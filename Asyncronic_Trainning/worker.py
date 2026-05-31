"""
=============================================================================
  WORKER — ASYNC-SGD DISTRIBUTED TRAINING (STALENESS-AWARE)
=============================================================================

The worker loop (pullWeights / calcGradient / pushGradient):
1. Connects to the server and receives initialization (shard info, model params).
2. In a loop:
   a. Receives weights + weight-timestamp (step_id = j) from server.
   b. Computes gradients on its next mini-batches using a persistent iterator.
   c. Sends gradients + timestamp j back to server.
   d. Server computes staleness τ = i − j and scales the gradient.
3. Stops when the server sends stop_signal.

The persistent dataloader iterator auto-resets on StopIteration so the worker
cycles through its shard continuously without manual epoch tracking.
=============================================================================
"""
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

import sys
import os
import torch
import torch.nn as nn
import socket
import time
import argparse
import numpy as np

# Agregar el directorio padre al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from defineNetwork import Net
from Protocol import MessageFromServer, MessageFromWorker, WorkerReadyMessage, TrainingConfig
from messageHandling import send_message, receive_message
from loadImageNet import get_imagenet_stream_dataloader

# Configuración
SOCKET_TIMEOUT = TrainingConfig.socket_timeout
SERVER_HOST = TrainingConfig.server_host
SERVER_PORT = TrainingConfig.server_port
BATCH_SIZE = TrainingConfig.batch_size
NUM_CLASSES = TrainingConfig.num_classes
HF_TOKEN = TrainingConfig.hf_token


class DistributedTrainingWorker:
    """
    Worker de Entrenamiento Distribuido para ImageNet.
    
    Se conecta al servidor y entrena los batches asignados del shard ImageNet.
    """
    
    def __init__(self, server_host, server_port, imagenet_split='train'):
        self.server_host = server_host
        self.server_port = server_port
        self.imagenet_split = imagenet_split
        self.hf_token = None  # Will be received from server
        
        # Modelo
        self.net = Net(num_classes=NUM_CLASSES)
        self.criterion = nn.CrossEntropyLoss()
        
        # Datos
        self.worker_id = None
        self.shard_size = None
        self.dataloader = None
        self.dataloader_iter = None
        self.global_batch_index = 0  # Track total batches processed globally
        
        # Socket
        self.socket = None
        
        # Configuración de device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)
        
        # Para AMP (Automatic Mixed Precision)
        # Bug #3 fix: create a single persistent optimizer for GradScaler.unscale_().
        # A new SGD was previously created each batch, corrupting the scaler state.
        if torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            self._amp_optimizer = torch.optim.SGD(self.net.parameters(), lr=0.0)
        else:
            self.scaler = None
            self._amp_optimizer = None
        
        print(f"Worker inicializado para ImageNet ({imagenet_split})")
    
    def connect_to_server(self):
        """Se conecta al servidor."""
        print(f"\n{'='*70}")
        print(f"  WORKER — CONECTANDO AL SERVIDOR")
        print(f"{'='*70}")
        print(f"  Intentando conectar a {self.server_host}:{self.server_port}...")
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(SOCKET_TIMEOUT)
            self.socket.connect((self.server_host, self.server_port))
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2097152)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2097152)
            print(f"  ✓ Conectado al servidor exitosamente")
            
        except ConnectionRefusedError:
            print(f"  ✗ Conexión rechazada. ¿El servidor está ejecutándose?")
            raise
        except socket.timeout:
            print(f"  ✗ Timeout conectando al servidor")
            raise
        except Exception as e:
            print(f"  ✗ Error conectando: {e}")
            raise
    
    def wait_for_initialization(self):
        """
        Espera el mensaje de sincronización inicial del servidor.
        
        Recibe shard_size y actualiza los parámetros del modelo.
        """
        print(f"\n{'='*70}")
        print(f"  ESPERANDO INICIALIZACIÓN DEL SERVIDOR")
        print(f"{'='*70}\n")
        
        try:
            # Recibir mensaje de sincronización
            print(f"  [Worker] Esperando mensaje de sincronización...")
            message = receive_message(self.socket)
            
            if not message.init_signal:
                raise RuntimeError("Mensaje de sincronización no recibido")
            
            print(f"  ✓ Recibido mensaje de sincronización del servidor")
            
            # Guardar shard_size
            self.shard_size = message.shard_size
            print(f"  ✓ Shard size asignado: {self.shard_size:,} imágenes")
            
            self.hf_token = message.hf_token
            self.worker_id = message.worker_id
            self.num_workers = message.num_workers
            
            # Actualizar parámetros del modelo
            self.update_model_params(message.params)
            print(f"  ✓ Parámetros del modelo actualizados")
            
            # Crear dataloader para el shard asignado (una sola vez)
            print(f"  ⏳ Inicializando dataloader de ImageNet Shard {self.worker_id}/{self.num_workers} ({self.imagenet_split})...")
            self.dataloader = get_imagenet_stream_dataloader(
                split=self.imagenet_split,
                token=self.hf_token,
                batch_size=BATCH_SIZE,
                shard_index=self.worker_id,
                num_shards=self.num_workers,
            )
            self.dataloader_iter = iter(self.dataloader)
            print(f"  ✓ Dataloader e iterador inicializados")
            
            # Enviar confirmación
            ready_msg = WorkerReadyMessage(
                worker_id=self.worker_id,
                dataset_size=self.shard_size
            )
            send_message(self.socket, ready_msg)
            print(f"  ✓ Confirmación de listo enviada al servidor")
            
        except Exception as e:
            print(f"  ✗ Error en inicialización: {e}")
            raise
    
    def update_model_params(self, params_dict):
        
        with torch.no_grad():
            current_state = self.net.state_dict()
            new_state = {}
            for name, current_val in current_state.items():
                if name in params_dict:
                    new_state[name] = torch.tensor(
                        params_dict[name],
                        dtype=current_val.dtype,
                        device=current_val.device
                    )
                else:
                    # Keep the local value for any key the server didn't send
                    new_state[name] = current_val
            self.net.load_state_dict(new_state)
    
    def compute_accuracy(self, outputs, labels):
        """Calcula la precisión"""
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        return 100 * correct / total
    
    def train_epoch(self, start_batch_id, num_batches, learning_rate):
        """
        Calcula gradientes para un paso (slice de una época) compuesto por num_batches.
        NO actualiza los pesos de forma local.
        
        Retorna:
            (gradients_dict, avg_loss, avg_accuracy, training_time, buffers_dict)
        """
        print(f"    Calculando gradientes para paso: {num_batches} batches...")
        
        tiempo_inicio = time.time()
        
        self.net.train()
        
        accumulated_grads = {}
        total_loss = 0.0
        total_accuracy = 0.0
        num_samples = 0
        batch_count = 0
        
        for _ in range(num_batches):
            try:
                if self.dataloader_iter is None:
                    self.dataloader_iter = iter(self.dataloader)
                inputs, labels = next(self.dataloader_iter)
            except StopIteration:
                print("    ℹ Dataloader agotado. Reiniciando iterador.")
                self.dataloader_iter = iter(self.dataloader)
                inputs, labels = next(self.dataloader_iter)
                
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.net.zero_grad()
            valid_grads = True
            
            # ── Forward + backward (sin optimizer.step() local) ─────────
            if self.scaler is not None:
                # En GPU con precisión mixta (AMP)
                with torch.cuda.amp.autocast():
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self._amp_optimizer)
                
                # Verificar overflow en AMP antes de acumular gradientes
                valid_grads = all(
                    p.grad is None or torch.isfinite(p.grad).all()
                    for p in self.net.parameters()
                )
                self.scaler.update()
            else:
                # En CPU o sin AMP
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
            # ─────────────────────────────────────────────────────────
            
            # Acumular gradientes (solo si no hubo overflow en AMP)
            if valid_grads:
                for name, param in self.net.named_parameters():
                    if param.grad is not None:
                        if name not in accumulated_grads:
                            accumulated_grads[name] = param.grad.detach().cpu().numpy().copy()
                        else:
                            accumulated_grads[name] += param.grad.detach().cpu().numpy()
            
            # Acumular métricas
            total_loss += loss.item()
            accuracy = self.compute_accuracy(outputs, labels)
            total_accuracy += accuracy * labels.size(0)
            num_samples += labels.size(0)
            
            batch_count += 1
            
            # Mostrar progreso cada 100 batches
            if batch_count % 100 == 0:
                print(f"      ... {batch_count}/{num_batches} batches procesados (Loss={loss.item():.4f})")
        
        tiempo_entrenamiento = time.time() - tiempo_inicio
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        avg_accuracy = total_accuracy / num_samples if num_samples > 0 else 0.0
        
        # Normalizar gradientes acumulados por el número de batches procesados en este paso
        if batch_count > 0:
            for name in accumulated_grads.keys():
                accumulated_grads[name] = accumulated_grads[name] / batch_count
        
        # Log de depuración: verificar magnitud de gradientes promediados
        if accumulated_grads:
            grad_norms = [np.linalg.norm(g.flatten()) for g in accumulated_grads.values() if g.size > 0]
            avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
            print(f"    ℹ Gradient norm promedio: {avg_grad_norm:.6f} (normalizado por {batch_count} batches)")
        
        # Sincronizar buffers de BatchNorm (running_mean, running_var)
        buffers = {
            name: tensor.cpu().numpy()
            for name, tensor in self.net.state_dict().items()
            if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name
        }
        
        print(f"    ✓ Paso completado: Loss={avg_loss:.4f}, Acc (train)={avg_accuracy:.2f}%")

        return accumulated_grads, avg_loss, avg_accuracy, tiempo_entrenamiento, buffers
    
    def training_loop(self):
        """
        Async worker loop: pullWeights → calcGradient → pushGradient.

        The worker receives weights with a global timestamp (step_id = j),
        computes gradients, and sends them back with the same timestamp
        so the server can compute staleness τ = i − j.

        The persistent dataloader iterator auto-resets via StopIteration,
        so the worker cycles through its shard continuously.
        """
        print(f"\n{'='*70}")
        print(f"  INICIANDO BUCLE DE ENTRENAMIENTO ASYNC")
        print(f"{'='*70}\n")
        
        round_count = 0
        
        while True:
            try:
                # ── pullWeights: receive weights + timestamp from server ──
                print(f"  [Worker] Esperando pesos del servidor...")
                message = receive_message(self.socket)

                # Check stop signal immediately
                if message.stop_signal:
                    print(f"\n  ✓ Stop signal recibido. Terminando worker.")
                    break
                
                round_count += 1
                
                # Extract step info
                num_batches_to_process = len(message.batch_ids)
                weight_timestamp = message.step_id  # j: the server's global_step when these weights were sent
                
                print(f"  ✓ Recibido: epoch={message.epoch}, "
                      f"weight_timestamp={weight_timestamp}, "
                      f"batches={num_batches_to_process}, "
                      f"round={round_count}")
                
                # ┌─── HANDSHAKE: skip re-init messages ───┐
                if message.init_signal and message.epoch == 0:
                    continue
                # └────────────────────────────────────────┘
                
                # Update model to the server's latest weights
                self.update_model_params(message.params)
                
                # ── calcGradient: compute on next mini-batches ───────────
                gradients, loss, accuracy, train_time, buffers = self.train_epoch(
                    start_batch_id=0,  # Informational only; persistent iterator handles position
                    num_batches=num_batches_to_process,
                    learning_rate=message.learning_rate
                )
                
                # ── pushGradient: send gradients + timestamp back ────────
                response = MessageFromWorker(
                    worker_id=self.worker_id,
                    epoch=message.epoch,
                    step_id=weight_timestamp,  # Echo back j so server computes τ = i − j
                    gradients=gradients,
                    loss=loss,
                    accuracy=accuracy,
                    training_time=train_time,
                    buffers=buffers,
                )
                
                print(f"    → Enviando gradientes (timestamp={weight_timestamp})...")
                send_message(self.socket, response)
                print(f"    ✓ Gradientes enviados (round {round_count})\n")
                
            except ConnectionError as e:
                print(f"\n  ✗ Conexión perdida con servidor: {e}")
                break
            except socket.timeout:
                print(f"\n  ✗ Timeout esperando mensaje del servidor")
                break
            except Exception as e:
                print(f"\n  ✗ Error en bucle de entrenamiento: {e}")
                import traceback
                traceback.print_exc()
                break
    
    def shutdown(self):
        """Cierra la conexión."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass


def start_worker(server_host, server_port, imagenet_split):
    """Inicia el worker de entrenamiento distribuido"""
    worker = DistributedTrainingWorker(server_host, server_port, imagenet_split)
    
    try:
        worker.connect_to_server()
        worker.wait_for_initialization()
        worker.training_loop()
    
    except Exception as e:
        print(f"\n✗ Error en worker: {e}")
    finally:
        worker.shutdown()
        print("\nWorker desconectado")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Worker para entrenamiento distribuido de ImageNet."
    )

    parser.add_argument(
        "--host",
        "-H",
        default=SERVER_HOST,
        help=f"Host del servidor (por defecto: {SERVER_HOST})",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=SERVER_PORT,
        help=f"Puerto del servidor (por defecto: {SERVER_PORT})",
    )
    parser.add_argument(
        "--split",
        type=str,
        default='train',
        choices=['train', 'val'],
        help="Split de ImageNet a usar (por defecto: train)",
    )

    args = parser.parse_args()

    start_worker(args.host, args.port, args.split)