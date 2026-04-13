"""
=============================================================================
  WORKER — ENTRENAMIENTO NEURAL DISTRIBUIDO CON SOCKETS
=============================================================================

El worker:
1. Se conecta al servidor
2. Para cada época recibe:
   - batch_ids: lista de identificadores de batches
   - params: parámetros globales del modelo
   - learning_rate
   - init_signal / stop_signal
3. Carga los batches basados en batch_ids
4. Entrena acumulando gradientes
5. Envía gradientes acumulados al servidor
6. Repite hasta recibir stop_signal

El dataset se carga localmente de manera que cada worker puede acceder a cualquier batch.
=============================================================================
"""

import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import socket
import time
import argparse

# Agregar el directorio padre al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from defineNetwork import Net
from Protocol import MessageFromServer, MessageFromWorker, WorkerReadyMessage, TrainingConfig
from messageHandling import send_message, receive_message

# Configuración
SOCKET_TIMEOUT = TrainingConfig.socket_timeout
SERVER_HOST = TrainingConfig.server_host
SERVER_PORT = TrainingConfig.server_port
BATCH_SIZE = TrainingConfig.batch_size
NUM_WORKERS = TrainingConfig.num_workers

class DistributedTrainingWorker:
    """
    Worker de Entrenamiento Distribuido.
    
    Se conecta al servidor y entrena los batches asignados.
    """
    
    def __init__(self, server_host, server_port):
        self.server_host = server_host
        self.server_port = server_port
        
        # Modelo
        self.net = Net()
        self.criterion = nn.CrossEntropyLoss()
        
        # Datos
        self.batches = list(TRAINLOADER)
        self.worker_id = None
        
        # Socket
        self.socket = None
        
        # Configuración de device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)
        
        # Para AMP (Automatic Mixed Precision)
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        print(f"Worker initialized: {len(self.batches)} batches loaded locally")
    
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
        
        Actualiza los parámetros del modelo y envía confirmación.
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
            
            # Actualizar parámetros del modelo
            self.update_model_params(message.params)
            print(f"  ✓ Parámetros del modelo actualizados")
            
            # Enviar confirmación
            ready_msg = WorkerReadyMessage(
                worker_id=0,  # Se asignará en el servidor
                dataset_size=len(self.batches)
            )
            send_message(self.socket, ready_msg)
            print(f"  ✓ Confirmación de listo enviada al servidor")
            
        except Exception as e:
            print(f"  ✗ Error en inicialización: {e}")
            raise
    
    def update_model_params(self, params_dict):
        """
        Actualiza los parámetros del modelo desde el servidor.
        
        Parámetros:
            params_dict: Dict con parámetros en formato numpy
        """
        with torch.no_grad():
            for name, param in self.net.named_parameters():
                if name in params_dict:
                    param_data = torch.tensor(params_dict[name], dtype=param.dtype, device=param.device)
                    param.data = param_data
    
    def compute_accuracy(self, outputs, labels):
        """Calcula la precisión"""
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        return 100 * correct / total
    
    def train_epoch(self, batch_ids):
        """
        Entrena una época con los batches asignados.
        
        Acumula gradientes de todos los batches y computa pérdida y precisión.
        
        Parámetros:
            batch_ids: Lista de identificadores de batches
        
        Retorna:
            (gradients_dict, avg_loss, avg_accuracy, training_time)
        """
        print(f"    Entrenando con {len(batch_ids)} batches...")
        
        tiempo_inicio = time.time()
        
        self.net.train()
        
        accumulated_grads = {}
        total_loss = 0.0
        total_accuracy = 0.0
        num_samples = 0
        
        # Procesamiento de batches
        for batch_idx in batch_ids:
            if batch_idx >= len(self.batches):
                print(f"    ⚠ Warning: batch_id {batch_idx} out of range ({len(self.batches)})")
                continue
            
            inputs, labels = self.batches[batch_idx]
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.net.zero_grad()
            
            # Forward pass
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Unscale para acumulación
                dummy_optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01)
                self.scaler.unscale_(dummy_optimizer)
            else:
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
            
            # Acumular gradientes
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
        
        tiempo_entrenamiento = time.time() - tiempo_inicio
        
        avg_loss = total_loss / len(batch_ids) if batch_ids else 0.0
        avg_accuracy = total_accuracy / num_samples if num_samples > 0 else 0.0
        
        print(f"    ✓ Entrenamiento completado: Loss={avg_loss:.4f}, Acc={avg_accuracy:.2f}%")
        
        return accumulated_grads, avg_loss, avg_accuracy, tiempo_entrenamiento
    
    def training_loop(self):
        """
        Bucle principal del worker.
        
        Recibe mensajes del servidor, entrena, envía gradientes.
        Continúa hasta recibir stop_signal.
        """
        print(f"\n{'='*70}")
        print(f"  INICIANDO BUCLE DE ENTRENAMIENTO")
        print(f"{'='*70}\n")
        
        epoch_count = 0
        
        while True:
            try:
                # Recibir mensaje del servidor
                print(f"  [Worker] Esperando mensaje del servidor...")
                message = receive_message(self.socket)
                
                epoch_count += 1
                
                print(f"  ✓ Recibido: epoch={message.epoch}, init={message.init_signal}, "
                      f"stop={message.stop_signal}, batches={len(message.batch_ids)}")
                
                # ┌─── HANDSHAKE: Responder a mensaje de sincronización ───┐
                if message.init_signal and message.epoch == 0:
                    # Ya manejado en wait_for_initialization
                    continue
                # └─────────────────────────────────────────────────┘
                
                # Actualizar parámetros del modelo
                self.update_model_params(message.params)
                print(f"    → Parámetros del modelo actualizados (epoch {message.epoch})")
                
                # Entrenar
                gradients, loss, accuracy, train_time = self.train_epoch(message.batch_ids)
                
                # Crear respuesta
                response = MessageFromWorker(
                    worker_id=0,
                    epoch=message.epoch,
                    gradients=gradients,
                    loss=loss,
                    accuracy=accuracy,
                    training_time=train_time
                )
                
                # Enviar gradientes
                print(f"    → Enviando gradientes...")
                send_message(self.socket, response)
                print(f"    ✓ Gradientes enviados")
                
                # Verificar stop signal
                if message.stop_signal:
                    print(f"\n  ✓ Stop signal recibido. Terminando worker.")
                    break
                
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


def start_worker():
    """Inicia el worker de entrenamiento distribuido"""
    worker = DistributedTrainingWorker(SERVER_HOST, SERVER_PORT)
    
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
        # permitir pasar parámetros por línea de comandos para el servidor
    parser = argparse.ArgumentParser(
        description="Worker para entrenamiento distribuido."
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
        "--particiones",
        "-n",
        type=int,
        default=NUM_WORKERS,
        help=f"Número de particiones/datos (por defecto: {NUM_WORKERS})",
    )

    
    args = parser.parse_args()
    SERVER_HOST = args.host
    SERVER_PORT = args.port
    NUM_WORKERS = args.particiones

    # Definir TRANSFORM localmente
    TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Crear dataset y dataloader
    TRAINSET = datasets.CIFAR10(root='./data', train=True, download=True, transform=TRANSFORM)
    TRAINLOADER = torch.utils.data.DataLoader(TRAINSET, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())

    start_worker()