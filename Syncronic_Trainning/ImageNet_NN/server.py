"""
=============================================================================
  SERVIDOR — ENTRENAMIENTO NEURONAL DISTRIBUIDO IMAGENET CON SOCKETS
=============================================================================

El servidor:
1. Carga el dataset ImageNet en modo streaming (HuggingFace)
2. Particiona el dataset en K shards (uno por worker)
3. Abre un socket servidor esperando conexiones de workers
4. Para cada época:
   - Envía a cada worker: epoch, batch_ids, shard_size, pesos globales, learning_rate, init/stop signal
   - Recibe de cada worker: gradientes calculados
   - Promedia los gradientes
   - Actualiza los pesos globales
5. Al final, evaluación en validación
=============================================================================
"""


import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import socket
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, List
import argparse

# Agregar el directorio padre al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from defineNetwork import Net
from Protocol import MessageFromServer, MessageFromWorker, WorkerReadyMessage, TrainingConfig, SHARD_SIZE
from messageHandling import send_message, receive_message
from loadImageNet import (
    get_imagenet_stream_dataloader, 
    get_hf_split_size,
    detect_data_source
)
from Utils.ModelPersistence import guardar_modelo

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DEL SERVIDOR
# ─────────────────────────────────────────────────────────────────────────────

# Importar constantes desde TrainingConfig
NUM_WORKERS = TrainingConfig.num_workers
LEARNING_RATE = TrainingConfig.learning_rate
INTERVALO_LOG = TrainingConfig.intervalo_log
SOCKET_TIMEOUT = TrainingConfig.socket_timeout
SERVER_HOST = TrainingConfig.server_host
SERVER_PORT = TrainingConfig.server_port
BATCH_SIZE = TrainingConfig.batch_size
SAVE_FILE = TrainingConfig.save_file
NUM_EPOCHS = TrainingConfig.epocas
NUM_CLASSES = TrainingConfig.num_classes
IMAGENET_SPLIT = TrainingConfig.imagenet_split
HF_TOKEN = TrainingConfig.hf_token

class DistributedTrainingServer:
    """
    Servidor de Entrenamiento Distribuido ImageNet.
    
    Maneja conexiones de múltiples workers y coordina el entrenamiento federado
    con shards de ImageNet.
    """
    
    def __init__(self, host, port, num_workers, epocas, learning_rate, hf_token, split='train', shard_size=10000, pretrained=False, freeze_backbone=False):
        self.host = host
        self.port = port
        self.num_workers = num_workers
        self.epocas = epocas
        self.hf_token = hf_token
        self.split = split
        
        # If fine-tuning a pretrained model (no freeze) and learning rate is the default 0.01,
        # adjust default to 0.0001 (1e-4) to avoid disrupting pretrained weights.
        if pretrained and not freeze_backbone and learning_rate == 0.01:
            print("  ℹ Fine-tuning pretrained model: adjusting learning rate to 0.0001 (1e-4)")
            self.learning_rate = 0.0001
        else:
            self.learning_rate = learning_rate
            
        # Modelo
        self.net = Net(num_classes=NUM_CLASSES, pretrained=pretrained, freeze_backbone=freeze_backbone)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)
        
        self.optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self.net.parameters()), 
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=1e-2
        )
        
        # Dataset size for scheduler
        self.total_dataset_size = get_hf_split_size(split)
        
        # Step-based synchronization configuration
        self.steps_per_epoch = TrainingConfig.steps_per_epoch  # Default 10 steps per epoch
        shard_size = SHARD_SIZE
        num_batches_per_worker = shard_size // BATCH_SIZE  # Total batches per worker per epoch
        self.batches_per_step = max(1, num_batches_per_worker // self.steps_per_epoch)  # Batches per sync point
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.01,
            epochs=epocas,
            steps_per_epoch=self.steps_per_epoch,
            pct_start=0.3,
            div_factor=10,
            final_div_factor=100
        )
        
        # Conexiones de workers
        self.worker_sockets: Dict[int, socket.socket] = {}
        self.worker_connected = {}
        
        # Datos sobre particiones
        self.shard_sizes = SHARD_SIZE  
        
        # Historial de checkpoints (epoch-level)
        self.historial_intervalo_epochs = []
        self.historial_intervalo_times = []
        self.historial_intervalo_loss = []
        self.historial_intervalo_acc_train = []
        
        # Historial de pasos (step-level, one entry per synchronization point)
        self.step_loss_history = []
        self.step_accuracy_history = []
        self.step_times_history = []
        self.step_ids_history = []
    

    def setup_socket_server(self):
        """Configura el socket servidor."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2097152)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2097152)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.num_workers)
        self.server_socket.settimeout(SOCKET_TIMEOUT)
        
        print(f"\n{'='*70}")
        print(f"  SERVIDOR DISTRIBUIDO IMAGENET — ESCUCHANDO EN {self.host}:{self.port}")
        print(f"{'='*70}")
        print(f"  Esperando {self.num_workers} conexiones de workers...")
    
    def wait_for_workers(self):
        """
        Espera a que se conecten todos los workers.
        Asigna worker_id basado en el orden de conexión.
        Envía mensaje de sincronización inicial a cada worker con shard_size.
        """
        # FASE 1: Aceptar todas las conexiones
        for worker_id in range(self.num_workers):
            try:
                print(f"\n  [Esperando] Worker {worker_id}...")
                client_socket, client_address = self.server_socket.accept()
                client_socket.settimeout(SOCKET_TIMEOUT)
                client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2097152)
                client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2097152)
                
                self.worker_sockets[worker_id] = client_socket
                self.worker_connected[worker_id] = True
                
                print(f"  ✓ Worker {worker_id} conectado desde {client_address}")
                
            except socket.timeout:
                print(f"\n  ✗ Timeout esperando worker {worker_id}")
                raise
            except Exception as e:
                print(f"\n  ✗ Error aceptando conexión: {e}")
                raise
        
        # FASE 2: Enviar mensaje de sincronización a todos los workers
        print(f"\n  {'─'*68}")
        print(f"  FASE DE SINCRONIZACIÓN — Enviando señales de inicio a workers")
        print(f"  {'─'*68}")
        
        for worker_id in range(self.num_workers):
            try:
                # Crear mensaje de sincronización (epoch=0, init_signal=True)
                # Bug #2 fix: send full state_dict (includes BN running_mean/var buffers)
                params = {name: tensor.cpu().numpy() for name, tensor in self.net.state_dict().items()}
                shard_size = self.shard_sizes
                
                message = MessageFromServer(
                    batch_ids=[],
                    epoch=0,
                    init_signal=True,
                    stop_signal=False,
                    learning_rate=self.learning_rate,
                    shard_size=shard_size,
                    params=params,
                    hf_token=self.hf_token,
                    worker_id=worker_id,
                    num_workers=self.num_workers,
                    steps_per_epoch=self.steps_per_epoch
                )
                
                # Enviar mensaje de sincronización
                sock = self.worker_sockets[worker_id]
                send_message(sock, message)
                
                print(f"    → Sincronización enviada a worker {worker_id} (shard_size={shard_size:,})")
                
            except Exception as e:
                print(f"    ✗ Error sincronizando worker {worker_id}: {e}")
                raise
        
        # FASE 3: Esperar confirmación (handshake) de todos los workers
        print(f"\n  {'─'*68}")
        print(f"  FASE DE HANDSHAKE — Esperando confirmación de workers")
        print(f"  {'─'*68}")
        
        for worker_id in range(self.num_workers):
            try:
                sock = self.worker_sockets[worker_id]
                ready_msg = receive_message(sock)
                
                print(f"    ✓ Worker {worker_id} listo (dataset_size={ready_msg.dataset_size:,})")
                
            except Exception as e:
                print(f"    ✗ Error esperando confirmación de worker {worker_id}: {e}")
                raise
        
        print(f"  ✓ Todos los workers sincronizados y listos para entrenar")
    
    def distribute_work(self, epoch, step=0, is_last_epoch=False):
        """
        Distribuye trabajo a todos los workers para un paso (step) específico dentro de una época.
        
        Envía a cada worker: epoch, step_id, batch_ids para este paso, pesos globales, learning_rate, etc.
        Cada step sincroniza un porción del dataset (batches_per_step batches).
        
        Args:
            epoch: Número de época (1-based)
            step: Número del paso dentro de la época (0-based, 0 to steps_per_epoch-1)
            is_last_epoch: True si esta es la última época
        """
        is_last_step = (step == self.steps_per_epoch - 1)
        
        print(f"\n  {'─'*68}")
        print(f"  ÉPOCA {epoch}/{self.epocas} — PASO {step+1}/{self.steps_per_epoch} — DISTRIBUYENDO TRABAJO")
        print(f"  {'─'*68}")
        
        for worker_id in range(self.num_workers):
            try:
                # Calcular número de batches según shard_size
                shard_size = self.shard_sizes
                num_batches = shard_size // BATCH_SIZE
                steps_per_epoch = self.steps_per_epoch
                
                # Calcular rango de batches para este step
                # Cada step procesa batches_per_step batches
                step_start_global = step * self.batches_per_step
                step_end_global = min((step + 1) * self.batches_per_step, num_batches)  # Don't exceed total batches
                
                # Distribuir estos batches entre workers
                # Worker i procesa: [step_start + i*batch_per_worker_per_step : step_start + (i+1)*batch_per_worker_per_step]
                batches_per_worker_per_step = (step_end_global - step_start_global) // self.num_workers
                start_batch = step_start_global + worker_id * batches_per_worker_per_step
                if worker_id == self.num_workers - 1:
                    # Last worker gets any remaining batches
                    end_batch = step_end_global
                else:
                    end_batch = start_batch + batches_per_worker_per_step
                
                batch_ids = list(range(start_batch, end_batch))
                
                # Bug #2 fix: send full state_dict (includes BN running_mean/var buffers)
                params = {name: tensor.cpu().numpy() for name, tensor in self.net.state_dict().items()}
                
                message = MessageFromServer(
                    batch_ids=batch_ids,
                    epoch=epoch,
                    step_id=step,
                    steps_per_epoch=steps_per_epoch,
                    init_signal=(step == 0),  # True only at start of epoch (step 0)
                    stop_signal=(is_last_epoch and is_last_step),  # True only at final step of final epoch
                    learning_rate=self.learning_rate,
                    shard_size=shard_size,
                    params=params,
                    hf_token=self.hf_token,
                    worker_id=worker_id,
                    num_workers=self.num_workers
                )
                
                # Enviar al worker
                sock = self.worker_sockets[worker_id]
                send_message(sock, message)
                
                print(f"    ✓ Enviado a worker {worker_id}: epoch={epoch},  steps_per_epoch={steps_per_epoch}, "
                      f"batches={len(batch_ids)} [{start_batch}-{end_batch-1}]")
                
            except Exception as e:
                print(f"    ✗ Error enviando a worker {worker_id}: {e}")
                raise
    
    def collect_results(self):
        """
        Recolecta resultados de todos los workers para la época actual.
        
        Recibe gradientes y métricas de cada worker.
        """
        print(f"\n  {'─'*68}")
        print(f"  RECOLECTANDO RESULTADOS DE WORKERS")
        print(f"  {'─'*68}")
        
        all_messages = []
        
        for worker_id in range(self.num_workers):
            try:
                sock = self.worker_sockets[worker_id]
                message = receive_message(sock)
                
                all_messages.append(message)
                print(f"    ✓ Worker {worker_id}: {message}")
                
            except Exception as e:
                print(f"    ✗ Error recibiendo de worker {worker_id}: {e}")
                raise
        
        return all_messages
    
    def average_gradients(self, messages_list):
        """
        Promedia los gradientes de todos los workers.
        
        IMPORTANTE: Los gradientes ya están normalizados por batch_count en el worker.
        Solo necesitamos promediarlos aquí.
        
        Retorna:
            Dict con gradientes promediados para cada parámetro
        """
        num_workers = len(messages_list)
        
        # Inicializar diccionario de gradientes promediados
        avg_grads = {}
        
        # Iterar sobre todas las claves de parámetros del primer worker
        if num_workers > 0:
            for param_name in messages_list[0].gradients.keys():
                # Promediar este parámetro de todos los workers
                grads_list = [msg.gradients[param_name] for msg in messages_list]
                avg_grads[param_name] = sum(grads_list) / num_workers
        
        # Log de depuración: verificar magnitud de gradientes promediados
        if avg_grads:
            grad_norms = [np.linalg.norm(g.flatten()) for g in avg_grads.values() if g.size > 0]
            avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
            print(f"    ℹ Server: Gradient norm promedio después de averaging: {avg_grad_norm:.6f} (across {num_workers} workers)")
        
        return avg_grads
    
    def update_model(self, avg_grads):
        """
        Actualiza los pesos del modelo usando los gradientes promediados.
        
        Los gradientes llegan ya:
        - Normalizados por batch_count desde el worker
        - Promediados entre workers
        """
        self.optimizer.zero_grad()
        
        # Asignar gradientes a los parámetros
        for name, param in self.net.named_parameters():
            if name in avg_grads:
                param.grad = torch.tensor(avg_grads[name], dtype=param.dtype, device=param.device)
        
        # Log de depuración: verificar norma total de gradientes antes de clipping
        total_norm = 0.0
        for p in self.net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f"    ℹ Server: Gradient norm total ANTES de clipping: {total_norm:.6f}")
        
        # Aplicar clipping para evitar exploding gradients
        clipped_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
        print(f"    ℹ Server: Gradient clipping aplicado (norm={clipped_norm:.6f})")
        
        # Actualizar pesos
        self.optimizer.step()
        # Note: scheduler.step() is called once per epoch in training_loop, not per step
    
    def apply_worker_buffers(self, messages_list):
        """
        Bug #2 fix: average BN running stats from all workers and apply to the
        server model so that saved checkpoints have correct normalization for eval.
        
        running_mean / running_var  → averaged across workers.
        num_batches_tracked          → taken from worker 0 (it's a counter, not a stat).
        """
        all_bufs = [msg.buffers for msg in messages_list if getattr(msg, 'buffers', None)]
        if not all_bufs or not all_bufs[0]:
            return  # workers didn't send buffers (old protocol) — skip silently
        
        current_state = self.net.state_dict()
        new_state = dict(current_state)  # shallow copy; we'll replace BN entries
        
        for buf_name in all_bufs[0].keys():
            buf_arrays = [b[buf_name] for b in all_bufs if buf_name in b]
            if not buf_arrays:
                continue
            
            target = current_state.get(buf_name)
            if target is None:
                continue
            
            if 'num_batches_tracked' in buf_name:
                # Counter — just take the value from the first worker
                merged = torch.tensor(buf_arrays[0], dtype=target.dtype, device=target.device)
            else:
                # running_mean / running_var — average across workers
                avg = sum(buf_arrays) / len(buf_arrays)
                merged = torch.tensor(avg, dtype=target.dtype, device=target.device)
            
            new_state[buf_name] = merged
        
        self.net.load_state_dict(new_state)
        print(f"    ℹ Server: BN running stats sincronizadas desde {len(all_bufs)} workers")
    
    def evaluate_global_model(self, epoch, tiempo_actual, avg_loss, avg_epoch_acc=None):
        """
        Evalúa el modelo global y guarda métricas en historial.
        
        Parámetros:
            epoch: int, número de época actual
            tiempo_actual: float, tiempo transcurrido desde el inicio del entrenamiento
            avg_loss: float, pérdida promedio de la época
            avg_epoch_acc: float, opcional, precisión promedio de la época
        """
        if epoch % INTERVALO_LOG == 0 or epoch == 1:
            self.historial_intervalo_epochs.append(epoch)
            self.historial_intervalo_times.append(round(tiempo_actual, 6))
            self.historial_intervalo_loss.append(round(avg_loss, 6))
            if avg_epoch_acc is not None:
                self.historial_intervalo_acc_train.append(round(avg_epoch_acc, 6))
            
            print(f"\n  {'─'*68}")
            print(f"  EVALUACIÓN GLOBAL — ÉPOCA {epoch}/{self.epocas}")
            print(f"  {'─'*68}")
            print(f"    ✓ GLOBAL → Loss: {avg_loss:.4f}")
            if avg_epoch_acc is not None:
                print(f"    ✓ GLOBAL → Accuracy: {avg_epoch_acc:.2f}%")
            print(f"    ⏱ Tiempo acumulado: {tiempo_actual:.2f}s")
    
    def training_loop(self):
        """
        Bucle principal de entrenamiento distribuido con sincronización por pasos (steps).
        
        Para cada época, realiza múltiples pasos de sincronización (steps_per_epoch).
        Cada paso: distribuye trabajo → recolecta gradientes → promedia → actualiza pesos.
        """
        print(f"\n{'='*70}")
        print(f"  INICIANDO ENTRENAMIENTO DISTRIBUIDO IMAGENET")
        print(f"  ({self.steps_per_epoch} pasos de sincronización por época)")
        print(f"{'='*70}\n")
        
        training_start = time.time()
        epoch_loss_history = []
        
        try:
            for epoch in range(1, self.epocas + 1):
                epoch_start = time.time()
                epoch_losses = []
                epoch_accs = []
                is_last_epoch = (epoch == self.epocas)
                
                # Bucle de pasos dentro de cada época
                for step in range(self.steps_per_epoch):
                    is_last_step = (step == self.steps_per_epoch - 1)
                    
                    # Distribuir trabajo para este paso
                    self.distribute_work(epoch, step=step, is_last_epoch=is_last_epoch)
                    
                    # Recolectar resultados de este paso
                    messages = self.collect_results()
                    
                    # Promediar gradientes y calcular pérdidas
                    avg_grads = self.average_gradients(messages)
                    avg_loss = sum(msg.loss for msg in messages) / len(messages) if messages else 0.0
                    avg_acc = sum(msg.accuracy for msg in messages) / len(messages) if messages else 0.0
                    
                    epoch_losses.append(avg_loss)
                    epoch_accs.append(avg_acc)
                    
                    # Actualizar pesos con gradientes promediados
                    self.update_model(avg_grads)
                    self.scheduler.step()
                    
                    # Sincronizar BN stats en cada paso para que el servidor distribuya
                    # los buffers de BatchNorm actualizados a todos los workers.
                    self.apply_worker_buffers(messages)
                    
                    # Registrar métricas a nivel de paso (step-level)
                    self.step_loss_history.append(round(avg_loss, 6))
                    self.step_accuracy_history.append(round(avg_acc, 6))
                    self.step_times_history.append(round(time.time() - training_start, 6))
                    self.step_ids_history.append([epoch, step])
                    
                    # Log del paso
                    step_time = time.time() - epoch_start
                    print(f"    ✓ Paso {step+1}/{self.steps_per_epoch} completado "
                          f"| Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}%\n")
                
                # Al final de la época, actualizar scheduler y registrar métricas
                epoch_time = time.time() - epoch_start
                total_time = time.time() - training_start
                avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
                avg_epoch_acc = sum(epoch_accs) / len(epoch_accs) if epoch_accs else 0.0
                
                # Registrar métricas en historial
                self.evaluate_global_model(epoch, total_time, avg_epoch_loss, avg_epoch_acc)
                
                print(f"  Epoch {epoch} completada en {epoch_time:.4f}s "
                      f"(Total: {total_time:.4f}s | Acc: {avg_epoch_acc:.2f}%)\n")
            
            print(f"\n{'='*70}")
            print(f"  ENTRENAMIENTO COMPLETADO")
            print(f"{'='*70}\n")
            
            # Calcular tiempo total de entrenamiento
            tiempo_total = time.time() - training_start

            nombre_modelo = input("\n  Ingrese un nombre para guardar el modelo: ").strip()
            
            # Guardar modelo PyTorch
            model_path = f"models/{nombre_modelo}_imagenet.pt"
            os.makedirs("models", exist_ok=True)
            torch.save(self.net.state_dict(), model_path)
            
            # Guardar modelo con métricas completas
            guardar_modelo(
                None, None, None, None,  # PyTorch model, not NumPy weights
                nombre_modelo=nombre_modelo,
                precision_test=0.0,
                epocas=self.epocas,
                learning_rate=self.learning_rate,
                training_time=tiempo_total,
                step_loss_history=self.step_loss_history,
                step_accuracy_history=self.step_accuracy_history,
                step_times_history=self.step_times_history,
                step_ids_history=self.step_ids_history,
                info_extra={
                    'num_workers': self.num_workers,
                    'architecture': 'ImageNet ResNet - Distributed with Sockets',
                    'server_host': self.host,
                    'server_port': self.port,
                    'tiempo_total_segundos': tiempo_total,
                    'historial_intervalo_epochs': self.historial_intervalo_epochs,
                    'historial_intervalo_times': self.historial_intervalo_times,
                    'historial_intervalo_loss': self.historial_intervalo_loss,
                    'historial_intervalo_acc_train': self.historial_intervalo_acc_train,
                    'model_path': model_path,
                    'dataset_split': self.split,
                    'num_classes': NUM_CLASSES,
                }
            )
        
        except Exception as e:
            print(f"\n✗ Error durante entrenamiento: {e}")
            raise
        finally:
            # Cerrar conexiones
            for worker_id, sock in self.worker_sockets.items():
                try:
                    sock.close()
                except:
                    pass
            self.server_socket.close()
    

def start_server(host, port, num_workers, epocas, learning_rate, hf_token, split, shard_size, pretrained, freeze_backbone):
    """Inicia el servidor de entrenamiento distribuido"""
    server = DistributedTrainingServer(
        host, port, num_workers, epocas, learning_rate, hf_token, split, shard_size,
        pretrained=pretrained, freeze_backbone=freeze_backbone
    )
    server.setup_socket_server()
    server.wait_for_workers()
    server.training_loop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Servidor para entrenamiento distribuido de ImageNet."
    )

    parser.add_argument(
        "--host",
        "-H",
        default=SERVER_HOST,
        help=f"Host en el que el servidor escuchará (por defecto: {SERVER_HOST})",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=SERVER_PORT,
        help=f"Puerto en el que el servidor escuchará (por defecto: {SERVER_PORT})",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=NUM_WORKERS,
        help=f"Número de workers (por defecto: {NUM_WORKERS})",
    )
    parser.add_argument(
        "--epocas",
        "-e",
        type=int,
        default=NUM_EPOCHS,
        help=f"Cantidad de épocas para entrenar (por defecto: {NUM_EPOCHS})",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help=f"Tasa de aprendizaje (por defecto: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=HF_TOKEN,
        help="Token de HuggingFace para acceso a ImageNet",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=IMAGENET_SPLIT,
        choices=['train', 'val'],
        help=f"Split de ImageNet a usar (por defecto: {IMAGENET_SPLIT})",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=10000,
        help="Tamaño de shard de datos por worker (por defecto: 10000)",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Usar un modelo ResNet-18 preentrenado",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Congelar los pesos del feature extractor (backbone) en el modelo preentrenado",
    )

    args = parser.parse_args()

    start_server(
        args.host,
        args.port,
        args.workers,
        args.epocas,
        args.lr,
        args.hf_token,
        args.split,
        args.shard_size,
        args.pretrained,
        args.freeze_backbone,
    )