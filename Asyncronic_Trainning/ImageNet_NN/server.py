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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from defineNetwork import Net
from Protocol import MessageFromServer, MessageFromWorker, WorkerReadyMessage, TrainingConfig, SHARD_SIZE
from messageHandling import send_message, receive_message
from Utils.loadImageNet import (
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
    
    def __init__(self, host, port, num_workers, epocas, learning_rate, hf_token, split='train', shard_size=10000):
        self.host = host
        self.port = port
        self.num_workers = num_workers
        self.epocas = epocas
        self.learning_rate = learning_rate
        self.hf_token = hf_token
        self.split = split
        
        # Modelo
        self.net = Net(num_classes=NUM_CLASSES)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)
        
        self.optimizer = optim.AdamW(
            self.net.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-2,
            betas=(0.9, 0.999), 
            eps=1e-8
        )
        
        # Dataset size for scheduler
        self.total_dataset_size = get_hf_split_size(split)
        batches_per_epoch = 1  # 1 weight update step per epoch in this local SGD setup
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.01,
            epochs=epocas,
            steps_per_epoch=batches_per_epoch,
            pct_start=0.3,
            div_factor=10,
            final_div_factor=100
        )
        
        # Conexiones de workers
        self.worker_sockets: Dict[int, socket.socket] = {}
        self.worker_connected = {}
        
        # Datos sobre particiones
        self.shard_sizes = shard_size
        
        # Historial de checkpoints
        self.historial_intervalo_epochs = []
        self.historial_intervalo_times = []
        self.historial_intervalo_loss = []
    

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
                    num_workers=self.num_workers
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
    
    def distribute_work(self, epoch, is_last_epoch=False):
        """
        Distribuye trabajo a todos los workers para una época.
        
        Envía a cada worker: epoch, batch_ids, shard_size, pesos globales, learning_rate, etc.
        Cada worker recibe diferentes batch_ids para trabajar en particiones distintas.
        """
        print(f"\n  {'─'*68}")
        print(f"  ÉPOCA {epoch}/{self.epocas} — DISTRIBUYENDO TRABAJO A WORKERS")
        print(f"  {'─'*68}")
        
        for worker_id in range(self.num_workers):
            try:
                # Calcular número de batches según shard_size
                shard_size = self.shard_sizes
                num_batches = shard_size // BATCH_SIZE
                
                # IMPORTANTE: Cada worker recibe diferentes batch_ids
                # Worker 0: [0, 1, 2, ...], Worker 1: [K, K+1, K+2, ...], etc.
                batches_per_worker = num_batches // self.num_workers
                start_batch = worker_id * batches_per_worker
                end_batch = start_batch + batches_per_worker if worker_id < self.num_workers - 1 else num_batches
                batch_ids = list(range(start_batch, end_batch))
                
                # Bug #2 fix: send full state_dict (includes BN running_mean/var buffers)
                params = {name: tensor.cpu().numpy() for name, tensor in self.net.state_dict().items()}
                
                message = MessageFromServer(
                    batch_ids=batch_ids,
                    epoch=epoch,
                    init_signal=True,
                    stop_signal=is_last_epoch,  # Bug #5 fix: True on final epoch
                    learning_rate=self.learning_rate,
                    shard_size=shard_size,
                    params=params,
                    hf_token=self.hf_token
                )
                
                # Enviar al worker
                sock = self.worker_sockets[worker_id]
                send_message(sock, message)
                
                print(f"    ✓ Enviado a worker {worker_id}: epoch={epoch}, "
                      f"shard_size={shard_size:,}, batches={len(batch_ids)} "
                      f"[{start_batch}-{end_batch-1}]")
                
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
        clipped_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        print(f"    ℹ Server: Gradient clipping aplicado (norm={clipped_norm:.6f})")
        
        # Actualizar pesos
        self.optimizer.step()
        self.scheduler.step()
    
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
    
    def evaluate_global_model(self, epoch, tiempo_actual, avg_loss):
        """
        Evalúa el modelo global y guarda métricas en historial.
        
        Parámetros:
            epoch: int, número de época actual
            tiempo_actual: float, tiempo transcurrido desde el inicio del entrenamiento
            avg_loss: float, pérdida promedio de la época
        """
        if epoch % INTERVALO_LOG == 0 or epoch == 1:
            self.historial_intervalo_epochs.append(epoch)
            self.historial_intervalo_times.append(round(tiempo_actual, 6))
            self.historial_intervalo_loss.append(round(avg_loss, 6))
            
            print(f"\n  {'─'*68}")
            print(f"  EVALUACIÓN GLOBAL — ÉPOCA {epoch}/{self.epocas}")
            print(f"  {'─'*68}")
            print(f"    ✓ GLOBAL → Loss: {avg_loss:.4f}")
            print(f"    ⏱ Tiempo acumulado: {tiempo_actual:.2f}s")
    
    def training_loop(self):
        """
        Bucle principal de entrenamiento.
        """
        print(f"\n{'='*70}")
        print(f"  INICIANDO ENTRENAMIENTO DISTRIBUIDO IMAGENET")
        print(f"{'='*70}\n")
        
        training_start = time.time()
        
        try:
            for epoch in range(1, self.epocas + 1):
                epoch_start = time.time()
                is_last = (epoch == self.epocas)
                
                # Distribuir trabajo — Bug #5 fix: send stop_signal on final epoch
                self.distribute_work(epoch, is_last_epoch=is_last)
                
                # Recolectar resultados
                messages = self.collect_results()
                
                # Promediar gradientes y calcular pérdida promedio
                avg_grads = self.average_gradients(messages)
                avg_loss = sum(msg.loss for msg in messages) / len(messages) if messages else 0.0
                avg_acc = sum(msg.accuracy for msg in messages) / len(messages) if messages else 0.0
                
                # Actualizar pesos con gradientes promediados
                self.update_model(avg_grads)
                
                # Bug #2 fix: sincronizar BN running stats desde los workers
                self.apply_worker_buffers(messages)
                
                epoch_time = time.time() - epoch_start
                total_time = time.time() - training_start
                
                # Registrar métricas en historial
                self.evaluate_global_model(epoch, total_time, avg_loss)
                
                print(f"  Epoch {epoch} completada en {epoch_time:.4f}s "
                      f"(Total: {total_time:.4f}s | Acc: {avg_acc:.2f}%)\n")
            
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
                info_extra={
                    'num_workers': self.num_workers,
                    'architecture': 'ImageNet ResNet - Distributed with Sockets',
                    'server_host': self.host,
                    'server_port': self.port,
                    'tiempo_total_segundos': tiempo_total,
                    'historial_intervalo_epochs': self.historial_intervalo_epochs,
                    'historial_intervalo_times': self.historial_intervalo_times,
                    'historial_intervalo_loss': self.historial_intervalo_loss,
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
    

def start_server(host, port, num_workers, epocas, learning_rate, hf_token, split, shard_size):
    """Inicia el servidor de entrenamiento distribuido"""
    server = DistributedTrainingServer(
        host, port, num_workers, epocas, learning_rate, hf_token, split, shard_size
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
    )