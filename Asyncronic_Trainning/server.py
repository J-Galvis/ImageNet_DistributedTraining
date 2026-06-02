"""
=============================================================================
  SERVIDOR — ENTRENAMIENTO DISTRIBUIDO ASYNC-SGD CON STALENESS-AWARE LR
=============================================================================

Implements the n-softsync protocol from:
  Zhang et al. (2016) — "Staleness-aware Async-SGD for Distributed Deep Learning"

Architecture:
  * Parameter Server with one thread per worker.
  * Workers pull weights, compute gradients, and push updates independently.
  * The server accumulates c = ⌊λ/n⌋ gradients before applying an update.
  * Each gradient is scaled by 1/max(1, τ) where τ = global_step − worker_step.
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
import threading
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


class AsyncDistributedTrainingServer:
    """
    Asynchronous Parameter Server with Staleness-Aware Learning Rate.

    Uses n-softsync: the server updates after collecting c = ⌊λ/n⌋ gradients,
    each scaled by 1/max(1, τ) where τ is the gradient's staleness.
    """

    def __init__(self, host, port, num_workers, epocas, learning_rate,
                 hf_token, split='train', shard_size=10000, splitting_n=None,
                 pretrained=False, freeze_backbone=False):
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

        # ── Model ────────────────────────────────────────────────────────
        self.net = Net(num_classes=NUM_CLASSES, pretrained=pretrained, freeze_backbone=freeze_backbone)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)

        self.optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=1e-2
        )

        # Dataset size
        self.total_dataset_size = get_hf_split_size(split)

        # ── Step / batch arithmetic ──────────────────────────────────────
        self.steps_per_epoch = TrainingConfig.steps_per_epoch  # Default 10
        shard_size = SHARD_SIZE
        num_batches_per_worker = shard_size // BATCH_SIZE
        self.batches_per_step = max(1, num_batches_per_worker // self.steps_per_epoch)

        # Total global weight updates to perform across all epochs
        self.total_updates = self.epocas * self.steps_per_epoch

        # ── n-softsync parameters ────────────────────────────────────────
        # splitting_n defaults to λ (= Downpour ASGD, c=1)
        if splitting_n is None:
            splitting_n = num_workers
        self.splitting_n = splitting_n
        self.c = max(1, num_workers // splitting_n)  # update threshold

        # ── Threading primitives ─────────────────────────────────────────
        self.lock = threading.Lock()
        self.update_cond = threading.Condition(self.lock)

        # Global weight version counter
        self.global_step = 0

        # Gradient accumulator (reset after each global update)
        self.accumulated_grads: Dict[str, np.ndarray] = {}
        self.accumulated_buffers: List[Dict] = []
        self.grad_count = 0

        # Per-worker loss/accuracy tracking for logging
        self.step_losses = []
        self.step_accuracies = []
        self.step_staleness = []

        # ── LR scheduler ─────────────────────────────────────────────────
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.01,
            epochs=epocas,
            steps_per_epoch=self.steps_per_epoch,
            pct_start=0.3,
            div_factor=10,
            final_div_factor=100
        )

        # ── Connections ──────────────────────────────────────────────────
        self.worker_sockets: Dict[int, socket.socket] = {}
        self.worker_connected = {}
        self.shard_sizes = SHARD_SIZE

        # ── History for stats ────────────────────────────────────────────
        self.historial_intervalo_epochs = []
        self.historial_intervalo_times = []
        self.historial_intervalo_loss = []
        self.historial_intervalo_acc_train = []

        # Historial de pasos (step-level, one entry per global update point)
        self.step_loss_history = []
        self.step_accuracy_history = []
        self.step_times_history = []
        self.step_ids_history = []

        # ── Training is done flag ────────────────────────────────────────
        self.training_done = False

    # ─────────────────────────────────────────────────────────────────────
    # SOCKET SETUP
    # ─────────────────────────────────────────────────────────────────────

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
        print(f"  SERVIDOR ASYNC-SGD IMAGENET — ESCUCHANDO EN {self.host}:{self.port}")
        print(f"{'='*70}")
        print(f"  Protocolo: {self.splitting_n}-softsync (c={self.c})")
        print(f"  Esperando {self.num_workers} conexiones de workers...")

    # ─────────────────────────────────────────────────────────────────────
    # WORKER CONNECTION & HANDSHAKE  (synchronous, before training starts)
    # ─────────────────────────────────────────────────────────────────────

    def wait_for_workers(self):
        """
        Espera a que se conecten todos los workers.
        Envía mensaje de sincronización inicial y espera handshake.
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
                params = {name: tensor.cpu().numpy()
                          for name, tensor in self.net.state_dict().items()}
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

    # ─────────────────────────────────────────────────────────────────────
    # SNAPSHOT: thread-safe copy of current model weights
    # ─────────────────────────────────────────────────────────────────────

    def _snapshot_params(self):
        """Return a numpy dict of the current model state_dict (under lock)."""
        return {name: tensor.cpu().numpy()
                for name, tensor in self.net.state_dict().items()}

    # ─────────────────────────────────────────────────────────────────────
    # HANDLE WORKER  (one thread per worker)
    # ─────────────────────────────────────────────────────────────────────

    def handle_worker(self, worker_id):
        """
        Per-worker loop running in its own thread.

        Protocol:
          1. Send current weights + global_step as timestamp (pullWeights).
          2. Receive gradient from worker (pushGradient).
          3. Compute staleness, scale & accumulate gradient.
          4. If grad_count >= c: apply update, notify all threads.
             Else: wait for another thread to complete the update.
          5. Repeat from 1 until training is done.
        """
        sock = self.worker_sockets[worker_id]
        shard_size = self.shard_sizes

        while True:
            try:
                # ────── CHECK COMPLETION (under lock) ────────────────────
                with self.lock:
                    if self.training_done:
                        self._send_stop(sock, worker_id, shard_size)
                        break

                    # Snapshot weights and current global step
                    current_step = self.global_step
                    params = self._snapshot_params()

                # ────── SEND WEIGHTS (outside lock for parallelism) ──────
                # Compute which batches this worker should process
                # Each "round" a worker processes batches_per_step batches
                # from its persistent iterator — batch_ids are informational.
                step_in_epoch = current_step % self.steps_per_epoch
                epoch_num = (current_step // self.steps_per_epoch) + 1

                num_batches = shard_size // BATCH_SIZE
                step_start = step_in_epoch * self.batches_per_step
                step_end = min((step_in_epoch + 1) * self.batches_per_step, num_batches)
                batch_ids = list(range(step_start, step_end))

                message = MessageFromServer(
                    batch_ids=batch_ids,
                    epoch=epoch_num,
                    step_id=current_step,           # ← weight timestamp j
                    steps_per_epoch=self.steps_per_epoch,
                    init_signal=False,
                    stop_signal=False,
                    learning_rate=self.learning_rate,
                    shard_size=shard_size,
                    params=params,
                    hf_token=self.hf_token,
                    worker_id=worker_id,
                    num_workers=self.num_workers
                )

                send_message(sock, message)

                # ────── RECEIVE GRADIENT ─────────────────────────────────
                response = receive_message(sock)

                # ────── ACCUMULATE UNDER LOCK ────────────────────────────
                with self.update_cond:
                    if self.training_done:
                        # Another thread finished training while we were computing
                        self._send_stop(sock, worker_id, shard_size)
                        break

                    # Compute staleness τ = i − j
                    worker_timestamp = response.step_id  # j: step when worker pulled weights
                    staleness = max(1, self.global_step - worker_timestamp)
                    lr_scale = 1.0 / staleness

                    # Scale and accumulate gradients
                    for name, grad in response.gradients.items():
                        scaled_grad = grad * lr_scale
                        if name in self.accumulated_grads:
                            self.accumulated_grads[name] += scaled_grad
                        else:
                            self.accumulated_grads[name] = scaled_grad.copy()

                    self.grad_count += 1

                    # Accumulate BN buffers
                    if getattr(response, 'buffers', None):
                        self.accumulated_buffers.append(response.buffers)

                    # Track metrics
                    self.step_losses.append(response.loss)
                    self.step_accuracies.append(response.accuracy)
                    self.step_staleness.append(staleness)

                    print(f"    ← Worker {worker_id}: loss={response.loss:.4f}, "
                          f"acc={response.accuracy:.2f}%, τ={staleness}, "
                          f"lr_scale={lr_scale:.4f}, "
                          f"grad_count={self.grad_count}/{self.c}")

                    # ── Check threshold ──────────────────────────────────
                    if self.grad_count >= self.c:
                        # Average the accumulated gradients
                        avg_grads = {name: g / self.grad_count
                                     for name, g in self.accumulated_grads.items()}

                        # Apply the update
                        self._apply_gradients(avg_grads)

                        # Apply BN buffers
                        if self.accumulated_buffers:
                            self._apply_buffers(self.accumulated_buffers)

                        # Step scheduler
                        self.scheduler.step()

                        # Log update
                        avg_loss = np.mean(self.step_losses) if self.step_losses else 0.0
                        avg_acc = np.mean(self.step_accuracies) if self.step_accuracies else 0.0
                        avg_stale = np.mean(self.step_staleness) if self.step_staleness else 0.0
                        current_epoch = (self.global_step // self.steps_per_epoch) + 1
                        step_in_ep = (self.global_step % self.steps_per_epoch) + 1

                        total_time = time.time() - self.training_start

                        # Registrar métricas a nivel de paso (step-level)
                        self.step_loss_history.append(round(avg_loss, 6))
                        self.step_accuracy_history.append(round(avg_acc, 6))
                        self.step_times_history.append(round(total_time, 6))
                        self.step_ids_history.append([current_epoch, step_in_ep])

                        print(f"\n  {'─'*68}")
                        print(f"  GLOBAL UPDATE #{self.global_step + 1}/{self.total_updates} "
                              f"(Epoch {current_epoch}, Step {step_in_ep}/{self.steps_per_epoch})")
                        print(f"    Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.2f}% "
                              f"| Avg Staleness: {avg_stale:.1f}")
                        print(f"  {'─'*68}\n")

                        # Evaluate / log at epoch boundaries
                        self.evaluate_global_model(current_epoch, total_time, avg_loss, avg_acc)

                        # Increment global step
                        self.global_step += 1

                        # Reset accumulators
                        self.accumulated_grads = {}
                        self.accumulated_buffers = []
                        self.grad_count = 0
                        self.step_losses = []
                        self.step_accuracies = []
                        self.step_staleness = []

                        # Check if training is complete
                        if self.global_step >= self.total_updates:
                            self.training_done = True

                        # Wake up all threads waiting for this update
                        self.update_cond.notify_all()
                    else:
                        # Wait until another thread triggers the update
                        # Use a timeout to re-check training_done flag
                        self.update_cond.wait(timeout=60.0)

            except (ConnectionError, BrokenPipeError, OSError) as e:
                print(f"\n  ✗ Worker {worker_id} disconnected: {e}")
                break
            except Exception as e:
                print(f"\n  ✗ Error in worker {worker_id} handler: {e}")
                import traceback
                traceback.print_exc()
                break

    def _send_stop(self, sock, worker_id, shard_size):
        """Send a stop signal to a worker."""
        try:
            params = self._snapshot_params()
            message = MessageFromServer(
                batch_ids=[],
                epoch=self.epocas,
                step_id=self.global_step,
                steps_per_epoch=self.steps_per_epoch,
                init_signal=False,
                stop_signal=True,
                learning_rate=self.learning_rate,
                shard_size=shard_size,
                params=params,
                hf_token=self.hf_token,
                worker_id=worker_id,
                num_workers=self.num_workers
            )
            send_message(sock, message)
            print(f"    → Stop signal enviado a worker {worker_id}")
        except Exception as e:
            print(f"    ✗ Error enviando stop a worker {worker_id}: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # GRADIENT APPLICATION
    # ─────────────────────────────────────────────────────────────────────

    def _apply_gradients(self, avg_grads):
        """Apply averaged gradients to the model (must be called under lock)."""
        self.optimizer.zero_grad()

        for name, param in self.net.named_parameters():
            if name in avg_grads:
                param.grad = torch.tensor(
                    avg_grads[name], dtype=param.dtype, device=param.device
                )

        # Log gradient norm before clipping
        total_norm = 0.0
        for p in self.net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f"    ℹ Gradient norm BEFORE clipping: {total_norm:.6f}")

        # Clip gradients
        clipped_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
        print(f"    ℹ Gradient clipping applied (norm={clipped_norm:.6f})")

        # Update weights
        self.optimizer.step()

    def _apply_buffers(self, buffer_list):
        """Average BN running stats from accumulated buffers and apply."""
        if not buffer_list or not buffer_list[0]:
            return

        current_state = self.net.state_dict()
        new_state = dict(current_state)

        for buf_name in buffer_list[0].keys():
            buf_arrays = [b[buf_name] for b in buffer_list if buf_name in b]
            if not buf_arrays:
                continue

            target = current_state.get(buf_name)
            if target is None:
                continue

            if 'num_batches_tracked' in buf_name:
                merged = torch.tensor(buf_arrays[0], dtype=target.dtype, device=target.device)
            else:
                avg = sum(buf_arrays) / len(buf_arrays)
                merged = torch.tensor(avg, dtype=target.dtype, device=target.device)

            new_state[buf_name] = merged

        self.net.load_state_dict(new_state)
        print(f"    ℹ BN running stats synced from {len(buffer_list)} worker contributions")

    # ─────────────────────────────────────────────────────────────────────
    # EVALUATION / LOGGING
    # ─────────────────────────────────────────────────────────────────────

    def evaluate_global_model(self, epoch, tiempo_actual, avg_loss, avg_acc):
        """Log epoch-level metrics."""
        if epoch % INTERVALO_LOG == 0 or epoch == 1:
            # Only log once per epoch (avoid duplicates from multiple updates in same epoch)
            if epoch not in self.historial_intervalo_epochs:
                self.historial_intervalo_epochs.append(epoch)
                self.historial_intervalo_times.append(round(tiempo_actual, 6))
                self.historial_intervalo_loss.append(round(avg_loss, 6))
                self.historial_intervalo_acc_train.append(round(avg_acc, 6))

                print(f"\n  {'─'*68}")
                print(f"  EVALUACIÓN GLOBAL — ÉPOCA {epoch}/{self.epocas}")
                print(f"  {'─'*68}")
                print(f"    ✓ GLOBAL → Loss: {avg_loss:.4f} | Acc (train): {avg_acc:.2f}%")
                print(f"    ⏱ Tiempo acumulado: {tiempo_actual:.2f}s")

    # ─────────────────────────────────────────────────────────────────────
    # MAIN TRAINING LOOP  (spawns threads)
    # ─────────────────────────────────────────────────────────────────────

    def training_loop(self):
        """
        Main async training loop.

        Spawns one thread per worker and waits for all threads to complete.
        """
        print(f"\n{'='*70}")
        print(f"  INICIANDO ENTRENAMIENTO ASYNC-SGD ({self.splitting_n}-softsync)")
        print(f"  Workers: {self.num_workers} | c={self.c} | "
              f"Total updates: {self.total_updates}")
        print(f"{'='*70}\n")

        self.training_start = time.time()

        try:
            # Spawn one handler thread per worker
            threads = []
            for worker_id in range(self.num_workers):
                t = threading.Thread(
                    target=self.handle_worker,
                    args=(worker_id,),
                    name=f"worker-{worker_id}",
                    daemon=True
                )
                t.start()
                threads.append(t)
                print(f"  ✓ Thread para worker {worker_id} iniciado")

            # Wait for all threads to finish
            for t in threads:
                t.join()

            print(f"\n{'='*70}")
            print(f"  ENTRENAMIENTO COMPLETADO")
            print(f"{'='*70}\n")

            # Compute total training time
            tiempo_total = time.time() - self.training_start

            nombre_modelo = input("\n  Ingrese un nombre para guardar el modelo: ").strip()

            # Save PyTorch model
            model_path = f"models/{nombre_modelo}_imagenet.pt"
            os.makedirs("models", exist_ok=True)
            torch.save(self.net.state_dict(), model_path)

            # Save model with complete metrics
            guardar_modelo(
                None, None, None, None,
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
                    'architecture': 'ImageNet ResNet - Async-SGD with Sockets',
                    'protocol': f'{self.splitting_n}-softsync (c={self.c})',
                    'server_host': self.host,
                    'server_port': self.port,
                    'tiempo_total_segundos': tiempo_total,
                    'total_global_updates': self.global_step,
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
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Close connections
            for worker_id, sock in self.worker_sockets.items():
                try:
                    sock.close()
                except:
                    pass
            self.server_socket.close()


def start_server(host, port, num_workers, epocas, learning_rate,
                 hf_token, split, shard_size, splitting_n, pretrained, freeze_backbone):
    """Inicia el servidor de entrenamiento distribuido async."""
    server = AsyncDistributedTrainingServer(
        host, port, num_workers, epocas, learning_rate,
        hf_token, split, shard_size, splitting_n,
        pretrained=pretrained, freeze_backbone=freeze_backbone
    )
    server.setup_socket_server()
    server.wait_for_workers()
    server.training_loop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Servidor para entrenamiento distribuido Async-SGD de ImageNet."
    )

    parser.add_argument(
        "--host", "-H",
        default=SERVER_HOST,
        help=f"Host en el que el servidor escuchará (por defecto: {SERVER_HOST})",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=SERVER_PORT,
        help=f"Puerto en el que el servidor escuchará (por defecto: {SERVER_PORT})",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=NUM_WORKERS,
        help=f"Número de workers λ (por defecto: {NUM_WORKERS})",
    )
    parser.add_argument(
        "--epocas", "-e",
        type=int,
        default=NUM_EPOCHS,
        help=f"Cantidad de épocas para entrenar (por defecto: {NUM_EPOCHS})",
    )
    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help=f"Tasa de aprendizaje base α₀ (por defecto: {LEARNING_RATE})",
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
        "--splitting-n",
        type=int,
        default=None,
        help="Parámetro n para n-softsync. "
             "c = ⌊λ/n⌋ gradientes se acumulan antes de actualizar. "
             "Default: n=λ (Downpour ASGD, c=1). "
             "n=1 da SGD síncrono (c=λ).",
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
        args.splitting_n,
        args.pretrained,
        args.freeze_backbone,
    )