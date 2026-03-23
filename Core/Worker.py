"""
=============================================================================
  WORKER — ENTRENAMIENTO NEURAL DISTRIBUIDO CON SOCKETS
=============================================================================

El worker:
1. Se conecta al servidor
2. Para cada época recibe:
   - batch_id: identificador de su partición
   - Pesos globales del servidor
   - learning_rate
   - init_signal / stop_signal
3. Carga su partición de datos basada en batch_id
4. Entrena una iteración con esos datos
5. Calcula gradientes
6. Envía gradientes al servidor
7. Repite hasta recibir stop_signal

El dataset se carga localmente y se particiona igual que en el servidor,
de manera que cada worker sabe exactamente qué datos le corresponden.
=============================================================================
"""

import sys
import os
import numpy as np
import pickle
import socket
import struct
import time

# ── Agregar el directorio padre al path para acceder al paquete Utils ─────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Utils.DatasetHandling import cargar_mnist, preprocesar, particionar_dataset
from Utils.Fuctions import forward, backward, cross_entropy, precision
from Utils.WeightsHandling import inicializar_pesos
from Utils.Protocol import MessageFromServer, MessageFromWorker, WorkerReadyMessage, TrainingConfig
from Utils.messageHandling import send_message, receive_message

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DEL WORKER
# ─────────────────────────────────────────────────────────────────────────────

SERVER_HOST = TrainingConfig.server_host
SERVER_PORT = TrainingConfig.server_port
SOCKET_TIMEOUT = TrainingConfig.socket_timeout
NUM_PARTICIONES = TrainingConfig.num_particiones
SERVER_RANDOM_SEED = TrainingConfig.server_random_seed

# ─────────────────────────────────────────────────────────────────────────────
# WORKER PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class DistributedTrainingWorker:
    """
    Worker de Entrenamiento Distribuido.
    
    Se conecta al servidor y entrena su asignada partición del dataset.
    """
    
    def __init__(self, server_host, server_port, server_particiones,server_random_seed):
        self.server_host = server_host
        self.server_port = server_port
        
        # Datos
        self.num_particiones = server_particiones
        self.particiones = None
        self.X_k = None
        self.Y_k = None
        self.y_k = None
        self.batch_id = None
        
        # Socket
        self.socket = None
        self.random_seed = server_random_seed
    
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
    
    def load_dataset(self):
        """
        Carga el dataset MNIST y lo particiona.
        
        Cada worker obtendrá su partición basándose en batch_id
        (asignado por el servidor).
        """
        print(f"\n{'='*70}")
        print(f"  PASO 1: CARGANDO DATASET MNIST")
        print(f"{'='*70}")

        X_all, y_all = cargar_mnist()
        X_train, Y_train, y_train, _, _, _ = preprocesar(X_all, y_all)
        
        # Particionar igual que el servidor
        self.particiones = particionar_dataset(X_train, Y_train, y_train, self.num_particiones, self.random_seed)
        print(f"  ✓ Dataset cargado: {X_train.shape[0]} muestras")
        print(f"  ✓ Dataset particionado en {self.num_particiones} particiones (local)")

    
    def get_batch(self, batch_id):
        """
        Obtiene la partición asignada basada en batch_id.
        
        batch_id es asignado por el servidor como el orden de conexión.
        """
        if batch_id >= len(self.particiones):
            raise ValueError(f"batch_id {batch_id} fuera de rango [0, {len(self.particiones)-1}]")
        
        self.batch_id = batch_id
        self.X_k, self.Y_k, self.y_k = self.particiones[batch_id]
        
        print(f"\n{'='*70}")
        print(f"  ASIGNACIÓN DE DATOS")
        print(f"{'='*70}")
        print(f"  Worker batch_id : {batch_id}")
        print(f"  Datos asignados : {self.X_k.shape[0]} muestras")
        print(f"  Clases presentes: {np.unique(self.y_k)}")
    
    def train_epoch(self, epoch, W1_global, b1_global, W2_global, b2_global, learning_rate):
        """
        Entrena una época con su partición asignada.
        
        Entrenamiento = Forward + Backward + Gradient Computation
        
        Retorna:
            (dW1, db1, dW2, db2, loss, accuracy, training_time)
        """
        tiempo_inicio = time.time()
        
        # Forward Pass
        Z1, A1, Z2, A2 = forward(self.X_k, W1_global, b1_global, W2_global, b2_global)
        
        # Calcular pérdida
        loss = cross_entropy(A2, self.Y_k)
        
        # Calcular precisión
        y_pred = np.argmax(A2, axis=1)
        acc = precision(y_pred, self.y_k)
        
        # Backward Pass — calcular gradientes
        dW1, db1, dW2, db2 = backward(self.X_k, self.Y_k, Z1, A1, A2, W2_global)
        
        tiempo_entrenamiento = time.time() - tiempo_inicio
        
        return dW1, db1, dW2, db2, loss, acc, tiempo_entrenamiento
    
    def training_loop(self):
        """
        Bucle principal del worker.
        
        Recibe mensajes del servidor, entrena, envía gradientes.
        Continúa hasta recibir stop_signal.
        """
        print(f"\n{'='*70}")
        print(f"  PASO 2: ESPERANDO INSTRUCCIONES DEL SERVIDOR")
        print(f"{'='*70}\n") 
        
        while True:
            try:
                # Recibir mensaje del servidor
                print(f"  [Worker {self.batch_id}] Esperando mensaje...")
                message = receive_message(self.socket)

                # Asignar batch si es la primera vez
                if self.batch_id is None:
                    self.get_batch(message.batch_id)
                
                print(f"  ✓ Recibido: epoch={message.epoch}, init={message.init_signal}, "
                      f"stop={message.stop_signal}")
                
                # ┌─── HANDSHAKE: Responder a mensaje de sincronización ───┐
                if message.init_signal and message.epoch == 0:
                    ready_msg = WorkerReadyMessage(
                        worker_id=self.batch_id,
                        batch_id=self.batch_id,
                        dataset_size=self.X_k.shape[0]
                    )
                    print(f"    → Enviando confirmación de listo al servidor...")
                    send_message(self.socket, ready_msg)
                    print(f"    ✓ Confirmación enviada")
                    continue  # Volver a esperar el primer mensaje de entrenamiento
                # └─────────────────────────────────────────────────┘
                
                # Entrenar
                print(f"    Entrenando epoch {message.epoch}...")
                dW1, db1, dW2, db2, loss, acc, train_time = self.train_epoch(
                    message.epoch,
                    message.W1, message.b1, message.W2, message.b2,
                    message.learning_rate
                )
                
                # Crear respuesta
                response = MessageFromWorker(
                    worker_id=self.batch_id,
                    batch_id=self.batch_id,
                    epoch=message.epoch,
                    dW1=dW1,
                    db1=db1,
                    dW2=dW2,
                    db2=db2,
                    loss=loss,
                    accuracy=acc,
                    training_time=train_time
                )
                
                # Enviar gradientes
                print(f"    Enviando gradientes... (Loss={loss:.4f}, Acc={acc:.1f}%)")
                send_message(self.socket, response)
                
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
                print(f"\n  ✓ Conexión cerrada")
            except:
                pass
    
    def run(self):
        """Ejecución principal del worker."""
        print("\n" + "=" * 70)
        print("  WORKER — ENTRENAMIENTO NEURONAL DISTRIBUIDO")
        print("  Conectado a Servidor via Sockets")
        print("=" * 70)
        
        try:
            # Conectar al servidor
            self.connect_to_server()
            
            self.load_dataset()
            
            # Bucle de entrenamiento
            self.training_loop()
            
        except KeyboardInterrupt:
            print("\n\n  ✗ Worker interrumpido por usuario")
        except Exception as e:
            print(f"\n\n  ✗ Error crítico: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown()
            print(f"\n{'='*70}")
            print(f"  WORKER FINALIZADO")
            print(f"{'='*70}")


import argparse

# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

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
        default=NUM_PARTICIONES,
        help=f"Número de particiones/datos (por defecto: {NUM_PARTICIONES})",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=SERVER_RANDOM_SEED,
        help=f"Semilla aleatoria para particionado (por defecto: {SERVER_RANDOM_SEED})",
    )
    
    args = parser.parse_args()

    worker = DistributedTrainingWorker(
        server_host=args.host,
        server_port=args.port,
        server_particiones=args.particiones,
        server_random_seed=args.seed,
    )

    worker.run()
