"""
=============================================================================
  SERVIDOR — ENTRENAMIENTO NEURONAL DISTRIBUIDO CON SOCKETS
=============================================================================

El servidor:
1. Carga y particiona el dataset MNIST en K particiones
2. Abre un socket servidor esperando conexiones de workers
3. Para cada época:
   - Envía a cada worker: epoch, pesos globales, learning_rate, init/stop signal
   - Recibe de cada worker: gradientes calculados
   - Promedia los gradientes
   - Actualiza los pesos globales
4. Al final, evaluación en test
=============================================================================
"""

import sys
import os
import numpy as np
import pickle
import socket
import struct
import time
import argparse
from typing import Dict, List, Tuple

# ── Agregar el directorio padre al path para acceder al paquete Utils ─────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Utils.DatasetHandling import cargar_mnist, preprocesar, particionar_dataset
from Utils.Fuctions import forward, backward, cross_entropy, precision, predecir
from Utils.WeightsHandling import inicializar_pesos, actualizar_pesos
from Utils.ModelPersistence import guardar_modelo
from Utils.Protocol import MessageFromServer, MessageFromWorker, WorkerReadyMessage, TrainingConfig
from Utils.messageHandling import send_message, receive_message

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DEL SERVIDOR
# ─────────────────────────────────────────────────────────────────────────────

NUM_PARTICIONES = TrainingConfig.num_particiones
EPOCAS = TrainingConfig.epocas
LEARNING_RATE = TrainingConfig.learning_rate
INTERVALO_LOG = TrainingConfig.intervalo_log
SERVER_HOST = TrainingConfig.server_host
SERVER_PORT = TrainingConfig.server_port
SOCKET_TIMEOUT = TrainingConfig.socket_timeout
SERVER_RANDOM_SEED = TrainingConfig.server_random_seed


def promediar_gradientes(lista_gradientes):
    """
    Promedia los gradientes de K workers tras entrenar con sus particiones.
    
    Parámetros:
        lista_gradientes: lista de tuplas (dW1, db1, dW2, db2)
    
    Retorna:
        (dW1_prom, db1_prom, dW2_prom, db2_prom)
    """
    K = len(lista_gradientes)
    
    lista_dW1 = [grad[0] for grad in lista_gradientes]
    lista_db1 = [grad[1] for grad in lista_gradientes]
    lista_dW2 = [grad[2] for grad in lista_gradientes]
    lista_db2 = [grad[3] for grad in lista_gradientes]
    
    dW1_prom = np.mean(np.array(lista_dW1), axis=0)
    db1_prom = np.mean(np.array(lista_db1), axis=0)
    dW2_prom = np.mean(np.array(lista_dW2), axis=0)
    db2_prom = np.mean(np.array(lista_db2), axis=0)
    
    return dW1_prom, db1_prom, dW2_prom, db2_prom


# ─────────────────────────────────────────────────────────────────────────────
# SERVIDOR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class DistributedTrainingServer:
    """
    Servidor de Entrenamiento Distribuido.
    
    Maneja conexiones de múltiples workers y coordina el entrenamiento federado.
    """
    
    def __init__(self, host, port, num_particiones, epocas, learning_rate, intervalo_log):
        self.host = host
        self.port = port
        self.num_particiones = num_particiones
        self.epocas = epocas
        self.learning_rate = learning_rate
        self.intervalo_log = intervalo_log
        
        # Pesos globales
        self.W1, self.b1, self.W2, self.b2 = inicializar_pesos()
        
        # Conexiones de workers
        self.worker_sockets: Dict[int, socket.socket] = {}  # batch_id -> socket
        self.worker_connected = {}  # batch_id -> bool
        
        
        # Historial de checkpoints por INTERVALO_LOG
        self.historial_intervalo_epochs = []      # Épocas en las que se guardó
        self.historial_intervalo_times = []       # Tiempos acumulados
        self.historial_intervalo_acc_train = []   # Precisión en entrenamiento
        self.historial_intervalo_loss = []        # Loss

        # Datos de entrenamiento
        self.particiones = None
        self.X_test = None
        self.Y_test = None
        self.y_test = None
    
    def setup_socket_server(self):
        """Configura el socket servidor."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(self.num_particiones)
        self.server_socket.settimeout(SOCKET_TIMEOUT)
        
        print(f"\n{'='*70}")
        print(f"  SERVIDOR DISTRIBUIDO — ESCUCHANDO EN {self.host}:{self.port}")
        print(f"{'='*70}")
        print(f"  Esperando {self.num_particiones} conexiones de workers...")
    
    def wait_for_workers(self):
        """
        Espera a que se conecten todos los workers.
        Asigna batch_id basado en el orden de conexión.
        Envía mensaje de sincronización inicial a cada worker.
        """
        # FASE 1: Aceptar todas las conexiones
        for batch_id in range(self.num_particiones):
            try:
                print(f"\n  [Esperando] Worker batch_id={batch_id}...")
                client_socket, client_address = self.server_socket.accept()
                client_socket.settimeout(SOCKET_TIMEOUT)
                
                self.worker_sockets[batch_id] = client_socket
                self.worker_connected[batch_id] = True
                
                print(f"  ✓ Worker {batch_id} conectado desde {client_address}")

                
            except socket.timeout:
                print(f"\n  ✗ Timeout esperando worker {batch_id}")
                raise
            except Exception as e:
                print(f"\n  ✗ Error aceptando conexión: {e}")
                raise
        
        # FASE 2: Enviar mensaje de sincronización a todos los workers
        print(f"\n  {'─'*68}")
        print(f"  FASE DE SINCRONIZACIÓN — Enviando señales de inicio a workers")
        print(f"  {'─'*68}")
        
        for batch_id in range(self.num_particiones):
            try:
                # Crear mensaje de sincronización (epoch=0, init_signal=True)
                message = MessageFromServer(
                    batch_id=batch_id,
                    epoch=0,
                    init_signal=True,
                    stop_signal=False,
                    learning_rate=self.learning_rate,
                    W1=np.copy(self.W1),
                    b1=np.copy(self.b1),
                    W2=np.copy(self.W2),
                    b2=np.copy(self.b2)
                )
                
                # Enviar mensaje de sincronización
                sock = self.worker_sockets[batch_id]
                send_message(sock, message)
                
                print(f"    → Sincronización enviada a worker {batch_id}")
                
            except Exception as e:
                print(f"    ✗ Error sincronizando worker {batch_id}: {e}")
                raise
        
        # FASE 3: Esperar confirmación (handshake) de todos los workers
        print(f"\n  {'─'*68}")
        print(f"  FASE DE HANDSHAKE — Esperando confirmación de workers")
        print(f"  {'─'*68}")
        
        for batch_id in range(self.num_particiones):
            try:
                sock = self.worker_sockets[batch_id]
                ready_msg: WorkerReadyMessage = receive_message(sock)
                
                print(f"    ✓ Worker {batch_id} listo (dataset_size={ready_msg.dataset_size})")
                
            except Exception as e:
                print(f"    ✗ Error esperando confirmación de worker {batch_id}: {e}")
                raise
        
        print(f"  ✓ Todos los workers sincronizados y listos para entrenar")
    
    def distribute_work(self, epoch):
        """
        Distribuye trabajo a todos los workers para una época.
        
        Envía a cada worker: epoch, pesos globales, learning_rate, etc.
        """
        print(f"\n  {'─'*68}")
        print(f"  ÉPOCA {epoch}/{self.epocas} — DISTRIBUYENDO TRABAJO A WORKERS")
        print(f"  {'─'*68}")
        
        for batch_id in range(self.num_particiones):
            try:
                # Crear mensaje para el worker
                message = MessageFromServer(
                    batch_id=batch_id,
                    epoch=epoch,
                    init_signal=(epoch == 1),
                    stop_signal=(epoch == self.epocas),
                    learning_rate=self.learning_rate,
                    W1=np.copy(self.W1),
                    b1=np.copy(self.b1),
                    W2=np.copy(self.W2),
                    b2=np.copy(self.b2)
                )
                
                # Enviar al worker
                sock = self.worker_sockets[batch_id]
                send_message(sock, message)
                
                print(f"    ✓ Enviado a worker {batch_id}: epoch={epoch}, pesos globales")
                
            except Exception as e:
                print(f"    ✗ Error enviando a worker {batch_id}: {e}")
                raise
    
    def collect_results(self):
        """
        Recolecta resultados de todos los workers para la época actual.
        
        Recibe gradientes y métricas de cada worker.
        """
        gradientes_epoca = []
        
        print(f"\n  {'─'*68}")
        print(f"  RECOLECTANDO RESULTADOS DE WORKERS")
        print(f"  {'─'*68}")
        
        for batch_id in range(self.num_particiones):
            try:
                sock = self.worker_sockets[batch_id]
                message: MessageFromWorker = receive_message(sock)
                
                gradientes_epoca.append((message.dW1, message.db1, message.dW2, message.db2))
                
                print(f"    ✓ Worker {batch_id} (epoch {message.epoch}): "
                      f"Loss={message.loss:.4f}, Acc={message.accuracy:.1f}%, "
                      f"Time={message.training_time:.4f}s")
                
            except Exception as e:
                print(f"    ✗ Error recibiendo de worker {batch_id}: {e}")
                raise
        
        return gradientes_epoca
    
    def update_global_weights(self, gradientes_epoca, epoch):
        """
        Promedia los gradientes y actualiza los pesos globales.
        
        Pasos:
        1. Promediar gradientes de todos los workers
        2. Actualizar pesos globales usando descenso de gradiente
        3. Evaluar en entrenamiento y test
        """
        # Promediar gradientes
        dW1_prom, db1_prom, dW2_prom, db2_prom = promediar_gradientes(gradientes_epoca)
        
        # Actualizar pesos globales
        self.W1, self.b1, self.W2, self.b2 = actualizar_pesos(
            self.W1, self.b1, self.W2, self.b2,
            dW1_prom, db1_prom, dW2_prom, db2_prom,
            self.learning_rate
        )
        
        print(f"    ✓ Pesos globales actualizados (promediado de {self.num_particiones} workers)")
    
    def evaluate_global_model(self, X_train, Y_train, y_train, epoch, tiempo_actual):
        """
        Evalúa el modelo global en entrenamiento y test.
        
        Parámetros:
            tiempo_actual: float, tiempo transcurrido desde el inicio del entrenamiento
        """
        # Evaluación en entrenamiento
        Z1_all, A1_all, Z2_all, A2_all = forward(X_train, self.W1, self.b1, self.W2, self.b2)
        loss = cross_entropy(A2_all, Y_train)
        acc_train = precision(np.argmax(A2_all, axis=1), y_train)
        
        # Evaluación en test
        y_pred_test = predecir(self.X_test, self.W1, self.b1, self.W2, self.b2)
        acc_test = precision(y_pred_test, self.y_test)
        
        self.historial_intervalo_epochs.append(epoch)
        self.historial_intervalo_times.append(round(tiempo_actual, 6))
        self.historial_intervalo_acc_train.append(round(acc_train, 6))
        self.historial_intervalo_loss.append(round(loss, 6))
            
        print(f"\n  {'─'*68}")
        print(f"  EVALUACIÓN GLOBAL — ÉPOCA {epoch}/{self.epocas}")
        print(f"  {'─'*68}")

        
        print(f"  {'─'*68}")
        print(f"    ✓ GLOBAL → Loss: {loss:.4f} │ "
                f"Acc Train: {acc_train:.1f}% │ "
                f"Acc Test: {acc_test:.1f}%")
        print(f"    ⏱ Tiempo acumulado: {tiempo_actual:.2f}s")
    
    def shutdown(self):
        """Cierra todas las conexiones de workers."""
        print(f"\n{'='*70}")
        print(f"  APAGANDO SERVIDOR")
        print(f"{'='*70}")
        
        for batch_id, sock in self.worker_sockets.items():
            try:
                sock.close()
                print(f"  ✓ Conexión con worker {batch_id} cerrada")
            except:
                pass
        
        self.server_socket.close()
        print(f"  ✓ Socket servidor cerrado")
    
    def train(self, X_train, Y_train, y_train, X_test, Y_test, y_test):
        """
        Ejecuta el bucle principal de entrenamiento distribuido.
        """
        self.X_test = X_test
        self.Y_test = Y_test
        self.y_test = y_test
        
        print("\n" + "=" * 70)
        print("  ALGORITMO DISTRIBUIDO — ENTRENAMIENTO FEDERADO CON SOCKETS")
        print("=" * 70)
        print(f"  Particiones       : {self.num_particiones}")
        print(f"  Épocas totales    : {self.epocas}")
        print(f"  Learning Rate     : {self.learning_rate}")
        print(f"  Muestras totales  : {X_train.shape[0]}")
        print(f"  Muestras por worker: ~{X_train.shape[0] // self.num_particiones}")
        
        # Esperar a que se conecten todos los workers
        self.wait_for_workers()
        
        print(f"\n{'='*70}")
        print(f"  PASO 1: INICIALIZANDO PESOS GLOBALES")
        print(f"{'='*70}")
        print(f"  W1: {self.W1.shape}  (pesos entrada → oculta)")
        print(f"  b1: {self.b1.shape}  (sesgos capa oculta)")
        print(f"  W2: {self.W2.shape}  (pesos oculta → salida)")
        print(f"  b2: {self.b2.shape}  (sesgos capa salida)")
        
        print(f"\n{'='*70}")
        print(f"  PASO 2: ENTRENAMIENTO FEDERADO DISTRIBUIDO")
        print(f"{'='*70}")
        
        # Inicio del cronómetro de entrenamiento
        inicio_entrenamiento = time.time()
        
        # Bucle principal de épocas
        for epoch in range(1, self.epocas + 1):
            tiempo_inicio_epoca = time.time()
            
            # Distribuir trabajo a workers
            self.distribute_work(epoch)
            
            # Recolectar resultados
            gradientes_epoca = self.collect_results()
            
            # Actualizar pesos globales
            self.update_global_weights(gradientes_epoca, epoch)
            
            # Tiempo transcurrido desde el inicio del entrenamiento
            tiempo_actual = time.time() - inicio_entrenamiento
            
            # Evaluar modelo global
            self.evaluate_global_model(X_train, Y_train, y_train, epoch, tiempo_actual)
            
            tiempo_epoca = time.time() - tiempo_inicio_epoca
            
            if epoch % self.intervalo_log == 0 or epoch == 1:
                print(f"    Tiempo época: {tiempo_epoca:.2f}s\n")
        
        # Evaluación final
        print(f"\n{'='*70}")
        print(f"  EVALUACIÓN FINAL")
        print(f"{'='*70}")
        y_pred_test = predecir(self.X_test, self.W1, self.b1, self.W2, self.b2)
        acc_final = precision(y_pred_test, self.y_test)
        print(f"\n  ✓ Precisión FINAL del modelo en TEST: {acc_final:.2f}%")
        
        # Calcular tiempo total de entrenamiento
        tiempo_total = time.time() - inicio_entrenamiento

        nombre_modelo = input("\n  Ingrese un nombre para guardar el modelo: ").strip()
        
        # Guardar modelo con métricas completas
        guardar_modelo(
            self.W1, self.b1, self.W2, self.b2,
            nombre_modelo=nombre_modelo,
            precision_test=acc_final,
            epocas=self.epocas,
            learning_rate=self.learning_rate,
            training_time=tiempo_total,
            info_extra={
                'num_particiones': self.num_particiones,
                'architecture': 'Distributed with Sockets',
                'server_host': self.host,
                'server_port': self.port,
                'tiempo_total_segundos': tiempo_total,
                'historial_intervalo_epochs': self.historial_intervalo_epochs,
                'historial_intervalo_times': self.historial_intervalo_times,
                'historial_intervalo_acc_train': self.historial_intervalo_acc_train,
                'historial_intervalo_loss': self.historial_intervalo_loss,
            }
        )

# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # permitir pasar parámetros por línea de comandos para el servidor
    parser = argparse.ArgumentParser(
        description="Servidor para entrenamiento distribuido."
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
        "--particiones",
        "-n",
        type=int,
        default=NUM_PARTICIONES,
        help=f"Número de particiones/datos (por defecto: {NUM_PARTICIONES})",
    )
    parser.add_argument(
        "--epocas",
        "-e",
        type=int,
        default=EPOCAS,
        help=f"Cantidad de épocas para entrenar (por defecto: {EPOCAS})",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help=f"Tasa de aprendizaje (por defecto: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--intervalo-log",
        type=int,
        default=INTERVALO_LOG,
        help=f"Intervalo de logging de métricas (por defecto: {INTERVALO_LOG})",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=SERVER_RANDOM_SEED,
        help=f"Semilla aleatoria para particionado (por defecto: {SERVER_RANDOM_SEED})",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  SERVIDOR — ENTRENAMIENTO NEURONAL DISTRIBUIDO")
    print("  Red Neuronal Federada — MNIST — Distribución por Sockets")
    print("=" * 70)
    
    # Cargar dataset
    print(f"\n{'='*70}")
    print(f"  CARGANDO DATASET MNIST")
    print(f"{'='*70}")
    X_all, y_all = cargar_mnist()
    X_train, Y_train, y_train, X_test, Y_test, y_test = preprocesar(X_all, y_all)
    
    # Crear servidor
    server = DistributedTrainingServer(
        host=args.host,
        port=args.port,
        num_particiones=args.particiones,
        epocas=args.epocas,
        learning_rate=args.lr,
        intervalo_log=args.intervalo_log
    )
    
    # Configurar socket
    server.setup_socket_server()
    
    try:
        # Entrenar
        server.train(X_train, Y_train, y_train, X_test, Y_test, y_test)
    
    except KeyboardInterrupt:
        print("\n\n  ✗ Entrenamiento interrumpido por usuario")
    except Exception as e:
        print(f"\n\n  ✗ Error en entrenamiento: {e}")
        import traceback
        traceback.print_exc()
    finally:
        server.shutdown()
