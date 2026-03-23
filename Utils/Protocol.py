"""
Protocolo de Comunicación para Entrenamiento Distribuido
=========================================================

Define la estructura de los mensajes intercambiados entre Server y Workers
mediante sockets y pickle.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class MessageFromServer:
    """
    Mensaje enviado por el servidor al worker.
    
    Atributos:
        batch_id: int - Identificador del batch/partición (0, 1, 2, 3...)
        epoch: int - Número de época actual
        init_signal: bool - True al inicio del entrenamiento
        stop_signal: bool - True para detener el worker
        learning_rate: float - Tasa de aprendizaje
        W1: np.array - Pesos capa oculta (entrada → oculta)
        b1: np.array - Sesgos capa oculta
        W2: np.array - Pesos capa salida (oculta → salida)
        b2: np.array - Sesgos capa salida
    """
    batch_id: int
    epoch: int
    init_signal: bool
    stop_signal: bool
    learning_rate: float
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    
    def __repr__(self):
        return (f"MessageFromServer(batch_id={self.batch_id}, epoch={self.epoch}, "
                f"init={self.init_signal}, stop={self.stop_signal}, "
                f"lr={self.learning_rate})")


@dataclass
class MessageFromWorker:
    """
    Mensaje enviado por el worker al servidor.
    
    Atributos:
        worker_id: int - Identificador del worker (basado en batch_id)
        batch_id: int - Identificador del batch procesado
        epoch: int - Número de época procesada
        dW1: np.array - Gradiente para W1
        db1: np.array - Gradiente para b1
        dW2: np.array - Gradiente para W2
        db2: np.array - Gradiente para b2
        loss: float - Pérdida computada en el batch
        accuracy: float - Precisión en el batch (%)
    """
    worker_id: int
    batch_id: int
    epoch: int
    dW1: np.ndarray
    db1: np.ndarray
    dW2: np.ndarray
    db2: np.ndarray
    loss: float
    accuracy: float
    training_time: float
    
    def __repr__(self):
        return (f"MessageFromWorker(worker_id={self.worker_id}, batch_id={self.batch_id}, "
                f"epoch={self.epoch}, loss={self.loss:.4f}, acc={self.accuracy:.1f}%)")


@dataclass
class WorkerReadyMessage:
    """
    Mensaje de confirmación enviado por el worker después de sincronización.
    
    Confirma que el worker ha recibido correctamente el mensaje de sincronización
    y está listo para comenzar el entrenamiento.
    
    Atributos:
        worker_id: int - Identificador del worker
        batch_id: int - Batch_id asignado
        dataset_size: int - Tamaño de la partición asignada
    """
    worker_id: int
    batch_id: int
    dataset_size: int
    
    def __repr__(self):
        return (f"WorkerReadyMessage(worker_id={self.worker_id}, "
                f"batch_id={self.batch_id}, dataset_size={self.dataset_size})")


@dataclass
class TrainingConfig:
    num_particiones: int = 1
    epocas: int = 100
    learning_rate: float = 0.1
    intervalo_log: int = 10
    server_host: str = 'localhost'
    server_port: int = 9999
    socket_timeout: int = 60  # segundos
    server_random_seed: int = 42  # Semilla