"""
Protocolo de Comunicación para Entrenamiento Distribuido CIFAR10
=================================================================

Define la estructura de los mensajes intercambiados entre Server y Workers
mediante sockets y pickle.
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class MessageFromServer:
    """
    Mensaje enviado por el servidor al worker.
    
    Atributos:
        batch_ids: list - Lista de identificadores de batch a procesar
        epoch: int - Número de época actual
        init_signal: bool - True al inicio del entrenamiento
        stop_signal: bool - True para detener el worker
        learning_rate: float - Tasa de aprendizaje
        params: Dict - Parámetros del modelo (PyTorch state_dict)
    """
    batch_ids: list
    epoch: int
    init_signal: bool
    stop_signal: bool
    learning_rate: float
    params: Dict
    
    def __repr__(self):
        return (f"MessageFromServer(epoch={self.epoch}, batches={len(self.batch_ids)}, "
                f"init={self.init_signal}, stop={self.stop_signal})")


@dataclass
class MessageFromWorker:
    """
    Mensaje enviado por el worker al servidor.
    
    Atributos:
        worker_id: int - Identificador del worker (basado en orden de conexión)
        epoch: int - Número de época procesada
        gradients: Dict - Gradientes acumulados para cada parámetro
        loss: float - Pérdida computada en los batches
        accuracy: float - Precisión en los batches (%)
        training_time: float - Tiempo de entrenamiento en segundos
    """
    worker_id: int
    epoch: int
    gradients: Dict
    loss: float
    accuracy: float
    training_time: float
    
    def __repr__(self):
        return (f"MessageFromWorker(worker_id={self.worker_id}, epoch={self.epoch}, "
                f"loss={self.loss:.4f}, acc={self.accuracy:.1f}%)")


@dataclass
class WorkerReadyMessage:
    """
    Mensaje de confirmación enviado por el worker después de sincronización.
    
    Confirma que el worker ha recibido correctamente el mensaje de sincronización
    y está listo para comenzar el entrenamiento.
    
    Atributos:
        worker_id: int - Identificador del worker
        dataset_size: int - Tamaño de la partición asignada
    """
    worker_id: int
    dataset_size: int
    
    def __repr__(self):
        return (f"WorkerReadyMessage(worker_id={self.worker_id}, "
                f"dataset_size={self.dataset_size})")


@dataclass
class TrainingConfig:
    """Configuración global de entrenamiento distribuido."""
    num_workers: int = 1
    epocas: int = 60
    learning_rate: float = 0.001
    intervalo_log: int = 1
    server_host: str = 'localhost'
    server_port: int = 6000
    socket_timeout: int = 500 # segundos
    batch_size: int = 32
    save_file: str = './Results/cifar10_trained_model.pth'
