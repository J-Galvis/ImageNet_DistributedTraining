"""
Protocolo de Comunicación para Entrenamiento Distribuido ImageNet
==================================================================

Define la estructura de los mensajes intercambiados entre Server y Workers
mediante sockets y pickle para entrenamiento con ImageNet en modo shards.
"""

from dataclasses import dataclass, field
from typing import Dict
import numpy as np

# Constante para tamaño de shard de datos
SHARD_SIZE = 50_000  # Imágenes por shard del dataset

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
        shard_size: int - Tamaño de la porción del dataset para este worker
        params: Dict - Parámetros del modelo (PyTorch state_dict)
    """
    batch_ids: list
    epoch: int
    init_signal: bool
    stop_signal: bool
    learning_rate: float
    shard_size: int
    params: Dict
    hf_token: str  # Token de HuggingFace para acceso a ImageNet
    worker_id: int = 0
    num_workers: int = 1
    
    def __repr__(self):
        return (f"MessageFromServer(epoch={self.epoch}, worker_id={self.worker_id}, num_workers={self.num_workers}, "
                f"batches={len(self.batch_ids)}, shard_size={self.shard_size}, init={self.init_signal}, stop={self.stop_signal})")


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
    buffers: Dict = field(default_factory=dict)  # BN running stats (running_mean, running_var)
    
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
        dataset_size: int - Tamaño de la partición asignada (shard_size)
    """
    worker_id: int
    dataset_size: int
    
    def __repr__(self):
        return (f"WorkerReadyMessage(worker_id={self.worker_id}, "
                f"dataset_size={self.dataset_size})")


@dataclass
class TrainingConfig:
    """Configuración global de entrenamiento distribuido para ImageNet."""
    num_workers: int = 1
    epocas: int = 10
    learning_rate: float = 0.01
    intervalo_log: int = 1
    server_host: str = 'localhost'
    server_port: int = 6000
    socket_timeout: int = 500  # segundos
    batch_size: int = 32
    num_classes: int = 1000  # ImageNet tiene 1000 clases
    save_file: str = './Results/imagenet_trained_model.pth'
    imagenet_split: str = 'train'  # 'train' o 'val'
    hf_token: str = ''  # Token de HuggingFace para ImageNet
