from __future__ import annotations

import os
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, Subset
from torchvision import datasets as tvd, transforms as T


NUM_CLASSES = 1000
IMAGE_SIZE = 224
SHARD_SIZE = 50_000  

HF_DATASET = "ILSVRC/imagenet-1k"  

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


# ══════════════════════════════════════════════════════════════════
# UTILIDADES COMUNES
# ══════════════════════════════════════════════════════════════════


def _default_data_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "Data", "ImageNet")


def _imagenet_transform(is_train: bool = False) -> T.Compose:
    """
    Transformación estándar ImageNet con aumentación para entrenamiento.
    """
    if is_train:
        return T.Compose(
            [
                T.RandomResizedCrop(IMAGE_SIZE),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=_MEAN, std=_STD),
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(IMAGE_SIZE),
                T.ToTensor(),
                T.Normalize(mean=_MEAN, std=_STD),
            ]
        )


def detect_data_source(data_dir: Optional[str] = None) -> str:
    """
    Detecta si el dataset está disponible en disco local o hay que usar streaming.

    Regla: si data_dir contiene train/ y val/ con al menos un subdirectorio
    cada uno, se usa el modo local. En cualquier otro caso, streaming.

    :param data_dir: Ruta al directorio raíz de ImageNet. None = Data/ImageNet/.
    :return: "local" si el dataset está en disco, "stream" si no.
    """
    if data_dir is None:
        data_dir = _default_data_dir()

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    def _has_subdirs(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        return any(os.path.isdir(os.path.join(path, e)) for e in os.listdir(path))

    if _has_subdirs(train_dir) and _has_subdirs(val_dir):
        return "local"
    return "stream"


# ══════════════════════════════════════════════════════════════════
# MODO STREAM — HuggingFace Hub
# ══════════════════════════════════════════════════════════════════

# Tamaños oficiales de ILSVRC-2012
_HF_SPLIT_SIZES = {
    "train": 1_281_167,
    "validation": 50_000,
}


def get_hf_split_size(split: str) -> int:
    """
    Retorna el tamaño conocido de un split de ImageNet en HuggingFace.

    :param split: "train" o "val" (o "validation").
    :return: Número de imágenes en el split.
    """
    key = "validation" if split in ("val", "validation") else split
    return _HF_SPLIT_SIZES.get(key, 0)


def _hf_split_name(split: str) -> str:
    """HuggingFace usa 'validation', no 'val'."""
    return "validation" if split == "val" else split


class _HFStreamDataset(IterableDataset):
    """
    Dataset iterable que lee ImageNet desde HuggingFace en modo streaming.

    Cada elemento es un dict con "image" (PIL.Image) y "label" (int).
    Esta clase lo convierte a tensores (C, H, W) float32 normalizados
    con etiquetas int64, exactamente igual que ImageFolder.

    El sharding se aplica antes de crear el dataset:
    ds.shard(num_shards, index) garantiza que cada Worker
    procese una porción distinta sin solapamiento.

    :param hf_split: Split de HuggingFace ("train" o "validation").
    :param token: Token HuggingFace con acceso a ILSVRC/imagenet-1k.
    :param shard_index: Índice de este shard (0-based).
    :param num_shards: Total de shards.
    :param start_index: Primer elemento a procesar dentro del shard
                        (para reanudar si se interrumpió).
    """

    def __init__(
        self,
        hf_split: str,
        token: str,
        shard_index: int = 0,
        num_shards: int = 1,
        start_index: int = 0,
    ) -> None:
        super().__init__()
        self._hf_split = hf_split
        self._token = token
        self._shard_index = shard_index
        self._num_shards = num_shards
        self._start_index = start_index
        self._transform = _imagenet_transform(is_train=(hf_split == "train"))

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int]]:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as e:
            raise ImportError(
                "El modo streaming requiere la librería 'datasets' de HuggingFace.\n"
                "Instala con: pip install datasets"
            ) from e

        ds = load_dataset(
            HF_DATASET,
            split=self._hf_split,
            token=self._token,
            streaming=True,
        )

        # Sharding: cada Worker toma su porción sin solapamiento
        if self._num_shards > 1:
            ds = ds.shard(
                num_shards=self._num_shards,
                index=self._shard_index,
            )

        # Saltar elementos ya procesados (reanudación)
        if self._start_index > 0:
            ds = ds.skip(self._start_index)

        for item in ds:
            img = item["image"]
            label = item["label"]

            # Asegurar que la imagen es RGB (algunas son escala de grises)
            if img.mode != "RGB":
                img = img.convert("RGB")

            tensor = self._transform(img)
            yield tensor, label


def get_imagenet_stream_dataloader(
    split: str = "train",
    token: str = "",
    batch_size: int = 64,
    shard_index: int = 0,
    num_shards: int = 1,
    start_index: int = 0,
) -> DataLoader:
    """
    DataLoader de ImageNet en modo streaming desde HuggingFace.

    No descarga el dataset completo — las imágenes llegan bajo demanda.
    Los features resultantes SÍ se cachean en disco (shards .npy),
    por lo que solo la primera sesión requiere internet.

    IMPORTANTE: num_workers=0 porque IterableDataset con HuggingFace
    no es compatible con multiprocessing de DataLoader. La paralelización
    la gestiona HuggingFace internamente.

    :param split: "train" o "val".
    :param token: Token HuggingFace. También acepta variable de entorno
                  HF_TOKEN si token="".
    :param batch_size: Imágenes por batch.
    :param shard_index: Índice de este Worker (0-based).
    :param num_shards: Total de Workers que procesan en paralelo.
    :param start_index: Elemento de inicio dentro del shard (para reanudación).
    :return: DataLoader iterable configurado.
    """
    # Leer token desde variable de entorno si no se pasó explícitamente
    resolved_token = token or os.environ.get("HF_TOKEN", "")
    if not resolved_token:
        raise ValueError(
            "Se requiere un token de HuggingFace para modo streaming.\n"
            "Opciones:\n"
            "  1. --hf-token <token> en la línea de comandos\n"
            "  2. Variable de entorno: export HF_TOKEN=<token>\n"
            "Obtén tu token en: https://huggingface.co/settings/tokens"
        )

    hf_split = _hf_split_name(split)
    dataset = _HFStreamDataset(
        hf_split=hf_split,
        token=resolved_token,
        shard_index=shard_index,
        num_shards=num_shards,
        start_index=start_index,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,  # IterableDataset + HF no admite multiprocessing
        pin_memory=torch.cuda.is_available(),
    )


def load_imagenet_labels_stream(
    split: str = "val",
    token: str = "",
    cache_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Carga etiquetas de ImageNet desde HuggingFace en modo streaming.

    Con caché en disco: la primera ejecución descarga etiquetas del stream
    y las guarda en .npy. Siguientes ejecuciones cargan desde .npy sin internet.

    Usa streaming=True para descargar los parquets de forma lazy —
    solo los necesarios para obtener las etiquetas, no el dataset completo.
    Para val (50k imgs) descarga ~1-2 parquets en lugar de 294.

    :param split: "train" o "val".
    :param token: Token HuggingFace. También lee HF_TOKEN del entorno.
    :param cache_dir: Directorio para almacenar etiquetas en caché (.npy).
                      Si None, usa Data/feature_cache/.
    :return: (N,) int32 con etiquetas en [0, 999].
    """
    # Resolver cache_dir
    if cache_dir is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(root, "Data", "feature_cache")

    # Crear directorio si no existe
    os.makedirs(cache_dir, exist_ok=True)

    # Generar nombre del archivo de caché basado en split
    # Usar el nombre de HuggingFace para la clave del archivo
    hf_split = _hf_split_name(split)
    cache_file = os.path.join(cache_dir, f"imagenet_{hf_split}_labels.npy")

    # INTENTO 1: Cargar desde caché si existe
    if os.path.exists(cache_file):
        try:
            arr = np.load(cache_file)
            if arr.dtype == np.int32:
                print(
                    f"[Loader] ✓ Etiquetas {hf_split} cargadas desde caché "
                    f"({len(arr):,} etiquetas). Sin streaming necesario."
                )
                return arr
            else:
                print(
                    f"[Loader] ⚠ Archivo de caché corrupto (dtype={arr.dtype}). "
                    f"Descargando nuevamente desde HuggingFace..."
                )
        except Exception as e:
            print(
                f"[Loader] ⚠ Error al cargar caché: {e}. "
                f"Descargando nuevamente desde HuggingFace..."
            )

    # INTENTO 2: Descargar desde HuggingFace si no hay caché o está corrupto
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise ImportError(
            "El modo streaming requiere 'datasets'. Instala con: pip install datasets"
        ) from e

    resolved_token = token or os.environ.get("HF_TOKEN", "")
    if not resolved_token:
        raise ValueError(
            "Se requiere HF_TOKEN para cargar etiquetas desde HuggingFace.\n"
            "Opciones:\n"
            "  1. --hf-token <token>\n"
            "  2. export HF_TOKEN=<token>"
        )

    total = _HF_SPLIT_SIZES.get(hf_split, 0)
    print(
        f"[Loader] Descargando etiquetas de {hf_split} ({total:,} imgs) "
        f"desde HuggingFace (streaming)..."
    )

    # streaming=True: los parquets se descargan bajo demanda.
    # Solo se leen los necesarios para obtener todas las etiquetas,
    # sin descargar nunca las imágenes completas (~150 GB).
    ds = load_dataset(
        HF_DATASET,
        split=hf_split,
        token=resolved_token,
        streaming=True,
    )

    labels = []
    for idx, item in enumerate(ds):
        labels.append(item["label"])
        # Progreso cada 100k etiquetas
        if (idx + 1) % 100_000 == 0:
            print(f"[Loader]   ... {idx + 1:,} etiquetas procesadas")

    arr = np.array(labels, dtype=np.int32)
    print(f"[Loader] {len(arr):,} etiquetas descargadas.")

    # GUARDADO en caché
    try:
        np.save(cache_file, arr)
        print(
            f"[Loader] ✓ Etiquetas guardadas en caché: {cache_file} "
            f"({arr.nbytes // 1024 // 1024} MB)"
        )
    except Exception as e:
        print(
            f"[Loader] ⚠ No se pudieron guardar etiquetas en caché: {e}. "
            f"Siguientes ejecuciones descargarán nuevamente."
        )

    return arr