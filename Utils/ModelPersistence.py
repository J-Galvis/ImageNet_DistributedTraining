import numpy as np
import os
import json
import pickle
from datetime import datetime


# ── Carpeta donde se guardan los modelos ─────────────────────────────────────
CARPETA_MODELOS = os.path.join(os.path.dirname(__file__), '../modelos_guardados')
CARPETA_STATS = os.path.join(os.path.dirname(__file__), '../stats')


# ─────────────────────────────────────────────────────────────────────────────
# GUARDAR MODELO
# ─────────────────────────────────────────────────────────────────────────────

def guardar_modelo(W1, b1, W2, b2, nombre_modelo, precision_test=None,
                   epocas=None, learning_rate=None, training_time=None, info_extra=None,
                   step_loss_history=None, step_accuracy_history=None, 
                   step_times_history=None, step_ids_history=None):
    """
    Guarda los pesos de una red neuronal entrenada en un archivo .pkl
    y los metadatos en un archivo JSON.
    
    Los pesos se guardan en: nombre_modelo.pkl
    Los metadatos se guardan en: nombre_modelo_metadata.json
    """
    # Crear la carpeta si no existe
    os.makedirs(CARPETA_MODELOS, exist_ok=True)
    os.makedirs(CARPETA_STATS, exist_ok=True)

    # Nombre de los archivos
    nombre_archivo_pkl = f"{nombre_modelo}.pkl"
    nombre_archivo_json = f"{nombre_modelo}_metadata.json"
    ruta_archivo_pkl = os.path.join(CARPETA_MODELOS, nombre_archivo_pkl)
    ruta_archivo_json = os.path.join(CARPETA_STATS, nombre_archivo_json)

    if not (W1 is None or b1 is None or W2 is None or b2 is None):
    # Empaquetar solo los pesos en el pickle
        datos_modelo = {
            'W1': W1,
            'b1': b1,
            'W2': W2,
            'b2': b2
        }

        # Guardar pesos con pickle
        with open(ruta_archivo_pkl, 'wb') as archivo:
            pickle.dump(datos_modelo, archivo)
        
                # Arquitectura
        entrada = int(W1.shape[0])
        oculta = int(W1.shape[1])
        salida = int(W2.shape[1])
    
    else:
        entrada = oculta = salida = None

    # Construir metadatos
    fecha_guardado = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    
    # Metadatos básicos
    metadatos = {
        'nombre_modelo': nombre_modelo,
        'fecha_guardado': fecha_guardado,
        'precision_test': precision_test,
        'epocas': epocas,
        'learning_rate': learning_rate,
        'training_time_seconds': training_time,
        'arquitectura': {
            'entrada': entrada,
            'oculta': oculta,
            'salida': salida,
        }   
    }
    
    # Agregar información extra
    if info_extra:
        metadatos['info_extra'] = info_extra
    
    # Agregar métricas a nivel de paso (step-level) si están disponibles
    if step_loss_history or step_accuracy_history or step_times_history or step_ids_history:
        metadatos['step_metrics'] = {
            'step_loss': step_loss_history if step_loss_history else [],
            'step_accuracy': step_accuracy_history if step_accuracy_history else [],
            'step_times': step_times_history if step_times_history else [],
            'step_ids': step_ids_history if step_ids_history else [],
        }
    
    # Guardar metadatos en JSON
    with open(ruta_archivo_json, 'w', encoding='utf-8') as archivo:
        json.dump(metadatos, archivo, indent=2, ensure_ascii=False)
    
    print(f"\n  ✓ Modelo guardado en: {ruta_archivo_pkl}")
    print(f"  ✓ Metadatos guardados en: {ruta_archivo_json}")
    
    return ruta_archivo_pkl


# ─────────────────────────────────────────────────────────────────────────────
# CARGAR MODELO
# ─────────────────────────────────────────────────────────────────────────────

def cargar_modelo(nombre_modelo=None, ruta_archivo=None):
    """
    Carga un modelo previamente guardado desde un archivo .pkl.
    """
    if ruta_archivo is None:
        # Buscar en la carpeta de modelos
        if not os.path.exists(CARPETA_MODELOS):
            raise FileNotFoundError(
                f"No se encontró la carpeta de modelos: {CARPETA_MODELOS}\n"
                f"Primero entrena y guarda un modelo."
            )

        # Listar todos los archivos .pkl
        archivos = [f for f in os.listdir(CARPETA_MODELOS) if f.endswith('.pkl')]

        if not archivos:
            raise FileNotFoundError(
                f"No hay modelos guardados en: {CARPETA_MODELOS}"
            )

        # Filtrar por nombre si se proporcionó
        if nombre_modelo:
            archivos = [f for f in archivos if nombre_modelo in f]
            if not archivos:
                raise FileNotFoundError(
                    f"No se encontró ningún modelo con nombre '{nombre_modelo}' "
                    f"en {CARPETA_MODELOS}"
                )

        # Tomar el más reciente (ordenar por nombre que incluye timestamp)
        archivos.sort()
        archivo_seleccionado = archivos[-1]
        ruta_archivo = os.path.join(CARPETA_MODELOS, archivo_seleccionado)

    # Cargar el archivo .pkl (solo pesos)
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_archivo}")

    with open(ruta_archivo, 'rb') as archivo:
        datos = pickle.load(archivo)

    W1 = datos['W1']
    b1 = datos['b1']
    W2 = datos['W2']
    b2 = datos['b2']

    return W1, b1, W2, b2


# ─────────────────────────────────────────────────────────────────────────────
# CARGAR METADATOS
# ─────────────────────────────────────────────────────────────────────────────

def cargar_metadatos(nombre_modelo):
    """
    Carga los metadatos de un modelo desde el archivo JSON.
    
    Parámetros:
        nombre_modelo: str, nombre del modelo (sin extensión)
    
    Retorna:
        dict con los metadatos del modelo
    """
    archivo_metadatos = os.path.join(CARPETA_STATS, f"{nombre_modelo}_metadata.json")
    
    if not os.path.exists(archivo_metadatos):
        raise FileNotFoundError(f"No se encontró archivo de metadatos: {archivo_metadatos}")
    
    with open(archivo_metadatos, 'r', encoding='utf-8') as f:
        metadatos = json.load(f)
    
    return metadatos
