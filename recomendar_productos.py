import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import Optional, List, Tuple, Dict
from tabulate import tabulate
import json

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constantes ---
PATH_PRODUCTOS_CSV = "data/iqos_products.csv"
MODELO_DEFECTO = 'all-MiniLM-L6-v2'
PATH_EMBEDDINGS_DEFECTO = f"data/embeddings_{MODELO_DEFECTO}.npy"


def cargar_datos(path_productos: str, path_embeddings: str) -> tuple:
    """Carga los datos de productos y los embeddings desde los archivos."""
    try:
        df = pd.read_csv(path_productos)
        embeddings = np.load(path_embeddings)
        return df, embeddings
    except FileNotFoundError as e:
        logger.error(f"Error al cargar datos: {e}. Asegúrate de que los archivos existen.")
        return None, None

def calcular_diversidad(recomendaciones_df, embeddings_originales):
    """Calcula la diversidad como la distancia promedio entre los ítems recomendados."""
    indices = recomendaciones_df.index.tolist()
    if len(indices) < 2:
        return 0.0
    
    embeddings_recomendados = embeddings_originales[indices]
    sim_matrix = cosine_similarity(embeddings_recomendados)
    dist_matrix = 1 - sim_matrix
    diversidad = np.mean(dist_matrix[np.triu_indices(len(indices), k=1)])
    return diversidad

def calcular_novedad(recomendaciones_df, historial_usuario):
    """Calcula la novedad como la proporción de ítems no vistos en el historial."""
    if historial_usuario is None or len(historial_usuario) == 0:
        return 1.0
        
    ids_recomendados = set(recomendaciones_df['id'].tolist())
    ids_historial = set(historial_usuario)
    
    novedad = len(ids_recomendados - ids_historial) / len(ids_recomendados)
    return novedad

def recomendar_productos(
    consulta: str,
    top_k: int = 3,
    path_productos: str = PATH_PRODUCTOS_CSV,
    path_embeddings: str = PATH_EMBEDDINGS_DEFECTO,
    path_metadata: Optional[str] = None,
    categoria: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Genera recomendaciones de productos basadas en una consulta de usuario.
    
    Args:
        consulta: La consulta del usuario en lenguaje natural.
        top_k: El número de recomendaciones a devolver.
        path_productos: Ruta al archivo CSV de productos.
        path_embeddings: Ruta al archivo .npy de embeddings.
        path_metadata: (Opcional) Ruta al archivo JSON de metadatos del modelo de embedding.
        categoria: (Opcional) La categoría de productos a filtrar.
    
    Returns:
        Un DataFrame con los productos recomendados y un diccionario con métricas.
    """
    df, embeddings = cargar_datos(path_productos, path_embeddings)
    if df is None:
        return pd.DataFrame(), {}

    # 1. Cargar el modelo de embedding
    model_name = MODELO_DEFECTO
    if path_metadata:
        try:
            with open(path_metadata, 'r') as f:
                metadata = json.load(f)
                model_name = metadata.get('model_name', MODELO_DEFECTO)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning(f"No se pudo leer el archivo de metadatos en {path_metadata}. Usando modelo por defecto.")
    
    logger.info(f"Cargando modelo de embedding: {model_name}")
    modelo_transformer = SentenceTransformer(model_name)
        
    # 2. Generar embedding para la consulta
    embedding_consulta = modelo_transformer.encode([consulta], show_progress_bar=False)
        
    # 3. Filtrar por categoría si se especifica
    df_filtrado = df
    indices_filtrados = df.index
    if categoria:
        df_filtrado = df[df['categoria'].str.lower() == categoria.lower()]
        indices_filtrados = df_filtrado.index
    
    if df_filtrado.empty:
        logger.warning(f"No se encontraron productos para la categoría '{categoria}'.")
        return pd.DataFrame(), {}
        
    embeddings_filtrados = embeddings[indices_filtrados]
        
    # 4. Calcular similitud y obtener los mejores K
    similitudes = cosine_similarity(embedding_consulta, embeddings_filtrados).flatten()
    indices_top_local = np.argsort(similitudes)[-top_k:][::-1]
    indices_top_global = indices_filtrados[indices_top_local]

    recomendaciones = df.loc[indices_top_global].copy()
    recomendaciones['score'] = similitudes[indices_top_local]
    
    # 5. Calcular métricas
    metricas = {
        "similitud_promedio": recomendaciones['score'].mean(),
        "diversidad": calcular_diversidad(recomendaciones, embeddings),
        "novedad": calcular_novedad(recomendaciones, None), # Historial no implementado aún
        "model_used": model_name
    }

    return recomendaciones, metricas

def mostrar_recomendaciones(df_recomendaciones: pd.DataFrame, metricas: dict):
    """Muestra las recomendaciones y sus metricas."""
    if df_recomendaciones.empty:
        print("No se pudieron generar recomendaciones.")
        return
        
    print("--- Recomendaciones Encontradas ---")
    print(tabulate(df_recomendaciones[['nombre', 'categoria', 'score']], headers='keys', tablefmt='psql'))
    print("\n--- Métricas de la Recomendación ---")
    print(f"Modelo de Embedding: {metricas.get('model_used', 'No especificado')}")
    print(f"Similitud Promedio: {metricas.get('similitud_promedio', 0):.4f}")
    print(f"Diversidad: {metricas.get('diversidad', 0):.4f}")
    print(f"Novedad: {metricas.get('novedad', 0):.4f}")


if __name__ == '__main__':
    # --- Ejemplo de uso ---
    consulta_de_prueba = "Busco un dispositivo con un diseño elegante y que sea fácil de usar."
    
    df_recs, metricas_res = recomendar_productos(
        consulta=consulta_de_prueba, 
        top_k=3
    )
    
    if not df_recs.empty:
        mostrar_recomendaciones(df_recs, metricas_res)