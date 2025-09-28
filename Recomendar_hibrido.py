import pandas as pd
from recomendar_productos import recomendar_productos
from recomendar_llm import recomendar_con_llm
import logging

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constantes de Configuración ---
MODELO_EMBEDDING_FILTRADO = 'all-mpnet-base-v2'
TOP_K_FILTRADO = 5 # Número de candidatos a pasar al LLM

def recomendar_hibrido(consulta: str):
    """
    Implementa un sistema de recomendación híbrido de dos etapas.
    
    Etapa 1: Filtrado rápido con Embeddings (SBERT) para generar candidatos.
    Etapa 2: Re-ranking y razonamiento con un LLM para la recomendación final.
    """
    print("="*80)
    print("      SISTEMA DE RECOMENDACIÓN HÍBRIDO (SBERT + LLM)")
    print("="*80)
    print(f"CONSULTA: '{consulta}'\n")

    # --- ETAPA 1: FILTRADO CON EMBEDDINGS ---
    logger.info(f"Iniciando Etapa 1: Filtrado con SBERT ({MODELO_EMBEDDING_FILTRADO})")
    print(f"--- Etapa 1: Filtrando los {TOP_K_FILTRADO} mejores candidatos con SBERT... ---\n")
    
    path_embeddings = f"data/embeddings_{MODELO_EMBEDDING_FILTRADO}.npy"
    path_metadata = f"data/metadata_{MODELO_EMBEDDING_FILTRADO}.json"
    
    # Obtenemos el DataFrame de recomendaciones de SBERT
    df_candidatos, metricas = recomendar_productos(
        consulta=consulta,
        top_k=TOP_K_FILTRADO,
        path_embeddings=path_embeddings,
        path_metadata=path_metadata
    )

    if df_candidatos is None or df_candidatos.empty:
        logger.warning("La etapa de filtrado no devolvió candidatos. Terminando proceso.")
        print("No se encontraron productos relevantes en la primera etapa.")
        return

    # Convertir el DataFrame a una lista de diccionarios para el LLM
    productos_candidatos = df_candidatos.to_dict(orient='records')
    
    print("Candidatos seleccionados por SBERT:")
    print(df_candidatos[['nombre', 'categoria', 'score']])
    print("\n" + "."*80 + "\n")

    # --- ETAPA 2: RE-RANKING CON LLM ---
    logger.info(f"Iniciando Etapa 2: Re-ranking de {len(productos_candidatos)} candidatos con LLM.")
    print("--- Etapa 2: LLM analiza los candidatos para la recomendación final... ---\n")

    respuesta_llm = recomendar_con_llm(consulta, productos_candidatos=productos_candidatos)
    
    print(respuesta_llm)
    
    # Return tanto la respuesta final como la lista de candidatos para evaluación
    return respuesta_llm, df_candidatos


if __name__ == '__main__':
    consulta_ejemplo = "Busco un dispositivo que sea elegante, moderno y fácil de llevar a todos lados, ideal para un profesional ocupado."
    recomendar_hibrido(consulta_ejemplo) 