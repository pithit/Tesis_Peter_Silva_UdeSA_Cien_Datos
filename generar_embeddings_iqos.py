import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import logging
import argparse
import json

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validar_dataset(df: pd.DataFrame) -> bool:
    """Valida que el DataFrame tenga las columnas necesarias."""
    columnas_requeridas = ['id', 'nombre', 'categoria', 'descripcion']
    if not all(col in df.columns for col in columnas_requeridas):
        logger.error(f"El dataset debe contener las columnas: {columnas_requeridas}")
        return False
    return True

def generar_embeddings(
    input_csv: str = "data/iqos_products.csv",
    output_file: str = "data/embeddings_iqos.npy",
    output_metadata_file: str = "data/metadata_iqos.json",
    modelo: str = "all-MiniLM-L6-v2",
    batch_size: int = 32
):
    """
    Genera embeddings para las descripciones de productos IQOS y guarda los metadatos.
    """
    try:
        if not os.path.exists(input_csv):
            logger.error(f"No se encontró el archivo de entrada: {input_csv}")
            return
        
        df = pd.read_csv(input_csv)
        if not validar_dataset(df):
            return
        
        logger.info(f"Cargando modelo SentenceTransformer: {modelo}")
        model = SentenceTransformer(modelo)
        
        logger.info("Generando embeddings para las descripciones...")
        embeddings = model.encode(
            df['descripcion'].tolist(), 
            batch_size=batch_size,
            show_progress_bar=True
        )
        
        # Guardar embeddings
        np.save(output_file, embeddings)
        logger.info(f"Embeddings guardados en {output_file} (Dimensiones: {embeddings.shape})")
        
        # Guardar metadatos
        metadata = {"model_name": modelo}
        with open(output_metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Metadatos guardados en {output_metadata_file}")
        
    except Exception as e:
        logger.error(f"Error al generar embeddings: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de Embeddings para productos IQOS")
    parser.add_argument(
        "--modelo",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Nombre del modelo SentenceTransformer a utilizar (ej. 'all-MiniLM-L6-v2' o 'all-mpnet-base-v2')"
    )
    args = parser.parse_args()
    
    if not os.path.exists('data'):
        os.makedirs('data')

    # Normalizar el nombre del modelo para usarlo en el nombre del archivo
    model_filename = args.modelo.replace("/", "_")
    output_path = f"data/embeddings_{model_filename}.npy"
    metadata_path = f"data/metadata_{model_filename}.json"

    print(f"--- Generando archivos para el modelo: {args.modelo} ---")
    generar_embeddings(
        output_file=output_path,
        output_metadata_file=metadata_path,
        modelo=args.modelo
    )
    print("--- Proceso completado ---") #fin
