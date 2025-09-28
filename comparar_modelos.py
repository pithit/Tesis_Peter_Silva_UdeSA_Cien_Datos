import os
from dotenv import load_dotenv
from recomendar_productos import recomendar_productos, mostrar_recomendaciones
from recomendar_llm import recomendar_con_llm

# Cargar variables de entorno desde el file .env
load_dotenv()

def comparar_recomendaciones(consulta: str):
    """
    Ejecuta y compara los resultados de los dos sistemas de recomendación:
    1. Basado en Embeddings (SBERT)
    2. Basado en un Modelo de Lenguaje Grande (LLM)
    """
    
    print("="*80)
    print("                COMPARACIÓN DE MODELOS DE RECOMENDACIÓN")
    print("="*80)
    print(f"\nCONSULTA DEL USUARIO: '{consulta}'\n")
    print("."*80)

    # --- 1. Modelo basado en Embeddings (SBERT) ---
    print("\nMODELO 1: Búsqueda Semántica con Embeddings (SBERT)\n")
    # Usamos el path por defecto que corresponde a all-MiniLM-L6-v2
    recomendaciones_sbert, metricas_sbert = recomendar_productos(
        consulta=consulta,
        top_k=3
    )
    if not recomendaciones_sbert.empty:
        mostrar_recomendaciones(recomendaciones_sbert, metricas_sbert)
    else:
        print("No se pudieron generar recomendaciones con SBERT.")
    
    print("\n" + "."*80)

    # --- 2. Modelo basado en LLM (Llama 3 con Groq) ---
    print("\nMODELO 2: Razonamiento con LLM (Llama 3 con Groq)\n")
    
    recomendar_con_llm(
        consulta_usuario=consulta,
        path_csv="data/iqos_products.csv"
    )
        
    print("\n" + "="*80)


if __name__ == "__main__":
    # --- Consulta de ejemplo para la comparación ---
    consulta_ejemplo = "Busco un dispositivo que sea elegante, moderno y fácil de llevar a todos lados, ideal para un profesional ocupado."
    
    comparar_recomendaciones(consulta_ejemplo)
