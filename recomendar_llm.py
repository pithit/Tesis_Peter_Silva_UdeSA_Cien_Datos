# recomendar_llm.py

import os
import pandas as pd
from typing import List, Dict, Optional
from dotenv import load_dotenv
import logging

# --- Cargar variables de entorno desde archivo .env ---
# Esto busca un archivo .env en el directorio raíz del proyecto
load_dotenv()

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------- CONFIGURACIÓN ----------
MODEL_NAME = "llama3-8b-8192"

def construir_prompt(consulta_usuario: str, productos: List[Dict]) -> str:
    """Construye el prompt para el LLM con técnicas de Prompt Engineering."""
    
    # 1. Instrucción de Rol (Persona)
    prompt = "Actúa como un experto vendedor de IQOS y asistente de compras personal. Tu objetivo es ayudar a un usuario a encontrar el producto perfecto para él.\n\n"
    
    # 2. Contexto y Tarea
    prompt += f"Un usuario ha realizado la siguiente consulta: '{consulta_usuario}'.\n\n"
    prompt += "A continuación, se presenta una lista de productos disponibles con sus descripciones:\n"
    for p in productos:
        prompt += f"- Nombre: {p['nombre']}, Descripción: {p['descripcion']}\n"
    
    # 3. Instrucción de Salida y Razonamiento (Chain-of-Thought)
    prompt += "\n**Instrucciones:**\n1.  **Analiza la consulta:** Lee atentamente la consulta del usuario para entender sus preferencias, necesidades y cualquier restricción.\n2.  **Evalúa los productos:** Revisa la siguiente lista de productos y compáralos con la consulta del usuario.\n3.  **Razonamiento paso a paso:** Antes de dar la recomendación final, explica brevemente tu proceso de pensamiento.\n4.  **Recomendación final:** Ofrece una recomendación clara y concisa.\n5.  **Formato de Salida Obligatorio:** Al final de toda tu respuesta, incluye una sección que comience EXACTAMENTE con la línea \"PRODUCTOS RECOMENDADOS:\" seguida de una lista numerada de los 3 productos principales que recomendaste.\n\n**Consulta del usuario:**\n\"{consulta_usuario}\"\n\n**Lista de productos:**\n"
    
    return prompt

def obtener_recomendaciones_llm(prompt: str) -> Optional[str]:
    """Llama a la API de Groq para obtener la respuesta del LLM."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("La variable de entorno GROQ_API_KEY no se encontró.")
        print("ADVERTENCIA: La variable de entorno GROQ_API_KEY no se encontró.")
        return None

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
        )
        return chat_completion.choices[0].message.content
    except ImportError:
        logger.error("La librería 'openai' no está instalada.")
        print("Error: La librería 'openai' no está instalada. Por favor, ejecute 'pip install -r requirements.txt'")
        return None
    except Exception as e:
        logger.error(f"Error al contactar la API de Groq: {e}")
        print(f"Error al contactar la API de Groq: {e}")
        return None

def recomendar_con_llm(consulta_usuario: str, productos_candidatos: Optional[List[Dict]] = None):
    """
    Obtiene recomendaciones de productos IQOS utilizando un LLM.
    Puede operar sobre todos los productos o sobre una lista de candidatos pre-filtrada.
    """
    productos_a_considerar = []
    if productos_candidatos is None:
        logger.info("No se proveyeron candidatos, cargando todos los productos del catálogo...")
    try:
            df_productos = pd.read_csv("src/productos_iqos.csv")
            productos_a_considerar = df_productos.to_dict(orient='records')
    except FileNotFoundError:
            logger.error("No se encontró el archivo 'src/productos_iqos.csv'.")
            print("Error: El archivo de productos no fue encontrado.")
            return None
    else:
        logger.info(f"Recibidos {len(productos_candidatos)} candidatos para re-ranking por el LLM.")
        productos_a_considerar = productos_candidatos

    prompt = construir_prompt(consulta_usuario, productos_a_considerar)
    respuesta = obtener_recomendaciones_llm(prompt)

    if respuesta:
        return respuesta

if __name__ == '__main__':
    print("--- Ejecutando recomendador LLM (Llama 3 con Groq) de forma individual ---")
    
    consultas_de_prueba = [
        "Busco un dispositivo con un diseño elegante y que sea fácil de usar.",
        "un dispositivo barato pero que se sienta premium",
        "Quiero algo bueno y práctico",
        "Kiero un dispositivo fasil de husar y que la vateria dure"
    ]

    for i, consulta in enumerate(consultas_de_prueba):
        print(f"\n--- PRUEBA {i+1}: CONSULTA = '{consulta}' ---")
        recomendacion = recomendar_con_llm(consulta)
        print("Respuesta del Asistente de Compras IQOS:")
        print(recomendacion)
        print("--------------------------------------------------")
