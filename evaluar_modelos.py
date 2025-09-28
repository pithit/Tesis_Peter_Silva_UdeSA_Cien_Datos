import json
import os
import pandas as pd
import re
from dotenv import load_dotenv

# Cargar las funciones de recomendación de los otros scripts
from recomendar_productos import recomendar_productos
from recomendar_llm import recomendar_con_llm
from Recomendar_hibrido import recomendar_hibrido

def parsear_recomendaciones_llm(respuesta_texto: str, top_k=3) -> list:
    """
    Parser mejorado - múltiples formatos de respuesta del LLM
    """
    
    # 1. Buscar sección formal "PRODUCTOS RECOMENDADOS:"
        match = re.search(r"PRODUCTOS RECOMENDADOS:\s*\n(.*?)$", respuesta_texto, re.DOTALL | re.IGNORECASE)
    if match:
        lista_texto = match.group(1)
        items = re.findall(r"^\s*\d+\.\s*(.+)$", lista_texto, re.MULTILINE)
        if items:
            productos = [limpiar_nombre_producto(item) for item in items[:top_k]]
            return rellenar_lista(productos, top_k)
    
    # 2. Buscar listas numeradas en cualquier parte del texto
    items_numerados = re.findall(r"^\s*\d+\.\s*(.+)$", respuesta_texto, re.MULTILINE)
    if items_numerados:
        productos = [limpiar_nombre_producto(item) for item in items_numerados[:top_k]]
        return rellenar_lista(productos, top_k)
    
    # 3. Buscar nombres de productos conocidos
    productos_conocidos = [
        "IQOS ILUMA PRIME", "IQOS ILUMA", "IQOS ILUMA ONE", "IQOS 3 DUO",
        "TEREA Amber", "TEREA Sienna", "TEREA Turquoise",
        "HEETS Amber Selection", "HEETS Sienna Selection", "HEETS Turquoise Selection",
        "Estación de Carga para IQOS 3 DUO", "Funda de Cuero para IQOS ILUMA", 
        "Contenedor de Viaje para TEREA"
    ]
    
    productos_encontrados = []
    texto_lower = respuesta_texto.lower()
    
    for producto in productos_conocidos:
        # Buscar variaciones del nombre del producto
        variaciones = generar_variaciones_nombre(producto)
        for variacion in variaciones:
            if variacion.lower() in texto_lower and producto not in productos_encontrados:
                productos_encontrados.append(producto)
                break
    
    if productos_encontrados:
        return rellenar_lista(productos_encontrados[:top_k], top_k)
    
    # 4. Si todo falla, devolver lista de "N/A"
    return ["N/A"] * top_k


def limpiar_nombre_producto(texto: str) -> str:
    """Limpia el nombre del producto extraído"""
    # Remover asteriscos, comillas, dos puntos, etc.
    limpio = re.sub(r'[\*\"\:\-]', '', texto).strip()
    
    # Remover texto después de ": " (descripción)
    if ": " in limpio:
        limpio = limpio.split(": ")[0].strip()
    
    # Mapear nombres comunes a nombres exactos
    mapeo_nombres = {
        "iqos iluma prime": "IQOS ILUMA PRIME",
        "iqos iluma one": "IQOS ILUMA ONE", 
        "iqos iluma": "IQOS ILUMA",
        "iqos 3 duo": "IQOS 3 DUO",
        "terea amber": "TEREA Amber",
        "terea sienna": "TEREA Sienna", 
        "terea turquoise": "TEREA Turquoise",
        "heets amber": "HEETS Amber Selection",
        "heets sienna": "HEETS Sienna Selection",
        "heets turquoise": "HEETS Turquoise Selection",
        "estación de carga": "Estación de Carga para IQOS 3 DUO",
        "funda de cuero": "Funda de Cuero para IQOS ILUMA",
        "contenedor": "Contenedor de Viaje para TEREA"
    }
    
    limpio_lower = limpio.lower()
    for clave, valor in mapeo_nombres.items():
        if clave in limpio_lower:
            return valor
    
    return limpio


def generar_variaciones_nombre(producto: str) -> list:
    """Genera variaciones del nombre del producto para búsqueda flexible"""
    variaciones = [producto, producto.lower()]
    
    # Variaciones específicas
    if "IQOS ILUMA PRIME" in producto:
        variaciones.extend(["iluma prime", "iqos prime", "prime"])
    elif "IQOS ILUMA ONE" in producto:
        variaciones.extend(["iluma one", "iqos one"])
    elif "IQOS ILUMA" in producto:
        variaciones.extend(["iluma", "iqos iluma"])
    elif "IQOS 3 DUO" in producto:
        variaciones.extend(["3 duo", "iqos 3", "duo"])
    elif "TEREA" in producto:
        variaciones.append(producto.replace("TEREA ", "").lower())
    elif "HEETS" in producto:
        variaciones.append(producto.replace("HEETS ", "").replace(" Selection", "").lower())
    
    return variaciones


def rellenar_lista(productos: list, top_k: int) -> list:
    """Rellena la lista hasta top_k elementos"""
    while len(productos) < top_k:
        productos.append("N/A")
    return productos[:top_k]


def ejecutar_evaluacion(k=3):
    """
    Ejecuta todos los modelos contra el ground truth y guarda los resultados.
    """
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("La variable de entorno GROQ_API_KEY no está configurada.")

    with open('eval/ground_truth.json', 'r', encoding='utf-8') as f:
        ground_truth_data = json.load(f)

    df_catalogo = pd.read_csv('src/productos_iqos.csv')

    resultados_evaluacion = []

    print("Ejecutando evaluación para todos los modelos...")

    for i, item in enumerate(ground_truth_data):
        consulta = item['consulta']
        print(f"Procesando consulta {i+1}/{len(ground_truth_data)}: \"{consulta[:50]}...\"")

        # 1. Modelo SBERT
        recomendaciones_sbert_df, _ = recomendar_productos(
            consulta, 
            path_embeddings=f"data/embeddings_all-mpnet-base-v2.npy",
            path_metadata=f"data/metadata_all-mpnet-base-v2.json",
            path_productos="data/iqos_products.csv",
            top_k=k
        )
        ranking_sbert = recomendaciones_sbert_df['nombre'].tolist()

        # 2. Modelo LLM Puro
        catalogo_dict = df_catalogo.to_dict(orient='records')
        respuesta_llm_puro = recomendar_con_llm(consulta, productos_candidatos=catalogo_dict)
        ranking_llm_puro = parsear_recomendaciones_llm(respuesta_llm_puro, top_k=k)

        # 3. Modelo Híbrido
        respuesta_hibrida_texto, df_candidatos_hibrido = recomendar_hibrido(consulta)
        # ✅ CORREGIDO: Usamos el re-ranking del LLM, no el ranking de SBERT
        ranking_hibrido = parsear_recomendaciones_llm(respuesta_hibrida_texto, top_k=k)


        resultados_evaluacion.append({
            "consulta": consulta,
            "ground_truth": item['relevancia'],
            "resultados": {
                "SBERT": ranking_sbert,
                "LLM_Puro": ranking_llm_puro,
                "Hibrido": ranking_hibrido
            },
            "respuesta_llm_puro": respuesta_llm_puro,
            "respuesta_hibrida": respuesta_hibrida_texto
        })

    os.makedirs('eval', exist_ok=True)
    with open('eval/resultados_modelos.json', 'w', encoding='utf-8') as f:
        json.dump(resultados_evaluacion, f, ensure_ascii=False, indent=4)

    print("\nEvaluación completada. Resultados guardados en 'eval/resultados_modelos.json'")

if __name__ == "__main__":
    ejecutar_evaluacion() 