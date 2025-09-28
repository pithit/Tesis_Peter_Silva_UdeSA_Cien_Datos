import json
import numpy as np
from sklearn.metrics import ndcg_score

def calcular_hit_rate_at_k(ranking: list, ground_truth: dict, k: int) -> float:
    """
    Calcula si alguno de los k primeros items recomendados es relevante (score > 0).
    """
    hits = 0
    for producto in ranking[:k]:
        if ground_truth.get(producto, 0) > 0:
            hits += 1
    return 1.0 if hits > 0 else 0.0

def calcular_ndcg_at_k(ranking: list, ground_truth: dict, k: int) -> float:
    """
    Calcula el NDCG@k.
    """
    # Vector de relevancia real (ground truth) para los items en el ranking
    true_relevance = np.zeros(k)
    for i, producto in enumerate(ranking[:k]):
        true_relevance[i] = ground_truth.get(producto, 0)

    # Vector de puntuaciones ideal (mejor ranking posible)
    # No es necesario para scikit-learn, que lo calcula internamente.
    
    # El ranking predicho tiene un score decreciente, pero para ndcg_score se usa la relevancia real
    predicted_scores = np.arange(k, 0, -1)

    # ndcg_score espera [[true_relevance]], [[predicted_scores]]
    return ndcg_score([true_relevance], [predicted_scores], k=k)

def analizar_resultados(k=3):
    """
    Carga los resultados y el ground truth, y calcula las métricas.
    """
    # Cargar los archivos JSON
    with open('eval/resultados_modelos.json', 'r', encoding='utf-8') as f:
        resultados_data = json.load(f)
    
    # Diccionario para almacenar los promedios de las métricas
    metricas_promedio = {
        'SBERT': {'NDCG@k': [], 'HitRate@k': []},
        'LLM_Puro': {'NDCG@k': [], 'HitRate@k': []},
        'Hibrido': {'NDCG@k': [], 'HitRate@k': []}
    }

    print(f"--- Análisis de Métricas (k={k}) ---")

    for i, resultado in enumerate(resultados_data):
        print(f"\nConsulta {i+1}: \"{resultado['consulta'][:40]}...\"")
        ground_truth = resultado['ground_truth']
        
        for modelo, ranking in resultado['resultados'].items():
            if not ranking or "Error" in ranking[0]:
                print(f"  - {modelo}: Ranking con errores, saltando cálculo.")
                continue

            ndcg = calcular_ndcg_at_k(ranking, ground_truth, k)
            hit_rate = calcular_hit_rate_at_k(ranking, ground_truth, k)
            
            metricas_promedio[modelo]['NDCG@k'].append(ndcg)
            metricas_promedio[modelo]['HitRate@k'].append(hit_rate)

            print(f"  - {modelo}: NDCG@{k} = {ndcg:.4f}, HitRate@{k} = {hit_rate:.4f}")

    # Calcular y mostrar los promedios finales
    print("\n--- Resultados Promedio ---")
    for modelo, metricas in metricas_promedio.items():
        avg_ndcg = np.mean(metricas['NDCG@k']) if metricas['NDCG@k'] else 0
        avg_hit_rate = np.mean(metricas['HitRate@k']) if metricas['HitRate@k'] else 0
        print(f"Modelo: {modelo}")
        print(f"  - NDCG@{k} Promedio: {avg_ndcg:.4f}")
        print(f"  - HitRate@{k} Promedio: {avg_hit_rate:.4f}")
        print("-" * 25)

if __name__ == "__main__":
    analizar_resultados(k=3) 