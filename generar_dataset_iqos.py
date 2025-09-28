# generar_dataset_iqos.py

import pandas as pd
import json
import os

def generar_dataset_iqos():
    """
    Genera un conjunto de datos simulado de productos IQOS y lo guarda en un archivo CSV.
    El dataset incluye dispositivos, consumibles (sticks) y accesorios,
    con descripciones para el análisis semántico.
    """
    productos = [
        # Dispositivos
        {
            "id": 1,
            "nombre": "IQOS ILUMA PRIME",
            "categoria": "Dispositivo",
            "descripcion": "El dispositivo más avanzado y elegante. Fabricado con aluminio anodizado y una exclusiva funda texturizada. Incorpora el revolucionario SMARTCORE INDUCTION SYSTEM™, que calienta el tabaco por inducción sin lámina, garantizando una experiencia sin limpieza, sin residuos de tabaco y con un sabor más consistente. Ofrece hasta 2 usos consecutivos por carga y una autonomía total para 20 usos."
        },
        {
            "id": 2,
            "nombre": "IQOS ILUMA",
            "categoria": "Dispositivo",
            "descripcion": "Un diseño icónico con tecnología avanzada. El IQOS ILUMA utiliza el SMARTCORE INDUCTION SYSTEM™ para una experiencia superior sin necesidad de limpieza. Es personalizable con una amplia gama de accesorios y colores. Ofrece 2 usos consecutivos y una batería de larga duración en su cargador de bolsillo."
        },
        {
            "id": 3,
            "nombre": "IQOS ILUMA ONE",
            "categoria": "Dispositivo",
            "descripcion": "Un dispositivo todo en uno, práctico y compacto. Ideal para llevar a cualquier parte, ofrece hasta 20 usos consecutivos con una sola carga completa. Su diseño integrado lo hace fácil de usar y perfecto para un estilo de vida activo. También cuenta con la tecnología de calentamiento por inducción SMARTCORE."
        },
        {
            "id": 4,
            "nombre": "IQOS 3 DUO",
            "categoria": "Dispositivo",
            "descripcion": "Un dispositivo versátil y rápido que permite dos usos consecutivos sin tener que esperar. Su diseño ergonómico y compacto lo hace cómodo de sostener. Utiliza la tecnología HeatControl™ con una lámina de calentamiento para un sabor consistente. Es robusto y fiable, ideal para el uso diario."
        },
        # Consumibles (Sticks)
        {
            "id": 5,
            "nombre": "TEREA Amber",
            "categoria": "Stick",
            "descripcion": "Una mezcla de tabaco tostado con notas a nuez y madera. Ofrece un sabor rico y equilibrado, con una intensidad media-alta. Diseñado exclusivamente para los dispositivos IQOS ILUMA."
        },
        {
            "id": 6,
            "nombre": "TEREA Sienna",
            "categoria": "Stick",
            "descripcion": "Una mezcla de tabaco redondeada y tostada con notas amaderadas y de té. Proporciona una experiencia de sabor suave pero satisfactoria. Diseñado para la gama IQOS ILUMA."
        },
        {
            "id": 7,
            "nombre": "TEREA Turquoise",
            "categoria": "Stick",
            "descripcion": "Una mezcla de tabaco ligeramente tostado con un refrescante sabor a mentol y notas aromáticas cítricas. Ideal para quienes prefieren una sensación fresca y vibrante. Exclusivo para IQOS ILUMA."
        },
        {
            "id": 8,
            "nombre": "HEETS Amber Selection",
            "categoria": "Stick",
            "descripcion": "Una mezcla de tabaco tostado con un aroma a nuez. Sabor intenso y con cuerpo. Compatible con dispositivos IQOS 3 DUO y modelos anteriores que utilizan lámina de calentamiento."
        },
        {
            "id": 9,
            "nombre": "HEETS Sienna Selection",
            "categoria": "Stick",
            "descripcion": "Un sabor a tabaco redondeado y amaderado. Equilibrado y con cuerpo. Para usar con dispositivos IQOS con tecnología HeatControl™."
        },
        {
            "id": 10,
            "nombre": "HEETS Turquoise Selection",
            "categoria": "Stick",
            "descripcion": "Una mezcla de tabaco mentolado que proporciona una sensación refrescante y suave. Para dispositivos IQOS que calientan con lámina."
        },
        # Accesorios
        {
            "id": 11,
            "nombre": "Funda de Cuero para IQOS ILUMA",
            "categoria": "Accesorio",
            "descripcion": "Una funda protectora elegante fabricada en cuero de alta calidad. Protege tu dispositivo de arañazos y golpes, a la vez que añade un toque de estilo premium. Disponible en varios colores."
        },
        {
            "id": 12,
            "nombre": "Estación de Carga para IQOS 3 DUO",
            "categoria": "Accesorio",
            "descripcion": "Una base de carga de escritorio elegante y funcional. Permite cargar tu dispositivo IQOS de forma cómoda y mantenerlo siempre listo para usar. Su diseño minimalista se adapta a cualquier entorno."
        },
        {
            "id": 13,
            "nombre": "Contenedor de Viaje para TEREA",
            "categoria": "Accesorio",
            "descripcion": "Un accesorio práctico y portátil para desechar los sticks de tabaco usados de forma limpia y discreta. Ideal para llevar en el coche o de viaje."
        }
    ]
    
    df = pd.DataFrame(productos)
    
    # Crear el directorio 'data' si no existe
    if not os.path.exists('data'):
        os.makedirs('data')
        
    output_path = 'data/iqos_products.csv'
    df.to_csv(output_path, index=False, quoting=1) # quoting=1 es para csv.QUOTE_ALL
    print(f"Dataset generado y guardado en '{output_path}'")

if __name__ == '__main__':
    generar_dataset_iqos()
