## 🏗️ ESTRUCTURA DE CARPETAS PRINCIPALES

```
proyecto_zero_shot_recomendaciones/
├── src/                          # files de python
├── data/                         # Datos y embeddings
├── eval/                         # Evaluaciones y métricas
├── graficos/                     # Visualizaciones generadas
├── requirements.txt              # Dependencias de Python
├── .env                          # Env. file
├── README.md                     
└── .gitignore                    
```


## 🛠️ CONFIGURACIÓN INICIAL REQUERIDA

### 1. **Entorno Virtual Python**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

### 2. **Instalación de Dependencias**
```bash
pip install -r requirements.txt
```


### 3. **Variables de Entorno** (`.env`)
```bash
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🔄 FLUJO DE EJECUCIÓN TÍPICO

### **Paso 1: Preparación de Datos**
```bash
python src/generar_dataset_iqos.py
python src/generar_embeddings_iqos.py
```

### **Paso 2: Evaluación de Sistemas**
```bash
python src/evaluar_modelos.py
```

### **Paso 3: Cálculo de Métricas**  
```bash
python src/calcular_metricas_con_guardado.py
```

---

## 📋 CHECKLIST DE CONFIGURACIÓN

- Crear estructura de carpetas
- Configurar entorno virtual Python  
- Instalar dependencias (`requirements.txt`)
- Configurar API keys (Groq/OpenAI)
- Preparar dataset de productos con descripciones curadas
- Crear ground truth manual para evaluación
- Ejecutar pipeline de generación de embeddings
- Probar los tres sistemas de recomendación
- Validar métricas de evaluación
- Generar visualizaciones iniciales

---


