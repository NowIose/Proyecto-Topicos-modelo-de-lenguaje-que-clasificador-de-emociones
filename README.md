# CritiqAI — Clasificador de sentimientos de reseñas en español

Aplicación web que clasifica reseñas de productos en español como **positivas** o
**negativas**, muestra cómo el pipeline de NLP transforma el texto y explica qué palabras
pesaron en la decisión.

Proyecto integrador del primer parcial de **Tópicos y Programación Avanzada**.

| Capa | Tecnología | Dónde |
|---|---|---|
| Datos y EDA | pandas, notebook | `data/`, `notebooks/`, `scripts/` |
| Pipeline de NLP | NLTK (Snowball), spaCy, scikit-learn (TF-IDF) | `src/nlp/` |
| Modelos | Regresión Logística y Naive Bayes Multinomial | `src/models/` |
| Backend | FastAPI | `api/` |
| Frontend | React + Vite | `frontend/` |

**Resultado en prueba (32 000 reseñas no vistas):** Regresión Logística F1 macro **0,911**,
Naive Bayes 0,890, línea base 0,333. El detalle está en [`docs/informe_tecnico.md`](docs/informe_tecnico.md).

## Arquitectura

```
Entrenamiento (python -m src.models.entrenar)
  train_binario.csv ──► PreprocesadorNLP ──► TF-IDF ──► Regresión Logística ──► artifacts/
                       (limpieza, tokens,                                       modelo_sentimientos.joblib
                        stopwords, stemming)                                    metricas_modelo.json

Uso (cada consulta)
  Frontend React ──HTTP──► FastAPI ──► modelo_sentimientos.joblib.predict(texto crudo)
                                 └──► metricas_modelo.json
```

El modelo guardado es **un solo pipeline de scikit-learn** con tres pasos:
`preprocessor` (el `PipelineNLP` del Módulo 1), `vectorizer` (TF-IDF) y `classifier`.
La API lo carga y le pasa el texto tal como lo escribe el usuario, así que el
preprocesamiento en producción es exactamente el del entrenamiento: el backend no
reimplementa ningún paso.

## Instalación desde cero

Requisitos: **Python 3.11** y **Node.js 18 o superior**.

```bash
git clone https://github.com/NowIose/Proyecto-Topicos-modelo-de-lenguaje-que-clasificador-de-emociones.git
cd Proyecto-Topicos-modelo-de-lenguaje-que-clasificador-de-emociones

# Entorno de Python
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt          # incluye el modelo es_core_news_sm de spaCy

# Frontend
cd frontend
npm install
cd ..
```

## Levantar el sistema

El repositorio ya trae el modelo entrenado en `artifacts/`, así que no hace falta
entrenar para usar la app.

**Opción A: todo en un puerto** (lo que se usa en la presentación)

```bash
cd frontend && npm run build && cd ..
uvicorn api.main:app --port 8000
```

Abre <http://localhost:8000>. La documentación interactiva de la API está en
<http://localhost:8000/docs>.

**Opción B: modo desarrollo** (el frontend se recarga solo al editar)

```bash
uvicorn api.main:app --reload --port 8000     # terminal 1
cd frontend && npm run dev                    # terminal 2 -> http://localhost:5173
```

**Opción C: Docker**

```bash
docker compose up --build                     # -> http://localhost:8000
docker compose run --rm entrenar              # reentrena dentro del contenedor
```

## Reentrenar

```bash
python -m src.models.entrenar            # dataset completo, ~9 minutos
python -m src.models.entrenar --rapido   # 8 000 reseñas, ~40 segundos
```

Hace todo con un solo comando y semilla fija (42): compara stemming, lematización y
sin reducción; compara línea base, Regresión Logística y Naive Bayes con unigramas y
con bigramas; ajusta hiperparámetros con `GridSearchCV`; elige el modelo por F1 de
validación cruzada; lo evalúa una sola vez en prueba y guarda el pipeline y las
métricas. Después de reentrenar, reinicia `uvicorn` para que cargue el modelo nuevo.

## Pruebas

```bash
pytest
```

Hay 233 pruebas: NLP (`tests/test_nlp.py`, `tests/test_preprocesador.py`), ML
(`tests/ml/`, incluido el entrenamiento de punta a punta) y API (`tests/api/`).

## Endpoints

| Método | Ruta | Qué hace |
|---|---|---|
| GET | `/api/health` | Estado del servicio y del modelo cargado |
| POST | `/api/predict` | `{"texto": "..."}` devuelve sentimiento, confianza, probabilidades y términos clave |
| POST | `/api/predict/lote` | `{"textos": [...]}` hasta 500 reseñas, con resumen por clase |
| POST | `/api/predict/archivo` | Archivo `.txt` (una reseña por línea) o `.csv` con columna `texto`, máx. 1 MB |
| POST | `/api/preprocess` | Cada paso del NLP: normalizado, tokens, stopwords quitadas, raíces y términos del vocabulario |
| GET | `/api/metrics` | Métricas, comparación de modelos, matriz de confusión y análisis de errores |

Errores: `422` con `{"error": "...", "detalle": [...]}` si el texto está vacío, supera
5000 caracteres, tiene un tipo incorrecto o no deja palabras tras el preprocesamiento;
`503` si no hay modelo entrenado.

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"texto": "Llegó roto y el vendedor no responde"}'
```

## Dónde cambiar cada cosa

| Quiero… | Archivo | Después |
|---|---|---|
| Cambiar stemming por lematización | `config/ml.yaml` → `nlp.metodo_morfologico` | reentrenar |
| Cambiar el rango de n-gramas o el vocabulario | `config/ml.yaml` → `nlp.ngram_range`, `max_features`, `min_df`, `max_df` | reentrenar |
| Agregar o quitar una stop word | `src/nlp/stopwords.py` (`NEGACIONES_Y_MODIFICADORES` son las que se conservan) | reentrenar |
| Cambiar los hiperparámetros explorados | `src/models/entrenar.py` → `GRILLAS` | reentrenar |
| Agregar una validación de entrada | `api/esquemas.py` → `_validar_texto` | reiniciar la API |
| Agregar un endpoint | `api/main.py` | reiniciar la API |
| Mostrar otro dato en la interfaz | `frontend/src/componentes/` | `npm run build` |

## Estructura

```
├── api/                 # FastAPI: main.py (endpoints), esquemas.py (validación), servicio.py (modelo)
├── artifacts/           # modelo_sentimientos.joblib, metricas_modelo.json, comparacion_modelos.csv
├── config/ml.yaml       # rutas, semilla, parámetros de NLP y TF-IDF
├── data/                # raw/ (Hugging Face), processed/ (train/test 80/20), features/
├── docs/                # informe técnico y planes de cada módulo
├── frontend/            # React + Vite
├── notebooks/           # análisis exploratorio
├── scripts/             # preparación del dataset y extracción de features
├── src/nlp/             # limpieza, tokenización, stopwords, stemming, lematización, TF-IDF
├── src/models/          # entrenamiento, validación cruzada, evaluación, inferencia
└── tests/
```

## Equipo

Ver [`CONTRIBUCIONES.md`](CONTRIBUCIONES.md).
