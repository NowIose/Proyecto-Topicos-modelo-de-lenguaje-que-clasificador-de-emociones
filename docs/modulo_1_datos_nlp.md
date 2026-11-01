# Documentación Técnica Detallada — Módulo 1: Datos y Procesamiento NLP

**Proyecto:** Análisis de sentimientos de reseñas de productos en español mediante NLP y Machine Learning (Opinómetro)  
**Dataset oficial:** `SetFit/amazon_reviews_multi_es`  
**Ubicación de datos crudos:** `data/raw/train.jsonl`  
**Responsable del módulo:** José (Integrante 1)  
**Fecha de finalización:** 31 de octubre de 2026  

---

## 1. Resumen Ejecutivo del Módulo

El **Módulo 1** establece los cimientos de datos y procesamiento de lenguaje natural del proyecto. Su función es transformar reseñas textuales sin estructurar en representaciones numéricas óptimas (TF-IDF) para el posterior entrenamiento de modelos de Machine Learning (Regresión Logística, Random Forest, etc.), asegurando:

1. **Inmutabilidad de datos originales:** `data/raw/train.jsonl` permanece intacto.
2. **Transformación a clasificación binaria:** Exclusión justificada de reseñas neutrales (3 estrellas).
3. **División sin fuga de información (*data leakage*):** Estratificación estricta (80% train / 20% test).
4. **Pipeline NLP con preservación semántica:** Normalización diacrítica protegiendo la `ñ` y **retención estricta de negaciones** (`no`, `nunca`, `jamás`, `sin`) para no invertir polaridades.
5. **Generación de variantes para experimentación:** Unigramas vs. Unigramas + Bigramas, y comparación morfológica Stemming vs. Lematización.
6. **Validación automática continua:** 14 pruebas unitarias con `pytest` pasando al 100%.

---

## 2. Mapa General de Archivos y Entregables del Módulo

```text
CritiqAI/
├── data/
│   ├── raw/
│   │   └── train.jsonl                          # Dataset original intacto (200.000 reseñas)
│   ├── processed/
│   │   ├── resenas_binarias.csv                 # 160.000 reseñas binarias completas
│   │   ├── train_binario.csv                    # 128.000 reseñas para entrenamiento (80%)
│   │   ├── test_binario.csv                     # 32.000 reseñas para evaluación aislada (20%)
│   │   └── dataset_final.csv                    # Dataset con columna 'text_limpio' preprocesada
│   └── features/
│       ├── tfidf_unigramas/                     # Configuración A (unigramas, 10.000 vocabulario)
│       │   ├── pipeline_unigramas.joblib
│       │   ├── X_train_tfidf.npz
│       │   └── X_test_tfidf.npz
│       └── tfidf_unigramas_bigramas/            # Configuración B (unigramas+bigramas, 25.000 vocabulario)
│           ├── pipeline_unigramas_bigramas.joblib
│           ├── X_train_tfidf.npz
│           └── X_test_tfidf.npz
├── notebooks/
│   └── 01_analisis_exploratorio.ipynb           # EDA exhaustivo con tablas y gráficos interactivos
├── src/
│   └── nlp/
│       ├── __init__.py                          # Exportación modular de funciones
│       ├── limpieza.py                          # Normalización, URLs, HTML, diacríticos y 'ñ'
│       ├── tokenizacion.py                      # Tokenizador regex especializado para español
│       ├── stopwords.py                         # Stopwords con preservación de negaciones
│       ├── stemming.py                          # Stemmer Snowball en español (alta velocidad)
│       ├── lematizacion.py                      # Lematizador neuronal spaCy (es_core_news_sm)
│       ├── vectorizacion.py                     # Wrapper VectorizadorTFIDF con joblib
│       └── pipeline_nlp.py                      # Clase PipelineNLP unificada con inspección paso a paso
├── scripts/
│   ├── probar_dataset.py                        # Script de validación de integridad UTF-8
│   ├── crear_eda_notebook.py                    # Generador del notebook de EDA
│   ├── preparar_dataset_binario.py              # Script de mapeo binario y partición train/test
│   └── extraer_features.py                      # Generador de matrices dispersas TF-IDF y modelos
├── tests/
│   └── test_nlp.py                              # 14 pruebas unitarias automatizadas
├── pytest.ini                                   # Configuración de ruta raíz para tests
└── requirements.txt                             # Dependencias del proyecto
```

---

## 3. Fase 1 — Obtención, Verificación y Ficha Técnica del Dataset

### Funcionalidad
Comprobar la integridad física y la codificación de caracteres en español del dataset descargado, documentar las fuentes, versiones y restricciones de licenciamiento para garantizar un desarrollo académico formal.

### Archivos involucrados
* **`data/raw/train.jsonl`**: Archivo de origen descargado de Hugging Face.
* **`scripts/probar_dataset.py`**: Script de verificación que lee registros línea por línea, valida la carga de JSON, chequea acentos gráficos (`á`, `é`, `í`, `ó`, `ú`, `ñ`) y calcula totales.

### Ficha Técnica del Dataset
* **Nombre:** `SetFit/amazon_reviews_multi_es`
* **Repositorio oficial:** <https://huggingface.co/datasets/SetFit/amazon_reviews_multi_es>
* **Corpus base:** *Multilingual Amazon Reviews Corpus* (subconjunto en español).
* **Licencia:** *Academic / Research Non-Commercial Use Only*. Uso exclusivo para investigación académica. Queda prohibido el lucro comercial, la redistribución de datos o cualquier intento de desanonimizar identidades de usuarios.
* **Formato:** JSON Lines (JSONL), UTF-8.
* **Cantidad de registros:** 200.000 reseñas.
* **Variables identificadas:**
  * `id` (`str`): Identificador único de la reseña (ej. `"es_0491108"`).
  * `text` (`str`): Contenido textual de la opinión.
  * `label` (`int`): Calificación original (0 a 4).
  * `label_text` (`str`): Representación textual de la etiqueta (`"0"` a `"4"`).
* **Distribución original:** Perfectamente balanceada (40.000 ejemplos por etiqueta del 0 al 4).

### Comando de verificación
```powershell
python scripts/probar_dataset.py
```

---

## 4. Fase 2 — Análisis Exploratorio de Datos (EDA)

### Funcionalidad
Analizar exhaustivamente la calidad de datos, balance de clases, distribución de longitud y patrones léxicos para fundamentar las decisiones de diseño del pipeline NLP.

### Archivos involucrados
* **`notebooks/01_analisis_exploratorio.ipynb`**: Cuaderno Jupyter ejecutable con análisis visuales, tablas estadísticas y gráficos generados con Matplotlib y Seaborn.
* **`scripts/crear_eda_notebook.py`**: Script que construye el notebook usando la API de `nbformat`.

### Hallazgos Principales del EDA

1. **Calidad de Datos:**
   * Nulos: 0 en todas las columnas.
   * Cadenas vacías o sólo espacios: 0.
   * IDs duplicados: 0 (todos son únicos).
   * Textos repetidos: 1.736 reseñas (0.87%). Corresponden a respuestas breves clichés del comercio electrónico (*"Buena relación calidad precio"*, *"Cumple las expectativas"*). No representan un error y se conservan.

2. **Distribución de Clases:**
   * Exactamente 40.000 reseñas por estrella (20% cada una). Sin desbalance inicial.

3. **Métricas de Longitud:**
   * **Caracteres:** Media: 151.3 | Mediana: 120 | Mínimo: 16 | Máximo: 3.086.
   * **Palabras:** Media: 27.8 | Mediana: 22 | Mínimo: 2 | Máximo: 551.
   * **Comportamiento por polaridad:** Las reseñas de 1 y 2 estrellas son sensiblemente más largas (mediana de 25-28 palabras) que las de 5 estrellas (mediana de 16 palabras), ya que los clientes insatisfechos detallan minuciosamente cada defecto encontrado.

4. **Patrón Léxico Crítico (Justificación de Stopwords):**
   * En opiniones muy negativas (clase 0), la palabra **`"no"` es el término #1 más frecuente** (46.621 apariciones en clase 0 vs sólo 11.534 en clase 4).
   * Eliminar negaciones en el preprocesamiento destruiría la principal señal predictiva del modelo.

### Comando de verificación
Abrir en el editor el notebook: `notebooks/01_analisis_exploratorio.ipynb`.

---

## 5. Fase 3 — Preparación de Etiquetas Binarias

### Funcionalidad
Mapear la escala ordinal de 5 estrellas a una formulación de clasificación binaria estándar: **negativo** vs **positivo**, resolviendo la ambigüedad inherente a las calificaciones intermedias.

### Regla de Mapeo
* **Etiquetas 0 y 1 (1 y 2 estrellas):** Convertidas a `'negativo'` (`polaridad = 0`). Total: 80.000 reseñas.
* **Etiqueta 2 (3 estrellas):** **Excluida del problema binario.** Las opiniones de 3 estrellas son valoraciones neutras o tibias (ej. *"bueno pero caro"*, *"regular"*) que introducen ruido en modelos binarios. Se apartan 40.000 registros.
* **Etiquetas 3 y 4 (4 y 5 estrellas):** Convertidas a `'positivo'` (`polaridad = 1`). Total: 80.000 reseñas.

### Archivos involucrados
* **`scripts/preparar_dataset_binario.py`**: Automatiza la exclusión de neutrales, crea las columnas `sentimiento` y `polaridad`, y exporta los archivos.
* **`data/processed/resenas_binarias.csv`**: Archivo resultante con las 160.000 reseñas binarias (50% positivo / 50% negativo).

### Comando de verificación
```powershell
python -c "import pandas as pd; df = pd.read_csv('data/processed/resenas_binarias.csv'); print(df['sentimiento'].value_counts())"
```

---

## 6. Fase 4 — Partición Estratificada Train / Test

### Funcionalidad
Dividir el conjunto de datos binario en conjuntos de entrenamiento y prueba de forma estratificada y reproducible, garantizando la ausencia de fuga de información (*data leakage*).

### Especificaciones de la Partición
* **Proporción:** 80% Entrenamiento / 20% Prueba.
* **Estratificación:** `stratify = df['sentimiento']` (garantiza exactamente 50% positivas y 50% negativas en ambos conjuntos).
* **Semilla aleatoria fija:** `random_state = 42`.
* **Conjunto de Entrenamiento:** 128.000 registros (64.000 neg / 64.000 pos).
* **Conjunto de Prueba:** 32.000 registros (16.000 neg / 16.000 pos).
* **Regla estricta:** El conjunto de prueba se mantiene intacto y jamás se utiliza para ajustar parámetros, diccionarios de stopwords ni vocabularios TF-IDF.

### Archivos involucrados
* **`data/processed/train_binario.csv`**: Partición de entrenamiento (128.000 filas).
* **`data/processed/test_binario.csv`**: Partición de prueba (32.000 filas).

---

## 7. Fase 5 — Pipeline Modular de NLP (`src/nlp/`)

### Funcionalidad
Implementación desacoplada y orientada a objetos de cada una de las técnicas de procesamiento vistas en clase.

### 7.1 Normalización (`src/nlp/limpieza.py`)
* Convierte a minúsculas.
* Remueve URLs (`https?://\S+|www\.\S+`) y etiquetas HTML (`<.*?>`).
* **Tratamiento de diacríticos y la letra `ñ`:** Remueve acentos gráficos (`pésimo` -> `pesimo`) mediante descomposición Unicode `NFKD`, pero **protege expresamente la letra `ñ`** para evitar la deformación semántica de vocablos como `año` o `diseño`.
* **Caracteres repetidos:** Utiliza expresiones regulares (`(.)\1{2,}`) para colapsar repeticiones enfáticas (`buenoooo` -> `bueno`, `pesimoooo` -> `pesimo`).
* Elimina caracteres especiales y signos de puntuación, preservando letras del abecedario español y espacios normalizados.

### 7.2 Tokenización (`src/nlp/tokenizacion.py`)
* Convierte cadenas de texto en listas de palabras (`List[str]`).
* Emplea un patrón de expresión regular compilado (`[a-záéíóúüñA-ZÁÉÍÓÚÜÑ0-9]+`) adaptado a caracteres ortográficos del español.

### 7.3 Stopwords con Preservación de Negaciones (`src/nlp/stopwords.py`)
* Utiliza como base la lista de stopwords en español de NLTK (313 términos).
* **Regla Crítica de Sentimiento:** NLTK incluye por defecto términos como `"no"`, `"ni"`, `"pero"`, `"sin"`, `"nada"`, `"muy"`. Nuestro módulo **excluye de forma mandatoria** estas palabras de la lista de eliminación:
  ```python
  NEGACIONES_Y_MODIFICADORES = {
      "no", "nada", "ni", "nunca", "jamas", "jamás",
      "tampoco", "sin", "muy", "pero", "mas", "más",
      "poco", "bueno", "buena", "mal", "mala", "malo"
  }
  ```
* Genera además variantes sin tilde de las stopwords para coincidir exactamente con el texto normalizado.

### 7.4 Stemming (`src/nlp/stemming.py`)
* Implementa el algoritmo `SnowballStemmer("spanish")` (Porter 2).
* Reduce las palabras a su raíz morfológica común (`productos` -> `product`, `comprando` -> `compr`, `funciona` -> `funcion`).
* **Ventaja:** Muy alta velocidad de ejecución (procesa las 128.000 reseñas en ~15 segundos).

### 7.5 Lematización (`src/nlp/lematizacion.py`)
* Implementa reducción lingüística con spaCy mediante el modelo liviano `es_core_news_sm`.
* Deshabilita los componentes pesados `parser` y `ner` para maximizar rendimiento.
* Reduce palabras a su forma canónica de diccionario (`productos` -> `producto`, `comprando` -> `comprar`, `funciona` -> `funcionar`).
* **Ventaja:** Gramaticalmente correcta y lemas interpretables.
* **Desventaja computacional:** Aproximadamente 25 veces más lenta que el stemming.

### 7.6 Pipeline Unificado (`src/nlp/pipeline_nlp.py`)
* Clase `PipelineNLP` que conecta todas las fases.
* Admite seleccionar `metodo_morfologico` (`'stemming'`, `'lematizacion'` o `'ninguno'`).
* Ofrece el método `inspeccionar_paso_a_paso(texto)` para retornar un diccionario con las transformaciones intermedias:
  ```python
  {
      "texto_original": "...",
      "texto_normalizado": "...",
      "tokens": [...],
      "tokens_sin_stopwords": [...],
      "tokens_stemming": [...],
      "texto_final": "...",
      "vector_tfidf": <matriz_dispersa>
  }
  ```
* Incluye métodos de serialización `guardar(ruta)` y `cargar(ruta)` con `joblib`.

### Comando de prueba interactiva
```powershell
python -c "from src.nlp.pipeline_nlp import PipelineNLP; p = PipelineNLP(); print(p.inspeccionar_paso_a_paso('¡El producto NO funciona! Es una pésima compra 😡'))"
```

---

## 8. Fase 6 y 7 — N-Gramas y Vectorización TF-IDF (`src/nlp/vectorizacion.py`)

### Funcionalidad
Transformar las cadenas textuales procesadas en matrices dispersas de valores numéricos TF-IDF con ponderación sublineal (`sublinear_tf=True`), evaluando dos configuraciones para el posterior modelado.

### Configuraciones Generadas

| Parámetro | Configuración A (Unigramas) | Configuración B (Unigramas + Bigramas) |
|---|:---:|:---:|
| **Rango de n-gramas** | `(1, 1)` | `(1, 2)` |
| **Límite de características (`max_features`)** | 10.000 | 25.000 |
| **Frecuencia mínima documental (`min_df`)** | 2 | 2 |
| **Frecuencia máxima documental (`max_df`)** | 0.90 | 0.90 |
| **Escalado sublineal (`sublinear_tf`)** | `True` ($1 + \log(\text{tf})$) | `True` ($1 + \log(\text{tf})$) |
| **Tamaño real de vocabulario aprendido** | 10.000 términos | 25.000 términos |
| **Forma de la matriz Train** | `(128000, 10000)` | `(128000, 25000)` |
| **Forma de la matriz Test** | `(32000, 10000)` | `(32000, 25000)` |
| **Tiempo de vectorización total** | ~19.6 segundos | ~22.4 segundos |
| **Ejemplos capturados** | `"product"`, `"calid"`, `"no"` | `"no funcion"`, `"excelent calid"`, `"mal servic"` |

### Archivos involucrados
* **`scripts/extraer_features.py`**: Script que ejecuta el ajuste y extracción sobre los datos.
* **`data/features/tfidf_unigramas/`**: Contiene `pipeline_unigramas.joblib`, `X_train_tfidf.npz` y `X_test_tfidf.npz`.
* **`data/features/tfidf_unigramas_bigramas/`**: Contiene `pipeline_unigramas_bigramas.joblib`, `X_train_tfidf.npz` y `X_test_tfidf.npz`.
* **`data/processed/dataset_final.csv`**: Archivo con los 160.000 registros con la columna precalculada `text_limpio`.

---

## 9. Fase 9 — Pruebas Unitarias Automatizadas (`tests/test_nlp.py`)

### Funcionalidad
Garantizar la estabilidad, calidad y correcto funcionamiento de todos los módulos del sistema NLP ante casos normales y casos límite (*edge cases*).

### Archivos involucrados
* **`tests/test_nlp.py`**: Suite de 14 pruebas con `pytest`.
* **`pytest.ini`**: Configuración con `pythonpath = .` para ejecución limpia en cualquier entorno.

### Detalle de las 14 pruebas implementadas
1. `test_limpieza_texto_vacio`: Valida cadenas vacías, `None` y sólo espacios.
2. `test_limpieza_mayusculas`: Comprueba la conversión adecuada a minúsculas.
3. `test_limpieza_tildes_preservando_ene`: Verifica desacentuación protegiendo la `ñ`.
4. `test_limpieza_urls_y_html`: Verifica eliminación de enlaces web y etiquetas HTML.
5. `test_limpieza_caracteres_repetidos`: Verifica reducción de repeticiones (`buenoooo`).
6. `test_limpieza_puntuacion_y_emojis`: Comprueba descarte de signos y emojis.
7. `test_tokenizacion_basica`: Valida división en palabras.
8. `test_tokenizacion_con_caracteres_especiales`: Valida tokens con acentos y `ñ`.
9. `test_preservacion_estricta_de_negaciones`: **Prueba crítica:** Confirma que `"no"`, `"nunca"`, `"sin"`, `"muy"` NO se eliminan.
10. `test_stemming_espanol`: Valida reducción morfológica con Snowball.
11. `test_lematizacion_espanol`: Valida lematización lingüística con spaCy.
12. `test_caso_ilustrativo_completo`: Valida la frase oficial de prueba del plan (`¡El producto NO funciona! Es una pésima compra 😡`).
13. `test_tfidf_unigramas_y_bigramas`: Valida dimensiones, ajuste exclusivo sobre train y persistencia con `joblib`.
14. `test_pipeline_nlp_end_to_end`: Valida el flujo completo desde el texto crudo hasta el vector TF-IDF y su serialización/recuperación en disco.

### Comando de ejecución de pruebas
```powershell
pytest tests/test_nlp.py -v
```
**Resultado:** `14 passed in 2.96s (100% de éxito)`

---

## 10. Contrato de Integración con los Siguientes Módulos

### Para el Módulo 2 (Machine Learning - Antony)
* **Datos listos para modelar:** Puede leer directamente las matrices dispersas ya vectorizadas con:
  ```python
  from scipy import sparse
  X_train = sparse.load_npz("data/features/tfidf_unigramas_bigramas/X_train_tfidf.npz")
  X_test = sparse.load_npz("data/features/tfidf_unigramas_bigramas/X_test_tfidf.npz")
  ```
* **Etiquetas correspondientes:**
  ```python
  import pandas as pd
  y_train = pd.read_csv("data/processed/train_binario.csv")["polaridad"]
  y_test = pd.read_csv("data/processed/test_binario.csv")["polaridad"]
  ```

### Para el Módulo 3 y 4 (Backend FastAPI y Frontend React - Arteaga)
* **Inferencia en producción:** Cuando el backend reciba una reseña escrita por un usuario en la web, cargará el pipeline con una sola línea de código:
  ```python
  from src.nlp.pipeline_nlp import PipelineNLP
  pipeline = PipelineNLP.cargar("data/features/tfidf_unigramas_bigramas/pipeline_unigramas_bigramas.joblib")
  vector_tfidf = pipeline.transform([texto_usuario])
  # O para mostrar las etapas en el frontend:
  pasos_intermedios = pipeline.inspeccionar_paso_a_paso(texto_usuario)
  ```
* Se garantiza que el preprocesamiento de la API será idéntico al utilizado durante el entrenamiento.
