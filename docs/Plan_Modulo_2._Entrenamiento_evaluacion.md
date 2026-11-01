# Módulo 2 — Entrenamiento, evaluación y selección de modelos de Machine Learning

**Proyecto:** Análisis de sentimientos de reseñas de productos en español mediante NLP y Machine Learning (CritiqAI)  
**Módulo:** 2 — Modelado predictivo, optimización y serialización  
**Estado:** Plan de trabajo y checklist de actividades  

Teniendo en cuenta el estado actual de CritiqAI, el Módulo 1 ya está desarrollado en gran parte: dataset descargado y preparado, análisis exploratorio, procesamiento NLP, pruebas y extracción de características TF-IDF. Por lo tanto, no vamos a repetir esas actividades ni crear fases innecesarias.

El Módulo 2 comenzará aprovechando los archivos que ya existen y avanzará hasta dejar un modelo entrenado, evaluado y listo para integrarse con FastAPI en el Módulo 3.

---

## 1. Estado actual del proyecto

Según la estructura de carpetas que compartiste, tenemos lo siguiente:

| Elemento | Estado |
|---|:---:|
| Dataset original `train.jsonl` | **Realizado** |
| Dataset binario y división entrenamiento/prueba | **Realizado** |
| Análisis exploratorio (EDA) | **Realizado** |
| Limpieza y normalización de texto | **Realizado** |
| Tokenización y eliminación de stopwords | **Realizado** |
| Stemming y lematización | **Realizado** |
| Pipeline de procesamiento NLP | **Realizado** |
| TF-IDF de unigramas | **Realizado** |
| TF-IDF de unigramas y bigramas | **Realizado** |
| Pruebas del módulo NLP | **Realizado** |
| Entrenamiento de modelos ML | *Pendiente* |
| Evaluación y comparación de modelos | *Pendiente* |
| Optimización mediante validación cruzada | *Pendiente* |
| Análisis de errores e interpretación | *Pendiente* |
| Selección y serialización del modelo final | *Pendiente* |
| Integración con FastAPI | *Módulo 3* |

### Estructura que ya tenemos y debemos aprovechar

```text
CritiqAI/
│
├── data/
│   ├── raw/
│   │   └── train.jsonl
│   │
│   ├── processed/
│   │   ├── resenas_binarias.csv
│   │   ├── train_binario.csv
│   │   ├── test_binario.csv
│   │   └── dataset_final.csv
│   │
│   └── features/
│       ├── tfidf_unigramas/
│       │   ├── pipeline_unigramas.joblib
│       │   ├── X_train_tfidf.npz
│       │   └── X_test_tfidf.npz
│       │
│       └── tfidf_unigramas_bigramas/
│           ├── pipeline_unigramas_bigramas.joblib
│           ├── X_train_tfidf.npz
│           └── X_test_tfidf.npz
│
├── src/
│   ├── models/
│   └── nlp/
│
├── scripts/
├── artifacts/
├── api/
├── frontend/
├── notebooks/
├── tests/
├── docs/
│
├── requirements.txt
└── README.md
```

---

## 2. Fases de desarrollo del Módulo 2

---

### Fase 1. Preparación y verificación de los datos para Machine Learning

**Objetivo:** preparar los datos de entrenamiento y prueba que serán utilizados por los modelos de clasificación, aprovechando los resultados del Módulo 1.

#### Actividades
- [ ] 1. Revisar los archivos `train_binario.csv` y `test_binario.csv`.
- [ ] 2. Identificar la columna que contiene las etiquetas de sentimiento.
- [ ] 3. Comprobar que las clases negativa y positiva estén correctamente distribuidas.
- [ ] 4. Cargar las matrices TF-IDF ya generadas:
  - `X_train_tfidf.npz`
  - `X_test_tfidf.npz`
- [ ] 5. Comprobar que las dimensiones de las matrices coincidan con las cantidades de registros de entrenamiento y prueba.
- [ ] 6. Verificar y obtener las etiquetas `y_train` y `y_test` directamente de los CSV, ya que en la estructura actual no aparecen archivos `.npy` con esas etiquetas.
- [ ] 7. Verificar que las matrices y etiquetas correspondan al mismo orden de registros.

#### Resultado esperado
Contar con los datos necesarios para entrenar y evaluar los modelos:
* `X_train`: características de entrenamiento.
* `y_train`: etiquetas de entrenamiento.
* `X_test`: características de prueba.
* `y_test`: etiquetas de prueba.

> [!NOTE]
> No se volverá a procesar el dataset ni a generar TF-IDF si los archivos existentes son correctos.

---

### Fase 2. Desarrollo y entrenamiento de los modelos de clasificación

**Objetivo:** entrenar modelos de Machine Learning capaces de clasificar reseñas de productos en español como positivas o negativas.

#### Actividades
- [ ] 1. Preparar un modelo base (Baseline) utilizando `DummyClassifier`.
- [ ] 2. Implementar el algoritmo de Regresión Logística (`LogisticRegression`).
- [ ] 3. Implementar el algoritmo Naive Bayes Multinomial (`MultinomialNB`).
- [ ] 4. Entrenar los modelos utilizando las características TF-IDF de unigramas.
- [ ] 5. Entrenar nuevamente los modelos utilizando las características TF-IDF de unigramas y bigramas.
- [ ] 6. Registrar los resultados de cada entrenamiento.
- [ ] 7. Mantener las mismas condiciones de entrenamiento para que la comparación sea justa.

#### Experimentos previstos

| Experimento | Representación | Modelo |
|:---:|---|---|
| 1 | TF-IDF unigramas | Baseline |
| 2 | TF-IDF unigramas | Regresión Logística |
| 3 | TF-IDF unigramas | Naive Bayes |
| 4 | TF-IDF unigramas + bigramas | Regresión Logística |
| 5 | TF-IDF unigramas + bigramas | Naive Bayes |

#### Resultado esperado
Obtener cinco experimentos entrenados y contar con resultados iniciales que permitan identificar qué algoritmo y representación ofrecen mejores resultados.

#### Archivos que se desarrollarán
```text
src/
└── models/
    ├── entrenar.py
    └── __init__.py
```

---

### Fase 3. Evaluación y comparación del rendimiento de los modelos

**Objetivo:** medir el desempeño de los modelos entrenados e identificar cuál ofrece mejores resultados para la clasificación de sentimientos.

#### Actividades
- [ ] 1. Evaluar los modelos utilizando los datos de entrenamiento y validación cuando corresponda.
- [ ] 2. Utilizar el conjunto de prueba únicamente para la evaluación final.
- [ ] 3. Calcular las siguientes métricas:
  - Accuracy.
  - Precision.
  - Recall.
  - F1-score.
- [ ] 4. Generar matrices de confusión.
- [ ] 5. Comparar los resultados de los cinco experimentos.
- [ ] 6. Identificar los modelos que presentan mejor rendimiento.
- [ ] 7. Generar tablas y gráficos para documentar los resultados.

#### Resultado esperado
Una comparación objetiva del rendimiento de los modelos.

#### Archivos que se desarrollarán
```text
src/
└── models/
    └── evaluar.py

artifacts/
├── metricas_modelos.json
└── comparacion_modelos.csv
```

---

### Fase 4. Optimización y ajuste de hiperparámetros

**Objetivo:** mejorar el rendimiento del modelo seleccionado mediante la búsqueda de combinaciones adecuadas de hiperparámetros.

#### Actividades
- [ ] 1. Seleccionar el modelo con mejores resultados iniciales.
- [ ] 2. Identificar los hiperparámetros que pueden optimizarse.
- [ ] 3. Aplicar `GridSearchCV` o `RandomizedSearchCV`.
- [ ] 4. Utilizar validación cruzada estratificada (`StratifiedKFold`).
- [ ] 5. Evaluar diferentes configuraciones de hiperparámetros.
- [ ] 6. Identificar la combinación que obtiene el mejor resultado de validación.
- [ ] 7. Comparar el modelo optimizado con su versión inicial.

#### Consideración técnica importante

> [!WARNING]
> La validación cruzada debe realizarse exclusivamente con los datos de entrenamiento. Además, si se ajusta el TF-IDF dentro de la validación cruzada, el vectorizador debe formar parte de un Pipeline de Scikit-learn para que su vocabulario y sus ponderaciones IDF se calculen dentro de cada partición.
>
> Como ya existen matrices TF-IDF calculadas, primero verificaremos cómo fueron generadas. Si el vectorizador se ajustó antes de la validación cruzada con todo el conjunto de entrenamiento, tendremos que corregir ese procedimiento para evitar fuga de información (*data leakage*).
>
> **El conjunto de prueba no debe utilizarse para seleccionar hiperparámetros.**

#### Resultado esperado
Un modelo optimizado y una comparación entre sus métricas antes y después del ajuste.

#### Archivo que se desarrollará
```text
src/
└── models/
    └── optimizar.py
```

---

### Fase 5. Análisis de errores e interpretación del modelo

**Objetivo:** comprender el comportamiento del modelo y analizar los errores que comete durante la clasificación de reseñas.

#### Actividades
- [ ] 1. Identificar reseñas clasificadas incorrectamente.
- [ ] 2. Separar los errores en:
  - Falsos positivos.
  - Falsos negativos.
- [ ] 3. Analizar reseñas que contienen negaciones, sarcasmo, expresiones coloquiales o palabras ambiguas.
- [ ] 4. Identificar las características que más influyen en las predicciones.
- [ ] 5. Examinar los coeficientes de la Regresión Logística, cuando sea el modelo seleccionado.
- [ ] 6. Analizar las limitaciones de los modelos utilizados.
- [ ] 7. Documentar ejemplos de predicciones correctas e incorrectas.

#### Resultado esperado
Un análisis que permita explicar no solamente cuánto acierta el modelo, sino también por qué puede equivocarse.

#### Archivo que se desarrollará
```text
src/
└── models/
    └── analizar_errores.py
```

---

### Fase 6. Selección, entrenamiento final y serialización del modelo

**Objetivo:** seleccionar el modelo definitivo y guardar todos los componentes necesarios para utilizarlo posteriormente desde el backend.

#### Actividades
- [ ] 1. Comparar los resultados obtenidos en las fases anteriores.
- [ ] 2. Seleccionar el modelo definitivo considerando F1-score, precisión, recall y matriz de confusión.
- [ ] 3. Entrenar la configuración seleccionada utilizando los datos de entrenamiento disponibles.
- [ ] 4. Evaluar el modelo final con el conjunto de prueba, que debe permanecer separado del entrenamiento y del ajuste.
- [ ] 5. Guardar el modelo entrenado mediante `joblib`.
- [ ] 6. Guardar el vectorizador y los componentes necesarios para reproducir el procesamiento de texto.
- [ ] 7. Guardar la configuración y las métricas finales.
- [ ] 8. Crear una función de predicción que permita recibir una reseña y devolver:
  - Sentimiento predicho.
  - Probabilidad estimada de cada clase.
- [ ] 9. Comprobar que el modelo serializado pueda cargarse y utilizarse correctamente.

#### Resultado esperado
Una versión final del modelo que pueda ser utilizada desde FastAPI sin volver a implementar manualmente el procesamiento NLP.

#### Estructura prevista
```text
artifacts/
├── modelo_sentimientos.joblib
├── vectorizador.joblib
├── metricas_modelo.json
└── configuracion_modelo.json

src/
└── models/
    └── predictor.py
```

> [!NOTE]
> Los archivos definitivos dependerán de cómo esté construido el pipeline actual. No necesariamente necesitaremos guardar un vectorizador independiente si ya está incluido dentro del objeto serializado.

---

### Fase 7. Automatización, pruebas y documentación del módulo

**Objetivo:** garantizar que el entrenamiento y la evaluación puedan reproducirse y que los componentes estén preparados para integrarse con el backend.

#### Actividades
- [ ] 1. Crear un script que ejecute el proceso de entrenamiento y evaluación.
- [ ] 2. Evitar que sea necesario ejecutar manualmente cada experimento.
- [ ] 3. Comprobar que el modelo final se guarde correctamente.
- [ ] 4. Probar la carga del modelo serializado.
- [ ] 5. Realizar predicciones con reseñas nuevas.
- [ ] 6. Verificar que las probabilidades devueltas sean válidas.
- [ ] 7. Documentar los algoritmos, hiperparámetros, métricas y resultados.
- [ ] 8. Registrar los cambios realizados en Git.

#### Resultado esperado
Poder ejecutar el proceso de entrenamiento mediante un solo comando, por ejemplo:
```powershell
python -m src.models.entrenar
```

Este comando deberá ejecutar el flujo definido y producir los artefactos correspondientes.

---

## 3. Estructura final prevista del Módulo 2

Esta es la estructura a la que queremos llegar, aprovechando los directorios que ya existen:

```text
CritiqAI/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── src/
│   ├── nlp/
│   │   ├── limpieza.py
│   │   ├── tokenizacion.py
│   │   ├── stopwords.py
│   │   ├── stemming.py
│   │   ├── lematizacion.py
│   │   ├── pipeline_nlp.py
│   │   └── vectorizacion.py
│   │
│   └── models/
│       ├── __init__.py
│       ├── entrenar.py
│       ├── evaluar.py
│       ├── optimizar.py
│       ├── analizar_errores.py
│       └── predictor.py
│
├── artifacts/
│   ├── modelo_sentimientos.joblib
│   ├── metricas_modelo.json
│   ├── comparacion_modelos.csv
│   └── configuracion_modelo.json
│
├── scripts/
├── api/
├── frontend/
├── notebooks/
├── tests/
├── docs/
│
├── requirements.txt
└── README.md
```

> [!TIP]
> No crearemos todos estos archivos de golpe. Los iremos incorporando cuando corresponda a cada fase y revisando primero si ya existe alguna implementación.

---

## 4. Orden de ejecución y entregables

| Orden | Fase | Entregable principal |
|:---:|---|---|
| **1** | Preparación de datos | Datos y etiquetas correctamente cargados |
| **2** | Entrenamiento | Cinco experimentos de clasificación |
| **3** | Evaluación | Métricas y comparación de modelos |
| **4** | Optimización | Mejor configuración de hiperparámetros |
| **5** | Análisis | Informe de errores e interpretación |
| **6** | Selección final | Modelo serializado y predictor |
| **7** | Automatización | Entrenamiento reproducible y pruebas |

---

## 5. Criterios para considerar terminado el Módulo 2

Antes de pasar al Módulo 3, debemos comprobar que:

- [ ] Se entrenaron y compararon los modelos establecidos.
- [ ] Se utilizó un modelo base como referencia.
- [ ] Se calcularon Accuracy, Precision, Recall y F1-score.
- [ ] Se generaron matrices de confusión.
- [ ] Se realizó optimización mediante validación cruzada.
- [ ] Se evitó utilizar el conjunto de prueba durante el ajuste.
- [ ] Se analizaron errores de clasificación.
- [ ] Se justificó la elección del modelo final.
- [ ] Se guardó el modelo y sus componentes mediante joblib.
- [ ] Se pueden realizar predicciones con reseñas nuevas.
- [ ] Se puede reproducir el entrenamiento mediante un comando.
- [ ] Se documentaron los resultados y se realizaron pruebas.

---

## 6. ¿Cómo comenzaremos?

No vamos a reinstalar dependencias, crear otro entorno virtual ni repetir el Módulo 1. Tampoco necesitamos ejecutar comandos de verificación como una fase independiente: la revisión de los archivos existentes será el primer paso de la Fase 1.

El primer trabajo será revisar, dentro de esa fase, tres elementos:
1. La columna de etiquetas de `train_binario.csv` y `test_binario.csv`.
2. Las dimensiones y correspondencia de las matrices TF-IDF.
3. El contenido de `scripts/extraer_features.py` y la función de los archivos `pipeline_*.joblib`.

Con esa revisión podremos decidir si aprovechamos directamente las matrices existentes o si necesitamos ajustar la forma en que se construye el vectorizador para la validación cruzada.

> [!IMPORTANT]
> **Recomendación:** Mantener este orden y no avanzar al entrenamiento hasta confirmar la correspondencia entre las matrices TF-IDF y sus etiquetas. Es un paso pequeño, pero evita entrenar modelos con datos desordenados o introducir fuga de información desde el principio.
