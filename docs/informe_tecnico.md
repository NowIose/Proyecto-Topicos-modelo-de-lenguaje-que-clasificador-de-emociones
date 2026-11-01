# Informe técnico — CritiqAI

Clasificación de sentimiento de reseñas de productos en español.
Tópicos y Programación Avanzada, primer parcial.

## 1. Problema

Una tienda en línea recibe miles de reseñas por día y necesita saber, sin leerlas una
por una, cuáles son quejas. La tarea es de **clasificación binaria**: dada una reseña en
español, predecir si es **positiva** o **negativa** y con qué probabilidad.

## 2. Dataset

| Campo | Valor |
|---|---|
| Fuente | `SetFit/amazon_reviews_multi_es` en Hugging Face (<https://huggingface.co/datasets/SetFit/amazon_reviews_multi_es>), subconjunto en español de *Multilingual Amazon Reviews Corpus* |
| Licencia | Uso académico y de investigación no comercial; prohibido redistribuir con fines de lucro o intentar desanonimizar a los usuarios |
| Fecha de obtención | 31 de octubre de 2026 |
| Registros crudos | 200 000 reseñas con calificación de 1 a 5 estrellas (`data/raw/train.jsonl`) |
| Etiquetado | 1–2 estrellas → negativo, 4–5 → positivo; las de 3 estrellas se excluyen por ambiguas |
| Registros usados | 160 000 (80 000 por clase) |
| División | 80/20 estratificada con semilla 42: 128 000 de entrenamiento y 32 000 de prueba |

El análisis exploratorio (`notebooks/01_analisis_exploratorio.ipynb`) muestra que no hay
valores faltantes ni IDs duplicados, que las clases quedan **balanceadas al 50 %** tras
excluir las de 3 estrellas (por eso no se aplicó ninguna técnica de rebalanceo) y que las
reseñas son cortas: 150 caracteres de promedio y un máximo de ~3100.

El conjunto de prueba solo se usa una vez, al final: la elección del método morfológico,
de los n-gramas, de los hiperparámetros y del modelo se hace con validación cruzada sobre
el conjunto de entrenamiento.

## 3. Pipeline de NLP

Implementado en `src/nlp/` (Módulo 1) y metido dentro del modelo mediante
`PreprocesadorNLP` (`src/nlp/preprocesador.py`), un transformador de scikit-learn.

| Paso | Decisión | Por qué |
|---|---|---|
| Normalización | Minúsculas; se quitan tildes **pero se conserva la ñ**; se eliminan URLs, HTML, números, signos y emojis | Los usuarios escriben con y sin tildes («envio»/«envío»), así que unificarlas reduce el vocabulario. La ñ distingue palabras («año»/«ano»). Los números casi nunca indican sentimiento por sí solos |
| Tokenización | Separación por palabras sobre el texto normalizado | Casos difíciles del dominio: «x» como abreviatura de «por», errores de tipeo («ka pantalla») y palabras pegadas a signos («¡¡pésimo!!»), que quedan resueltos por la normalización previa |
| Stop words | Lista de NLTK en español (381 palabras), **conservando las negaciones e intensificadores** (`no`, `nunca`, `nada`, `muy`, `poco`…) | En sentimiento, «no» invierte el significado: «no funciona» y «funciona» quedarían iguales si se borrara |
| Stemming y lematización | Ambos implementados (Snowball de NLTK y spaCy `es_core_news_sm`) y comparados; queda **stemming** | Ver tabla 3.1 |
| N-gramas | Se compararon unigramas contra unigramas + bigramas; quedan **unigramas + bigramas** | Ver tabla 4.1 |
| TF-IDF | `max_features=10000`, `min_df=2`, `max_df=0.90`, `sublinear_tf=True` | 10 000 términos cubren el vocabulario útil sin sobreajustar; `min_df=2` descarta errores de tipeo únicos; `max_df=0.9` quita términos presentes en casi todas las reseñas; la escala sublineal evita que una palabra repetida domine |

Ejemplo antes y después (lo muestra también la interfaz, en «Cómo ve el modelo esta reseña»):

| Etapa | Resultado |
|---|---|
| Original | No funciona, se apagó a los dos días y el vendedor no responde. |
| Normalizado | no funciona se apago a los dos dias y el vendedor no responde |
| Stop words quitadas | se, a, los, y, el |
| Stemming | no funcion apag dos dias vendedor no respond |
| Términos TF-IDF | no, funcion, apag, dos, dias, vendedor, respond, no funcion, dos dias, vendedor no, no respond |

### 3.1 Stemming contra lematización

Regresión Logística, validación cruzada de 5 pliegues sobre una muestra estratificada de
20 000 reseñas de entrenamiento (lematizar las 128 000 con spaCy tarda unos 8 minutos).

| Reducción | F1 macro (CV) | Vocabulario distinto | Tiempo de preprocesamiento |
|---|---|---|---|
| **Stemming (Snowball)** | **0,8886** | 9 482 | 11 s |
| Lematización (spaCy) | 0,8861 | 15 722 | 91 s |
| Sin reducción | 0,8831 | 19 751 | 2 s |

Las dos reducciones mejoran frente a no reducir porque agrupan variantes («funciona»,
«funcionó», «funcionaba» → «funcion»). Stemming gana por poco y es **8 veces más
rápido**, lo que también importa en la API. La lematización produce formas más legibles,
pero spaCy se equivoca con faltas de ortografía frecuentes en reseñas.

## 4. Modelos

Se usan los dos modelos que la consigna sugiere para clasificar sentimiento:

- **Regresión Logística**: modelo lineal que aprende un peso por término y pasa la suma
  ponderada por una sigmoide para obtener una probabilidad. Con TF-IDF de alta dimensión y
  datos abundantes suele ser el clasificador lineal más fuerte, y sus coeficientes se
  pueden interpretar (la interfaz los muestra).
- **Naive Bayes Multinomial**: aplica el teorema de Bayes suponiendo que los términos son
  independientes dada la clase. Es muy rápido y una referencia clásica para texto, pero esa
  suposición falla cuando las palabras dependen unas de otras («no» + «funciona»).

Además, una **línea base** (`DummyClassifier`, siempre predice la clase más frecuente) para
demostrar que los modelos aportan valor.

### 4.1 Unigramas contra unigramas + bigramas

F1 macro con validación cruzada de 5 pliegues sobre las 128 000 reseñas de entrenamiento:

| Modelo | Unigramas | Unigramas + bigramas |
|---|---|---|
| Regresión Logística | 0,8823 | **0,9069** |
| Naive Bayes | 0,8483 | 0,8888 |
| Línea base | 0,3333 | 0,3333 |

Los bigramas capturan negaciones y frases hechas («no funciona», «muy bien», «no
recomiendo») que como unigramas sueltos pierden el sentido.

### 4.2 Ajuste de hiperparámetros

`GridSearchCV` con `StratifiedKFold` de 5 pliegues, optimizando F1 macro, con el TF-IDF
dentro del pipeline para que se ajuste solo con los datos de cada pliegue (sin fuga de
información).

| Modelo | Grilla | Mejor valor | F1 CV |
|---|---|---|---|
| Regresión Logística | C ∈ {0,5; 1; 2; 4; 8; 16} | C = 2 | 0,9083 |
| Naive Bayes | alpha ∈ {0,05; 0,1; 0,3; 0,5; 1} | alpha = 1 | 0,8888 |

En la Regresión Logística, C es la inversa de la regularización: con C chico el modelo
subajusta y con C grande sobreajusta; el máximo queda en el medio de la grilla.

### 4.3 Comparación final en prueba (32 000 reseñas)

| Modelo | Accuracy | Precisión macro | Recall macro | F1 macro |
|---|---|---|---|---|
| **Regresión Logística (producción)** | **0,9110** | **0,9111** | **0,9110** | **0,9110** |
| Naive Bayes Multinomial | 0,8904 | 0,8906 | 0,8904 | 0,8904 |
| Línea base | 0,5000 | 0,2500 | 0,5000 | 0,3333 |

Matriz de confusión de la Regresión Logística:

| | predicho positivo | predicho negativo |
|---|---|---|
| **real positivo** | 14 426 | 1 574 |
| **real negativo** | 1 275 | 14 725 |

**Queda en producción la Regresión Logística**: gana en validación cruzada (criterio de
elección) y lo confirma en prueba, con 2 puntos más de F1 que Naive Bayes y un
comportamiento equilibrado entre clases (recall 0,90 en positivas y 0,92 en negativas).

## 5. Análisis de errores

El modelo comete 2849 errores en prueba (8,9 %): 1275 falsos positivos y 1574 falsos
negativos. Los errores en los que el modelo estaba más seguro muestran tres causas:

1. **Etiquetas ruidosas.** La etiqueta sale de las estrellas, no del texto. «Cumple
   perfectamente con su función» o «Pieza perfecta todo en orden» están marcadas como
   negativas (1–2 estrellas) aunque el texto es claramente positivo. Aquí el modelo no se
   equivoca: el dataset sí.
2. **Negaciones que cambian el sentido de una palabra negativa.** «Esta marca nunca
   decepciona» se clasifica negativa porque «decepcion» es uno de los términos más
   negativos del vocabulario, y el bigrama «nunca decepcion» es poco frecuente.
3. **Contraste y matices.** «Uno no funciona pero los tres han sido eficaces» o «Aunque el
   semáforo no funciona» mezclan una queja con una valoración positiva; un modelo de bolsa
   de palabras no entiende qué parte pesa más.

Términos con más peso en el modelo: hacia positiva `perfect`, `encant`, `buen`,
`excelent`, `genial`, `nad mal`, `muy bien`; hacia negativa `no recom`, `no`, `no cumpl`,
`decepcion`, `no volveri`, `dej funcion`, `devuelt`. Varios son bigramas con negación, lo
que confirma la decisión de conservar «no» y de usar bigramas.

## 6. Arquitectura

```
                 python -m src.models.entrenar  (un comando, semilla 42)
 data/processed ─────────────────────────────────────────────────────────► artifacts/
 train/test CSV   PreprocesadorNLP → TF-IDF → GridSearchCV → evaluación     modelo_sentimientos.joblib
                                                                           metricas_modelo.json
                                                                                 │ joblib.load
 Navegador ── React (frontend/) ── HTTP JSON ── FastAPI (api/) ─────────────────┘
```

- **Pipeline reproducible.** `src/models/entrenar.py` va de los CSV al modelo guardado con
  un solo comando y versiones fijadas en `requirements.txt`. El artefacto es un único
  `sklearn.Pipeline` (`preprocessor` → `vectorizer` → `classifier`) que recibe texto crudo;
  antes de guardarlo, el script verifica que prediga exactamente igual que el modelo
  evaluado.
- **Backend (FastAPI).** Se eligió porque valida la entrada con Pydantic, genera la
  documentación OpenAPI sola (`/docs`) y está en Python, así carga el modelo con el mismo
  código con que se entrenó. El modelo se carga una vez al arrancar, a través del
  `SentimentClassifier` del Módulo 2. Endpoints: `/api/health`, `/api/predict`,
  `/api/predict/lote`, `/api/predict/archivo`, `/api/preprocess` y `/api/metrics`. La API
  valida texto vacío, demasiado largo (más de 5000 caracteres), tipo incorrecto, JSON mal
  formado, lotes vacíos o de más de 500 textos, archivos de formato no soportado y textos
  que no dejan palabras tras el preprocesamiento.
- **Frontend (React + Vite).** Tres vistas: analizar una reseña (veredicto, términos que
  pesaron, reseña marcada y paso a paso del NLP), analizar un archivo `.txt` o `.csv`, y
  el panel de métricas con la comparación de modelos.
- **Pruebas.** 233 pruebas con pytest: NLP, ML (incluido el entrenamiento completo con un
  dataset diminuto) y API con `TestClient`.

## 7. Limitaciones y trabajo futuro

- El etiquetado por estrellas introduce ruido (sección 5); revisar a mano una muestra
  daría una medida más honesta del techo del modelo.
- Un modelo de bolsa de palabras no capta el orden ni el contraste («pero», «aunque»);
  modelos secuenciales o de tipo transformer lo harían mejor, pero quedan fuera de los
  modelos vistos en la materia.
- Las reseñas de 3 estrellas (neutras) se excluyeron; el sistema siempre responde
  positivo o negativo.

## 8. Uso de asistentes de IA

Se usó **Claude Code** (Anthropic) como asistente de programación en la integración del
pipeline de NLP con los modelos, el script de entrenamiento, el backend, el frontend y esta
documentación. Los commits en los que intervino llevan la línea
`Co-Authored-By: Claude`. Todo el código fue revisado, ejecutado y probado por el equipo, y
cada integrante puede explicar y modificar lo que subió.
