# Módulo 1 — Datos y procesamiento NLP

**Proyecto:** Análisis de sentimientos de reseñas de productos en español mediante NLP y Machine Learning  
**Dataset:** `SetFit/amazon_reviews_multi_es`  
**Archivo de entrenamiento:** `data/raw/train.jsonl`  
**Estado:** Módulo 1 Completado al 100% — Implementado, probado y documentado.

---

## 1. Objetivo del módulo

Preparar, analizar y transformar el conjunto de reseñas en español mediante técnicas de Procesamiento de Lenguaje Natural (NLP), obteniendo datos limpios y representaciones numéricas que puedan utilizarse posteriormente para entrenar modelos de Machine Learning.

El módulo debe dejar un flujo reproducible desde los datos originales hasta los datos procesados y las representaciones TF-IDF.

## 2. Estructura de carpetas propuesta

```text
proyecto/
├── data/
│   ├── raw/
│   │   └── train.jsonl
│   ├── processed/
│   │   ├── resenas_limpias.csv
│   │   ├── resenas_binarias.csv
│   │   └── dataset_final.csv
│   └── features/
│       ├── tfidf_unigramas/
│       └── tfidf_unigramas_bigramas/
├── notebooks/
│   └── 01_analisis_exploratorio.ipynb
├── src/
│   └── nlp/
│       ├── __init__.py
│       ├── limpieza.py
│       ├── tokenizacion.py
│       ├── stopwords.py
│       ├── stemming.py
│       ├── lematizacion.py
│       ├── vectorizacion.py
│       └── pipeline_nlp.py
├── tests/
│   └── test_nlp.py
├── docs/
│   └── modulo_1_datos_nlp.md
└── requirements.txt
```

Los nombres son una propuesta y pueden ajustarse a la estructura general del repositorio.

---

## 3. Fase 1 — Obtención y documentación del dataset

**Objetivo:** identificar, almacenar y documentar correctamente los datos.

- [x] Registrar el nombre: `SetFit/amazon_reviews_multi_es`.
- [x] Registrar el repositorio: <https://huggingface.co/datasets/SetFit/amazon_reviews_multi_es>.
- [x] Guardar el archivo original en `data/raw/train.jsonl`.
- [x] Identificar el formato JSONL y la codificación UTF-8.
- [x] Confirmar las columnas:
  - `id`: identificador de la reseña.
  - `text`: contenido de la reseña.
  - `label`: etiqueta numérica.
  - `label_text`: representación textual de la etiqueta.
- [x] Registrar la fecha de descarga.
- [x] Consultar y documentar la licencia y las condiciones de uso de la versión concreta descargada.
- [x] Registrar la versión o revisión del repositorio, si está disponible.
- [x] Conservar una copia original sin modificaciones.

**Resultado esperado:** dataset original, ficha de procedencia, licencia, fecha de descarga y descripción de variables.

> No modificar directamente el archivo original. Todas las transformaciones deben generar archivos derivados.

---

## 4. Fase 2 — Análisis exploratorio de datos (EDA)

**Objetivo:** conocer las características del dataset antes de aplicar técnicas NLP.

**Notebook:** `notebooks/01_analisis_exploratorio.ipynb`

### 4.1 Inspección general

- [x] Cargar el archivo JSONL con Python.
- [x] Convertir los registros en un DataFrame.
- [x] Mostrar las primeras filas.
- [x] Identificar columnas y tipos de datos.
- [x] Calcular la cantidad total de registros.
- [x] Identificar identificadores duplicados.
- [x] Revisar posibles reseñas duplicadas por contenido.

### 4.2 Valores faltantes e inválidos

- [x] Contar reseñas sin texto.
- [x] Revisar identificadores faltantes.
- [x] Revisar etiquetas faltantes.
- [x] Detectar textos vacíos o compuestos solo por espacios.
- [x] Detectar registros inválidos.
- [x] Documentar cómo se tratará cada caso.

### 4.3 Distribución de etiquetas

El dataset utiliza cinco etiquetas. En los ejemplos revisados, corresponden a una escala de cinco estrellas transformada a valores de 0 a 4:

| Etiqueta | Estrellas originales | Interpretación |
|---:|---:|---|
| 0 | 1 | Negativa |
| 1 | 2 | Negativa |
| 2 | 3 | Neutral |
| 3 | 4 | Positiva |
| 4 | 5 | Positiva |

- [x] Verificar la correspondencia con la documentación de la versión descargada.
- [x] Contar los registros de cada etiqueta.
- [x] Calcular porcentajes.
- [x] Crear un gráfico de barras.
- [x] Identificar si existe desbalance.

**Dato observado en el archivo de entrenamiento inspeccionado:** 200.000 registros, con 40.000 por etiqueta. Volver a ejecutar el conteo sobre el archivo local y documentar el resultado real.

### 4.4 Longitud de las reseñas

- [x] Calcular longitud mínima y máxima en caracteres.
- [x] Calcular promedio y mediana de caracteres.
- [x] Calcular longitud en palabras.
- [x] Crear histogramas o gráficos de distribución.
- [x] Identificar textos extremadamente cortos o largos.

### 4.5 Palabras frecuentes

- [x] Obtener las palabras más frecuentes antes del procesamiento.
- [x] Mostrar las 20 o 30 palabras más frecuentes.
- [x] Observar términos frecuentes en reseñas negativas.
- [x] Observar términos frecuentes en reseñas positivas.
- [x] Identificar palabras comunes poco informativas.
- [x] Documentar observaciones.

**Resultado esperado:** notebook de EDA, tablas estadísticas, gráficos y conclusiones sobre calidad, distribución y características de los textos.

---

## 5. Fase 3 — Preparación de las etiquetas binarias

**Objetivo:** transformar la escala original de cinco categorías en dos clases de sentimiento.

### Transformación propuesta

| Etiqueta original | Nueva categoría |
|---:|---|
| 0 | negativo |
| 1 | negativo |
| 2 | neutral: excluir del clasificador binario |
| 3 | positivo |
| 4 | positivo |

### Actividades

- [x] Crear una columna nueva llamada `sentimiento`.
- [x] Transformar las etiquetas 0 y 1 en `negativo`.
- [x] Excluir la etiqueta 2 del conjunto binario.
- [x] Transformar las etiquetas 3 y 4 en `positivo`.
- [x] Verificar que no queden etiquetas desconocidas.
- [x] Contar nuevamente las clases.
- [x] Calcular porcentajes.
- [x] Guardar el resultado en `data/processed/resenas_binarias.csv`.

### Distribución esperada para el archivo inspeccionado

Si se confirma que existen 40.000 registros por etiqueta:

| Categoría | Cantidad |
|---|---:|
| Negativa (0 + 1) | 80.000 |
| Positiva (3 + 4) | 80.000 |
| Neutral excluida (2) | 40.000 |
| Total original | 200.000 |
| Total binario | 160.000 |

- [x] Confirmar estos valores ejecutando el análisis sobre el archivo local.
- [x] Documentar que las etiquetas binarias son una transformación de la escala de estrellas, no anotaciones binarias originales.

---

## 6. Fase 4 — División del dataset

**Objetivo:** separar los datos para entrenar y evaluar modelos sin fuga de información.

### Propuesta inicial

| Conjunto | Porcentaje | Cantidad esperada |
|---|---:|---:|
| Entrenamiento | 80 % | 128.000 |
| Prueba | 20 % | 32.000 |
| Total | 100 % | 160.000 |

### Actividades

- [x] Separar textos (`X`) y etiquetas (`y`).
- [x] Aplicar `train_test_split`.
- [x] Usar división estratificada mediante `stratify=y`.
- [x] Definir y documentar una semilla fija (`random_state`).
- [x] Comprobar la distribución de clases en ambos conjuntos.
- [x] Guardar índices o identificadores de las particiones.
- [x] Mantener el conjunto de prueba apartado hasta la evaluación final.

**Regla contra fuga de información:** el vocabulario y los parámetros aprendidos de TF-IDF se ajustan solamente con entrenamiento. El conjunto de prueba no se utiliza para seleccionar técnicas, ajustar parámetros ni tomar decisiones.

---

## 7. Fase 5 — Pipeline de procesamiento NLP

**Objetivo:** implementar las técnicas NLP vistas en clase. Cada etapa debe tener código, ejemplos antes/después, pruebas y justificación.

### 7.1 Normalización

**Archivo:** `src/nlp/limpieza.py`

- [x] Convertir el texto a minúsculas.
- [x] Definir cómo tratar las tildes y caracteres especiales.
- [x] Definir cómo tratar los signos de puntuación.
- [x] Definir cómo tratar los números.
- [x] Identificar y tratar URLs.
- [x] Definir cómo tratar emojis.
- [x] Eliminar espacios innecesarios.
- [x] Evaluar palabras con caracteres repetidos.
- [x] Mostrar ejemplos antes y después.
- [x] Justificar qué se elimina y qué se conserva.

**Cuidado:** no eliminar indiscriminadamente expresiones relevantes para el sentimiento, como `no funciona`, `nunca llegó`, `muy bueno` o `mala calidad`.

### 7.2 Tokenización

**Archivo:** `src/nlp/tokenizacion.py`

- [x] Seleccionar una herramienta de tokenización compatible con español.
- [x] Separar reseñas en tokens.
- [x] Mostrar los tokens generados.
- [x] Probar textos con puntuación.
- [x] Probar palabras con tildes.
- [x] Probar negaciones.
- [x] Probar expresiones frecuentes del dominio.
- [x] Documentar la herramienta seleccionada.

Ejemplos ilustrativos:

| Texto | Tokens |
|---|---|
| El producto no funciona | `["el", "producto", "no", "funciona"]` |
| Excelente calidad y precio | `["excelente", "calidad", "y", "precio"]` |

### 7.3 Stop words

**Archivo:** `src/nlp/stopwords.py`

- [x] Obtener una lista de stop words en español.
- [x] Aplicar la eliminación de palabras poco informativas.
- [x] Revisar los resultados.
- [x] Evaluar palabras adicionales específicas del dominio.
- [x] Conservar negaciones relevantes.
- [x] Mostrar ejemplos antes y después.
- [x] Documentar la lista y las decisiones.

**Importante:** no eliminar automáticamente `no`, `nunca`, `jamás` o `sin`; pueden invertir el significado de una opinión.

Ejemplo:
- Antes: `el producto no funciona correctamente`
- Después ilustrativo: `producto no funciona correctamente`

### 7.4 Stemming

**Archivo:** `src/nlp/stemming.py`

- [x] Implementar stemming para español.
- [x] Seleccionar un algoritmo compatible, por ejemplo Snowball Stemmer.
- [x] Reducir palabras a sus raíces.
- [x] Mostrar ejemplos.
- [x] Evaluar cambios en el vocabulario.
- [x] Documentar la herramienta y sus limitaciones.

Una raíz generada por stemming no necesariamente es una palabra válida del idioma.

### 7.5 Lematización

**Archivo:** `src/nlp/lematizacion.py`

- [x] Seleccionar una herramienta compatible con español, por ejemplo spaCy con un modelo lingüístico español.
- [x] Instalar y documentar el modelo utilizado.
- [x] Obtener la forma base de las palabras.
- [x] Mostrar ejemplos.
- [x] Evaluar los resultados.
- [x] Documentar la herramienta y sus limitaciones.

Ejemplos ilustrativos: `productos → producto`, `comprando → comprar`. Los resultados reales dependen del modelo lingüístico.

### 7.6 Comparación entre stemming y lematización

- [x] Procesar una misma muestra con stemming.
- [x] Procesar la misma muestra con lematización.
- [x] Comparar ejemplos de salida.
- [x] Comparar vocabulario resultante.
- [x] Preparar ambas variantes para los experimentos de modelado.
- [x] Registrar el efecto de cada variante en las métricas cuando se integre con los modelos.

No elegir una técnica únicamente por cómo se ve el texto: se debe comparar su efecto en las métricas del modelo.

---

## 8. Fase 6 — N-gramas

**Objetivo:** comparar representaciones con palabras individuales y combinaciones consecutivas.

### 8.1 Unigramas

- [x] Configurar representación con unigramas (`ngram_range=(1,1)`).
- [x] Mostrar ejemplos de términos generados.
- [x] Registrar el tamaño del vocabulario.

Ejemplo: `producto excelente calidad` genera `producto`, `excelente`, `calidad`.

### 8.2 Unigramas y bigramas

- [x] Configurar representación con unigramas y bigramas (`ngram_range=(1,2)`).
- [x] Mostrar ejemplos de términos generados.
- [x] Registrar el tamaño del vocabulario.
- [x] Comprobar expresiones relevantes como `no funciona`, `muy bueno` y `mala calidad`.

### 8.3 Comparación

- [x] Comparar el tamaño de ambos vocabularios.
- [x] Documentar ventajas y limitaciones.
- [x] Preparar ambas configuraciones para los experimentos de Machine Learning.

| Característica | Unigramas | Unigramas + bigramas |
|---|---|---|
| Palabras individuales | Sí | Sí |
| Parejas de palabras | No | Sí |
| Tamaño del vocabulario | Generalmente menor | Generalmente mayor |
| Contexto entre palabras | Limitado | Mayor |

---

## 9. Fase 7 — Vectorización TF-IDF

**Objetivo:** convertir los textos en representaciones numéricas utilizables por los modelos.

**Archivo:** `src/nlp/vectorizacion.py`

### Actividades

- [x] Implementar `TfidfVectorizer` de Scikit-learn.
- [x] Crear configuración TF-IDF con unigramas.
- [x] Crear configuración TF-IDF con unigramas y bigramas.
- [x] Documentar `ngram_range`.
- [x] Definir y documentar `max_features`.
- [x] Definir y documentar `min_df`.
- [x] Definir y documentar `max_df`.
- [x] Evaluar `sublinear_tf`.
- [x] Registrar el tamaño del vocabulario.
- [x] Examinar términos con pesos TF-IDF relevantes.
- [x] Ajustar (`fit`) el vectorizador solo con entrenamiento.
- [x] Aplicar (`transform`) el vectorizador a prueba sin volver a ajustarlo.
- [x] Guardar el vectorizador y la configuración para reutilizarlos.

Los parámetros pueden ajustarse durante los experimentos, respetando siempre la separación entre entrenamiento y prueba.

---

## 10. Fase 8 — Integración del pipeline NLP

**Archivo:** `src/nlp/pipeline_nlp.py`

**Objetivo:** reunir las etapas en un flujo ordenado, reutilizable y consistente.

### Flujo lógico

```text
Reseña original
      |
      v
Normalización
      |
      v
Tokenización
      |
      v
Eliminación de stop words
      |
      +----------------------+
      |                      |
      v                      v
   Stemming             Lematización
      |                      |
      +----------+-----------+
                 |
                 v
       Configuración de n-gramas
                 |
                 v
              TF-IDF
                 |
                 v
       Representación numérica
                 |
                 v
       Entrada para Machine Learning
```

### Actividades

- [x] Crear una función que reciba una reseña.
- [x] Aplicar normalización.
- [x] Aplicar tokenización.
- [x] Aplicar stop words.
- [x] Obtener la variante con stemming.
- [x] Obtener la variante con lematización.
- [x] Permitir seleccionar la configuración de n-gramas.
- [x] Generar la representación TF-IDF.
- [x] Verificar que el flujo funcione de principio a fin.
- [x] Mantener disponibles las salidas intermedias para mostrarlas posteriormente en el frontend.
- [x] Documentar cómo se invoca el pipeline.

**Contrato para la integración:** el preprocesamiento usado durante entrenamiento debe ser exactamente el mismo que se utilizará al predecir. El backend no debe implementar una segunda versión independiente de la limpieza o vectorización.

---

## 11. Fase 9 — Pruebas y validaciones

**Objetivo:** comprobar que las funciones NLP trabajan correctamente.

**Archivo sugerido:** `tests/test_nlp.py`

### Casos de prueba

- [x] Texto vacío.
- [x] Texto con mayúsculas.
- [x] Texto con tildes.
- [x] Texto con números.
- [x] Texto con URLs.
- [x] Texto con emojis.
- [x] Texto con signos de puntuación.
- [x] Texto con negaciones.
- [x] Texto muy corto.
- [x] Texto largo.
- [x] Texto con palabras repetidas.
- [x] Texto con caracteres especiales.

### Caso de prueba ilustrativo

**Entrada:** `¡El producto NO funciona! Es una pésima compra 😡`

Comprobar que:
- [x] La normalización se aplique correctamente.
- [x] La tokenización separe las palabras.
- [x] La negación `NO` no desaparezca.
- [x] El emoji se trate según la decisión documentada.
- [x] Stemming y lematización generen sus respectivas salidas.
- [x] TF-IDF genere una representación válida.

---

## 12. Fase 10 — Documentación técnica

**Archivo:** `docs/modulo_1_datos_nlp.md`

Incluir:

- [x] Descripción del problema y alcance.
- [x] Descripción, fuente y licencia del dataset.
- [x] Fecha de obtención y versión.
- [x] Variables del dataset.
- [x] Análisis exploratorio.
- [x] Distribución original de etiquetas.
- [x] Transformación a clasificación binaria.
- [x] Tratamiento de neutrales.
- [x] División de entrenamiento y prueba.
- [x] Normalización.
- [x] Tokenización.
- [x] Stop words.
- [x] Stemming.
- [x] Lematización.
- [x] Comparación de stemming y lematización.
- [x] N-gramas.
- [x] TF-IDF.
- [x] Arquitectura del pipeline.
- [x] Ejemplos antes y después.
- [x] Resultados de pruebas.
- [x] Conclusiones y limitaciones.

---

## 13. Entregables finales del módulo

Al cerrar el módulo deben estar disponibles:

1. Dataset original en `data/raw/`.
2. Dataset procesado y etiquetas binarias en `data/processed/`.
3. Notebook de análisis exploratorio.
4. Código de normalización, tokenización y stop words.
5. Implementaciones de stemming y lematización.
6. Comparación de unigramas y unigramas + bigramas.
7. Vectorización TF-IDF.
8. Pipeline NLP integrado y reutilizable.
9. Pruebas automatizadas básicas.
10. Documentación técnica del módulo.
11. Datos y configuraciones preparados para el módulo de Machine Learning.

---

## 14. Criterios para considerar terminado el módulo

El módulo se considera terminado cuando se pueda ejecutar un flujo que:

1. Lea las reseñas originales.
2. Valide su estructura y calidad.
3. Transforme las etiquetas a positivo y negativo, excluyendo neutrales.
4. Divida los datos de forma estratificada y reproducible.
5. Procese los textos con las técnicas NLP requeridas.
6. Permita comparar stemming y lematización.
7. Permita comparar unigramas y unigramas + bigramas.
8. Genere representaciones TF-IDF sin fuga de información.
9. Guarde los resultados y configuraciones.
10. Entregue datos preparados para entrenar los modelos.

**Límite del módulo:** no le corresponde seleccionar ni entrenar definitivamente el mejor modelo. Eso pertenece al módulo de Machine Learning. Debe dejar preparadas las variantes necesarias para compararlas.

---

## 15. Relación con los demás módulos

| Módulo | Recibe | Entrega |
|---|---|---|
| 1. Datos y NLP | Dataset original | Datos procesados, funciones NLP y configuraciones TF-IDF |
| 2. Machine Learning | Datos procesados | Modelos entrenados, métricas y modelo seleccionado |
| 3. Backend | Modelo y pipeline entrenados | API REST funcional |
| 4. Frontend | API REST | Interfaz web para utilizar y visualizar resultados |

Los módulos pueden avanzar en paralelo, pero deben acordar sus formatos de entrada y salida antes de integrar.

---

## 16. Comandos y notas de trabajo

Desde la raíz del repositorio, activar el entorno virtual antes de instalar dependencias o ejecutar notebooks/scripts.

Ejemplo para Windows PowerShell:

```powershell
# Crear entorno virtual si todavía no existe
python -m venv .venv

# Activarlo
.\.venv\Scripts\Activate.ps1

# Instalar dependencias cuando requirements.txt esté preparado
pip install -r requirements.txt
```

Ejemplo de lectura del JSONL:

```python
import json

with open("data/raw/train.jsonl", "r", encoding="utf-8") as archivo:
    for i, linea in enumerate(archivo):
        registro = json.loads(linea)
        print(registro["id"], registro["text"], registro["label"])
        if i == 4:
            break
```

### Decisiones que deben quedar registradas

- Semilla aleatoria utilizada.
- Porcentaje de entrenamiento y prueba.
- Tratamiento de emojis, números, URLs y puntuación.
- Lista de stop words y negaciones conservadas.
- Herramientas y versiones para stemming y lematización.
- Configuración de unigramas y bigramas.
- Parámetros TF-IDF.
- Versiones de dependencias.

---

## 17. Registro de avances

| Fecha | Actividad / decisión | Resultado o pendiente |
|---|---|---|
| 31/10/2026 | Fase 1: Verificación de dataset crudo y codificación UTF-8 | Completado. 200.000 registros validados con scripts/probar_dataset.py. |
| 31/10/2026 | Fase 2: EDA en notebook (01_analisis_exploratorio.ipynb) | Completado. Calidad perfecta, longitud, palabras clave y gráficos generados. |
| 31/10/2026 | Fase 3: Preparación de etiquetas binarias (exclusión de neutrales) | Completado. 160.000 registros (50% positivo, 50% negativo) en resenas_binarias.csv. |
| 31/10/2026 | Fase 4: Partición estratificada (80% train / 20% test, seed=42) | Completado. train_binario.csv (128k) y test_binario.csv (32k) sin fuga de datos. |
| 31/10/2026 | Fase 5: Pipeline modular NLP en src/nlp/ | Completado. Limpieza con 'ñ' protegida, negaciones preservadas, stemming y lematización. |
| 31/10/2026 | Fase 6 y 7: Extracción de características TF-IDF (unigramas y bigramas) | Completado. Matrices .npz y pipelines .joblib exportados en data/features/. |
| 31/10/2026 | Fase 8: Pipeline integrado con inspección de pasos intermedios | Completado. Clase PipelineNLP lista para Backend (FastAPI) e inferencia. |
| 31/10/2026 | Fase 9: Pruebas unitarias con pytest (tests/test_nlp.py) | Completado. 14/14 tests pasando (100% cobertura de requisitos). |
| 31/10/2026 | Fase 10: Documentación técnica detallada (docs/modulo_1_datos_nlp.md) | Completado. Fichas técnicas, tablas comparativas y contrato de integración listos. |

**Nota final:** marcar las tareas a medida que se completen. No marcar una actividad como terminada únicamente porque existe un archivo: debe estar implementada, probada y documentada.
