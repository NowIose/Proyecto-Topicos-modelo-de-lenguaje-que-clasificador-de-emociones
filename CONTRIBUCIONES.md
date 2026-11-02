# Contribuciones

Cada sección resume lo que hizo cada integrante y coincide con su historial de commits
(`git log --author=<correo>`). Los merges de pull requests no se cuentan como aporte.

## José — Datos y NLP (Módulo 1)

GitHub: `NowIose` · correo de commits: `dainerdual@gmail.com`

- Estructura inicial del repositorio y descarga del dataset `amazon_reviews_multi_es`
  con prueba de codificación UTF-8 (`965800d`).
- Análisis exploratorio en notebook y script que lo genera (`4608dde`).
- Mapeo a etiquetas binarias excluyendo las reseñas neutras y división estratificada
  80/20 (`4676d9e`).
- Pipeline de NLP: limpieza con soporte de ñ, tokenización, stop words que conservan
  negaciones, stemming y lematización (`77737cd`).
- `PipelineNLP` integrado y extracción de matrices TF-IDF de unigramas y bigramas
  (`1ad7f92`).
- Documentación técnica del Módulo 1 y plan del Módulo 2 (`afcde9e`, `3a5aaf6`).

## Gabriel — Modelado (Módulo 2)

GitHub: `GaboTex2004` · correo de commits: `gabrielchoque222@gmail.com`

- Entorno Docker de ML, configuración `config/ml.yaml` y validación y partición del
  dataset (`8a82937`).
- Pipelines de Regresión Logística y Naive Bayes Multinomial (`e13e907`).
- Entrenamiento reproducible de los modelos candidatos (`0f1df49`).
- Métricas de evaluación y análisis de errores (`7f2f5da`).
- Validación cruzada estratificada y `GridSearchCV` (`681d486`).
- Comparación de unigramas contra bigramas con TF-IDF (`efdf607`).
- Comparación de stemming contra lematización (`c164fbf`).
- Inferencia y carga de modelos: `SentimentClassifier` (`fe8e485`).

## Santiago Arteaga — Integración, backend y frontend (Módulos 3 y 4)

GitHub: `santiagoarteaga0704` · correo de commits: `santiagoarteaga0704@gmail.com`

- Revisión e integración a `main` de los pull requests #5 a #11 del Módulo 2.
- **NLP y ML:** `PreprocesadorNLP`, que mete el `PipelineNLP` del Módulo 1 dentro del
  pipeline de scikit-learn para que el modelo guardado incluya el preprocesamiento
  (`ce77a9e`); configuración apuntando a los datos reales con sección `nlp` (`cbedef3`);
  `python -m src.models.entrenar`, el entrenamiento de punta a punta con línea base,
  comparaciones, `GridSearchCV`, evaluación en prueba y guardado (`c210cd6`); modelo
  entrenado con el dataset completo (`027debe`).
- **Backend:** API REST con FastAPI (predicción simple, por lote y por archivo,
  preprocesamiento paso a paso, métricas y estado), validaciones y errores en español,
  pruebas con `TestClient` y versiones fijadas en `requirements.txt`.
- **Frontend:** aplicación React + Vite con análisis de una reseña, reseña marcada, paso a
  paso del NLP, análisis por archivo y panel de métricas.
- **Entrega:** README, informe técnico, Docker y este archivo.

Los commits de Santiago se hicieron con asistencia de Claude Code y lo declaran con la
línea `Co-Authored-By: Claude` (ver sección 8 del informe técnico).
