# Contratos e integración del módulo Machine Learning

Este documento delimita la preparación de Antony. Aún no existen un dataset
definitivo, experimentos ejecutados ni un modelo serializado.

## Contrato de datos y NLP (integración con José)

El entrenamiento recibirá una tabla con exactamente estas columnas mínimas:

| Campo | Tipo esperado | Restricción |
| --- | --- | --- |
| `texto` | texto no vacío | reseña original en español |
| `sentimiento` | texto | solo `positivo` o `negativo` |

Las filas inválidas, los duplicados y la política sobre valores ausentes deben
ser resueltos de común acuerdo cuando esté disponible el dataset real. La ruta
configurada (`data/reviews.csv`) es solo la ubicación prevista.

El módulo ML importará el preprocesamiento oficial desde `src.nlp`; no mantendrá
una segunda implementación. El contrato recomendado es que José exponga un
transformador compatible con scikit-learn (`fit`/`transform`) o una fábrica que
lo construya. Ese transformador será el primer paso del futuro pipeline.

El preprocesamiento, la futura representación numérica y el estimador se
guardarán juntos como un único pipeline con joblib. Predicción cargará ese
objeto y recibirá el texto original: no volverá a limpiar el texto manualmente.
Así se evita cualquier diferencia entre entrenamiento e inferencia.

## Contrato del artefacto (integración con Arteaga)

La salida prevista es `artifacts/sentiment_pipeline.joblib`; este archivo no se
crea en esta fase. El backend podrá cargarlo posteriormente con joblib y deberá:

1. enviar al pipeline una colección de textos originales;
2. obtener `positivo` o `negativo` mediante `predict`;
3. obtener probabilidades mediante `predict_proba` y asociarlas usando el orden
   real de `classes_`, nunca mediante posiciones asumidas;
4. devolver probabilidades con las claves `positivo` y `negativo`.

Para que ese contrato sea posible, el estimador que se elija más adelante debe
ofrecer probabilidades. La versión de Python y las dependencias usadas para
servir el artefacto deben ser compatibles con las usadas al serializarlo.

## Docker y flujo de trabajo

El servicio `ml` de `compose.ml.yaml` es deliberadamente una tarea finita, no un
servidor. Actualmente ejecuta las pruebas. Desde PowerShell, en la raíz:

```powershell
docker compose -f compose.ml.yaml build ml
docker compose -f compose.ml.yaml run --rm ml
docker compose -f compose.ml.yaml run --rm ml python -m pytest -q
```

Los montajes `data/` y `artifacts/` hacen que las entradas y los futuros modelos
permanezcan en el repositorio de Windows aunque el contenedor termine. El código
y la configuración se montan en modo de solo lectura.

Cuando `train.py` esté implementado, el comando previsto será:

```powershell
docker compose -f compose.ml.yaml run --rm ml python -m src.models.train
```

Hasta entonces no debe presentarse ese comando como un entrenamiento exitoso.

## Incorporación futura al Compose general

No se modifica el Compose de Arteaga. Cuando exista entrenamiento real, el
equipo podrá trasladar esta definición al archivo general como un servicio de
inicialización, por ejemplo `ml-train`, con `restart: "no"` y el comando de
entrenamiento. El backend podrá declarar una dependencia con la condición
`service_completed_successfully`; así `docker compose up --build` entrenará (o
validará) el artefacto antes de iniciar la API. Como alternativa, el equipo puede
exigir un artefacto previamente generado y ejecutar el entrenamiento solo bajo
demanda. Esa decisión se tomará al integrar, no en este esqueleto.
