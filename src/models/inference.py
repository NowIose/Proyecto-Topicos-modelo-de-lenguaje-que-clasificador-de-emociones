"""Inferencia reproducible y estructurada para modelos de Machine Learning entrenados.

Este módulo expone la interfaz de inferencia consumible por futuros servicios
backend (ej. FastAPI por Arteaga). Permite cargar pipelines previamente guardados
con joblib, validar su compatibilidad y predecir el sentimiento de textos nuevos
junto con sus probabilidades de clase estructuradas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
from sklearn.pipeline import Pipeline

from .config import PROJECT_ROOT, MLConfig, load_config


@dataclass(frozen=True)
class Prediction:
    """Resultado estructurado de la inferencia sobre un texto.

    Attributes:
        label: Etiqueta de clase predicha (ej. 'positivo' o 'negativo').
        probabilities: Diccionario asociando cada clase a su probabilidad [0.0, 1.0].
    """

    label: str
    probabilities: Mapping[str, float]

    @property
    def confidence(self) -> float:
        """Devuelve la probabilidad numérica asignada a la clase predicha."""
        return float(self.probabilities.get(self.label, 0.0))

    def to_dict(self) -> dict[str, Any]:
        """Convierte la predicción a un diccionario listo para serializar a JSON / Pydantic."""
        return {
            "label": self.label,
            "confidence": self.confidence,
            "probabilities": dict(self.probabilities),
        }


def get_pipeline_classes(pipeline: Any) -> list[str]:
    """Obtiene el orden real de las clases desde el atributo 'classes_' del pipeline.

    Evita asumir posiciones fijas (ej. que 0 sea negativo y 1 positivo).
    Inspecciona el pipeline y su estimador final ('classifier').

    Args:
        pipeline: Pipeline de scikit-learn entrenado.

    Returns:
        Lista ordenada de nombres de clases como cadenas.

    Raises:
        ValueError: Si el pipeline no ha sido entrenado o carece del atributo 'classes_'.
    """
    classes = getattr(pipeline, "classes_", None)
    if classes is None:
        classifier = getattr(pipeline, "named_steps", {}).get("classifier", None)
        if classifier is None and hasattr(pipeline, "steps") and len(pipeline.steps) > 0:
            classifier = pipeline.steps[-1][1]
        classes = getattr(classifier, "classes_", None)

    if classes is None:
        raise ValueError(
            "El pipeline proporcionado no expone el atributo 'classes_'. "
            "Asegúrese de que el pipeline haya sido ajustado (fitted) antes de la inferencia."
        )

    return [str(cls) for cls in classes]


def validate_pipeline(pipeline: Any) -> Pipeline:
    """Verifica que un objeto sea un pipeline compatible y entrenado.

    Args:
        pipeline: Objeto a validar.

    Returns:
        El objeto validado como Pipeline.

    Raises:
        TypeError: Si el objeto no es un Pipeline o carece de métodos 'predict' y 'predict_proba'.
        ValueError: Si el pipeline no está entrenado (sin 'classes_').
    """
    if pipeline is None:
        raise TypeError("El pipeline no puede ser None.")

    has_predict = callable(getattr(pipeline, "predict", None))
    has_predict_proba = callable(getattr(pipeline, "predict_proba", None))

    if not (isinstance(pipeline, Pipeline) or (has_predict and has_predict_proba)):
        raise TypeError(
            f"Se esperaba un Pipeline compatible de scikit-learn con métodos 'predict' y 'predict_proba', "
            f"pero se recibió: {type(pipeline).__name__}."
        )

    # Validar que esté ajustado y exponga classes_
    get_pipeline_classes(pipeline)

    return pipeline


def load_pipeline(
    path: str | Path | None = None,
    *,
    config: MLConfig | None = None,
) -> Pipeline:
    """Carga con joblib un pipeline previamente guardado y valida su compatibilidad.

    Si no se especifica ruta, utiliza la ruta configurada por defecto en config/ml.yaml.

    Args:
        path: Ruta al archivo .joblib del pipeline. Si es None, toma la ruta de configuración.
        config: Configuración MLConfig opcional para resolver la ruta por defecto.

    Returns:
        Instancia de Pipeline validada y lista para inferencia.

    Raises:
        FileNotFoundError: Si el archivo no existe en el sistema de archivos.
        ValueError: Si la ruta está vacía o el objeto cargado no está entrenado.
        TypeError: Si la ruta no es válida o el objeto cargado no es un pipeline compatible.
    """
    if path is None:
        active_config = config or load_config()
        resolved_path = active_config.artifacts.pipeline_path
    elif isinstance(path, (str, Path)):
        path_str = str(path).strip()
        if not path_str:
            raise ValueError("La ruta del pipeline no puede estar vacía.")
        candidate_path = Path(path)
        resolved_path = candidate_path if candidate_path.is_absolute() else PROJECT_ROOT / candidate_path
    else:
        raise TypeError(
            f"La ruta debe ser str, Path o None, pero se recibió: {type(path).__name__}."
        )

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"No se encontró el pipeline de Machine Learning en la ruta: '{resolved_path}'. "
            f"Asegúrese de que el modelo haya sido entrenado y guardado previamente."
        )

    if not resolved_path.is_file():
        raise ValueError(
            f"La ruta especificada existe pero no es un archivo regular: '{resolved_path}'."
        )

    try:
        loaded = joblib.load(resolved_path)
    except Exception as exc:
        raise ValueError(
            f"Error al deserializar el archivo con joblib desde '{resolved_path}': {exc}"
        ) from exc

    return validate_pipeline(loaded)


def _validate_text_input(text: Any) -> str:
    """Valida que la entrada sea una cadena de texto no vacía."""
    if text is None:
        raise TypeError("El texto de entrada no puede ser None.")
    if not isinstance(text, str):
        raise TypeError(
            f"El texto de entrada debe ser de tipo str, pero se recibió: {type(text).__name__}."
        )
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("El texto de entrada no puede estar vacío ni contener solo espacios en blanco.")
    return cleaned


def predict(pipeline: Any, text: str) -> Prediction:
    """Predice el sentimiento y las probabilidades para una reseña individual.

    El pipeline ejecuta internamente el preprocesamiento, vectorización y clasificación
    sin ajustar nada sobre el texto nuevo (cero data leakage).

    Args:
        pipeline: Pipeline de scikit-learn entrenado y compatible.
        text: Texto de la reseña a clasificar (cadena no vacía).

    Returns:
        Instancia de Prediction con la etiqueta predicha y el mapeo de probabilidades.

    Raises:
        TypeError: Si el pipeline o el texto no son del tipo esperado.
        ValueError: Si el texto está vacío o el pipeline no expone 'classes_'.
    """
    valid_pipe = validate_pipeline(pipeline)
    cleaned = _validate_text_input(text)

    classes = get_pipeline_classes(valid_pipe)

    # Inferencia directa sin ajuste ni modificación del vectorizador
    pred_label = str(valid_pipe.predict([cleaned])[0])
    proba_matrix = valid_pipe.predict_proba([cleaned])
    proba_row = proba_matrix[0]

    probabilities: dict[str, float] = {
        cls: float(prob) for cls, prob in zip(classes, proba_row)
    }

    return Prediction(
        label=pred_label,
        probabilities=probabilities,
    )


def predict_batch(pipeline: Any, texts: Sequence[str]) -> list[Prediction]:
    """Predice el sentimiento y las probabilidades para una colección de reseñas.

    Ejecuta la inferencia de manera vectorizada y eficiente sobre el batch completo.

    Args:
        pipeline: Pipeline de scikit-learn entrenado y compatible.
        texts: Secuencia de textos a clasificar (ninguno vacío).

    Returns:
        Lista de instancias de Prediction correspondientes a cada texto.

    Raises:
        TypeError: Si el pipeline o la secuencia de textos son inválidos.
        ValueError: Si la secuencia está vacía o contiene textos vacíos.
    """
    valid_pipe = validate_pipeline(pipeline)

    if isinstance(texts, str):
        raise TypeError(
            "Se esperaba una secuencia de textos (list o tuple), pero se recibió un str. "
            "Para un único texto utilice 'predict(pipeline, text)'."
        )
    if not isinstance(texts, Sequence):
        raise TypeError(
            f"Se esperaba una Sequence[str], pero se recibió: {type(texts).__name__}."
        )

    if len(texts) == 0:
        raise ValueError("La colección de textos a clasificar no puede estar vacía.")

    cleaned_texts = [_validate_text_input(t) for t in texts]
    classes = get_pipeline_classes(valid_pipe)

    pred_labels = valid_pipe.predict(cleaned_texts)
    proba_matrix = valid_pipe.predict_proba(cleaned_texts)

    results: list[Prediction] = []
    for label, row in zip(pred_labels, proba_matrix):
        probs = {cls: float(prob) for cls, prob in zip(classes, row)}
        results.append(Prediction(label=str(label), probabilities=probs))

    return results


class SentimentClassifier:
    """Servicio de inferencia de sentimientos de alto nivel para integración con FastAPI.

    Permite instanciar una sola vez el clasificador (ej. en el ciclo de vida o estado
    de FastAPI) y reutilizarlo de forma concurrente para predecir textos entrantes.

    Example:
        >>> classifier = SentimentClassifier("artifacts/sentiment_pipeline.joblib")
        >>> result = classifier.predict("Excelente servicio, muy rápido.")
        >>> print(result.label)
        'positivo'
        >>> print(result.confidence)
        0.95
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        pipeline: Any | None = None,
        config: MLConfig | None = None,
    ) -> None:
        if pipeline is not None:
            self._pipeline = validate_pipeline(pipeline)
            self._model_path = Path(model_path) if model_path is not None else None
        else:
            self._pipeline = load_pipeline(path=model_path, config=config)
            self._model_path = Path(model_path) if model_path is not None else None

        self._classes = get_pipeline_classes(self._pipeline)

    @property
    def pipeline(self) -> Pipeline:
        """Pipeline subyacente cargado."""
        return self._pipeline

    @property
    def classes(self) -> list[str]:
        """Lista de clases soportadas por el modelo."""
        return list(self._classes)

    @property
    def model_path(self) -> Path | None:
        """Ruta al archivo de modelo utilizado (si se cargó desde disco)."""
        return self._model_path

    def predict(self, text: str) -> Prediction:
        """Predice el sentimiento de un texto individual."""
        return predict(self._pipeline, text)

    def predict_batch(self, texts: Sequence[str]) -> list[Prediction]:
        """Predice el sentimiento de un conjunto de textos en batch."""
        return predict_batch(self._pipeline, texts)
