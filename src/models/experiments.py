"""Construcción, entrenamiento y predicción para los modelos candidatos de ML."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .config import MLConfig
from .split import DatasetSplit


@dataclass(frozen=True)
class ExperimentSpec:
    """Describe un experimento sin ejecutar ni asumir un estimador concreto."""

    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentResult:
    """Estructura para métricas reales que se obtendrán en la fase futura."""

    name: str
    metrics: Mapping[str, float]


def run_experiments(
    specs: Sequence[ExperimentSpec], config: MLConfig
) -> Sequence[ExperimentResult]:
    """Ejecutará y comparará experimentos definidos explícitamente."""

    del specs, config
    raise NotImplementedError("La experimentación se implementará en una fase posterior.")


def build_tfidf_vectorizer(
    *,
    ngram_range: tuple[int, int] = (1, 1),
    lowercase: bool = True,
    **kwargs: Any,
) -> TfidfVectorizer:
    """Construye un vectorizador TF-IDF provisional y configurable.

    Por defecto utiliza unigramas y conversión a minúsculas. Permite
    sobrescribir o agregar parámetros para futura compatibilidad con el
    módulo de NLP.

    Args:
        ngram_range: Rango de n-gramas a extraer (por defecto unigramas: (1, 1)).
        lowercase: Si se convierten los textos a minúsculas antes de tokenizar.
        **kwargs: Opciones adicionales admitidas por TfidfVectorizer.

    Returns:
        Instancia de TfidfVectorizer lista para ser incluida en un Pipeline.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        lowercase=lowercase,
        **kwargs,
    )


def build_logistic_regression_pipeline(
    *,
    random_state: int | None = 42,
    max_iter: int = 1000,
    vectorizer: Any | None = None,
    config: MLConfig | None = None,
    **classifier_kwargs: Any,
) -> Pipeline:
    """Construye un Pipeline de scikit-learn con TF-IDF y Regresión Logística.

    Cumple el contrato de la fase:
    - Estimador: LogisticRegression con random_state=42 y max_iter=1000.
    - Transformador: TfidfVectorizer básico (unigramas), configurable o reemplazable.

    Args:
        random_state: Semilla para reproducibilidad. Si se pasa `config`, se
            prioriza `config.experiment.random_seed`. Por defecto 42.
        max_iter: Límite de iteraciones del solver para asegurar convergencia (por defecto 1000).
        vectorizer: Transformador opcional compatible con scikit-learn. Si es None,
            se usa `build_tfidf_vectorizer()`.
        config: Configuración `MLConfig` opcional para resolver parámetros por defecto.
        **classifier_kwargs: Parámetros adicionales para LogisticRegression.

    Returns:
        Pipeline con pasos 'vectorizer' y 'classifier'.
    """
    seed = (
        config.experiment.random_seed
        if config is not None
        else (random_state if random_state is not None else 42)
    )

    tfidf_step = vectorizer if vectorizer is not None else build_tfidf_vectorizer()
    classifier_step = LogisticRegression(
        random_state=seed,
        max_iter=max_iter,
        **classifier_kwargs,
    )

    return Pipeline(
        steps=[
            ("vectorizer", tfidf_step),
            ("classifier", classifier_step),
        ]
    )


def build_multinomial_nb_pipeline(
    *,
    vectorizer: Any | None = None,
    **classifier_kwargs: Any,
) -> Pipeline:
    """Construye un Pipeline de scikit-learn con TF-IDF y Multinomial Naive Bayes.

    Cumple el contrato de la fase:
    - Estimador: MultinomialNB determinista.
    - Transformador: TfidfVectorizer básico (unigramas), configurable o reemplazable.

    Args:
        vectorizer: Transformador opcional compatible con scikit-learn. Si es None,
            se usa `build_tfidf_vectorizer()`.
        **classifier_kwargs: Parámetros adicionales para MultinomialNB (ej. alpha, fit_prior).

    Returns:
        Pipeline con pasos 'vectorizer' y 'classifier'.
    """
    tfidf_step = vectorizer if vectorizer is not None else build_tfidf_vectorizer()
    classifier_step = MultinomialNB(**classifier_kwargs)

    return Pipeline(
        steps=[
            ("vectorizer", tfidf_step),
            ("classifier", classifier_step),
        ]
    )


def get_candidate_pipelines(
    *,
    random_seed: int = 42,
    max_iter: int = 1000,
    vectorizer: Any | None = None,
    config: MLConfig | None = None,
) -> dict[str, Pipeline]:
    """Retorna un mapeo con instancias nuevas de ambos modelos candidatos.

    Returns:
        Diccionario con claves 'logistic_regression' y 'multinomial_nb'.
    """
    return {
        "logistic_regression": build_logistic_regression_pipeline(
            random_state=random_seed,
            max_iter=max_iter,
            vectorizer=vectorizer,
            config=config,
        ),
        "multinomial_nb": build_multinomial_nb_pipeline(
            vectorizer=vectorizer,
        ),
    }


def train_pipeline(
    pipeline: Pipeline,
    X_train: Sequence[str] | pd.Series | DatasetSplit,
    y_train: Sequence[str] | pd.Series | None = None,
) -> Pipeline:
    """Entrena el pipeline exclusivamente con los datos de entrenamiento.

    Ajusta el vectorizador y el clasificador únicamente sobre X_train e y_train,
    garantizando la ausencia total de fuga de información (data leakage).

    Args:
        pipeline: Instancia de Pipeline de scikit-learn.
        X_train: Datos de entrenamiento (pd.Series o secuencia de textos)
            o un objeto `DatasetSplit` retornado por `split_dataset`.
        y_train: Etiquetas de entrenamiento. Requerido si X_train no es `DatasetSplit`.

    Returns:
        El pipeline entrenado (fitted).

    Raises:
        TypeError: Si `pipeline` no es un `Pipeline` de scikit-learn.
        ValueError: Si los datos están vacíos, no coinciden en longitud o hay parámetros contradictorios.
    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError(
            f"Se esperaba un objeto sklearn.pipeline.Pipeline, pero se recibió: {type(pipeline).__name__}."
        )

    if isinstance(X_train, DatasetSplit):
        if y_train is not None:
            raise ValueError(
                "No se debe proporcionar 'y_train' cuando 'X_train' es un objeto DatasetSplit."
            )
        train_x = X_train.X_train
        train_y = X_train.y_train
    else:
        if y_train is None:
            raise ValueError(
                "Se requiere 'y_train' cuando 'X_train' es una secuencia o Serie."
            )
        train_x = X_train
        train_y = y_train

    if len(train_x) == 0:
        raise ValueError("X_train no puede estar vacío.")
    if len(train_y) == 0:
        raise ValueError("y_train no puede estar vacío.")
    if len(train_x) != len(train_y):
        raise ValueError(
            f"La cantidad de textos en X_train ({len(train_x)}) no coincide con las etiquetas y_train ({len(train_y)})."
        )

    pipeline.fit(train_x, train_y)
    return pipeline


def _normalize_texts_input(texts: str | Sequence[str] | pd.Series) -> list[str]:
    """Valida y normaliza la entrada de textos a una lista de cadenas no vacía."""
    if isinstance(texts, str):
        normalized = [texts]
    elif isinstance(texts, pd.Series):
        normalized = texts.tolist()
    elif isinstance(texts, Sequence):
        normalized = list(texts)
    else:
        raise TypeError(
            f"Se esperaba str, Sequence[str] o pd.Series, pero se recibió: {type(texts).__name__}."
        )

    if len(normalized) == 0:
        raise ValueError("La colección de textos a evaluar no puede estar vacía.")

    return normalized


def predict(
    pipeline: Pipeline,
    texts: str | Sequence[str] | pd.Series,
) -> np.ndarray:
    """Obtiene las predicciones de clase para uno o varios textos.

    Args:
        pipeline: Pipeline entrenado.
        texts: Texto individual o colección/Serie de textos a predecir.

    Returns:
        np.ndarray unidimensional con las etiquetas predichas.

    Raises:
        TypeError: Si el pipeline o la entrada no corresponden a los tipos esperados.
        ValueError: Si la colección de textos está vacía.
    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError(
            f"Se esperaba un objeto sklearn.pipeline.Pipeline, pero se recibió: {type(pipeline).__name__}."
        )

    input_texts = _normalize_texts_input(texts)
    return pipeline.predict(input_texts)


def predict_proba(
    pipeline: Pipeline,
    texts: str | Sequence[str] | pd.Series,
) -> np.ndarray:
    """Obtiene la matriz de probabilidades brutas predichas para cada texto.

    Args:
        pipeline: Pipeline entrenado.
        texts: Texto individual o colección/Serie de textos a predecir.

    Returns:
        np.ndarray de tamaño (n_muestras, n_clases) con las probabilidades numéricas.

    Raises:
        TypeError: Si el pipeline o la entrada no corresponden a los tipos esperados.
        ValueError: Si la colección de textos está vacía.
    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError(
            f"Se esperaba un objeto sklearn.pipeline.Pipeline, pero se recibió: {type(pipeline).__name__}."
        )

    input_texts = _normalize_texts_input(texts)
    return pipeline.predict_proba(input_texts)


def predict_probabilities(
    pipeline: Pipeline,
    texts: str | Sequence[str] | pd.Series,
) -> list[dict[str, float]]:
    """Obtiene las probabilidades estructuradas como diccionario {clase: probabilidad}.

    Garantiza el cumplimiento estricto del contrato: asocia las probabilidades
    utilizando el orden real de ``classes_`` del estimador, sin asumir posiciones fijas.

    Args:
        pipeline: Pipeline entrenado.
        texts: Texto individual o colección/Serie de textos a predecir.

    Returns:
        Lista de diccionarios donde cada elemento mapea el nombre de la clase a su probabilidad.

    Raises:
        TypeError: Si el pipeline o la entrada no corresponden a los tipos esperados.
        ValueError: Si la colección de textos está vacía o el pipeline no expone 'classes_'.
    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError(
            f"Se esperaba un objeto sklearn.pipeline.Pipeline, pero se recibió: {type(pipeline).__name__}."
        )

    input_texts = _normalize_texts_input(texts)

    classes = getattr(pipeline, "classes_", None)
    if classes is None:
        classifier = pipeline.named_steps.get("classifier", pipeline.steps[-1][1])
        classes = getattr(classifier, "classes_", None)

    if classes is None:
        raise ValueError(
            "El pipeline no ha sido entrenado o no expone el atributo contractual 'classes_'."
        )

    proba_matrix = pipeline.predict_proba(input_texts)
    return [
        {str(cls): float(prob) for cls, prob in zip(classes, row)}
        for row in proba_matrix
    ]


# Alias convenientes para interoperabilidad de nombres
predict_pipeline = predict
predict_proba_pipeline = predict_proba
