"""Construcción, entrenamiento, predicción y validación cruzada para modelos ML."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
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
    """Estructura para métricas reales obtenidas en los experimentos."""

    name: str
    metrics: Mapping[str, float]
    fold_metrics: Mapping[str, Sequence[float]] = field(default_factory=dict)


def run_experiments(
    specs: Sequence[ExperimentSpec], config: MLConfig
) -> Sequence[ExperimentResult]:
    """Ejecutará y comparará experimentos definidos explícitamente."""

    del specs, config
    raise NotImplementedError("La experimentación se implementará en una fase posterior.")


class TextPreprocessorAdapter(BaseEstimator, TransformerMixin):
    """Adaptador compatible con scikit-learn para preprocesamiento de texto.

    Permite integrar componentes de NLP externos (como stemming o lematización
    provistos por `src.nlp`) dentro de un Pipeline de scikit-learn de manera
    transparente, serializable y sin fuga de datos.

    Acepta:
    - Una función callable: `func(text: str) -> str` (o `func(texts) -> texts` si batch=True).
    - Un transformador/estimador con método `transform` (y opcionalmente `fit`).
    - None (actúa como passthrough/identidad).

    Args:
        func: Objeto callable o transformador subyacente.
        name: Identificador opcional del transformador (ej. 'stemming', 'lemmatization').
        batch: Si es True, pasa la colección completa de textos al callable de una sola vez.
    """

    def __init__(
        self,
        func: Any = None,
        name: str | None = None,
        batch: bool = False,
    ) -> None:
        self.func = func
        self.name = name
        self.batch = batch

    def fit(self, X: Any, y: Any = None) -> TextPreprocessorAdapter:
        """Ajusta el preprocesador si el componente subyacente soporta fit."""
        if self.func is not None and hasattr(self.func, "fit"):
            self.func.fit(X, y)
        return self

    def transform(self, X: Any) -> list[str]:
        """Aplica la transformación de texto sobre la entrada.

        Args:
            X: Colección de textos (str, list, pd.Series o np.ndarray).

        Returns:
            Lista de cadenas de texto transformadas.
        """
        if isinstance(X, str):
            texts = [X]
        elif isinstance(X, pd.Series):
            texts = X.tolist()
        elif isinstance(X, np.ndarray):
            texts = X.tolist()
        elif isinstance(X, Sequence):
            texts = list(X)
        else:
            texts = list(X)

        if self.func is None:
            return [str(t) for t in texts]

        if hasattr(self.func, "transform"):
            transformed = self.func.transform(texts)
            if isinstance(transformed, (pd.Series, np.ndarray)):
                return [str(t) for t in transformed.tolist()]
            return [str(t) for t in transformed]

        if callable(self.func):
            if self.batch:
                transformed = self.func(texts)
                return [str(t) for t in transformed]
            else:
                return [str(self.func(text)) for text in texts]

        raise TypeError(
            f"El componente func debe ser un estimador con 'transform' o un callable, "
            f"pero se recibió: {type(self.func).__name__}."
        )


def _resolve_preprocessor(preprocessor: Any | None) -> Any | None:
    """Resuelve y normaliza un preprocesador a un transformador compatible con scikit-learn.

    Args:
        preprocessor: Estimador con fit/transform, callable o None.

    Returns:
        Transformador compatible con scikit-learn o None.
    """
    if preprocessor is None:
        return None
    if hasattr(preprocessor, "fit") and hasattr(preprocessor, "transform"):
        return preprocessor
    if callable(preprocessor):
        return TextPreprocessorAdapter(func=preprocessor)
    raise TypeError(
        f"El preprocesador debe ser un estimador de scikit-learn (con fit y transform) "
        f"o un callable, pero se recibió: {type(preprocessor).__name__}."
    )


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
    ngram_range: tuple[int, int] = (1, 1),
    vectorizer: Any | None = None,
    preprocessor: Any | None = None,
    config: MLConfig | None = None,
    **classifier_kwargs: Any,
) -> Pipeline:
    """Construye un Pipeline de scikit-learn con TF-IDF y Regresión Logística.

    Cumple el contrato de la fase:
    - Estimador: LogisticRegression con random_state=42 y max_iter=1000.
    - Transformador: TfidfVectorizer configurable o reemplazable.
    - Preprocesador: Opcional componente o adaptador NLP (ej. stemming o lematización).

    Args:
        random_state: Semilla para reproducibilidad. Si se pasa `config`, se
            prioriza `config.experiment.random_seed`. Por defecto 42.
        max_iter: Límite de iteraciones del solver para asegurar convergencia (por defecto 1000).
        ngram_range: Rango de n-gramas a utilizar si no se pasa `vectorizer` (por defecto unigramas: (1, 1)).
        vectorizer: Transformador opcional compatible con scikit-learn. Si es None,
            se usa `build_tfidf_vectorizer(ngram_range=ngram_range)`.
        preprocessor: Transformador opcional o callable de NLP a ejecutar antes de vectorizar.
        config: Configuración `MLConfig` opcional para resolver parámetros por defecto.
        **classifier_kwargs: Parámetros adicionales para LogisticRegression.

    Returns:
        Pipeline con pasos 'vectorizer' y 'classifier' (y opcionalmente 'preprocessor').
    """
    seed = (
        config.experiment.random_seed
        if config is not None
        else (random_state if random_state is not None else 42)
    )

    tfidf_step = (
        vectorizer
        if vectorizer is not None
        else build_tfidf_vectorizer(ngram_range=ngram_range)
    )
    classifier_step = LogisticRegression(
        random_state=seed,
        max_iter=max_iter,
        **classifier_kwargs,
    )

    steps: list[tuple[str, Any]] = []
    prep_step = _resolve_preprocessor(preprocessor)
    if prep_step is not None:
        steps.append(("preprocessor", prep_step))
    steps.append(("vectorizer", tfidf_step))
    steps.append(("classifier", classifier_step))

    return Pipeline(steps=steps)


def build_multinomial_nb_pipeline(
    *,
    ngram_range: tuple[int, int] = (1, 1),
    vectorizer: Any | None = None,
    preprocessor: Any | None = None,
    **classifier_kwargs: Any,
) -> Pipeline:
    """Construye un Pipeline de scikit-learn con TF-IDF y Multinomial Naive Bayes.

    Cumple el contrato de la fase:
    - Estimador: MultinomialNB determinista.
    - Transformador: TfidfVectorizer configurable o reemplazable.
    - Preprocesador: Opcional componente o adaptador NLP (ej. stemming o lematización).

    Args:
        ngram_range: Rango de n-gramas a utilizar si no se pasa `vectorizer` (por defecto unigramas: (1, 1)).
        vectorizer: Transformador opcional compatible con scikit-learn. Si es None,
            se usa `build_tfidf_vectorizer(ngram_range=ngram_range)`.
        preprocessor: Transformador opcional o callable de NLP a ejecutar antes de vectorizar.
        **classifier_kwargs: Parámetros adicionales para MultinomialNB (ej. alpha, fit_prior).

    Returns:
        Pipeline con pasos 'vectorizer' y 'classifier' (y opcionalmente 'preprocessor').
    """
    tfidf_step = (
        vectorizer
        if vectorizer is not None
        else build_tfidf_vectorizer(ngram_range=ngram_range)
    )
    classifier_step = MultinomialNB(**classifier_kwargs)

    steps: list[tuple[str, Any]] = []
    prep_step = _resolve_preprocessor(preprocessor)
    if prep_step is not None:
        steps.append(("preprocessor", prep_step))
    steps.append(("vectorizer", tfidf_step))
    steps.append(("classifier", classifier_step))

    return Pipeline(steps=steps)


def get_candidate_pipelines(
    *,
    random_seed: int = 42,
    max_iter: int = 1000,
    vectorizer: Any | None = None,
    preprocessor: Any | None = None,
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
            preprocessor=preprocessor,
            config=config,
        ),
        "multinomial_nb": build_multinomial_nb_pipeline(
            vectorizer=vectorizer,
            preprocessor=preprocessor,
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


# ---------------------------------------------------------------------------
# Validación cruzada estratificada (Fase 3.4A)
# ---------------------------------------------------------------------------


def build_stratified_kfold(
    config: MLConfig | None = None,
    *,
    n_splits: int | None = None,
    random_state: int | None = None,
    shuffle: bool = True,
) -> StratifiedKFold:
    """Construye un StratifiedKFold reproducible respetando la configuración central.

    Args:
        config: Configuración MLConfig opcional. Si es None, carga config/ml.yaml.
        n_splits: Número de particiones (por defecto toma config.experiment.cv_folds, 5).
        random_state: Semilla pseudoaleatoria (por defecto toma config.experiment.random_seed, 42).
        shuffle: Si se barajan los datos antes de dividir (por defecto True).

    Returns:
        Instancia de StratifiedKFold configurada.

    Raises:
        ValueError: Si `n_splits` es menor a 2.
    """
    from .config import load_config

    active_config = config or load_config()

    resolved_splits = (
        n_splits if n_splits is not None else active_config.experiment.cv_folds
    )
    # En scikit-learn, random_state debe ser None si shuffle es False
    resolved_seed = (
        (
            random_state
            if random_state is not None
            else active_config.experiment.random_seed
        )
        if shuffle
        else None
    )

    if resolved_splits < 2:
        raise ValueError(
            f"El número de particiones (n_splits) debe ser al menos 2. Se recibió: {resolved_splits}."
        )

    return StratifiedKFold(
        n_splits=resolved_splits,
        shuffle=shuffle,
        random_state=resolved_seed,
    )


def validate_cv_sample_counts(
    y: Sequence[str] | pd.Series,
    n_splits: int,
    expected_labels: Sequence[str] | None = None,
) -> None:
    """Verifica que cada clase tenga suficientes muestras para la validación cruzada estratificada.

    Args:
        y: Vector de etiquetas de entrenamiento (y_train).
        n_splits: Cantidad de particiones (folds) a generar.
        expected_labels: Clases requeridas por contrato (opcional).

    Raises:
        ValueError: Si faltan clases contractuales o si alguna clase tiene menos de `n_splits` muestras.
    """
    y_series = pd.Series(y)
    if y_series.empty or len(y_series) == 0:
        raise ValueError("El vector de etiquetas para validación cruzada no puede estar vacío.")

    counts = y_series.value_counts()
    check_labels = expected_labels if expected_labels is not None else counts.index.tolist()

    for label in check_labels:
        if label not in counts or counts[label] == 0:
            raise ValueError(
                f"La clase contractual '{label}' no está presente en los datos de entrenamiento para CV."
            )
        count = int(counts[label])
        if count < n_splits:
            raise ValueError(
                f"Muestras insuficientes para la clase '{label}': se encontraron {count} muestras, "
                f"pero se requieren al menos {n_splits} para ejecutar StratifiedKFold con {n_splits} particiones."
            )


def cross_validate_pipeline(
    pipeline: Pipeline,
    X_train: Sequence[str] | pd.Series,
    y_train: Sequence[str] | pd.Series,
    *,
    cv: StratifiedKFold | int | None = None,
    config: MLConfig | None = None,
    name: str = "pipeline",
) -> ExperimentResult:
    """Evalúa un pipeline mediante validación cruzada estratificada.

    Garantiza que TF-IDF se ajuste exclusivamente dentro del pliegue de entrenamiento
    de cada partición, previniendo fuga de información (data leakage).

    Args:
        pipeline: Pipeline de scikit-learn con vectorizador y clasificador.
        X_train: Textos del conjunto de entrenamiento.
        y_train: Etiquetas del conjunto de entrenamiento.
        cv: Objeto StratifiedKFold o entero con número de particiones.
        config: Configuración MLConfig opcional.
        name: Identificador del modelo o experimento.

    Returns:
        ExperimentResult con promedios, desviaciones estándar y métricas por fold.
    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError(
            f"Se esperaba un objeto sklearn.pipeline.Pipeline, pero se recibió: {type(pipeline).__name__}."
        )

    from .config import load_config

    active_config = config or load_config()

    if isinstance(cv, StratifiedKFold):
        cv_obj = cv
    elif isinstance(cv, int):
        cv_obj = build_stratified_kfold(active_config, n_splits=cv)
    else:
        cv_obj = build_stratified_kfold(active_config)

    X_list = list(X_train)
    y_list = list(y_train)

    if len(X_list) == 0 or len(y_list) == 0:
        raise ValueError("Los datos de entrenamiento para CV no pueden estar vacíos.")
    if len(X_list) != len(y_list):
        raise ValueError(
            f"La cantidad de textos ({len(X_list)}) no coincide con la cantidad de etiquetas ({len(y_list)})."
        )

    validate_cv_sample_counts(
        y_list,
        n_splits=cv_obj.get_n_splits(),
        expected_labels=active_config.dataset.labels,
    )

    scoring = {
        "f1_macro": "f1_macro",
        "accuracy": "accuracy",
    }

    cv_raw = cross_validate(
        estimator=pipeline,
        X=X_list,
        y=y_list,
        cv=cv_obj,
        scoring=scoring,
        return_train_score=False,
    )

    f1_scores = tuple(float(s) for s in cv_raw["test_f1_macro"])
    acc_scores = tuple(float(s) for s in cv_raw["test_accuracy"])

    metrics: dict[str, float] = {
        "f1_macro_mean": float(np.mean(f1_scores)),
        "f1_macro_std": float(np.std(f1_scores)),
        "accuracy_mean": float(np.mean(acc_scores)),
        "accuracy_std": float(np.std(acc_scores)),
    }

    fold_metrics = {
        "f1_macro": f1_scores,
        "accuracy": acc_scores,
    }

    return ExperimentResult(
        name=name,
        metrics=metrics,
        fold_metrics=fold_metrics,
    )


def _build_pipelines_from_specs(
    specs: Sequence[ExperimentSpec],
    config: MLConfig,
) -> dict[str, Pipeline]:
    """Construye pipelines a partir de especificaciones de experimento."""
    pipelines: dict[str, Pipeline] = {}
    for spec in specs:
        if spec.name == "logistic_regression":
            pipelines[spec.name] = build_logistic_regression_pipeline(
                config=config, **spec.parameters
            )
        elif spec.name == "multinomial_nb":
            pipelines[spec.name] = build_multinomial_nb_pipeline(**spec.parameters)
        else:
            raise ValueError(f"Modelo o especificación desconocida: '{spec.name}'.")
    return pipelines


def run_cross_validation(
    X_train: Sequence[str] | pd.Series | DatasetSplit,
    y_train: Sequence[str] | pd.Series | None = None,
    *,
    pipelines: Mapping[str, Pipeline] | None = None,
    specs: Sequence[ExperimentSpec] | None = None,
    config: MLConfig | None = None,
    cv: StratifiedKFold | int | None = None,
    primary_metric: str = "f1_macro",
) -> list[ExperimentResult]:
    """Ejecuta validación cruzada estratificada sobre X_train e y_train para comparar modelos.

    Utiliza exclusivamente los datos de entrenamiento. Nunca toca ni evalúa X_test ni y_test.
    Calcula F1 Macro y Accuracy por cada fold, y ordena los resultados de forma descendente
    por la métrica principal (por defecto F1 Macro promedio).

    Args:
        X_train: Datos de entrenamiento (pd.Series o secuencia) o un `DatasetSplit`.
        y_train: Etiquetas de entrenamiento. Requerido si X_train no es `DatasetSplit`.
        pipelines: Diccionario opcional {nombre: Pipeline}. Si no se provee, evalúa los dos candidatos.
        specs: Opcional secuencia de ExperimentSpec para construir pipelines.
        config: Configuración MLConfig opcional.
        cv: Objeto StratifiedKFold o entero con número de particiones.
        primary_metric: Métrica principal para ordenar los resultados (por defecto 'f1_macro').

    Returns:
        Lista de `ExperimentResult` ordenada de mejor a peor desempeño según primary_metric.
    """
    from .config import load_config

    active_config = config or load_config()

    if isinstance(X_train, DatasetSplit):
        if y_train is not None:
            raise ValueError(
                "No se debe proporcionar 'y_train' cuando 'X_train' es un objeto DatasetSplit."
            )
        # IMPORTANTE: Utiliza solamente X_train e y_train. X_test e y_test permanecen reservados.
        actual_X = X_train.X_train
        actual_y = X_train.y_train
    else:
        if y_train is None:
            raise ValueError("Se requiere 'y_train' cuando 'X_train' no es DatasetSplit.")
        actual_X = X_train
        actual_y = y_train

    if pipelines is not None:
        eval_pipelines = dict(pipelines)
    elif specs is not None:
        eval_pipelines = _build_pipelines_from_specs(specs, active_config)
    else:
        eval_pipelines = get_candidate_pipelines(config=active_config)

    results: list[ExperimentResult] = []
    for name, pipe in eval_pipelines.items():
        res = cross_validate_pipeline(
            pipeline=pipe,
            X_train=actual_X,
            y_train=actual_y,
            cv=cv,
            config=active_config,
            name=name,
        )
        results.append(res)

    # Ordenar resultados de mejor a peor según la métrica principal promedio
    rank_key = f"{primary_metric}_mean"
    results.sort(key=lambda r: r.metrics.get(rank_key, 0.0), reverse=True)
    return results


evaluate_candidates_cv = run_cross_validation


def format_cv_comparison(
    results: Sequence[ExperimentResult],
    primary_metric: str = "f1_macro",
) -> str:
    """Formatea la tabla comparativa de los resultados de validación cruzada."""
    lines = [
        "=" * 75,
        "COMPARACIÓN DE MODELOS MEDIANTE VALIDACIÓN CRUZADA ESTRATIFICADA",
        "=" * 75,
    ]
    for rank, res in enumerate(results, start=1):
        f1_m = res.metrics.get("f1_macro_mean", 0.0)
        f1_s = res.metrics.get("f1_macro_std", 0.0)
        acc_m = res.metrics.get("accuracy_mean", 0.0)
        acc_s = res.metrics.get("accuracy_std", 0.0)

        lines.append(f"[{rank}] Modelo: {res.name}")
        lines.append(f"    F1 Macro promedio: {f1_m:.4f} ± {f1_s:.4f}")
        lines.append(f"    Accuracy promedio: {acc_m:.4f} ± {acc_s:.4f}")

        if "f1_macro" in res.fold_metrics:
            folds_f1 = ", ".join(f"{v:.4f}" for v in res.fold_metrics["f1_macro"])
            lines.append(f"    Folds F1 Macro:   [{folds_f1}]")
        if "accuracy" in res.fold_metrics:
            folds_acc = ", ".join(f"{v:.4f}" for v in res.fold_metrics["accuracy"])
            lines.append(f"    Folds Accuracy:   [{folds_acc}]")
        lines.append("-" * 75)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optimización de hiperparámetros con GridSearchCV (Fase 3.4B)
# ---------------------------------------------------------------------------

DEFAULT_PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "logistic_regression": {
        "classifier__C": [0.1, 1.0, 10.0],
    },
    "multinomial_nb": {
        "classifier__alpha": [0.1, 1.0, 10.0],
    },
}


def get_default_param_grid(model_name: str) -> dict[str, list[Any]]:
    """Devuelve una copia del grid de hiperparámetros por defecto para el modelo dado.

    Args:
        model_name: Nombre del modelo ('logistic_regression' o 'multinomial_nb').

    Returns:
        Diccionario con listas de valores para los hiperparámetros del pipeline.

    Raises:
        ValueError: Si el modelo no está en DEFAULT_PARAM_GRIDS.
    """
    if model_name not in DEFAULT_PARAM_GRIDS:
        raise ValueError(
            f"Modelo desconocido '{model_name}'. Opciones disponibles: {list(DEFAULT_PARAM_GRIDS.keys())}."
        )
    return {k: list(v) for k, v in DEFAULT_PARAM_GRIDS[model_name].items()}


@dataclass(frozen=True)
class GridSearchResult:
    """Resultado estructurado de la optimización de hiperparámetros con GridSearchCV.

    Attributes:
        model_name: Identificador del modelo (ej. 'logistic_regression').
        best_params: Mejor combinación de hiperparámetros encontrada.
        best_score: Mejor puntaje de validación cruzada para la métrica principal.
        best_estimator: Pipeline reajustado sobre X_train con los mejores parámetros.
        cv_results: Diccionario de resultados de scikit-learn (cv_results_).
        param_grid: Grid de parámetros explorado.
        primary_metric: Métrica empleada como criterio de optimización (ej. 'f1_macro').
    """

    model_name: str
    best_params: Mapping[str, Any]
    best_score: float
    best_estimator: Pipeline
    cv_results: Mapping[str, Any]
    param_grid: Mapping[str, Sequence[Any]]
    primary_metric: str = "f1_macro"

    def to_dataframe(self) -> pd.DataFrame:
        """Devuelve un DataFrame resumen con las combinaciones evaluadas y sus puntajes."""
        return cv_results_to_dataframe(self.cv_results, self.primary_metric)


def cv_results_to_dataframe(
    cv_results: Mapping[str, Any],
    metric: str = "f1_macro",
) -> pd.DataFrame:
    """Convierte el diccionario cv_results_ de scikit-learn a un pd.DataFrame ordenado."""
    data = []
    params_list = cv_results.get("params", [])
    mean_key = f"mean_test_{metric}"
    std_key = f"std_test_{metric}"
    rank_key = f"rank_test_{metric}"

    for i, p in enumerate(params_list):
        row = {
            "params": p,
            f"{metric}_mean": float(cv_results[mean_key][i]) if mean_key in cv_results else None,
            f"{metric}_std": float(cv_results[std_key][i]) if std_key in cv_results else None,
            "rank": int(cv_results[rank_key][i]) if rank_key in cv_results else i + 1,
        }
        data.append(row)

    df = pd.DataFrame(data)
    if "rank" in df.columns:
        df = df.sort_values(by="rank").reset_index(drop=True)
    return df


def tune_pipeline_grid_search(
    pipeline: Pipeline,
    param_grid: Mapping[str, Sequence[Any]],
    X_train: Sequence[str] | pd.Series,
    y_train: Sequence[str] | pd.Series,
    *,
    cv: StratifiedKFold | int | None = None,
    scoring: str | None = None,
    config: MLConfig | None = None,
    name: str = "pipeline",
    refit: bool = True,
    n_jobs: int | None = None,
) -> GridSearchResult:
    """Ejecuta GridSearchCV sobre un pipeline utilizando exclusivamente datos de entrenamiento.

    Mantiene el vectorizador TF-IDF dentro del Pipeline durante cada partición de
    la validación cruzada interna, asegurando la ausencia total de fuga de información.

    Args:
        pipeline: Pipeline de scikit-learn (ej. vectorizer + classifier).
        param_grid: Diccionario de hiperparámetros a optimizar.
        X_train: Textos del conjunto de entrenamiento.
        y_train: Etiquetas del conjunto de entrenamiento.
        cv: Objeto StratifiedKFold o entero con número de pliegues (por defecto 5 folds de config).
        scoring: Métrica de optimización (por defecto 'f1_macro' de config).
        config: Configuración MLConfig opcional.
        name: Identificador del modelo (ej. 'logistic_regression').
        refit: Si es True, reajusta el mejor estimador sobre todo X_train.
        n_jobs: Número de hilos en paralelo (por defecto None / 1 para reproducibilidad).

    Returns:
        GridSearchResult con best_params, best_score, best_estimator y cv_results.
    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError(
            f"Se esperaba un objeto sklearn.pipeline.Pipeline, pero se recibió: {type(pipeline).__name__}."
        )

    from .config import load_config

    active_config = config or load_config()

    if isinstance(cv, StratifiedKFold):
        cv_obj = cv
    elif isinstance(cv, int):
        cv_obj = build_stratified_kfold(active_config, n_splits=cv)
    else:
        cv_obj = build_stratified_kfold(active_config)

    resolved_scoring = scoring or active_config.experiment.primary_metric

    X_list = list(X_train)
    y_list = list(y_train)

    if len(X_list) == 0 or len(y_list) == 0:
        raise ValueError("Los datos de entrenamiento para GridSearchCV no pueden estar vacíos.")
    if len(X_list) != len(y_list):
        raise ValueError(
            f"La cantidad de textos ({len(X_list)}) no coincide con la cantidad de etiquetas ({len(y_list)})."
        )

    validate_cv_sample_counts(
        y_list,
        n_splits=cv_obj.get_n_splits(),
        expected_labels=active_config.dataset.labels,
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=dict(param_grid),
        scoring=resolved_scoring,
        cv=cv_obj,
        refit=refit,
        n_jobs=n_jobs,
        return_train_score=False,
    )

    grid_search.fit(X_list, y_list)

    return GridSearchResult(
        model_name=name,
        best_params=dict(grid_search.best_params_),
        best_score=float(grid_search.best_score_),
        best_estimator=grid_search.best_estimator_,
        cv_results=dict(grid_search.cv_results_),
        param_grid={k: list(v) for k, v in param_grid.items()},
        primary_metric=resolved_scoring,
    )


def run_grid_search(
    X_train: Sequence[str] | pd.Series | DatasetSplit,
    y_train: Sequence[str] | pd.Series | None = None,
    *,
    param_grids: Mapping[str, Mapping[str, Sequence[Any]]] | None = None,
    pipelines: Mapping[str, Pipeline] | None = None,
    config: MLConfig | None = None,
    cv: StratifiedKFold | int | None = None,
    scoring: str | None = None,
    n_jobs: int | None = None,
) -> dict[str, GridSearchResult]:
    """Ejecuta la optimización de hiperparámetros con GridSearchCV para ambos modelos candidatos.

    Utiliza EXCLUSIVAMENTE X_train e y_train. Nunca accede ni evalúa X_test ni y_test.
    Mantiene TF-IDF encapsulado dentro del Pipeline durante cada fold.

    Args:
        X_train: Conjunto de entrenamiento o un objeto DatasetSplit.
        y_train: Etiquetas de entrenamiento. Requerido si X_train no es DatasetSplit.
        param_grids: Diccionario opcional {nombre_modelo: grid_de_hiperparametros}.
            Por defecto usa DEFAULT_PARAM_GRIDS:
            - logistic_regression: {'classifier__C': [0.1, 1.0, 10.0]}
            - multinomial_nb: {'classifier__alpha': [0.1, 1.0, 10.0]}
        pipelines: Opcional diccionario {nombre_modelo: Pipeline}. Si es None,
            construye instancias limpias de ambos candidatos.
        config: Configuración MLConfig opcional.
        cv: Objeto StratifiedKFold o entero con número de particiones.
        scoring: Métrica objetivo (por defecto 'f1_macro' según config).
        n_jobs: Paralelización para GridSearchCV.

    Returns:
        Diccionario {nombre_modelo: GridSearchResult}.
    """
    from .config import load_config

    active_config = config or load_config()

    if isinstance(X_train, DatasetSplit):
        if y_train is not None:
            raise ValueError(
                "No se debe proporcionar 'y_train' cuando 'X_train' es un objeto DatasetSplit."
            )
        # IMPORTANTE: Utiliza exclusivamente X_train e y_train. X_test e y_test permanecen reservados.
        actual_X = X_train.X_train
        actual_y = X_train.y_train
    else:
        if y_train is None:
            raise ValueError("Se requiere 'y_train' cuando 'X_train' no es DatasetSplit.")
        actual_X = X_train
        actual_y = y_train

    eval_pipelines = (
        dict(pipelines)
        if pipelines is not None
        else get_candidate_pipelines(config=active_config)
    )
    resolved_param_grids = (
        dict(param_grids) if param_grids is not None else DEFAULT_PARAM_GRIDS
    )

    results: dict[str, GridSearchResult] = {}
    for name, pipe in eval_pipelines.items():
        grid = resolved_param_grids.get(
            name,
            get_default_param_grid(name) if name in DEFAULT_PARAM_GRIDS else {},
        )
        res = tune_pipeline_grid_search(
            pipeline=pipe,
            param_grid=grid,
            X_train=actual_X,
            y_train=actual_y,
            cv=cv,
            scoring=scoring,
            config=active_config,
            name=name,
            n_jobs=n_jobs,
        )
        results[name] = res

    return results


def format_grid_search_comparison(
    results: Mapping[str, GridSearchResult] | Sequence[GridSearchResult],
    primary_metric: str = "f1_macro",
) -> str:
    """Genera una tabla comparativa legible con los resultados de GridSearchCV.

    No selecciona un modelo definitivo; presenta de manera transparente las
    mejores configuraciones encontradas y el historial de combinaciones probadas.
    """
    results_list: list[GridSearchResult] = (
        list(results.values()) if isinstance(results, Mapping) else list(results)
    )

    lines = [
        "=" * 80,
        "OPTIMIZACIÓN DE HIPERPARÁMETROS CON GRIDSEARCHCV (5 FOLDS ESTRATIFICADOS)",
        "=" * 80,
        f"Métrica principal de optimización: {primary_metric}",
        "",
    ]

    for rank, res in enumerate(results_list, start=1):
        lines.append(f"[{rank}] Modelo candidato: {res.model_name}")
        lines.append(f"    Mejor combinación encontrada: {res.best_params}")
        lines.append(f"    Mejor {primary_metric} en CV: {res.best_score:.4f}")
        lines.append("    Configuraciones exploradas:")

        mean_key = f"mean_test_{primary_metric}"
        std_key = f"std_test_{primary_metric}"
        rank_key = f"rank_test_{primary_metric}"

        params_list = res.cv_results.get("params", [])
        for i, p in enumerate(params_list):
            m_val = res.cv_results[mean_key][i] if mean_key in res.cv_results else 0.0
            s_val = res.cv_results[std_key][i] if std_key in res.cv_results else 0.0
            r_val = res.cv_results[rank_key][i] if rank_key in res.cv_results else i + 1
            param_str = ", ".join(f"{k.split('__')[-1]}={v}" for k, v in p.items())
            lines.append(
                f"      * {param_str:<20} -> {primary_metric}: {m_val:.4f} ± {s_val:.4f} (Rank {r_val})"
            )

        clf = res.best_estimator.named_steps.get("classifier", res.best_estimator[-1])
        vec = res.best_estimator.named_steps.get("vectorizer", res.best_estimator[0])
        lines.append(
            f"    Mejor estimador ajustado: Pipeline[{type(vec).__name__} -> {type(clf).__name__}]"
        )
        lines.append("-" * 80)

    lines.append(
        "(Nota: No se selecciona modelo definitivo en esta subfase; X_test permanece reservado)."
    )
    lines.append("=" * 80)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparación de representaciones TF-IDF: Unigramas vs Bigramas (Fase 3.5A)
# ---------------------------------------------------------------------------

NGRAM_CONFIGURATIONS: dict[str, tuple[int, int]] = {
    "unigrams": (1, 1),
    "unigrams_and_bigrams": (1, 2),
}


@dataclass(frozen=True)
class NgramComparisonResult:
    """Resultado estructurado de la comparación entre unigramas y unigramas+bigramas.

    Attributes:
        results: Mapeo {nombre_combinacion: ExperimentResult} con las 4 combinaciones evaluadas.
        cv_folds: Cantidad de particiones estratificadas utilizadas.
        primary_metric: Métrica principal de comparación (por defecto 'f1_macro').
        ranking: Tupla ordenada de ExperimentResult de mayor a menor según la métrica principal.
        ngram_configs: Mapeo de nombres a rangos (ej. {'unigrams': (1, 1), ...}).
    """

    results: Mapping[str, ExperimentResult]
    cv_folds: int = 5
    primary_metric: str = "f1_macro"
    ranking: Sequence[ExperimentResult] = ()
    ngram_configs: Mapping[str, tuple[int, int]] = field(
        default_factory=lambda: dict(NGRAM_CONFIGURATIONS)
    )

    def to_dataframe(self) -> pd.DataFrame:
        """Convierte los resultados a un pd.DataFrame estructurado para análisis."""
        data = []
        for rank, res in enumerate(self.ranking, start=1):
            parts = res.name.split("__")
            model_name = parts[0] if len(parts) > 1 else res.name
            rep_name = parts[1] if len(parts) > 1 else "custom"
            ngram_range = self.ngram_configs.get(rep_name, None)

            f1_mean = res.metrics.get("f1_macro_mean", None)
            f1_std = res.metrics.get("f1_macro_std", None)
            acc_mean = res.metrics.get("accuracy_mean", None)
            acc_std = res.metrics.get("accuracy_std", None)

            data.append(
                {
                    "rank": rank,
                    "combination": res.name,
                    "model": model_name,
                    "representation": rep_name,
                    "ngram_range": str(ngram_range) if ngram_range else "-",
                    "f1_macro_mean": f1_mean,
                    "f1_macro_std": f1_std,
                    "accuracy_mean": acc_mean,
                    "accuracy_std": acc_std,
                }
            )
        return pd.DataFrame(data)

    def summary(self) -> str:
        """Genera el reporte textual de comparación entre unigramas y bigramas."""
        return format_ngram_comparison(self, self.primary_metric)


def build_ngram_comparison_pipelines(
    *,
    config: MLConfig | None = None,
    ngram_configs: Mapping[str, tuple[int, int]] | None = None,
    vectorizer_factory: Any | None = None,
    **classifier_kwargs: Any,
) -> dict[str, Pipeline]:
    """Construye las combinaciones de pipelines para comparar unigramas vs unigramas+bigramas.

    Genera cuatro pipelines:
    - 'logistic_regression__unigrams': Regresión Logística con TF-IDF (1, 1)
    - 'logistic_regression__unigrams_and_bigrams': Regresión Logística con TF-IDF (1, 2)
    - 'multinomial_nb__unigrams': Multinomial Naive Bayes con TF-IDF (1, 1)
    - 'multinomial_nb__unigrams_and_bigrams': Multinomial Naive Bayes con TF-IDF (1, 2)

    Permite inyectar un `vectorizer_factory` personalizado para futura compatibilidad
    con transformadores del módulo NLP de José.

    Args:
        config: Configuración MLConfig opcional.
        ngram_configs: Mapeo opcional de nombres a rangos (por defecto NGRAM_CONFIGURATIONS).
        vectorizer_factory: Fábrica opcional `func(ngram_range=...) -> Transformer`.
        **classifier_kwargs: Hiperparámetros adicionales por clasificador.

    Returns:
        Diccionario con las cuatro instancias de Pipeline listas para evaluar.
    """
    from .config import load_config

    active_config = config or load_config()

    resolved_ngrams = (
        dict(ngram_configs) if ngram_configs is not None else NGRAM_CONFIGURATIONS
    )

    pipelines: dict[str, Pipeline] = {}
    for ngram_name, ngram_range in resolved_ngrams.items():
        if vectorizer_factory is not None:
            lr_vec = vectorizer_factory(ngram_range=ngram_range)
            nb_vec = vectorizer_factory(ngram_range=ngram_range)
        else:
            lr_vec = build_tfidf_vectorizer(ngram_range=ngram_range)
            nb_vec = build_tfidf_vectorizer(ngram_range=ngram_range)

        lr_name = f"logistic_regression__{ngram_name}"
        nb_name = f"multinomial_nb__{ngram_name}"

        pipelines[lr_name] = build_logistic_regression_pipeline(
            config=active_config,
            vectorizer=lr_vec,
            **classifier_kwargs.get("logistic_regression", {}),
        )
        pipelines[nb_name] = build_multinomial_nb_pipeline(
            vectorizer=nb_vec,
            **classifier_kwargs.get("multinomial_nb", {}),
        )

    return pipelines


def compare_ngram_representations(
    X_train: Sequence[str] | pd.Series | DatasetSplit,
    y_train: Sequence[str] | pd.Series | None = None,
    *,
    config: MLConfig | None = None,
    cv: StratifiedKFold | int | None = None,
    primary_metric: str = "f1_macro",
    ngram_configs: Mapping[str, tuple[int, int]] | None = None,
    vectorizer_factory: Any | None = None,
) -> NgramComparisonResult:
    """Ejecuta la comparación reproducible de TF-IDF (1, 1) vs TF-IDF (1, 2) con CV estratificada.

    Evalúa cuatro combinaciones sobre exactamente los mismos pliegues estratificados
    para garantizar una comparación justa y determinista.

    Utiliza EXCLUSIVAMENTE X_train e y_train. X_test e y_test permanecen reservados.

    Args:
        X_train: Datos de entrenamiento (pd.Series, secuencia) o un `DatasetSplit`.
        y_train: Etiquetas de entrenamiento. Requerido si X_train no es `DatasetSplit`.
        config: Configuración MLConfig opcional.
        cv: Objeto StratifiedKFold o entero con número de pliegues.
        primary_metric: Métrica principal de comparación (por defecto 'f1_macro').
        ngram_configs: Mapeo opcional de nombres de representación a rangos de n-gramas.
        vectorizer_factory: Fábrica opcional para construir el vectorizador con n-gramas.

    Returns:
        NgramComparisonResult con los resultados estructurados de las 4 combinaciones.
    """
    from .config import load_config

    active_config = config or load_config()

    if isinstance(X_train, DatasetSplit):
        if y_train is not None:
            raise ValueError(
                "No se debe proporcionar 'y_train' cuando 'X_train' es un objeto DatasetSplit."
            )
        # IMPORTANTE: Utiliza exclusivamente X_train e y_train. X_test e y_test permanecen reservados.
        actual_X = X_train.X_train
        actual_y = X_train.y_train
    else:
        if y_train is None:
            raise ValueError("Se requiere 'y_train' cuando 'X_train' no es DatasetSplit.")
        actual_X = X_train
        actual_y = y_train

    # Resolver objeto StratifiedKFold idéntico para todas las combinaciones
    if isinstance(cv, StratifiedKFold):
        cv_obj = cv
    elif isinstance(cv, int):
        cv_obj = build_stratified_kfold(active_config, n_splits=cv)
    else:
        cv_obj = build_stratified_kfold(active_config)

    resolved_ngrams = (
        dict(ngram_configs) if ngram_configs is not None else NGRAM_CONFIGURATIONS
    )

    pipelines = build_ngram_comparison_pipelines(
        config=active_config,
        ngram_configs=resolved_ngrams,
        vectorizer_factory=vectorizer_factory,
    )

    results: dict[str, ExperimentResult] = {}
    for name, pipe in pipelines.items():
        res = cross_validate_pipeline(
            pipeline=pipe,
            X_train=actual_X,
            y_train=actual_y,
            cv=cv_obj,
            config=active_config,
            name=name,
        )
        results[name] = res

    rank_key = f"{primary_metric}_mean"
    ranked = sorted(
        results.values(),
        key=lambda r: r.metrics.get(rank_key, 0.0),
        reverse=True,
    )

    return NgramComparisonResult(
        results=results,
        cv_folds=cv_obj.get_n_splits(),
        primary_metric=primary_metric,
        ranking=tuple(ranked),
        ngram_configs=resolved_ngrams,
    )


def format_ngram_comparison(
    comparison: NgramComparisonResult | Sequence[ExperimentResult] | Mapping[str, ExperimentResult],
    primary_metric: str = "f1_macro",
) -> str:
    """Genera una tabla comparativa legible para las representaciones de n-gramas.

    Presenta de forma neutral las cuatro combinaciones evaluadas, sin declarar
    un ganador definitivo y preservando el aislamiento del conjunto de prueba.
    """
    if isinstance(comparison, NgramComparisonResult):
        ranking_list = list(comparison.ranking)
        cv_folds = comparison.cv_folds
        ngram_map = comparison.ngram_configs
    elif isinstance(comparison, Mapping):
        ranking_list = sorted(
            comparison.values(),
            key=lambda r: r.metrics.get(f"{primary_metric}_mean", 0.0),
            reverse=True,
        )
        cv_folds = 5
        ngram_map = NGRAM_CONFIGURATIONS
    else:
        ranking_list = sorted(
            comparison,
            key=lambda r: r.metrics.get(f"{primary_metric}_mean", 0.0),
            reverse=True,
        )
        cv_folds = 5
        ngram_map = NGRAM_CONFIGURATIONS

    lines = [
        "=" * 85,
        "COMPARACIÓN DE REPRESENTACIONES: UNIGRAMAS (1,1) VS UNIGRAMAS + BIGRAMAS (1,2)",
        "=" * 85,
        f"Validación cruzada estratificada: {cv_folds} pliegues (shuffle=True, random_state=42)",
        "Datos evaluados: Exclusivamente conjunto de entrenamiento (X_train, y_train)",
        "",
        f"Ranking preliminar por {primary_metric} (promedio ± desviación estándar):",
        "-" * 85,
    ]

    for rank, res in enumerate(ranking_list, start=1):
        parts = res.name.split("__")
        model_part = parts[0] if len(parts) > 1 else res.name
        rep_part = parts[1] if len(parts) > 1 else "desconocido"
        ngram_tuple = ngram_map.get(rep_part, "-")

        model_display = (
            "Regresión Logística"
            if "logistic" in model_part
            else ("Multinomial Naive Bayes" if "nb" in model_part else model_part)
        )
        rep_display = (
            "Unigramas (1, 1)"
            if rep_part == "unigrams"
            else ("Unigramas + Bigramas (1, 2)" if "bigram" in rep_part else str(ngram_tuple))
        )

        f1_m = res.metrics.get("f1_macro_mean", 0.0)
        f1_s = res.metrics.get("f1_macro_std", 0.0)
        acc_m = res.metrics.get("accuracy_mean", 0.0)
        acc_s = res.metrics.get("accuracy_std", 0.0)

        lines.append(f"[{rank}] Combinación: {res.name}")
        lines.append(f"    Algoritmo:      {model_display}")
        lines.append(f"    Representación: {rep_display}")
        lines.append(f"    F1 Macro:       {f1_m:.4f} ± {f1_s:.4f}")
        lines.append(f"    Accuracy:       {acc_m:.4f} ± {acc_s:.4f}")

        if "f1_macro" in res.fold_metrics:
            folds_f1 = ", ".join(f"{v:.4f}" for v in res.fold_metrics["f1_macro"])
            lines.append(f"    Folds F1 Macro: [{folds_f1}]")
        lines.append("-" * 85)

    lines.append(
        "(Nota metodológica: No se selecciona modelo definitivo ni se evalúa X_test en la Fase 3.5A)."
    )
    lines.append("=" * 85)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparación de técnicas de preprocesamiento NLP: Stemming vs Lematización (Fase 3.5B)
# ---------------------------------------------------------------------------

PREPROCESSING_TECHNIQUES: tuple[str, ...] = ("stemming", "lemmatization")


def get_nlp_preprocessors() -> dict[str, Any] | None:
    """Intenta descubrir e instanciar los preprocesadores de stemming y lematización de José en `src.nlp`.

    Inspecciona si el módulo `src.nlp` expone implementaciones de stemming y lematización.
    Si José aún no los ha implementado, devuelve None sin lanzar excepciones no deseadas.

    Returns:
        Diccionario {'stemming': ..., 'lemmatization': ...} si ambos están disponibles,
        o None si aún no han sido implementados en `src.nlp`.
    """
    try:
        import src.nlp as nlp  # type: ignore
    except (ImportError, Exception):
        return None

    stemmer = (
        getattr(nlp, "stemmer", None)
        or getattr(nlp, "Stemmer", None)
        or getattr(nlp, "stem_text", None)
        or getattr(nlp, "SpanishStemmer", None)
        or getattr(nlp, "get_stemmer", None)
    )
    lemmatizer = (
        getattr(nlp, "lemmatizer", None)
        or getattr(nlp, "Lemmatizer", None)
        or getattr(nlp, "lemmatize_text", None)
        or getattr(nlp, "SpanishLemmatizer", None)
        or getattr(nlp, "get_lemmatizer", None)
    )

    if stemmer is None or lemmatizer is None:
        return None

    actual_stemmer = (
        stemmer()
        if callable(stemmer)
        and not hasattr(stemmer, "transform")
        and getattr(stemmer, "__name__", "").startswith("get_")
        else stemmer
    )
    actual_lemmatizer = (
        lemmatizer()
        if callable(lemmatizer)
        and not hasattr(lemmatizer, "transform")
        and getattr(lemmatizer, "__name__", "").startswith("get_")
        else lemmatizer
    )

    return {
        "stemming": actual_stemmer,
        "lemmatization": actual_lemmatizer,
    }


@dataclass(frozen=True)
class PreprocessingComparisonResult:
    """Estructura para almacenar y presentar la comparación de técnicas de preprocesamiento NLP.

    Attributes:
        results: Mapeo de nombre de combinación a ExperimentResult.
        cv_folds: Número de particiones de validación cruzada.
        primary_metric: Métrica principal de comparación (por defecto 'f1_macro').
        ranking: Tupla ordenada de ExperimentResult de mayor a menor según la métrica principal.
        techniques: Lista de nombres de técnicas evaluadas (ej. ['stemming', 'lemmatization']).
        ngram_range: Rango de n-gramas configurado para TF-IDF.
    """

    results: Mapping[str, ExperimentResult]
    cv_folds: int = 5
    primary_metric: str = "f1_macro"
    ranking: Sequence[ExperimentResult] = ()
    techniques: Sequence[str] = ()
    ngram_range: tuple[int, int] = (1, 1)

    def to_dataframe(self) -> pd.DataFrame:
        """Convierte los resultados a un pd.DataFrame estructurado para análisis."""
        data = []
        for rank, res in enumerate(self.ranking, start=1):
            parts = res.name.split("__")
            model_name = parts[0] if len(parts) > 1 else res.name
            tech_name = parts[1] if len(parts) > 1 else "custom"

            f1_mean = res.metrics.get("f1_macro_mean", None)
            f1_std = res.metrics.get("f1_macro_std", None)
            acc_mean = res.metrics.get("accuracy_mean", None)
            acc_std = res.metrics.get("accuracy_std", None)

            data.append(
                {
                    "rank": rank,
                    "combination": res.name,
                    "model": model_name,
                    "technique": tech_name,
                    "ngram_range": str(self.ngram_range),
                    "f1_macro_mean": f1_mean,
                    "f1_macro_std": f1_std,
                    "accuracy_mean": acc_mean,
                    "accuracy_std": acc_std,
                }
            )
        return pd.DataFrame(data)

    def summary(self) -> str:
        """Genera el reporte textual de comparación entre técnicas de preprocesamiento."""
        return format_preprocessing_comparison(self, self.primary_metric)


def build_preprocessing_comparison_pipelines(
    preprocessors: Mapping[str, Any],
    *,
    config: MLConfig | None = None,
    ngram_range: tuple[int, int] = (1, 1),
    vectorizer_factory: Any | None = None,
    **classifier_kwargs: Any,
) -> dict[str, Pipeline]:
    """Construye las combinaciones de pipelines para comparar técnicas de preprocesamiento NLP.

    Para cada técnica en `preprocessors` (ej. 'stemming', 'lemmatization'), genera
    dos pipelines:
    - 'logistic_regression__{technique}'
    - 'multinomial_nb__{technique}'

    Args:
        preprocessors: Mapeo {nombre_tecnica: transformador_o_callable}.
        config: Configuración MLConfig opcional.
        ngram_range: Rango de n-gramas a utilizar para TF-IDF (por defecto (1, 1)).
        vectorizer_factory: Fábrica opcional `func(ngram_range=...) -> Transformer`.
        **classifier_kwargs: Hiperparámetros adicionales por estimador.

    Returns:
        Diccionario {nombre_combinacion: Pipeline}.
    """
    from .config import load_config

    active_config = config or load_config()

    if not isinstance(preprocessors, Mapping):
        raise TypeError(
            f"Se esperaba un Mapping para 'preprocessors', pero se recibió: {type(preprocessors).__name__}."
        )
    if len(preprocessors) == 0:
        raise ValueError("El mapeo de 'preprocessors' no puede estar vacío.")

    pipelines: dict[str, Pipeline] = {}
    for tech_name, prep_obj in preprocessors.items():
        if vectorizer_factory is not None:
            lr_vec = vectorizer_factory(ngram_range=ngram_range)
            nb_vec = vectorizer_factory(ngram_range=ngram_range)
        else:
            lr_vec = build_tfidf_vectorizer(ngram_range=ngram_range)
            nb_vec = build_tfidf_vectorizer(ngram_range=ngram_range)

        lr_name = f"logistic_regression__{tech_name}"
        nb_name = f"multinomial_nb__{tech_name}"

        pipelines[lr_name] = build_logistic_regression_pipeline(
            config=active_config,
            vectorizer=lr_vec,
            preprocessor=prep_obj,
            **classifier_kwargs.get("logistic_regression", {}),
        )
        pipelines[nb_name] = build_multinomial_nb_pipeline(
            vectorizer=nb_vec,
            preprocessor=prep_obj,
            **classifier_kwargs.get("multinomial_nb", {}),
        )

    return pipelines


def compare_preprocessing_techniques(
    X_train: Sequence[str] | pd.Series | DatasetSplit,
    y_train: Sequence[str] | pd.Series | None = None,
    *,
    preprocessors: Mapping[str, Any] | None = None,
    config: MLConfig | None = None,
    cv: StratifiedKFold | int | None = None,
    ngram_range: tuple[int, int] = (1, 1),
    vectorizer_factory: Any | None = None,
    primary_metric: str = "f1_macro",
) -> PreprocessingComparisonResult:
    """Ejecuta la comparación reproducible entre técnicas de preprocesamiento NLP con CV estratificada.

    Evalúa las combinaciones de (modelo x técnica de preprocesamiento) sobre
    exactamente las mismas particiones estratificadas de entrenamiento.

    El preprocesamiento se ejecuta dentro del Pipeline de scikit-learn, evitando
    cualquier fuga de información hacia los pliegues de validación.

    Utiliza EXCLUSIVAMENTE X_train e y_train. X_test e y_test permanecen estrictamente reservados.

    Args:
        X_train: Datos de entrenamiento (pd.Series, secuencia) o un `DatasetSplit`.
        y_train: Etiquetas de entrenamiento. Requerido si X_train no es `DatasetSplit`.
        preprocessors: Mapeo {nombre_tecnica: transformador_o_callable}. Si es None,
            intenta obtener los preprocesadores de José en `src.nlp`. Si no están
            implementados, lanza NotImplementedError explicativo.
        config: Configuración MLConfig opcional.
        cv: Objeto StratifiedKFold o entero con número de pliegues.
        ngram_range: Rango de n-gramas para TF-IDF (por defecto (1, 1)).
        vectorizer_factory: Fábrica opcional para el vectorizador TF-IDF.
        primary_metric: Métrica principal de comparación (por defecto 'f1_macro').

    Returns:
        PreprocessingComparisonResult con los resultados estructurados de cada combinación.

    Raises:
        NotImplementedError: Si preprocessors es None y José aún no ha implementado el módulo `src.nlp`.
        ValueError: Si los datos de entrenamiento están vacíos o las dimensiones no coinciden.
    """
    from .config import load_config

    active_config = config or load_config()

    if isinstance(X_train, DatasetSplit):
        if y_train is not None:
            raise ValueError(
                "No se debe proporcionar 'y_train' cuando 'X_train' es un objeto DatasetSplit."
            )
        actual_X = X_train.X_train
        actual_y = X_train.y_train
    else:
        if y_train is None:
            raise ValueError("Se requiere 'y_train' cuando 'X_train' no es DatasetSplit.")
        actual_X = X_train
        actual_y = y_train

    resolved_preprocessors = preprocessors
    if resolved_preprocessors is None:
        resolved_preprocessors = get_nlp_preprocessors()
        if resolved_preprocessors is None:
            raise NotImplementedError(
                "Los componentes de preprocesamiento de NLP (stemming y lematización) "
                "aún no han sido implementados por José en 'src.nlp'. Para evaluar la "
                "comparación o realizar pruebas, inyecte un diccionario en el parámetro "
                "'preprocessors' (ej. {'stemming': transformador_o_func, 'lemmatization': transformador_o_func})."
            )

    if not isinstance(resolved_preprocessors, Mapping) or len(resolved_preprocessors) == 0:
        raise ValueError("El parámetro 'preprocessors' debe ser un Mapping no vacío.")

    if isinstance(cv, StratifiedKFold):
        cv_obj = cv
    elif isinstance(cv, int):
        cv_obj = build_stratified_kfold(active_config, n_splits=cv)
    else:
        cv_obj = build_stratified_kfold(active_config)

    pipelines = build_preprocessing_comparison_pipelines(
        preprocessors=resolved_preprocessors,
        config=active_config,
        ngram_range=ngram_range,
        vectorizer_factory=vectorizer_factory,
    )

    results: dict[str, ExperimentResult] = {}
    for name, pipe in pipelines.items():
        res = cross_validate_pipeline(
            pipeline=pipe,
            X_train=actual_X,
            y_train=actual_y,
            cv=cv_obj,
            config=active_config,
            name=name,
        )
        results[name] = res

    rank_key = f"{primary_metric}_mean"
    ranked = sorted(
        results.values(),
        key=lambda r: r.metrics.get(rank_key, 0.0),
        reverse=True,
    )

    return PreprocessingComparisonResult(
        results=results,
        cv_folds=cv_obj.get_n_splits(),
        primary_metric=primary_metric,
        ranking=tuple(ranked),
        techniques=tuple(resolved_preprocessors.keys()),
        ngram_range=ngram_range,
    )


def format_preprocessing_comparison(
    comparison: PreprocessingComparisonResult | Sequence[ExperimentResult] | Mapping[str, ExperimentResult],
    primary_metric: str = "f1_macro",
) -> str:
    """Genera una tabla comparativa legible para las técnicas de preprocesamiento NLP.

    Presenta de forma neutral las combinaciones evaluadas, sin declarar
    un ganador definitivo y preservando el aislamiento del conjunto de prueba.
    """
    if isinstance(comparison, PreprocessingComparisonResult):
        ranking_list = list(comparison.ranking)
        cv_folds = comparison.cv_folds
        ngram_range = comparison.ngram_range
    elif isinstance(comparison, Mapping):
        ranking_list = sorted(
            comparison.values(),
            key=lambda r: r.metrics.get(f"{primary_metric}_mean", 0.0),
            reverse=True,
        )
        cv_folds = 5
        ngram_range = (1, 1)
    else:
        ranking_list = sorted(
            comparison,
            key=lambda r: r.metrics.get(f"{primary_metric}_mean", 0.0),
            reverse=True,
        )
        cv_folds = 5
        ngram_range = (1, 1)

    lines = [
        "=" * 85,
        "COMPARACIÓN DE PREPROCESAMIENTO NLP: STEMMING VS LEMATIZACIÓN",
        "=" * 85,
        f"Validación cruzada estratificada: {cv_folds} pliegues (shuffle=True, random_state=42)",
        f"Configuración TF-IDF: ngram_range={ngram_range}",
        "Datos evaluados: Exclusivamente conjunto de entrenamiento (X_train, y_train)",
        "",
        f"Ranking preliminar por {primary_metric} (promedio ± desviación estándar):",
        "-" * 85,
    ]

    for rank, res in enumerate(ranking_list, start=1):
        parts = res.name.split("__")
        model_part = parts[0] if len(parts) > 1 else res.name
        tech_part = parts[1] if len(parts) > 1 else "desconocido"

        model_display = (
            "Regresión Logística"
            if "logistic" in model_part
            else ("Multinomial Naive Bayes" if "nb" in model_part else model_part)
        )
        tech_display = (
            "Stemming"
            if "stem" in tech_part.lower()
            else ("Lematización" if "lem" in tech_part.lower() else tech_part)
        )

        f1_m = res.metrics.get("f1_macro_mean", 0.0)
        f1_s = res.metrics.get("f1_macro_std", 0.0)
        acc_m = res.metrics.get("accuracy_mean", 0.0)
        acc_s = res.metrics.get("accuracy_std", 0.0)

        lines.append(f"[{rank}] Combinación: {res.name}")
        lines.append(f"    Algoritmo:      {model_display}")
        lines.append(f"    Técnica NLP:    {tech_display}")
        lines.append(f"    F1 Macro:       {f1_m:.4f} ± {f1_s:.4f}")
        lines.append(f"    Accuracy:       {acc_m:.4f} ± {acc_s:.4f}")

        if "f1_macro" in res.fold_metrics:
            folds_f1 = ", ".join(f"{v:.4f}" for v in res.fold_metrics["f1_macro"])
            lines.append(f"    Folds F1 Macro: [{folds_f1}]")
        lines.append("-" * 85)

    lines.append(
        "(Nota metodológica: No se selecciona modelo definitivo ni se evalúa X_test en la Fase 3.5B)."
    )
    lines.append("=" * 85)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Selección reproducible del mejor modelo por validación cruzada (Fase 3.6B)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateEvaluation:
    """Representación normalizada interna de un candidato evaluado con validación cruzada.

    Attributes:
        name: Identificador único del candidato (ej. 'logistic_regression__unigrams').
        f1_macro_mean: Promedio de F1 Macro a través de los folds.
        f1_macro_std: Desviación estándar de F1 Macro a través de los folds.
        accuracy_mean: Promedio de Accuracy a través de los folds.
        cv_folds: Número de pliegues empleados en la evaluación.
        accuracy_std: Desviación estándar de Accuracy a través de los folds.
        configuration: Diccionario con la configuración del modelo o hiperparámetros.
        fold_metrics: Desglose de métricas por pliegue.
        estimator: Instancia del estimador o Pipeline (si está disponible).
    """

    name: str
    f1_macro_mean: float
    f1_macro_std: float
    accuracy_mean: float
    cv_folds: int
    accuracy_std: float = 0.0
    configuration: Mapping[str, Any] = field(default_factory=dict)
    fold_metrics: Mapping[str, Sequence[float]] = field(default_factory=dict)
    estimator: Any | None = None


@dataclass(frozen=True)
class ModelSelectionResult:
    """Resultado estructurado de la selección del mejor modelo candidato por validación cruzada.

    Attributes:
        selected_model_name: Nombre de la combinación o modelo ganador.
        cv_metrics: Métricas del modelo ganador (f1_macro_mean, f1_macro_std, accuracy_mean, etc.).
        configuration: Diccionario con la configuración del modelo ganador.
        justification: Explicación detallada del motivo de selección y desempate.
        ranking: Lista ordenada de diccionarios con todos los candidatos comparados.
        pipeline: Estimador / Pipeline del modelo seleccionado (si está disponible).
        cv_folds: Número de pliegues estratificados utilizados en la comparación.
        decision_criteria: Lista de criterios de decisión aplicados en orden jerárquico.
    """

    selected_model_name: str
    cv_metrics: Mapping[str, float]
    configuration: Mapping[str, Any]
    justification: str
    ranking: Sequence[Mapping[str, Any]]
    pipeline: Pipeline | None = None
    cv_folds: int = 5
    decision_criteria: Sequence[str] = (
        "1. Mayor F1 Macro promedio (f1_macro_mean)",
        "2. Menor desviación estándar de F1 Macro (f1_macro_std)",
        "3. Mayor Accuracy promedio (accuracy_mean)",
        "4. Identificador estable determinista (orden alfabético)",
    )

    def to_dataframe(self) -> pd.DataFrame:
        """Convierte el ranking de candidatos evaluados a un pd.DataFrame estructurado."""
        return pd.DataFrame(self.ranking)

    def summary(self) -> str:
        """Genera el reporte textual de la selección del mejor modelo."""
        return format_model_selection(self)


def validate_candidates_compatibility(
    candidates: Sequence[CandidateEvaluation],
    *,
    expected_cv_folds: int | None = None,
) -> None:
    """Verifica que todos los candidatos provengan del mismo protocolo de evaluación.

    Comprueba:
    - Que la colección de candidatos no esté vacía.
    - Que todos tengan el mismo número de folds de validación cruzada.
    - Que las métricas sean valores numéricos finitos y válidos en [0.0, 1.0].
    - Que la desviación estándar sea no negativa.
    - Que los pliegues detallados (si existen) tengan longitudes idénticas.

    Raises:
        ValueError: Si existe inconsistencia en folds, métricas inválidas o lista vacía.
    """
    if len(candidates) == 0:
        raise ValueError("No se proporcionaron candidatos para la selección del mejor modelo.")

    reference_folds = (
        expected_cv_folds if expected_cv_folds is not None else candidates[0].cv_folds
    )

    for c in candidates:
        if c.cv_folds != reference_folds:
            raise ValueError(
                f"Incompatibilidad en el protocolo de validación cruzada: el candidato '{c.name}' "
                f"fue evaluado con {c.cv_folds} folds, pero se esperaban {reference_folds} folds. "
                "Para una comparación justa todos los candidatos deben usar el mismo esquema de partición."
            )

        for metric_name, val in [
            ("f1_macro_mean", c.f1_macro_mean),
            ("f1_macro_std", c.f1_macro_std),
            ("accuracy_mean", c.accuracy_mean),
            ("accuracy_std", c.accuracy_std),
        ]:
            if val is None or not np.isfinite(val):
                raise ValueError(
                    f"El candidato '{c.name}' contiene un valor no numérico o no finito en '{metric_name}': {val}."
                )

        if not (0.0 <= c.f1_macro_mean <= 1.0):
            raise ValueError(
                f"El candidato '{c.name}' tiene 'f1_macro_mean' fuera del rango [0.0, 1.0]: {c.f1_macro_mean}."
            )
        if not (0.0 <= c.accuracy_mean <= 1.0):
            raise ValueError(
                f"El candidato '{c.name}' tiene 'accuracy_mean' fuera del rango [0.0, 1.0]: {c.accuracy_mean}."
            )
        if c.f1_macro_std < 0.0:
            raise ValueError(
                f"El candidato '{c.name}' tiene una desviación estándar negativa: {c.f1_macro_std}."
            )

        if "f1_macro" in c.fold_metrics and len(c.fold_metrics["f1_macro"]) != reference_folds:
            raise ValueError(
                f"El candidato '{c.name}' reporta {len(c.fold_metrics['f1_macro'])} pliegues en 'fold_metrics', "
                f"pero se esperaban {reference_folds} folds."
            )


def _candidate_sort_key(c: CandidateEvaluation) -> tuple[float, float, float, str]:
    """Genera la clave de ordenamiento jerárquica para la selección del modelo.

    Criterios en orden de prioridad:
    1. Mayor F1 Macro promedio (-f1_macro_mean para orden ascendente)
    2. Menor desviación estándar (+f1_macro_std para orden ascendente)
    3. Mayor Accuracy promedio (-accuracy_mean para orden ascendente)
    4. Identificador estable determinista (orden alfabético del nombre)
    """
    return (
        -float(c.f1_macro_mean),
        float(c.f1_macro_std),
        -float(c.accuracy_mean),
        str(c.name),
    )


def _build_justification(
    winner: CandidateEvaluation,
    runner_up: CandidateEvaluation | None,
    total_candidates: int,
) -> str:
    """Genera la justificación metodológica de la selección del modelo ganador."""
    if runner_up is None or total_candidates <= 1:
        return (
            f"El modelo '{winner.name}' fue seleccionado como único candidato evaluado "
            f"con F1 Macro promedio de {winner.f1_macro_mean:.4f} (± {winner.f1_macro_std:.4f}) "
            f"y Accuracy de {winner.accuracy_mean:.4f}."
        )

    f1_diff = winner.f1_macro_mean - runner_up.f1_macro_mean
    std_diff = runner_up.f1_macro_std - winner.f1_macro_std
    acc_diff = winner.accuracy_mean - runner_up.accuracy_mean

    if abs(f1_diff) > 1e-9:
        return (
            f"El modelo '{winner.name}' fue seleccionado por obtener el mayor F1 Macro promedio "
            f"({winner.f1_macro_mean:.4f} ± {winner.f1_macro_std:.4f}) frente a {runner_up.f1_macro_mean:.4f} "
            f"(± {runner_up.f1_macro_std:.4f}) de '{runner_up.name}', superándolo por {f1_diff:+.4f} puntos."
        )

    if abs(std_diff) > 1e-9:
        return (
            f"El modelo '{winner.name}' empató en F1 Macro promedio ({winner.f1_macro_mean:.4f}) con "
            f"'{runner_up.name}', pero fue seleccionado por el primer criterio de desempate al presentar "
            f"una menor desviación estándar ({winner.f1_macro_std:.4f} frente a {runner_up.f1_macro_std:.4f}), "
            f"demostrando mayor estabilidad y menor variabilidad entre las {winner.cv_folds} particiones."
        )

    if abs(acc_diff) > 1e-9:
        return (
            f"El modelo '{winner.name}' empató en F1 Macro ({winner.f1_macro_mean:.4f}) y desviación estándar "
            f"({winner.f1_macro_std:.4f}) con '{runner_up.name}', pero fue seleccionado por el segundo criterio "
            f"de desempate al obtener un mayor Accuracy promedio ({winner.accuracy_mean:.4f} frente a "
            f"{runner_up.accuracy_mean:.4f})."
        )

    return (
        f"El modelo '{winner.name}' empató en todas las métricas cuantitativas de validación cruzada "
        f"(F1 Macro: {winner.f1_macro_mean:.4f} ± {winner.f1_macro_std:.4f}, Accuracy: {winner.accuracy_mean:.4f}) "
        f"con '{runner_up.name}'. Fue seleccionado mediante el desempate final determinista basado en el "
        f"orden alfabético del identificador estable."
    )


def _normalize_candidate_evaluations(
    candidates: Any,
    default_cv_folds: int = 5,
) -> list[CandidateEvaluation]:
    """Convierte cualquier estructura de resultados previa a una lista de CandidateEvaluation."""
    if candidates is None:
        raise ValueError("Los resultados de candidatos no pueden ser None.")

    # Caso 1: NgramComparisonResult
    if isinstance(candidates, NgramComparisonResult):
        normalized: list[CandidateEvaluation] = []
        for name, res in candidates.results.items():
            f1_m = float(res.metrics.get("f1_macro_mean", 0.0))
            f1_s = float(res.metrics.get("f1_macro_std", 0.0))
            acc_m = float(res.metrics.get("accuracy_mean", 0.0))
            acc_s = float(res.metrics.get("accuracy_std", 0.0))
            parts = name.split("__")
            rep = parts[1] if len(parts) > 1 else "unigrams"
            ngram_range = candidates.ngram_configs.get(rep, (1, 1))
            normalized.append(
                CandidateEvaluation(
                    name=name,
                    f1_macro_mean=f1_m,
                    f1_macro_std=f1_s,
                    accuracy_mean=acc_m,
                    accuracy_std=acc_s,
                    cv_folds=candidates.cv_folds,
                    configuration={
                        "algorithm": parts[0],
                        "representation": rep,
                        "ngram_range": ngram_range,
                    },
                    fold_metrics=dict(res.fold_metrics),
                )
            )
        return normalized

    # Caso 2: PreprocessingComparisonResult
    if isinstance(candidates, PreprocessingComparisonResult):
        normalized = []
        for name, res in candidates.results.items():
            f1_m = float(res.metrics.get("f1_macro_mean", 0.0))
            f1_s = float(res.metrics.get("f1_macro_std", 0.0))
            acc_m = float(res.metrics.get("accuracy_mean", 0.0))
            acc_s = float(res.metrics.get("accuracy_std", 0.0))
            parts = name.split("__")
            tech = parts[1] if len(parts) > 1 else "none"
            normalized.append(
                CandidateEvaluation(
                    name=name,
                    f1_macro_mean=f1_m,
                    f1_macro_std=f1_s,
                    accuracy_mean=acc_m,
                    accuracy_std=acc_s,
                    cv_folds=candidates.cv_folds,
                    configuration={
                        "algorithm": parts[0],
                        "technique": tech,
                        "ngram_range": candidates.ngram_range,
                    },
                    fold_metrics=dict(res.fold_metrics),
                )
            )
        return normalized

    # Caso 3: Mapping (dict de GridSearchResult, ExperimentResult o dicts)
    if isinstance(candidates, Mapping):
        normalized = []
        for name, item in candidates.items():
            if isinstance(item, GridSearchResult):
                mean_key = f"mean_test_{item.primary_metric}"
                std_key = f"std_test_{item.primary_metric}"
                best_idx = 0
                if mean_key in item.cv_results and len(item.cv_results[mean_key]) > 0:
                    best_idx = int(np.argmax(item.cv_results[mean_key]))
                f1_s = (
                    float(item.cv_results[std_key][best_idx])
                    if std_key in item.cv_results
                    else 0.0
                )
                acc_m = (
                    float(item.cv_results["mean_test_accuracy"][best_idx])
                    if "mean_test_accuracy" in item.cv_results
                    else float(item.best_score)
                )
                normalized.append(
                    CandidateEvaluation(
                        name=name,
                        f1_macro_mean=float(item.best_score),
                        f1_macro_std=f1_s,
                        accuracy_mean=acc_m,
                        cv_folds=default_cv_folds,
                        configuration=dict(item.best_params),
                        estimator=item.best_estimator,
                    )
                )
            elif isinstance(item, CandidateEvaluation):
                normalized.append(item)
            elif isinstance(item, ExperimentResult):
                f1_m = float(item.metrics.get("f1_macro_mean", item.metrics.get("f1_macro", 0.0)))
                f1_s = float(item.metrics.get("f1_macro_std", 0.0))
                acc_m = float(item.metrics.get("accuracy_mean", item.metrics.get("accuracy", 0.0)))
                acc_s = float(item.metrics.get("accuracy_std", 0.0))
                folds_count = (
                    len(item.fold_metrics["f1_macro"])
                    if "f1_macro" in item.fold_metrics
                    else default_cv_folds
                )
                normalized.append(
                    CandidateEvaluation(
                        name=name,
                        f1_macro_mean=f1_m,
                        f1_macro_std=f1_s,
                        accuracy_mean=acc_m,
                        accuracy_std=acc_s,
                        cv_folds=folds_count,
                        fold_metrics=dict(item.fold_metrics),
                    )
                )
            elif isinstance(item, Mapping):
                f1_m = float(item.get("f1_macro_mean", item.get("f1_macro", 0.0)))
                f1_s = float(item.get("f1_macro_std", item.get("std", 0.0)))
                acc_m = float(item.get("accuracy_mean", item.get("accuracy", 0.0)))
                acc_s = float(item.get("accuracy_std", 0.0))
                folds_count = int(item.get("cv_folds", default_cv_folds))
                config = item.get("configuration", item.get("params", {}))
                est = item.get("pipeline", item.get("estimator", None))
                normalized.append(
                    CandidateEvaluation(
                        name=name,
                        f1_macro_mean=f1_m,
                        f1_macro_std=f1_s,
                        accuracy_mean=acc_m,
                        accuracy_std=acc_s,
                        cv_folds=folds_count,
                        configuration=dict(config) if isinstance(config, Mapping) else {},
                        estimator=est,
                    )
                )
            else:
                raise TypeError(
                    f"Tipo de resultado no reconocido para el candidato '{name}': {type(item).__name__}."
                )
        return normalized

    # Caso 4: Sequence (list o tuple de ExperimentResult, GridSearchResult o dicts)
    if isinstance(candidates, Sequence):
        normalized = []
        for i, item in enumerate(candidates):
            if isinstance(item, CandidateEvaluation):
                normalized.append(item)
            elif isinstance(item, ExperimentResult):
                f1_m = float(item.metrics.get("f1_macro_mean", item.metrics.get("f1_macro", 0.0)))
                f1_s = float(item.metrics.get("f1_macro_std", 0.0))
                acc_m = float(item.metrics.get("accuracy_mean", item.metrics.get("accuracy", 0.0)))
                acc_s = float(item.metrics.get("accuracy_std", 0.0))
                folds_count = (
                    len(item.fold_metrics["f1_macro"])
                    if "f1_macro" in item.fold_metrics
                    else default_cv_folds
                )
                normalized.append(
                    CandidateEvaluation(
                        name=item.name,
                        f1_macro_mean=f1_m,
                        f1_macro_std=f1_s,
                        accuracy_mean=acc_m,
                        accuracy_std=acc_s,
                        cv_folds=folds_count,
                        fold_metrics=dict(item.fold_metrics),
                    )
                )
            elif isinstance(item, GridSearchResult):
                mean_key = f"mean_test_{item.primary_metric}"
                std_key = f"std_test_{item.primary_metric}"
                best_idx = 0
                if mean_key in item.cv_results and len(item.cv_results[mean_key]) > 0:
                    best_idx = int(np.argmax(item.cv_results[mean_key]))
                f1_s = (
                    float(item.cv_results[std_key][best_idx])
                    if std_key in item.cv_results
                    else 0.0
                )
                acc_m = (
                    float(item.cv_results["mean_test_accuracy"][best_idx])
                    if "mean_test_accuracy" in item.cv_results
                    else float(item.best_score)
                )
                normalized.append(
                    CandidateEvaluation(
                        name=item.model_name,
                        f1_macro_mean=float(item.best_score),
                        f1_macro_std=f1_s,
                        accuracy_mean=acc_m,
                        cv_folds=default_cv_folds,
                        configuration=dict(item.best_params),
                        estimator=item.best_estimator,
                    )
                )
            elif isinstance(item, Mapping):
                c_name = str(item.get("name", f"candidate_{i+1}"))
                f1_m = float(item.get("f1_macro_mean", item.get("f1_macro", 0.0)))
                f1_s = float(item.get("f1_macro_std", item.get("std", 0.0)))
                acc_m = float(item.get("accuracy_mean", item.get("accuracy", 0.0)))
                acc_s = float(item.get("accuracy_std", 0.0))
                folds_count = int(item.get("cv_folds", default_cv_folds))
                config = item.get("configuration", item.get("params", {}))
                est = item.get("pipeline", item.get("estimator", None))
                normalized.append(
                    CandidateEvaluation(
                        name=c_name,
                        f1_macro_mean=f1_m,
                        f1_macro_std=f1_s,
                        accuracy_mean=acc_m,
                        accuracy_std=acc_s,
                        cv_folds=folds_count,
                        configuration=dict(config) if isinstance(config, Mapping) else {},
                        estimator=est,
                    )
                )
            else:
                raise TypeError(
                    f"Elemento no reconocido en la secuencia de candidatos (índice {i}): {type(item).__name__}."
                )
        return normalized

    raise TypeError(
        f"Se esperaba una colección de candidatos (Sequence, Mapping, NgramComparisonResult o "
        f"PreprocessingComparisonResult), pero se recibió: {type(candidates).__name__}."
    )


def select_best_model(
    candidates: (
        Sequence[ExperimentResult]
        | Mapping[str, ExperimentResult]
        | NgramComparisonResult
        | PreprocessingComparisonResult
        | Mapping[str, GridSearchResult]
        | Sequence[GridSearchResult]
        | Sequence[Mapping[str, Any]]
    ),
    *,
    expected_cv_folds: int | None = None,
) -> ModelSelectionResult:
    """Selecciona de forma reproducible la mejor configuración según métricas de validación cruzada.

    Aplica de manera determinista los criterios contractuales jerárquicos:
    1. Criterio principal: Mayor F1 Macro promedio (f1_macro_mean).
    2. Primer desempate: Menor desviación estándar de F1 Macro (f1_macro_std).
    3. Segundo desempate: Mayor Accuracy promedio (accuracy_mean).
    4. Desempate final: Identificador estable determinista (orden alfabético).

    Verifica que todos los candidatos comparados correspondan al mismo número de pliegues
    y protocolo de validación cruzada, rechazando evaluaciones incompatibles.

    Utiliza EXCLUSIVAMENTE los resultados de entrenamiento en CV. Nunca accede a X_test ni y_test.

    Args:
        candidates: Resultados de validación cruzada procedentes de run_cross_validation,
            compare_ngram_representations, compare_preprocessing_techniques o run_grid_search.
        expected_cv_folds: Número contractual esperado de folds (opcional; si se omite,
            se deduce del primer candidato).

    Returns:
        Instancia estructurada de ModelSelectionResult con el ganador, métricas,
        configuración, justificación analítica y ranking completo.

    Raises:
        ValueError: Si los candidatos están vacíos, incompatibles en folds o tienen métricas no finitas.
        TypeError: Si la estructura de entrada es de un tipo no admitido.
    """
    normalized = _normalize_candidate_evaluations(
        candidates,
        default_cv_folds=expected_cv_folds if expected_cv_folds is not None else 5,
    )

    validate_candidates_compatibility(normalized, expected_cv_folds=expected_cv_folds)

    # Ordenamiento determinista con clave jerárquica
    ranked = sorted(normalized, key=_candidate_sort_key)

    winner = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None

    justification = _build_justification(winner, runner_up, len(ranked))

    # Construir el ranking exportable
    ranking_data: list[dict[str, Any]] = []
    for rank_idx, c in enumerate(ranked, start=1):
        ranking_data.append(
            {
                "rank": rank_idx,
                "name": c.name,
                "f1_macro_mean": c.f1_macro_mean,
                "f1_macro_std": c.f1_macro_std,
                "accuracy_mean": c.accuracy_mean,
                "accuracy_std": c.accuracy_std,
                "cv_folds": c.cv_folds,
                "configuration": dict(c.configuration),
            }
        )

    winner_metrics = {
        "f1_macro_mean": winner.f1_macro_mean,
        "f1_macro_std": winner.f1_macro_std,
        "accuracy_mean": winner.accuracy_mean,
        "accuracy_std": winner.accuracy_std,
    }

    winner_pipeline: Pipeline | None = (
        winner.estimator if isinstance(winner.estimator, Pipeline) else None
    )

    return ModelSelectionResult(
        selected_model_name=winner.name,
        cv_metrics=winner_metrics,
        configuration=dict(winner.configuration),
        justification=justification,
        ranking=tuple(ranking_data),
        pipeline=winner_pipeline,
        cv_folds=winner.cv_folds,
    )


# Alias sinónimo
select_best_candidate = select_best_model


def format_model_selection(selection: ModelSelectionResult) -> str:
    """Genera una presentación formateada y detallada de la selección del modelo ganador."""
    lines = [
        "=" * 85,
        "SELECCIÓN REPRODUCIBLE DEL MEJOR MODELO POR VALIDACIÓN CRUZADA",
        "=" * 85,
        f"Pliegues de validación cruzada: {selection.cv_folds} (estratificados)",
        f"Modelo ganador seleccionado:    {selection.selected_model_name}",
        f"F1 Macro promedio:              {selection.cv_metrics.get('f1_macro_mean', 0.0):.4f} ± {selection.cv_metrics.get('f1_macro_std', 0.0):.4f}",
        f"Accuracy promedio:              {selection.cv_metrics.get('accuracy_mean', 0.0):.4f} ± {selection.cv_metrics.get('accuracy_std', 0.0):.4f}",
        "",
        "Configuración del modelo seleccionado:",
    ]
    if selection.configuration:
        for k, v in selection.configuration.items():
            lines.append(f"  - {k}: {v}")
    else:
        lines.append("  (Configuración por defecto del pipeline)")

    lines.append("")
    lines.append("Justificación metodológica:")
    lines.append(f"  {selection.justification}")
    lines.append("")
    lines.append("Criterios de desempate aplicados:")
    for crit in selection.decision_criteria:
        lines.append(f"  {crit}")
    lines.append("")
    lines.append("Ranking completo de candidatos evaluados:")
    lines.append("-" * 85)

    for row in selection.ranking:
        rank = row.get("rank", "-")
        name = row.get("name", "desconocido")
        f1_m = row.get("f1_macro_mean", 0.0)
        f1_s = row.get("f1_macro_std", 0.0)
        acc_m = row.get("accuracy_mean", 0.0)
        lines.append(f"[{rank}] {name:<40} | F1: {f1_m:.4f} ± {f1_s:.4f} | Acc: {acc_m:.4f}")

    lines.append("-" * 85)
    lines.append(
        "(Nota metodológica: La selección se basa exclusivamente en CV sobre datos de entrenamiento; "
        "el conjunto X_test permanece reservado para la evaluación final)."
    )
    lines.append("=" * 85)
    return "\n".join(lines)
