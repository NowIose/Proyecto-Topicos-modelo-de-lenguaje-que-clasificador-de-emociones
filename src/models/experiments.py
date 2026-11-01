"""Construcción, entrenamiento, predicción y validación cruzada para modelos ML."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
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
