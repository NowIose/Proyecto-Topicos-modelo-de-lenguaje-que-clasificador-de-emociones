"""Herramientas de Machine Learning para el clasificador de sentimientos.

El paquete contiene contratos, configuración, validación de dataset, partición
estratificada y los pipelines candidatos de clasificación (Regresión Logística
y Multinomial Naive Bayes) con TF-IDF.
"""

from .config import DEFAULT_CONFIG_PATH, MLConfig, load_config
from .dataset import (
    DatasetValidationError,
    load_csv,
    load_dataset,
    validate_columns,
    validate_dataset,
    validate_labels,
    validate_no_duplicates,
    validate_no_empty_texts,
    validate_no_nulls,
)
from .experiments import (
    ExperimentResult,
    ExperimentSpec,
    build_logistic_regression_pipeline,
    build_multinomial_nb_pipeline,
    build_tfidf_vectorizer,
    get_candidate_pipelines,
    predict,
    predict_pipeline,
    predict_proba,
    predict_proba_pipeline,
    predict_probabilities,
    run_experiments,
    train_pipeline,
)
from .split import DatasetSplit, split_dataset

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "MLConfig",
    "load_config",
    "DatasetValidationError",
    "load_csv",
    "load_dataset",
    "validate_columns",
    "validate_dataset",
    "validate_labels",
    "validate_no_duplicates",
    "validate_no_empty_texts",
    "validate_no_nulls",
    "DatasetSplit",
    "split_dataset",
    "ExperimentSpec",
    "ExperimentResult",
    "run_experiments",
    "build_tfidf_vectorizer",
    "build_logistic_regression_pipeline",
    "build_multinomial_nb_pipeline",
    "get_candidate_pipelines",
    "train_pipeline",
    "predict",
    "predict_pipeline",
    "predict_proba",
    "predict_proba_pipeline",
    "predict_probabilities",
]
