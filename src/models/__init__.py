"""Herramientas de Machine Learning para el clasificador de sentimientos.

El paquete contiene por ahora los contratos y la configuración de la futura
fase de experimentación. No entrena ni publica modelos todavía.
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
]
