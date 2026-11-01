"""División estratificada y reproducible de datasets para Machine Learning."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import EXPECTED_LABELS, DatasetConfig, MLConfig, load_config
from .dataset import (
    DatasetValidationError,
    load_dataset,
    validate_dataset,
)


class DatasetSplit(NamedTuple):
    """Contenedor inmutable para los subconjuntos de entrenamiento y prueba.

    Permite tanto el desempaquetado posicional:
        X_train, X_test, y_train, y_test = split_dataset(df)
    como el acceso por atributo:
        split.X_train, split.y_test
    """

    X_train: pd.Series
    X_test: pd.Series
    y_train: pd.Series
    y_test: pd.Series


def validate_split_sample_counts(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    expected_labels: Sequence[str],
) -> None:
    """Verifica que cada clase cuente con suficientes muestras para la división.

    Garantiza que:
    1. Todas las clases contractuales estén presentes en los datos.
    2. Cada clase tenga al menos 2 muestras para posibilitar la partición.
    3. Para la proporción `test_size` dada, cada clase tenga al menos 1 muestra
       tanto en el conjunto de entrenamiento como en el de prueba.

    Args:
        df: DataFrame validado.
        target_column: Nombre de la columna objetivo.
        test_size: Fracción asignada al conjunto de prueba (0 < test_size < 1).
        expected_labels: Clases requeridas por el contrato.

    Raises:
        DatasetValidationError: Si alguna clase no tiene suficientes muestras.
    """
    class_counts = df[target_column].value_counts()

    # Primero: verificar presencia de todas las clases contractuales
    for label in expected_labels:
        if label not in class_counts or class_counts[label] == 0:
            raise DatasetValidationError(
                f"La clase contractual '{label}' no está presente en los datos."
            )

    # Segundo: verificar suficiencia de muestras para estratificación
    for label in expected_labels:
        count = int(class_counts[label])
        if count < 2:
            raise DatasetValidationError(
                f"Muestras insuficientes para la clase '{label}': se encontraron {count}, "
                "pero se requieren al menos 2 muestras para realizar una división estratificada."
            )

        expected_test_count = round(count * test_size)
        expected_train_count = count - expected_test_count
        if expected_test_count < 1 or expected_train_count < 1:
            raise DatasetValidationError(
                f"Muestras insuficientes para la clase '{label}' ({count} muestras) con test_size={test_size}: "
                f"se estiman {expected_train_count} muestras en entrenamiento y {expected_test_count} en prueba. "
                "Se requiere al menos 1 muestra de cada clase en ambos conjuntos."
            )


def validate_no_overlap(X_train: pd.Series, X_test: pd.Series) -> None:
    """Garantiza la ausencia de reseñas superpuestas entre entrenamiento y prueba.

    Args:
        X_train: Serie de textos del conjunto de entrenamiento.
        X_test: Serie de textos del conjunto de prueba.

    Raises:
        DatasetValidationError: Si se detecta alguna reseña idéntica en ambos conjuntos.
    """
    train_texts = set(X_train.astype(str).str.strip())
    test_texts = set(X_test.astype(str).str.strip())
    overlap = train_texts.intersection(test_texts)

    if overlap:
        examples = sorted(list(overlap))[:3]
        raise DatasetValidationError(
            f"Se detectó superposición de {len(overlap)} reseña(s) entre entrenamiento "
            f"y prueba. Ejemplos: {examples}."
        )


def split_dataset(
    data: pd.DataFrame | str | Path | DatasetConfig | MLConfig | None = None,
    *,
    test_size: float | None = None,
    random_seed: int | None = None,
    text_column: str | None = None,
    target_column: str | None = None,
    expected_labels: Sequence[str] | None = None,
    config: MLConfig | None = None,
) -> DatasetSplit:
    """Divide un dataset en subconjuntos de entrenamiento y prueba de forma estratificada.

    Por defecto toma los parámetros desde la configuración central del proyecto
    (`config/ml.yaml`): 80% entrenamiento, 20% prueba y semilla 42.

    Aplica validaciones estrictas:
    - Validación del dataset (columnas, nulos, textos vacíos, duplicados, etiquetas).
    - Muestras suficientes por clase para mantener estratificación en train y test.
    - Ausencia total de superposición de reseñas entre train y test.

    Args:
        data: DataFrame pre-cargado, ruta al archivo CSV, configuración o None (resuelve config).
        test_size: Proporción para el conjunto de prueba (por defecto toma de config: 0.20).
        random_seed: Semilla para reproducibilidad (por defecto toma de config: 42).
        text_column: Nombre de columna de texto (por defecto toma de config: 'texto').
        target_column: Nombre de columna objetivo (por defecto toma de config: 'sentimiento').
        expected_labels: Clases permitidas (por defecto toma de config: ('positivo', 'negativo')).
        config: Configuración `MLConfig` opcional para sobrescribir los valores por defecto.

    Returns:
        DatasetSplit conteniendo (X_train, X_test, y_train, y_test).

    Raises:
        ValueError: Si `test_size` no está en el rango (0, 1).
        DatasetValidationError: Si los datos o la partición violan las restricciones.
        FileNotFoundError: Si se especifica o resuelve una ruta CSV inexistente.
    """
    active_config = config
    if active_config is None:
        if isinstance(data, MLConfig):
            active_config = data
        else:
            active_config = load_config()

    resolved_test_size = (
        test_size if test_size is not None else active_config.experiment.test_size
    )
    resolved_seed = (
        random_seed if random_seed is not None else active_config.experiment.random_seed
    )
    resolved_text_col = (
        text_column if text_column is not None else active_config.dataset.text_column
    )
    resolved_target_col = (
        target_column
        if target_column is not None
        else active_config.dataset.target_column
    )
    resolved_labels = (
        expected_labels
        if expected_labels is not None
        else active_config.dataset.labels
    )

    if not 0.0 < resolved_test_size < 1.0:
        raise ValueError(
            f"'test_size' debe ser un número estrictamente entre 0 y 1. Se recibió: {resolved_test_size}."
        )

    # Cargar o validar DataFrame según la fuente proporcionada
    if isinstance(data, pd.DataFrame):
        df = validate_dataset(
            data,
            text_column=resolved_text_col,
            target_column=resolved_target_col,
            expected_labels=resolved_labels,
        )
    else:
        df = load_dataset(
            data,
            text_column=resolved_text_col,
            target_column=resolved_target_col,
            expected_labels=resolved_labels,
        )

    # Validar suficiencia de muestras por clase antes de la división
    validate_split_sample_counts(
        df=df,
        target_column=resolved_target_col,
        test_size=resolved_test_size,
        expected_labels=resolved_labels,
    )

    X = df[resolved_text_col]
    y = df[resolved_target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=resolved_test_size,
        random_state=resolved_seed,
        stratify=y,
    )

    # Validar post-condiciones: presencia de ambas clases en ambos subconjuntos
    for label in resolved_labels:
        if label not in y_train.values or label not in y_test.values:
            raise DatasetValidationError(
                f"La partición estratificada no incluyó la clase '{label}' en ambos subconjuntos."
            )

    # Validar post-condición: cero superposición de textos
    validate_no_overlap(X_train, X_test)

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
