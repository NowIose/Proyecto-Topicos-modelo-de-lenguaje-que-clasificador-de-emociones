"""Carga y validación del dataset para el módulo de Machine Learning.

Este módulo implementa la lectura de archivos CSV y las reglas de validación
de datos que garantizan que el dataset cumpla el contrato mínimo para el
futuro entrenamiento del clasificador de sentimientos en español.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from .config import (
    EXPECTED_LABELS,
    PROJECT_ROOT,
    DatasetConfig,
    MLConfig,
    load_config,
)


class DatasetValidationError(ValueError):
    """Excepción lanzada cuando el dataset no cumple las restricciones requeridas."""


def load_csv(path: str | Path) -> pd.DataFrame:
    """Carga un archivo CSV utilizando pandas con codificación UTF-8.

    Args:
        path: Ruta al archivo CSV.

    Returns:
        DataFrame con los datos crudos del CSV.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si la ruta especificada corresponde a un directorio.
        DatasetValidationError: Si el archivo está vacío o tiene formato inválido.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"El archivo del dataset no existe: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"La ruta del dataset no es un archivo válido: {file_path}")

    try:
        return pd.read_csv(file_path, encoding="utf-8")
    except pd.errors.EmptyDataError as exc:
        raise DatasetValidationError(
            f"El archivo CSV está vacío o sin datos legibles: {file_path}"
        ) from exc
    except pd.errors.ParserError as exc:
        raise DatasetValidationError(
            f"Error de formato al analizar el archivo CSV '{file_path}': {exc}"
        ) from exc


def validate_columns(
    df: pd.DataFrame,
    text_column: str = "texto",
    target_column: str = "sentimiento",
) -> None:
    """Valida la presencia de las columnas requeridas y que el DataFrame contenga filas.

    Args:
        df: DataFrame a validar.
        text_column: Nombre de la columna que contiene el texto.
        target_column: Nombre de la columna objetivo con las etiquetas.

    Raises:
        DatasetValidationError: Si la estructura o columnas son inválidas.
    """
    if not isinstance(df, pd.DataFrame):
        raise DatasetValidationError(
            f"Se esperaba un objeto pandas DataFrame, pero se recibió: {type(df).__name__}."
        )

    if df.empty or len(df) == 0:
        raise DatasetValidationError("El dataset está vacío (0 filas).")

    missing = [col for col in (text_column, target_column) if col not in df.columns]
    if missing:
        raise DatasetValidationError(
            f"Faltan columnas requeridas en el dataset: {missing}. "
            f"Columnas presentes: {list(df.columns)}."
        )


def validate_no_nulls(
    df: pd.DataFrame,
    text_column: str = "texto",
    target_column: str = "sentimiento",
) -> None:
    """Detecta y rechaza valores nulos en las columnas clave.

    Args:
        df: DataFrame a validar.
        text_column: Nombre de la columna de texto.
        target_column: Nombre de la columna objetivo.

    Raises:
        DatasetValidationError: Si existen valores nulos o NaN.
    """
    null_text = int(df[text_column].isna().sum())
    null_target = int(df[target_column].isna().sum())

    if null_text > 0 and null_target > 0:
        raise DatasetValidationError(
            f"El dataset contiene valores nulos: {null_text} en '{text_column}' "
            f"y {null_target} en '{target_column}'."
        )
    if null_text > 0:
        raise DatasetValidationError(
            f"El dataset contiene {null_text} valores nulos en la columna '{text_column}'."
        )
    if null_target > 0:
        raise DatasetValidationError(
            f"El dataset contiene {null_target} valores nulos en la columna '{target_column}'."
        )


def validate_no_empty_texts(
    df: pd.DataFrame,
    text_column: str = "texto",
) -> None:
    """Detecta y rechaza textos vacíos, con solo espacios en blanco o tipos no texto.

    Args:
        df: DataFrame a validar.
        text_column: Nombre de la columna de texto.

    Raises:
        DatasetValidationError: Si hay tipos no texto o cadenas vacías.
    """
    non_str_mask = ~df[text_column].apply(lambda x: isinstance(x, str))
    if non_str_mask.any():
        invalid_types = sorted(
            {type(x).__name__ for x in df.loc[non_str_mask, text_column]}
        )
        raise DatasetValidationError(
            f"La columna '{text_column}' debe contener únicamente texto (str). "
            f"Se encontraron tipos incompatibles: {invalid_types}."
        )

    empty_mask = df[text_column].str.strip() == ""
    empty_count = int(empty_mask.sum())
    if empty_count > 0:
        raise DatasetValidationError(
            f"El dataset contiene {empty_count} fila(s) con texto vacío o solo espacios en blanco."
        )


def validate_no_duplicates(
    df: pd.DataFrame,
    text_column: str = "texto",
) -> None:
    """Detecta y rechaza textos duplicados en el dataset.

    Se normalizan los extremos de espacios para asegurar que variantes con
    espacios accidentales sean reconocidas como duplicados.

    Args:
        df: DataFrame a validar.
        text_column: Nombre de la columna de texto.

    Raises:
        DatasetValidationError: Si se detectan textos repetidos.
    """
    normalized_texts = df[text_column].astype(str).str.strip()
    duplicated_mask = normalized_texts.duplicated(keep="first")
    duplicate_count = int(duplicated_mask.sum())

    if duplicate_count > 0:
        examples = normalized_texts[duplicated_mask].head(3).tolist()
        raise DatasetValidationError(
            f"El dataset contiene {duplicate_count} texto(s) duplicado(s). "
            f"Ejemplos: {examples}."
        )


def validate_labels(
    df: pd.DataFrame,
    target_column: str = "sentimiento",
    expected_labels: Sequence[str] = EXPECTED_LABELS,
) -> None:
    """Valida que existan únicamente las etiquetas permitidas y que ambas clases estén presentes.

    Args:
        df: DataFrame a validar.
        target_column: Nombre de la columna objetivo.
        expected_labels: Secuencia de etiquetas esperadas contractuales.

    Raises:
        DatasetValidationError: Si hay etiquetas no permitidas o faltan clases.
    """
    unique_labels = set(df[target_column].unique())
    allowed = set(expected_labels)

    invalid_labels = unique_labels - allowed
    if invalid_labels:
        raise DatasetValidationError(
            f"El dataset contiene etiquetas no permitidas: {sorted(str(lbl) for lbl in invalid_labels)}. "
            f"Solo se aceptan: {list(expected_labels)}."
        )

    missing_labels = allowed - unique_labels
    if missing_labels:
        raise DatasetValidationError(
            f"El dataset no contiene ejemplos de todas las clases requeridas. "
            f"Clases faltantes: {sorted(missing_labels)}."
        )


def validate_dataset(
    df: pd.DataFrame,
    text_column: str = "texto",
    target_column: str = "sentimiento",
    expected_labels: Sequence[str] = EXPECTED_LABELS,
) -> pd.DataFrame:
    """Ejecuta todas las validaciones contractuales sobre el DataFrame.

    Aplica de manera secuencial:
    1. Existencia del DataFrame y presencia de columnas mínimas.
    2. Detección de valores nulos en texto y sentimiento.
    3. Detección de textos vacíos, con solo espacios o tipos incompatibles.
    4. Detección de textos duplicados.
    5. Aceptación exclusiva de etiquetas válidas ('positivo', 'negativo').
    6. Comprobación de que ambas clases estén presentes en el dataset.

    Args:
        df: DataFrame a validar.
        text_column: Nombre de la columna de texto.
        target_column: Nombre de la columna de sentimiento.
        expected_labels: Clases permitidas requeridas.

    Returns:
        El DataFrame validado si cumple con todas las restricciones.

    Raises:
        DatasetValidationError: Si cualquier regla de validación falla.
    """
    validate_columns(df, text_column=text_column, target_column=target_column)
    validate_no_nulls(df, text_column=text_column, target_column=target_column)
    validate_no_empty_texts(df, text_column=text_column)
    validate_no_duplicates(df, text_column=text_column)
    validate_labels(df, target_column=target_column, expected_labels=expected_labels)
    return df


def load_dataset(
    path: str | Path | DatasetConfig | MLConfig | None = None,
    *,
    text_column: str | None = None,
    target_column: str | None = None,
    expected_labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Carga y valida un dataset en formato CSV para Machine Learning.

    Si no se indica una ruta explícita ni configuración, se resuelve la ruta
    definida en la configuración contractual del proyecto (`config/ml.yaml`).

    Args:
        path: Ruta al archivo CSV, objeto `DatasetConfig`, `MLConfig`, o None.
        text_column: Nombre opcional de la columna de texto (sobreescribe la config).
        target_column: Nombre opcional de la columna objetivo (sobreescribe la config).
        expected_labels: Etiquetas contractuales esperadas (sobreescribe la config).

    Returns:
        DataFrame validado y listo para la futura etapa de entrenamiento.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        DatasetValidationError: Si los datos no cumplen con alguna validación.
    """
    resolved_path: Path
    resolved_text_column = text_column
    resolved_target_column = target_column
    resolved_labels = expected_labels

    if isinstance(path, MLConfig):
        cfg = path.dataset
        resolved_path = cfg.path
        resolved_text_column = resolved_text_column or cfg.text_column
        resolved_target_column = resolved_target_column or cfg.target_column
        resolved_labels = resolved_labels or cfg.labels
    elif isinstance(path, DatasetConfig):
        resolved_path = path.path
        resolved_text_column = resolved_text_column or path.text_column
        resolved_target_column = resolved_target_column or path.target_column
        resolved_labels = resolved_labels or path.labels
    elif path is None:
        cfg = load_config().dataset
        resolved_path = cfg.path
        resolved_text_column = resolved_text_column or cfg.text_column
        resolved_target_column = resolved_target_column or cfg.target_column
        resolved_labels = resolved_labels or cfg.labels
    else:
        candidate_path = Path(path)
        if not candidate_path.is_absolute():
            if not candidate_path.exists() and (PROJECT_ROOT / candidate_path).exists():
                candidate_path = PROJECT_ROOT / candidate_path
        resolved_path = candidate_path

    resolved_text_column = resolved_text_column or "texto"
    resolved_target_column = resolved_target_column or "sentimiento"
    resolved_labels = resolved_labels or EXPECTED_LABELS

    df = load_csv(resolved_path)
    return validate_dataset(
        df,
        text_column=resolved_text_column,
        target_column=resolved_target_column,
        expected_labels=resolved_labels,
    )
