"""Pruebas unitarias para la división estratificada y reproducible del dataset ML."""

from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from src.models.config import load_config
from src.models.dataset import DatasetValidationError
from src.models.split import (
    DatasetSplit,
    split_dataset,
    validate_no_overlap,
    validate_split_sample_counts,
)


@pytest.fixture
def balanced_dataframe_10() -> pd.DataFrame:
    """Dataset sintético balanceado de 10 muestras (5 positivas, 5 negativas)."""
    return pd.DataFrame(
        {
            "texto": [
                "Excelente producto y servicio impecable",
                "Muy buena atención recibida",
                "Me encantó la rapidez del envío",
                "Experiencia genial y recomendable",
                "Totalmente satisfecho con la compra",
                "Pésima atención, no lo recomiendo",
                "Muy mala calidad del producto",
                "Llegó tarde y roto el paquete",
                "No sirve para nada este servicio",
                "Terrible experiencia con el vendedor",
            ],
            "sentimiento": [
                "positivo",
                "positivo",
                "positivo",
                "positivo",
                "positivo",
                "negativo",
                "negativo",
                "negativo",
                "negativo",
                "negativo",
            ],
        }
    )


@pytest.fixture
def balanced_dataframe_20() -> pd.DataFrame:
    """Dataset sintético balanceado de 20 muestras (10 positivas, 10 negativas)."""
    return pd.DataFrame(
        {
            "texto": [f"Reseña positiva número {i}" for i in range(10)]
            + [f"Reseña negativa número {j}" for j in range(10)],
            "sentimiento": ["positivo"] * 10 + ["negativo"] * 10,
        }
    )


@pytest.fixture
def imbalanced_dataframe_20() -> pd.DataFrame:
    """Dataset sintético desbalanceado (15 positivas, 5 negativas)."""
    return pd.DataFrame(
        {
            "texto": [f"Opinión positiva {i}" for i in range(15)]
            + [f"Opinión negativa {j}" for j in range(5)],
            "sentimiento": ["positivo"] * 15 + ["negativo"] * 5,
        }
    )


# ---------------------------------------------------------------------------
# Tamaños y proporciones contractuales (80% train, 20% test)
# ---------------------------------------------------------------------------


def test_split_dataset_default_proportions(balanced_dataframe_10: pd.DataFrame) -> None:
    split = split_dataset(balanced_dataframe_10)

    assert isinstance(split, DatasetSplit)
    assert len(split.X_train) == 8  # 80%
    assert len(split.X_test) == 2  # 20%
    assert len(split.y_train) == 8
    assert len(split.y_test) == 2


def test_split_dataset_unpacking_and_types(balanced_dataframe_10: pd.DataFrame) -> None:
    X_train, X_test, y_train, y_test = split_dataset(balanced_dataframe_10)

    assert isinstance(X_train, pd.Series)
    assert isinstance(X_test, pd.Series)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert (X_train.index == y_train.index).all()
    assert (X_test.index == y_test.index).all()


def test_split_dataset_custom_test_size(balanced_dataframe_20: pd.DataFrame) -> None:
    split = split_dataset(balanced_dataframe_20, test_size=0.30)

    assert len(split.X_train) == 14  # 70%
    assert len(split.X_test) == 6  # 30%


# ---------------------------------------------------------------------------
# Estratificación por clase
# ---------------------------------------------------------------------------


def test_split_dataset_is_stratified_balanced(balanced_dataframe_10: pd.DataFrame) -> None:
    split = split_dataset(balanced_dataframe_10)

    # Con 10 muestras (5 de cada una) y 20% test:
    # Train tiene 4 pos y 4 neg; Test tiene 1 pos y 1 neg
    train_counts = split.y_train.value_counts().to_dict()
    test_counts = split.y_test.value_counts().to_dict()

    assert train_counts == {"positivo": 4, "negativo": 4}
    assert test_counts == {"positivo": 1, "negativo": 1}


def test_split_dataset_is_stratified_imbalanced(imbalanced_dataframe_20: pd.DataFrame) -> None:
    split = split_dataset(imbalanced_dataframe_20, test_size=0.20)

    # 15 positivas (80% train = 12, 20% test = 3)
    # 5 negativas (80% train = 4, 20% test = 1)
    train_counts = split.y_train.value_counts().to_dict()
    test_counts = split.y_test.value_counts().to_dict()

    assert train_counts == {"positivo": 12, "negativo": 4}
    assert test_counts == {"positivo": 3, "negativo": 1}


def test_split_dataset_guarantees_both_classes_in_splits(balanced_dataframe_10: pd.DataFrame) -> None:
    split = split_dataset(balanced_dataframe_10)

    assert set(split.y_train.unique()) == {"positivo", "negativo"}
    assert set(split.y_test.unique()) == {"positivo", "negativo"}


# ---------------------------------------------------------------------------
# Reproducibilidad (semilla 42 y configurable)
# ---------------------------------------------------------------------------


def test_split_dataset_is_reproducible_with_default_seed(balanced_dataframe_20: pd.DataFrame) -> None:
    split_1 = split_dataset(balanced_dataframe_20)
    split_2 = split_dataset(balanced_dataframe_20)

    assert_series_equal(split_1.X_train, split_2.X_train)
    assert_series_equal(split_1.X_test, split_2.X_test)
    assert_series_equal(split_1.y_train, split_2.y_train)
    assert_series_equal(split_1.y_test, split_2.y_test)


def test_split_dataset_different_seeds_produce_different_splits(balanced_dataframe_20: pd.DataFrame) -> None:
    split_a = split_dataset(balanced_dataframe_20, random_seed=42)
    split_b = split_dataset(balanced_dataframe_20, random_seed=99)

    # Los índices seleccionados deben diferir entre distintas semillas
    assert not split_a.X_train.index.equals(split_b.X_train.index)


# ---------------------------------------------------------------------------
# Garantía de no superposición de reseñas
# ---------------------------------------------------------------------------


def test_split_dataset_has_zero_overlap(balanced_dataframe_20: pd.DataFrame) -> None:
    split = split_dataset(balanced_dataframe_20)

    train_texts = set(split.X_train.values)
    test_texts = set(split.X_test.values)

    assert len(train_texts.intersection(test_texts)) == 0


def test_validate_no_overlap_raises_when_overlap_exists() -> None:
    train_series = pd.Series(["Buena película", "Regular"])
    test_series = pd.Series(["Pésima película", "Buena película"])

    with pytest.raises(DatasetValidationError, match="superposición de 1 reseña"):
        validate_no_overlap(train_series, test_series)


# ---------------------------------------------------------------------------
# Integración con validación de dataset existente
# ---------------------------------------------------------------------------


def test_split_dataset_rejects_data_with_duplicates() -> None:
    df_with_duplicates = pd.DataFrame(
        {
            "texto": ["Mismo texto", "Mismo texto", "Otro texto 1", "Otro texto 2"],
            "sentimiento": ["positivo", "positivo", "negativo", "negativo"],
        }
    )
    with pytest.raises(DatasetValidationError, match="texto.*duplicado"):
        split_dataset(df_with_duplicates)


def test_split_dataset_rejects_data_with_nulls() -> None:
    df_with_null = pd.DataFrame(
        {
            "texto": ["Texto 1", None, "Texto 3", "Texto 4"],
            "sentimiento": ["positivo", "positivo", "negativo", "negativo"],
        }
    )
    with pytest.raises(DatasetValidationError, match="valores nulos"):
        split_dataset(df_with_null)


def test_split_dataset_rejects_data_with_invalid_labels() -> None:
    df_invalid_labels = pd.DataFrame(
        {
            "texto": [f"Texto {i}" for i in range(4)],
            "sentimiento": ["positivo", "neutral", "negativo", "negativo"],
        }
    )
    with pytest.raises(DatasetValidationError, match="etiquetas no permitidas"):
        split_dataset(df_invalid_labels)


def test_split_dataset_loads_from_csv_path(tmp_path: Path, balanced_dataframe_10: pd.DataFrame) -> None:
    csv_file = tmp_path / "dataset.csv"
    balanced_dataframe_10.to_csv(csv_file, index=False, encoding="utf-8")

    split = split_dataset(csv_file)
    assert len(split.X_train) == 8
    assert len(split.X_test) == 2


# ---------------------------------------------------------------------------
# Validación de suficiencia de muestras y parámetros
# ---------------------------------------------------------------------------


def test_split_dataset_rejects_class_with_only_one_sample() -> None:
    df = pd.DataFrame(
        {
            "texto": [f"Texto {i}" for i in range(6)],
            "sentimiento": ["positivo"] * 5 + ["negativo"] * 1,
        }
    )
    with pytest.raises(DatasetValidationError, match="Muestras insuficientes para la clase 'negativo'"):
        split_dataset(df)


def test_split_dataset_rejects_insufficient_samples_for_test_size() -> None:
    # 2 muestras de negativo con test_size=0.20 asignan round(2 * 0.20) = 0 a prueba
    df = pd.DataFrame(
        {
            "texto": [f"Texto {i}" for i in range(10)],
            "sentimiento": ["positivo"] * 8 + ["negativo"] * 2,
        }
    )
    with pytest.raises(DatasetValidationError, match="Muestras insuficientes para la clase 'negativo'"):
        split_dataset(df, test_size=0.20)


def test_validate_split_sample_counts_missing_class() -> None:
    df = pd.DataFrame({"sentimiento": ["positivo", "positivo"]})
    with pytest.raises(DatasetValidationError, match="no está presente"):
        validate_split_sample_counts(df, "sentimiento", 0.20, ("positivo", "negativo"))


@pytest.mark.parametrize("invalid_test_size", [0.0, 1.0, -0.2, 1.5])
def test_split_dataset_rejects_invalid_test_size(
    balanced_dataframe_10: pd.DataFrame, invalid_test_size: float
) -> None:
    with pytest.raises(ValueError, match="'test_size' debe ser un número estrictamente entre 0 y 1"):
        split_dataset(balanced_dataframe_10, test_size=invalid_test_size)


def test_split_dataset_raises_file_not_found_on_missing_default_path() -> None:
    # Al llamar sin data, intenta cargar data/reviews.csv que aún no existe
    with pytest.raises(FileNotFoundError, match="El archivo del dataset no existe"):
        split_dataset()


def test_split_dataset_respects_custom_ml_config(balanced_dataframe_20: pd.DataFrame) -> None:
    cfg = load_config()
    split = split_dataset(balanced_dataframe_20, config=cfg)
    assert len(split.X_train) == 16
    assert len(split.X_test) == 4
