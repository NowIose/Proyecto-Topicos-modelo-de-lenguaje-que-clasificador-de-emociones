"""Pruebas unitarias para la carga y validación del dataset del módulo ML."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.config import DatasetConfig, MLConfig, load_config
from src.models.dataset import (
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


@pytest.fixture
def valid_dataframe() -> pd.DataFrame:
    """Fixture con un dataset sintético válido."""
    return pd.DataFrame(
        {
            "texto": [
                "Excelente producto y servicio impecable",
                "Pésima atención, no lo recomiendo para nada",
                "La experiencia fue bastante buena en general",
                "Muy mala calidad, se rompió al primer uso",
            ],
            "sentimiento": [
                "positivo",
                "negativo",
                "positivo",
                "negativo",
            ],
        }
    )


# ---------------------------------------------------------------------------
# Casos correctos (Happy path)
# ---------------------------------------------------------------------------


def test_validate_dataset_succeeds_on_valid_data(valid_dataframe: pd.DataFrame) -> None:
    result = validate_dataset(valid_dataframe)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4
    assert set(result["sentimiento"]) == {"positivo", "negativo"}


def test_validate_dataset_allows_extra_columns(valid_dataframe: pd.DataFrame) -> None:
    valid_dataframe["id"] = [1, 2, 3, 4]
    valid_dataframe["fuente"] = ["web", "app", "web", "tienda"]
    result = validate_dataset(valid_dataframe)
    assert "id" in result.columns
    assert "fuente" in result.columns


def test_validate_dataset_allows_different_column_order() -> None:
    df = pd.DataFrame(
        {
            "sentimiento": ["positivo", "negativo"],
            "texto": ["Gran servicio al cliente", "Horrible experiencia"],
        }
    )
    result = validate_dataset(df)
    assert len(result) == 2


def test_load_csv_reads_valid_csv(tmp_path: Path, valid_dataframe: pd.DataFrame) -> None:
    csv_file = tmp_path / "valid.csv"
    valid_dataframe.to_csv(csv_file, index=False, encoding="utf-8")

    loaded_df = load_csv(csv_file)
    assert len(loaded_df) == len(valid_dataframe)
    assert list(loaded_df.columns) == ["texto", "sentimiento"]


def test_load_dataset_with_filepath(tmp_path: Path, valid_dataframe: pd.DataFrame) -> None:
    csv_file = tmp_path / "valid.csv"
    valid_dataframe.to_csv(csv_file, index=False, encoding="utf-8")

    loaded_df = load_dataset(csv_file)
    assert len(loaded_df) == 4
    assert set(loaded_df["sentimiento"]) == {"positivo", "negativo"}


def test_load_dataset_with_dataset_config(tmp_path: Path, valid_dataframe: pd.DataFrame) -> None:
    csv_file = tmp_path / "valid.csv"
    valid_dataframe.to_csv(csv_file, index=False, encoding="utf-8")

    cfg = DatasetConfig(
        path=csv_file,
        text_column="texto",
        target_column="sentimiento",
        labels=("positivo", "negativo"),
    )
    loaded_df = load_dataset(cfg)
    assert len(loaded_df) == 4


def test_load_dataset_with_ml_config(tmp_path: Path, valid_dataframe: pd.DataFrame) -> None:
    csv_file = tmp_path / "valid.csv"
    valid_dataframe.to_csv(csv_file, index=False, encoding="utf-8")

    base_config = load_config()
    custom_dataset_cfg = DatasetConfig(
        path=csv_file,
        text_column="texto",
        target_column="sentimiento",
        labels=("positivo", "negativo"),
    )
    ml_cfg = MLConfig(
        dataset=custom_dataset_cfg,
        experiment=base_config.experiment,
        artifacts=base_config.artifacts,
    )

    loaded_df = load_dataset(ml_cfg)
    assert len(loaded_df) == 4


# ---------------------------------------------------------------------------
# Validación estructural y de columnas
# ---------------------------------------------------------------------------


def test_validate_dataset_rejects_non_dataframe() -> None:
    with pytest.raises(DatasetValidationError, match="Se esperaba un objeto pandas DataFrame"):
        validate_dataset([{"texto": "bueno", "sentimiento": "positivo"}])  # type: ignore[arg-type]


def test_validate_dataset_rejects_empty_dataframe() -> None:
    empty_df = pd.DataFrame(columns=["texto", "sentimiento"])
    with pytest.raises(DatasetValidationError, match="El dataset está vacío"):
        validate_dataset(empty_df)


def test_validate_dataset_rejects_missing_text_column() -> None:
    df = pd.DataFrame({"opinion": ["muy bueno", "muy malo"], "sentimiento": ["positivo", "negativo"]})
    with pytest.raises(DatasetValidationError, match="Faltan columnas requeridas"):
        validate_dataset(df)


def test_validate_dataset_rejects_missing_target_column() -> None:
    df = pd.DataFrame({"texto": ["muy bueno", "muy malo"], "clase": ["positivo", "negativo"]})
    with pytest.raises(DatasetValidationError, match="Faltan columnas requeridas"):
        validate_dataset(df)


def test_validate_dataset_rejects_missing_both_columns() -> None:
    df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
    with pytest.raises(DatasetValidationError, match="Faltan columnas requeridas"):
        validate_dataset(df)


# ---------------------------------------------------------------------------
# Validación de valores nulos
# ---------------------------------------------------------------------------


def test_validate_dataset_rejects_nulls_in_text() -> None:
    df = pd.DataFrame(
        {
            "texto": ["Buen servicio", None, "Mala atención"],
            "sentimiento": ["positivo", "negativo", "negativo"],
        }
    )
    with pytest.raises(DatasetValidationError, match="valores nulos.*'texto'"):
        validate_dataset(df)


def test_validate_dataset_rejects_nan_in_target() -> None:
    df = pd.DataFrame(
        {
            "texto": ["Buen servicio", "Mala atención"],
            "sentimiento": ["positivo", np.nan],
        }
    )
    with pytest.raises(DatasetValidationError, match="valores nulos.*'sentimiento'"):
        validate_dataset(df)


def test_validate_dataset_rejects_nulls_in_both_columns() -> None:
    df = pd.DataFrame(
        {
            "texto": ["Buen servicio", None],
            "sentimiento": [None, "negativo"],
        }
    )
    with pytest.raises(DatasetValidationError, match="valores nulos.*'texto'.*'sentimiento'"):
        validate_dataset(df)


# ---------------------------------------------------------------------------
# Validación de textos vacíos y tipos no-string
# ---------------------------------------------------------------------------


def test_validate_dataset_rejects_empty_string_text() -> None:
    df = pd.DataFrame(
        {
            "texto": ["Buen servicio", ""],
            "sentimiento": ["positivo", "negativo"],
        }
    )
    with pytest.raises(DatasetValidationError, match="texto vacío o solo espacios"):
        validate_dataset(df)


def test_validate_dataset_rejects_whitespace_only_text() -> None:
    df = pd.DataFrame(
        {
            "texto": ["Buen servicio", "   \t \n  "],
            "sentimiento": ["positivo", "negativo"],
        }
    )
    with pytest.raises(DatasetValidationError, match="texto vacío o solo espacios"):
        validate_dataset(df)


def test_validate_dataset_rejects_non_string_type_in_text() -> None:
    df = pd.DataFrame(
        {
            "texto": ["Buen servicio", 12345],
            "sentimiento": ["positivo", "negativo"],
        }
    )
    with pytest.raises(DatasetValidationError, match="debe contener únicamente texto"):
        validate_dataset(df)


# ---------------------------------------------------------------------------
# Validación de duplicados
# ---------------------------------------------------------------------------


def test_validate_dataset_rejects_identical_rows() -> None:
    df = pd.DataFrame(
        {
            "texto": ["Excelente película", "Excelente película", "Mala película"],
            "sentimiento": ["positivo", "positivo", "negativo"],
        }
    )
    with pytest.raises(DatasetValidationError, match="texto.*duplicado"):
        validate_dataset(df)


def test_validate_dataset_rejects_duplicate_text_with_conflicting_label() -> None:
    df = pd.DataFrame(
        {
            "texto": ["Excelente película", "Excelente película"],
            "sentimiento": ["positivo", "negativo"],
        }
    )
    with pytest.raises(DatasetValidationError, match="texto.*duplicado"):
        validate_dataset(df)


def test_validate_dataset_rejects_duplicate_text_with_spaces() -> None:
    df = pd.DataFrame(
        {
            "texto": ["Excelente película", "  Excelente película  ", "Pésima película"],
            "sentimiento": ["positivo", "positivo", "negativo"],
        }
    )
    with pytest.raises(DatasetValidationError, match="texto.*duplicado"):
        validate_dataset(df)


# ---------------------------------------------------------------------------
# Validación de etiquetas y clases
# ---------------------------------------------------------------------------


def test_validate_dataset_rejects_invalid_labels() -> None:
    df = pd.DataFrame(
        {
            "texto": ["Gran película", "Mala película", "Película regular"],
            "sentimiento": ["positivo", "negativo", "neutral"],
        }
    )
    with pytest.raises(DatasetValidationError, match="etiquetas no permitidas.*neutral"):
        validate_dataset(df)


def test_validate_dataset_rejects_only_positive_class() -> None:
    df = pd.DataFrame(
        {
            "texto": ["Gran servicio", "Me encantó todo"],
            "sentimiento": ["positivo", "positivo"],
        }
    )
    with pytest.raises(DatasetValidationError, match="Clases faltantes.*negativo"):
        validate_dataset(df)


def test_validate_dataset_rejects_only_negative_class() -> None:
    df = pd.DataFrame(
        {
            "texto": ["Pésimo servicio", "No vuelvo nunca"],
            "sentimiento": ["negativo", "negativo"],
        }
    )
    with pytest.raises(DatasetValidationError, match="Clases faltantes.*positivo"):
        validate_dataset(df)


# ---------------------------------------------------------------------------
# Validación de carga de archivos CSV
# ---------------------------------------------------------------------------


def test_load_csv_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="El archivo del dataset no existe"):
        load_csv("data/non_existent_file.csv")


def test_load_csv_raises_on_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no es un archivo válido"):
        load_csv(tmp_path)


def test_load_csv_raises_on_empty_file(tmp_path: Path) -> None:
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="vacío o sin datos legibles"):
        load_csv(empty_csv)


def test_load_dataset_raises_when_default_file_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _config_con_dataset_inexistente(monkeypatch, "src.models.dataset")
    with pytest.raises(FileNotFoundError, match="El archivo del dataset no existe"):
        load_dataset()


def test_load_dataset_raises_on_invalid_csv_data(tmp_path: Path) -> None:
    csv_file = tmp_path / "invalid_data.csv"
    pd.DataFrame(
        {
            "texto": ["Texto 1", "Texto 2"],
            "sentimiento": ["positivo", "positivo"],  # falta 'negativo'
        }
    ).to_csv(csv_file, index=False)

    with pytest.raises(DatasetValidationError, match="Clases faltantes.*negativo"):
        load_dataset(csv_file)


def _config_con_dataset_inexistente(monkeypatch: pytest.MonkeyPatch, *modulos: str) -> None:
    """Hace que la config por defecto apunte a un CSV que no existe."""
    import dataclasses
    import importlib

    from src.models.config import PROJECT_ROOT
    from src.models.config import load_config as cargar_real

    real = cargar_real()
    falsa = dataclasses.replace(
        real,
        dataset=dataclasses.replace(real.dataset, path=PROJECT_ROOT / "data" / "no_existe.csv"),
    )
    for modulo in modulos:
        monkeypatch.setattr(importlib.import_module(modulo), "load_config", lambda *a, **k: falsa)
