"""Pruebas unitarias para la orquestación reproducible del entrenamiento en train.py."""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.models.config import load_config
from src.models.dataset import DatasetValidationError
from src.models.split import DatasetSplit
from src.models.train import TrainingResult, format_training_summary, main, train

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture
def synthetic_dataframe_10() -> pd.DataFrame:
    """Dataset sintético balanceado de 10 muestras para pruebas."""
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
def synthetic_csv_path(tmp_path: Path, synthetic_dataframe_10: pd.DataFrame) -> Path:
    """Guarda el dataset de prueba en un archivo CSV temporal."""
    csv_file = tmp_path / "reviews_sample.csv"
    synthetic_dataframe_10.to_csv(csv_file, index=False, encoding="utf-8")
    return csv_file


# ---------------------------------------------------------------------------
# Flujo de entrenamiento y resultado estructurado
# ---------------------------------------------------------------------------


def test_train_with_dataframe_returns_valid_training_result(
    synthetic_dataframe_10: pd.DataFrame,
) -> None:
    result = train(data=synthetic_dataframe_10)

    assert isinstance(result, TrainingResult)
    assert isinstance(result.split, DatasetSplit)

    # Verificación de proporciones (80% train / 20% test para 10 muestras)
    assert len(result.split.X_train) == 8
    assert len(result.split.X_test) == 2
    assert len(result.split.y_train) == 8
    assert len(result.split.y_test) == 2

    # Verificación de modelos candidatos
    assert set(result.models.keys()) == {"logistic_regression", "multinomial_nb"}

    lr_pipeline = result.logistic_regression
    nb_pipeline = result.multinomial_nb

    assert isinstance(lr_pipeline, Pipeline)
    assert isinstance(nb_pipeline, Pipeline)
    assert isinstance(lr_pipeline.named_steps["classifier"], LogisticRegression)
    assert isinstance(nb_pipeline.named_steps["classifier"], MultinomialNB)

    # Ambos modelos deben estar ajustados (fitted)
    assert hasattr(lr_pipeline.named_steps["classifier"], "classes_")
    assert hasattr(lr_pipeline.named_steps["vectorizer"], "vocabulary_")
    assert hasattr(nb_pipeline.named_steps["classifier"], "classes_")
    assert hasattr(nb_pipeline.named_steps["vectorizer"], "vocabulary_")


def test_train_from_csv_file_path(synthetic_csv_path: Path) -> None:
    result = train(data=synthetic_csv_path)

    assert isinstance(result, TrainingResult)
    assert len(result.split.X_train) == 8
    assert len(result.split.X_test) == 2
    assert "logistic_regression" in result.models
    assert "multinomial_nb" in result.models


def test_train_with_config_argument(synthetic_dataframe_10: pd.DataFrame) -> None:
    config = load_config()
    result = train(data=synthetic_dataframe_10, config=config)

    assert result.config is config
    assert result.logistic_regression.named_steps["classifier"].random_state == config.experiment.random_seed


# ---------------------------------------------------------------------------
# Reproducibilidad y aislamiento del subconjunto de prueba
# ---------------------------------------------------------------------------


def test_train_is_strictly_reproducible(synthetic_dataframe_10: pd.DataFrame) -> None:
    result_1 = train(data=synthetic_dataframe_10)
    result_2 = train(data=synthetic_dataframe_10)

    # La partición debe ser exactamente idéntica
    assert_series_equal(result_1.split.X_train, result_2.split.X_train)
    assert_series_equal(result_1.split.X_test, result_2.split.X_test)
    assert_series_equal(result_1.split.y_train, result_2.split.y_train)
    assert_series_equal(result_1.split.y_test, result_2.split.y_test)

    # Las predicciones sobre X_test deben ser idénticas
    lr_pred_1 = result_1.logistic_regression.predict(result_1.split.X_test)
    lr_pred_2 = result_2.logistic_regression.predict(result_2.split.X_test)
    assert (lr_pred_1 == lr_pred_2).all()

    nb_pred_1 = result_1.multinomial_nb.predict(result_1.split.X_test)
    nb_pred_2 = result_2.multinomial_nb.predict(result_2.split.X_test)
    assert (nb_pred_1 == nb_pred_2).all()


def test_train_does_not_leak_test_data_into_vectorizer(
    synthetic_dataframe_10: pd.DataFrame,
) -> None:
    """Verifica que los textos en X_test no se utilizan para ajustar el vectorizador."""
    result = train(data=synthetic_dataframe_10)

    # Encontrar palabras que solo están en X_test
    train_tokens = set(" ".join(result.split.X_train).lower().split())
    test_tokens = set(" ".join(result.split.X_test).lower().split())
    unique_test_tokens = test_tokens - train_tokens

    # Si hay palabras exclusivas en test, no deben estar en el vocabulario aprendido
    lr_vocab = result.logistic_regression.named_steps["vectorizer"].vocabulary_
    nb_vocab = result.multinomial_nb.named_steps["vectorizer"].vocabulary_

    for token in unique_test_tokens:
        # Los unigramas de solo test no deben haberse aprendido
        assert token not in lr_vocab
        assert token not in nb_vocab


# ---------------------------------------------------------------------------
# Resumen del entrenamiento
# ---------------------------------------------------------------------------


def test_format_training_summary(synthetic_dataframe_10: pd.DataFrame) -> None:
    result = train(data=synthetic_dataframe_10)
    summary = result.summary()

    assert "RESUMEN DEL ENTRENAMIENTO DE MODELOS CANDIDATOS" in summary
    assert "Total de registros procesados: 10" in summary
    assert "Conjunto de entrenamiento (X_train, y_train): 8 registros" in summary
    assert "Conjunto de prueba reservado (X_test, y_test): 2 registros" in summary
    assert "positivo" in summary
    assert "negativo" in summary
    assert "logistic_regression" in summary
    assert "multinomial_nb" in summary

    # Garantizar que no se inventen métricas de rendimiento (f1, accuracy, etc.)
    assert "f1_score" not in summary.lower()
    assert "accuracy:" not in summary.lower()
    assert "precision:" not in summary.lower()


# ---------------------------------------------------------------------------
# Control de errores y CLI
# ---------------------------------------------------------------------------


def test_train_raises_file_not_found_on_missing_path() -> None:
    with pytest.raises(FileNotFoundError):
        train(data="archivo_que_definitivamente_no_existe.csv")


def test_cli_main_success_with_data_flag(
    synthetic_csv_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["--data", str(synthetic_csv_path)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "RESUMEN DEL ENTRENAMIENTO DE MODELOS CANDIDATOS" in captured.out
    assert "Total de registros procesados: 10" in captured.out
    assert captured.err == ""


def test_cli_main_missing_dataset_returns_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["--data", "archivo_inexistente_999.csv"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "No se encontró el dataset" in captured.err


def test_cli_main_default_missing_dataset(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Por defecto apunta a data/reviews.csv que aún no existe
    exit_code = main([])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "No se encontró el dataset" in captured.err


def test_cli_main_invalid_dataset_format(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid_csv = tmp_path / "invalid.csv"
    invalid_csv.write_text("columna_invalida,otra_columna\ntexto1,positivo\n", encoding="utf-8")

    exit_code = main(["--data", str(invalid_csv)])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error de validación del dataset:" in captured.err
