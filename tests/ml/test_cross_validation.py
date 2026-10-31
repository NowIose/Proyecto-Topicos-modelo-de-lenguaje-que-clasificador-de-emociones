"""Pruebas unitarias para la validación cruzada estratificada (Fase 3.4A)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.models.config import EXPECTED_LABELS, load_config
from src.models.experiments import (
    ExperimentResult,
    ExperimentSpec,
    build_logistic_regression_pipeline,
    build_multinomial_nb_pipeline,
    build_stratified_kfold,
    cross_validate_pipeline,
    evaluate_candidates_cv,
    format_cv_comparison,
    run_cross_validation,
    validate_cv_sample_counts,
)
from src.models.split import DatasetSplit, split_dataset

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# Fixtures sintéticas con suficiencia muestral para 5 folds
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_cv_dataframe_20() -> pd.DataFrame:
    """Dataset balanceado de 20 muestras (10 positivas, 10 negativas).

    Garantiza al menos 10 muestras por clase, suficiente para k=5 folds
    (2 muestras de cada clase por pliegue).
    """
    return pd.DataFrame(
        {
            "texto": [
                "Excelente producto y servicio impecable",
                "Muy buena atención recibida en la tienda",
                "Me encantó la rapidez del envío del pedido",
                "Experiencia genial y totalmente recomendable",
                "Totalmente satisfecho con la compra realizada",
                "El paquete llegó en perfecto estado y rápido",
                "Gran calidad de materiales y buen precio",
                "La mejor compra que he realizado este año",
                "Atención al cliente fabulosa y muy eficiente",
                "Quedé encantado con el servicio y la atención",
                "Pésima atención, no lo recomiendo en absoluto",
                "Muy mala calidad del producto recibido",
                "Llegó tarde y roto el paquete comprado",
                "No sirve para nada este servicio tan deficiente",
                "Terrible experiencia vivida con este vendedor",
                "El producto dejó de funcionar al primer día",
                "Muy decepcionado con el servicio posventa",
                "Una completa estafa, producto falso y roto",
                "Demoraron semanas y no responden los mensajes",
                "Horrible servicio y peor atención recibida",
            ],
            "sentimiento": [
                "positivo",
                "positivo",
                "positivo",
                "positivo",
                "positivo",
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
                "negativo",
                "negativo",
                "negativo",
                "negativo",
                "negativo",
            ],
        }
    )


# ---------------------------------------------------------------------------
# Construcción y validación de StratifiedKFold
# ---------------------------------------------------------------------------


def test_build_stratified_kfold_contract_and_defaults() -> None:
    config = load_config()
    cv = build_stratified_kfold(config)

    assert isinstance(cv, StratifiedKFold)
    assert cv.get_n_splits() == 5
    assert cv.shuffle is True
    assert cv.random_state == 42


def test_build_stratified_kfold_custom_parameters() -> None:
    cv = build_stratified_kfold(n_splits=3, random_state=99, shuffle=True)

    assert cv.get_n_splits() == 3
    assert cv.shuffle is True
    assert cv.random_state == 99

    cv_no_shuffle = build_stratified_kfold(n_splits=4, shuffle=False)
    assert cv_no_shuffle.get_n_splits() == 4
    assert cv_no_shuffle.shuffle is False
    assert cv_no_shuffle.random_state is None

    with pytest.raises(ValueError, match="al menos 2"):
        build_stratified_kfold(n_splits=1)


def test_validate_cv_sample_counts_success() -> None:
    y = ["positivo"] * 5 + ["negativo"] * 5
    # No debe levantar excepción
    validate_cv_sample_counts(y, n_splits=5, expected_labels=EXPECTED_LABELS)


def test_validate_cv_sample_counts_insufficient_samples() -> None:
    # 4 muestras para 'negativo' cuando se requieren 5
    y = ["positivo"] * 10 + ["negativo"] * 4

    with pytest.raises(ValueError, match="Muestras insuficientes para la clase 'negativo'"):
        validate_cv_sample_counts(y, n_splits=5)


def test_validate_cv_sample_counts_missing_class() -> None:
    y = ["positivo"] * 10

    with pytest.raises(ValueError, match="no está presente"):
        validate_cv_sample_counts(y, n_splits=5, expected_labels=EXPECTED_LABELS)


# ---------------------------------------------------------------------------
# Evaluación de un pipeline mediante cross_validate
# ---------------------------------------------------------------------------


def test_cross_validate_pipeline_metrics_and_folds(
    synthetic_cv_dataframe_20: pd.DataFrame,
) -> None:
    pipeline = build_logistic_regression_pipeline()
    X = synthetic_cv_dataframe_20["texto"]
    y = synthetic_cv_dataframe_20["sentimiento"]

    result = cross_validate_pipeline(
        pipeline=pipeline,
        X_train=X,
        y_train=y,
        name="logistic_regression",
    )

    assert isinstance(result, ExperimentResult)
    assert result.name == "logistic_regression"

    # Verificar presencia de promedios y desviaciones estándar
    for metric_name in (
        "f1_macro_mean",
        "f1_macro_std",
        "accuracy_mean",
        "accuracy_std",
    ):
        assert metric_name in result.metrics
        val = result.metrics[metric_name]
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0

    # Verificar métricas individuales por fold (5 particiones)
    assert "f1_macro" in result.fold_metrics
    assert "accuracy" in result.fold_metrics
    assert len(result.fold_metrics["f1_macro"]) == 5
    assert len(result.fold_metrics["accuracy"]) == 5

    # Consistencia matemática entre los folds y el promedio reportado
    expected_f1_mean = float(np.mean(result.fold_metrics["f1_macro"]))
    expected_f1_std = float(np.std(result.fold_metrics["f1_macro"]))
    assert result.metrics["f1_macro_mean"] == pytest.approx(expected_f1_mean)
    assert result.metrics["f1_macro_std"] == pytest.approx(expected_f1_std)


def test_cross_validate_pipeline_no_data_leakage(
    synthetic_cv_dataframe_20: pd.DataFrame,
) -> None:
    """Verifica que el Pipeline encapsula la vectorización dentro de cada fold."""
    pipeline = build_multinomial_nb_pipeline()
    X = synthetic_cv_dataframe_20["texto"]
    y = synthetic_cv_dataframe_20["sentimiento"]

    result = cross_validate_pipeline(
        pipeline=pipeline,
        X_train=X,
        y_train=y,
        name="multinomial_nb",
    )

    # El estimador no debe ser mutado de forma global por cross_validate
    assert not hasattr(pipeline.named_steps["vectorizer"], "vocabulary_")
    assert result.name == "multinomial_nb"


# ---------------------------------------------------------------------------
# Comparación y ordenamiento de ambos candidatos
# ---------------------------------------------------------------------------


def test_run_cross_validation_compares_and_ranks_candidates(
    synthetic_cv_dataframe_20: pd.DataFrame,
) -> None:
    X = synthetic_cv_dataframe_20["texto"]
    y = synthetic_cv_dataframe_20["sentimiento"]

    results = run_cross_validation(X, y)

    assert isinstance(results, list)
    assert len(results) == 2

    model_names = {r.name for r in results}
    assert model_names == {"logistic_regression", "multinomial_nb"}

    # Ordenamiento estricto por F1 Macro descendente
    assert results[0].metrics["f1_macro_mean"] >= results[1].metrics["f1_macro_mean"]


def test_run_cross_validation_alias_evaluate_candidates_cv(
    synthetic_cv_dataframe_20: pd.DataFrame,
) -> None:
    X = synthetic_cv_dataframe_20["texto"]
    y = synthetic_cv_dataframe_20["sentimiento"]

    r1 = run_cross_validation(X, y)
    r2 = evaluate_candidates_cv(X, y)

    assert len(r1) == len(r2)
    assert [r.name for r in r1] == [r.name for r in r2]


def test_run_cross_validation_with_dataset_split_preserves_test_set(
    synthetic_cv_dataframe_20: pd.DataFrame,
) -> None:
    """Garantiza que la validación cruzada opere EXCLUSIVAMENTE sobre X_train e y_train.

    X_test e y_test nunca deben ser accedidos ni evaluados en esta fase.
    """
    split = split_dataset(
        synthetic_cv_dataframe_20,
        test_size=0.20,
        random_seed=42,
    )

    # Inyectamos datos inválidos deliberadamente en X_test e y_test
    # para asegurar que si se llegasen a tocar, fallaría estrepitosamente.
    tampered_test_split = DatasetSplit(
        X_train=split.X_train,
        X_test=pd.Series(["TEXTO_CORRUPTO_QUE_NUNCA_DEBE_LEERSE"] * len(split.X_test)),
        y_train=split.y_train,
        y_test=pd.Series(["ETIQUETA_INVALIDA"] * len(split.y_test)),
    )

    # Debe ejecutarse perfectamente sin tocar el conjunto de prueba
    results = run_cross_validation(tampered_test_split)

    assert len(results) == 2
    for r in results:
        assert "f1_macro_mean" in r.metrics


def test_run_cross_validation_with_specs(
    synthetic_cv_dataframe_20: pd.DataFrame,
) -> None:
    specs = [
        ExperimentSpec("logistic_regression", {"C": 0.5, "max_iter": 500}),
        ExperimentSpec("multinomial_nb", {"alpha": 0.2}),
    ]
    X = synthetic_cv_dataframe_20["texto"]
    y = synthetic_cv_dataframe_20["sentimiento"]

    results = run_cross_validation(X, y, specs=specs)
    assert len(results) == 2
    assert {r.name for r in results} == {"logistic_regression", "multinomial_nb"}


def test_format_cv_comparison(
    synthetic_cv_dataframe_20: pd.DataFrame,
) -> None:
    X = synthetic_cv_dataframe_20["texto"]
    y = synthetic_cv_dataframe_20["sentimiento"]

    results = run_cross_validation(X, y)
    table = format_cv_comparison(results)

    assert "COMPARACIÓN DE MODELOS MEDIANTE VALIDACIÓN CRUZADA ESTRATIFICADA" in table
    assert "logistic_regression" in table
    assert "multinomial_nb" in table
    assert "F1 Macro promedio:" in table
    assert "Accuracy promedio:" in table
    assert "Folds F1 Macro:" in table


# ---------------------------------------------------------------------------
# Manejo de errores y casos inválidos
# ---------------------------------------------------------------------------


def test_cv_error_handling(synthetic_cv_dataframe_20: pd.DataFrame) -> None:
    X = synthetic_cv_dataframe_20["texto"]
    y = synthetic_cv_dataframe_20["sentimiento"]

    # 1. No pipeline
    with pytest.raises(TypeError, match="sklearn.pipeline.Pipeline"):
        cross_validate_pipeline("no_es_pipeline", X, y)  # type: ignore

    # 2. X e y discordantes en longitud
    with pytest.raises(ValueError, match="no coincide"):
        cross_validate_pipeline(build_logistic_regression_pipeline(), X, y.iloc[:5])

    # 3. Datos vacíos
    with pytest.raises(ValueError, match="no pueden estar vacíos"):
        cross_validate_pipeline(build_logistic_regression_pipeline(), [], [])

    # 4. DatasetSplit con y_train explícito redundante
    split = split_dataset(synthetic_cv_dataframe_20, test_size=0.20, random_seed=42)
    with pytest.raises(ValueError, match="No se debe proporcionar 'y_train'"):
        run_cross_validation(split, y_train=split.y_train)

    # 5. X_train sin y_train cuando no es DatasetSplit
    with pytest.raises(ValueError, match="Se requiere 'y_train'"):
        run_cross_validation(X)
