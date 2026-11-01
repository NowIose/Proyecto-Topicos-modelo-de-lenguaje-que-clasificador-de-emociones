"""Pruebas unitarias para la selección reproducible del mejor modelo por CV (Fase 3.6B)."""

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.models.experiments import (
    CandidateEvaluation,
    ExperimentResult,
    GridSearchResult,
    ModelSelectionResult,
    NgramComparisonResult,
    PreprocessingComparisonResult,
    format_model_selection,
    select_best_candidate,
    select_best_model,
    validate_candidates_compatibility,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# Helpers para crear resultados de validación cruzada simulados
# ---------------------------------------------------------------------------


def make_experiment_result(
    name: str,
    f1_mean: float,
    f1_std: float,
    acc_mean: float,
    folds: int = 5,
) -> ExperimentResult:
    """Crea un ExperimentResult simulado con pliegues coherentes."""
    # Folds sintéticos centrados en la media
    f1_folds = [f1_mean + ((-1) ** i) * (f1_std / 2) for i in range(folds)]
    acc_folds = [acc_mean + ((-1) ** i) * 0.01 for i in range(folds)]

    return ExperimentResult(
        name=name,
        metrics={
            "f1_macro_mean": f1_mean,
            "f1_macro_std": f1_std,
            "accuracy_mean": acc_mean,
            "accuracy_std": 0.01,
        },
        fold_metrics={
            "f1_macro": f1_folds,
            "accuracy": acc_folds,
        },
    )


# ---------------------------------------------------------------------------
# Pruebas de criterios de selección y reglas de desempate jerárquicas
# ---------------------------------------------------------------------------


def test_select_best_model_by_f1_macro_clear_winner() -> None:
    """Verifica que gane el modelo con mayor F1 Macro promedio (criterio principal)."""
    candidates = [
        make_experiment_result("multinomial_nb", f1_mean=0.8200, f1_std=0.0100, acc_mean=0.8300),
        make_experiment_result("logistic_regression", f1_mean=0.8850, f1_std=0.0300, acc_mean=0.8900),
    ]

    selection = select_best_model(candidates)

    assert isinstance(selection, ModelSelectionResult)
    assert selection.selected_model_name == "logistic_regression"
    assert selection.cv_metrics["f1_macro_mean"] == pytest.approx(0.8850)
    assert "mayor F1 Macro promedio" in selection.justification
    assert "logistic_regression" in selection.justification


def test_select_best_model_tiebreaker_1_lower_std() -> None:
    """Verifica el primer desempate: ante igual F1 Macro, gana la menor desviación estándar."""
    candidates = [
        make_experiment_result("model_variable", f1_mean=0.8500, f1_std=0.0450, acc_mean=0.8700),
        make_experiment_result("model_stable", f1_mean=0.8500, f1_std=0.0120, acc_mean=0.8600),
    ]

    selection = select_best_model(candidates)

    assert selection.selected_model_name == "model_stable"
    assert selection.cv_metrics["f1_macro_std"] == pytest.approx(0.0120)
    assert "menor desviación estándar" in selection.justification
    assert "mayor estabilidad" in selection.justification


def test_select_best_model_tiebreaker_2_higher_accuracy() -> None:
    """Verifica el segundo desempate: ante igual F1 Macro y desviación, gana mayor Accuracy."""
    candidates = [
        make_experiment_result("model_acc_lower", f1_mean=0.8500, f1_std=0.0200, acc_mean=0.8400),
        make_experiment_result("model_acc_higher", f1_mean=0.8500, f1_std=0.0200, acc_mean=0.8800),
    ]

    selection = select_best_model(candidates)

    assert selection.selected_model_name == "model_acc_higher"
    assert selection.cv_metrics["accuracy_mean"] == pytest.approx(0.8800)
    assert "mayor Accuracy promedio" in selection.justification


def test_select_best_model_tiebreaker_3_stable_identifier() -> None:
    """Verifica el desempate final determinista por orden alfabético estable."""
    candidates = [
        make_experiment_result("zebra_model", f1_mean=0.8500, f1_std=0.0200, acc_mean=0.8600),
        make_experiment_result("alpha_model", f1_mean=0.8500, f1_std=0.0200, acc_mean=0.8600),
    ]

    selection = select_best_model(candidates)

    assert selection.selected_model_name == "alpha_model"
    assert "desempate final determinista" in selection.justification
    assert "alpha_model" in selection.justification


def test_select_best_model_single_candidate() -> None:
    """Verifica que un único candidato sea seleccionado con justificación adecuada."""
    candidates = [
        make_experiment_result("only_model", f1_mean=0.9100, f1_std=0.0150, acc_mean=0.9200),
    ]

    selection = select_best_model(candidates)

    assert selection.selected_model_name == "only_model"
    assert len(selection.ranking) == 1
    assert "único candidato evaluado" in selection.justification


def test_select_best_candidate_alias_is_identical() -> None:
    """Verifica que select_best_candidate sea un alias funcional idéntico de select_best_model."""
    candidates = [
        make_experiment_result("model_a", f1_mean=0.80, f1_std=0.01, acc_mean=0.81),
        make_experiment_result("model_b", f1_mean=0.85, f1_std=0.01, acc_mean=0.86),
    ]

    res1 = select_best_model(candidates)
    res2 = select_best_candidate(candidates)

    assert res1.selected_model_name == res2.selected_model_name
    assert res1.cv_metrics == res2.cv_metrics


# ---------------------------------------------------------------------------
# Compatibilidad con estructuras de resultados existentes (Fase 3.4 y 3.5)
# ---------------------------------------------------------------------------


def test_select_best_model_from_ngram_comparison_result() -> None:
    """Verifica la selección directa desde un NgramComparisonResult (Fase 3.5A)."""
    results = {
        "logistic_regression__unigrams": make_experiment_result(
            "logistic_regression__unigrams", 0.85, 0.02, 0.86
        ),
        "logistic_regression__unigrams_and_bigrams": make_experiment_result(
            "logistic_regression__unigrams_and_bigrams", 0.89, 0.01, 0.90
        ),
        "multinomial_nb__unigrams": make_experiment_result(
            "multinomial_nb__unigrams", 0.81, 0.03, 0.82
        ),
        "multinomial_nb__unigrams_and_bigrams": make_experiment_result(
            "multinomial_nb__unigrams_and_bigrams", 0.83, 0.02, 0.84
        ),
    }

    ngram_comp = NgramComparisonResult(
        results=results,
        cv_folds=5,
        primary_metric="f1_macro",
        ranking=tuple(results.values()),
        ngram_configs={"unigrams": (1, 1), "unigrams_and_bigrams": (1, 2)},
    )

    selection = select_best_model(ngram_comp)

    assert selection.selected_model_name == "logistic_regression__unigrams_and_bigrams"
    assert selection.cv_metrics["f1_macro_mean"] == pytest.approx(0.89)
    assert selection.configuration["ngram_range"] == (1, 2)
    assert selection.configuration["representation"] == "unigrams_and_bigrams"


def test_select_best_model_from_preprocessing_comparison_result() -> None:
    """Verifica la selección directa desde un PreprocessingComparisonResult (Fase 3.5B)."""
    results = {
        "logistic_regression__stemming": make_experiment_result(
            "logistic_regression__stemming", 0.87, 0.02, 0.88
        ),
        "logistic_regression__lemmatization": make_experiment_result(
            "logistic_regression__lemmatization", 0.91, 0.01, 0.92
        ),
        "multinomial_nb__stemming": make_experiment_result(
            "multinomial_nb__stemming", 0.84, 0.02, 0.85
        ),
        "multinomial_nb__lemmatization": make_experiment_result(
            "multinomial_nb__lemmatization", 0.86, 0.02, 0.87
        ),
    }

    prep_comp = PreprocessingComparisonResult(
        results=results,
        cv_folds=5,
        primary_metric="f1_macro",
        ranking=tuple(results.values()),
        techniques=("stemming", "lemmatization"),
        ngram_range=(1, 1),
    )

    selection = select_best_model(prep_comp)

    assert selection.selected_model_name == "logistic_regression__lemmatization"
    assert selection.cv_metrics["f1_macro_mean"] == pytest.approx(0.91)
    assert selection.configuration["technique"] == "lemmatization"


def test_select_best_model_from_grid_search_mapping() -> None:
    """Verifica la selección directa desde un diccionario de GridSearchResult (Fase 3.4B)."""
    gs_lr = GridSearchResult(
        model_name="logistic_regression",
        best_params={"classifier__C": 1.0},
        best_score=0.9250,
        best_estimator=None,  # type: ignore
        cv_results={
            "mean_test_f1_macro": [0.85, 0.925, 0.91],
            "std_test_f1_macro": [0.03, 0.015, 0.02],
            "mean_test_accuracy": [0.86, 0.93, 0.92],
        },
        param_grid={"classifier__C": [0.1, 1.0, 10.0]},
    )

    gs_nb = GridSearchResult(
        model_name="multinomial_nb",
        best_params={"classifier__alpha": 0.1},
        best_score=0.8800,
        best_estimator=None,  # type: ignore
        cv_results={
            "mean_test_f1_macro": [0.88, 0.86, 0.84],
            "std_test_f1_macro": [0.02, 0.02, 0.03],
            "mean_test_accuracy": [0.89, 0.87, 0.85],
        },
        param_grid={"classifier__alpha": [0.1, 1.0, 10.0]},
    )

    selection = select_best_model({"logistic_regression": gs_lr, "multinomial_nb": gs_nb})

    assert selection.selected_model_name == "logistic_regression"
    assert selection.cv_metrics["f1_macro_mean"] == pytest.approx(0.9250)
    assert selection.configuration["classifier__C"] == 1.0


# ---------------------------------------------------------------------------
# Pruebas de validación de compatibilidad y rechazo de comparaciones incompatibles
# ---------------------------------------------------------------------------


def test_validation_rejects_incompatible_cv_folds() -> None:
    """Verifica que se rechace comparar candidatos evaluados con diferente número de pliegues."""
    c1 = CandidateEvaluation("model_5_folds", 0.85, 0.02, 0.86, cv_folds=5)
    c2 = CandidateEvaluation("model_10_folds", 0.88, 0.02, 0.89, cv_folds=10)

    with pytest.raises(ValueError) as exc_info:
        select_best_model([c1, c2])

    assert "Incompatibilidad en el protocolo" in str(exc_info.value)
    assert "5 folds" in str(exc_info.value)
    assert "10 folds" in str(exc_info.value)


def test_validation_rejects_empty_candidates() -> None:
    """Verifica que se rechace una colección vacía de candidatos."""
    with pytest.raises(ValueError, match="No se proporcionaron candidatos"):
        select_best_model([])

    with pytest.raises(ValueError, match="No se proporcionaron candidatos"):
        select_best_model({})


def test_validation_rejects_nan_metrics() -> None:
    """Verifica que se rechacen métricas NaN o no finitas."""
    c_nan = CandidateEvaluation("model_nan", float("nan"), 0.02, 0.86, cv_folds=5)
    with pytest.raises(ValueError, match="no numérico o no finito"):
        select_best_model([c_nan])


def test_validation_rejects_out_of_bounds_metrics() -> None:
    """Verifica que se rechacen métricas fuera del rango válido [0.0, 1.0]."""
    c_high = CandidateEvaluation("model_high", 1.5, 0.02, 0.86, cv_folds=5)
    with pytest.raises(ValueError, match="fuera del rango"):
        select_best_model([c_high])

    c_negative = CandidateEvaluation("model_negative", -0.2, 0.02, 0.86, cv_folds=5)
    with pytest.raises(ValueError, match="fuera del rango"):
        select_best_model([c_negative])


def test_validation_rejects_inconsistent_fold_lengths() -> None:
    """Verifica que se rechacen candidatos donde fold_metrics no coincide con cv_folds."""
    c_inconsistent = CandidateEvaluation(
        "model_inconsistent",
        0.85,
        0.02,
        0.86,
        cv_folds=5,
        fold_metrics={"f1_macro": [0.84, 0.86]},  # solo 2 pliegues en vez de 5
    )

    with pytest.raises(ValueError, match="pliegues en 'fold_metrics'"):
        select_best_model([c_inconsistent])


# ---------------------------------------------------------------------------
# Pruebas de exportación a DataFrame y formateo textual
# ---------------------------------------------------------------------------


def test_model_selection_to_dataframe_and_formatting() -> None:
    """Verifica que el ranking se convierta a DataFrame y que el reporte textual sea completo."""
    candidates = [
        make_experiment_result("candidate_c", f1_mean=0.75, f1_std=0.02, acc_mean=0.76),
        make_experiment_result("candidate_a", f1_mean=0.88, f1_std=0.01, acc_mean=0.89),
        make_experiment_result("candidate_b", f1_mean=0.82, f1_std=0.03, acc_mean=0.83),
    ]

    selection = select_best_model(candidates)

    df = selection.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df["rank"]) == [1, 2, 3]
    assert df.iloc[0]["name"] == "candidate_a"

    text_summary = selection.summary()
    assert "SELECCIÓN REPRODUCIBLE DEL MEJOR MODELO POR VALIDACIÓN CRUZADA" in text_summary
    assert "candidate_a" in text_summary
    assert "Justificación metodológica" in text_summary
    assert "Criterios de desempate aplicados" in text_summary
    assert "Nota metodológica" in text_summary
