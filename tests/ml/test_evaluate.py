"""Pruebas unitarias para las funciones de evaluación y análisis de errores en evaluate.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.config import EXPECTED_LABELS
from src.models.evaluate import (
    EvaluationReport,
    Misclassification,
    compute_accuracy,
    compute_class_metrics,
    compute_classification_report,
    compute_confusion_matrix,
    compute_f1_macro,
    compute_f1_scores,
    compute_f1_weighted,
    evaluate,
    evaluate_pipeline,
    evaluate_predictions,
    format_error_analysis,
    identify_misclassifications,
    misclassifications_to_dataframe,
)
from src.models.experiments import build_logistic_regression_pipeline, train_pipeline

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# Fixtures con ejemplos pequeños y resultados calculables a mano
# ---------------------------------------------------------------------------


@pytest.fixture
def exact_eval_data() -> dict[str, list[str]]:
    """Dataset pequeño con métricas exactas calculables a mano.

    Distribución:
      - 3 reales 'positivo', 1 real 'negativo'
      - y_true: ['positivo', 'positivo', 'positivo', 'negativo']
      - y_pred: ['positivo', 'positivo', 'negativo', 'negativo']

    Cálculos analíticos:
      - Total = 4
      - Aciertos = 3 (muestras 0, 1 y 3) -> Accuracy = 3/4 = 0.75
      - Clase 'positivo':
          TP = 2 (muestras 0 y 1)
          FP = 0
          FN = 1 (muestra 2 era positivo y se predijo negativo)
          Precision = 2 / (2 + 0) = 1.0
          Recall    = 2 / (2 + 1) = 2/3 ≈ 0.66667
          F1-score  = 2 * (1.0 * 2/3) / (1.0 + 2/3) = 4/5 = 0.80
          Support   = 3
      - Clase 'negativo':
          TP = 1 (muestra 3)
          FP = 1 (muestra 2)
          FN = 0
          Precision = 1 / (1 + 1) = 0.50
          Recall    = 1 / (1 + 0) = 1.0
          F1-score  = 2 * (0.50 * 1.0) / (0.50 + 1.0) = 2/3 ≈ 0.66667
          Support   = 1
      - F1 Macro    = (0.80 + 2/3) / 2 = 11/15 ≈ 0.73333
      - F1 Weighted = (0.80 * 3 + (2/3) * 1) / 4 = (2.4 + 2/3) / 4 = 23/30 ≈ 0.76667
      - Matriz de confusión con labels=('positivo', 'negativo'):
          Fila 0 (positivo real): [[2, 1],
          Fila 1 (negativo real):  [0, 1]]
    """
    return {
        "texts": [
            "Excelente servicio y atención",
            "Muy buen producto recomendado",
            "Buena experiencia en general",
            "Pésimo estado, llegó defectuoso",
        ],
        "y_true": ["positivo", "positivo", "positivo", "negativo"],
        "y_pred": ["positivo", "positivo", "negativo", "negativo"],
    }


# ---------------------------------------------------------------------------
# Pruebas de métricas individuales
# ---------------------------------------------------------------------------


def test_compute_accuracy_exact(exact_eval_data: dict[str, list[str]]) -> None:
    acc = compute_accuracy(exact_eval_data["y_true"], exact_eval_data["y_pred"])
    assert acc == pytest.approx(0.75)


def test_compute_class_metrics_exact(exact_eval_data: dict[str, list[str]]) -> None:
    metrics = compute_class_metrics(
        exact_eval_data["y_true"],
        exact_eval_data["y_pred"],
        labels=EXPECTED_LABELS,
    )

    assert set(metrics.keys()) == {"positivo", "negativo"}

    pos = metrics["positivo"]
    assert pos["precision"] == pytest.approx(1.0)
    assert pos["recall"] == pytest.approx(2 / 3)
    assert pos["f1_score"] == pytest.approx(0.80)
    assert pos["support"] == 3

    neg = metrics["negativo"]
    assert neg["precision"] == pytest.approx(0.50)
    assert neg["recall"] == pytest.approx(1.0)
    assert neg["f1_score"] == pytest.approx(2 / 3)
    assert neg["support"] == 1


def test_compute_f1_scores_exact(exact_eval_data: dict[str, list[str]]) -> None:
    f1_m = compute_f1_macro(exact_eval_data["y_true"], exact_eval_data["y_pred"])
    f1_w = compute_f1_weighted(exact_eval_data["y_true"], exact_eval_data["y_pred"])
    f1_dict = compute_f1_scores(exact_eval_data["y_true"], exact_eval_data["y_pred"])

    expected_macro = (0.80 + (2 / 3)) / 2  # 11/15 ≈ 0.733333
    expected_weighted = (0.80 * 3 + (2 / 3) * 1) / 4  # 23/30 ≈ 0.766667

    assert f1_m == pytest.approx(expected_macro)
    assert f1_w == pytest.approx(expected_weighted)
    assert f1_dict["f1_macro"] == pytest.approx(expected_macro)
    assert f1_dict["f1_weighted"] == pytest.approx(expected_weighted)


def test_compute_confusion_matrix_contract_and_order(
    exact_eval_data: dict[str, list[str]],
) -> None:
    # Orden contractual estándar: ('positivo', 'negativo')
    cm = compute_confusion_matrix(
        exact_eval_data["y_true"],
        exact_eval_data["y_pred"],
        labels=("positivo", "negativo"),
    )
    expected_cm = np.array([[2, 1], [0, 1]])
    np.testing.assert_array_equal(cm, expected_cm)

    # Inversión de orden documentado: ('negativo', 'positivo')
    cm_reversed = compute_confusion_matrix(
        exact_eval_data["y_true"],
        exact_eval_data["y_pred"],
        labels=("negativo", "positivo"),
    )
    expected_reversed = np.array([[1, 0], [1, 2]])
    np.testing.assert_array_equal(cm_reversed, expected_reversed)


def test_compute_classification_report(exact_eval_data: dict[str, list[str]]) -> None:
    # Como texto legible
    report_text = compute_classification_report(
        exact_eval_data["y_true"], exact_eval_data["y_pred"]
    )
    assert isinstance(report_text, str)
    assert "positivo" in report_text
    assert "negativo" in report_text
    assert "accuracy" in report_text

    # Como diccionario estructurado
    report_dict = compute_classification_report(
        exact_eval_data["y_true"], exact_eval_data["y_pred"], output_dict=True
    )
    assert isinstance(report_dict, dict)
    assert report_dict["accuracy"] == pytest.approx(0.75)
    assert "positivo" in report_dict
    assert "negativo" in report_dict


# ---------------------------------------------------------------------------
# Pruebas de análisis de errores
# ---------------------------------------------------------------------------


def test_identify_misclassifications_isolates_bad_predictions(
    exact_eval_data: dict[str, list[str]],
) -> None:
    texts = exact_eval_data["texts"]
    y_true = exact_eval_data["y_true"]
    y_pred = exact_eval_data["y_pred"]

    errors = identify_misclassifications(texts, y_true, y_pred)

    # Solo la muestra 2 está mal clasificada (real: positivo, predicho: negativo)
    assert len(errors) == 1
    err = errors[0]
    assert isinstance(err, Misclassification)
    assert err.text == texts[2]
    assert err.true_label == "positivo"
    assert err.predicted_label == "negativo"
    assert err.index == 2


def test_identify_misclassifications_with_perfect_predictions() -> None:
    texts = ["Excelente producto", "Muy malo todo"]
    labels = ["positivo", "negativo"]

    errors = identify_misclassifications(texts, labels, labels)
    assert len(errors) == 0

    formatted = format_error_analysis(errors)
    assert "0 errores" in formatted


def test_misclassifications_to_dataframe_and_formatting(
    exact_eval_data: dict[str, list[str]],
) -> None:
    errors = identify_misclassifications(
        exact_eval_data["texts"],
        exact_eval_data["y_true"],
        exact_eval_data["y_pred"],
        probabilities=[{"positivo": 0.9, "negativo": 0.1}] * 4,
    )

    df = misclassifications_to_dataframe(errors)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert list(df.columns) == ["indice", "texto", "etiqueta_real", "etiqueta_predicha"]
    assert df.loc[0, "texto"] == exact_eval_data["texts"][2]

    formatted = format_error_analysis(errors)
    assert "Se encontraron 1 muestra(s)" in formatted
    assert "Buena experiencia en general" in formatted
    assert "Real: 'positivo' -> Predicho: 'negativo'" in formatted


# ---------------------------------------------------------------------------
# Pruebas de evaluate_predictions y EvaluationReport
# ---------------------------------------------------------------------------


def test_evaluate_predictions_full_report(exact_eval_data: dict[str, list[str]]) -> None:
    report = evaluate_predictions(
        y_true=exact_eval_data["y_true"],
        y_pred=exact_eval_data["y_pred"],
        texts=exact_eval_data["texts"],
    )

    assert isinstance(report, EvaluationReport)
    assert report.accuracy == pytest.approx(0.75)
    assert report.f1_macro == pytest.approx((0.80 + 2 / 3) / 2)
    assert report.f1_weighted == pytest.approx((0.80 * 3 + (2 / 3) * 1) / 4)
    assert len(report.misclassifications) == 1

    # Acceso por contrato .metrics plano
    flat_metrics = report.metrics
    assert flat_metrics["accuracy"] == pytest.approx(0.75)
    assert "positivo_f1_score" in flat_metrics
    assert "negativo_precision" in flat_metrics

    # Formateo del resumen
    summary_text = report.summary()
    assert "REPORTE DE EVALUACIÓN DEL MODELO" in summary_text
    assert "Accuracy global: 0.7500" in summary_text
    assert "Matriz de confusión" in summary_text


# ---------------------------------------------------------------------------
# Pruebas de evaluación con Pipeline entrenado
# ---------------------------------------------------------------------------


def test_evaluate_pipeline_with_trained_estimator() -> None:
    train_df = pd.DataFrame(
        {
            "texto": [
                "Excelente producto muy recomendado",
                "Buena calidad y entrega rápida",
                "Pésima atención al cliente",
                "Muy mala experiencia producto roto",
            ],
            "sentimiento": ["positivo", "positivo", "negativo", "negativo"],
        }
    )
    pipeline = build_logistic_regression_pipeline()
    train_pipeline(pipeline, train_df["texto"], train_df["sentimiento"])

    eval_texts = [
        "Excelente servicio recomendado",
        "Pésima experiencia recibido roto",
    ]
    eval_expected = ["positivo", "negativo"]

    report = evaluate_pipeline(pipeline, eval_texts, eval_expected)

    assert isinstance(report, EvaluationReport)
    assert 0.0 <= report.accuracy <= 1.0
    assert 0.0 <= report.f1_macro <= 1.0
    assert report.confusion_matrix.shape == (2, 2)
    assert len(report.labels) == 2


def test_polymorphic_evaluate_entrypoint(exact_eval_data: dict[str, list[str]]) -> None:
    # 1. Con vectores de predicciones directas
    rep1 = evaluate(
        pipeline_or_predictions=exact_eval_data["y_pred"],
        texts_or_y_true=exact_eval_data["y_true"],
        expected_labels=exact_eval_data["texts"],
    )
    assert rep1.accuracy == pytest.approx(0.75)
    assert len(rep1.misclassifications) == 1

    # 2. Con Pipeline entrenado
    pipe = build_logistic_regression_pipeline()
    train_pipeline(
        pipe,
        exact_eval_data["texts"],
        exact_eval_data["y_true"],
    )
    rep2 = evaluate(
        pipeline_or_predictions=pipe,
        texts_or_y_true=exact_eval_data["texts"],
        expected_labels=exact_eval_data["y_true"],
    )
    assert isinstance(rep2, EvaluationReport)


# ---------------------------------------------------------------------------
# Validación y manejo de errores
# ---------------------------------------------------------------------------


def test_evaluation_error_handling() -> None:
    # Vectores vacíos
    with pytest.raises(ValueError, match="no puede estar vacío"):
        compute_accuracy([], [])

    # Tamaños discordantes
    with pytest.raises(ValueError, match="no coincide"):
        compute_accuracy(["positivo"], ["positivo", "negativo"])

    # Dimensiones discordantes en análisis de errores
    with pytest.raises(ValueError, match="no coinciden"):
        identify_misclassifications(["t1"], ["positivo"], ["positivo", "negativo"])

    # Pipeline sin método predict
    with pytest.raises(TypeError, match="predict"):
        evaluate_pipeline("no_es_pipeline", ["t1"], ["positivo"])

    # Llamada a evaluate con pipeline pero sin expected_labels
    pipe = LogisticRegression()
    with pytest.raises(ValueError, match="expected_labels"):
        evaluate(pipe, ["texto"])
