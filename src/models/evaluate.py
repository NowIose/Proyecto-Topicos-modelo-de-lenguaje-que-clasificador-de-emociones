"""Módulo de evaluación y diagnóstico de modelos de Machine Learning.

Implementa el cálculo de métricas estándar de clasificación (accuracy, precision,
recall, F1-scores, matriz de confusión y reporte de clasificación) así como
el análisis detallado de errores en reseñas, desacoplado de la fase de entrenamiento.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .config import EXPECTED_LABELS


@dataclass(frozen=True)
class Misclassification:
    """Detalle de una reseña clasificada erróneamente por un modelo.

    Attributes:
        text: Texto original evaluado.
        true_label: Etiqueta real u observada.
        predicted_label: Etiqueta predicha por el modelo.
        index: Índice o identificador original de la muestra (opcional).
        probabilities: Probabilidades estimadas por clase (opcional).
    """

    text: str
    true_label: str
    predicted_label: str
    index: int | str | None = None
    probabilities: Mapping[str, float] | None = None


@dataclass(frozen=True)
class EvaluationReport:
    """Contenedor neutral y estructurado para métricas calculadas y diagnóstico.

    Conserva el contrato contractual de clases ('positivo', 'negativo') y ofrece
    acceso tanto por atributos estructurados como por un mapeo plano `metrics`.
    """

    accuracy: float
    f1_macro: float
    f1_weighted: float
    per_class_metrics: Mapping[str, Mapping[str, float]]
    confusion_matrix: np.ndarray
    classification_report_str: str
    misclassifications: Sequence[Misclassification] = ()
    labels: tuple[str, ...] = EXPECTED_LABELS

    @property
    def metrics(self) -> Mapping[str, float]:
        """Diccionario plano de métricas para trazabilidad y contratos."""
        flat: dict[str, float] = {
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
        }
        for label, scores in self.per_class_metrics.items():
            for metric_name, val in scores.items():
                flat[f"{label}_{metric_name}"] = float(val)
        return flat

    def summary(self) -> str:
        """Formatea un resumen textual completo de la evaluación."""
        return format_evaluation_summary(self)


# ---------------------------------------------------------------------------
# Validaciones de vectores de etiquetas
# ---------------------------------------------------------------------------


def _validate_label_vectors(
    y_true: Sequence[str] | pd.Series,
    y_pred: Sequence[str] | pd.Series,
) -> tuple[list[str], list[str]]:
    """Valida que los vectores de etiquetas sean consistentes y no vacíos."""
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)

    if len(y_true_list) == 0:
        raise ValueError("El vector de etiquetas reales 'y_true' no puede estar vacío.")
    if len(y_pred_list) == 0:
        raise ValueError("El vector de etiquetas predichas 'y_pred' no puede estar vacío.")
    if len(y_true_list) != len(y_pred_list):
        raise ValueError(
            f"La cantidad de etiquetas reales ({len(y_true_list)}) no coincide con las predichas ({len(y_pred_list)})."
        )

    return y_true_list, y_pred_list


# ---------------------------------------------------------------------------
# Funciones individuales de cálculo de métricas (scikit-learn)
# ---------------------------------------------------------------------------


def compute_accuracy(
    y_true: Sequence[str] | pd.Series,
    y_pred: Sequence[str] | pd.Series,
) -> float:
    """Calcula la exactitud global (accuracy) de las predicciones.

    Args:
        y_true: Etiquetas reales observadas.
        y_pred: Etiquetas predichas.

    Returns:
        Exactitud en el rango [0.0, 1.0].
    """
    yt, yp = _validate_label_vectors(y_true, y_pred)
    return float(accuracy_score(yt, yp))


def compute_class_metrics(
    y_true: Sequence[str] | pd.Series,
    y_pred: Sequence[str] | pd.Series,
    *,
    labels: Sequence[str] = EXPECTED_LABELS,
) -> dict[str, dict[str, float]]:
    """Calcula precisión, recall y F1-score de forma individual para cada clase.

    Args:
        y_true: Etiquetas reales observadas.
        y_pred: Etiquetas predichas.
        labels: Orden y lista fija de clases contractuales (por defecto: ('positivo', 'negativo')).

    Returns:
        Diccionario con las métricas calculadas por clase:
        {
            "positivo": {"precision": ..., "recall": ..., "f1_score": ..., "support": ...},
            "negativo": {"precision": ..., "recall": ..., "f1_score": ..., "support": ...}
        }
    """
    yt, yp = _validate_label_vectors(y_true, y_pred)
    precision, recall, fscore, support = precision_recall_fscore_support(
        yt,
        yp,
        labels=labels,
        zero_division=0,
    )

    metrics_by_class: dict[str, dict[str, float]] = {}
    for i, label in enumerate(labels):
        metrics_by_class[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(fscore[i]),
            "support": float(support[i]),
        }
    return metrics_by_class


def compute_f1_macro(
    y_true: Sequence[str] | pd.Series,
    y_pred: Sequence[str] | pd.Series,
    *,
    labels: Sequence[str] = EXPECTED_LABELS,
) -> float:
    """Calcula el F1 macro (promedio no ponderado del F1 de cada clase)."""
    yt, yp = _validate_label_vectors(y_true, y_pred)
    return float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0))


def compute_f1_weighted(
    y_true: Sequence[str] | pd.Series,
    y_pred: Sequence[str] | pd.Series,
    *,
    labels: Sequence[str] = EXPECTED_LABELS,
) -> float:
    """Calcula el F1 weighted (promedio ponderado por el soporte de cada clase)."""
    yt, yp = _validate_label_vectors(y_true, y_pred)
    return float(f1_score(yt, yp, labels=labels, average="weighted", zero_division=0))


def compute_f1_scores(
    y_true: Sequence[str] | pd.Series,
    y_pred: Sequence[str] | pd.Series,
    *,
    labels: Sequence[str] = EXPECTED_LABELS,
) -> dict[str, float]:
    """Calcula conjuntamente el F1 macro y el F1 weighted.

    Returns:
        Diccionario con claves 'f1_macro' y 'f1_weighted'.
    """
    return {
        "f1_macro": compute_f1_macro(y_true, y_pred, labels=labels),
        "f1_weighted": compute_f1_weighted(y_true, y_pred, labels=labels),
    }


def compute_confusion_matrix(
    y_true: Sequence[str] | pd.Series,
    y_pred: Sequence[str] | pd.Series,
    *,
    labels: Sequence[str] = EXPECTED_LABELS,
) -> np.ndarray:
    """Calcula la matriz de confusión con orden de clases documentado y determinista.

    Contrato del orden de clases:
    - Las filas corresponden a las etiquetas reales (y_true).
    - Las columnas corresponden a las etiquetas predichas (y_pred).
    - Para `labels=('positivo', 'negativo')`:
        Fila 0 = 'positivo' real; Fila 1 = 'negativo' real.
        Columna 0 = 'positivo' predicho; Columna 1 = 'negativo' predicho.
        [[TP_pos, FN_pos],
         [FP_pos, TN_pos]]

    Args:
        y_true: Etiquetas reales observadas.
        y_pred: Etiquetas predichas.
        labels: Secuencia con el orden de las clases (por defecto EXPECTED_LABELS).

    Returns:
        np.ndarray bidimensional de forma (len(labels), len(labels)).
    """
    yt, yp = _validate_label_vectors(y_true, y_pred)
    return confusion_matrix(yt, yp, labels=labels)


def compute_classification_report(
    y_true: Sequence[str] | pd.Series,
    y_pred: Sequence[str] | pd.Series,
    *,
    labels: Sequence[str] = EXPECTED_LABELS,
    output_dict: bool = False,
) -> str | dict[str, Any]:
    """Genera el reporte de clasificación detallado de scikit-learn.

    Args:
        y_true: Etiquetas reales observadas.
        y_pred: Etiquetas predichas.
        labels: Secuencia de clases a evaluar.
        output_dict: Si es True, retorna un diccionario estructurado; si es False,
            retorna una cadena formateada legible.

    Returns:
        Cadena o diccionario con métricas globales y por clase.
    """
    yt, yp = _validate_label_vectors(y_true, y_pred)
    return classification_report(
        yt,
        yp,
        labels=labels,
        zero_division=0,
        output_dict=output_dict,
    )


# ---------------------------------------------------------------------------
# Análisis de errores
# ---------------------------------------------------------------------------


def identify_misclassifications(
    texts: Sequence[str] | pd.Series,
    y_true: Sequence[str] | pd.Series,
    y_pred: Sequence[str] | pd.Series,
    probabilities: Sequence[Mapping[str, float]] | None = None,
) -> list[Misclassification]:
    """Identifica las muestras donde la predicción del modelo difiere de la etiqueta real.

    Permite aislar las reseñas mal clasificadas preservando el texto original,
    la etiqueta real y la etiqueta predicha para diagnóstico de NLP.

    Args:
        texts: Colección de reseñas o textos evaluados.
        y_true: Etiquetas reales observadas.
        y_pred: Etiquetas predichas por el modelo.
        probabilities: Probabilidades estimadas por clase para cada muestra (opcional).

    Returns:
        Lista de instancias de `Misclassification` ordenadas por aparición original.

    Raises:
        ValueError: Si las dimensiones de los argumentos no concuerdan.
    """
    texts_list = list(texts)
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)

    if not (len(texts_list) == len(y_true_list) == len(y_pred_list)):
        raise ValueError(
            f"Las dimensiones no coinciden: texts={len(texts_list)}, "
            f"y_true={len(y_true_list)}, y_pred={len(y_pred_list)}."
        )

    if probabilities is not None and len(probabilities) != len(texts_list):
        raise ValueError(
            f"La cantidad de probabilidades ({len(probabilities)}) no coincide con la de textos ({len(texts_list)})."
        )

    indices = None
    if isinstance(texts, (pd.Series, pd.DataFrame)):
        indices = list(texts.index)
    elif isinstance(y_true, (pd.Series, pd.DataFrame)):
        indices = list(y_true.index)

    misclassified: list[Misclassification] = []
    for i, (text, true_lbl, pred_lbl) in enumerate(
        zip(texts_list, y_true_list, y_pred_list)
    ):
        if true_lbl != pred_lbl:
            sample_idx = indices[i] if indices is not None else i
            sample_probs = probabilities[i] if probabilities is not None else None
            misclassified.append(
                Misclassification(
                    text=str(text),
                    true_label=str(true_lbl),
                    predicted_label=str(pred_lbl),
                    index=sample_idx,
                    probabilities=sample_probs,
                )
            )

    return misclassified


def misclassifications_to_dataframe(
    misclassifications: Sequence[Misclassification],
) -> pd.DataFrame:
    """Convierte una lista de clasificaciones erróneas a un pd.DataFrame."""
    data = [
        {
            "indice": m.index,
            "texto": m.text,
            "etiqueta_real": m.true_label,
            "etiqueta_predicha": m.predicted_label,
        }
        for m in misclassifications
    ]
    return pd.DataFrame(data)


def format_error_analysis(
    misclassifications: Sequence[Misclassification],
    max_examples: int = 10,
) -> str:
    """Genera un reporte textual estructurado de los errores identificados."""
    total = len(misclassifications)
    if total == 0:
        return "No se encontraron clasificaciones erróneas (0 errores)."

    lines = [
        f"Se encontraron {total} muestra(s) clasificadas erróneamente:",
        "-" * 65,
    ]
    for i, item in enumerate(misclassifications[:max_examples], start=1):
        idx_str = f" [Índice: {item.index}]" if item.index is not None else ""
        lines.append(f"[{i}]{idx_str}")
        lines.append(f'  Texto: "{item.text}"')
        lines.append(f"  Real: '{item.true_label}' -> Predicho: '{item.predicted_label}'")
        if item.probabilities:
            probs_str = ", ".join(f"{k}: {v:.3f}" for k, v in item.probabilities.items())
            lines.append(f"  Probabilidades: {{{probs_str}}}")
        lines.append("-" * 65)

    if total > max_examples:
        lines.append(f"... y {total - max_examples} muestra(s) con error adicionales.")

    return "\n".join(lines)


def format_evaluation_summary(report: EvaluationReport) -> str:
    """Formatea un resumen legible de las métricas y la matriz de confusión."""
    lines = [
        "=" * 65,
        "REPORTE DE EVALUACIÓN DEL MODELO",
        "=" * 65,
        f"Accuracy global: {report.accuracy:.4f}",
        f"F1 Macro:        {report.f1_macro:.4f}",
        f"F1 Weighted:     {report.f1_weighted:.4f}",
        "",
        "Métricas por clase:",
    ]
    for label, metrics in report.per_class_metrics.items():
        prec = metrics.get("precision", 0.0)
        rec = metrics.get("recall", 0.0)
        f1 = metrics.get("f1_score", 0.0)
        sup = int(metrics.get("support", 0))
        lines.append(
            f"  - {label:<10} -> Precisión: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} (Soporte: {sup})"
        )

    lines.append("")
    lines.append(
        f"Matriz de confusión (Filas: Real, Columnas: Predicho | Clases: {report.labels}):"
    )
    header = "          " + "  ".join(f"{lbl[:8]:>8}" for lbl in report.labels)
    lines.append(header)
    for i, true_lbl in enumerate(report.labels):
        row_vals = "  ".join(f"{val:>8d}" for val in report.confusion_matrix[i])
        lines.append(f"{true_lbl[:8]:>8}: {row_vals}")

    lines.append("")
    lines.append(f"Total de muestras mal clasificadas: {len(report.misclassifications)}")
    lines.append("=" * 65)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orquestación de la evaluación (desacoplada del entrenamiento)
# ---------------------------------------------------------------------------


def evaluate_predictions(
    y_true: Sequence[str] | pd.Series,
    y_pred: Sequence[str] | pd.Series,
    texts: Sequence[str] | pd.Series | None = None,
    probabilities: Sequence[Mapping[str, float]] | None = None,
    *,
    labels: Sequence[str] = EXPECTED_LABELS,
) -> EvaluationReport:
    """Evalúa un vector de predicciones contra las etiquetas reales.

    No requiere estimador ni reentrenamiento. Si se suministran los textos
    originales, genera automáticamente el análisis de errores.

    Args:
        y_true: Etiquetas reales observadas.
        y_pred: Etiquetas predichas.
        texts: Textos originales opcionales para análisis de errores.
        probabilities: Probabilidades opcionales por muestra.
        labels: Orden y lista fija de clases contractuales.

    Returns:
        Instancia de `EvaluationReport` con todas las métricas calculadas.
    """
    yt, yp = _validate_label_vectors(y_true, y_pred)
    tuple_labels = tuple(labels)

    acc = compute_accuracy(yt, yp)
    class_metrics = compute_class_metrics(yt, yp, labels=tuple_labels)
    f1_m = compute_f1_macro(yt, yp, labels=tuple_labels)
    f1_w = compute_f1_weighted(yt, yp, labels=tuple_labels)
    conf_mat = compute_confusion_matrix(yt, yp, labels=tuple_labels)
    rep_str = compute_classification_report(yt, yp, labels=tuple_labels, output_dict=False)

    misclassified: list[Misclassification] = []
    if texts is not None:
        misclassified = identify_misclassifications(
            texts=texts,
            y_true=yt,
            y_pred=yp,
            probabilities=probabilities,
        )

    return EvaluationReport(
        accuracy=acc,
        f1_macro=f1_m,
        f1_weighted=f1_w,
        per_class_metrics=metrics_by_class_to_mapping(class_metrics),
        confusion_matrix=conf_mat,
        classification_report_str=str(rep_str),
        misclassifications=tuple(misclassified),
        labels=tuple_labels,
    )


def metrics_by_class_to_mapping(
    raw: dict[str, dict[str, float]],
) -> Mapping[str, Mapping[str, float]]:
    """Convierte el diccionario de métricas por clase a un Mapping inmutable."""
    return {k: dict(v) for k, v in raw.items()}


def evaluate_pipeline(
    pipeline: Any,
    texts: Sequence[str] | pd.Series,
    expected_labels: Sequence[str] | pd.Series,
    *,
    labels: Sequence[str] = EXPECTED_LABELS,
) -> EvaluationReport:
    """Evalúa un pipeline previamente entrenado sobre una colección de textos.

    No altera ni entrena el pipeline. Obtiene predicciones y probabilidades
    mediante el estimador y genera el `EvaluationReport`.

    Args:
        pipeline: Pipeline entrenado con métodos `predict` y opcionalmente `predict_proba`.
        texts: Textos a clasificar.
        expected_labels: Etiquetas verdaderas para validar.
        labels: Clases evaluadas en orden fijo.

    Returns:
        Instancia de `EvaluationReport`.
    """
    if not hasattr(pipeline, "predict"):
        raise TypeError(
            f"El estimador suministrado no cuenta con método 'predict': {type(pipeline).__name__}."
        )

    clean_texts = list(texts)
    y_pred = pipeline.predict(clean_texts)

    # Obtener probabilidades si el estimador las soporta y está entrenado
    probabilities: list[dict[str, float]] | None = None
    if hasattr(pipeline, "predict_proba"):
        try:
            from .experiments import predict_probabilities

            probabilities = predict_probabilities(pipeline, clean_texts)
        except Exception:
            probabilities = None

    return evaluate_predictions(
        y_true=expected_labels,
        y_pred=y_pred,
        texts=texts,
        probabilities=probabilities,
        labels=labels,
    )


def evaluate(
    pipeline_or_predictions: Any,
    texts_or_y_true: Sequence[str] | pd.Series,
    expected_labels: Sequence[str] | pd.Series | None = None,
    *,
    labels: Sequence[str] = EXPECTED_LABELS,
) -> EvaluationReport:
    """Punto de entrada neutral y polimórfico para evaluación de modelos.

    Soporta dos modalidades:
    1. Evaluación de pipeline:
       `evaluate(pipeline, texts, expected_labels)`
    2. Evaluación directa de predicciones:
       `evaluate(y_pred, y_true, texts=texts_opcionales)`

    Args:
        pipeline_or_predictions: Pipeline entrenado o vector de predicciones.
        texts_or_y_true: Colección de textos (si el 1er arg es pipeline) o etiquetas reales.
        expected_labels: Etiquetas reales (si el 1er arg es pipeline) o textos opcionales.
        labels: Clases contractuales.

    Returns:
        Instancia de `EvaluationReport`.
    """
    if hasattr(pipeline_or_predictions, "predict"):
        if expected_labels is None:
            raise ValueError(
                "Se requieren 'texts' y 'expected_labels' al evaluar un pipeline."
            )
        return evaluate_pipeline(
            pipeline=pipeline_or_predictions,
            texts=texts_or_y_true,
            expected_labels=expected_labels,
            labels=labels,
        )

    # Modalidad directa con predicciones
    y_pred = pipeline_or_predictions
    y_true = texts_or_y_true
    texts = expected_labels if isinstance(expected_labels, (Sequence, pd.Series)) else None

    return evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        texts=texts,
        labels=labels,
    )
