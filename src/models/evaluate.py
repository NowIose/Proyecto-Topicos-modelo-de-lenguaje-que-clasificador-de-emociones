"""Interfaces para la futura evaluación de modelos con datos reales."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class EvaluationReport:
    """Contenedor neutral para métricas calculadas, nunca inventadas."""

    metrics: Mapping[str, float]


def evaluate(
    pipeline: Any, texts: Sequence[str], expected_labels: Sequence[str]
) -> EvaluationReport:
    """Evaluará un pipeline ya entrenado en una fase posterior."""

    del pipeline, texts, expected_labels
    raise NotImplementedError("La evaluación se implementará en una fase posterior.")
