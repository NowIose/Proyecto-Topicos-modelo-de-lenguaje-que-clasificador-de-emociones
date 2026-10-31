"""Contratos para la futura comparación reproducible de experimentos."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .config import MLConfig


@dataclass(frozen=True)
class ExperimentSpec:
    """Describe un experimento sin ejecutar ni asumir un estimador concreto."""

    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentResult:
    """Estructura para métricas reales que se obtendrán en la fase futura."""

    name: str
    metrics: Mapping[str, float]


def run_experiments(
    specs: Sequence[ExperimentSpec], config: MLConfig
) -> Sequence[ExperimentResult]:
    """Ejecutará y comparará experimentos definidos explícitamente."""

    del specs, config
    raise NotImplementedError("La experimentación se implementará en una fase posterior.")
