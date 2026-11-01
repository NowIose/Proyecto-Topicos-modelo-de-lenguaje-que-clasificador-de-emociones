"""Contrato de carga y predicción que consumirá el futuro backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class Prediction:
    """Respuesta interna esperada por la integración con la API."""

    label: str
    probabilities: Mapping[str, float]


def load_pipeline(path: str | Path) -> Any:
    """Cargará con joblib un pipeline previamente entrenado y versionado."""

    del path
    raise NotImplementedError("La carga del pipeline se implementará posteriormente.")


def predict(pipeline: Any, text: str) -> Prediction:
    """Predecirá una etiqueta y probabilidades usando el mismo pipeline."""

    del pipeline, text
    raise NotImplementedError("La inferencia se implementará posteriormente.")
