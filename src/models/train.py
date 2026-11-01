"""Punto de entrada del futuro entrenamiento reproducible.

TODO(ML): integrar el preprocesamiento de ``src.nlp``, construir el pipeline,
entrenarlo con datos reales y persistir el pipeline completo con joblib.
"""

from __future__ import annotations

from pathlib import Path

from .config import MLConfig


def train(config: MLConfig) -> Path:
    """Entrena y guarda el futuro pipeline completo.

    La firma define el límite del módulo. Se mantiene deliberadamente sin
    implementación para no adelantar la fase de entrenamiento.
    """

    del config
    raise NotImplementedError("El entrenamiento se implementará en una fase posterior.")


def main() -> None:
    """Evita que el esqueleto se confunda con un entrenamiento terminado."""

    raise SystemExit(
        "Entrenamiento no disponible: esta fase solo prepara el entorno ML."
    )


if __name__ == "__main__":
    main()
