"""Orquestación reproducible del entrenamiento de modelos candidatos de ML."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Mapping, Sequence

import pandas as pd
from sklearn.pipeline import Pipeline

from .config import MLConfig, load_config
from .dataset import DatasetValidationError, load_dataset, validate_dataset
from .experiments import get_candidate_pipelines, train_pipeline
from .split import DatasetSplit, split_dataset


@dataclass(frozen=True)
class TrainingResult:
    """Resultado del entrenamiento estructurado y reproducible de los modelos candidatos.

    Contiene los pipelines ajustados sobre los datos de entrenamiento y la partición
    utilizada (manteniendo el subconjunto de prueba intacto para futuras fases).
    """

    split: DatasetSplit
    models: Mapping[str, Pipeline]
    config: MLConfig
    data_source: str = ""

    @property
    def logistic_regression(self) -> Pipeline:
        """Devuelve el pipeline entrenado de Regresión Logística."""
        return self.models["logistic_regression"]

    @property
    def multinomial_nb(self) -> Pipeline:
        """Devuelve el pipeline entrenado de Multinomial Naive Bayes."""
        return self.models["multinomial_nb"]

    def summary(self) -> str:
        """Genera el resumen estructurado del entrenamiento."""
        return format_training_summary(self)


def format_training_summary(result: TrainingResult) -> str:
    """Genera un resumen textual del entrenamiento sin inventar métricas."""
    train_counts = result.split.y_train.value_counts()
    test_counts = result.split.y_test.value_counts()
    n_train = len(result.split.X_train)
    n_test = len(result.split.X_test)
    n_total = n_train + n_test

    lines = [
        "=" * 70,
        "RESUMEN DEL ENTRENAMIENTO DE MODELOS CANDIDATOS",
        "=" * 70,
        f"Fuente de datos: {result.data_source}",
        f"Total de registros procesados: {n_total}",
        f"División estratificada (test_size={result.config.experiment.test_size}, random_seed={result.config.experiment.random_seed}):",
        f"  - Conjunto de entrenamiento (X_train, y_train): {n_train} registros ({n_train / n_total * 100:.1f}%)",
        f"  - Conjunto de prueba reservado (X_test, y_test): {n_test} registros ({n_test / n_total * 100:.1f}%)",
        "",
        "Distribución de clases en Entrenamiento:",
    ]
    for label in result.config.dataset.labels:
        count = int(train_counts.get(label, 0))
        pct = (count / n_train * 100) if n_train > 0 else 0
        lines.append(f"  - {label}: {count} ({pct:.1f}%)")

    lines.append("")
    lines.append("Distribución de clases en Prueba (reservada para futura evaluación):")
    for label in result.config.dataset.labels:
        count = int(test_counts.get(label, 0))
        pct = (count / n_test * 100) if n_test > 0 else 0
        lines.append(f"  - {label}: {count} ({pct:.1f}%)")

    lines.append("")
    lines.append("Modelos candidatos entrenados:")
    for name, model in result.models.items():
        clf = model.named_steps.get("classifier", model[-1])
        clf_name = type(clf).__name__
        vec = model.named_steps.get("vectorizer", model[0])
        vec_name = type(vec).__name__
        lines.append(f"  - {name} ({clf_name}): Pipeline[{vec_name} -> {clf_name}]")

    lines.append("=" * 70)
    return "\n".join(lines)


def train(
    data: str | Path | pd.DataFrame | MLConfig | None = None,
    *,
    config: MLConfig | None = None,
) -> TrainingResult:
    """Orquesta el flujo reproducible de entrenamiento para ambos modelos candidatos.

    1. Carga y valida el dataset especificado o el configurado por defecto.
    2. Divide los datos en train y test de manera estratificada y reproducible.
    3. Construye los pipelines candidatos (Regresión Logística y MultinomialNB).
    4. Entrena ambos modelos exclusivamente sobre X_train e y_train.
    5. Preserva X_test e y_test reservados e inalterados para la posterior evaluación.

    Args:
        data: Ruta al archivo CSV, DataFrame en memoria, MLConfig o None para ruta default.
        config: Configuración MLConfig opcional para sobrescribir parámetros.

    Returns:
        Instancia de TrainingResult conteniendo la partición de datos y los modelos entrenados.

    Raises:
        FileNotFoundError: Si el dataset CSV indicado no existe.
        DatasetValidationError: Si los datos no cumplen con las validaciones de contrato.
    """
    if isinstance(data, MLConfig):
        active_config = data
        data_source = active_config.dataset.path
    else:
        active_config = config or load_config()
        data_source = data if data is not None else active_config.dataset.path

    # Cargar y validar dataset
    if isinstance(data_source, pd.DataFrame):
        df = validate_dataset(
            data_source,
            text_column=active_config.dataset.text_column,
            target_column=active_config.dataset.target_column,
            expected_labels=active_config.dataset.labels,
        )
        source_desc = "pandas.DataFrame (en memoria)"
    else:
        df = load_dataset(
            data_source,
            text_column=active_config.dataset.text_column,
            target_column=active_config.dataset.target_column,
            expected_labels=active_config.dataset.labels,
        )
        source_desc = str(data_source)

    # Dividir de forma estratificada y reproducible
    split = split_dataset(
        df,
        test_size=active_config.experiment.test_size,
        random_seed=active_config.experiment.random_seed,
        text_column=active_config.dataset.text_column,
        target_column=active_config.dataset.target_column,
        expected_labels=active_config.dataset.labels,
        config=active_config,
    )

    # Construir pipelines candidatos
    candidate_pipelines = get_candidate_pipelines(
        random_seed=active_config.experiment.random_seed,
        config=active_config,
    )

    # Entrenar ambos modelos sobre el mismo conjunto X_train e y_train
    trained_models: dict[str, Pipeline] = {}
    for name, pipeline in candidate_pipelines.items():
        trained_pipeline = train_pipeline(pipeline, split)
        trained_models[name] = trained_pipeline

    return TrainingResult(
        split=split,
        models=trained_models,
        config=active_config,
        data_source=source_desc,
    )


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento reproducible de los modelos candidatos de clasificación de sentimientos."
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default=None,
        help="Ruta al archivo CSV con las reseñas para el entrenamiento.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Ruta alternativa al archivo de configuración YAML.",
    )
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> int:
    """Punto de entrada de línea de comandos para la orquestación del entrenamiento."""
    parsed = parse_args(args)

    try:
        config = load_config(parsed.config) if parsed.config else load_config()
        result = train(data=parsed.data, config=config)
        print(result.summary())
        return 0
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(
            "No se encontró el dataset. Proporcione una ruta válida mediante '--data RUTA_CSV' "
            "o verifique la ruta configurada en config/ml.yaml.",
            file=sys.stderr,
        )
        return 1
    except DatasetValidationError as exc:
        print(f"Error de validación del dataset: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error inesperado durante el entrenamiento: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
