"""Carga y validación de la configuración del módulo de Machine Learning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "ml.yaml"
EXPECTED_LABELS = ("positivo", "negativo")


@dataclass(frozen=True)
class DatasetConfig:
    """Contrato mínimo del dataset que consumirá el entrenamiento."""

    path: Path
    text_column: str
    target_column: str
    labels: tuple[str, ...]
    test_path: Path | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    """Parámetros reproducibles comunes a los futuros experimentos."""

    random_seed: int
    test_size: float
    cv_folds: int
    primary_metric: str


@dataclass(frozen=True)
class ArtifactConfig:
    """Ubicación prevista para el pipeline serializado."""

    directory: Path
    pipeline_filename: str
    metrics_filename: str = "metricas_modelo.json"

    @property
    def pipeline_path(self) -> Path:
        """Devuelve la ruta completa prevista, sin crear el artefacto."""

        return self.directory / self.pipeline_filename

    @property
    def metrics_path(self) -> Path:
        """Ruta del JSON con las métricas que publica el backend."""
        return self.directory / self.metrics_filename


@dataclass(frozen=True)
class NLPConfig:
    """Parámetros del preprocesamiento y del TF-IDF del modelo en producción."""

    metodo_morfologico: str = "stemming"
    ngram_range: tuple[int, int] = (1, 2)
    max_features: int = 10000
    min_df: int | float = 2
    max_df: int | float = 0.90
    sublinear_tf: bool = True


@dataclass(frozen=True)
class MLConfig:
    """Configuración validada de la futura fase de Machine Learning."""

    dataset: DatasetConfig
    experiment: ExperimentConfig
    artifacts: ArtifactConfig
    nlp: NLPConfig = NLPConfig()


def _require_mapping(value: Any, section: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"La sección '{section}' debe ser un objeto YAML.")
    return value


def _project_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{field}' debe ser una ruta no vacía.")

    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{field}' debe ser un texto no vacío.")
    return value.strip()


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> MLConfig:
    """Carga y valida la configuración YAML sin asumir que exista un dataset.

    Args:
        path: Ruta al archivo YAML. Las rutas internas relativas se resuelven
            desde la raíz del repositorio, no desde el directorio actual.

    Raises:
        FileNotFoundError: Si el archivo de configuración no existe.
        ValueError: Si falta una clave o un valor no cumple el contrato.
    """

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with config_path.open("r", encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)

    root = _require_mapping(raw, "raíz")
    dataset = _require_mapping(root.get("dataset"), "dataset")
    experiment = _require_mapping(root.get("experiment"), "experiment")
    artifacts = _require_mapping(root.get("artifacts"), "artifacts")

    raw_labels = dataset.get("labels")
    if not isinstance(raw_labels, (list, tuple)):
        raise ValueError("'dataset.labels' debe ser una lista.")

    try:
        labels = tuple(raw_labels)
        random_seed = int(experiment["random_seed"])
        test_size = float(experiment["test_size"])
        cv_folds = int(experiment["cv_folds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("La configuración ML contiene valores requeridos inválidos.") from exc

    if labels != EXPECTED_LABELS:
        raise ValueError(
            "Las etiquetas deben conservar el orden contractual: "
            f"{list(EXPECTED_LABELS)}."
        )
    if random_seed < 0:
        raise ValueError("'experiment.random_seed' no puede ser negativo.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("'experiment.test_size' debe estar entre 0 y 1.")
    if cv_folds < 2:
        raise ValueError("'experiment.cv_folds' debe ser al menos 2.")

    text_column = _non_empty_string(dataset.get("text_column"), "dataset.text_column")
    target_column = _non_empty_string(
        dataset.get("target_column"), "dataset.target_column"
    )
    primary_metric = _non_empty_string(
        experiment.get("primary_metric"), "experiment.primary_metric"
    )
    pipeline_filename = _non_empty_string(
        artifacts.get("pipeline_filename"), "artifacts.pipeline_filename"
    )

    if not pipeline_filename.endswith(".joblib"):
        raise ValueError("El pipeline previsto debe usar la extensión '.joblib'.")

    test_path = (
        _project_path(dataset["test_path"], "dataset.test_path")
        if dataset.get("test_path") is not None
        else None
    )
    metrics_filename = artifacts.get("metrics_filename", "metricas_modelo.json")

    return MLConfig(
        dataset=DatasetConfig(
            path=_project_path(dataset.get("path"), "dataset.path"),
            text_column=text_column,
            target_column=target_column,
            labels=labels,
            test_path=test_path,
        ),
        experiment=ExperimentConfig(
            random_seed=random_seed,
            test_size=test_size,
            cv_folds=cv_folds,
            primary_metric=primary_metric,
        ),
        artifacts=ArtifactConfig(
            directory=_project_path(artifacts.get("directory"), "artifacts.directory"),
            pipeline_filename=pipeline_filename,
            metrics_filename=_non_empty_string(metrics_filename, "artifacts.metrics_filename"),
        ),
        nlp=_load_nlp_config(root.get("nlp")),
    )


def _load_nlp_config(raw: Any) -> NLPConfig:
    """Lee la sección opcional 'nlp'; si falta, usa los valores por defecto."""
    if raw is None:
        return NLPConfig()
    nlp = _require_mapping(raw, "nlp")
    defaults = NLPConfig()
    metodo = nlp.get("metodo_morfologico", defaults.metodo_morfologico)
    if metodo not in ("stemming", "lematizacion", "ninguno"):
        raise ValueError(
            "'nlp.metodo_morfologico' debe ser 'stemming', 'lematizacion' o 'ninguno'."
        )
    ngram = tuple(nlp.get("ngram_range", defaults.ngram_range))
    if len(ngram) != 2 or not 1 <= ngram[0] <= ngram[1]:
        raise ValueError("'nlp.ngram_range' debe ser [min, max] con 1 <= min <= max.")
    return NLPConfig(
        metodo_morfologico=metodo,
        ngram_range=(int(ngram[0]), int(ngram[1])),
        max_features=int(nlp.get("max_features", defaults.max_features)),
        min_df=nlp.get("min_df", defaults.min_df),
        max_df=nlp.get("max_df", defaults.max_df),
        sublinear_tf=bool(nlp.get("sublinear_tf", defaults.sublinear_tf)),
    )
