"""Pruebas de la configuración reproducible del módulo ML."""

from pathlib import Path

from src.models.config import EXPECTED_LABELS, PROJECT_ROOT, load_config


def test_default_config_loads_expected_contract() -> None:
    config = load_config()

    assert config.experiment.random_seed == 42
    assert config.dataset.text_column == "texto"
    assert config.dataset.target_column == "sentimiento"
    assert config.dataset.labels == EXPECTED_LABELS
    assert config.artifacts.pipeline_path.name == "modelo_sentimientos.joblib"


def test_config_paths_are_resolved_from_project_root() -> None:
    config = load_config()

    assert config.dataset.path.is_absolute()
    assert config.artifacts.directory.is_absolute()
    assert config.dataset.path == PROJECT_ROOT / "data" / "processed" / "train_binario.csv"
    assert config.artifacts.directory == Path(PROJECT_ROOT) / "artifacts"


def test_config_incluye_test_metricas_y_nlp() -> None:
    config = load_config()

    assert config.dataset.test_path == PROJECT_ROOT / "data" / "processed" / "test_binario.csv"
    assert config.artifacts.metrics_path.name == "metricas_modelo.json"
    assert config.nlp.metodo_morfologico in ("stemming", "lematizacion", "ninguno")
    assert config.nlp.ngram_range[0] <= config.nlp.ngram_range[1]
