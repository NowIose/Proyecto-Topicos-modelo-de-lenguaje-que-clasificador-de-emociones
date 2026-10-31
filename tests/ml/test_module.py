"""Pruebas del esqueleto ML; no requieren ni simulan un modelo entrenado."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "src.models",
        "src.models.config",
        "src.models.dataset",
        "src.models.split",
        "src.models.train",
        "src.models.experiments",
        "src.models.evaluate",
        "src.models.inference",
    ],
)
def test_ml_modules_can_be_imported(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None
