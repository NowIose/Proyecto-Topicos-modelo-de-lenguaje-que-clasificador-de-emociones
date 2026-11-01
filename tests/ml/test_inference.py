"""Pruebas unitarias para la inferencia de modelos ML entrenados (Fase 3.6A)."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from src.models.experiments import (
    TextPreprocessorAdapter,
    build_logistic_regression_pipeline,
    build_multinomial_nb_pipeline,
)
from src.models.inference import (
    Prediction,
    SentimentClassifier,
    get_pipeline_classes,
    load_pipeline,
    predict,
    predict_batch,
    validate_pipeline,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# Fixtures para entrenamiento rápido de pipelines pequeños en pruebas
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_training_data() -> tuple[list[str], list[str]]:
    """Colección pequeña de 10 reseñas balanceadas para entrenar pipelines en pruebas."""
    texts = [
        "Excelente producto y servicio totalmente impecable",
        "Muy buena atención recibida en la tienda online",
        "Me encantó la rapidez del envío del pedido",
        "Experiencia genial y absolutamente recomendable",
        "Totalmente satisfecho con la compra realizada",
        "Pésima atención al cliente, no lo recomiendo",
        "Muy mala calidad del producto que llegó roto",
        "Llegó con mucho retraso y el paquete roto",
        "No sirve para nada este servicio tan deficiente",
        "Terrible experiencia vivida con este pésimo vendedor",
    ]
    labels = [
        "positivo",
        "positivo",
        "positivo",
        "positivo",
        "positivo",
        "negativo",
        "negativo",
        "negativo",
        "negativo",
        "negativo",
    ]
    return texts, labels


@pytest.fixture
def trained_lr_pipeline(sample_training_data: tuple[list[str], list[str]]) -> Pipeline:
    """Pipeline de Regresión Logística ajustado con datos sintéticos."""
    texts, labels = sample_training_data
    pipe = build_logistic_regression_pipeline(random_state=42)
    pipe.fit(texts, labels)
    return pipe


@pytest.fixture
def trained_nb_pipeline(sample_training_data: tuple[list[str], list[str]]) -> Pipeline:
    """Pipeline de Multinomial Naive Bayes ajustado con datos sintéticos."""
    texts, labels = sample_training_data
    pipe = build_multinomial_nb_pipeline()
    pipe.fit(texts, labels)
    return pipe


# ---------------------------------------------------------------------------
# Pruebas de carga del pipeline (load_pipeline) y validación
# ---------------------------------------------------------------------------


def test_load_pipeline_success(trained_lr_pipeline: Pipeline, tmp_path: Path) -> None:
    """Verifica que un pipeline compatible previamente guardado se cargue correctamente."""
    model_path = tmp_path / "sentiment_model.joblib"
    joblib.dump(trained_lr_pipeline, model_path)

    loaded = load_pipeline(model_path)

    assert isinstance(loaded, Pipeline)
    assert hasattr(loaded, "classes_") or hasattr(loaded.named_steps["classifier"], "classes_")
    classes = get_pipeline_classes(loaded)
    assert set(classes) == {"positivo", "negativo"}


def test_load_pipeline_file_not_found(tmp_path: Path) -> None:
    """Verifica que se lance FileNotFoundError si el archivo joblib no existe."""
    non_existent = tmp_path / "does_not_exist.joblib"

    with pytest.raises(FileNotFoundError) as exc_info:
        load_pipeline(non_existent)

    assert "No se encontró el pipeline" in str(exc_info.value)


def test_load_pipeline_invalid_path_types() -> None:
    """Verifica que se validen tipos y rutas vacías."""
    with pytest.raises(ValueError, match="no puede estar vacía"):
        load_pipeline("")

    with pytest.raises(ValueError, match="no puede estar vacía"):
        load_pipeline("   ")

    with pytest.raises(TypeError, match="La ruta debe ser str, Path o None"):
        load_pipeline(12345)  # type: ignore


def test_load_pipeline_directory_instead_of_file(tmp_path: Path) -> None:
    """Verifica que se lance ValueError si la ruta corresponde a un directorio."""
    with pytest.raises(ValueError, match="no es un archivo regular"):
        load_pipeline(tmp_path)


def test_load_pipeline_incompatible_object(tmp_path: Path) -> None:
    """Verifica que se lance TypeError si el archivo contiene un objeto que no es Pipeline."""
    invalid_file = tmp_path / "not_a_pipeline.joblib"
    joblib.dump({"modelo": "falso"}, invalid_file)

    with pytest.raises(TypeError) as exc_info:
        load_pipeline(invalid_file)

    assert "Pipeline compatible" in str(exc_info.value)


def test_load_pipeline_unfitted_pipeline(tmp_path: Path) -> None:
    """Verifica que se rechace un pipeline que no ha sido entrenado (sin classes_)."""
    unfitted_pipe = build_logistic_regression_pipeline()
    unfitted_file = tmp_path / "unfitted_model.joblib"
    joblib.dump(unfitted_pipe, unfitted_file)

    with pytest.raises(ValueError) as exc_info:
        load_pipeline(unfitted_file)

    assert "classes_" in str(exc_info.value)


def test_load_pipeline_corrupted_file(tmp_path: Path) -> None:
    """Verifica que se informe un error claro si el archivo está corrupto."""
    corrupted_file = tmp_path / "corrupted.joblib"
    corrupted_file.write_bytes(b"esto no es un archivo joblib valido")

    with pytest.raises(ValueError, match="Error al deserializar"):
        load_pipeline(corrupted_file)


# ---------------------------------------------------------------------------
# Pruebas de inferencia unitaria: predict(pipeline, text)
# ---------------------------------------------------------------------------


def test_predict_positive_review(trained_lr_pipeline: Pipeline) -> None:
    """Verifica la predicción de una reseña positiva clara."""
    text = "Excelente atención y entrega sumamente rápida"
    result = predict(trained_lr_pipeline, text)

    assert isinstance(result, Prediction)
    assert result.label == "positivo"
    assert "positivo" in result.probabilities
    assert "negativo" in result.probabilities
    assert result.probabilities["positivo"] > result.probabilities["negativo"]
    assert result.confidence == pytest.approx(result.probabilities["positivo"])
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-5)


def test_predict_negative_review(trained_nb_pipeline: Pipeline) -> None:
    """Verifica la predicción de una reseña negativa clara."""
    text = "Pésimo servicio, muy mala calidad y llegó roto"
    result = predict(trained_nb_pipeline, text)

    assert isinstance(result, Prediction)
    assert result.label == "negativo"
    assert result.probabilities["negativo"] > result.probabilities["positivo"]
    assert result.confidence == pytest.approx(result.probabilities["negativo"])
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("empty_input", ["", "   ", "\t\n\r  "])
def test_predict_empty_text_raises(trained_lr_pipeline: Pipeline, empty_input: str) -> None:
    """Verifica que se rechacen textos vacíos o formados solo por espacios."""
    with pytest.raises(ValueError, match="no puede estar vacío"):
        predict(trained_lr_pipeline, empty_input)


@pytest.mark.parametrize("invalid_input", [None, 123, 45.6, ["reseña"], {"texto": "hola"}])
def test_predict_invalid_text_type_raises(trained_lr_pipeline: Pipeline, invalid_input: Any) -> None:
    """Verifica que se rechacen tipos de datos no permitidos para el texto."""
    with pytest.raises(TypeError):
        predict(trained_lr_pipeline, invalid_input)  # type: ignore


def test_predict_invalid_pipeline_raises() -> None:
    """Verifica que se rechace un pipeline inválido o nulo."""
    with pytest.raises(TypeError, match="pipeline no puede ser None"):
        predict(None, "Texto de prueba")

    with pytest.raises(TypeError, match="Pipeline compatible"):
        predict("no_es_pipeline", "Texto de prueba")


# ---------------------------------------------------------------------------
# Pruebas de classes_ dinámico y probabilidades
# ---------------------------------------------------------------------------


def test_predict_classes_order_is_dynamic() -> None:
    """Verifica que el orden de clases se obtenga de classes_ sin asumir posiciones fijas."""
    texts = ["positivo uno", "positivo dos", "negativo uno", "negativo dos"]

    # Caso 1: clases ordenadas A -> B
    pipe_a = build_logistic_regression_pipeline()
    pipe_a.fit(texts, ["alegria", "alegria", "tristeza", "tristeza"])
    classes_a = get_pipeline_classes(pipe_a)
    pred_a = predict(pipe_a, "alegria total")

    assert set(pred_a.probabilities.keys()) == set(classes_a)
    for c in classes_a:
        assert c in pred_a.probabilities

    # Caso 2: clases con etiquetas invertidas
    pipe_b = build_logistic_regression_pipeline()
    pipe_b.fit(texts, ["Z_label", "Z_label", "A_label", "A_label"])
    classes_b = get_pipeline_classes(pipe_b)
    pred_b = predict(pipe_b, "Z_label texto")

    assert list(pred_b.probabilities.keys()) == classes_b


# ---------------------------------------------------------------------------
# Pruebas de inferencia en batch: predict_batch(pipeline, texts)
# ---------------------------------------------------------------------------


def test_predict_batch_vectorized(trained_lr_pipeline: Pipeline) -> None:
    """Verifica la inferencia vectorizada de múltiples reseñas a la vez."""
    batch_texts = [
        "Excelente servicio y atención impecable",
        "Pésimo servicio, muy mala calidad y paquete roto",
        "Muy buena atención y me encantó el producto",
    ]

    predictions = predict_batch(trained_lr_pipeline, batch_texts)

    assert len(predictions) == 3
    assert all(isinstance(p, Prediction) for p in predictions)
    assert predictions[0].label == "positivo"
    assert predictions[1].label == "negativo"
    assert predictions[2].label == "positivo"


def test_predict_batch_validations(trained_lr_pipeline: Pipeline) -> None:
    """Verifica validaciones para la colección de textos en predict_batch."""
    # Rechaza strings individuales en predict_batch (debe usarse lista)
    with pytest.raises(TypeError, match="secuencia de textos"):
        predict_batch(trained_lr_pipeline, "un solo string")  # type: ignore

    # Rechaza lista vacía
    with pytest.raises(ValueError, match="no puede estar vacía"):
        predict_batch(trained_lr_pipeline, [])

    # Rechaza lista que contiene un texto vacío
    with pytest.raises(ValueError, match="no puede estar vacío"):
        predict_batch(trained_lr_pipeline, ["Texto válido", "   "])

    # Rechaza lista con elementos no string
    with pytest.raises(TypeError, match="debe ser de tipo str"):
        predict_batch(trained_lr_pipeline, ["Texto válido", 123])  # type: ignore


# ---------------------------------------------------------------------------
# Pruebas de la clase de servicio SentimentClassifier (Integración FastAPI)
# ---------------------------------------------------------------------------


def test_sentiment_classifier_service(trained_lr_pipeline: Pipeline, tmp_path: Path) -> None:
    """Verifica la clase de servicio SentimentClassifier para Arteaga."""
    model_path = tmp_path / "model.joblib"
    joblib.dump(trained_lr_pipeline, model_path)

    # Instanciación desde ruta de archivo
    classifier = SentimentClassifier(model_path)

    assert set(classifier.classes) == {"positivo", "negativo"}
    assert classifier.model_path == model_path

    # Predicción unitaria
    pred = classifier.predict("Excelente servicio y atención rápida")
    assert isinstance(pred, Prediction)
    assert pred.label == "positivo"

    # Predicción en batch
    batch_preds = classifier.predict_batch([
        "Me encantó el servicio impecable",
        "Horrible y pésimo servicio roto",
    ])
    assert len(batch_preds) == 2
    assert batch_preds[0].label == "positivo"
    assert batch_preds[1].label == "negativo"

    # Instanciación con pipeline ya cargado en memoria
    classifier_mem = SentimentClassifier(pipeline=trained_lr_pipeline)
    assert classifier_mem.predict("Excelente producto").label == "positivo"


def test_prediction_to_dict_serialization(trained_lr_pipeline: Pipeline) -> None:
    """Verifica la serialización a diccionario de Prediction para respuestas JSON en FastAPI."""
    pred = predict(trained_lr_pipeline, "Excelente atención recibida")
    data = pred.to_dict()

    assert isinstance(data, dict)
    assert "label" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert data["label"] == "positivo"
    assert isinstance(data["confidence"], float)
    assert isinstance(data["probabilities"], dict)
    assert set(data["probabilities"].keys()) == {"positivo", "negativo"}


# ---------------------------------------------------------------------------
# Pruebas de compatibilidad con preprocesamiento en el Pipeline
# ---------------------------------------------------------------------------


class DummyNLPTransformer(BaseEstimator, TransformerMixin):
    """Transformador mock que verifica que el preprocesamiento corra en inferencia sin fit."""

    def __init__(self) -> None:
        self.fit_called_ = 0
        self.transform_called_ = 0

    def fit(self, X: Any, y: Any = None) -> "DummyNLPTransformer":
        self.fit_called_ += 1
        return self

    def transform(self, X: Any) -> list[str]:
        self.transform_called_ += 1
        return [str(t).lower() for t in X]


def test_inference_runs_preprocessing_without_fitting(tmp_path: Path) -> None:
    """Verifica que el pipeline con preprocesador ejecute transform pero NUNCA fit durante inferencia."""
    nlp_step = DummyNLPTransformer()
    pipe = build_logistic_regression_pipeline(preprocessor=nlp_step)

    train_x = ["Excelente servicio", "Mala atencion"]
    train_y = ["positivo", "negativo"]
    pipe.fit(train_x, train_y)

    assert nlp_step.fit_called_ == 1
    fit_count_after_training = nlp_step.fit_called_

    # Guardar y cargar
    model_path = tmp_path / "pipe_with_nlp.joblib"
    joblib.dump(pipe, model_path)
    loaded_pipe = load_pipeline(model_path)

    # Inferencia
    result = predict(loaded_pipe, "EXCELENTE SERVICIO NUEVO")
    assert result.label == "positivo"

    # Verificar que el preprocesador interno se transformó pero NO se reajustó
    loaded_nlp = loaded_pipe.named_steps["preprocessor"]
    assert loaded_nlp.fit_called_ == fit_count_after_training
    assert loaded_nlp.transform_called_ >= 1
