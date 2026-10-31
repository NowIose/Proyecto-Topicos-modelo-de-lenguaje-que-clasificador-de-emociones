"""Pruebas unitarias para la construcción, entrenamiento e inferencia de modelos candidatos."""

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

from src.models.config import load_config
from src.models.experiments import (
    ExperimentResult,
    ExperimentSpec,
    build_logistic_regression_pipeline,
    build_multinomial_nb_pipeline,
    build_tfidf_vectorizer,
    get_candidate_pipelines,
    predict,
    predict_pipeline,
    predict_proba,
    predict_proba_pipeline,
    predict_probabilities,
    run_experiments,
    train_pipeline,
)
from src.models.split import DatasetSplit, split_dataset


@pytest.fixture
def sample_training_df() -> pd.DataFrame:
    """Dataset pequeño sintético balanceado para entrenamiento de pruebas."""
    return pd.DataFrame(
        {
            "texto": [
                "Excelente producto muy recomendado y genial",
                "Buena calidad y atención impecable",
                "Maravillosa experiencia totalmente satisfecho",
                "El envío fue rápido y el producto perfecto",
                "Muy contento con el servicio recibido",
                "Pésima atención y pésimo producto",
                "Muy mala calidad, totalmente decepcionado",
                "Llegó roto y con mucho retraso",
                "No lo recomiendo para nada, pésimo",
                "Terrible compra, una completa estafa",
            ],
            "sentimiento": [
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
            ],
        }
    )


# ---------------------------------------------------------------------------
# Construcción del vectorizador TF-IDF
# ---------------------------------------------------------------------------


def test_build_tfidf_vectorizer_defaults() -> None:
    vec = build_tfidf_vectorizer()

    assert isinstance(vec, TfidfVectorizer)
    assert vec.ngram_range == (1, 1)
    assert vec.lowercase is True


def test_build_tfidf_vectorizer_custom_params() -> None:
    vec = build_tfidf_vectorizer(ngram_range=(1, 2), max_features=50, lowercase=False)

    assert isinstance(vec, TfidfVectorizer)
    assert vec.ngram_range == (1, 2)
    assert vec.max_features == 50
    assert vec.lowercase is False


# ---------------------------------------------------------------------------
# Construcción del Pipeline de Regresión Logística
# ---------------------------------------------------------------------------


def test_build_logistic_regression_pipeline_contract_defaults() -> None:
    pipeline = build_logistic_regression_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert "vectorizer" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps

    vectorizer = pipeline.named_steps["vectorizer"]
    classifier = pipeline.named_steps["classifier"]

    assert isinstance(vectorizer, TfidfVectorizer)
    assert vectorizer.ngram_range == (1, 1)
    assert isinstance(classifier, LogisticRegression)
    assert classifier.random_state == 42
    assert classifier.max_iter == 1000


def test_build_logistic_regression_pipeline_custom_params() -> None:
    custom_vec = TfidfVectorizer(ngram_range=(1, 2))
    pipeline = build_logistic_regression_pipeline(
        random_state=123,
        max_iter=500,
        vectorizer=custom_vec,
        C=0.5,
    )

    classifier = pipeline.named_steps["classifier"]
    assert pipeline.named_steps["vectorizer"] is custom_vec
    assert classifier.random_state == 123
    assert classifier.max_iter == 500
    assert classifier.C == 0.5


def test_build_logistic_regression_pipeline_with_config() -> None:
    config = load_config()
    pipeline = build_logistic_regression_pipeline(config=config)

    classifier = pipeline.named_steps["classifier"]
    assert classifier.random_state == config.experiment.random_seed


# ---------------------------------------------------------------------------
# Construcción del Pipeline de Multinomial Naive Bayes
# ---------------------------------------------------------------------------


def test_build_multinomial_nb_pipeline_contract_defaults() -> None:
    pipeline = build_multinomial_nb_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert "vectorizer" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps

    vectorizer = pipeline.named_steps["vectorizer"]
    classifier = pipeline.named_steps["classifier"]

    assert isinstance(vectorizer, TfidfVectorizer)
    assert vectorizer.ngram_range == (1, 1)
    assert isinstance(classifier, MultinomialNB)


def test_build_multinomial_nb_pipeline_custom_params() -> None:
    custom_vec = TfidfVectorizer(ngram_range=(1, 2))
    pipeline = build_multinomial_nb_pipeline(
        vectorizer=custom_vec,
        alpha=0.25,
    )

    classifier = pipeline.named_steps["classifier"]
    assert pipeline.named_steps["vectorizer"] is custom_vec
    assert classifier.alpha == 0.25


# ---------------------------------------------------------------------------
# Obtención de ambos candidatos
# ---------------------------------------------------------------------------


def test_get_candidate_pipelines() -> None:
    candidates = get_candidate_pipelines()

    assert set(candidates.keys()) == {"logistic_regression", "multinomial_nb"}
    assert isinstance(candidates["logistic_regression"].named_steps["classifier"], LogisticRegression)
    assert isinstance(candidates["multinomial_nb"].named_steps["classifier"], MultinomialNB)


# ---------------------------------------------------------------------------
# Entrenamiento de pipelines y prevención de fuga de información
# ---------------------------------------------------------------------------


def test_train_pipeline_with_dataset_split(sample_training_df: pd.DataFrame) -> None:
    split = split_dataset(sample_training_df, test_size=0.20, random_seed=42)
    pipeline = build_logistic_regression_pipeline()

    fitted = train_pipeline(pipeline, split)

    assert fitted is pipeline
    assert hasattr(fitted.named_steps["classifier"], "classes_")
    assert hasattr(fitted.named_steps["vectorizer"], "vocabulary_")


def test_train_pipeline_with_series(sample_training_df: pd.DataFrame) -> None:
    split = split_dataset(sample_training_df, test_size=0.20, random_seed=42)
    pipeline = build_multinomial_nb_pipeline()

    fitted = train_pipeline(pipeline, split.X_train, split.y_train)

    assert fitted is pipeline
    assert hasattr(fitted.named_steps["classifier"], "classes_")
    assert hasattr(fitted.named_steps["vectorizer"], "vocabulary_")


def test_train_pipeline_no_data_leakage() -> None:
    """Verifica que el vocabulario del vectorizador se ajusta únicamente con datos de entrenamiento."""
    # Palabra única presente solo en el conjunto de prueba
    train_texts = pd.Series([
        "buen servicio recibido",
        "excelente compra realizada",
        "pesima atencion al cliente",
        "muy malo todo",
    ])
    train_labels = pd.Series(["positivo", "positivo", "negativo", "negativo"])

    pipeline = build_logistic_regression_pipeline()
    train_pipeline(pipeline, train_texts, train_labels)

    vocabulary = pipeline.named_steps["vectorizer"].vocabulary_
    assert "servicio" in vocabulary
    assert "excelente" in vocabulary
    assert "palabraineditadeltest" not in vocabulary


def test_train_pipeline_validation_errors(sample_training_df: pd.DataFrame) -> None:
    split = split_dataset(sample_training_df, test_size=0.20, random_seed=42)
    pipeline = build_logistic_regression_pipeline()

    with pytest.raises(TypeError, match="sklearn.pipeline.Pipeline"):
        train_pipeline("no_es_pipeline", split)  # type: ignore

    with pytest.raises(ValueError, match="No se debe proporcionar 'y_train'"):
        train_pipeline(pipeline, split, y_train=split.y_train)

    with pytest.raises(ValueError, match="Se requiere 'y_train'"):
        train_pipeline(pipeline, split.X_train)

    with pytest.raises(ValueError, match="X_train no puede estar vacío"):
        train_pipeline(pipeline, pd.Series(dtype=str), pd.Series(dtype=str))

    with pytest.raises(ValueError, match="y_train no puede estar vacío"):
        train_pipeline(pipeline, split.X_train, pd.Series(dtype=str))

    with pytest.raises(ValueError, match="no coincide"):
        train_pipeline(pipeline, split.X_train, split.y_train.iloc[:1])


# ---------------------------------------------------------------------------
# Predicción y probabilidades
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_factory", [
    build_logistic_regression_pipeline,
    build_multinomial_nb_pipeline,
])
def test_predict_and_predict_proba_contracts(
    model_factory,
    sample_training_df: pd.DataFrame,
) -> None:
    split = split_dataset(sample_training_df, test_size=0.20, random_seed=42)
    pipeline = train_pipeline(model_factory(), split)

    # 1. Predicciones con Series
    preds = predict(pipeline, split.X_test)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(split.X_test)
    for p in preds:
        assert p in ("positivo", "negativo")

    # 2. Predicción con texto individual
    single_pred = predict(pipeline, "Excelente atención y entrega rápida")
    assert isinstance(single_pred, np.ndarray)
    assert len(single_pred) == 1

    # 3. Probabilidades en matriz
    proba_matrix = predict_proba(pipeline, split.X_test)
    assert isinstance(proba_matrix, np.ndarray)
    assert proba_matrix.shape == (len(split.X_test), 2)
    # Las probabilidades por fila deben sumar aproximadamente 1.0
    row_sums = np.sum(proba_matrix, axis=1)
    assert np.allclose(row_sums, 1.0)

    # 4. Probabilidades estructuradas en diccionario
    proba_dicts = predict_probabilities(pipeline, split.X_test)
    assert isinstance(proba_dicts, list)
    assert len(proba_dicts) == len(split.X_test)
    for p_dict in proba_dicts:
        assert isinstance(p_dict, dict)
        assert set(p_dict.keys()) == {"negativo", "positivo"}
        for prob in p_dict.values():
            assert isinstance(prob, float)
            assert 0.0 <= prob <= 1.0
        assert np.isclose(sum(p_dict.values()), 1.0)

    # 5. Probabilidades estructuradas con texto único
    single_prob_dict = predict_probabilities(pipeline, "Producto muy malo y defectuoso")
    assert len(single_prob_dict) == 1
    assert set(single_prob_dict[0].keys()) == {"negativo", "positivo"}


def test_predict_pipeline_aliases(sample_training_df: pd.DataFrame) -> None:
    split = split_dataset(sample_training_df, test_size=0.20, random_seed=42)
    pipeline = train_pipeline(build_logistic_regression_pipeline(), split)

    assert (predict_pipeline(pipeline, split.X_test) == predict(pipeline, split.X_test)).all()
    assert np.allclose(
        predict_proba_pipeline(pipeline, split.X_test),
        predict_proba(pipeline, split.X_test),
    )


def test_prediction_error_handling() -> None:
    pipeline = build_logistic_regression_pipeline()

    with pytest.raises(TypeError, match="sklearn.pipeline.Pipeline"):
        predict("no_pipeline", ["texto"])  # type: ignore

    with pytest.raises(TypeError, match="sklearn.pipeline.Pipeline"):
        predict_proba("no_pipeline", ["texto"])  # type: ignore

    with pytest.raises(TypeError, match="sklearn.pipeline.Pipeline"):
        predict_probabilities("no_pipeline", ["texto"])  # type: ignore

    with pytest.raises(ValueError, match="no puede estar vacía"):
        predict(pipeline, [])

    with pytest.raises(ValueError, match="no puede estar vacía"):
        predict_proba(pipeline, [])

    with pytest.raises(ValueError, match="no puede estar vacía"):
        predict_probabilities(pipeline, [])

    with pytest.raises(ValueError, match="classes_"):
        # Pipeline no entrenado aún
        predict_probabilities(pipeline, ["texto de prueba"])


# ---------------------------------------------------------------------------
# Respeto de estructuras existentes de experimentos
# ---------------------------------------------------------------------------


def test_unimplemented_run_experiments_contract() -> None:
    spec = ExperimentSpec(name="provisional_test")
    config = load_config()

    with pytest.raises(NotImplementedError):
        run_experiments([spec], config)


def test_experiment_spec_and_result_dataclasses() -> None:
    spec = ExperimentSpec(name="exp_1", parameters={"alpha": 1.0})
    assert spec.name == "exp_1"
    assert spec.parameters == {"alpha": 1.0}

    result = ExperimentResult(name="exp_1", metrics={"f1": 0.85})
    assert result.name == "exp_1"
    assert result.metrics["f1"] == 0.85
