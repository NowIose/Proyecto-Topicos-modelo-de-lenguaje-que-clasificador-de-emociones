"""Pruebas unitarias para la comparación reproducible entre técnicas de preprocesamiento NLP (Fase 3.5B)."""

import io
import tempfile
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline

from src.models.experiments import (
    PREPROCESSING_TECHNIQUES,
    PreprocessingComparisonResult,
    TextPreprocessorAdapter,
    build_logistic_regression_pipeline,
    build_multinomial_nb_pipeline,
    build_preprocessing_comparison_pipelines,
    compare_preprocessing_techniques,
    format_preprocessing_comparison,
    get_nlp_preprocessors,
)
from src.models.split import DatasetSplit, split_dataset

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# Funciones y transformadores mock exclusivos para testing
# ---------------------------------------------------------------------------


def mock_stemmer_func(text: str) -> str:
    """Función de prueba que simula un stemming básico truncando sufijos."""
    words = text.split()
    stems = [
        w[:-5] if w.endswith("mente") else (w[:-3] if w.endswith("ado") or w.endswith("ido") else w)
        for w in words
    ]
    return " ".join(stems)


def mock_lemmatizer_func(text: str) -> str:
    """Función de prueba que simula una lematización convirtiendo a minúsculas y normalizando."""
    return text.lower().strip()


class MockTrackingTransformer(BaseEstimator, TransformerMixin):
    """Transformador mock que registra las muestras vistas en `fit` para verificar cero fuga."""

    fitted_batches: list[list[str]] = []

    def __init__(self, prefix: str = "mock") -> None:
        self.prefix = prefix
        self.fitted_samples_: list[str] = []

    def fit(self, X: Any, y: Any = None) -> "MockTrackingTransformer":
        samples = list(X)
        self.fitted_samples_ = samples
        MockTrackingTransformer.fitted_batches.append(samples)
        return self

    def transform(self, X: Any) -> list[str]:
        return [f"{self.prefix}_{t}" for t in X]


# ---------------------------------------------------------------------------
# Fixture sintética balanceada con 20 muestras (suficiente para 5 folds)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_dataframe_20() -> pd.DataFrame:
    """Dataset balanceado de 20 textos (10 positivos, 10 negativos) para validación cruzada."""
    return pd.DataFrame(
        {
            "texto": [
                "Excelente producto y servicio totalmente impecable",
                "Muy buena atención recibida en la tienda online",
                "Me encantó la rapidez del envío del pedido realizado",
                "Experiencia genial y absolutamente recomendable para todos",
                "Totalmente satisfecho con la compra realizada hoy",
                "El paquete llegó en perfecto estado y muy rápido",
                "Gran calidad de materiales y excelente precio final",
                "La mejor compra que he realizado en este sitio web",
                "Atención al cliente fabulosa y servicio muy eficiente",
                "Quedé gratamente sorprendido con la calidad y la atención",
                "Pésima atención al cliente, no lo recomiendo para nada",
                "Muy mala calidad del producto que llegó roto",
                "Llegó con mucho retraso y el paquete roto",
                "No sirve para nada este servicio tan deficiente",
                "Terrible experiencia vivida con este pésimo vendedor",
                "El producto dejó de funcionar exactamente al primer día",
                "Muy decepcionado con el servicio de entrega tardío",
                "Una completa estafa, producto falso y pésima calidad",
                "Demoraron demasiadas semanas y nunca responden mensajes",
                "Horrible servicio y peor atención nunca jamás compraré",
            ],
            "sentimiento": [
                "positivo",
                "positivo",
                "positivo",
                "positivo",
                "positivo",
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
                "negativo",
                "negativo",
                "negativo",
                "negativo",
                "negativo",
            ],
        }
    )


# ---------------------------------------------------------------------------
# Pruebas de contrato y disponibilidad del módulo NLP de José
# ---------------------------------------------------------------------------


def test_preprocessing_techniques_contract() -> None:
    """Verifica las constantes contractuales de técnicas NLP esperadas."""
    assert PREPROCESSING_TECHNIQUES == ("stemming", "lemmatization")


def test_get_nlp_preprocessors_missing_in_current_state() -> None:
    """Verifica que get_nlp_preprocessors retorna None mientras José no implemente src.nlp."""
    preprocessors = get_nlp_preprocessors()
    # Actualmente src/nlp/__init__.py está vacío
    assert preprocessors is None


def test_compare_preprocessing_techniques_missing_nlp_raises() -> None:
    """Comprueba que si José aún no implementó el NLP, se lanza NotImplementedError explicativo."""
    X = ["Texto uno", "Texto dos", "Texto tres", "Texto cuatro"]
    y = ["positivo", "negativo", "positivo", "negativo"]

    with pytest.raises(NotImplementedError) as exc_info:
        compare_preprocessing_techniques(X, y, preprocessors=None)

    assert "src.nlp" in str(exc_info.value)
    assert "stemming y lematización" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Pruebas del adaptador TextPreprocessorAdapter
# ---------------------------------------------------------------------------


def test_text_preprocessor_adapter_with_callable() -> None:
    """Verifica la adaptación de funciones normales de preprocesamiento elemento a elemento."""
    adapter = TextPreprocessorAdapter(func=mock_stemmer_func, name="stemmer")

    texts = ["servicio excelente", "totalmente satisfecho"]
    adapter.fit(texts)
    transformed = adapter.transform(texts)

    assert len(transformed) == 2
    assert isinstance(transformed[0], str)
    assert transformed == ["servicio excelente", "total satisfecho"]


def test_text_preprocessor_adapter_batch_mode() -> None:
    """Verifica que el adaptador admita modo batch cuando la función procesa la lista completa."""
    def uppercase_batch(batch: Sequence[str]) -> list[str]:
        return [t.upper() for t in batch]

    adapter = TextPreprocessorAdapter(func=uppercase_batch, batch=True)
    texts = ["hola mundo", "prueba de texto"]
    transformed = adapter.transform(texts)

    assert transformed == ["HOLA MUNDO", "PRUEBA DE TEXTO"]


def test_text_preprocessor_adapter_with_transformer_class() -> None:
    """Verifica que el adaptador delegue correctamente fit y transform en clases estimadoras."""
    mock_transformer = MockTrackingTransformer(prefix="test")
    adapter = TextPreprocessorAdapter(func=mock_transformer)

    texts = ["muestra 1", "muestra 2"]
    adapter.fit(texts)
    assert mock_transformer.fitted_samples_ == texts

    transformed = adapter.transform(texts)
    assert transformed == ["test_muestra 1", "test_muestra 2"]


def test_text_preprocessor_adapter_clone_and_pickle() -> None:
    """Verifica que el adaptador sea compatible con sklearn.base.clone y con serialización joblib."""
    adapter = TextPreprocessorAdapter(func=mock_lemmatizer_func, name="lemmatizer")

    cloned = clone(adapter)
    assert isinstance(cloned, TextPreprocessorAdapter)
    assert cloned.name == "lemmatizer"
    assert cloned.func == mock_lemmatizer_func

    # Serialización y deserialización con joblib
    buf = io.BytesIO()
    joblib.dump(adapter, buf)
    buf.seek(0)
    loaded = joblib.load(buf)

    assert isinstance(loaded, TextPreprocessorAdapter)
    assert loaded.transform(["PRUEBA"]) == ["prueba"]


# ---------------------------------------------------------------------------
# Construcción de Pipelines con preprocesador integrado
# ---------------------------------------------------------------------------


def test_pipeline_build_with_preprocessor_step() -> None:
    """Comprueba que build_*_pipeline incluya el paso 'preprocessor' en primer lugar."""
    lr_pipe = build_logistic_regression_pipeline(preprocessor=mock_stemmer_func)
    nb_pipe = build_multinomial_nb_pipeline(preprocessor=mock_lemmatizer_func)

    assert list(lr_pipe.named_steps.keys()) == ["preprocessor", "vectorizer", "classifier"]
    assert list(nb_pipe.named_steps.keys()) == ["preprocessor", "vectorizer", "classifier"]

    # Sin preprocesador conserva los 2 pasos originales (retrocompatibilidad)
    plain_lr = build_logistic_regression_pipeline(preprocessor=None)
    plain_nb = build_multinomial_nb_pipeline(preprocessor=None)
    assert list(plain_lr.named_steps.keys()) == ["vectorizer", "classifier"]
    assert list(plain_nb.named_steps.keys()) == ["vectorizer", "classifier"]


def test_build_preprocessing_comparison_pipelines_structure() -> None:
    """Verifica que se generen los 4 pipelines candidatos requeridos (2 modelos x 2 técnicas)."""
    preprocessors = {
        "stemming": mock_stemmer_func,
        "lemmatization": mock_lemmatizer_func,
    }
    pipelines = build_preprocessing_comparison_pipelines(preprocessors)

    expected_keys = {
        "logistic_regression__stemming",
        "logistic_regression__lemmatization",
        "multinomial_nb__stemming",
        "multinomial_nb__lemmatization",
    }
    assert set(pipelines.keys()) == expected_keys

    for pipe in pipelines.values():
        assert isinstance(pipe, Pipeline)
        assert "preprocessor" in pipe.named_steps
        assert "vectorizer" in pipe.named_steps
        assert "classifier" in pipe.named_steps


# ---------------------------------------------------------------------------
# Comparación reproducible de Stemming vs Lematización con CV estratificada
# ---------------------------------------------------------------------------


def test_compare_preprocessing_techniques_execution(synthetic_dataframe_20: pd.DataFrame) -> None:
    """Verifica la ejecución de la validación cruzada con preprocesadores inyectados."""
    preprocessors = {
        "stemming": mock_stemmer_func,
        "lemmatization": mock_lemmatizer_func,
    }

    result = compare_preprocessing_techniques(
        X_train=synthetic_dataframe_20["texto"],
        y_train=synthetic_dataframe_20["sentimiento"],
        preprocessors=preprocessors,
        cv=5,
    )

    assert isinstance(result, PreprocessingComparisonResult)
    assert result.cv_folds == 5
    assert len(result.results) == 4
    assert len(result.ranking) == 4

    # Ranking debe estar ordenado de mayor a menor por f1_macro_mean
    f1_scores = [r.metrics["f1_macro_mean"] for r in result.ranking]
    assert f1_scores == sorted(f1_scores, reverse=True)

    for res in result.ranking:
        assert 0.0 <= res.metrics["f1_macro_mean"] <= 1.0
        assert 0.0 <= res.metrics["accuracy_mean"] <= 1.0
        assert len(res.fold_metrics["f1_macro"]) == 5
        assert len(res.fold_metrics["accuracy"]) == 5


def test_compare_preprocessing_techniques_isolates_test_set(
    synthetic_dataframe_20: pd.DataFrame,
) -> None:
    """Garantiza que X_test e y_test no sean accedidos ni evaluados al pasar DatasetSplit."""
    split = split_dataset(synthetic_dataframe_20, test_size=0.20, random_seed=42)

    preprocessors = {
        "stemming": mock_stemmer_func,
        "lemmatization": mock_lemmatizer_func,
    }

    result = compare_preprocessing_techniques(
        X_train=split,
        preprocessors=preprocessors,
        cv=5,
    )

    assert isinstance(result, PreprocessingComparisonResult)
    # 20 muestras en total, test_size=0.2 -> 16 train, 4 test
    assert len(split.X_train) == 16
    assert len(split.X_test) == 4


def test_preprocessing_cv_no_data_leakage(synthetic_dataframe_20: pd.DataFrame) -> None:
    """Verifica que el preprocesador se ajuste exclusivamente dentro de cada pliegue de entrenamiento."""
    MockTrackingTransformer.fitted_batches.clear()
    tracker = MockTrackingTransformer()
    preprocessors = {"custom_tracker": tracker}

    # Ejecutar comparación sobre 20 muestras con 5 folds (16 entrenamiento y 4 validación por fold)
    compare_preprocessing_techniques(
        X_train=synthetic_dataframe_20["texto"],
        y_train=synthetic_dataframe_20["sentimiento"],
        preprocessors=preprocessors,
        cv=5,
    )

    # Cada fold evalúa 2 modelos (LR y MNB) x 5 folds = 10 llamadas a fit sobre clones independientes
    assert len(MockTrackingTransformer.fitted_batches) == 10
    # En cada una de las 10 llamadas a fit, se pasaron exactamente 16 muestras (cero fuga de los 4 de test/val)
    for batch in MockTrackingTransformer.fitted_batches:
        assert len(batch) == 16


def test_preprocessing_comparison_to_dataframe_and_summary(
    synthetic_dataframe_20: pd.DataFrame,
) -> None:
    """Verifica la conversión a DataFrame y el formateo textual neutral."""
    preprocessors = {
        "stemming": mock_stemmer_func,
        "lemmatization": mock_lemmatizer_func,
    }

    result = compare_preprocessing_techniques(
        X_train=synthetic_dataframe_20["texto"],
        y_train=synthetic_dataframe_20["sentimiento"],
        preprocessors=preprocessors,
        cv=5,
    )

    df = result.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    expected_cols = [
        "rank",
        "combination",
        "model",
        "technique",
        "ngram_range",
        "f1_macro_mean",
        "f1_macro_std",
        "accuracy_mean",
        "accuracy_std",
    ]
    assert list(df.columns) == expected_cols

    summary_text = result.summary()
    assert "COMPARACIÓN DE PREPROCESAMIENTO NLP: STEMMING VS LEMATIZACIÓN" in summary_text
    assert "Regresión Logística" in summary_text
    assert "Multinomial Naive Bayes" in summary_text
    assert "Stemming" in summary_text
    assert "Lematización" in summary_text
    assert "Nota metodológica" in summary_text


def test_full_pipeline_with_preprocessor_serialization(
    synthetic_dataframe_20: pd.DataFrame,
) -> None:
    """Verifica que un pipeline completo con preprocesador sea serializable y predecible tras reload."""
    pipe = build_logistic_regression_pipeline(preprocessor=mock_lemmatizer_func)

    X_train = synthetic_dataframe_20["texto"]
    y_train = synthetic_dataframe_20["sentimiento"]
    pipe.fit(X_train, y_train)

    test_samples = ["Excelente servicio", "Pésimo y horrible producto"]
    preds_before = pipe.predict(test_samples)
    probas_before = pipe.predict_proba(test_samples)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = tmp.name

    joblib.dump(pipe, tmp_path)
    loaded_pipe = joblib.load(tmp_path)

    preds_after = loaded_pipe.predict(test_samples)
    probas_after = loaded_pipe.predict_proba(test_samples)

    np.testing.assert_array_equal(preds_before, preds_after)
    np.testing.assert_allclose(probas_before, probas_after)
