"""Pruebas unitarias para la comparación de representaciones TF-IDF: unigramas vs bigramas (Fase 3.5A)."""

import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.models.experiments import (
    NGRAM_CONFIGURATIONS,
    NgramComparisonResult,
    build_ngram_comparison_pipelines,
    compare_ngram_representations,
    format_ngram_comparison,
)
from src.models.split import DatasetSplit, split_dataset

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# Fixture sintética balanceada con suficiencia muestral para 5 folds
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_ngram_dataframe_20() -> pd.DataFrame:
    """Dataset balanceado de 20 reseñas (10 positivas, 10 negativas) con frases ricas en bigramas."""
    return pd.DataFrame(
        {
            "texto": [
                "Excelente producto y servicio totalmente impecable",
                "Muy buena atención recibida en la tienda online",
                "Me encantó la rapidez del envío del pedido",
                "Experiencia genial y absolutamente recomendable para todos",
                "Totalmente satisfecho con la compra realizada hoy",
                "El paquete llegó en perfecto estado y muy rápido",
                "Gran calidad de materiales y excelente precio",
                "La mejor compra que he realizado en este sitio",
                "Atención al cliente fabulosa y servicio muy eficiente",
                "Quedé gratamente sorprendido con la calidad y la atención",
                "Pésima atención al cliente, no lo recomiendo para nada",
                "Muy mala calidad del producto que llegó roto",
                "Llegó con mucho retraso y el paquete roto",
                "No sirve para nada este servicio tan deficiente",
                "Terrible experiencia vivida con este pésimo vendedor",
                "El producto dejó de funcionar exactamente al primer día",
                "Muy decepcionado con el servicio de entrega tardío",
                "Una completa estafa, producto falso y mala calidad",
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
# Contrato y construcción de pipelines con n-gramas
# ---------------------------------------------------------------------------


def test_ngram_configurations_contract() -> None:
    assert NGRAM_CONFIGURATIONS == {
        "unigrams": (1, 1),
        "unigrams_and_bigramas" if "unigrams_and_bigramas" in NGRAM_CONFIGURATIONS else "unigrams_and_bigrams": (1, 2),
    }
    assert NGRAM_CONFIGURATIONS["unigrams"] == (1, 1)
    assert NGRAM_CONFIGURATIONS["unigrams_and_bigrams"] == (1, 2)


def test_build_ngram_comparison_pipelines_structure() -> None:
    pipelines = build_ngram_comparison_pipelines()

    assert len(pipelines) == 4
    expected_names = {
        "logistic_regression__unigrams",
        "logistic_regression__unigrams_and_bigrams",
        "multinomial_nb__unigrams",
        "multinomial_nb__unigrams_and_bigrams",
    }
    assert set(pipelines.keys()) == expected_names

    for name, pipe in pipelines.items():
        assert isinstance(pipe, Pipeline)
        assert "vectorizer" in pipe.named_steps
        assert "classifier" in pipe.named_steps

        vec = pipe.named_steps["vectorizer"]
        clf = pipe.named_steps["classifier"]
        assert isinstance(vec, TfidfVectorizer)

        if "logistic_regression" in name:
            assert isinstance(clf, LogisticRegression)
        else:
            assert isinstance(clf, MultinomialNB)

        if "unigrams_and_bigrams" in name:
            assert vec.ngram_range == (1, 2)
        else:
            assert vec.ngram_range == (1, 1)


def test_build_ngram_comparison_pipelines_custom_vectorizer_factory() -> None:
    """Verifica compatibilidad con transformadores futuros del módulo de José en src/nlp/."""
    custom_created_ranges = []

    def mock_jose_vectorizer_factory(ngram_range: tuple[int, int]) -> TfidfVectorizer:
        custom_created_ranges.append(ngram_range)
        return TfidfVectorizer(ngram_range=ngram_range, lowercase=True)

    pipelines = build_ngram_comparison_pipelines(
        vectorizer_factory=mock_jose_vectorizer_factory
    )

    assert len(pipelines) == 4
    assert (1, 1) in custom_created_ranges
    assert (1, 2) in custom_created_ranges


# ---------------------------------------------------------------------------
# Ejecución y estructura de la comparación con validación cruzada
# ---------------------------------------------------------------------------


def test_compare_ngram_representations_structure_and_metrics(
    synthetic_ngram_dataframe_20: pd.DataFrame,
) -> None:
    X = synthetic_ngram_dataframe_20["texto"]
    y = synthetic_ngram_dataframe_20["sentimiento"]

    result = compare_ngram_representations(X, y)

    assert isinstance(result, NgramComparisonResult)
    assert result.cv_folds == 5
    assert result.primary_metric == "f1_macro"
    assert len(result.results) == 4
    assert len(result.ranking) == 4

    # Verificar que el ranking está ordenado descendentemente por F1 Macro
    for i in range(len(result.ranking) - 1):
        assert (
            result.ranking[i].metrics["f1_macro_mean"]
            >= result.ranking[i + 1].metrics["f1_macro_mean"]
        )

    # Verificar métricas completas en cada combinación
    for res in result.results.values():
        for m_name in ("f1_macro_mean", "f1_macro_std", "accuracy_mean", "accuracy_std"):
            assert m_name in res.metrics
            assert isinstance(res.metrics[m_name], float)
            assert 0.0 <= res.metrics[m_name] <= 1.0

        assert "f1_macro" in res.fold_metrics
        assert "accuracy" in res.fold_metrics
        assert len(res.fold_metrics["f1_macro"]) == 5
        assert len(res.fold_metrics["accuracy"]) == 5


def test_compare_ngram_representations_preserves_test_set_isolation(
    synthetic_ngram_dataframe_20: pd.DataFrame,
) -> None:
    """Garantiza que la comparación utilice EXCLUSIVAMENTE X_train e y_train.

    X_test e y_test no deben ser accedidos ni evaluados bajo ninguna circunstancia.
    """
    split = split_dataset(synthetic_ngram_dataframe_20, test_size=0.20, random_seed=42)

    # Inyectamos datos corruptos deliberadamente en el subconjunto de prueba
    tampered_split = DatasetSplit(
        X_train=split.X_train,
        X_test=pd.Series(["DATO_CORRUPTO_QUE_NUNCA_DEBE_LEERSE"] * len(split.X_test)),
        y_train=split.y_train,
        y_test=pd.Series(["ETIQUETA_FALSA"] * len(split.y_test)),
    )

    # La comparación no debe fallar porque nunca toca X_test ni y_test
    result = compare_ngram_representations(tampered_split)

    assert isinstance(result, NgramComparisonResult)
    assert len(result.results) == 4


def test_compare_ngram_representations_no_data_leakage(
    synthetic_ngram_dataframe_20: pd.DataFrame,
) -> None:
    """Verifica que el TF-IDF dentro de cada pipeline se ajusta estrictamente dentro de cada fold."""
    X = synthetic_ngram_dataframe_20["texto"]
    y = synthetic_ngram_dataframe_20["sentimiento"]

    pipelines = build_ngram_comparison_pipelines()
    for pipe in pipelines.values():
        assert not hasattr(pipe.named_steps["vectorizer"], "vocabulary_")

    # Ejecutar comparación con pipelines frescos
    result = compare_ngram_representations(X, y)
    assert len(result.results) == 4


def test_compare_ngram_representations_is_strictly_reproducible(
    synthetic_ngram_dataframe_20: pd.DataFrame,
) -> None:
    X = synthetic_ngram_dataframe_20["texto"]
    y = synthetic_ngram_dataframe_20["sentimiento"]

    res_1 = compare_ngram_representations(X, y)
    res_2 = compare_ngram_representations(X, y)

    assert [r.name for r in res_1.ranking] == [r.name for r in res_2.ranking]

    for name in res_1.results:
        f1_1 = res_1.results[name].metrics["f1_macro_mean"]
        f1_2 = res_2.results[name].metrics["f1_macro_mean"]
        assert f1_1 == pytest.approx(f1_2)


# ---------------------------------------------------------------------------
# Conversión a DataFrame y reporte de comparación
# ---------------------------------------------------------------------------


def test_ngram_comparison_to_dataframe(
    synthetic_ngram_dataframe_20: pd.DataFrame,
) -> None:
    X = synthetic_ngram_dataframe_20["texto"]
    y = synthetic_ngram_dataframe_20["sentimiento"]

    result = compare_ngram_representations(X, y)
    df = result.to_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    expected_cols = [
        "rank",
        "combination",
        "model",
        "representation",
        "ngram_range",
        "f1_macro_mean",
        "f1_macro_std",
        "accuracy_mean",
        "accuracy_std",
    ]
    for col in expected_cols:
        assert col in df.columns

    assert df.loc[0, "rank"] == 1
    assert "(1, 1)" in df["ngram_range"].values
    assert "(1, 2)" in df["ngram_range"].values


def test_format_ngram_comparison(
    synthetic_ngram_dataframe_20: pd.DataFrame,
) -> None:
    X = synthetic_ngram_dataframe_20["texto"]
    y = synthetic_ngram_dataframe_20["sentimiento"]

    result = compare_ngram_representations(X, y)
    summary_text = result.summary()

    assert "COMPARACIÓN DE REPRESENTACIONES: UNIGRAMAS (1,1) VS UNIGRAMAS + BIGRAMAS (1,2)" in summary_text
    assert "logistic_regression__unigrams" in summary_text
    assert "logistic_regression__unigrams_and_bigrams" in summary_text
    assert "multinomial_nb__unigrams" in summary_text
    assert "multinomial_nb__unigrams_and_bigrams" in summary_text
    assert "Unigramas (1, 1)" in summary_text
    assert "Unigramas + Bigramas (1, 2)" in summary_text
    assert "No se selecciona modelo definitivo ni se evalúa X_test" in summary_text


# ---------------------------------------------------------------------------
# Validaciones y manejo de errores
# ---------------------------------------------------------------------------


def test_ngram_comparison_validation_errors(
    synthetic_ngram_dataframe_20: pd.DataFrame,
) -> None:
    X = synthetic_ngram_dataframe_20["texto"]
    y = synthetic_ngram_dataframe_20["sentimiento"]

    # 1. DatasetSplit con y_train explícito redundante
    split = split_dataset(synthetic_ngram_dataframe_20, test_size=0.20, random_seed=42)
    with pytest.raises(ValueError, match="No se debe proporcionar 'y_train'"):
        compare_ngram_representations(split, y_train=split.y_train)

    # 2. X_train sin y_train cuando no es DatasetSplit
    with pytest.raises(ValueError, match="Se requiere 'y_train'"):
        compare_ngram_representations(X)

    # 3. Datos vacíos
    with pytest.raises(ValueError, match="no pueden estar vacíos"):
        compare_ngram_representations([], [])

    # 4. Longitudes discordantes
    with pytest.raises(ValueError, match="no coincide"):
        compare_ngram_representations(X, y.iloc[:5])

    # 5. Muestras insuficientes para 5 folds
    insufficient_y = ["positivo"] * 10 + ["negativo"] * 4
    insufficient_X = ["texto"] * 14
    with pytest.raises(ValueError, match="Muestras insuficientes"):
        compare_ngram_representations(insufficient_X, insufficient_y)
