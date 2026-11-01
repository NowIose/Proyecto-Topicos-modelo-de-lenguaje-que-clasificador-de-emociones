"""Pruebas unitarias para la optimización de hiperparámetros con GridSearchCV (Fase 3.4B)."""

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.experiments import (
    DEFAULT_PARAM_GRIDS,
    GridSearchResult,
    build_logistic_regression_pipeline,
    build_multinomial_nb_pipeline,
    cv_results_to_dataframe,
    format_grid_search_comparison,
    get_default_param_grid,
    run_grid_search,
    tune_pipeline_grid_search,
)
from src.models.split import DatasetSplit, split_dataset

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# Fixture con suficientes muestras para validación cruzada estratificada (5 folds)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_grid_dataframe_20() -> pd.DataFrame:
    """Dataset balanceado de 20 muestras (10 positivas, 10 negativas) para pruebas."""
    return pd.DataFrame(
        {
            "texto": [
                "Excelente producto y servicio impecable",
                "Muy buena atención recibida en la tienda",
                "Me encantó la rapidez del envío del pedido",
                "Experiencia genial y totalmente recomendable",
                "Totalmente satisfecho con la compra realizada",
                "El paquete llegó en perfecto estado y rápido",
                "Gran calidad de materiales y buen precio",
                "La mejor compra que he realizado este año",
                "Atención al cliente fabulosa y muy eficiente",
                "Quedé encantado con el servicio y la atención",
                "Pésima atención, no lo recomiendo en absoluto",
                "Muy mala calidad del producto recibido",
                "Llegó tarde y roto el paquete comprado",
                "No sirve para nada este servicio tan deficiente",
                "Terrible experiencia vivida con este vendedor",
                "El producto dejó de funcionar al primer día",
                "Muy decepcionado con el servicio posventa",
                "Una completa estafa, producto falso y roto",
                "Demoraron semanas y no responden los mensajes",
                "Horrible servicio y peor atención recibida",
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
# Configuración y contrato de los grids de parámetros
# ---------------------------------------------------------------------------


def test_default_param_grids_contract() -> None:
    assert "logistic_regression" in DEFAULT_PARAM_GRIDS
    assert "multinomial_nb" in DEFAULT_PARAM_GRIDS

    lr_grid = DEFAULT_PARAM_GRIDS["logistic_regression"]
    nb_grid = DEFAULT_PARAM_GRIDS["multinomial_nb"]

    assert lr_grid == {"classifier__C": [0.1, 1.0, 10.0]}
    assert nb_grid == {"classifier__alpha": [0.1, 1.0, 10.0]}


def test_get_default_param_grid() -> None:
    grid = get_default_param_grid("logistic_regression")
    assert grid == {"classifier__C": [0.1, 1.0, 10.0]}

    # Verificar que devuelve una copia modificable sin alterar la constante
    grid["classifier__C"].append(100.0)
    assert 100.0 not in DEFAULT_PARAM_GRIDS["logistic_regression"]["classifier__C"]

    with pytest.raises(ValueError, match="desconocido"):
        get_default_param_grid("modelo_no_existente")


# ---------------------------------------------------------------------------
# Optimización individual por pipeline con GridSearchCV
# ---------------------------------------------------------------------------


def test_tune_pipeline_grid_search_logistic_regression(
    synthetic_grid_dataframe_20: pd.DataFrame,
) -> None:
    X = synthetic_grid_dataframe_20["texto"]
    y = synthetic_grid_dataframe_20["sentimiento"]

    pipeline = build_logistic_regression_pipeline()
    param_grid = DEFAULT_PARAM_GRIDS["logistic_regression"]

    result = tune_pipeline_grid_search(
        pipeline=pipeline,
        param_grid=param_grid,
        X_train=X,
        y_train=y,
        name="logistic_regression",
    )

    assert isinstance(result, GridSearchResult)
    assert result.model_name == "logistic_regression"
    assert "classifier__C" in result.best_params
    assert result.best_params["classifier__C"] in [0.1, 1.0, 10.0]
    assert isinstance(result.best_score, float)
    assert 0.0 <= result.best_score <= 1.0

    # Estimador devuelto debe estar reajustado sobre todo X_train con el mejor C
    best_estimator = result.best_estimator
    assert isinstance(best_estimator, Pipeline)
    assert hasattr(best_estimator.named_steps["classifier"], "classes_")
    assert best_estimator.named_steps["classifier"].C == result.best_params["classifier__C"]

    # cv_results debe registrar las 3 combinaciones probadas
    assert len(result.cv_results["params"]) == 3


def test_tune_pipeline_grid_search_multinomial_nb(
    synthetic_grid_dataframe_20: pd.DataFrame,
) -> None:
    X = synthetic_grid_dataframe_20["texto"]
    y = synthetic_grid_dataframe_20["sentimiento"]

    pipeline = build_multinomial_nb_pipeline()
    param_grid = DEFAULT_PARAM_GRIDS["multinomial_nb"]

    result = tune_pipeline_grid_search(
        pipeline=pipeline,
        param_grid=param_grid,
        X_train=X,
        y_train=y,
        name="multinomial_nb",
    )

    assert isinstance(result, GridSearchResult)
    assert result.model_name == "multinomial_nb"
    assert "classifier__alpha" in result.best_params
    assert result.best_params["classifier__alpha"] in [0.1, 1.0, 10.0]
    assert isinstance(result.best_score, float)
    assert 0.0 <= result.best_score <= 1.0

    best_estimator = result.best_estimator
    assert best_estimator.named_steps["classifier"].alpha == result.best_params["classifier__alpha"]


# ---------------------------------------------------------------------------
# Optimización conjunta de ambos modelos candidatos
# ---------------------------------------------------------------------------


def test_run_grid_search_both_models(
    synthetic_grid_dataframe_20: pd.DataFrame,
) -> None:
    X = synthetic_grid_dataframe_20["texto"]
    y = synthetic_grid_dataframe_20["sentimiento"]

    results = run_grid_search(X, y)

    assert isinstance(results, dict)
    assert set(results.keys()) == {"logistic_regression", "multinomial_nb"}

    lr_res = results["logistic_regression"]
    nb_res = results["multinomial_nb"]

    assert lr_res.best_params["classifier__C"] in [0.1, 1.0, 10.0]
    assert nb_res.best_params["classifier__alpha"] in [0.1, 1.0, 10.0]


def test_run_grid_search_with_dataset_split_preserves_test_set(
    synthetic_grid_dataframe_20: pd.DataFrame,
) -> None:
    """Verifica que GridSearchCV opera exclusivamente sobre X_train e y_train.

    X_test e y_test no deben ser accedidos bajo ninguna circunstancia.
    """
    split = split_dataset(synthetic_grid_dataframe_20, test_size=0.20, random_seed=42)

    # Inyectamos valores corruptos en test split para garantizar que nunca se lean
    tampered_split = DatasetSplit(
        X_train=split.X_train,
        X_test=pd.Series(["DATO_CORRUPTO_TEST"] * len(split.X_test)),
        y_train=split.y_train,
        y_test=pd.Series(["ETIQUETA_ERRONEA"] * len(split.y_test)),
    )

    results = run_grid_search(tampered_split)

    assert len(results) == 2
    assert "logistic_regression" in results
    assert "multinomial_nb" in results


def test_grid_search_no_data_leakage(
    synthetic_grid_dataframe_20: pd.DataFrame,
) -> None:
    """Garantiza que el Pipeline original no es mutado de forma externa previa al fit."""
    pipeline = build_logistic_regression_pipeline()
    assert not hasattr(pipeline.named_steps["vectorizer"], "vocabulary_")

    X = synthetic_grid_dataframe_20["texto"]
    y = synthetic_grid_dataframe_20["sentimiento"]

    result = tune_pipeline_grid_search(
        pipeline=pipeline,
        param_grid={"classifier__C": [1.0]},
        X_train=X,
        y_train=y,
    )

    # El best_estimator_ tiene vocabulario ajustado solo sobre X_train
    assert hasattr(result.best_estimator.named_steps["vectorizer"], "vocabulary_")


def test_grid_search_is_strictly_reproducible(
    synthetic_grid_dataframe_20: pd.DataFrame,
) -> None:
    X = synthetic_grid_dataframe_20["texto"]
    y = synthetic_grid_dataframe_20["sentimiento"]

    res_1 = run_grid_search(X, y)
    res_2 = run_grid_search(X, y)

    # Mismos mejores hiperparámetros
    assert res_1["logistic_regression"].best_params == res_2["logistic_regression"].best_params
    assert res_1["multinomial_nb"].best_params == res_2["multinomial_nb"].best_params

    # Mismos puntajes exactos
    assert res_1["logistic_regression"].best_score == pytest.approx(
        res_2["logistic_regression"].best_score
    )
    assert res_1["multinomial_nb"].best_score == pytest.approx(
        res_2["multinomial_nb"].best_score
    )


def test_grid_search_custom_param_grid(
    synthetic_grid_dataframe_20: pd.DataFrame,
) -> None:
    X = synthetic_grid_dataframe_20["texto"]
    y = synthetic_grid_dataframe_20["sentimiento"]

    custom_grid = {
        "logistic_regression": {"classifier__C": [0.5, 2.5]},
        "multinomial_nb": {"classifier__alpha": [0.05, 0.5]},
    }

    results = run_grid_search(X, y, param_grids=custom_grid)

    assert results["logistic_regression"].best_params["classifier__C"] in [0.5, 2.5]
    assert len(results["logistic_regression"].cv_results["params"]) == 2
    assert results["multinomial_nb"].best_params["classifier__alpha"] in [0.05, 0.5]
    assert len(results["multinomial_nb"].cv_results["params"]) == 2


# ---------------------------------------------------------------------------
# Conversión a DataFrame y formateo de comparación
# ---------------------------------------------------------------------------


def test_grid_search_result_to_dataframe(
    synthetic_grid_dataframe_20: pd.DataFrame,
) -> None:
    X = synthetic_grid_dataframe_20["texto"]
    y = synthetic_grid_dataframe_20["sentimiento"]

    results = run_grid_search(X, y)
    lr_res = results["logistic_regression"]

    df = lr_res.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "params" in df.columns
    assert "f1_macro_mean" in df.columns
    assert "f1_macro_std" in df.columns
    assert "rank" in df.columns
    assert df.loc[0, "rank"] == 1


def test_format_grid_search_comparison(
    synthetic_grid_dataframe_20: pd.DataFrame,
) -> None:
    X = synthetic_grid_dataframe_20["texto"]
    y = synthetic_grid_dataframe_20["sentimiento"]

    results = run_grid_search(X, y)
    text = format_grid_search_comparison(results)

    assert "OPTIMIZACIÓN DE HIPERPARÁMETROS CON GRIDSEARCHCV" in text
    assert "logistic_regression" in text
    assert "multinomial_nb" in text
    assert "Mejor combinación encontrada:" in text
    assert "Mejor f1_macro en CV:" in text
    assert "Configuraciones exploradas:" in text
    assert "No se selecciona modelo definitivo en esta subfase" in text


# ---------------------------------------------------------------------------
# Manejo de errores
# ---------------------------------------------------------------------------


def test_grid_search_validation_errors(
    synthetic_grid_dataframe_20: pd.DataFrame,
) -> None:
    X = synthetic_grid_dataframe_20["texto"]
    y = synthetic_grid_dataframe_20["sentimiento"]

    # 1. Objeto no Pipeline
    with pytest.raises(TypeError, match="sklearn.pipeline.Pipeline"):
        tune_pipeline_grid_search(
            "no_es_pipeline",  # type: ignore
            {"classifier__C": [1.0]},
            X,
            y,
        )

    # 2. Datos vacíos
    with pytest.raises(ValueError, match="no pueden estar vacíos"):
        tune_pipeline_grid_search(
            build_logistic_regression_pipeline(),
            {"classifier__C": [1.0]},
            [],
            [],
        )

    # 3. Muestras discordantes
    with pytest.raises(ValueError, match="no coincide"):
        tune_pipeline_grid_search(
            build_logistic_regression_pipeline(),
            {"classifier__C": [1.0]},
            X,
            y.iloc[:5],
        )

    # 4. DatasetSplit con y_train explícito redundante
    split = split_dataset(synthetic_grid_dataframe_20, test_size=0.20, random_seed=42)
    with pytest.raises(ValueError, match="No se debe proporcionar 'y_train'"):
        run_grid_search(split, y_train=split.y_train)

    # 5. Muestras insuficientes para 5 folds
    insufficient_y = ["positivo"] * 10 + ["negativo"] * 4
    insufficient_X = ["texto"] * 14
    with pytest.raises(ValueError, match="Muestras insuficientes"):
        run_grid_search(insufficient_X, insufficient_y)
