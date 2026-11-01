"""Entrenamiento reproducible de punta a punta con un solo comando.

    python -m src.models.entrenar            # entrenamiento completo
    python -m src.models.entrenar --rapido   # muestra reducida, para pruebas en vivo

Flujo (todo sobre el conjunto de entrenamiento; la prueba solo se usa al final):
1. Carga la partición estratificada 80/20 del Módulo 1.
2. Compara stemming, lematización y sin reducción (validación cruzada).
3. Preprocesa los textos con el método de config/ml.yaml (nlp.metodo_morfologico).
4. Compara línea base, Regresión Logística y Naive Bayes con unigramas
   y con unigramas + bigramas (validación cruzada estratificada).
5. Ajusta hiperparámetros de ambos modelos con GridSearchCV.
6. Elige el modelo de producción por F1 macro de validación cruzada.
7. Evalúa una sola vez en prueba: métricas, matriz de confusión, errores.
8. Guarda el pipeline completo (preprocesador NLP + TF-IDF + clasificador)
   y las métricas en artifacts/.

Las funciones de validación cruzada, grid search y métricas son las del
Módulo 2 (experiments.py y evaluate.py); este script las orquesta.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.nlp import STOPWORDS_ESP, PreprocesadorNLP

from .config import MLConfig, NLPConfig, load_config
from .evaluate import evaluate_predictions
from .experiments import cross_validate_pipeline, predict_probabilities, tune_pipeline_grid_search

# Los CSV del Módulo 1 llaman 'text' a la columna; el contrato del Módulo 2 usa 'texto'.
COLUMNA_TEXTO_CSV = "text"

# Muestra usada para comparar stemming contra lematización: spaCy procesa
# ~270 reseñas/s, así que lematizar las 128 000 tardaría ~8 minutos.
MUESTRA_COMPARACION_MORFOLOGICA = 20_000
MUESTRA_RAPIDA = 8_000

NOMBRES_MODELOS = {
    "baseline": "Línea base (clase más frecuente)",
    "logistic_regression": "Regresión Logística",
    "multinomial_nb": "Naive Bayes Multinomial",
}

# Hiperparámetros explorados con GridSearchCV.
GRILLAS = {
    "logistic_regression": {"classifier__C": [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]},
    "multinomial_nb": {"classifier__alpha": [0.05, 0.1, 0.3, 0.5, 1.0]},
}


def log(mensaje: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {mensaje}", flush=True)


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------


def cargar_particion(ruta: Path, config: MLConfig) -> pd.DataFrame:
    """Lee un CSV de la partición y lo deja con las columnas del contrato."""
    df = pd.read_csv(ruta, encoding="utf-8")
    df = df.rename(columns={COLUMNA_TEXTO_CSV: config.dataset.text_column})
    df = df[[config.dataset.text_column, config.dataset.target_column]].dropna()
    return df.reset_index(drop=True)


def muestra_estratificada(df: pd.DataFrame, n: int, config: MLConfig) -> pd.DataFrame:
    if len(df) <= n:
        return df
    muestra, _ = train_test_split(
        df,
        train_size=n,
        stratify=df[config.dataset.target_column],
        random_state=config.experiment.random_seed,
    )
    return muestra.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Construcción de pipelines
# ---------------------------------------------------------------------------


def construir_vectorizador(nlp: NLPConfig, ngram_range: tuple[int, int] | None = None) -> TfidfVectorizer:
    """TF-IDF con los parámetros documentados en config/ml.yaml.

    lowercase=False porque el preprocesador NLP ya normalizó el texto.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range or nlp.ngram_range,
        max_features=nlp.max_features,
        min_df=nlp.min_df,
        max_df=nlp.max_df,
        sublinear_tf=nlp.sublinear_tf,
        lowercase=False,
    )


def construir_clasificador(nombre: str, seed: int) -> Any:
    if nombre == "baseline":
        return DummyClassifier(strategy="most_frequent")
    if nombre == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if nombre == "multinomial_nb":
        return MultinomialNB()
    raise ValueError(f"Modelo desconocido: {nombre}")


def pipeline_sobre_texto_procesado(
    nombre: str, config: MLConfig, ngram_range: tuple[int, int] | None = None
) -> Pipeline:
    """Pipeline TF-IDF + clasificador para textos YA preprocesados.

    El preprocesamiento NLP no aprende nada (cada texto se trata por separado),
    así que hacerlo una sola vez antes de la validación cruzada no filtra
    información; el TF-IDF sí se ajusta dentro de cada fold.
    """
    return Pipeline(
        [
            ("vectorizer", construir_vectorizador(config.nlp, ngram_range)),
            ("classifier", construir_clasificador(nombre, config.experiment.random_seed)),
        ]
    )


def ensamblar_pipeline_produccion(ajustado: Pipeline, metodo: str) -> Pipeline:
    """Antepone el preprocesador NLP a un pipeline ya entrenado.

    El resultado recibe texto crudo, que es lo que manda la API.
    """
    preprocesador = PreprocesadorNLP(metodo).fit([])
    return Pipeline(
        [
            ("preprocessor", preprocesador),
            ("vectorizer", ajustado.named_steps["vectorizer"]),
            ("classifier", ajustado.named_steps["classifier"]),
        ]
    )


# ---------------------------------------------------------------------------
# Experimentos
# ---------------------------------------------------------------------------


def resumen_cv(resultado: Any) -> dict[str, float]:
    m = resultado.metrics
    return {
        "f1_macro": round(float(m["f1_macro_mean"]), 4),
        "f1_macro_std": round(float(m["f1_macro_std"]), 4),
        "accuracy": round(float(m["accuracy_mean"]), 4),
    }


def comparar_morfologia(train: pd.DataFrame, config: MLConfig, n: int) -> list[dict[str, Any]]:
    """Stemming vs lematización vs sin reducción, con Regresión Logística."""
    muestra = muestra_estratificada(train, n, config)
    textos = muestra[config.dataset.text_column].tolist()
    y = muestra[config.dataset.target_column]
    cv = StratifiedKFold(config.experiment.cv_folds, shuffle=True, random_state=config.experiment.random_seed)

    filas = []
    for metodo in ("stemming", "lematizacion", "ninguno"):
        inicio = time.time()
        procesados = PreprocesadorNLP(metodo).transform(textos)
        segundos_proceso = time.time() - inicio
        resultado = cross_validate_pipeline(
            pipeline_sobre_texto_procesado("logistic_regression", config),
            procesados,
            y,
            cv=cv,
            config=config,
            name=metodo,
        )
        vocabulario = len(set(" ".join(procesados).split()))
        fila = {
            "metodo": metodo,
            "muestra": len(muestra),
            **resumen_cv(resultado),
            "vocabulario_distinto": vocabulario,
            "segundos_preprocesamiento": round(segundos_proceso, 1),
        }
        filas.append(fila)
        log(f"  {metodo:<13} F1={fila['f1_macro']:.4f}  vocabulario={vocabulario}  ({segundos_proceso:.0f}s)")
    return filas


def comparar_modelos_y_ngramas(
    textos: Sequence[str], y: pd.Series, config: MLConfig
) -> list[dict[str, Any]]:
    """Los cinco experimentos del plan del Módulo 2 (más NB/baseline con bigramas)."""
    cv = StratifiedKFold(config.experiment.cv_folds, shuffle=True, random_state=config.experiment.random_seed)
    filas = []
    for ngram in ((1, 1), (1, 2)):
        for nombre in ("baseline", "logistic_regression", "multinomial_nb"):
            resultado = cross_validate_pipeline(
                pipeline_sobre_texto_procesado(nombre, config, ngram),
                textos,
                y,
                cv=cv,
                config=config,
                name=f"{nombre}_{ngram}",
            )
            fila = {
                "modelo": nombre,
                "ngram_range": list(ngram),
                "representacion": "unigramas" if ngram == (1, 1) else "unigramas + bigramas",
                **resumen_cv(resultado),
            }
            filas.append(fila)
            log(f"  {NOMBRES_MODELOS[nombre]:<34} {fila['representacion']:<21} F1={fila['f1_macro']:.4f}")
    return filas


def ajustar_hiperparametros(
    textos: Sequence[str], y: pd.Series, config: MLConfig
) -> dict[str, Any]:
    cv = StratifiedKFold(config.experiment.cv_folds, shuffle=True, random_state=config.experiment.random_seed)
    resultados = {}
    for nombre, grilla in GRILLAS.items():
        busqueda = tune_pipeline_grid_search(
            pipeline_sobre_texto_procesado(nombre, config),
            grilla,
            textos,
            y,
            cv=cv,
            scoring=config.experiment.primary_metric,
            config=config,
            name=nombre,
            n_jobs=-1,
        )
        cv_df = pd.DataFrame(busqueda.cv_results)
        param = next(iter(grilla))
        resultados[nombre] = {
            "estimador": busqueda.best_estimator,
            "parametro": param.replace("classifier__", ""),
            "mejor_valor": busqueda.best_params[param],
            "mejor_f1_cv": round(float(busqueda.best_score), 4),
            "grilla": [
                {"valor": v, "f1_macro": round(float(s), 4)}
                for v, s in zip(cv_df[f"param_{param}"], cv_df["mean_test_score"])
            ],
        }
        log(
            f"  {NOMBRES_MODELOS[nombre]:<24} mejor {resultados[nombre]['parametro']}="
            f"{resultados[nombre]['mejor_valor']}  F1 CV={resultados[nombre]['mejor_f1_cv']:.4f}"
        )
    return resultados


# ---------------------------------------------------------------------------
# Evaluación final en prueba
# ---------------------------------------------------------------------------


def metricas_prueba(
    pipeline: Pipeline, textos_proc: Sequence[str], textos_orig: Sequence[str], y: pd.Series, labels: Sequence[str]
) -> dict[str, Any]:
    y_pred = pipeline.predict(list(textos_proc))
    probabilidades = (
        predict_probabilities(pipeline, list(textos_proc)) if hasattr(pipeline, "predict_proba") else None
    )
    reporte = evaluate_predictions(y, y_pred, texts=list(textos_orig), probabilities=probabilidades, labels=labels)
    return {
        "accuracy": round(reporte.accuracy, 4),
        "precision_macro": round(float(precision_score(y, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(reporte.f1_macro, 4),
        "por_clase": {
            etiqueta: {k: round(float(v), 4) for k, v in valores.items()}
            for etiqueta, valores in reporte.per_class_metrics.items()
        },
        "matriz_confusion": {
            "etiquetas": list(labels),
            "valores": reporte.confusion_matrix.astype(int).tolist(),
        },
        "_errores": reporte.misclassifications,
    }


def analizar_errores(errores: Sequence[Any], por_tipo: int = 6) -> dict[str, Any]:
    """Los errores más 'seguros' de cada tipo: los más útiles para explicar fallas."""

    def confianza(e: Any) -> float:
        return float((e.probabilities or {}).get(e.predicted_label, 0.0))

    def a_dict(e: Any) -> dict[str, Any]:
        return {
            "texto": e.text,
            "real": e.true_label,
            "predicho": e.predicted_label,
            "confianza": round(confianza(e), 4),
        }

    falsos_positivos = sorted((e for e in errores if e.predicted_label == "positivo"), key=confianza, reverse=True)
    falsos_negativos = sorted((e for e in errores if e.predicted_label == "negativo"), key=confianza, reverse=True)
    return {
        "total_errores": len(errores),
        "falsos_positivos": len(falsos_positivos),
        "falsos_negativos": len(falsos_negativos),
        "ejemplos_falsos_positivos": [a_dict(e) for e in falsos_positivos[:por_tipo]],
        "ejemplos_falsos_negativos": [a_dict(e) for e in falsos_negativos[:por_tipo]],
    }


def terminos_influyentes(pipeline: Pipeline, n: int = 15) -> dict[str, list[dict[str, Any]]]:
    """Términos con más peso hacia cada clase (coeficientes o log-probabilidades)."""
    vocabulario = pipeline.named_steps["vectorizer"].get_feature_names_out()
    clf = pipeline.named_steps["classifier"]
    clases = list(clf.classes_)
    if hasattr(clf, "coef_"):
        # En binario coef_ apunta hacia classes_[1].
        pesos = clf.coef_[0]
    else:
        pesos = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    orden = np.argsort(pesos)
    return {
        clases[1]: [{"termino": vocabulario[i], "peso": round(float(pesos[i]), 4)} for i in orden[::-1][:n]],
        clases[0]: [{"termino": vocabulario[i], "peso": round(float(pesos[i]), 4)} for i in orden[:n]],
    }


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------


def entrenar(config: MLConfig, rapido: bool = False) -> dict[str, Any]:
    inicio_total = time.time()
    np.random.seed(config.experiment.random_seed)
    ds = config.dataset
    if ds.test_path is None:
        raise ValueError("config/ml.yaml debe definir dataset.test_path.")

    log("1/7 Cargando partición estratificada del Módulo 1")
    train = cargar_particion(ds.path, config)
    test = cargar_particion(ds.test_path, config)
    if rapido:
        train = muestra_estratificada(train, MUESTRA_RAPIDA, config)
        test = muestra_estratificada(test, MUESTRA_RAPIDA // 4, config)
    distribucion = train[ds.target_column].value_counts().to_dict()
    log(f"    train={len(train)}  test={len(test)}  clases={distribucion}")

    log("2/7 Comparando stemming, lematización y sin reducción")
    n_morf = min(MUESTRA_COMPARACION_MORFOLOGICA, len(train)) if not rapido else min(2_000, len(train))
    morfologia = comparar_morfologia(train, config, n_morf)

    metodo = config.nlp.metodo_morfologico
    log(f"3/7 Preprocesando todo con '{metodo}' (nlp.metodo_morfologico)")
    preprocesador = PreprocesadorNLP(metodo)
    X_train = preprocesador.transform(train[ds.text_column].tolist())
    X_test = preprocesador.transform(test[ds.text_column].tolist())
    y_train, y_test = train[ds.target_column], test[ds.target_column]

    log("4/7 Línea base vs Regresión Logística vs Naive Bayes, unigramas vs + bigramas")
    experimentos = comparar_modelos_y_ngramas(X_train, y_train, config)

    log(f"5/7 GridSearchCV con n-gramas {config.nlp.ngram_range}")
    ajustes = ajustar_hiperparametros(X_train, y_train, config)

    produccion = max(ajustes, key=lambda k: ajustes[k]["mejor_f1_cv"])
    log(f"6/7 Modelo en producción: {NOMBRES_MODELOS[produccion]} (mejor F1 de validación cruzada)")

    log("7/7 Evaluación única en el conjunto de prueba")
    baseline = pipeline_sobre_texto_procesado("baseline", config).fit(X_train, y_train)
    candidatos = {"baseline": baseline, **{k: v["estimador"] for k, v in ajustes.items()}}
    prueba = {}
    for nombre, pipe in candidatos.items():
        prueba[nombre] = metricas_prueba(pipe, X_test, test[ds.text_column], y_test, ds.labels)
        log(f"  {NOMBRES_MODELOS[nombre]:<34} accuracy={prueba[nombre]['accuracy']:.4f}  F1={prueba[nombre]['f1_macro']:.4f}")
    errores = analizar_errores(prueba[produccion]["_errores"])

    modelo_final = ensamblar_pipeline_produccion(ajustes[produccion]["estimador"], metodo)
    # Verificación: el pipeline guardado (texto crudo) predice igual que el evaluado.
    control = test[ds.text_column].head(50).tolist()
    if list(modelo_final.predict(control)) != list(ajustes[produccion]["estimador"].predict(X_test[:50])):
        raise RuntimeError("El pipeline de producción no reproduce las predicciones evaluadas.")

    modelos = {}
    for nombre in candidatos:
        modelos[nombre] = {
            "nombre": NOMBRES_MODELOS[nombre],
            "en_produccion": nombre == produccion,
            "hiperparametros": (
                {ajustes[nombre]["parametro"]: ajustes[nombre]["mejor_valor"]}
                if nombre in ajustes
                else {"strategy": "most_frequent"}
            ),
            "f1_cv": ajustes[nombre]["mejor_f1_cv"] if nombre in ajustes else None,
            "prueba": {k: v for k, v in prueba[nombre].items() if not k.startswith("_")},
        }

    metricas = {
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modo": "rapido" if rapido else "completo",
        "segundos_entrenamiento": round(time.time() - inicio_total, 1),
        "modelo_en_produccion": produccion,
        "dataset": {
            "fuente": "amazon_reviews_multi (es), reseñas de 1-2 estrellas negativas y 4-5 positivas",
            "train": len(train),
            "test": len(test),
            "distribucion_train": distribucion,
        },
        "nlp": {
            "metodo_morfologico": metodo,
            "ngram_range": list(config.nlp.ngram_range),
            "max_features": config.nlp.max_features,
            "min_df": config.nlp.min_df,
            "max_df": config.nlp.max_df,
            "sublinear_tf": config.nlp.sublinear_tf,
            "vocabulario": len(modelo_final.named_steps["vectorizer"].vocabulary_),
            "stopwords": len(STOPWORDS_ESP),
        },
        "validacion": {
            "cv_folds": config.experiment.cv_folds,
            "metrica_principal": config.experiment.primary_metric,
            "semilla": config.experiment.random_seed,
        },
        "comparacion_morfologica": morfologia,
        "experimentos": experimentos,
        "ajuste_hiperparametros": {
            k: {kk: vv for kk, vv in v.items() if kk != "estimador"} for k, v in ajustes.items()
        },
        "modelos": modelos,
        "analisis_errores": errores,
        "terminos_influyentes": terminos_influyentes(ajustes[produccion]["estimador"]),
        "versiones": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__},
    }
    return {"modelo": modelo_final, "metricas": metricas}


def guardar(resultado: dict[str, Any], config: MLConfig) -> None:
    art = config.artifacts
    art.directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(resultado["modelo"], art.pipeline_path, compress=3)
    with art.metrics_path.open("w", encoding="utf-8") as f:
        json.dump(resultado["metricas"], f, ensure_ascii=False, indent=2)
    filas = [
        {"modelo": NOMBRES_MODELOS[k], **{m: v["prueba"][m] for m in ("accuracy", "precision_macro", "recall_macro", "f1_macro")}}
        for k, v in resultado["metricas"]["modelos"].items()
    ]
    pd.DataFrame(filas).to_csv(art.directory / "comparacion_modelos.csv", index=False)
    log(f"Guardado {art.pipeline_path} y {art.metrics_path}")


def main(args: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Entrena y guarda el clasificador de sentimientos.")
    parser.add_argument("--config", "-c", default=None, help="Ruta alternativa a config/ml.yaml")
    parser.add_argument(
        "--rapido",
        action="store_true",
        help=f"Usa {MUESTRA_RAPIDA} reseñas de entrenamiento (para cambios en vivo).",
    )
    parsed = parser.parse_args(args)
    config = load_config(parsed.config) if parsed.config else load_config()
    resultado = entrenar(config, rapido=parsed.rapido)
    guardar(resultado, config)
    m = resultado["metricas"]
    prod = m["modelos"][m["modelo_en_produccion"]]
    log(f"Listo en {m['segundos_entrenamiento']}s. {prod['nombre']}: F1 prueba = {prod['prueba']['f1_macro']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
