"""Prueba de punta a punta del script de entrenamiento con un dataset diminuto."""
import dataclasses
import json

import joblib
import pandas as pd

from src.models import entrenar as modulo
from src.models.config import load_config

POSITIVAS = ["excelente producto me encanta", "muy bueno lo recomiendo", "llego rapido y funciona genial",
             "calidad increible vale la pena", "perfecto tal como se describe"]
NEGATIVAS = ["pesimo no funciona nada", "malo se rompio enseguida", "no lo recomiendo es basura",
             "llego roto y tarde", "horrible calidad no sirve"]


def _config_temporal(tmp_path):
    filas = [(t, "positivo") for t in POSITIVAS] + [(t, "negativo") for t in NEGATIVAS]
    df = pd.DataFrame(filas * 4, columns=["text", "sentimiento"])
    df["text"] = df["text"] + " " + pd.Series(range(len(df))).astype(str).map(lambda i: f"x{i}")
    df.to_csv(tmp_path / "train.csv", index=False)
    df.head(20).to_csv(tmp_path / "test.csv", index=False)

    real = load_config()
    return dataclasses.replace(
        real,
        dataset=dataclasses.replace(real.dataset, path=tmp_path / "train.csv", test_path=tmp_path / "test.csv"),
        experiment=dataclasses.replace(real.experiment, cv_folds=2),
        nlp=dataclasses.replace(real.nlp, min_df=1, max_df=1.0),
        artifacts=dataclasses.replace(real.artifacts, directory=tmp_path / "artifacts"),
    )


def test_entrenar_guarda_modelo_que_recibe_texto_crudo_y_metricas(tmp_path):
    config = _config_temporal(tmp_path)

    resultado = modulo.entrenar(config)
    modulo.guardar(resultado, config)

    modelo = joblib.load(config.artifacts.pipeline_path)
    assert list(modelo.named_steps) == ["preprocessor", "vectorizer", "classifier"]
    assert modelo.predict(["¡Excelente producto, me encanta!"])[0] in ("positivo", "negativo")

    metricas = json.loads(config.artifacts.metrics_path.read_text(encoding="utf-8"))
    assert metricas["modelo_en_produccion"] in ("logistic_regression", "multinomial_nb")
    assert set(metricas["modelos"]) == {"baseline", "logistic_regression", "multinomial_nb"}
    assert metricas["modelos"]["baseline"]["prueba"]["f1_macro"] <= 0.5
    assert len(metricas["experimentos"]) == 6
    assert {f["metodo"] for f in metricas["comparacion_morfologica"]} == {"stemming", "lematizacion", "ninguno"}
    matriz = metricas["modelos"]["logistic_regression"]["prueba"]["matriz_confusion"]["valores"]
    assert sum(map(sum, matriz)) == 20
