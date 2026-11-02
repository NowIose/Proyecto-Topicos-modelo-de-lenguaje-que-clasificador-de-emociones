"""Pruebas de la API con un modelo pequeño entrenado al vuelo."""
import dataclasses
import json

import joblib
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.esquemas import MAX_CARACTERES, MAX_TEXTOS_POR_LOTE
from src.models.config import load_config
from src.models.experiments import build_logistic_regression_pipeline
from src.nlp import PreprocesadorNLP

POSITIVAS = ["excelente producto me encanta", "muy bueno lo recomiendo", "llego rapido y funciona genial",
             "calidad increible vale la pena", "perfecto tal como se describe"]
NEGATIVAS = ["pesimo no funciona nada", "malo se rompio enseguida", "no lo recomiendo es basura",
             "llego roto y tarde", "horrible calidad no sirve"]


@pytest.fixture()
def cliente(tmp_path, monkeypatch):
    real = load_config()
    config = dataclasses.replace(real, artifacts=dataclasses.replace(real.artifacts, directory=tmp_path))
    pipe = build_logistic_regression_pipeline(preprocessor=PreprocesadorNLP("stemming"))
    pipe.fit(POSITIVAS * 3 + NEGATIVAS * 3, ["positivo"] * 15 + ["negativo"] * 15)
    joblib.dump(pipe, config.artifacts.pipeline_path)
    config.artifacts.metrics_path.write_text(
        json.dumps({
            "generado": "2026-11-01T00:00:00+00:00",
            "modelo_en_produccion": "logistic_regression",
            "modelos": {"logistic_regression": {"nombre": "Regresión Logística"}},
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(api_main.servicio, "config", config)
    with TestClient(api_main.app) as c:
        yield c


@pytest.fixture()
def cliente_sin_modelo(tmp_path, monkeypatch):
    real = load_config()
    config = dataclasses.replace(real, artifacts=dataclasses.replace(real.artifacts, directory=tmp_path / "vacio"))
    monkeypatch.setattr(api_main.servicio, "config", config)
    with TestClient(api_main.app) as c:
        yield c


# ---------------------------------------------------------------- estado
def test_health_con_modelo(cliente):
    r = cliente.get("/api/health")
    assert r.status_code == 200
    assert r.json()["estado"] == "ok"
    assert r.json()["modelo"] == "Regresión Logística"


def test_health_y_predict_sin_modelo(cliente_sin_modelo):
    assert cliente_sin_modelo.get("/api/health").json()["modelo_cargado"] is False
    r = cliente_sin_modelo.post("/api/predict", json={"texto": "muy bueno"})
    assert r.status_code == 503
    assert "src.models.entrenar" in r.json()["error"]


# ------------------------------------------------------------ predicción
def test_predict_devuelve_clase_y_probabilidades(cliente):
    r = cliente.post("/api/predict", json={"texto": "¡Excelente producto, me encanta!"})
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["sentimiento"] == "positivo"
    assert set(cuerpo["probabilidades"]) == {"positivo", "negativo"}
    assert abs(sum(cuerpo["probabilidades"].values()) - 1) < 1e-3
    assert cuerpo["confianza"] == cuerpo["probabilidades"]["positivo"]
    assert cuerpo["texto_procesado"]


def test_predict_terminos_clave_apuntan_al_sentimiento(cliente):
    cuerpo = cliente.post("/api/predict", json={"texto": "horrible, se rompio"}).json()
    assert cuerpo["sentimiento"] == "negativo"
    assert cuerpo["terminos_clave"][0]["aporte"] < 0


def test_predict_lote_resume_por_clase(cliente):
    r = cliente.post("/api/predict/lote", json={"textos": ["me encanta, excelente", "pesimo, no sirve"]})
    assert r.status_code == 200
    assert r.json()["total"] == 2
    assert r.json()["resumen"] == {"negativo": 1, "positivo": 1}


def test_predict_archivo_txt(cliente):
    archivo = "excelente producto\n\npesimo no funciona\n".encode("utf-8")
    r = cliente.post("/api/predict/archivo", files={"archivo": ("resenas.txt", archivo, "text/plain")})
    assert r.status_code == 200
    assert r.json()["total"] == 2


def test_predict_archivo_csv_busca_columna_texto(cliente):
    archivo = "id,texto\n1,me encanta\n2,\"malo, se rompio\"\n".encode("utf-8")
    r = cliente.post("/api/predict/archivo", files={"archivo": ("r.csv", archivo, "text/csv")})
    assert r.status_code == 200
    assert [p["texto"] for p in r.json()["predicciones"]] == ["me encanta", "malo, se rompio"]


# ------------------------------------------------------------ validación
@pytest.mark.parametrize("texto", ["", "    ", "\n\t"])
def test_rechaza_texto_vacio(cliente, texto):
    r = cliente.post("/api/predict", json={"texto": texto})
    assert r.status_code == 422
    assert "vacío" in r.json()["detalle"][0]


def test_rechaza_texto_demasiado_largo(cliente):
    r = cliente.post("/api/predict", json={"texto": "a" * (MAX_CARACTERES + 1)})
    assert r.status_code == 422
    assert str(MAX_CARACTERES) in r.json()["detalle"][0]


def test_rechaza_campo_faltante_y_tipo_incorrecto(cliente):
    assert cliente.post("/api/predict", json={}).json()["detalle"] == ["Falta el campo 'texto'."]
    r = cliente.post("/api/predict", json={"texto": 123})
    assert r.status_code == 422
    assert "Formato incorrecto" in r.json()["detalle"][0]


def test_rechaza_json_mal_formado(cliente):
    r = cliente.post("/api/predict", content=b"{texto: hola", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_rechaza_texto_sin_palabras_analizables(cliente):
    r = cliente.post("/api/predict", json={"texto": "!!! 123 ???"})
    assert r.status_code == 422
    assert "palabras analizables" in r.json()["error"]


def test_rechaza_lote_vacio_y_demasiado_grande(cliente):
    assert cliente.post("/api/predict/lote", json={"textos": []}).status_code == 422
    r = cliente.post("/api/predict/lote", json={"textos": ["ok"] * (MAX_TEXTOS_POR_LOTE + 1)})
    assert r.status_code == 422


def test_rechaza_archivo_de_formato_no_soportado(cliente):
    r = cliente.post("/api/predict/archivo", files={"archivo": ("a.pdf", b"%PDF", "application/pdf")})
    assert r.status_code == 422
    assert ".txt" in r.json()["error"]


def test_rechaza_csv_sin_columna_de_texto(cliente):
    r = cliente.post("/api/predict/archivo", files={"archivo": ("a.csv", b"id,nota\n1,5\n", "text/csv")})
    assert r.status_code == 422
    assert "columna" in r.json()["error"]


# ------------------------------------------------------- NLP y métricas
def test_preprocess_muestra_cada_paso(cliente):
    r = cliente.post("/api/preprocess", json={"texto": "El envío NO llegó, ¡pésimo!"})
    assert r.status_code == 200
    pasos = r.json()
    assert pasos["texto_normalizado"] == "el envio no llego pesimo"
    assert "el" in pasos["stopwords_eliminadas"]
    assert "no" in pasos["tokens_sin_stopwords"]
    assert pasos["metodo_morfologico"] == "stemming"


def test_metrics_devuelve_el_json_del_entrenamiento(cliente):
    r = cliente.get("/api/metrics")
    assert r.status_code == 200
    assert r.json()["modelo_en_produccion"] == "logistic_regression"


def test_cuerpo_ilegible_y_ruta_inexistente_responden_en_espanol(cliente):
    r = cliente.post("/api/predict", content=b"\xff\xfe", headers={"Content-Type": "application/json"})
    assert r.status_code in (400, 422)
    assert "error" in r.json()
    r = cliente.get("/api/no-existe")
    assert r.status_code == 404
    assert "Ruta no encontrada" in r.json()["error"]
