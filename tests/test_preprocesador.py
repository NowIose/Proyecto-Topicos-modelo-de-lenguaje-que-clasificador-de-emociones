"""Pruebas del puente entre el NLP del Módulo 1 y los pipelines de scikit-learn."""
import joblib
import pytest
from sklearn.pipeline import Pipeline

from src.models.experiments import build_logistic_regression_pipeline, get_nlp_preprocessors
from src.nlp import PipelineNLP, PreprocesadorNLP


def test_transform_usa_el_mismo_procesamiento_que_pipeline_nlp():
    textos = ["El producto NO funciona, ¡pésimo!", "Excelente compra, llegó rápido"]
    esperado = [PipelineNLP("stemming").procesar_texto(t) for t in textos]
    assert PreprocesadorNLP("stemming").fit(textos).transform(textos) == esperado


def test_conserva_la_negacion():
    salida = PreprocesadorNLP("stemming").transform(["no me gusto"])[0]
    assert "no" in salida.split()


def test_metodo_invalido_falla_en_fit():
    with pytest.raises(ValueError):
        PreprocesadorNLP("otro").fit(["hola"])


def test_inspeccionar_lista_stopwords_eliminadas():
    pasos = PreprocesadorNLP("stemming").inspeccionar("El envío de la caja fue malo")
    assert pasos["texto_normalizado"] == "el envio de la caja fue malo"
    assert "el" in pasos["stopwords_eliminadas"]
    assert "el" not in pasos["tokens_sin_stopwords"]
    assert pasos["texto_final"] == " ".join(pasos["tokens_finales"])


def test_get_nlp_preprocessors_encuentra_stemming_y_lematizacion():
    encontrados = get_nlp_preprocessors()
    assert encontrados is not None
    assert encontrados["stemming"].metodo_morfologico == "stemming"
    assert encontrados["lemmatization"].metodo_morfologico == "lematizacion"


def test_pipeline_con_preprocesador_se_serializa_y_predice(tmp_path):
    textos = ["excelente producto me encanta", "pesimo no funciona nada",
              "muy bueno lo recomiendo", "malo se rompio enseguida"] * 3
    etiquetas = ["positivo", "negativo", "positivo", "negativo"] * 3
    pipe = build_logistic_regression_pipeline(preprocessor=PreprocesadorNLP("stemming"))
    pipe.fit(textos, etiquetas)

    ruta = tmp_path / "modelo.joblib"
    joblib.dump(pipe, ruta)
    cargado: Pipeline = joblib.load(ruta)

    assert list(cargado.named_steps) == ["preprocessor", "vectorizer", "classifier"]
    assert cargado.predict(["excelente, me encanta"])[0] == "positivo"
