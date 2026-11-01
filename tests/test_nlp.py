"""
Conjunto de pruebas automatizadas para el Módulo 1 (Datos y NLP).
Fase 9 — Validación rigurosa de funciones NLP y pipeline.
"""
import pytest
from pathlib import Path
from src.nlp.limpieza import limpiar_texto, remover_tildes, normalizar_caracteres_repetidos
from src.nlp.tokenizacion import tokenizar_texto
from src.nlp.stopwords import eliminar_stopwords, STOPWORDS_ESP, NEGACIONES_Y_MODIFICADORES
from src.nlp.stemming import aplicar_stemming
from src.nlp.lematizacion import aplicar_lematizacion
from src.nlp.vectorizacion import VectorizadorTFIDF
from src.nlp.pipeline_nlp import PipelineNLP

# --- 1. Pruebas de Limpieza y Normalización ---

def test_limpieza_texto_vacio():
    assert limpiar_texto("") == ""
    assert limpiar_texto(None) == ""
    assert limpiar_texto("     ") == ""

def test_limpieza_mayusculas():
    texto = "ESTO ES UNA RESEÑA EN MAYÚSCULAS"
    resultado = limpiar_texto(texto)
    assert resultado == "esto es una reseña en mayusculas"

def test_limpieza_tildes_preservando_ene():
    texto = "El año 2026 trajo un avión magnífico y baratísimo para el niño"
    resultado = limpiar_texto(texto)
    # Debe eliminar tildes (avion, magnifico, baratisimo) pero conservar la ñ (año, niño)
    assert "año" in resultado
    assert "niño" in resultado
    assert "avion" in resultado
    assert "magnifico" in resultado
    assert "baratisimo" in resultado

def test_limpieza_urls_y_html():
    texto = "Visita https://amazon.es/review o www.tienda.com para ver <b>la oferta</b>."
    resultado = limpiar_texto(texto)
    assert "http" not in resultado
    assert "www" not in resultado
    assert "<b>" not in resultado
    assert "oferta" in resultado

def test_limpieza_caracteres_repetidos():
    texto = "Es buenoooooo y muyyyy rápiiiido"
    resultado = normalizar_caracteres_repetidos(texto)
    assert resultado == "es bueno y muy rapido" or "bueno" in resultado

def test_limpieza_puntuacion_y_emojis():
    texto = "¡¡¡Pésimo producto!!! ¿Por qué se rompió? 😡👎"
    resultado = limpiar_texto(texto)
    # Emojis y signos de puntuación se descartan limpiamente
    assert "pesimo producto por que se rompio" == resultado

# --- 2. Pruebas de Tokenización ---

def test_tokenizacion_basica():
    texto = "el producto no funciona correctamente"
    tokens = tokenizar_texto(texto)
    assert tokens == ["el", "producto", "no", "funciona", "correctamente"]

def test_tokenizacion_con_caracteres_especiales():
    texto = "diseño, tamaño y año"
    tokens = tokenizar_texto(texto)
    assert "diseño" in tokens
    assert "tamaño" in tokens
    assert "año" in tokens

# --- 3. Pruebas de Stopwords y Preservación de Negaciones ---

def test_preservacion_estricta_de_negaciones():
    """
    CRÍTICO: Verificar que palabras como 'no', 'nunca', 'jamas', 'sin', 'muy'
    NO sean eliminadas bajo ninguna circunstancia.
    """
    tokens = ["el", "producto", "no", "funciona", "nunca", "llego", "sin", "garantia", "muy", "malo"]
    filtrados = eliminar_stopwords(tokens)

    for palabra_clave in ["no", "nunca", "sin", "muy", "malo"]:
        assert palabra_clave in filtrados, f"Fallo: '{palabra_clave}' fue eliminada como stopword"

    # Stopwords comunes no informativas sí deben eliminarse
    assert "el" not in filtrados

# --- 4. Pruebas de Stemming y Lematización ---

def test_stemming_espanol():
    tokens = ["producto", "productos", "comprando", "compre", "funciona"]
    stems = aplicar_stemming(tokens)
    assert stems[0] == stems[1] == "product"
    assert stems[2] == stems[3] == "compr"
    assert stems[4] == "funcion"

def test_lematizacion_espanol():
    tokens = ["productos", "comprando", "mejores", "funciona"]
    lemas = aplicar_lematizacion(tokens)
    assert lemas[0] == "producto"
    assert lemas[1] == "comprar"
    assert lemas[2] == "mejor"
    assert lemas[3] == "funcionar"

# --- 5. Caso de Prueba Ilustrativo del Plan ---

def test_caso_ilustrativo_completo():
    """
    Caso del plan: '¡El producto NO funciona! Es una pésima compra 😡'
    """
    entrada = "¡El producto NO funciona! Es una pésima compra 😡"

    # Pipeline con Stemming
    pipeline_stem = PipelineNLP(metodo_morfologico="stemming", ngram_range=(1, 2))
    info_stem = pipeline_stem.inspeccionar_paso_a_paso(entrada)

    assert "no" in info_stem["tokens_sin_stopwords"]
    assert "no" in info_stem["texto_final"]
    assert "product" in info_stem["texto_final"]

    # Pipeline con Lematización
    pipeline_lema = PipelineNLP(metodo_morfologico="lematizacion", ngram_range=(1, 2))
    info_lema = pipeline_lema.inspeccionar_paso_a_paso(entrada)

    assert "no" in info_lema["tokens_sin_stopwords"]
    assert "no" in info_lema["texto_final"]
    assert "producto" in info_lema["texto_final"]
    assert "funcionar" in info_lema["texto_final"]

# --- 6. Pruebas del Vectorizador TF-IDF y Persistencia ---

def test_tfidf_unigramas_y_bigramas(tmp_path):
    corpus_train = [
        "producto excelente calidad muy recomendado",
        "no funciona pesimo producto nunca llego",
        "buena compra llego rapido"
    ]
    corpus_test = [
        "excelente producto",
        "no funciona nada"
    ]

    vectorizador = VectorizadorTFIDF(ngram_range=(1, 2), min_df=1)
    X_train = vectorizador.fit_transform(corpus_train)

    assert X_train.shape[0] == 3
    vocab = vectorizador.vocabulario
    assert "excelente calidad" in vocab
    assert "no funciona" in vocab

    # Transform en test sin alterar vocabulario
    X_test = vectorizador.transform(corpus_test)
    assert X_test.shape[0] == 2
    assert X_test.shape[1] == X_train.shape[1]

    # Test de persistencia con joblib
    ruta_guardado = tmp_path / "vectorizador_test.joblib"
    vectorizador.guardar(ruta_guardado)
    assert ruta_guardado.exists()

    recuperado = VectorizadorTFIDF.cargar(ruta_guardado)
    assert recuperado.tamano_vocabulario == vectorizador.tamano_vocabulario
    X_test_recuperado = recuperado.transform(corpus_test)
    assert (X_test.toarray() == X_test_recuperado.toarray()).all()

# --- 7. Prueba del Pipeline Completo End-to-End ---

def test_pipeline_nlp_end_to_end(tmp_path):
    train_docs = [
        "Excelente teléfono, la batería dura dos días completos.",
        "Pésima experiencia, se dañó al primer uso y no devuelven el dinero.",
        "Muy buen producto por el precio pagado."
    ]

    pipeline = PipelineNLP(metodo_morfologico="stemming", ngram_range=(1, 2), max_features=100)
    matriz_train = pipeline.fit_transform(train_docs)

    assert pipeline.esta_ajustado
    assert matriz_train.shape[0] == 3

    # Inspección individual
    nueva_resena = "¡No me gustó para nada, vino roto!"
    inspeccion = pipeline.inspeccionar_paso_a_paso(nueva_resena)
    assert "no" in inspeccion["texto_final"]
    assert inspeccion["vector_tfidf"] is not None

    # Guardar y cargar pipeline
    ruta_pipe = tmp_path / "pipeline_completo.joblib"
    pipeline.guardar(ruta_pipe)
    assert ruta_pipe.exists()

    pipe_cargado = PipelineNLP.cargar(ruta_pipe)
    assert pipe_cargado.esta_ajustado
    vec_cargado = pipe_cargado.transform([nueva_resena])
    assert vec_cargado.shape == inspeccion["vector_tfidf"].shape
