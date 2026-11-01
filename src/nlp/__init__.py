"""
Paquete NLP de CritiqAI (Opinómetro).
Contiene módulos para normalización, tokenización, stopwords, stemming,
lematización, vectorización y pipeline unificado.
"""
from src.nlp.limpieza import limpiar_texto, remover_tildes, remover_urls
from src.nlp.tokenizacion import tokenizar_texto
from src.nlp.stopwords import eliminar_stopwords, obtener_stopwords, STOPWORDS_ESP, NEGACIONES_Y_MODIFICADORES
from src.nlp.stemming import aplicar_stemming, aplicar_stemming_texto
from src.nlp.lematizacion import aplicar_lematizacion, aplicar_lematizacion_texto
from src.nlp.vectorizacion import VectorizadorTFIDF
from src.nlp.pipeline_nlp import PipelineNLP

__all__ = [
    "limpiar_texto",
    "remover_tildes",
    "remover_urls",
    "tokenizar_texto",
    "eliminar_stopwords",
    "obtener_stopwords",
    "STOPWORDS_ESP",
    "NEGACIONES_Y_MODIFICADORES",
    "aplicar_stemming",
    "aplicar_stemming_texto",
    "aplicar_lematizacion",
    "aplicar_lematizacion_texto",
    "VectorizadorTFIDF",
    "PipelineNLP",
]
