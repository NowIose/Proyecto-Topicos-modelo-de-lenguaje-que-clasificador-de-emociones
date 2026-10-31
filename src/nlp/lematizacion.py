"""
Módulo de Lematización para NLP en Español.
Módulo 1 — Fase 5.5

Implementa la reducción de palabras a su forma canónica o lema de diccionario
mediante spaCy y el modelo neuronal liviano 'es_core_news_sm'.
Se deshabilitan 'parser' y 'ner' para optimizar la velocidad de inferencia.
"""
from typing import List, Union
import spacy

_NLP_MODEL = None

def obtener_modelo_spacy():
    """Carga y almacena en caché el modelo de spaCy optimizado para lematización."""
    global _NLP_MODEL
    if _NLP_MODEL is None:
        try:
            _NLP_MODEL = spacy.load("es_core_news_sm", disable=["parser", "ner"])
        except OSError:
            import os
            # Si no está instalado, intentar descargarlo
            os.system("python -m spacy download es_core_news_sm")
            _NLP_MODEL = spacy.load("es_core_news_sm", disable=["parser", "ner"])
    return _NLP_MODEL

def aplicar_lematizacion(entrada: Union[str, List[str]]) -> List[str]:
    """
    Aplica lematización sobre un texto o lista de tokens.

    Args:
        entrada: Cadena de texto completa o lista de tokens.

    Returns:
        Lista de lemas lingüísticamente válidos en español.

    Ejemplos:
        'productos' -> 'producto'
        'comprando' -> 'comprar'
        'mejores'   -> 'mejor'
        'funciona'  -> 'funcionar'
    """
    if not entrada:
        return []

    nlp = obtener_modelo_spacy()

    if isinstance(entrada, list):
        texto = " ".join(entrada)
    else:
        texto = entrada

    if not texto.strip():
        return []

    doc = nlp(texto)
    # Extraer lema en minúsculas ignorando espacios residuales
    return [token.lemma_.lower() for token in doc if not token.is_space]

def aplicar_lematizacion_texto(texto: str) -> str:
    """
    Lematiza un texto y devuelve los lemas unidos en una sola cadena.

    Args:
        texto: Texto a procesar.

    Returns:
        Cadena con las palabras lematizadas separadas por espacio.
    """
    lemas = aplicar_lematizacion(texto)
    return " ".join(lemas)
