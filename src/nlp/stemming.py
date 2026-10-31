"""
Módulo de Stemming (Derivación Morfológica) para NLP en Español.
Módulo 1 — Fase 5.4

Implementa la reducción de palabras a su raíz morfológica común mediante
el algoritmo Snowball Stemmer (Porter 2) adaptado para el idioma español.
"""
from typing import List
from nltk.stem.snowball import SnowballStemmer

# Inicialización del stemmer para español
_STEMMER_ESPANOL = SnowballStemmer("spanish")

def obtener_stemmer() -> SnowballStemmer:
    """Retorna la instancia configurada del stemmer para español."""
    return _STEMMER_ESPANOL

def aplicar_stemming(tokens: List[str]) -> List[str]:
    """
    Aplica stemming sobre una lista de tokens en español.

    Args:
        tokens: Lista de palabras normalizadas.

    Returns:
        Lista de raíces morfológicas (stems).

    Nota:
        Las raíces resultantes no son necesariamente palabras completas o gramaticalmente
        válidas en el diccionario (ej. 'productos' -> 'product', 'comprando' -> 'compr').
    """
    if not tokens:
        return []
    return [_STEMMER_ESPANOL.stem(token) for token in tokens]

def aplicar_stemming_texto(texto: str) -> str:
    """
    Aplica stemming sobre una cadena de texto (asumiendo palabras separadas por espacios).

    Args:
        texto: Texto procesado con palabras separadas por espacio.

    Returns:
        Cadena con las palabras reducidas a sus raíces.
    """
    if not texto:
        return ""
    palabras = texto.split()
    stems = aplicar_stemming(palabras)
    return " ".join(stems)
