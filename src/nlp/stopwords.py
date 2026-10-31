"""
Módulo de Gestión y Filtrado de Stopwords para NLP en Español.
Módulo 1 — Fase 5.3

Proporciona la lista de palabras vacías (stopwords) con preservación
explícita de negaciones y modificadores de intensidad para evitar la inversión
o pérdida del sentido en el análisis de sentimientos.
"""
from typing import List, Set, Optional
import nltk
from nltk.corpus import stopwords as nltk_stopwords

# Palabras clave que alteran o definen la polaridad del sentimiento
# y NUNCA deben eliminarse en un sistema de análisis de opiniones
NEGACIONES_Y_MODIFICADORES: Set[str] = {
    "no",
    "nada",
    "ni",
    "nunca",
    "jamas",
    "jamás",
    "tampoco",
    "sin",
    "muy",
    "pero",
    "mas",
    "más",
    "poco",
    "bueno",
    "buena",
    "mal",
    "mala",
    "malo"
}

def obtener_stopwords(
    preservar_negaciones: bool = True,
    desacentuar_stopwords: bool = True,
    palabras_extra: Optional[List[str]] = None
) -> Set[str]:
    """
    Retorna el conjunto de stopwords en español listo para usar.

    Args:
        preservar_negaciones: Si es True, retira del conjunto de eliminación las palabras
                              de negación e intensificadores (ej. 'no', 'nunca', 'muy', 'sin').
        desacentuar_stopwords: Si es True, agrega variantes sin tilde de las stopwords
                               (ej. 'él' -> 'el', 'más' -> 'mas') para coincidir con texto normalizado.
        palabras_extra: Lista opcional de términos adicionales a incluir como stopwords.

    Returns:
        Set de palabras vacías a eliminar.
    """
    try:
        base_sw = set(nltk_stopwords.words("spanish"))
    except LookupError:
        nltk.download("stopwords")
        base_sw = set(nltk_stopwords.words("spanish"))

    if desacentuar_stopwords:
        import unicodedata
        sw_sin_tildes = set()
        for w in base_sw:
            w_norm = "".join(
                c for c in unicodedata.normalize("NFKD", w)
                if not unicodedata.combining(c)
            )
            sw_sin_tildes.add(w_norm)
        base_sw.update(sw_sin_tildes)

    if palabras_extra:
        base_sw.update(palabras_extra)

    if preservar_negaciones:
        base_sw.difference_update(NEGACIONES_Y_MODIFICADORES)

    return base_sw

# Instancia global predeterminada
STOPWORDS_ESP: Set[str] = obtener_stopwords(preservar_negaciones=True)

def eliminar_stopwords(
    tokens: List[str],
    stopwords_set: Optional[Set[str]] = None
) -> List[str]:
    """
    Filtra una lista de tokens eliminando las stopwords especificadas.

    Args:
        tokens: Lista de palabras a filtrar.
        stopwords_set: Conjunto de stopwords. Si es None, usa STOPWORDS_ESP.

    Returns:
        Lista de tokens sin stopwords.
    """
    if not tokens:
        return []

    sw = stopwords_set if stopwords_set is not None else STOPWORDS_ESP
    return [t for t in tokens if t.lower() not in sw]
