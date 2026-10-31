"""
Módulo de Tokenización para NLP en Español.
Módulo 1 — Fase 5.2

Transforma cadenas de texto en listas de unidades léxicas (tokens).
Ofrece opciones mediante expresiones regulares especializadas para español y
compatibilidad con NLTK.
"""
import re
from typing import List, Optional

def tokenizar_texto(texto: Optional[str], metodo: str = "regex") -> List[str]:
    """
    Divide un texto en tokens individuales.

    Args:
        texto: Cadena de texto a tokenizar.
        metodo: Método de tokenización ('regex' o 'split').
                'regex' extrae secuencias alfanuméricas considerando caracteres españoles (á, é, í, ó, ú, ü, ñ).
                'split' separa directamente por espacios en blanco.

    Returns:
        Lista de tokens (palabras).
    """
    if not texto or not isinstance(texto, str):
        return []

    texto = texto.strip()
    if not texto:
        return []

    if metodo == "split":
        return [t for t in texto.split() if t]

    # Por defecto 'regex' adaptado a caracteres en español
    patron = r"[a-záéíóúüñA-ZÁÉÍÓÚÜÑ0-9]+"
    tokens = re.findall(patron, texto)
    return tokens
