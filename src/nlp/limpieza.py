"""
Módulo de Normalización y Limpieza de Texto para NLP.
Módulo 1 — Fase 5.1

Realiza operaciones de limpieza estructuradas para preparar reseñas de e-commerce:
- Conversión a minúsculas.
- Eliminación de URLs y menciones HTML.
- Preservación selectiva de la 'ñ' al remover acentos diacríticos.
- Reducción de repeticiones enfáticas de caracteres (ej. 'buenoooo' -> 'bueno').
- Tratamiento de signos de puntuación y números.
- Normalización de espacios en blanco.
"""
import re
import unicodedata
from typing import Optional

def remover_urls(texto: str) -> str:
    """Elimina enlaces web (http, https, www)."""
    patron_url = r"https?://\S+|www\.\S+"
    return re.sub(patron_url, " ", texto)

def remover_html(texto: str) -> str:
    """Elimina etiquetas HTML que puedan estar presentes en los textos."""
    patron_html = r"<.*?>"
    return re.sub(patron_html, " ", texto)

def normalizar_caracteres_repetidos(texto: str) -> str:
    """
    Reduce repeticiones exageradas de caracteres a un solo carácter.
    Ejemplo: 'holaaaaa' -> 'hola', 'preeeeecio' -> 'precio'.
    """
    return re.sub(r"(.)\1{2,}", r"\1", texto)

def remover_tildes(texto: str) -> str:
    """
    Elimina tildes diacríticas unificando el vocabulario (ej. 'pésimo' -> 'pesimo'),
    pero conservando la 'ñ' y 'Ñ' intactas para preservar la ortografía en español.
    """
    # Proteger temporalmente la ñ
    texto = texto.replace("ñ", "__ene__").replace("Ñ", "__ENE__")
    
    # Descomponer caracteres con tildes y filtrar diacríticos
    texto_descompuesto = unicodedata.normalize("NFKD", texto)
    texto_sin_tildes = "".join(
        c for c in texto_descompuesto if not unicodedata.combining(c)
    )
    
    # Restaurar la ñ
    texto_final = texto_sin_tildes.replace("__ene__", "ñ").replace("__ENE__", "ñ")
    return texto_final

def limpiar_texto(
    texto: Optional[str],
    a_minusculas: bool = True,
    eliminar_urls: bool = True,
    eliminar_html: bool = True,
    desacentuar: bool = True,
    reducir_repeticiones: bool = True,
    eliminar_numeros: bool = True,
    mantener_caracteres_espanol: bool = True
) -> str:
    """
    Pipeline completo de normalización para una reseña.

    Args:
        texto: Cadena de texto a procesar.
        a_minusculas: Si es True, pasa todo a minúsculas.
        eliminar_urls: Si es True, elimina URLs.
        eliminar_html: Si es True, remueve etiquetas HTML.
        desacentuar: Si es True, remueve tildes preservando la 'ñ'.
        reducir_repeticiones: Si es True, acorta secuencias de 3 o más letras iguales.
        eliminar_numeros: Si es True, elimina dígitos numéricos.
        mantener_caracteres_espanol: Si es True, conserva únicamente letras a-z y ñ.

    Returns:
        Cadena normalizada y limpia.
    """
    if texto is None or not isinstance(texto, str):
        return ""

    resultado = texto

    # 1. Eliminar URLs y etiquetas HTML
    if eliminar_urls:
        resultado = remover_urls(resultado)
    if eliminar_html:
        resultado = remover_html(resultado)

    # 2. Conversión a minúsculas
    if a_minusculas:
        resultado = resultado.lower()

    # 3. Reducción de repeticiones enfáticas
    if reducir_repeticiones:
        resultado = normalizar_caracteres_repetidos(resultado)

    # 4. Remover tildes preservando ñ
    if desacentuar:
        resultado = remover_tildes(resultado)

    # 5. Filtrar signos y opcionalmente números
    if mantener_caracteres_espanol:
        if eliminar_numeros:
            # Conservar sólo letras del alfabeto español y espacios
            resultado = re.sub(r"[^a-záéíóúüñ\s]", " ", resultado)
        else:
            # Conservar letras y números
            resultado = re.sub(r"[^a-záéíóúüñ0-9\s]", " ", resultado)

    # 6. Normalizar espacios en blanco (múltiples espacios a uno solo)
    resultado = re.sub(r"\s+", " ", resultado).strip()

    return resultado
