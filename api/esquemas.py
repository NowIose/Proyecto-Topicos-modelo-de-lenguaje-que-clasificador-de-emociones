"""Modelos de entrada y salida de la API (validados con Pydantic)."""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# Límites de entrada. Una reseña de Amazon promedia ~150 caracteres y la más
# larga del dataset tiene ~3100, así que 5000 deja margen sin aceptar libros.
MAX_CARACTERES = 5000
MAX_TEXTOS_POR_LOTE = 500


def _validar_texto(valor: str) -> str:
    texto = valor.strip()
    if not texto:
        raise ValueError("El texto está vacío. Escribe una reseña para analizar.")
    if len(texto) > MAX_CARACTERES:
        raise ValueError(
            f"El texto tiene {len(texto)} caracteres; el máximo permitido es {MAX_CARACTERES}."
        )
    return texto


class EntradaTexto(BaseModel):
    texto: str = Field(
        ...,
        description=f"Reseña en español (1 a {MAX_CARACTERES} caracteres).",
        examples=["El producto llegó roto y nadie me respondió. No lo recomiendo."],
    )

    @field_validator("texto")
    @classmethod
    def texto_valido(cls, valor: str) -> str:
        return _validar_texto(valor)


class EntradaLote(BaseModel):
    textos: List[str] = Field(
        ...,
        description=f"Entre 1 y {MAX_TEXTOS_POR_LOTE} reseñas.",
        examples=[["Me encantó, llegó antes de tiempo", "Pésima calidad, se rompió"]],
    )

    @field_validator("textos")
    @classmethod
    def textos_validos(cls, valores: List[str]) -> List[str]:
        if not valores:
            raise ValueError("La lista de textos está vacía.")
        if len(valores) > MAX_TEXTOS_POR_LOTE:
            raise ValueError(
                f"Se recibieron {len(valores)} textos; el máximo por solicitud es {MAX_TEXTOS_POR_LOTE}."
            )
        validados = []
        for i, valor in enumerate(valores, start=1):
            try:
                validados.append(_validar_texto(valor))
            except ValueError as exc:
                raise ValueError(f"Texto {i}: {exc}") from None
        return validados


class TerminoClave(BaseModel):
    termino: str
    aporte: float = Field(..., description="> 0 empuja hacia 'positivo', < 0 hacia 'negativo'.")


class Prediccion(BaseModel):
    texto: str
    sentimiento: str = Field(..., description="Clase predicha: 'positivo' o 'negativo'.")
    confianza: float = Field(..., description="Probabilidad de la clase predicha (0 a 1).")
    probabilidades: Dict[str, float]
    texto_procesado: str = Field(..., description="Texto tras el preprocesamiento NLP.")
    terminos_clave: List[TerminoClave] = Field(
        default_factory=list, description="Términos del texto con más peso en la decisión."
    )


class ResultadoLote(BaseModel):
    total: int
    resumen: Dict[str, int] = Field(..., description="Cantidad de reseñas por sentimiento.")
    predicciones: List[Prediccion]


class Preprocesamiento(BaseModel):
    metodo_morfologico: str
    texto_original: str
    texto_normalizado: str
    tokens: List[str]
    stopwords_eliminadas: List[str]
    tokens_sin_stopwords: List[str]
    tokens_finales: List[str]
    texto_final: str
    terminos_en_vocabulario: List[str] = Field(
        ..., description="Términos (o bigramas) del texto que el modelo conoce."
    )


class Estado(BaseModel):
    estado: str
    modelo_cargado: bool
    modelo: Optional[str] = None
    entrenado: Optional[str] = None
    version_api: str


class Error(BaseModel):
    error: str
    detalle: Optional[List[str]] = None
