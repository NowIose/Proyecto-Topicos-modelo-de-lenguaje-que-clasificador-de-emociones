"""
Pipeline Integrado de Procesamiento NLP para Clasificación de Sentimientos.
Módulo 1 — Fase 8

Unifica secuencialmente:
1. Normalización (limpieza, minúsculas, remoción de URLs/HTML, diacríticos preservando 'ñ').
2. Tokenización especializada.
3. Filtrado de stopwords (preservando rigurosamente negaciones e intensificadores).
4. Reducción morfológica: Stemming (Snowball) o Lematización (spaCy).
5. Vectorización numérica TF-IDF (unigramas o unigramas + bigramas).

Garantiza la consistencia estricta entre entrenamiento e inferencia (contrato con Backend/Frontend).
"""
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import joblib
from scipy.sparse import spmatrix

from src.nlp.limpieza import limpiar_texto
from src.nlp.tokenizacion import tokenizar_texto
from src.nlp.stopwords import eliminar_stopwords, STOPWORDS_ESP
from src.nlp.stemming import aplicar_stemming
from src.nlp.lematizacion import aplicar_lematizacion
from src.nlp.vectorizacion import VectorizadorTFIDF

class PipelineNLP:
    """
    Pipeline integral reutilizable para preprocesamiento y vectorización de reseñas.
    """

    def __init__(
        self,
        metodo_morfologico: str = "stemming",
        ngram_range: tuple = (1, 2),
        max_features: int = 10000,
        min_df: Union[int, float] = 2,
        max_df: Union[int, float] = 0.90,
        sublinear_tf: bool = True
    ):
        """
        Inicializa el pipeline de procesamiento.

        Args:
            metodo_morfologico: 'stemming', 'lematizacion' o 'ninguno'.
            ngram_range: (1, 1) para unigramas, (1, 2) para unigramas + bigramas.
            max_features: Número máximo de términos en el vocabulario TF-IDF.
            min_df: Frecuencia mínima de documento.
            max_df: Frecuencia máxima de documento.
            sublinear_tf: Escalado sublineal para TF-IDF.
        """
        if metodo_morfologico not in ("stemming", "lematizacion", "ninguno"):
            raise ValueError("metodo_morfologico debe ser 'stemming', 'lematizacion' o 'ninguno'")

        self.metodo_morfologico = metodo_morfologico
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf

        self.vectorizador = VectorizadorTFIDF(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf
        )
        self.esta_ajustado = False

    def procesar_texto(self, texto: str) -> str:
        """
        Ejecuta el flujo secuencial de limpieza sobre un texto y retorna la cadena final
        lista para vectorización.
        """
        # 1. Normalización
        limpio = limpiar_texto(texto)
        if not limpio:
            return ""

        # 2. Tokenización
        tokens = tokenizar_texto(limpio)

        # 3. Stopwords con preservación de negaciones
        tokens_filtrados = eliminar_stopwords(tokens, stopwords_set=STOPWORDS_ESP)

        # 4. Reducción morfológica
        if self.metodo_morfologico == "stemming":
            tokens_finales = aplicar_stemming(tokens_filtrados)
        elif self.metodo_morfologico == "lematizacion":
            tokens_finales = aplicar_lematizacion(tokens_filtrados)
        else:
            tokens_finales = tokens_filtrados

        return " ".join(tokens_finales)

    def inspeccionar_paso_a_paso(self, texto: str) -> Dict[str, Any]:
        """
        Ejecuta el pipeline capturando todas las salidas intermedias.
        Diseñado para su visualización y depuración en el Frontend y pruebas.

        Returns:
            Diccionario estructurado con cada etapa del pipeline.
        """
        normalizado = limpiar_texto(texto)
        tokens = tokenizar_texto(normalizado)
        tokens_sin_sw = eliminar_stopwords(tokens, stopwords_set=STOPWORDS_ESP)

        if self.metodo_morfologico == "stemming":
            tokens_morfologicos = aplicar_stemming(tokens_sin_sw)
        elif self.metodo_morfologico == "lematizacion":
            tokens_morfologicos = aplicar_lematizacion(tokens_sin_sw)
        else:
            tokens_morfologicos = tokens_sin_sw

        texto_final = " ".join(tokens_morfologicos)

        tfidf_vec = None
        if self.esta_ajustado and texto_final:
            tfidf_vec = self.vectorizador.transform([texto_final])

        return {
            "texto_original": texto,
            "texto_normalizado": normalizado,
            "tokens": tokens,
            "tokens_sin_stopwords": tokens_sin_sw,
            f"tokens_{self.metodo_morfologico}": tokens_morfologicos,
            "texto_final": texto_final,
            "vector_tfidf": tfidf_vec
        }

    def fit(self, textos: List[str]) -> "PipelineNLP":
        """
        Ajusta el vectorizador TF-IDF con los textos procesados del conjunto de entrenamiento.
        """
        textos_procesados = [self.procesar_texto(t) for t in textos]
        self.vectorizador.fit(textos_procesados)
        self.esta_ajustado = True
        return self

    def transform(self, textos: List[str]) -> spmatrix:
        """
        Aplica el preprocesamiento y transforma los textos a matriz TF-IDF.
        """
        if not self.esta_ajustado:
            raise RuntimeError("El pipeline no ha sido ajustado (fit) aún.")
        textos_procesados = [self.procesar_texto(t) for t in textos]
        return self.vectorizador.transform(textos_procesados)

    def fit_transform(self, textos: List[str]) -> spmatrix:
        """
        Ajusta y transforma en una sola operación (exclusivo para entrenamiento).
        """
        textos_procesados = [self.procesar_texto(t) for t in textos]
        matriz = self.vectorizador.fit_transform(textos_procesados)
        self.esta_ajustado = True
        return matriz

    def guardar(self, ruta_archivo: Union[str, Path]) -> None:
        """Serializa y guarda el pipeline completo (incluyendo el vectorizador ajustado)."""
        ruta = Path(ruta_archivo)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, ruta)

    @classmethod
    def cargar(cls, ruta_archivo: Union[str, Path]) -> "PipelineNLP":
        """Carga un pipeline serializado desde el disco."""
        ruta = Path(ruta_archivo)
        if not ruta.exists():
            raise FileNotFoundError(f"No existe el archivo de pipeline en: {ruta}")
        return joblib.load(ruta)
