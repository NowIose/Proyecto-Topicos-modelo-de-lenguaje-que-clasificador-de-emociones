"""
Transformador scikit-learn que envuelve el PipelineNLP del Módulo 1.

Permite meter la limpieza, tokenización, stopwords y stemming/lematización
DENTRO del Pipeline de scikit-learn que se entrena y se serializa. Así el
artefacto guardado contiene el mismo preprocesamiento que se usó al entrenar
y el backend solo lo carga: no reimplementa ningún paso de NLP.
"""
from typing import Any, Dict, Iterable, List

from sklearn.base import BaseEstimator, TransformerMixin

from src.nlp.pipeline_nlp import PipelineNLP


class PreprocesadorNLP(BaseEstimator, TransformerMixin):
    """
    Paso 'preprocessor' del pipeline de clasificación.

    No aprende nada en fit (cada texto se procesa por separado), por lo que
    no hay fuga de información entre entrenamiento y prueba.

    Args:
        metodo_morfologico: 'stemming', 'lematizacion' o 'ninguno'.
    """

    def __init__(self, metodo_morfologico: str = "stemming"):
        self.metodo_morfologico = metodo_morfologico

    def _pipeline(self) -> PipelineNLP:
        # Se reutiliza la clase de José tal cual; solo se usan sus pasos de texto,
        # la vectorización TF-IDF la hace el paso 'vectorizer' del pipeline sklearn.
        return PipelineNLP(metodo_morfologico=self.metodo_morfologico)

    def fit(self, X: Iterable[str], y: Any = None) -> "PreprocesadorNLP":
        self._pipeline()  # valida el método morfológico
        return self

    def transform(self, X: Iterable[str]) -> List[str]:
        pipeline = self._pipeline()
        return [pipeline.procesar_texto(str(texto)) for texto in X]

    def inspeccionar(self, texto: str) -> Dict[str, Any]:
        """
        Devuelve cada etapa del preprocesamiento de un texto, para mostrarla
        en el frontend. Incluye la lista de stopwords que se eliminaron.
        """
        pasos = self._pipeline().inspeccionar_paso_a_paso(texto)
        conservados = set(pasos["tokens_sin_stopwords"])
        return {
            "metodo_morfologico": self.metodo_morfologico,
            "texto_original": pasos["texto_original"],
            "texto_normalizado": pasos["texto_normalizado"],
            "tokens": pasos["tokens"],
            "stopwords_eliminadas": [t for t in pasos["tokens"] if t not in conservados],
            "tokens_sin_stopwords": pasos["tokens_sin_stopwords"],
            "tokens_finales": pasos[f"tokens_{self.metodo_morfologico}"],
            "texto_final": pasos["texto_final"],
        }


# Nombres que busca src.models.experiments.get_nlp_preprocessors() para
# comparar stemming contra lematización con el NLP real del Módulo 1.
stemmer = PreprocesadorNLP("stemming")
lemmatizer = PreprocesadorNLP("lematizacion")
