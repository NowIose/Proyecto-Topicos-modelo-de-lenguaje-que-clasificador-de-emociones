"""
Módulo de Vectorización TF-IDF para Modelado NLP.
Módulo 1 — Fase 6 y Fase 7

Implementa la extracción de características textuales mediante TfidfVectorizer
de Scikit-learn, soportando configuraciones de unigramas y bigramas, control de
vocabulario (max_features, min_df, max_df) y persistencia serializada con joblib.
"""
from pathlib import Path
from typing import Tuple, List, Optional, Union
import joblib
import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

class VectorizadorTFIDF:
    """
    Encapsula la configuración, ajuste, transformación y persistencia
    de representaciones numéricas TF-IDF sin fuga de información.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 1),
        max_features: Optional[int] = 10000,
        min_df: Union[int, float] = 2,
        max_df: Union[int, float] = 0.90,
        sublinear_tf: bool = True
    ):
        """
        Inicializa el vectorizador TF-IDF con los hiperparámetros especificados.

        Args:
            ngram_range: Tupla (min_n, max_n). (1, 1) para unigramas, (1, 2) para unigramas + bigramas.
            max_features: Cantidad máxima de términos más representativos a conservar.
            min_df: Frecuencia mínima de documento para incluir un término.
            max_df: Frecuencia máxima relativa/absoluta de documento para excluir términos omnipresentes.
            sublinear_tf: Aplica escalado sublineal (1 + log(tf)) para moderar el impacto de frecuencias altas.
        """
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf

        self._crear_vectorizador(min_df=self.min_df, max_df=self.max_df)
        self.ajustado = False

    def _crear_vectorizador(self, min_df, max_df):
        self.vectorizador = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=self.sublinear_tf
        )

    def _ajustar_dfs_para_muestras_pequenas(self, n_doc: int):
        # Si el número de documentos es muy pequeño (ej. en tests rápidos),
        # evitar que max_df < min_df
        min_count = self.min_df if isinstance(self.min_df, int) else self.min_df * n_doc
        max_count = self.max_df if isinstance(self.max_df, int) else self.max_df * n_doc
        if max_count < min_count or n_doc <= 5:
            self._crear_vectorizador(min_df=1, max_df=1.0)

    def fit(self, textos: List[str]) -> "VectorizadorTFIDF":
        """
        Aprende el vocabulario y las ponderaciones IDF EXCLUSIVAMENTE sobre datos de entrenamiento.
        """
        self._ajustar_dfs_para_muestras_pequenas(len(textos))
        self.vectorizador.fit(textos)
        self.ajustado = True
        return self

    def transform(self, textos: List[str]) -> spmatrix:
        """
        Transforma una lista de textos a la matriz dispersa TF-IDF sin alterar el vocabulario aprendido.
        """
        if not self.ajustado:
            raise RuntimeError("El vectorizador debe ser ajustado (fit) antes de transform().")
        return self.vectorizador.transform(textos)

    def fit_transform(self, textos: List[str]) -> spmatrix:
        """
        Ajusta y transforma en una sola operación (sólo para conjunto de entrenamiento).
        """
        self._ajustar_dfs_para_muestras_pequenas(len(textos))
        matriz = self.vectorizador.fit_transform(textos)
        self.ajustado = True
        return matriz

    @property
    def vocabulario(self) -> np.ndarray:
        """Retorna el array con los nombres de las características (términos) del vocabulario."""
        if not self.ajustado:
            raise RuntimeError("El vectorizador no ha sido ajustado aún.")
        return self.vectorizador.get_feature_names_out()

    @property
    def tamano_vocabulario(self) -> int:
        """Retorna el número total de características aprendidas."""
        return len(self.vocabulario)

    def guardar(self, ruta_archivo: Union[str, Path]) -> None:
        """Serializa y guarda el vectorizador ajustado en disco."""
        ruta = Path(ruta_archivo)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, ruta)

    @classmethod
    def cargar(cls, ruta_archivo: Union[str, Path]) -> "VectorizadorTFIDF":
        """Carga un vectorizador serializado desde el disco."""
        ruta = Path(ruta_archivo)
        if not ruta.exists():
            raise FileNotFoundError(f"No existe el archivo de vectorizador en: {ruta}")
        return joblib.load(ruta)
