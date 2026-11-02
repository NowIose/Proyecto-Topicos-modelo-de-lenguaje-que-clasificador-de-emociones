"""Carga del modelo entrenado y predicción.

La API no reimplementa el NLP: todo pasa por el pipeline serializado por
`python -m src.models.entrenar` (preprocesador -> TF-IDF -> clasificador),
cargado con el SentimentClassifier del Módulo 2.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.models.config import MLConfig, load_config
from src.models.inference import SentimentClassifier


class TextoSinContenido(ValueError):
    """El texto no conserva ninguna palabra después del preprocesamiento."""


class ServicioModelo:
    def __init__(self, config: Optional[MLConfig] = None):
        self.config = config or load_config()
        self.clasificador: Optional[SentimentClassifier] = None
        self.metricas: Optional[Dict[str, Any]] = None
        self.error_carga: Optional[str] = None

    # ------------------------------------------------------------------ carga
    def cargar(self) -> None:
        ruta_modelo = self.config.artifacts.pipeline_path
        ruta_metricas = self.config.artifacts.metrics_path
        try:
            self.clasificador = SentimentClassifier(ruta_modelo)
            self.metricas = json.loads(Path(ruta_metricas).read_text(encoding="utf-8"))
            self.error_carga = None
        except (FileNotFoundError, ValueError, TypeError) as exc:
            self.clasificador = None
            self.metricas = None
            self.error_carga = (
                f"{exc}. Ejecuta 'python -m src.models.entrenar' para generar el modelo."
            )

    @property
    def listo(self) -> bool:
        return self.clasificador is not None

    @property
    def pipeline(self):
        return self.clasificador.pipeline

    @property
    def nombre_modelo(self) -> Optional[str]:
        if not self.metricas:
            return None
        clave = self.metricas["modelo_en_produccion"]
        return self.metricas["modelos"][clave]["nombre"]

    # ------------------------------------------------------------- predicción
    def _procesar(self, texto: str) -> str:
        return self.pipeline.named_steps["preprocessor"].transform([texto])[0]

    def _terminos_clave(self, texto_procesado: str, n: int = 8) -> List[Dict[str, Any]]:
        """Términos del texto con mayor aporte a la decisión (peso x TF-IDF).

        Aporte > 0 empuja hacia 'positivo'; < 0 hacia 'negativo'.
        """
        vectorizador = self.pipeline.named_steps["vectorizer"]
        clf = self.pipeline.named_steps["classifier"]
        if hasattr(clf, "coef_"):
            pesos = clf.coef_[0]
        elif hasattr(clf, "feature_log_prob_"):
            pesos = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        else:
            return []
        signo = 1.0 if list(clf.classes_)[1] == "positivo" else -1.0
        fila = vectorizador.transform([texto_procesado])
        indices = fila.indices
        aportes = signo * pesos[indices] * fila.data
        nombres = vectorizador.get_feature_names_out()
        orden = np.argsort(-np.abs(aportes))[:n]
        return [{"termino": str(nombres[indices[i]]), "aporte": round(float(aportes[i]), 4)} for i in orden]

    def predecir(self, texto: str) -> Dict[str, Any]:
        procesado = self._procesar(texto)
        if not procesado:
            raise TextoSinContenido(
                "El texto no contiene palabras analizables (solo signos, números o stopwords)."
            )
        resultado = self.clasificador.predict(texto)
        return {
            "texto": texto,
            "sentimiento": resultado.label,
            "confianza": round(resultado.confidence, 4),
            "probabilidades": {k: round(float(v), 4) for k, v in resultado.probabilities.items()},
            "texto_procesado": procesado,
            "terminos_clave": self._terminos_clave(procesado),
        }

    def predecir_lote(self, textos: List[str]) -> Dict[str, Any]:
        predicciones = []
        for i, texto in enumerate(textos, start=1):
            try:
                predicciones.append(self.predecir(texto))
            except TextoSinContenido as exc:
                raise TextoSinContenido(f"Texto {i}: {exc}") from None
        resumen: Dict[str, int] = {etiqueta: 0 for etiqueta in self.clasificador.classes}
        for p in predicciones:
            resumen[p["sentimiento"]] += 1
        return {"total": len(predicciones), "resumen": resumen, "predicciones": predicciones}

    def preprocesar(self, texto: str) -> Dict[str, Any]:
        pasos = self.pipeline.named_steps["preprocessor"].inspeccionar(texto)
        vocabulario = self.pipeline.named_steps["vectorizer"].vocabulary_
        terminos = []
        if pasos["texto_final"]:
            analizador = self.pipeline.named_steps["vectorizer"].build_analyzer()
            terminos = list(dict.fromkeys(t for t in analizador(pasos["texto_final"]) if t in vocabulario))
        return {**pasos, "terminos_en_vocabulario": terminos}
