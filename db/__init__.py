"""Paquete de base de datos local SQLite para CritiqAI."""
from db.database import (
    inicializar_bd,
    guardar_prediccion,
    registrar_feedback,
    obtener_historial,
    limpiar_historial,
)

__all__ = [
    "inicializar_bd",
    "guardar_prediccion",
    "registrar_feedback",
    "obtener_historial",
    "limpiar_historial",
]
