"""Módulo de base de datos local SQLite para historial de predicciones y feedback.

Permite almacenar las reseñas analizadas, registrar el feedback del usuario
y consultar el historial reciente de forma segura y sin dependencias externas.
"""
from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional

RUTA_DB_DEFAULT = Path(__file__).resolve().parents[1] / "data" / "historial.db"


def obtener_conexion(ruta_db: Optional[Path] = None) -> sqlite3.Connection:
    """Abre conexión con la base de datos SQLite con timeout seguro."""
    ruta = ruta_db or RUTA_DB_DEFAULT
    ruta.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(ruta), timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def inicializar_bd(ruta_db: Optional[Path] = None) -> None:
    """Crea la tabla de predicciones si aún no existe."""
    try:
        with obtener_conexion(ruta_db) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predicciones (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fecha TEXT NOT NULL,
                    texto TEXT NOT NULL,
                    sentimiento TEXT NOT NULL,
                    confianza REAL NOT NULL,
                    prob_positivo REAL NOT NULL,
                    prob_negativo REAL NOT NULL,
                    feedback INTEGER DEFAULT NULL
                );
                """
            )
            conn.commit()
    except Exception as e:
        print(f"[DB Warning] Error inicializando base de datos SQLite: {e}")


def guardar_prediccion(
    texto: str,
    sentimiento: str,
    confianza: float,
    prob_positivo: float,
    prob_negativo: float,
    ruta_db: Optional[Path] = None,
) -> Optional[int]:
    """Guarda una predicción en SQLite y devuelve su ID autogenerado."""
    try:
        fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with obtener_conexion(ruta_db) as conn:
            cur = conn.execute(
                """
                INSERT INTO predicciones (fecha, texto, sentimiento, confianza, prob_positivo, prob_negativo)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (fecha_actual, texto, sentimiento, confianza, prob_positivo, prob_negativo),
            )
            conn.commit()
            return cur.lastrowid
    except Exception as e:
        print(f"[DB Warning] No se pudo guardar la predicción: {e}")
        return None


def registrar_feedback(id_prediccion: int, es_correcto: bool, ruta_db: Optional[Path] = None) -> bool:
    """Actualiza si la predicción fue acertada (1) o errónea (0)."""
    try:
        with obtener_conexion(ruta_db) as conn:
            valor = 1 if es_correcto else 0
            cur = conn.execute(
                "UPDATE predicciones SET feedback = ? WHERE id = ?",
                (valor, id_prediccion),
            )
            conn.commit()
            return cur.rowcount > 0
    except Exception as e:
        print(f"[DB Warning] No se pudo registrar feedback: {e}")
        return False


def obtener_historial(limite: int = 50, ruta_db: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Devuelve las últimas predicciones almacenadas ordenadas de más reciente a más antigua."""
    try:
        with obtener_conexion(ruta_db) as conn:
            cur = conn.execute(
                """
                SELECT id, fecha, texto, sentimiento, confianza, prob_positivo, prob_negativo, feedback
                FROM predicciones
                ORDER BY id DESC
                LIMIT ?
                """,
                (limite,),
            )
            filas = cur.fetchall()
            return [dict(f) for f in filas]
    except Exception as e:
        print(f"[DB Warning] Error consultando historial: {e}")
        return []


def limpiar_historial(ruta_db: Optional[Path] = None) -> bool:
    """Elimina todos los registros de la tabla de predicciones."""
    try:
        with obtener_conexion(ruta_db) as conn:
            conn.execute("DELETE FROM predicciones")
            conn.commit()
            return True
    except Exception as e:
        print(f"[DB Warning] Error limpiando historial: {e}")
        return False
