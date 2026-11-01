"""API REST del clasificador de sentimientos (FastAPI).

Levantar:  uvicorn api.main:app --reload
Docs:      http://localhost:8000/docs

Se eligió FastAPI porque valida la entrada con Pydantic, genera la
documentación OpenAPI sola y comparte lenguaje con el pipeline de ML,
así el modelo se carga con el mismo código con que se entrenó.
"""
import csv
import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.esquemas import (
    MAX_TEXTOS_POR_LOTE,
    EntradaLote,
    EntradaTexto,
    Error,
    Estado,
    Prediccion,
    Preprocesamiento,
    ResultadoLote,
)
from api.servicio import ServicioModelo, TextoSinContenido

VERSION_API = "1.0.0"
MAX_BYTES_ARCHIVO = 1_000_000
COLUMNAS_TEXTO = ("texto", "text", "resena", "reseña", "review", "comentario")

servicio = ServicioModelo()


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    # El modelo se carga una sola vez al arrancar, no en cada solicitud.
    servicio.cargar()
    yield


app = FastAPI(
    title="CritiqAI — Clasificador de sentimientos",
    description=(
        "Clasifica reseñas de productos en español como positivas o negativas. "
        "El preprocesamiento NLP y el modelo vienen del mismo pipeline serializado "
        "que se usó al entrenar (`python -m src.models.entrenar`)."
    ),
    version=VERSION_API,
    lifespan=ciclo_de_vida,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

RESPUESTAS_ERROR = {
    422: {"model": Error, "description": "Entrada inválida"},
    503: {"model": Error, "description": "Modelo no disponible"},
}


# --------------------------------------------------------------------- errores
@app.exception_handler(RequestValidationError)
async def error_validacion(request: Request, exc: RequestValidationError):
    detalle = []
    for err in exc.errors():
        mensaje = str(err.get("msg", "")).removeprefix("Value error, ")
        if err.get("type") == "missing":
            mensaje = f"Falta el campo '{err['loc'][-1]}'."
        elif err.get("type") in ("string_type", "list_type", "json_invalid", "model_attributes_type"):
            mensaje = f"Formato incorrecto en '{'.'.join(str(p) for p in err['loc'][1:]) or 'cuerpo'}': {mensaje}"
        detalle.append(mensaje)
    return JSONResponse(status_code=422, content={"error": "Entrada inválida", "detalle": detalle})


@app.exception_handler(StarletteHTTPException)
async def error_http(request: Request, exc: StarletteHTTPException):
    # Cubre también los errores propios de Starlette (404, cuerpo ilegible, etc.).
    mensajes = {400: "No se pudo leer el cuerpo de la solicitud. Envía JSON válido en UTF-8.",
                404: "Ruta no encontrada. Consulta /docs para ver los endpoints."}
    detalle = exc.detail if isinstance(exc, HTTPException) else mensajes.get(exc.status_code, exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"error": detalle})


def requerir_modelo() -> None:
    if not servicio.listo:
        raise HTTPException(503, servicio.error_carga or "El modelo no está cargado.")


def predecir_o_422(funcion, *args):
    try:
        return funcion(*args)
    except TextoSinContenido as exc:
        raise HTTPException(422, str(exc)) from None


# ------------------------------------------------------------------- endpoints
@app.get("/api/health", response_model=Estado, tags=["Servicio"], summary="Estado del servicio")
def health():
    return Estado(
        estado="ok" if servicio.listo else "sin_modelo",
        modelo_cargado=servicio.listo,
        modelo=servicio.nombre_modelo,
        entrenado=servicio.metricas.get("generado") if servicio.metricas else None,
        version_api=VERSION_API,
    )


@app.post(
    "/api/predict",
    response_model=Prediccion,
    responses=RESPUESTAS_ERROR,
    tags=["Predicción"],
    summary="Clasifica una reseña",
)
def predict(entrada: EntradaTexto):
    requerir_modelo()
    return predecir_o_422(servicio.predecir, entrada.texto)


@app.post(
    "/api/predict/lote",
    response_model=ResultadoLote,
    responses=RESPUESTAS_ERROR,
    tags=["Predicción"],
    summary=f"Clasifica hasta {MAX_TEXTOS_POR_LOTE} reseñas",
)
def predict_lote(entrada: EntradaLote):
    requerir_modelo()
    return predecir_o_422(servicio.predecir_lote, entrada.textos)


def leer_textos_de_archivo(nombre: str, contenido: bytes) -> List[str]:
    """Extrae reseñas de un .txt (una por línea) o de un .csv (columna de texto)."""
    try:
        texto = contenido.decode("utf-8-sig")
    except UnicodeDecodeError:
        texto = contenido.decode("latin-1")

    extension = Path(nombre or "").suffix.lower()
    if extension == ".txt":
        return [linea.strip() for linea in texto.splitlines() if linea.strip()]
    if extension == ".csv":
        filas = list(csv.reader(io.StringIO(texto)))
        if not filas:
            return []
        encabezado = [c.strip().lower() for c in filas[0]]
        columna = next((encabezado.index(c) for c in COLUMNAS_TEXTO if c in encabezado), None)
        if columna is None:
            raise HTTPException(
                422,
                "El CSV debe tener una columna llamada 'texto' (o text, resena, review, comentario).",
            )
        return [f[columna].strip() for f in filas[1:] if len(f) > columna and f[columna].strip()]
    raise HTTPException(422, "Formato de archivo no soportado. Usa .txt (una reseña por línea) o .csv.")


@app.post(
    "/api/predict/archivo",
    response_model=ResultadoLote,
    responses=RESPUESTAS_ERROR,
    tags=["Predicción"],
    summary="Clasifica las reseñas de un archivo .txt o .csv",
)
async def predict_archivo(archivo: UploadFile = File(..., description=".txt o .csv, máx. 1 MB")):
    requerir_modelo()
    contenido = await archivo.read(MAX_BYTES_ARCHIVO + 1)
    if len(contenido) > MAX_BYTES_ARCHIVO:
        raise HTTPException(422, "El archivo supera el máximo de 1 MB.")
    textos = leer_textos_de_archivo(archivo.filename, contenido)
    try:
        lote = EntradaLote(textos=textos)
    except ValueError as exc:
        mensajes = [e["msg"].removeprefix("Value error, ") for e in exc.errors()]
        raise HTTPException(422, f"Archivo inválido: {' '.join(mensajes)}") from None
    return predecir_o_422(servicio.predecir_lote, lote.textos)


@app.post(
    "/api/preprocess",
    response_model=Preprocesamiento,
    responses=RESPUESTAS_ERROR,
    tags=["NLP"],
    summary="Muestra el preprocesamiento paso a paso",
)
def preprocess(entrada: EntradaTexto):
    requerir_modelo()
    return servicio.preprocesar(entrada.texto)


@app.get(
    "/api/metrics",
    responses={503: RESPUESTAS_ERROR[503]},
    tags=["Modelo"],
    summary="Métricas y comparación de modelos",
)
def metrics():
    requerir_modelo()
    return servicio.metricas


# Si el frontend ya fue compilado (frontend/dist), la API también lo sirve,
# así todo el sistema queda en un solo puerto.
DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if DIST.is_dir():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
