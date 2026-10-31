import json
import sys
from pathlib import Path

# Asegurar codificación utf-8 para la salida en consola Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Buscar train.jsonl en posibles rutas relativas
posibles_rutas = [
    Path("data/raw/train.jsonl"),
    Path("../data/raw/train.jsonl"),
    Path("train.jsonl")
]

archivo_path = None
for p in posibles_rutas:
    if p.exists():
        archivo_path = p
        break

if not archivo_path:
    print("Error: No se encontró el archivo train.jsonl en data/raw/")
    sys.exit(1)

print(f"Leyendo dataset desde: {archivo_path.resolve()}\n")

total_registros = 0
etiquetas = {}
columnas = None

with open(archivo_path, "r", encoding="utf-8") as archivo:
    for i, linea in enumerate(archivo):
        registro = json.loads(linea)
        total_registros += 1

        if i == 0:
            columnas = list(registro.keys())

        lbl = registro.get("label")
        etiquetas[lbl] = etiquetas.get(lbl, 0) + 1

        if i < 5:
            print(f"--- Registro {i+1} ---")
            print("ID:", registro.get("id"))
            print("Texto:", registro.get("text"))
            print("Etiqueta (numérica):", registro.get("label"))
            print("Etiqueta (texto):", registro.get("label_text"))
            print("-" * 50)

print(f"\nResumen del Dataset:")
print(f"Total registros: {total_registros}")
print(f"Columnas detectadas: {columnas}")
print(f"Distribución de etiquetas: {sorted(etiquetas.items())}")
