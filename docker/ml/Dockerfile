FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements-ml.txt ./requirements-ml.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-ml.txt

COPY config ./config
COPY src ./src
COPY tests ./tests

RUN mkdir -p /app/data /app/artifacts

# El contenedor es un ejecutor de tareas: las pruebas terminan y el contenedor
# se detiene. El comando de entrenamiento se añadirá en una fase posterior.
CMD ["python", "-m", "pytest", "-q"]
