"""
Script para Fase 6, Fase 7 y Fase 8:
Extracción y serialización de representaciones TF-IDF (unigramas y bigramas)
para el entrenamiento de modelos de Machine Learning (Módulo 2).

Garantiza:
1. Ajuste (fit) únicamente sobre entrenamiento (128.000 reseñas).
2. Transformación (transform) sobre prueba (32.000 reseñas) sin fuga de información.
3. Generación de artefactos en data/features/ y data/processed/.
"""
import sys
import time
from pathlib import Path

# Agregar raíz del repositorio a sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd
from scipy import sparse

from src.nlp.pipeline_nlp import PipelineNLP

# Asegurar codificación utf-8 para la salida en consola Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def ejecutar_extraccion_features():
    dir_features = Path("data/features")
    dir_unigramas = dir_features / "tfidf_unigramas"
    dir_bigramas = dir_features / "tfidf_unigramas_bigramas"
    dir_unigramas.mkdir(parents=True, exist_ok=True)
    dir_bigramas.mkdir(parents=True, exist_ok=True)

    print("1. Cargando particiones de datos (train y test)...")
    df_train = pd.read_csv("data/processed/train_binario.csv")
    df_test = pd.read_csv("data/processed/test_binario.csv")

    print(f"   Train: {len(df_train):,} registros")
    print(f"   Test:  {len(df_test):,} registros")

    # -------------------------------------------------------------
    # Configuración A: TF-IDF Unigramas (1, 1)
    # -------------------------------------------------------------
    print("\n2. Entrenando Pipeline A: TF-IDF Unigramas (1, 1)...")
    t0 = time.time()
    pipe_unigramas = PipelineNLP(
        metodo_morfologico="stemming",
        ngram_range=(1, 1),
        max_features=10000,
        sublinear_tf=True
    )
    X_train_uni = pipe_unigramas.fit_transform(df_train['text'].tolist())
    X_test_uni = pipe_unigramas.transform(df_test['text'].tolist())
    tiempo_uni = time.time() - t0

    print(f"   [OK] Pipeline Unigramas completado en {tiempo_uni:.2f} s")
    print(f"   Vocabulario unigramas: {pipe_unigramas.vectorizador.tamano_vocabulario:,} términos")
    print(f"   Forma matriz train: {X_train_uni.shape}")
    print(f"   Forma matriz test:  {X_test_uni.shape}")

    # Guardar artefactos Configuración A
    pipe_unigramas.guardar(dir_unigramas / "pipeline_unigramas.joblib")
    sparse.save_npz(dir_unigramas / "X_train_tfidf.npz", X_train_uni)
    sparse.save_npz(dir_unigramas / "X_test_tfidf.npz", X_test_uni)
    print("   [OK] Artefactos de unigramas guardados en data/features/tfidf_unigramas/")

    # -------------------------------------------------------------
    # Configuración B: TF-IDF Unigramas + Bigramas (1, 2)
    # -------------------------------------------------------------
    print("\n3. Entrenando Pipeline B: TF-IDF Unigramas + Bigramas (1, 2)...")
    t0 = time.time()
    pipe_bigramas = PipelineNLP(
        metodo_morfologico="stemming",
        ngram_range=(1, 2),
        max_features=25000,
        sublinear_tf=True
    )
    X_train_bi = pipe_bigramas.fit_transform(df_train['text'].tolist())
    X_test_bi = pipe_bigramas.transform(df_test['text'].tolist())
    tiempo_bi = time.time() - t0

    print(f"   [OK] Pipeline Unigramas+Bigramas completado en {tiempo_bi:.2f} s")
    print(f"   Vocabulario unigramas+bigramas: {pipe_bigramas.vectorizador.tamano_vocabulario:,} términos")
    print(f"   Forma matriz train: {X_train_bi.shape}")
    print(f"   Forma matriz test:  {X_test_bi.shape}")

    # Guardar artefactos Configuración B
    pipe_bigramas.guardar(dir_bigramas / "pipeline_unigramas_bigramas.joblib")
    sparse.save_npz(dir_bigramas / "X_train_tfidf.npz", X_train_bi)
    sparse.save_npz(dir_bigramas / "X_test_tfidf.npz", X_test_bi)
    print("   [OK] Artefactos de unigramas+bigramas guardados en data/features/tfidf_unigramas_bigramas/")

    # -------------------------------------------------------------
    # Guardar dataset procesado con texto limpio para Módulo 2
    # -------------------------------------------------------------
    print("\n4. Generando dataset limpio con texto preprocesado...")
    df_train['text_limpio'] = [pipe_unigramas.procesar_texto(t) for t in df_train['text']]
    df_test['text_limpio'] = [pipe_unigramas.procesar_texto(t) for t in df_test['text']]

    df_train['split'] = 'train'
    df_test['split'] = 'test'
    df_final = pd.concat([df_train, df_test], ignore_index=True)

    ruta_dataset_final = Path("data/processed/dataset_final.csv")
    df_final.to_csv(ruta_dataset_final, index=False, encoding="utf-8")
    print(f"   [OK] Dataset final exportado a {ruta_dataset_final.resolve()} ({len(df_final):,} filas)")

    print("\n--- Resumen Comparativo de Representaciones TF-IDF ---")
    print(f"{'Configuración':<25} | {'N-gramas':<10} | {'Max Features':<12} | {'Vocabulario':<12} | {'Tiempo (s)':<10}")
    print("-" * 75)
    print(f"{'Unigramas':<25} | {'(1, 1)':<10} | {10000:<12} | {pipe_unigramas.vectorizador.tamano_vocabulario:<12} | {tiempo_uni:<10.2f}")
    print(f"{'Unigramas + Bigramas':<25} | {'(1, 2)':<10} | {25000:<12} | {pipe_bigramas.vectorizador.tamano_vocabulario:<12} | {tiempo_bi:<10.2f}")
    print("\n¡Extracción y almacenamiento de características finalizada con éxito!")

if __name__ == "__main__":
    ejecutar_extraccion_features()
