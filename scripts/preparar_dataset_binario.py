"""
Script para Fase 3 (Preparación de etiquetas binarias) y Fase 4 (División estratificada)
del Módulo 1.
"""
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Asegurar codificación utf-8 para la salida en consola Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def preparar_dataset_binario():
    # 1. Rutas
    ruta_raw = Path("data/raw/train.jsonl")
    dir_processed = Path("data/processed")
    dir_processed.mkdir(parents=True, exist_ok=True)

    print(f"1. Cargando datos originales desde: {ruta_raw.resolve()}")
    df = pd.read_json(ruta_raw, lines=True)
    total_original = len(df)
    print(f"   Total registros originales: {total_original:,}")

    # 2. Excluir neutrales (etiqueta 2)
    df_filtrado = df[df['label'] != 2].copy()
    total_excluidos = total_original - len(df_filtrado)
    print(f"   Neutrales (etiqueta 2) excluidos: {total_excluidos:,}")

    # 3. Mapear a sentimiento binario
    # 0, 1 -> negativo (0)
    # 3, 4 -> positivo (1)
    mapeo_sentimiento = {0: "negativo", 1: "negativo", 3: "positivo", 4: "positivo"}
    mapeo_polaridad = {0: 0, 1: 0, 3: 1, 4: 1}

    df_filtrado['sentimiento'] = df_filtrado['label'].map(mapeo_sentimiento)
    df_filtrado['polaridad'] = df_filtrado['label'].map(mapeo_polaridad)

    # Verificar que no queden nulos
    assert df_filtrado['sentimiento'].isnull().sum() == 0, "Error: Existen etiquetas no asignadas"
    
    conteo_binario = df_filtrado['sentimiento'].value_counts()
    print("\n2. Distribución en el conjunto binario (160.000 esperados):")
    for clase, cant in conteo_binario.items():
        print(f"   - {clase}: {cant:,} ({cant/len(df_filtrado)*100:.2f}%)")

    # 4. División estratificada Train / Test (80% / 20%)
    print("\n3. Realizando partición estratificada (80% train / 20% test, random_state=42)...")
    train_df, test_df = train_test_split(
        df_filtrado,
        test_size=0.20,
        random_state=42,
        stratify=df_filtrado['sentimiento']
    )

    df_filtrado['split'] = 'train'
    df_filtrado.loc[test_df.index, 'split'] = 'test'

    print(f"   - Entrenamiento (train): {len(train_df):,} registros")
    print(f"     Distribución train:\n{train_df['sentimiento'].value_counts().to_string(header=False)}")
    print(f"   - Prueba (test): {len(test_df):,} registros")
    print(f"     Distribución test:\n{test_df['sentimiento'].value_counts().to_string(header=False)}")

    # 5. Guardar archivos procesados
    ruta_binario_completo = dir_processed / "resenas_binarias.csv"
    ruta_train = dir_processed / "train_binario.csv"
    ruta_test = dir_processed / "test_binario.csv"

    print("\n4. Guardando datasets procesados en data/processed/...")
    columnas_guardar = ['id', 'text', 'label', 'label_text', 'sentimiento', 'polaridad', 'split']
    df_filtrado[columnas_guardar].to_csv(ruta_binario_completo, index=False, encoding='utf-8')
    train_df[['id', 'text', 'sentimiento', 'polaridad']].to_csv(ruta_train, index=False, encoding='utf-8')
    test_df[['id', 'text', 'sentimiento', 'polaridad']].to_csv(ruta_test, index=False, encoding='utf-8')

    print(f"   [OK] {ruta_binario_completo.resolve()} ({len(df_filtrado):,} filas)")
    print(f"   [OK] {ruta_train.resolve()} ({len(train_df):,} filas)")
    print(f"   [OK] {ruta_test.resolve()} ({len(test_df):,} filas)")
    print("\nFase 3 y Fase 4 completadas exitosamente.")

if __name__ == "__main__":
    preparar_dataset_binario()
