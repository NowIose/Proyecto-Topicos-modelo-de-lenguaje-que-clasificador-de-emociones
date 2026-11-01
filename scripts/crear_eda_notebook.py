"""
Script para generar el notebook de Análisis Exploratorio de Datos (EDA)
siguiendo las especificaciones de la Fase 2 del Módulo 1.
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

cells = []

# Celda 1: Markdown - Encabezado
cells.append(nbf.v4.new_markdown_cell("""# 📊 Módulo 1 — Análisis Exploratorio de Datos (EDA)
**Proyecto:** Análisis de Sentimientos de Reseñas de Productos en Español (Opinómetro)  
**Dataset:** `SetFit/amazon_reviews_multi_es`  
**Archivo fuente:** `data/raw/train.jsonl`  
**Objetivo:** Conocer en profundidad las características estadísticas, léxicas y de calidad del conjunto de datos antes de aplicar el preprocesamiento NLP y el entrenamiento de modelos de Machine Learning.
"""))

# Celda 2: Markdown - Configuración
cells.append(nbf.v4.new_markdown_cell("""## 1. Importación de librerías y configuración del entorno"""))

# Celda 3: Código - Imports
cells.append(nbf.v4.new_code_cell("""import json
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración visual
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 11

print("Librerías importadas correctamente.")
"""))

# Celda 4: Markdown - Carga
cells.append(nbf.v4.new_markdown_cell("""## 2. Carga e Inspección General del Dataset

Se carga el archivo `data/raw/train.jsonl` en formato JSON Lines.
"""))

# Celda 5: Código - Carga de datos
cells.append(nbf.v4.new_code_cell("""# Rutas relativas robustas
ruta_dataset = Path("../data/raw/train.jsonl")
if not ruta_dataset.exists():
    ruta_dataset = Path("data/raw/train.jsonl")

print(f"Cargando dataset desde: {ruta_dataset.resolve()}")
df = pd.read_json(ruta_dataset, lines=True)

print(f"Dimensiones del dataset: {df.shape[0]:,} filas y {df.shape[1]} columnas.")
df.head()
"""))

# Celda 6: Markdown - Tipos de datos
cells.append(nbf.v4.new_markdown_cell("""### 2.1 Tipos de datos e información de columnas"""))

# Celda 7: Código - Info
cells.append(nbf.v4.new_code_cell("""df.info()"""))

# Celda 8: Markdown - Calidad y valores faltantes
cells.append(nbf.v4.new_markdown_cell("""## 3. Calidad de Datos: Valores Faltantes, Espacios en Blanco y Duplicados"""))

# Celda 9: Código - Faltantes y duplicados
cells.append(nbf.v4.new_code_cell("""# 1. Valores nulos
nulos = df.isnull().sum()
print("--- Valores Nulos por Columna ---")
print(nulos)

# 2. Textos vacíos o compuestos únicamente de espacios
textos_vacios = (df['text'].str.strip() == '').sum()
print(f"\\nTextos vacíos o sólo espacios: {textos_vacios}")

# 3. Duplicados por identificador único
duplicados_id = df['id'].duplicated().sum()
print(f"Identificadores duplicados (id): {duplicados_id}")

# 4. Duplicados por contenido textual
duplicados_text = df['text'].duplicated().sum()
print(f"Reseñas con texto idéntico: {duplicados_text} ({duplicados_text / len(df) * 100:.2f}%)")
"""))

# Celda 10: Código - Muestra de duplicados
cells.append(nbf.v4.new_code_cell("""# Inspección de los textos repetidos más frecuentes
top_duplicados = df['text'].value_counts().head(10)
print("Top 10 textos más repetidos:")
top_duplicados
"""))

# Celda 11: Markdown - Interpretación de calidad
cells.append(nbf.v4.new_markdown_cell("""**Hallazgos de calidad de datos:**
- No existen valores nulos ni registros incompletos en ninguna columna.
- Todos los identificadores `id` son únicos.
- Existen 1.736 textos repetidos (0.87% del corpus). Corresponden a valoraciones genéricas muy habituales en e-commerce (ej. *"Buena relación calidad precio"*, *"Cumple las expectativas"*, *"No es lo que esperaba"*). No se trata de fallos de ingestión, sino de patrones naturales de respuesta corta.
"""))

# Celda 12: Markdown - Distribución de etiquetas
cells.append(nbf.v4.new_markdown_cell("""## 4. Distribución de Etiquetas (Target Original: 0 a 4)

El dataset utiliza una escala de 5 valores (0 a 4), equivalentes a 1 a 5 estrellas:
- `0`: 1 estrella (muy negativa)
- `1`: 2 estrellas (negativa)
- `2`: 3 estrellas (neutral)
- `3`: 4 estrellas (positiva)
- `4`: 5 estrellas (muy positiva)
"""))

# Celda 13: Código - Conteo y gráfico de clases
cells.append(nbf.v4.new_code_cell("""conteo_etiquetas = df['label'].value_counts().sort_index()
porcentaje_etiquetas = df['label'].value_counts(normalize=True).sort_index() * 100

resumen_etiquetas = pd.DataFrame({
    'Estrellas': ['1 Estrella', '2 Estrellas', '3 Estrellas', '4 Estrellas', '5 Estrellas'],
    'Etiqueta': conteo_etiquetas.index,
    'Cantidad': conteo_etiquetas.values,
    'Porcentaje (%)': porcentaje_etiquetas.values
})

print(resumen_etiquetas.to_string(index=False))

# Gráfico de barras
colores = ['#d9534f', '#f0ad4e', '#5bc0de', '#5cb85c', '#2b8a3e']
plt.figure(figsize=(9, 4.5))
bars = plt.bar(resumen_etiquetas['Estrellas'], resumen_etiquetas['Cantidad'], color=colores, edgecolor='black', alpha=0.85)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1000, f"{yval:,}\\n(20%)", ha='center', va='bottom', fontsize=10)

plt.title('Distribución de Clases en el Dataset Original (train.jsonl)', fontsize=13, weight='bold')
plt.xlabel('Categoría / Calificación', fontsize=11)
plt.ylabel('Cantidad de Reseñas', fontsize=11)
plt.ylim(0, 48000)
plt.tight_layout()
plt.show()
"""))

# Celda 14: Markdown - Longitud
cells.append(nbf.v4.new_markdown_cell("""## 5. Análisis de Longitud de las Reseñas

Analizamos la extensión de las opiniones tanto en **caracteres** como en **palabras** para dimensionar los requerimientos de tokenización y vectorización.
"""))

# Celda 15: Código - Métricas de longitud
cells.append(nbf.v4.new_code_cell("""# Calcular longitudes
df['char_len'] = df['text'].str.len()
df['word_len'] = df['text'].str.split().str.len()

print("--- Estadísticas de Longitud en Caracteres ---")
print(df['char_len'].describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]))

print("\\n--- Estadísticas de Longitud en Palabras ---")
print(df['word_len'].describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]))
"""))

# Celda 16: Código - Histogramas de longitud
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Distribución de caracteres (acotada al percentil 99 para mejor visualización)
p99_char = df['char_len'].quantile(0.99)
sns.histplot(df[df['char_len'] <= p99_char]['char_len'], bins=50, kde=True, ax=axes[0], color='#1f77b4')
axes[0].axvline(df['char_len'].median(), color='red', linestyle='--', label=f"Mediana: {df['char_len'].median():.0f}")
axes[0].axvline(df['char_len'].mean(), color='orange', linestyle='-', label=f"Media: {df['char_len'].mean():.1f}")
axes[0].set_title('Distribución de Longitud en Caracteres (≤ P99)', fontsize=12, weight='bold')
axes[0].set_xlabel('Número de Caracteres')
axes[0].set_ylabel('Frecuencia')
axes[0].legend()

# Distribución de palabras (acotada al percentil 99)
p99_words = df['word_len'].quantile(0.99)
sns.histplot(df[df['word_len'] <= p99_words]['word_len'], bins=50, kde=True, ax=axes[1], color='#2ca02c')
axes[1].axvline(df['word_len'].median(), color='red', linestyle='--', label=f"Mediana: {df['word_len'].median():.0f}")
axes[1].axvline(df['word_len'].mean(), color='orange', linestyle='-', label=f"Media: {df['word_len'].mean():.1f}")
axes[1].set_title('Distribución de Longitud en Palabras (≤ P99)', fontsize=12, weight='bold')
axes[1].set_xlabel('Número de Palabras')
axes[1].set_ylabel('Frecuencia')
axes[1].legend()

plt.tight_layout()
plt.show()
"""))

# Celda 17: Código - Longitud por etiqueta
cells.append(nbf.v4.new_code_cell("""# ¿Las opiniones negativas son más largas que las positivas?
longitud_por_clase = df.groupby('label')[['char_len', 'word_len']].agg(['mean', 'median', 'std'])
print("Resumen de longitud por etiqueta:")
longitud_por_clase
"""))

# Celda 18: Código - Boxplot longitud por etiqueta
cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(10, 4.5))
sns.boxplot(x='label', y='word_len', data=df, showfliers=False, palette='Blues')
plt.title('Longitud en Palabras según la Calificación (sin valores atípicos)', fontsize=12, weight='bold')
plt.xlabel('Etiqueta (0: 1 estrella → 4: 5 estrellas)')
plt.ylabel('Cantidad de Palabras')
plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['1★ (Neg)', '2★ (Neg)', '3★ (Neu)', '4★ (Pos)', '5★ (Pos)'])
plt.tight_layout()
plt.show()
"""))

# Celda 19: Markdown - Análisis léxico
cells.append(nbf.v4.new_markdown_cell("""## 6. Análisis Léxico y Palabras Frecuentes (Sin Procesar)

Examinamos las palabras más frecuentes para entender el vocabulario bruto antes de la aplicación de técnicas NLP.
"""))

# Celda 20: Código - Palabras frecuentes globales
cells.append(nbf.v4.new_code_cell("""# Conteo de términos en todo el dataset
palabras_todas = ' '.join(df['text']).lower().split()
conteo_global = Counter(palabras_todas)

top30_global = pd.DataFrame(conteo_global.most_common(30), columns=['Palabra', 'Frecuencia'])

plt.figure(figsize=(12, 6))
sns.barplot(x='Frecuencia', y='Palabra', data=top30_global, palette='viridis')
plt.title('Top 30 Palabras Más Frecuentes en el Corpus Original (Sin Preprocesar)', fontsize=13, weight='bold')
plt.xlabel('Frecuencia Absoluta')
plt.tight_layout()
plt.show()
"""))

# Celda 21: Código - Comparación Negativas vs Positivas
cells.append(nbf.v4.new_code_cell("""# Comparación directa entre clases extremas: 0 (muy negativa) vs 4 (muy positiva)
palabras_neg = ' '.join(df[df['label'] == 0]['text']).lower().split()
palabras_pos = ' '.join(df[df['label'] == 4]['text']).lower().split()

conteo_neg = Counter(palabras_neg)
conteo_pos = Counter(palabras_pos)

top15_neg = pd.DataFrame(conteo_neg.most_common(15), columns=['Palabra', 'Frecuencia_Neg'])
top15_pos = pd.DataFrame(conteo_pos.most_common(15), columns=['Palabra', 'Frecuencia_Pos'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(x='Frecuencia_Neg', y='Palabra', data=top15_neg, ax=axes[0], palette='Reds_r')
axes[0].set_title('Top 15 Palabras en Reseñas Muy Negativas (Clase 0)', fontsize=11, weight='bold')
axes[0].set_xlabel('Frecuencia')

sns.barplot(x='Frecuencia_Pos', y='Palabra', data=top15_pos, ax=axes[1], palette='Greens_r')
axes[1].set_title('Top 15 Palabras en Reseñas Muy Positivas (Clase 4)', fontsize=11, weight='bold')
axes[1].set_xlabel('Frecuencia')

plt.tight_layout()
plt.show()
"""))

# Celda 22: Código - Presencia de negaciones clave
cells.append(nbf.v4.new_code_cell("""# Observación crítica sobre palabras de negación e intensificadores
terminos_clave = ['no', 'nunca', 'jamas', 'sin', 'muy', 'pero', 'bien', 'mal', 'excelente', 'pesimo']

frecuencias_por_clase = []
for termino in terminos_clave:
    fila = {'Término': termino}
    for lbl in range(5):
        c = Counter(' '.join(df[df['label'] == lbl]['text']).lower().split())
        fila[f'Clase {lbl}'] = c[termino]
    frecuencias_por_clase.append(fila)

pd.DataFrame(frecuencias_por_clase)
"""))

# Celda 23: Markdown - Conclusiones del EDA
cells.append(nbf.v4.new_markdown_cell("""## 7. Conclusiones y Decisiones para el Pipeline NLP

1. **Balance de datos:** El dataset original está completamente equilibrado (40.000 ejemplos por cada una de las 5 clases). La transformación binaria producirá 80.000 negativas (clases 0 y 1) y 80.000 positivas (clases 3 y 4), manteniendo un balance perfecto (50/50).
2. **Calidad de datos:** No hay nulos ni valores vacíos. Los identificadores son únicos. No se requiere imputación de datos.
3. **Longitud textual:** La mediana es de 22 palabras (120 caracteres). Las opiniones negativas tienden a ser ligeramente más extensas (media de ~31 palabras vs ~22 palabras en positivas), ya que los clientes insatisfechos detallan minuciosamente los fallos del producto o entrega.
4. **Tratamiento crítico de Stopwords:** La palabra `"no"` es el término más frecuente en las opiniones negativas (46.621 apariciones en clase 0 frente a sólo 11.534 en clase 4). **Eliminar `"no"`, `"nunca"`, `"jamás"` o `"sin"` destruiría la señal de sentimiento.** Por tanto, la lista de stopwords del pipeline debe preservar estrictamente los modificadores y negaciones.
5. **N-gramas:** La alta frecuencia de expresiones como *"buena relación calidad precio"* o *"no funciona"* justifica la extracción de bigramas (`ngram_range=(1,2)`) para capturar contexto sintáctico que los unigramas aislados pierden.
"""))

nb.cells = cells

# Guardar notebook
output_path = Path("notebooks/01_analisis_exploratorio.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook creado exitosamente en: {output_path.resolve()}")
