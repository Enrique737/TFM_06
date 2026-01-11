# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="BD0srljZjAzA"
# # **ML CHECKLIST**

# %% [markdown] id="6Jzag3nCgdKX"
# # **1. Análisis univariante de los datos**
#
#

# %% [markdown] id="16yT27I_giLF"
# **1.1 Tamaño del Dataset (tamaño en memoria, número de registros y atributos, etc.)**

# %% colab={"base_uri": "https://localhost:8080/"} id="45zCFaXkm1b9" outputId="e419f5f7-28a0-4cf8-e2d2-9c932f8edd44"

# PASO 1: Cargar las herramientas necesarias
import pandas as pd
import matplotlib.pyplot as plt

print("=== INICIO DEL ANÁLISIS ===")
print()

# %% colab={"base_uri": "https://localhost:8080/"} id="Z-bSUm8Cm7Ac" outputId="704128b7-0bb5-4f80-9d35-6dd1f6cf3809"
# PASO 2: Cargar los datos
print("1. CARGANDO DATOS...")
datos = pd.read_csv("https://www.dropbox.com/s/sxl5bpi2620p496/sample_mmp.csv?dl=1")
print(f"   ✓ Datos cargados: {len(datos)} filas, {len(datos.columns)} columnas")
print()

# %% colab={"base_uri": "https://localhost:8080/"} id="3NdkOi9PnETT" outputId="2eb9c0c1-46b1-4569-e962-58aa8b013695"
datos.shape

# %% [markdown] id="H6dU8EN4gngt"
# **1.2 Visualización directa de los datos (head)**

# %% colab={"base_uri": "https://localhost:8080/"} id="v1xcz26wnOmp" outputId="b7d6cd03-bf58-4394-9c73-a69a6fa7f98e"
# PASO 3: Ver primeras filas
print("2. PRIMERAS 5 FILAS:")
print(datos.head())
print()

# %% colab={"base_uri": "https://localhost:8080/", "height": 461} id="fc0oGmEoBnmq" outputId="502321a7-1d52-49ff-9432-ca13ae0aa697"
datos

# %% [markdown] id="fDz_aA7HgvNL"
# **1.3 Tipo de atributos disponibles (numéricos, categóricos)**

# %% colab={"base_uri": "https://localhost:8080/"} id="pMCgXneBnVuP" outputId="f2a10fa2-16c7-4784-bfb0-bbd8d539f750"
# PASO 4: Tipo de datos
datos.info()

# %% colab={"base_uri": "https://localhost:8080/"} id="gHjB3bzZnwn0" outputId="cf587d04-d47a-4986-dd14-3443f25e3216"
# Contar tipos
numericas = len(datos.select_dtypes(include=['int64', 'float64']).columns)
categoricas = len(datos.select_dtypes(include=['object']).columns)
print(f"   Total numéricas: {numericas}")
print(f"   Total categóricas (texto): {categoricas}")
print()

# %% [markdown] id="UGG1LimKgz-D"
# **1.4 Estadísticos descriptivos (valores medios, dispersión, percentiles, etc.)**

# %% colab={"base_uri": "https://localhost:8080/"} id="aFzXoKahn94x" outputId="d87b97cf-816a-4eb0-a279-30cc625c5cd4"
# PASO 5: Estadísticas básicas
print("4. ESTADÍSTICAS BÁSICAS:")
print("-" * 40)
print("Estadísticas de columnas numéricas:")
print(datos.describe())
print()

# %% [markdown] id="aiQyPhK7g3XT"
# **1.5 Número de valores nulos**

# %% colab={"base_uri": "https://localhost:8080/"} id="us-zVqbPrxyq" outputId="cce351d8-0771-40e6-9527-2adc33d4fb99"
# PASO 6: Valores faltantes
print("5. VALORES FALTANTES:")
nulos = datos.isnull().sum()
columnas_con_nulos = nulos[nulos > 0]

if len(columnas_con_nulos) > 0:
    print("Columnas con valores faltantes:")
    for col, cant in columnas_con_nulos.items():
        porc = (cant / len(datos)) * 100
        print(f"   {col}: {cant} faltantes ({porc:.1f}%)")
else:
    print("   ✓ No hay valores faltantes")
print()

# %% id="uXBeFUG5pW4v"
del datos['DefaultBrowsersIdentifier']

# %% id="1scc3IavqFZ_"
del datos['Census_ProcessorClass']

# %% [markdown] id="z6FelqH0g6p6"
# **1.6 Distribución / rango de valores del target (sólo en clasificación supervisada)**

# %% colab={"base_uri": "https://localhost:8080/"} id="KqHjGXVKr8DH" outputId="920594d7-0191-4c90-fddc-b8b55f7c2795"
# PASO 7: Variable que queremos predecir
print("6. VARIABLE A PREDECIR (HasDetections):")
print("-" * 40)
conteo = datos['HasDetections'].value_counts()
print(f"   Valor 0 (NO tiene malware): {conteo[0]} máquinas")
print(f"   Valor 1 (SÍ tiene malware): {conteo[1]} máquinas")

porcentaje = (conteo[1] / len(datos)) * 100
print(f"   → {porcentaje:.1f}% de las máquinas tienen malware")
print()





# %% colab={"base_uri": "https://localhost:8080/", "height": 426} id="A8F2PlfxsB3_" outputId="7347768f-c30e-43c1-b125-f3a1b1d4ec49"
# Gráfico simple
print("📊 GRÁFICO 1: Distribución de malware")
plt.figure(figsize=(6, 4))
plt.bar(['Sin malware', 'Con malware'], conteo.values, color=['green', 'red'])
plt.title('¿Cuántas máquinas tienen malware?')
plt.ylabel('Número de máquinas')
plt.show()
print()

# %% [markdown] id="ChMMld_ig9qZ"
# **1.7 Identificación de outliers**

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="WHjVDm90qwr5" outputId="739d3331-dd9b-46e4-f30d-bb8fd5c5e277"
"""
=================================================================
IDENTIFICACIÓN DE OUTLIERS Y DATOS ERRÓNEOS
=================================================================
Estrategia: Analizar solo las columnas más importantes
=================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
# %matplotlib inline

print("🔍 IDENTIFICANDO OUTLIERS Y DATOS ERRÓNEOS")
print("=" * 60)

# 1. CARGAR DATOS
datos = pd.read_csv("https://www.dropbox.com/s/sxl5bpi2620p496/sample_mmp.csv?dl=1")

# 2. SELECCIONAR COLUMNAS IMPORTANTES (por intuición)
print("\n📋 SELECCIÓN DE COLUMNAS IMPORTANTES:")
print("-" * 40)

# Columnas que SEGURO son importantes para malware
columnas_importantes = [
    # Configuración seguridad
    'IsProtected',                    # ¿Está protegido?
    'Firewall',                       # Firewall activado
    'AVProductsInstalled',            # Número de antivirus
    'AVProductStatesIdentifier',      # Configuración antivirus

    # Hardware (puede afectar vulnerabilidad)
    'Census_TotalPhysicalRAM',        # Memoria RAM
    'Census_PrimaryDiskTotalCapacity', # Tamaño disco
    'Census_ProcessorCoreCount',      # Núcleos CPU

    # Sistema operativo
    'Census_OSVersion',               # Versión Windows
    'Census_OSBuildRevision',         # Build específico

    # Target
    'HasDetections'                   # Lo que queremos predecir
]

# Filtrar solo las que existen en el dataset
columnas_existentes = [col for col in columnas_importantes if col in datos.columns]
print(f"Se analizarán {len(columnas_existentes)} columnas importantes:")
for i, col in enumerate(columnas_existentes, 1):
    print(f"  {i:2}. {col}")

# ============================================================================
# PARTE 1: OUTLIERS EN VARIABLES NUMÉRICAS (con Boxplots)
# ============================================================================
print("\n" + "=" * 60)
print("📊 PARTE 1: OUTLIERS EN VARIABLES NUMÉRICAS")
print("=" * 60)

# Separar variables numéricas de nuestras importantes
columnas_numericas = []
for col in columnas_existentes:
    if datos[col].dtype in ['int64', 'float64'] and col != 'HasDetections':
        columnas_numericas.append(col)

print(f"\n🔢 Variables numéricas a analizar ({len(columnas_numericas)}):")
for col in columnas_numericas:
    print(f"  • {col}")

# Función para analizar cada variable numérica
def analizar_variable_numerica(df, columna):
    """
    Analiza una variable numérica y detecta outliers
    """
    print(f"\n📈 ANALIZANDO: {columna}")
    print("-" * 40)

    # Estadísticas básicas
    print(f"Tipo: {df[columna].dtype}")
    print(f"Valores únicos: {df[columna].nunique()}")
    print(f"Valores nulos: {df[columna].isnull().sum()} ({df[columna].isnull().sum()/len(df)*100:.1f}%)")

    # Estadísticas descriptivas
    stats = df[columna].describe()
    print(f"\n📊 Estadísticas:")
    print(f"  Mínimo: {stats['min']:.2f}")
    print(f"  Máximo: {stats['max']:.2f}")
    print(f"  Media: {stats['mean']:.2f}")
    print(f"  Mediana: {stats['50%']:.2f}")
    print(f"  Desviación estándar: {stats['std']:.2f}")

    # Detectar outliers con método IQR
    Q1 = stats['25%']
    Q3 = stats['75%']
    IQR = Q3 - Q1

    if IQR > 0:  # Evitar división por cero
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        # Contar outliers
        outliers = df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)]
        num_outliers = len(outliers)
        porcentaje = (num_outliers / len(df)) * 100

        print(f"\n🔍 OUTLIERS (método IQR):")
        print(f"  Límite inferior: {limite_inferior:.2f}")
        print(f"  Límite superior: {limite_superior:.2f}")
        print(f"  Outliers encontrados: {num_outliers} ({porcentaje:.1f}%)")

        # Verificar valores prohibidos
        if stats['min'] < 0 and columna in ['Census_TotalPhysicalRAM', 'Census_PrimaryDiskTotalCapacity']:
            print(f"\n⚠️  ¡VALORES PROHIBIDOS DETECTADOS!")
            print(f"  {columna} no puede tener valores negativos")
            valores_negativos = df[df[columna] < 0][columna].count()
            print(f"  Valores negativos: {valores_negativos}")

        # Recomendación
        print(f"\n💡 RECOMENDACIÓN:")
        if num_outliers == 0:
            print("  No hay outliers - mantener datos como están")
        elif porcentaje < 5:
            print("  Pocos outliers (<5%) - considerar eliminarlos")
        elif porcentaje < 15:
            print("  Algunos outliers (5-15%) - revisar si son errores")
        else:
            print("  Muchos outliers (>15%) - probablemente distribución natural")

    return Q1, Q3, IQR

# Crear boxplots para cada variable numérica
print("\n📦 CREANDO BOXPLOTS PARA DETECTAR OUTLIERS VISUALMENTE:")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, col in enumerate(columnas_numericas[:6]):  # Mostrar solo primeras 6
    # Boxplot
    bp = axes[idx].boxplot(datos[col].dropna(), patch_artist=True, vert=True)

    # Colorear
    bp['boxes'][0].set_facecolor('lightblue')
    bp['medians'][0].set_color('red')

    # Calcular outliers
    Q1 = datos[col].quantile(0.25)
    Q3 = datos[col].quantile(0.75)
    IQR = Q3 - Q1

    # Añadir información
    axes[idx].set_title(f'{col}\nIQR: {IQR:.0f}', fontsize=10)
    axes[idx].set_ylabel('Valor')

    # Analizar estadísticas
    analizar_variable_numerica(datos, col)

# Ocultar ejes vacíos si hay menos de 6 columnas
for idx in range(len(columnas_numericas[:6]), 6):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.show()

# ============================================================================
# PARTE 2: DATOS ERRÓNEOS EN VARIABLES CATEGÓRICAS
# ============================================================================
print("\n" + "=" * 60)
print("🔤 PARTE 2: DATOS ERRÓNEOS EN VARIABLES CATEGÓRICAS")
print("=" * 60)

# Separar variables categóricas de nuestras importantes
columnas_categoricas = []
for col in columnas_existentes:
    if datos[col].dtype == 'object':
        columnas_categoricas.append(col)

print(f"\n🔤 Variables categóricas a analizar ({len(columnas_categoricas)}):")
for col in columnas_categoricas:
    print(f"  • {col}")

def analizar_variable_categorica(df, columna):
    """
    Analiza una variable categórica y detecta problemas
    """
    print(f"\n📊 ANALIZANDO: {columna}")
    print("-" * 40)

    # Estadísticas básicas
    valores_unicos = df[columna].nunique()
    print(f"Valores únicos: {valores_unicos}")
    print(f"Valores nulos: {df[columna].isnull().sum()} ({df[columna].isnull().sum()/len(df)*100:.1f}%)")

    # Distribución de valores
    conteo = df[columna].value_counts()

    print(f"\n📈 DISTRIBUCIÓN (Top 5 valores):")
    for i, (valor, frecuencia) in enumerate(conteo.head(5).items(), 1):
        porcentaje = (frecuencia / len(df)) * 100
        print(f"  {i}. '{valor}': {frecuencia} veces ({porcentaje:.1f}%)")

    # Detectar categorías raras (<1% de frecuencia)
    categorias_raras = conteo[conteo / len(df) < 0.01]
    num_categorias_raras = len(categorias_raras)

    print(f"\n🔍 CATEGORÍAS RARAS (<1% frecuencia):")
    print(f"  Número de categorías raras: {num_categorias_raras}")

    if num_categorias_raras > 0:
        print("  Ejemplos de categorías raras:")
        for valor, frecuencia in categorias_raras.head(3).items():
            print(f"    • '{valor}' ({frecuencia} veces)")

    # Detectar valores sospechosos
    valores_sospechosos = []
    if df[columna].dtype == 'object':
        for valor in df[columna].dropna().unique():
            if isinstance(valor, str):
                valor_str = str(valor).lower()
                if ('unknown' in valor_str or
                    'null' in valor_str or
                    'nan' in valor_str or
                    'none' in valor_str or
                    'not' in valor_str):
                    valores_sospechosos.append(valor)

    print(f"\n⚠️  VALORES SOSPECHOSOS:")
    if valores_sospechosos:
        print(f"  Encontrados: {len(valores_sospechosos)}")
        for valor in valores_sospechosos[:3]:  # Mostrar solo 3
            print(f"    • '{valor}'")
    else:
        print("  No se encontraron valores sospechosos evidentes")

    # Recomendación
    print(f"\n💡 RECOMENDACIÓN:")
    if valores_unicos > 20:
        print("  Demasiados valores únicos - considerar agrupar en 'Otros'")
    elif num_categorias_raras > 5:
        print("  Muchas categorías raras - agrupar las menos frecuentes")
    else:
        print("  Distribución aceptable - mantener como está")

# Crear gráficos para variables categóricas
print("\n📊 GRÁFICOS DE VARIABLES CATEGÓRICAS:")
num_categoricas = min(4, len(columnas_categoricas))  # Máximo 4 gráficos
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, col in enumerate(columnas_categoricas[:num_categoricas]):
    # Tomar top 10 categorías para no saturar gráfico
    top_categorias = datos[col].value_counts().head(10)

    # Gráfico de barras
    bars = axes[idx].bar(range(len(top_categorias)), top_categorias.values,
                        color=plt.cm.Set3(range(len(top_categorias))))

    # Configurar
    axes[idx].set_title(f'{col}\n({datos[col].nunique()} valores únicos)', fontsize=10)
    axes[idx].set_xlabel('Categorías')
    axes[idx].set_ylabel('Frecuencia')

    # Rotar etiquetas
    axes[idx].set_xticks(range(len(top_categorias)))
    axes[idx].set_xticklabels([str(x)[:10] + '...' if len(str(x)) > 10 else str(x)
                              for x in top_categorias.index], rotation=45, ha='right')

    # Analizar la variable
    analizar_variable_categorica(datos, col)

# Ocultar ejes vacíos
for idx in range(num_categoricas, 4):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.show()

# ============================================================================
# PARTE 3: RELACIÓN CON EL TARGET (HasDetections)
# ============================================================================
print("\n" + "=" * 60)
print("🎯 PARTE 3: RELACIÓN CON HASDETECTIONS (Target)")
print("=" * 60)

print("\n📊 ¿CÓMO SE RELACIONAN LAS VARIABLES CON EL MALWARE?")
print("-" * 50)

# Para variables numéricas: correlación
print("\n🔢 CORRELACIÓN DE VARIABLES NUMÉRICAS CON HASDETECTIONS:")
for col in columnas_numericas:
    if col in datos.columns and 'HasDetections' in datos.columns:
        correlacion = datos[col].corr(datos['HasDetections'])
        print(f"  {col:<30}: {correlacion:+.4f}")

# Para variables categóricas: tasa de malware por categoría
print("\n🔤 TASA DE MALWARE POR CATEGORÍA (variables más importantes):")

# Analizar solo 2 categóricas clave
categoricas_clave = ['IsProtected', 'Firewall'] if 'IsProtected' in columnas_categoricas else columnas_categoricas[:2]

for col in categoricas_clave:
    print(f"\n  📍 Variable: {col}")
    print("  " + "-" * 30)

    # Calcular tasa de malware por categoría
    for categoria in datos[col].dropna().unique()[:5]:  # Top 5 categorías
        subset = datos[datos[col] == categoria]
        if len(subset) > 0:
            tasa_malware = subset['HasDetections'].mean() * 100
            print(f"    '{categoria}': {tasa_malware:.1f}% tiene malware")

# ============================================================================
# RESUMEN Y RECOMENDACIONES FINALES
# ============================================================================
print("\n" + "=" * 60)
print("📋 RESUMEN Y RECOMENDACIONES FINALES")
print("=" * 60)

print("\n🎯 COLUMNAS MÁS IMPORTANTES IDENTIFICADAS:")
print("-" * 40)

# Clasificar por importancia
columnas_problematicas = []
columnas_ok = []

for col in columnas_existentes:
    if col in columnas_numericas:
        # Para numéricas: revisar outliers
        Q1 = datos[col].quantile(0.25)
        Q3 = datos[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:
            outliers = datos[(datos[col] < Q1 - 1.5*IQR) | (datos[col] > Q3 + 1.5*IQR)]
            if len(outliers) / len(datos) > 0.10:  # >10% outliers
                columnas_problematicas.append((col, f"{len(outliers)/len(datos)*100:.1f}% outliers"))
            else:
                columnas_ok.append(col)
    elif col in columnas_categoricas:
        # Para categóricas: revisar categorías raras
        if datos[col].nunique() > 20:
            columnas_problematicas.append((col, f"{datos[col].nunique()} valores únicos"))
        else:
            columnas_ok.append(col)

print("\n⚠️  COLUMNAS PROBLEMÁTICAS (necesitan tratamiento):")
if columnas_problematicas:
    for col, problema in columnas_problematicas:
        print(f"  • {col}: {problema}")
else:
    print("  ¡No hay columnas problemáticas graves!")

print("\n✅ COLUMNAS EN BUEN ESTADO:")
for col in columnas_ok:
    print(f"  • {col}")

print("\n💡 ACCIONES RECOMENDADAS:")
print("1. Para outliers numéricos (>10%):")
print("   • Considerar transformación logarítmica")
print("   • O truncar en percentiles 1% y 99%")
print("\n2. Para categóricas con muchos valores:")
print("   • Agrupar categorías poco frecuentes en 'Otros'")
print("   • Usar target encoding en lugar de one-hot")
print("\n3. Para tu modelo:")
print("   • Incluir todas estas columnas importantes")
print("   • Tratar outliers antes de entrenar")
print("   • Documentar decisiones en el notebook")

print("\n" + "=" * 60)
print("✅ ANÁLISIS COMPLETADO")
print("=" * 60)

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="3d2CEIZlsk1R" outputId="7b7c03d9-f58c-4b37-8fd8-ff274329982f"
datos['Census_OSVersion'].value_counts()

# %% id="TYhdqzb_uoha"
# Identificar valores con menos de 100 ocurrencias
conteos = datos['Census_OSVersion'].value_counts()
valores_a_reemplazar = conteos[conteos < 1000].index.tolist()

# Reemplazar esos valores por 'Otros'
datos['Census_OSVersion'] = datos['Census_OSVersion'].replace(valores_a_reemplazar, 'Otros')

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="TqgJnwVywpzc" outputId="575f2d33-0df3-4082-eae4-fb86e4d94870"
datos['Census_OSVersion'].value_counts()

# %% [markdown] id="P8JY_ktIhBuh"
# **1.8 Identificación de datos erróneos**

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="MxkUk5hUJGec" outputId="18269d91-707b-428d-e377-a9ffb3b76778"
"""
=================================================================
1.8 IDENTIFICACIÓN DE DATOS ERRÓNEOS
=================================================================
Estrategia: Comprender qué valores puede tomar cada columna
y buscar valores imposibles o sospechosos.
=================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("🔍 1.8 IDENTIFICACIÓN DE DATOS ERRÓNEOS")
print("=" * 60)

# Cargar datos
datos = pd.read_csv("https://www.dropbox.com/s/sxl5bpi2620p496/sample_mmp.csv?dl=1")

print("\n📋 ESTRATEGIA DEL PROFESOR:")
print("-" * 40)
print("1. Entender qué información lleva cada columna")
print("2. Conocer valores posibles que puede tomar")
print("3. Buscar valores prohibidos o imposibles")
print("4. Identificar valores extremos (outliers problemáticos)")
print()

# ============================================================================
# PASO 1: SELECCIONAR COLUMNAS IMPORTANTES (por intuición y correlación)
# ============================================================================
print("🎯 PASO 1: SELECCIÓN DE COLUMNAS CLAVE")
print("-" * 40)

# Basado en lo que dijo el profesor y correlación encontrada
columnas_prioritarias = [
    # Seguridad (muy importantes para malware)
    'IsProtected',                # ¿Está protegido? → True/False
    'Firewall',                   # Firewall → ON/OFF
    'AVProductsInstalled',        # Nº antivirus → 0, 1, 2, 3...
    'AVProductStatesIdentifier',  # ID configuración antivirus

    # Hardware (valores deben tener sentido físico)
    'Census_TotalPhysicalRAM',           # RAM en MB → >0
    'Census_PrimaryDiskTotalCapacity',   # Disco en MB → >0
    'Census_ProcessorCoreCount',         # Núcleos CPU → 1, 2, 4, 8...
    'Census_InternalPrimaryDiagonalDisplaySizeInInches',  # Pantalla → 10-30"

    # Sistema
    'Census_OSVersion',           # Versión Windows → 6.1, 6.2, 6.3, 10.0
    'Census_OSBuildRevision',     # Build específico
    'OsBuild',                    # Número de build
]

# Filtrar las que existen
columnas_existentes = [col for col in columnas_prioritarias if col in datos.columns]
print(f"Analizando {len(columnas_existentes)} columnas prioritarias:")
for i, col in enumerate(columnas_existentes[:10], 1):
    print(f"  {i:2}. {col}")
if len(columnas_existentes) > 10:
    print(f"  ... y {len(columnas_existentes)-10} más")

# ============================================================================
# PASO 2: FUNCIÓN PARA ANALIZAR VALORES POSIBLES
# ============================================================================
def analizar_valores_posibles(df, columna):
    """
    Analiza si los valores de una columna tienen sentido
    según lo que debería contener
    """
    print(f"\n📊 COLUMNA: {columna}")
    print("-" * 40)

    # Información básica
    print(f"Tipo de dato: {df[columna].dtype}")
    print(f"Valores únicos: {df[columna].nunique()}")
    print(f"Valores nulos: {df[columna].isnull().sum()} ({df[columna].isnull().sum()/len(df)*100:.1f}%)")

    # ============================================
    # ANÁLISIS ESPECÍFICO POR TIPO DE COLUMNA
    # ============================================

    if columna in ['Census_TotalPhysicalRAM', 'Census_PrimaryDiskTotalCapacity']:
        print(f"\n🔧 ANÁLISIS ESPECÍFICO: {columna}")
        print("Valores esperados: Números positivos (MB/GB)")

        # Buscar valores imposibles
        valores_negativos = df[df[columna] < 0][columna]
        valores_cero = df[df[columna] == 0][columna]

        if len(valores_negativos) > 0:
            print(f"⚠️  VALORES PROHIBIDOS: {len(valores_negativos)} valores negativos")
            print(f"   Ejemplos: {valores_negativos.iloc[:3].tolist()}")
        else:
            print("✅ No hay valores negativos")

        if len(valores_cero) > 0:
            print(f"⚠️  VALORES SOSPECHOSOS: {len(valores_cero)} valores igual a 0")
            print(f"   (¿Máquina sin RAM o sin disco?)")

    elif columna == 'Census_ProcessorCoreCount':
        print(f"\n🔧 ANÁLISIS ESPECÍFICO: {columna}")
        print("Valores esperados: 1, 2, 4, 6, 8, 12, 16, 32...")

        valores = df[columna].dropna().unique()
        valores_raros = [v for v in valores if v not in [1, 2, 4, 6, 8, 12, 16, 32, 64]]

        if valores_raros:
            print(f"⚠️  VALORES INUSUALES: {valores_raros}")
            print("   (Número de núcleos poco común)")

    elif columna == 'Census_InternalPrimaryDiagonalDisplaySizeInInches':
        print(f"\n🔧 ANÁLISIS ESPECÍFICO: {columna}")
        print("Valores esperados: Entre 10 y 30 pulgadas (para PCs)")

        valores_fuera_rango = df[(df[columna] < 10) | (df[columna] > 30)][columna]

        if len(valores_fuera_rango) > 0:
            print(f"⚠️  VALORES FUERA DE RANGO: {len(valores_fuera_rango)} valores")
            print(f"   Rango encontrado: {df[columna].min():.1f} a {df[columna].max():.1f}")

    elif columna == 'AVProductsInstalled':
        print(f"\n🔧 ANÁLISIS ESPECÍFICO: {columna}")
        print("Valores esperados: 0, 1, 2, 3, 4...")

        # ¿Valores negativos o muy altos?
        valores_negativos = df[df[columna] < 0][columna]
        valores_muy_altos = df[df[columna] > 10][columna]

        if len(valores_negativos) > 0:
            print(f"⚠️  VALORES PROHIBIDOS: {len(valores_negativos)} antivirus negativos")

        if len(valores_muy_altos) > 0:
            print(f"⚠️  VALORES SOSPECHOSOS: {len(valores_muy_altos)} con >10 antivirus")
            print(f"   (¿Realista tener 10+ antivirus instalados?)")

    elif columna in ['IsProtected', 'Firewall']:
        print(f"\n🔧 ANÁLISIS ESPECÍFICO: {columna}")
        print("Valores esperados: True/False, ON/OFF, 1/0, etc.")

        valores_unicos = df[columna].dropna().unique()
        print(f"Valores encontrados: {valores_unicos}")

        # Buscar valores sospechosos
        if df[columna].dtype == 'object':
            valores_texto = [str(v).lower() for v in valores_unicos if isinstance(v, str)]
            sospechosos = [v for v in valores_texto if any(palabra in v
                          for palabra in ['unknown', 'null', 'nan', 'none', 'not'])]

            if sospechosos:
                print(f"⚠️  VALORES SOSPECHOSOS: {sospechosos}")

    # ============================================
    # ESTADÍSTICAS GENERALES PARA TODAS
    # ============================================
    print(f"\n📈 ESTADÍSTICAS GENERALES:")
    stats = df[columna].describe()

    if df[columna].dtype in ['int64', 'float64']:
        print(f"  Mínimo: {stats['min']:.2f}")
        print(f"  Máximo: {stats['max']:.2f}")
        print(f"  Media: {stats['mean']:.2f}")
        print(f"  Mediana: {stats['50%']:.2f}")

        # Detección de outliers extremos
        Q1 = stats['25%']
        Q3 = stats['75%']
        IQR = Q3 - Q1

        if IQR > 0:
            limite_inferior = Q1 - 3 * IQR  # Límite más estricto
            limite_superior = Q3 + 3 * IQR

            outliers_extremos = df[(df[columna] < limite_inferior) |
                                  (df[columna] > limite_superior)]

            if len(outliers_extremos) > 0:
                print(f"⚠️  OUTLIERS EXTREMOS: {len(outliers_extremos)} valores")
                print(f"   (Fuera de 3*IQR: [{limite_inferior:.2f}, {limite_superior:.2f}])")

    else:  # Columnas categóricas
        print(f"\n📊 DISTRIBUCIÓN (Top 5):")
        top_5 = df[columna].value_counts().head(5)
        for valor, frecuencia in top_5.items():
            porcentaje = (frecuencia / len(df)) * 100
            print(f"  '{valor}': {frecuencia:,} ({porcentaje:.1f}%)")

        # Categorías muy raras
        todas_categorias = df[columna].value_counts()
        categorias_raras = todas_categorias[todas_categorias / len(df) < 0.001]  # <0.1%

        if len(categorias_raras) > 0:
            print(f"⚠️  CATEGORÍAS MUY RARAS (<0.1%): {len(categorias_raras)}")
            print(f"   Ejemplos: {categorias_raras.index[:3].tolist()}")

    return len(df[columna].isnull())

# ============================================================================
# PASO 3: APLICAR ANÁLISIS A COLUMNAS PRIORITARIAS
# ============================================================================
print("\n" + "=" * 60)
print("🔍 APLICANDO ANÁLISIS DE DATOS ERRÓNEOS")
print("=" * 60)

problemas_encontrados = []

for idx, col in enumerate(columnas_existentes[:8]):  # Analizar solo primeras 8
    print(f"\n{'='*50}")
    print(f"ANÁLISIS {idx+1}/{min(8, len(columnas_existentes))}")

    try:
        n_nulos = analizar_valores_posibles(datos, col)

        # Verificar si hay problemas graves
        if n_nulos / len(datos) > 0.3:  # >30% nulos
            problemas_encontrados.append((col, f"{n_nulos/len(datos)*100:.1f}% nulos"))

    except Exception as e:
        print(f"⚠️  Error analizando {col}: {e}")
        problemas_encontrados.append((col, f"Error: {str(e)[:50]}..."))

# ============================================================================
# PASO 4: ANÁLISIS VISUAL DE VARIABLES NUMÉRICAS CRÍTICAS
# ============================================================================
print("\n" + "=" * 60)
print("📊 ANÁLISIS VISUAL DE VARIABLES CRÍTICAS")
print("=" * 60)

# Seleccionar 4 variables numéricas críticas
variables_criticas = ['Census_TotalPhysicalRAM', 'Census_PrimaryDiskTotalCapacity',
                      'Census_ProcessorCoreCount', 'AVProductsInstalled']

# Filtrar las que existen
variables_a_graficar = [v for v in variables_criticas if v in datos.columns]

if variables_a_graficar:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, col in enumerate(variables_a_graficar[:4]):
        ax = axes[idx]

        # Histograma para ver distribución
        valores = datos[col].dropna()

        # Excluir outliers extremos para mejor visualización
        Q1 = valores.quantile(0.25)
        Q3 = valores.quantile(0.75)
        IQR = Q3 - Q1

        if IQR > 0:
            limite_inferior = Q1 - 3 * IQR
            limite_superior = Q3 + 3 * IQR
            valores_filtrados = valores[(valores >= limite_inferior) & (valores <= limite_superior)]
        else:
            valores_filtrados = valores

        # Crear histograma
        ax.hist(valores_filtrados, bins=30, edgecolor='black', alpha=0.7)
        ax.set_title(f'Distribución de {col}')
        ax.set_xlabel('Valor')
        ax.set_ylabel('Frecuencia')

        # Marcar valores sospechosos
        if col in ['Census_TotalPhysicalRAM', 'Census_PrimaryDiskTotalCapacity']:
            # Marcar área de valores negativos (no debería haber)
            ax.axvspan(-1000, 0, color='red', alpha=0.2, label='Valores imposibles')
            ax.legend()

        # Añadir estadísticas
        stats_text = f"Mín: {valores.min():.0f}\nMáx: {valores.max():.0f}\nMed: {valores.median():.0f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Ocultar ejes no usados
    for idx in range(len(variables_a_graficar[:4]), 4):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()

# ============================================================================
# PASO 5: BÚSQUEDA SISTEMÁTICA DE ERRORES COMUNES
# ============================================================================
print("\n" + "=" * 60)
print("🕵️ BÚSQUEDA SISTEMÁTICA DE ERRORES")
print("=" * 60)

print("\n🔍 BUSCANDO PATRONES COMUNES DE ERROR:")

# 1. Valores negativos donde no deberían
print("\n1. VALORES NEGATIVOS EN VARIABLES FÍSICAS:")
variables_fisicas = ['Census_TotalPhysicalRAM', 'Census_PrimaryDiskTotalCapacity',
                     'Census_ProcessorCoreCount']

for var in variables_fisicas:
    if var in datos.columns:
        negativos = datos[datos[var] < 0][var].count()
        if negativos > 0:
            print(f"  ⚠️  {var}: {negativos} valores negativos")

# 2. Valores cero problemáticos
print("\n2. VALORES CERO PROBLEMÁTICOS:")
if 'Census_TotalPhysicalRAM' in datos.columns:
    ceros_ram = datos[datos['Census_TotalPhysicalRAM'] == 0]['Census_TotalPhysicalRAM'].count()
    if ceros_ram > 0:
        print(f"  ⚠️  Census_TotalPhysicalRAM: {ceros_ram} máquinas con 0 RAM")

# 3. Valores de texto en columnas que deberían ser booleanas
print("\n3. VALORES DE TEXTO EN COLUMNAS BOOLEANAS:")
columnas_booleanas = ['IsProtected', 'Firewall']

for col in columnas_booleanas:
    if col in datos.columns and datos[col].dtype == 'object':
        valores_unicos = datos[col].dropna().unique()
        # Verificar si hay algo que no sea True/False, 1/0, ON/OFF
        valores_extraños = [v for v in valores_unicos
                           if str(v).lower() not in ['true', 'false', '1', '0', 'on', 'off', 'yes', 'no']]
        if valores_extraños:
            print(f"  ⚠️  {col}: Valores extraños encontrados: {valores_extraños[:3]}")

# 4. Valores fuera de rango lógico
print("\n4. VALORES FUERA DE RANGO LÓGICO:")
if 'Census_ProcessorCoreCount' in datos.columns:
    valores_raros = datos[~datos['Census_ProcessorCoreCount'].isin([1, 2, 4, 6, 8, 12, 16, 32])]
    valores_raros_count = valores_raros['Census_ProcessorCoreCount'].nunique()
    if valores_raros_count > 0:
        print(f"  ⚠️  Census_ProcessorCoreCount: {valores_raros_count} valores de núcleos inusuales")

# ============================================================================
# PASO 6: RESUMEN Y RECOMENDACIONES
# ============================================================================
print("\n" + "=" * 60)
print("📋 RESUMEN DE DATOS ERRÓNEOS ENCONTRADOS")
print("=" * 60)

if problemas_encontrados:
    print("\n⚠️  PROBLEMAS IDENTIFICADOS:")
    for col, problema in problemas_encontrados:
        print(f"  • {col}: {problema}")
else:
    print("\n✅ No se encontraron problemas graves de datos erróneos")

print("\n💡 RECOMENDACIONES PARA LIMPIEZA:")
print("1. Para valores negativos en RAM/Disco:")
print("   • Establecer como missing (NaN)")
print("   • O reemplazar con la mediana")
print("\n2. Para valores cero problemáticos:")
print("   • Verificar si son errores o datos reales")
print("   • Considerar eliminación si son pocos")
print("\n3. Para categorías muy raras:")
print("   • Agrupar en categoría 'Otros'")
print("\n4. Para outliers extremos:")
print("   • Investigar si son errores o casos especiales")
print("   • Considerar winsorization (limitar en percentiles)")

print("\n" + "=" * 60)
print("✅ 1.8 IDENTIFICACIÓN DE DATOS ERRÓNEOS - COMPLETADO")
print("=" * 60)

# %% [markdown] id="bhu52aL6hW5O"
# **1.9. Correlación de variables con el target**
#
#
#

# %% colab={"base_uri": "https://localhost:8080/"} id="TZcknes4PWNa" outputId="6e46cf53-aee7-4d7c-cc36-6fd4dbc47524"
"""
=================================================================
1.9 CORRELACIÓN DE VARIABLES CON EL TARGET (HasDetections)
=================================================================
Objetivo: Encontrar qué variables están más relacionadas con tener malware
=================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("🔗 1.9 CORRELACIÓN DE VARIABLES CON EL TARGET")
print("=" * 60)

# Cargar datos
datos = pd.read_csv("https://www.dropbox.com/s/sxl5bpi2620p496/sample_mmp.csv?dl=1")

# ============================================================================
# PARTE 1.9: CORRELACIÓN PARA VARIABLES NUMÉRICAS
# ============================================================================
print("\n📊 PARTE 1.9: CORRELACIÓN NUMÉRICA CON HASDETECTIONS")
print("-" * 50)

# 1. Encontrar columnas numéricas
columnas_numericas = datos.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Quitar el target de la lista para no correlacionarlo consigo mismo
if 'HasDetections' in columnas_numericas:
    columnas_numericas.remove('HasDetections')

print(f"Analizando {len(columnas_numericas)} columnas numéricas")
print("(Esto puede tardar un momento...)")

# 2. Calcular correlación de CADA columna numérica con el target
print("\n🔢 CALCULANDO CORRELACIONES...")

correlaciones = {}
for col in columnas_numericas[:30]:  # Solo primeras 30 para no saturar
    # Correlación de Pearson (mide relación lineal)
    correlacion = datos[col].corr(datos['HasDetections'])
    correlaciones[col] = correlacion

# 3. Ordenar de mayor a menor correlación (valor absoluto)
print("\n🏆 TOP 15 VARIABLES CON MAYOR CORRELACIÓN:")
print("(Ordenadas por importancia, sin importar si es + o -)")
print("-" * 60)

# Crear DataFrame con correlaciones
df_correlaciones = pd.DataFrame({
    'Variable': list(correlaciones.keys()),
    'Correlación': list(correlaciones.values()),
    'Corr_Abs': [abs(c) for c in correlaciones.values()]  # Valor absoluto
})

# Ordenar por valor absoluto (más importante)
df_correlaciones = df_correlaciones.sort_values('Corr_Abs', ascending=False)

# Mostrar top 15
for i, (_, fila) in enumerate(df_correlaciones.head(15).iterrows(), 1):
    signo = "+" if fila['Correlación'] > 0 else "-"
    print(f"{i:2}. {fila['Variable']:<35}: {signo}{abs(fila['Correlación']):.4f}")

# 4. Interpretación simple
print("\n💡 ¿QUÉ SIGNIFICA LA CORRELACIÓN?")
print("• +0.10: Aumenta variable → Aumenta probabilidad de malware")
print("• -0.10: Aumenta variable → Disminuye probabilidad de malware")
print("• Cerca de 0: Poca relación con malware")

# %% [markdown] id="mJStEVU8hYid"
# **10. Correlación de las variables con la clase (unidimensional)**

# %% id="CRri_zV1iPX-" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="d12e2a3c-762b-48f6-852f-ef9e8650ed21"
corr=datos.corr(numeric_only=True)
corr
corr.style.background_gradient(cmap='coolwarm')

# %% id="8fPgSTCZn9zH"
# Analizando la correlación que tiene la variable HasDetections con AVProductStatesIdentifier con un 0,1161;
# y la variable AVProductsInstalled con un -0,1487; podríamos contemplar que son las variables que más
# correlación tienen con HasDetections.

# En este caso, a más productos antivirus, menos malware (para el caso de AVProductsInstalled).
# Por otro lado, ciertas configuraciones antivirus tienen más riesgo (para el caso AVProductStatesIdentifier).

# La variable IsProtected con un 0.0562 de correlación nos dice que si el antivirus hizo un chequeo por última vez, tenemos más probabilidades de tener malware.
# La variable Census_ProcessorCoreCount con un 0.0561 nos dice que si tenemos más números de procesadores en el ordenador, tenemos más probabilidades de tener malware.
# La variable Census_PrimaryDiskTotalCapacity con un 0.0480 nos dice que si tenemos más espacio en el disco primario en MB, tendremos más probabilidades de tener malware.


# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="r4Q1Lwv5QrXU" outputId="bc09d567-7e7b-4a4b-f29a-f219068b7771"
# ============================================================================
# PARTE 10: CORRELACIÓN UNIDIMENSIONAL CON LA CLASE
# ============================================================================
print("\n\n" + "=" * 60)
print("🎯 PARTE 10: ANÁLISIS POR CLASE (0=Sin malware, 1=Con malware)")
print("=" * 60)

print("\n📈 COMPARANDO VARIABLES PARA MÁQUINAS CON Y SIN MALWARE")
print("-" * 50)

# Seleccionar 5 variables numéricas importantes de la parte anterior
top_5_variables = df_correlaciones.head(5)['Variable'].tolist()

print(f"\nAnalizando las 5 variables más importantes:")
for var in top_5_variables:
    print(f"  • {var}")

# Crear gráficos de comparación
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, col in enumerate(top_5_variables[:6]):
    ax = axes[idx]

    # Separar datos por clase
    datos_sin_malware = datos[datos['HasDetections'] == 0][col].dropna()
    datos_con_malware = datos[datos['HasDetections'] == 1][col].dropna()

    # Crear histogramas superpuestos
    ax.hist(datos_sin_malware, bins=30, alpha=0.5, label='SIN Malware', color='green')
    ax.hist(datos_con_malware, bins=30, alpha=0.5, label='CON Malware', color='red')

    ax.set_title(f'{col}\nCorr: {correlaciones[col]:+.3f}', fontsize=11)
    ax.set_xlabel('Valor')
    ax.set_ylabel('Frecuencia')
    ax.legend()

    # Añadir líneas de mediana
    mediana_sin = datos_sin_malware.median()
    mediana_con = datos_con_malware.median()

    ax.axvline(mediana_sin, color='darkgreen', linestyle='--', alpha=0.7, linewidth=2)
    ax.axvline(mediana_con, color='darkred', linestyle='--', alpha=0.7, linewidth=2)

    # Texto informativo
    info_text = f"Mediana:\nSin: {mediana_sin:.0f}\nCon: {mediana_con:.0f}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Ocultar eje extra si hay menos de 6 variables
for idx in range(len(top_5_variables[:6]), 6):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.show()

# ============================================================================
# ANÁLISIS DE VARIABLES CATEGÓRICAS IMPORTANTES
# ============================================================================
print("\n" + "=" * 60)
print("🔤 CORRELACIÓN CON VARIABLES CATEGÓRICAS IMPORTANTES")
print("=" * 60)

print("\n📊 TASA DE MALWARE POR CATEGORÍA (variables clave):")
print("-" * 50)

# Seleccionar algunas variables categóricas importantes
variables_categoricas = ['IsProtected', 'Firewall', 'Census_OSVersion']

for col in variables_categoricas:
    if col in datos.columns:
        print(f"\n📍 VARIABLE: {col}")
        print(f"   Valores únicos: {datos[col].nunique()}")

        # Calcular tasa de malware para cada categoría (top 5)
        valores_top = datos[col].value_counts().head(5).index

        print(f"   Tasa de malware por categoría (Top {len(valores_top)}):")
        for categoria in valores_top:
            # Filtrar datos para esta categoría
            subset = datos[datos[col] == categoria]
            if len(subset) > 0:
                tasa_malware = subset['HasDetections'].mean() * 100
                conteo = len(subset)
                print(f"     • '{categoria}': {tasa_malware:.1f}% malware ({conteo:,} máquinas)")

# ============================================================================
# RESUMEN Y CONCLUSIONES
# ============================================================================
print("\n" + "=" * 60)
print("📋 RESUMEN DE CORRELACIONES ENCONTRADAS")
print("=" * 60)

print("\n🎯 VARIABLES MÁS RELACIONADAS CON MALWARE:")

# Variables con correlación positiva (aumentan riesgo)
print("\n🔴 AUMENTAN RIESGO (correlación positiva):")
variables_positivas = df_correlaciones[df_correlaciones['Correlación'] > 0].head(3)
for _, fila in variables_positivas.iterrows():
    print(f"  • {fila['Variable']}: +{fila['Correlación']:.4f}")
    print(f"    (A mayor valor → más probabilidad de malware)")

# Variables con correlación negativa (disminuyen riesgo)
print("\n🟢 DISMINUYEN RIESGO (correlación negativa):")
variables_negativas = df_correlaciones[df_correlaciones['Correlación'] < 0].head(3)
for _, fila in variables_negativas.iterrows():
    print(f"  • {fila['Variable']}: {fila['Correlación']:.4f}")
    print(f"    (A mayor valor → menos probabilidad de malware)")

print("\n💡 INTERPRETACIÓN PARA WINDOWS DEFENDER:")
print("1. Variables con correlación positiva:")
print("   • Podrían indicar configuraciones vulnerables")
print("   • Ejemplo: Si 'AVProductStatesIdentifier' alto = más riesgo")
print("   → Acción: Recomendar cambiar configuración")

print("\n2. Variables con correlación negativa:")
print("   • Podrían indicar factores protectores")
print("   • Ejemplo: Si 'AVProductsInstalled' alto = menos riesgo")
print("   → Acción: Promover buenas prácticas")

print("\n3. Variables categóricas importantes:")
print("   • 'IsProtected' y 'Firewall' deberían tener baja tasa de malware")
print("   → Verificar si la protección realmente funciona")

print("\n" + "=" * 60)
print("✅ ANÁLISIS DE CORRELACIÓN COMPLETADO")
print("=" * 60)

# %% [markdown] id="x5Jd47Tphf58"
# **11. Visualización gráfica de las distribuciones**
#
# ✓ Numéricas: histogramas, box-plots, violin-plots, vista por deciles, etc.
#
# ✓ Categóricas: bar-charts, conteo directo, etc.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="mC9T8g8wVdWo" outputId="152e82e7-4102-4ba9-8d46-2f241c453d07"
"""
=================================================================
11. VISUALIZACIÓN GRÁFICA DE LAS DISTRIBUCIONES
=================================================================
Objetivo: Mostrar cómo se distribuyen los datos de forma visual
=================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("📊 11. VISUALIZACIÓN GRÁFICA DE LAS DISTRIBUCIONES")
print("=" * 60)

# Cargar datos
datos = pd.read_csv("https://www.dropbox.com/s/sxl5bpi2620p496/sample_mmp.csv?dl=1")

# ============================================================================
# PARTE A: VARIABLES NUMÉRICAS
# ============================================================================
print("\n📈 PARTE A: VARIABLES NUMÉRICAS")
print("-" * 50)

# Seleccionar 4 variables numéricas importantes
variables_numericas = ['Census_TotalPhysicalRAM', 'AVProductsInstalled',
                       'Census_ProcessorCoreCount', 'OsBuild']

# Filtrar las que existen
vars_a_mostrar = [v for v in variables_numericas if v in datos.columns]

print(f"Mostrando gráficos para {len(vars_a_mostrar)} variables numéricas:")

fig = plt.figure(figsize=(15, 10))

# ============================================
# 1. HISTOGRAMAS (distribución de valores)
# ============================================
print("\n📊 1. HISTOGRAMAS:")
print("   Muestra cuántas máquinas tienen cada valor")

for idx, col in enumerate(vars_a_mostrar[:4]):
    ax1 = plt.subplot(4, 4, idx*4 + 1)

    # Crear histograma
    valores = datos[col].dropna()
    ax1.hist(valores, bins=30, edgecolor='black', alpha=0.7, color='skyblue')

    ax1.set_title(f'Histograma: {col}', fontsize=10)
    ax1.set_xlabel('Valor')
    ax1.set_ylabel('Frecuencia')
    ax1.grid(True, alpha=0.3)

    # Añadir estadísticas
    stats_text = f"N: {len(valores):,}\nMed: {valores.median():.0f}"
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes,
             fontsize=8, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ============================================
# 2. BOX-PLOTS (valores atípicos y distribución)
# ============================================
print("\n📦 2. BOX-PLOTS:")
print("   Muestra valores normales y atípicos (outliers)")

for idx, col in enumerate(vars_a_mostrar[:4]):
    ax2 = plt.subplot(4, 4, idx*4 + 2)

    # Crear boxplot
    box = ax2.boxplot(datos[col].dropna(), patch_artist=True, vert=True)
    box['boxes'][0].set_facecolor('lightgreen')
    box['medians'][0].set_color('red')

    ax2.set_title(f'Box-plot: {col}', fontsize=10)
    ax2.set_ylabel('Valor')
    ax2.grid(True, alpha=0.3)

    # Contar outliers
    Q1 = datos[col].quantile(0.25)
    Q3 = datos[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = datos[(datos[col] < Q1 - 1.5*IQR) | (datos[col] > Q3 + 1.5*IQR)]

    info_text = f"Outliers: {len(outliers)}\n({len(outliers)/len(datos)*100:.1f}%)"
    ax2.text(0.95, 0.95, info_text, transform=ax2.transAxes,
             fontsize=8, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# ============================================
# 3. VIOLIN-PLOTS (distribución + densidad)
# ============================================
print("\n🎻 3. VIOLIN-PLOTS:")
print("   Combina boxplot con densidad (más información)")

for idx, col in enumerate(vars_a_mostrar[:4]):
    ax3 = plt.subplot(4, 4, idx*4 + 3)

    # Crear violin plot
    parts = ax3.violinplot(datos[col].dropna(), vert=True, showmedians=True)

    # Colorear
    for pc in parts['bodies']:
        pc.set_facecolor('orange')
        pc.set_alpha(0.6)

    ax3.set_title(f'Violin-plot: {col}', fontsize=10)
    ax3.set_ylabel('Valor')
    ax3.grid(True, alpha=0.3)

    # Información sobre forma
    skewness = datos[col].skew()
    if abs(skewness) > 1:
        forma = "Muy asimétrica"
    elif abs(skewness) > 0.5:
        forma = "Asimétrica"
    else:
        forma = "Simétrica"

    info_text = f"Forma: {forma}\nAsimetría: {skewness:.2f}"
    ax3.text(0.95, 0.95, info_text, transform=ax3.transAxes,
             fontsize=8, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ============================================
# 4. VISTA POR DECILES (percentiles)
# ============================================
print("\n📊 4. DECILES (Percentiles 10% a 90%):")
print("   Divide los datos en 10 partes iguales")

for idx, col in enumerate(vars_a_mostrar[:4]):
    ax4 = plt.subplot(4, 4, idx*4 + 4)

    # Calcular deciles (10%, 20%, ..., 90%)
    deciles = [datos[col].quantile(i/10) for i in range(1, 10)]

    # Crear gráfico de deciles
    ax4.plot(range(1, 10), deciles, 'o-', color='purple', linewidth=2)

    ax4.set_title(f'Deciles: {col}', fontsize=10)
    ax4.set_xlabel('Decil (10%, 20%, ..., 90%)')
    ax4.set_ylabel('Valor')
    ax4.grid(True, alpha=0.3)

    # Añadir valores
    for i, val in enumerate(deciles):
        if i % 2 == 0:  # Mostrar cada 2 deciles
            ax4.text(i+1, val, f'{val:.0f}', fontsize=7,
                    ha='center', va='bottom')

plt.tight_layout()
plt.show()

# ============================================================================
# PARTE B: VARIABLES CATEGÓRICAS
# ============================================================================
print("\n\n🔤 PARTE B: VARIABLES CATEGÓRICAS")
print("-" * 50)

# Seleccionar 4 variables categóricas importantes
variables_categoricas = ['IsProtected', 'Firewall', 'Census_OSSkuName', 'Census_OSEdition']

# Filtrar las que existen
cats_a_mostrar = [v for v in variables_categoricas if v in datos.columns]

print(f"Mostrando gráficos para {len(cats_a_mostrar)} variables categóricas:")

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
axes2 = axes2.flatten()

for idx, col in enumerate(cats_a_mostrar[:4]):
    ax = axes2[idx]

    # ============================================
    # 1. BAR-CHART (gráfico de barras)
    # ============================================
    print(f"\n📊 {col}: Gráfico de barras")
    print(f"   Muestra cuántas máquinas tienen cada valor")

    # Contar valores (top 10 si hay muchos)
    conteo = datos[col].value_counts()

    # Si hay muchos valores únicos, mostrar solo top 10
    if len(conteo) > 10:
        top_10 = conteo.head(10)
        otros = conteo[10:].sum()
        datos_grafico = top_10.copy()
        datos_grafico['Otros'] = otros
        titulo_extra = " (Top 10 + Otros)"
    else:
        datos_grafico = conteo
        titulo_extra = ""

    # Crear barras
    bars = ax.bar(range(len(datos_grafico)), datos_grafico.values,
                  color=plt.cm.Set3(range(len(datos_grafico))))

    ax.set_title(f'{col}{titulo_extra}', fontsize=12)
    ax.set_xlabel('Valores')
    ax.set_ylabel('Número de máquinas')

    # Configurar etiquetas en X
    etiquetas = [str(x)[:15] + '...' if len(str(x)) > 15 else str(x)
                 for x in datos_grafico.index]
    ax.set_xticks(range(len(datos_grafico)))
    ax.set_xticklabels(etiquetas, rotation=45, ha='right')

    # Añadir números encima de las barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(datos_grafico.values)*0.01,
                f'{int(height):,}', ha='center', va='bottom', fontsize=9)

    # Estadísticas
    total = datos[col].count()
    nulos = datos[col].isnull().sum()
    unicos = datos[col].nunique()

    stats_text = f"Total: {total:,}\nNulos: {nulos:,}\nÚnicos: {unicos}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ============================================
    # 2. CONTEO DIRECTO (texto adicional)
    # ============================================
    print(f"   Conteo directo de valores:")

    # Mostrar top 5 valores en consola
    top_5 = conteo.head(5)
    for i, (valor, cantidad) in enumerate(top_5.items(), 1):
        porcentaje = (cantidad / total) * 100
        print(f"     {i}. '{valor}': {cantidad:,} ({porcentaje:.1f}%)")

# Ocultar ejes no usados
for idx in range(len(cats_a_mostrar[:4]), 4):
    axes2[idx].set_visible(False)

plt.tight_layout()
plt.show()

# ============================================================================
# PARTE C: GRÁFICOS ESPECIALES PARA TU PROYECTO
# ============================================================================
print("\n\n🎨 PARTE C: GRÁFICOS ESPECIALES PARA ANÁLISIS DE MALWARE")
print("-" * 50)

print("\n📊 COMPARACIÓN CON Y SIN MALWARE:")

# Crear gráfico especial: comparación de una variable importante
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 1. Comparación de RAM
if 'Census_TotalPhysicalRAM' in datos.columns:
    # Preparar datos
    ram_sin_malware = datos[datos['HasDetections'] == 0]['Census_TotalPhysicalRAM'].dropna()
    ram_con_malware = datos[datos['HasDetections'] == 1]['Census_TotalPhysicalRAM'].dropna()

    # Boxplots comparativos
    ax1.boxplot([ram_sin_malware, ram_con_malware],
                labels=['Sin Malware', 'Con Malware'],
                patch_artist=True)

    # Colorear
    ax1.patches[0].set_facecolor('lightgreen')
    ax1.patches[1].set_facecolor('lightcoral')

    ax1.set_title('RAM: Comparación por presencia de malware')
    ax1.set_ylabel('RAM (MB)')
    ax1.grid(True, alpha=0.3)

    # Estadísticas
    mediana_sin = ram_sin_malware.median()
    mediana_con = ram_con_malware.median()
    ax1.text(0.5, 0.95, f'Mediana sin: {mediana_sin:,.0f} MB\nMediana con: {mediana_con:,.0f} MB',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 2. Comparación de número de antivirus
if 'AVProductsInstalled' in datos.columns:
    # Calcular % malware por número de antivirus
    tasas_malware = []
    valores_av = sorted(datos['AVProductsInstalled'].unique())

    for num_av in valores_av[:5]:  # Solo 0-4 antivirus
        subset = datos[datos['AVProductsInstalled'] == num_av]
        if len(subset) > 0:
            tasa = subset['HasDetections'].mean() * 100
            tasas_malware.append(tasa)
        else:
            tasas_malware.append(0)

    # Gráfico de barras
    bars = ax2.bar([str(v) for v in valores_av[:5]], tasas_malware[:5],
                   color=['red' if t > 50 else 'green' for t in tasas_malware[:5]])

    ax2.set_title('% Malware por número de antivirus instalados')
    ax2.set_xlabel('Número de productos antivirus')
    ax2.set_ylabel('% con malware')
    ax2.grid(True, alpha=0.3)

    # Añadir valores
    for bar, tasa in zip(bars, tasas_malware[:5]):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{tasa:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# ============================================================================
# RESUMEN Y EXPLICACIÓN
# ============================================================================
print("\n" + "=" * 60)
print("📋 RESUMEN DE TIPOS DE GRÁFICOS UTILIZADOS")
print("=" * 60)

print("\n🎨 PARA VARIABLES NUMÉRICAS:")
print("1. Histograma: Muestra distribución de valores")
print("2. Box-plot: Muestra valores normales y atípicos")
print("3. Violin-plot: Combina distribución y densidad")
print("4. Deciles: Divide datos en 10 partes iguales")

print("\n🎨 PARA VARIABLES CATEGÓRICAS:")
print("1. Bar-chart: Número de máquinas por categoría")
print("2. Conteo directo: Tabla con valores y porcentajes")

print("\n💡 INTERPRETACIÓN DE LOS GRÁFICOS:")
print("• Distribución normal: Forma de campana (histograma)")
print("• Outliers: Puntos fuera de los bigotes (box-plot)")
print("• Asimetría: Cola larga hacia un lado (violin-plot)")
print("• Categorías dominantes: Barras muy altas (bar-chart)")

print("\n🎯 PARA TU PROYECTO DE MALWARE:")
print("• Las variables con distribución diferente entre")
print("  máquinas con/sin malware son más predictivas")
print("• Los outliers pueden indicar errores o casos especiales")
print("• Las categorías poco frecuentes pueden agruparse")

print("\n" + "=" * 60)
print("✅ VISUALIZACIÓN GRÁFICA COMPLETADA")
print("=" * 60)

# %% [markdown] id="uFkhZGgNikL0"
# # **1.2 Análisis Multi-variante de los datos**

# %% [markdown] id="tvHfLoUDirwS"
# **1. Distribución de variables 2 a 2 (scatter-plots)**

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="tI6nMwxWzT-b" outputId="47c67b66-5639-4deb-f776-20890240df0c"
# -----------------------------------------------------------------
# PASO 12: RELACIONES ENTRE VARIABLES
# -----------------------------------------------------------------
print("\n🔗 PASO 12: RELACIONES ENTRE VARIABLES...")

print("1. Scatter plots de variables con mayor correlación con el target:")

# Variables con mayor correlación (de tu lista)
variables_correlacion = [
    'AVProductsInstalled',              # -0.1488
    'AVProductStatesIdentifier',        # +0.1162
    'IsProtected',                      # +0.0562
    'Census_ProcessorCoreCount',        # +0.0561
    'Census_PrimaryDiskTotalCapacity',  # +0.0480
    'AVProductsEnabled',                # -0.0419
    'RtpStateBitfield',                 # +0.0403
    'IsSxsPassiveMode'                  # -0.0340
]

# Filtrar solo las que existen en los datos
variables_existentes = [v for v in variables_correlacion if v in datos.columns]
print(f"   Variables encontradas: {len(variables_existentes)} de {len(variables_correlacion)}")

if len(variables_existentes) >= 4:
    # Crear figura con 2x4 = 8 subplots (una por variable)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, variable in enumerate(variables_existentes[:8]):  # Máximo 8 gráficos
        if idx < len(axes):
            ax = axes[idx]

            # Obtener coeficiente de correlación real
            if datos[variable].dtype in ['int64', 'float64']:
                # Filtrar filas donde ambas variables no sean NaN
                mask = datos[[variable, 'HasDetections']].notna().all(axis=1)
                x_data = datos.loc[mask, variable]
                y_data = datos.loc[mask, 'HasDetections']

                corr_valor = x_data.corr(y_data)
                color = 'red' if corr_valor < 0 else 'green'
                signo = '-' if corr_valor < 0 else '+'

                # Scatter plot para variables numéricas
                ax.scatter(x_data, y_data,
                          alpha=0.4, s=15, color='steelblue', edgecolor='black', linewidth=0.2)

                # Añadir línea de tendencia (solo si hay datos suficientes)
                if len(x_data) > 1:
                    try:
                        z = np.polyfit(x_data, y_data, 1)
                        p = np.poly1d(z)
                        sorted_idx = np.argsort(x_data)
                        ax.plot(x_data.iloc[sorted_idx],
                               p(x_data.iloc[sorted_idx]),
                               color='darkred', linewidth=2, alpha=0.7)
                    except:
                        pass  # Si falla el polyfit, continuar sin línea de tendencia

                # Etiqueta con correlación
                ax.text(0.05, 0.95, f'ρ = {signo}{abs(corr_valor):.3f}',
                       transform=ax.transAxes, fontsize=10, fontweight='bold',
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

                # Añadir conteo de datos
                ax.text(0.05, 0.05, f'n = {len(x_data):,}',
                       transform=ax.transAxes, fontsize=8,
                       verticalalignment='bottom',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

            else:
                # Para variables categóricas, hacer boxplot
                # Tomar top 8 categorías para mejor visualización
                top_cats = datos[variable].dropna().value_counts().nlargest(8).index
                datos_filtrados = datos[datos[variable].isin(top_cats)]

                # Crear boxplot manualmente para mejor control
                data_to_plot = []
                labels = []
                for cat in top_cats:
                    subset = datos_filtrados[datos_filtrados[variable] == cat]['HasDetections']
                    if len(subset) > 0:
                        data_to_plot.append(subset.dropna().values)
                        labels.append(str(cat)[:15])  # Truncar etiquetas largas

                if data_to_plot:  # Solo si hay datos
                    bp = ax.boxplot(data_to_plot, patch_artist=True, labels=labels)

                    # Colorear las cajas
                    for patch in bp['boxes']:
                        patch.set_facecolor('lightblue')
                        patch.set_alpha(0.6)

                    ax.tick_params(axis='x', rotation=45, labelsize=8)

                    # Calcular diferencia de medias entre categorías
                    if len(data_to_plot) >= 2:
                        mean_diff = np.mean(data_to_plot[0]) - np.mean(data_to_plot[-1])
                        ax.text(0.05, 0.95, f'Δμ = {mean_diff:.3f}',
                               transform=ax.transAxes, fontsize=9,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
                else:
                    ax.text(0.5, 0.5, 'Sin datos suficientes',
                           transform=ax.transAxes, ha='center', va='center')

            ax.set_title(f'{variable[:20]}', fontsize=11, fontweight='bold')
            ax.set_xlabel(variable[:15] + '...' if len(variable) > 15 else variable, fontsize=9)
            ax.set_ylabel('Malware (0/1)', fontsize=9)
            ax.grid(True, alpha=0.2)

    # Ajustar layout
    plt.suptitle('VARIABLES CON MAYOR CORRELACIÓN CON DETECCIÓN DE MALWARE\n' +
                '(Rojo=correlación negativa, Verde=correlación positiva)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    # 2. Heatmap de correlaciones entre estas variables
    print("\n2. Matriz de correlación entre variables importantes:")

    # Filtrar solo variables numéricas
    vars_numericas = [v for v in variables_existentes if datos[v].dtype in ['int64', 'float64']]
    vars_numericas.append('HasDetections')  # Añadir el target

    if len(vars_numericas) > 2:
        # Filtrar filas completas (sin NaN en ninguna de las variables)
        datos_completos = datos[vars_numericas].dropna()
        if len(datos_completos) > 0:
            corr_matrix = datos_completos.corr()

            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Máscara para triángulo superior
            heatmap = sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r',
                                 center=0, fmt='.2f', linewidths=1,
                                 square=True, cbar_kws={"shrink": 0.8})

            plt.title('MATRIZ DE CORRELACIÓN ENTRE VARIABLES IMPORTANTES\n' +
                     f'(Basado en {len(datos_completos):,} filas completas)',
                     fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()

            # Mostrar correlaciones más fuertes con el target
            print(f"\n3. Resumen de correlaciones con HasDetections (n={len(datos_completos):,}):")
            target_corrs = corr_matrix['HasDetections'].drop('HasDetections').sort_values(key=abs, ascending=False)
            for var, corr in target_corrs.head(10).items():
                signo = '  +' if corr > 0 else '  -'
                print(f"   {var:35} : {signo}{abs(corr):.4f}")
        else:
            print("   No hay suficientes filas completas para calcular correlaciones")

# %% [markdown] id="Nm4S52FUivpp"
# **2. Correlación de las variables 2 a 2 (correlación lineal)**

# %% colab={"base_uri": "https://localhost:8080/", "height": 612} id="SDMQL1EOee0s" outputId="9920bad0-028a-4521-9c3a-d8573c57104a"
print("\n2. Matriz de correlación (todas las variables):")

# Matriz de correlación simplificada
variables_corr = ['AVProductsInstalled', 'AVProductStatesIdentifier',
                 'Census_TotalPhysicalRAM', 'Census_ProcessorCoreCount',
                 'HasDetections']

datos_corr = datos[variables_corr].corr()

plt.figure(figsize=(8, 6))
plt.imshow(datos_corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Correlación')
plt.xticks(range(len(variables_corr)), variables_corr, rotation=45)
plt.yticks(range(len(variables_corr)), variables_corr)
plt.title('Matriz de Correlación')

# Añadir valores
for i in range(len(variables_corr)):
    for j in range(len(variables_corr)):
        plt.text(j, i, f'{datos_corr.iloc[i, j]:.2f}',
                ha='center', va='center', color='white' if abs(datos_corr.iloc[i, j]) > 0.5 else 'black')

plt.tight_layout()
plt.show()


# %% id="capIgIB-kS5n"
# AVProductStatesIdentifier y AVProductsInstalled presentan una correlación de 0.63.
# Por un lado, AVProductsInstalled es el número de antivirus instalados (0,1,2,3...)
# Por otro lado, AVProductStatesIdentifier es la configuración específica del antivirus.
# La correlación nos dice que: a medida que aumenta el ID de configuración del antivirus, también aumenta el número de antivirus instalados.
# O viceversa: a más antivirus instalados, es más probable tener ciertas configuraciones.


# %% id="J8XiDOJUlTJV"
# Census_ProcessorCoreCount y Census_TotalPysicalRAM presentan una correlación de 0.61.
# Por un lado, Censes_ProcessorCoreCount es el número de núcleos del CPU.
# Por otro lado, Census_TotalPysicalRAM es la cantidad de memoria de RAM (en MB).
# La correlación nos dice que: a medida que tenemos ordenadores con procesadores más potentes (más núcleos), también tienen más memoria RAM.
# Esto tiene sentido: el hardware de gama alta suele venir con componentes balanceados.


# %% [markdown] id="7awCwporiz6J"
# **3. Cross-tabs**

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="OZULw_VIefJE" outputId="47626a45-db8a-49fb-d883-2b199aa9cd08"
# -----------------------------------------------------------------
# PASO 13: TABLAS CRUZADAS
# -----------------------------------------------------------------
print("\n📋 PASO 13: TABLAS CRUZADAS...")

print("1. Tablas cruzadas de variables categóricas importantes:")

# Variables con mayor correlación (de tu lista)
variables_correlacion = [
    'AVProductsInstalled',              # -0.1488
    'AVProductStatesIdentifier',        # +0.1162
    'IsProtected',                      # +0.0562
    'Census_ProcessorCoreCount',        # +0.0561
    'Census_PrimaryDiskTotalCapacity',  # +0.0480
    'AVProductsEnabled',                # -0.0419
    'RtpStateBitfield',                 # +0.0403
    'IsSxsPassiveMode'                  # -0.0340
]

# Filtrar solo las que existen en los datos
variables_existentes = [v for v in variables_correlacion if v in datos.columns]

# Separar variables numéricas y categóricas
variables_numericas = [v for v in variables_existentes if datos[v].dtype in ['int64', 'float64']]
variables_categoricas = [v for v in variables_existentes if datos[v].dtype == 'object' or datos[v].nunique() < 20]

print(f"   Variables categóricas: {len(variables_categoricas)}")
print(f"   Variables numéricas: {len(variables_numericas)}")

# Análisis de tablas cruzadas para variables categóricas
if variables_categoricas:
    print("\n2. Tablas cruzadas con HasDetections:")

    for i, variable in enumerate(variables_categoricas[:6]):  # Mostrar máximo 6
        print(f"\n   {i+1}. {variable}:")

        # Crear tabla cruzada
        tabla = pd.crosstab(datos[variable], datos['HasDetections'], margins=True)

        # Calcular porcentajes por fila
        tabla_porcentaje = pd.crosstab(datos[variable], datos['HasDetections'], normalize='index') * 100

        # Combinar conteos y porcentajes
        tabla_completa = pd.concat([
            tabla.iloc[:-1],  # Conteos sin "All"
            tabla_porcentaje.round(1).astype(str) + '%'
        ], axis=1, keys=['Conteo', 'Porcentaje'])

        # Mostrar tabla formateada
        print(tabla_completa.to_string())

        # Calcular y mostrar odds ratio si es binaria
        if datos[variable].nunique() == 2:
            valores = datos[variable].dropna().unique()
            if len(valores) == 2:
                a = len(datos[(datos[variable] == valores[0]) & (datos['HasDetections'] == 1)])
                b = len(datos[(datos[variable] == valores[0]) & (datos['HasDetections'] == 0)])
                c = len(datos[(datos[variable] == valores[1]) & (datos['HasDetections'] == 1)])
                d = len(datos[(datos[variable] == valores[1]) & (datos['HasDetections'] == 0)])

                if b > 0 and c > 0 and d > 0:
                    odds_ratio = (a * d) / (b * c)
                    print(f"      Odds Ratio: {odds_ratio:.3f}")
                    print(f"      Interpretación: {valores[0]} vs {valores[1]}")

        # Calcular chi-cuadrado
        from scipy.stats import chi2_contingency
        tabla_sin_total = pd.crosstab(datos[variable], datos['HasDetections'])
        chi2, p, dof, expected = chi2_contingency(tabla_sin_total)
        print(f"      Chi-cuadrado: {chi2:.2f}, p-valor: {p:.4f}")

        # Gráfico de barras
        if i < 3:  # Solo hacer gráficos para las primeras 3
            plt.figure(figsize=(10, 5))

            # Obtener top 10 categorías
            top_cats = datos[variable].value_counts().nlargest(10).index
            datos_filtrados = datos[datos[variable].isin(top_cats)]

            # Crear tabla para el gráfico
            tabla_graf = pd.crosstab(datos_filtrados[variable], datos_filtrados['HasDetections'], normalize='index')

            # Gráfico de barras apiladas
            tabla_graf.plot(kind='bar', stacked=True, colormap='RdYlBu_r', figsize=(10, 5))
            plt.title(f'Distribución de Malware por {variable[:20]}', fontweight='bold')
            plt.xlabel(variable[:15] + '...' if len(variable) > 15 else variable)
            plt.ylabel('Proporción')
            plt.legend(title='Malware', labels=['No', 'Sí'])
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

# Análisis de variables numéricas con tablas cruzadas (discretizadas)
if variables_numericas:
    print("\n3. Variables numéricas discretizadas (quintiles):")

    for i, variable in enumerate(variables_numericas[:4]):  # Mostrar máximo 4
        print(f"\n   {i+1}. {variable}:")

        # Discretizar en quintiles
        datos[f'{variable}_q'] = pd.qcut(datos[variable], q=5, duplicates='drop')

        # Crear tabla cruzada
        tabla = pd.crosstab(datos[f'{variable}_q'], datos['HasDetections'], margins=True)

        # Calcular porcentajes por fila
        tabla_porcentaje = pd.crosstab(datos[f'{variable}_q'], datos['HasDetections'], normalize='index') * 100

        # Combinar
        tabla_completa = pd.concat([
            tabla.iloc[:-1],
            tabla_porcentaje.round(1).astype(str) + '%'
        ], axis=1, keys=['Conteo', 'Tasa Malware'])

        print(tabla_completa.to_string())

        # Gráfico de tendencia
        if i < 2:  # Solo hacer gráficos para las primeras 2
            plt.figure(figsize=(10, 5))

            # Calcular tasa de malware por quintil
            quintil_orden = sorted(datos[f'{variable}_q'].dropna().unique())
            tasas = []
            for q in quintil_orden:
                tasa = datos[datos[f'{variable}_q'] == q]['HasDetections'].mean() * 100
                tasas.append(tasa)

            # Gráfico de línea
            plt.plot(range(len(quintil_orden)), tasas, 'o-', linewidth=2, markersize=8)
            plt.fill_between(range(len(quintil_orden)), tasas, alpha=0.2)

            plt.title(f'Tasa de Malware por Quintil de {variable}', fontweight='bold')
            plt.xlabel('Quintil (1=Menor, 5=Mayor)')
            plt.ylabel('Tasa de Malware (%)')
            plt.grid(True, alpha=0.3)

            # Añadir etiquetas
            for j, (q, tasa) in enumerate(zip(quintil_orden, tasas)):
                plt.text(j, tasa + 0.5, f'{tasa:.1f}%', ha='center', fontsize=9)

            plt.tight_layout()
            plt.show()

# 4. Análisis de combinaciones importantes
print("\n4. Combinaciones importantes de variables:")

if 'AVProductsInstalled' in datos.columns and 'AVProductsEnabled' in datos.columns:
    print("\n   AVProductsInstalled vs AVProductsEnabled:")

    # Crear tabla cruzada doble
    tabla_doble = pd.crosstab(
        datos['AVProductsInstalled'],
        datos['AVProductsEnabled'],
        datos['HasDetections'],
        aggfunc='mean'
    ).round(3) * 100

    print(tabla_doble.to_string())

    # Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(tabla_doble, annot=True, fmt='.1f', cmap='RdYlBu_r',
                linewidths=1, linecolor='black')
    plt.title('Tasa de Malware por Combinación\nAV Instalados vs AV Habilitados',
              fontweight='bold')
    plt.xlabel('AV Products Enabled')
    plt.ylabel('AV Products Installed')
    plt.tight_layout()
    plt.show()

# 5. Resumen estadístico
print("\n5. Resumen estadístico de correlaciones:")
for variable in variables_existentes[:8]:
    if variable in datos.columns:
        if datos[variable].dtype in ['int64', 'float64']:
            corr = datos[variable].corr(datos['HasDetections'])
            print(f"   {variable:35} : {corr:+.4f}")
        else:
            # Para categóricas, calcular Cramer's V
            from scipy.stats import chi2_contingency
            tabla = pd.crosstab(datos[variable], datos['HasDetections'])
            chi2, p, dof, expected = chi2_contingency(tabla)
            n = tabla.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(tabla.shape) - 1)))
            print(f"   {variable:35} : Cramer's V = {cramers_v:.4f} (p={p:.4f})")

# %% [markdown] id="4e5j2yici3Lo"
# **4. Correlación de combinaciones de variables con la clase**
#
#

# %% colab={"base_uri": "https://localhost:8080/"} id="_CHehB-0efml" outputId="29630fb3-ac9c-414e-e908-6b3a1a4e4daf"
# -----------------------------------------------------------------
# PASO 15: COMBINACIONES DE VARIABLES
# -----------------------------------------------------------------
print("\n🧩 PASO 15: COMBINACIONES DE VARIABLES...")

print("Creando combinaciones interesantes:")

# 1. Combinación: RAM por núcleo
if 'Census_TotalPhysicalRAM' in datos.columns and 'Census_ProcessorCoreCount' in datos.columns:
    datos['RAM_por_Nucleo'] = datos['Census_TotalPhysicalRAM'] / datos['Census_ProcessorCoreCount'].replace(0, 1)
    corr = datos['RAM_por_Nucleo'].corr(datos['HasDetections'])
    print(f"  RAM por Núcleo: Correlación con malware = {corr:.3f}")

# 2. Combinación: Antivirus por RAM
if 'AVProductsInstalled' in datos.columns and 'Census_TotalPhysicalRAM' in datos.columns:
    datos['Antivirus_por_RAM'] = datos['AVProductsInstalled'] / (datos['Census_TotalPhysicalRAM'] / 1000).replace(0, 1)
    corr = datos['Antivirus_por_RAM'].corr(datos['HasDetections'])
    print(f"  Antivirus por GB de RAM: Correlación = {corr:.3f}")

# 3. Protección completa (ambas activas)
if 'IsProtected' in datos.columns and 'Firewall' in datos.columns:
    # Asumir que 'True' y 'ON' son buenos
    datos['Proteccion_Completa'] = ((datos['IsProtected'] == 'True') &
                                   (datos['Firewall'] == 'ON')).astype(int)

    print("\nTasa de malware por tipo de protección:")
    for proteccion in [0, 1]:
        subset = datos[datos['Proteccion_Completa'] == proteccion]
        tasa = subset['HasDetections'].mean() * 100
        tipo = "COMPLETA" if proteccion == 1 else "INCOMPLETA"
        print(f"  Protección {tipo}: {tasa:.1f}% tiene malware")

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="a8z1Clmg0eiF" outputId="432ba654-943f-4c30-d23c-b225ab2fb56d"
"""
RANDOM FOREST SENCILLO PARA PREDECIR MALWARE
Con todas las importaciones incluidas
"""

print("🌲 ENTRENANDO RANDOM FOREST SENCILLO")
print("=" * 50)

# -----------------------------------------------------------------
# PASO 0: IMPORTAR TODO LO NECESARIO
# -----------------------------------------------------------------
print("\n0️⃣ IMPORTANDO LIBRERÍAS...")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score

print("✅ Librerías importadas correctamente")

# -----------------------------------------------------------------
# PASO 1: CARGAR DATOS SI NO ESTÁN CARGADOS
# -----------------------------------------------------------------
print("\n1️⃣ CARGANDO DATOS...")

# Si 'datos' no existe, cargarlo
if 'datos' not in locals():
    url = "https://www.dropbox.com/s/sxl5bpi2620p496/sample_mmp.csv?dl=1"
    datos = pd.read_csv(url)
    print(f"✓ Datos cargados: {datos.shape[0]:,} filas")
else:
    print(f"✓ Datos ya cargados: {datos.shape[0]:,} filas")

# -----------------------------------------------------------------
# PASO 2: SELECCIONAR VARIABLES
# -----------------------------------------------------------------
print("\n2️⃣ SELECCIONANDO VARIABLAS IMPORTANTES")

# Variables numéricas importantes
variables = [
    'AVProductsInstalled',          # Nº antivirus
    'AVProductStatesIdentifier',    # Config antivirus
    'Census_TotalPhysicalRAM',      # Memoria RAM
    'Census_ProcessorCoreCount',    # Núcleos CPU
    'Census_PrimaryDiskTotalCapacity',  # Disco duro
    'OsBuild',                      # Build del sistema
    'CountryIdentifier',            # País
    'HasDetections'                 # Target (lo que queremos predecir)
]

# Filtrar solo las que existen
variables_existentes = []
for var in variables:
    if var in datos.columns:
        variables_existentes.append(var)

print(f"✓ Usando {len(variables_existentes)-1} variables:")
for var in variables_existentes:
    if var != 'HasDetections':
        print(f"  • {var}")

# -----------------------------------------------------------------
# PASO 3: MANEJAR VALORES NULOS
# -----------------------------------------------------------------
print("\n3️⃣ MANEJANDO VALORES NULOS")

# Crear copia solo con las variables que necesitamos
datos_modelo = datos[variables_existentes].copy()

print("Valores nulos encontrados:")
for col in datos_modelo.columns:
    nulos = datos_modelo[col].isnull().sum()
    if nulos > 0:
        porcentaje = (nulos / len(datos_modelo)) * 100
        print(f"  {col}: {nulos:,} nulos ({porcentaje:.1f}%)")

        # Rellenar con la MEDIANA si es numérica
        if datos_modelo[col].dtype in ['int64', 'float64']:
            valor_relleno = datos_modelo[col].median()
            datos_modelo[col] = datos_modelo[col].fillna(valor_relleno)
            print(f"    → Rellenados con: {valor_relleno:.0f}")

print("✓ Valores nulos manejados")

# -----------------------------------------------------------------
# PASO 4: PREPARAR DATOS
# -----------------------------------------------------------------
print("\n4️⃣ PREPARANDO DATOS")

# Separar X (características) y y (target)
X = datos_modelo.drop('HasDetections', axis=1)
y = datos_modelo['HasDetections']

print(f"✓ X (características): {X.shape[1]} columnas")
print(f"✓ y (target): {len(y)} valores")

# Ver distribución del target
print(f"\nDistribución del target (HasDetections):")
print(f"  0 (Sin malware): {(y == 0).sum():,} máquinas")
print(f"  1 (Con malware): {(y == 1).sum():,} máquinas")
print(f"  % con malware: {(y == 1).sum()/len(y)*100:.1f}%")

# Dividir en entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\n✓ Datos de ENTRENAMIENTO: {X_train.shape[0]:,} máquinas")
print(f"✓ Datos de PRUEBA: {X_test.shape[0]:,} máquinas")

# -----------------------------------------------------------------
# PASO 5: ENTRENAR RANDOM FOREST
# -----------------------------------------------------------------
print("\n5️⃣ ENTRENANDO RANDOM FOREST")

# Crear modelo Random Forest
modelo = RandomForestClassifier(
    n_estimators=50,        # 50 árboles
    max_depth=10,           # Profundidad máxima
    random_state=42,        # Para resultados reproducibles
    n_jobs=-1               # Usar todos los núcleos del CPU
)

print("Entrenando modelo...")
modelo.fit(X_train, y_train)
print("✓ Modelo entrenado!")

# -----------------------------------------------------------------
# PASO 6: HACER PREDICCIONES
# -----------------------------------------------------------------
print("\n6️⃣ HACIENDO PREDICCIONES")

# Predecir en datos de prueba
y_pred = modelo.predict(X_test)
y_pred_prob = modelo.predict_proba(X_test)[:, 1]  # Probabilidad de tener malware

print(f"✓ Predicciones realizadas")
print(f"  Ejemplo de primeras 5 predicciones: {y_pred[:5]}")
print(f"  Ejemplo de probabilidades: {y_pred_prob[:5].round(3)}")

# -----------------------------------------------------------------
# PASO 7: EVALUAR EL MODELO
# -----------------------------------------------------------------
print("\n7️⃣ EVALUANDO EL MODELO")

# 1. Calcular precisión
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 PRECISIÓN DEL MODELO: {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"   (De cada 100 predicciones, acertamos en {int(accuracy*100)})")

# 2. Matriz de confusión
print("\n📋 MATRIZ DE CONFUSIÓN:")
cm = confusion_matrix(y_test, y_pred)

# Mostrar matriz de forma simple
print(f"""
        PREDICCIÓN
       ------------
       | {cm[0,0]:^5} | {cm[0,1]:^5} |   ← REAL: Sin malware
       ------------
       | {cm[1,0]:^5} | {cm[1,1]:^5} |   ← REAL: Con malware
       ------------
         ↑       ↑
    Pred:     Pred:
   No malw.  Con malw.
""")

# Explicar qué significa
print("💡 ¿QUÉ SIGNIFICA?")
print(f"• {cm[0,0]} casos: REAL sin malware, PREDICHO sin malware ✓")
print(f"• {cm[0,1]} casos: REAL sin malware, PREDICHO con malware ✗ (Falso positivo)")
print(f"• {cm[1,0]} casos: REAL con malware, PREDICHO sin malware ✗ (Falso negativo)")
print(f"• {cm[1,1]} casos: REAL con malware, PREDICHO con malware ✓")

# 3. Métricas importantes
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"\n📈 OTRAS MÉTRICAS IMPORTANTES:")
print(f"• Precision: {precision:.3f}")
print(f"  (De los que predigo con malware, cuántos realmente lo tienen)")
print(f"• Recall: {recall:.3f}")
print(f"  (De los que realmente tienen malware, cuántos detecto)")

# 4. Reporte completo
print("\n📊 REPORTE COMPLETO:")
print(classification_report(y_test, y_pred,
                          target_names=['Sin Malware', 'Con Malware']))

# -----------------------------------------------------------------
# PASO 8: VER IMPORTANCIA DE VARIABLES
# -----------------------------------------------------------------
print("\n8️⃣ IMPORTANCIA DE LAS VARIABLES")

# Obtener importancia de cada variable
importancias = modelo.feature_importances_

# Crear tabla ordenada
importancia_df = pd.DataFrame({
    'Variable': X.columns,
    'Importancia': importancias
}).sort_values('Importancia', ascending=False)

print("\n🏆 VARIABLES MÁS IMPORTANTES PARA EL MODELO:")
print("-" * 40)
for i, (_, fila) in enumerate(importancia_df.iterrows(), 1):
    print(f"{i:2}. {fila['Variable']:<30}: {fila['Importancia']:.3f}")

# Gráfico simple de importancia
plt.figure(figsize=(10, 5))
plt.barh(importancia_df['Variable'], importancia_df['Importancia'], color='skyblue')
plt.xlabel('Importancia')
plt.title('Variables más importantes para detectar malware')
plt.gca().invert_yaxis()  # Más importante arriba
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------
# PASO 9: USAR EL MODELO
# -----------------------------------------------------------------
print("\n9️⃣ CÓMO USAR EL MODELO")

# Función para predecir nuevas máquinas
def predecir_malware(nueva_maquina):
    """
    Predice si una máquina tiene malware
    """
    # Hacer predicción
    probabilidad = modelo.predict_proba(nueva_maquina)[0, 1]
    prediccion = modelo.predict(nueva_maquina)[0]

    return probabilidad, prediccion

print("✓ Función 'predecir_malware()' creada")

# Ejemplo de uso
print("\n📝 EJEMPLO DE USO:")
ejemplo_maquina = X_test.head(1).copy()
prob_ejemplo, pred_ejemplo = predecir_malware(ejemplo_maquina)
print(f"  Probabilidad de malware: {prob_ejemplo:.1%}")
print(f"  Predicción: {'CON malware' if pred_ejemplo == 1 else 'SIN malware'}")

# -----------------------------------------------------------------
# PASO 10: RESUMEN
# -----------------------------------------------------------------
print("\n" + "=" * 50)
print("📋 RESUMEN FINAL")
print("=" * 50)

print(f"""
🎯 RESULTADOS:
• Precisión: {accuracy:.1%}
• Variables más importantes:
  1. {importancia_df.iloc[0]['Variable']}
  2. {importancia_df.iloc[1]['Variable']}
• Tamaño del modelo: {modelo.n_estimators} árboles

💡 RECOMENDACIONES:
1. Para Windows Defender: Usar umbral de 60% para alertas
2. Priorizar máquinas con {importancia_df.iloc[0]['Variable']} problemático
3. Combinar con otras reglas para reducir falsos positivos

✅ MODELO LISTO PARA USAR
""")

print("\n" + "=" * 50)
print("🌲 RANDOM FOREST ENTRENADO EXITOSAMENTE")
print("=" * 50)

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="-50InZbo6KHv" outputId="024f48e5-08f6-4f12-9795-fdadd82f645b"
"""
MÉTRICAS DE REGRESIÓN PARA RANDOM FOREST YA ENTRENADO
MAE, MSE, RMSE, R² para Train y Test
"""

print("📊 MÉTRICAS DE REGRESIÓN - RANDOM FOREST YA ENTRENADO")
print("=" * 60)

# -----------------------------------------------------------------
# PASO 1: VERIFICAR QUE TENEMOS EL MODELO Y DATOS
# -----------------------------------------------------------------
print("\n1️⃣ VERIFICANDO MODELO Y DATOS...")

# Verificar que tenemos las variables necesarias
variables_necesarias = ['modelo', 'X_train', 'X_test', 'y_train', 'y_test']
variables_faltantes = []

for var in variables_necesarias:
    if var not in locals():
        variables_faltantes.append(var)

if variables_faltantes:
    print(f"⚠️  Variables faltantes: {variables_faltantes}")
    print("   Entrenando modelo rápido...")

    # Entrenar modelo rápido si falta
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Usar variables simples
    vars_simple = ['AVProductsInstalled', 'Census_TotalPhysicalRAM', 'HasDetections']
    datos_simple = datos[[v for v in vars_simple if v in datos.columns]].copy()

    # Rellenar nulos
    for col in datos_simple.columns:
        if col != 'HasDetections' and datos_simple[col].isnull().any():
            datos_simple[col] = datos_simple[col].fillna(datos_simple[col].median())

    X = datos_simple.drop('HasDetections', axis=1)
    y = datos_simple['HasDetections']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    modelo = RandomForestClassifier(n_estimators=50, random_state=42)
    modelo.fit(X_train, y_train)

    print(f"✓ Modelo rápido entrenado con {X.shape[1]} variables")
else:
    print("✓ Modelo y datos disponibles")

print(f"✓ Train samples: {X_train.shape[0]:,}")
print(f"✓ Test samples:  {X_test.shape[0]:,}")

# -----------------------------------------------------------------
# PASO 2: OBTENER PREDICCIONES Y PROBABILIDADES
# -----------------------------------------------------------------
print("\n2️⃣ OBTENIENDO PREDICCIONES Y PROBABILIDADES...")

# Para clasificación: predicciones (0 o 1)
y_train_pred_class = modelo.predict(X_train)
y_test_pred_class = modelo.predict(X_test)

# Para regresión: probabilidades (valores entre 0 y 1)
# Estas son las que usaremos para métricas de regresión
y_train_pred_prob = modelo.predict_proba(X_train)[:, 1]  # Probabilidad de clase 1 (malware)
y_test_pred_prob = modelo.predict_proba(X_test)[:, 1]    # Probabilidad de clase 1 (malware)

print("✓ Predicciones obtenidas")
print(f"  Train - Ejemplo probabilidades: {y_train_pred_prob[:5].round(3)}")
print(f"  Test  - Ejemplo probabilidades: {y_test_pred_prob[:5].round(3)}")

# -----------------------------------------------------------------
# PASO 3: CALCULAR MÉTRICAS DE REGRESIÓN
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("3️⃣ CALCULANDO MÉTRICAS DE REGRESIÓN (MAE, MSE, RMSE, R²)")
print("=" * 60)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Calcular métricas para TRAIN
mae_train = mean_absolute_error(y_train, y_train_pred_prob)
mse_train = mean_squared_error(y_train, y_train_pred_prob)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_train_pred_prob)

# Calcular métricas para TEST
mae_test = mean_absolute_error(y_test, y_test_pred_prob)
mse_test = mean_squared_error(y_test, y_test_pred_prob)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_test_pred_prob)

# -----------------------------------------------------------------
# PASO 4: MOSTRAR RESULTADOS EN TABLA
# -----------------------------------------------------------------
print("\n📊 RESULTADOS DE REGRESIÓN:")
print("=" * 70)
print(f"{'Métrica':<10} {'Train':<15} {'Test':<15} {'Diferencia':<15}")
print("=" * 70)

# Crear lista de métricas
metricas = [
    ('MAE', mae_train, mae_test),
    ('MSE', mse_train, mse_test),
    ('RMSE', rmse_train, rmse_test),
    ('R²', r2_train, r2_test)
]

for nombre, train_val, test_val in metricas:
    diferencia = train_val - test_val
    signo = "+" if diferencia > 0 else ""

    if nombre == 'R²':
        # R² se formatea diferente
        print(f"{nombre:<10} {train_val:<15.4f} {test_val:<15.4f} {signo}{diferencia:<14.4f}")
    else:
        print(f"{nombre:<10} {train_val:<15.4f} {test_val:<15.4f} {signo}{diferencia:<14.4f}")

print("=" * 70)

# -----------------------------------------------------------------
# PASO 5: EXPLICACIÓN DE LAS MÉTRICAS
# -----------------------------------------------------------------
print("\n💡 ¿QUÉ SIGNIFICAN ESTAS MÉTRICAS?")
print("-" * 50)

print("""
MAE (Error Absoluto Medio):
• Promedio de los errores absolutos
• Ejemplo: MAE = 0.25 → Error promedio de 25% en probabilidades
• Más bajo = mejor

MSE (Error Cuadrático Medio):
• Promedia los errores al cuadrado
• Penaliza más los errores grandes
• Más bajo = mejor

RMSE (Raíz del Error Cuadrático Medio):
• Raíz cuadrada del MSE
• En las mismas unidades que la variable original
• Más bajo = mejor

R² (Coeficiente de Determinación):
• Qué porcentaje de la variabilidad explica el modelo
• Rango: -∞ a 1.0
• 1.0 = perfecto, 0 = no explica nada, negativo = peor que el promedio
• Más alto = mejor
""")

# -----------------------------------------------------------------
# PASO 6: INTERPRETACIÓN DE LOS RESULTADOS
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("4️⃣ INTERPRETACIÓN DE LOS RESULTADOS")
print("=" * 60)

print("\n🔍 ANÁLISIS DE NUESTRO MODELO:")

# 1. Analizar R²
print(f"\n1. R² SCORE:")
print(f"   • Train: {r2_train:.4f}")
print(f"   • Test:  {r2_test:.4f}")

if r2_test < 0:
    print("   ⚠️  R² NEGATIVO en Test: El modelo es PEOR que usar el promedio simple")
elif r2_test < 0.1:
    print("   ⚠️  R² BAJO (< 0.1): El modelo explica muy poca variabilidad")
elif r2_test < 0.3:
    print("   ✅ R² MODERADO (0.1-0.3): El modelo explica algo de variabilidad")
elif r2_test < 0.5:
    print("   ✅✅ R² BUENO (0.3-0.5): El modelo explica bastante variabilidad")
else:
    print("   ✅✅✅ R² EXCELENTE (> 0.5): El modelo explica mucha variabilidad")

# 2. Analizar diferencias entre Train y Test
print(f"\n2. DIFERENCIAS TRAIN vs TEST:")
print(f"   • Diferencia en MAE: {mae_train - mae_test:+.4f}")
print(f"   • Diferencia en MSE: {mse_train - mse_test:+.4f}")
print(f"   • Diferencia en R²:  {r2_train - r2_test:+.4f}")

# Verificar overfitting
if (r2_train - r2_test) > 0.1:
    print("   ⚠️  POSIBLE OVERFITTING: R² Train mucho mayor que R² Test")
elif abs(r2_train - r2_test) < 0.05:
    print("   ✅ BUENA GENERALIZACIÓN: R² similar en Train y Test")

# 3. Analizar errores absolutos
print(f"\n3. ERRORES ABSOLUTOS:")
print(f"   • MAE Test: {mae_test:.4f}")
print(f"     → Error promedio de {mae_test*100:.1f}% en probabilidades")

if mae_test < 0.3:
    print("   ✅ Error bajo: Buen rendimiento del modelo")
elif mae_test < 0.4:
    print("   ⚠️  Error moderado: Rendimiento aceptable")
else:
    print("   ⚠️⚠️  Error alto: El modelo podría mejorar")

# -----------------------------------------------------------------
# PASO 7: GRÁFICOS DE COMPARACIÓN
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("5️⃣ VISUALIZACIÓN DE RESULTADOS")
print("=" * 60)

import matplotlib.pyplot as plt

# Gráfico 1: Comparación de métricas de error (MAE, MSE, RMSE)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Gráfico de barras para errores
metricas_errores = ['MAE', 'MSE', 'RMSE']
valores_train = [mae_train, mse_train, rmse_train]
valores_test = [mae_test, mse_test, rmse_test]

x = np.arange(len(metricas_errores))
width = 0.35

axes[0].bar(x - width/2, valores_train, width, label='Train', color='blue', alpha=0.7)
axes[0].bar(x + width/2, valores_test, width, label='Test', color='red', alpha=0.7)
axes[0].set_xlabel('Métricas de Error')
axes[0].set_ylabel('Valor')
axes[0].set_title('Métricas de Error: Train vs Test\n(Más bajo = mejor)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metricas_errores)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Gráfico para R²
axes[1].bar(['Train', 'Test'], [r2_train, r2_test],
           color=['blue', 'red'], alpha=0.7)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1].set_xlabel('Conjunto de datos')
axes[1].set_ylabel('Valor R²')
axes[1].set_title('R² Score: Train vs Test\n(Más alto = mejor)')
axes[1].set_ylim(min(r2_train, r2_test, 0) - 0.1, 1.0)
axes[1].grid(True, alpha=0.3, axis='y')

# Añadir valores en las barras de R²
for i, valor in enumerate([r2_train, r2_test]):
    axes[1].text(i, valor + 0.02, f'{valor:.3f}',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# Gráfico 2: Valores reales vs predichos (scatter plot)
print("\n📈 GRÁFICO: Valores Reales vs Probabilidades Predichas (Test)")

plt.figure(figsize=(10, 4))

# Ordenar para mejor visualización
indices_ordenados = np.argsort(y_test.values)
y_test_ordenado = y_test.values[indices_ordenados]
y_pred_prob_ordenado = y_test_pred_prob[indices_ordenados]

plt.subplot(1, 2, 1)
plt.scatter(range(len(y_test_ordenado)), y_test_ordenado,
           alpha=0.5, s=10, label='Real', color='blue')
plt.scatter(range(len(y_pred_prob_ordenado)), y_pred_prob_ordenado,
           alpha=0.5, s=10, label='Predicho', color='red')
plt.xlabel('Muestra (ordenada)')
plt.ylabel('Valor (0=No malware, 1=Malware)')
plt.title('Valores Reales vs Probabilidades Predichas')
plt.legend()
plt.grid(True, alpha=0.3)

# Histograma de errores
plt.subplot(1, 2, 2)
errores = y_test.values - y_test_pred_prob
plt.hist(errores, bins=50, alpha=0.7, color='green', edgecolor='black')
plt.xlabel('Error (Real - Predicho)')
plt.ylabel('Frecuencia')
plt.title('Distribución de Errores en Test')
plt.axvline(x=0, color='red', linestyle='--', label='Error cero')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------
# PASO 8: RESUMEN PARA EL REPORTE
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("6️⃣ RESUMEN PARA TU REPORTE DEL MÁSTER")
print("=" * 60)

print(f"""
📋 RESUMEN DE MÉTRICAS DE REGRESIÓN:

1. RENDIMIENTO DEL MODELO:
   • R² en Test: {r2_test:.4f} → El modelo explica el {r2_test*100:.1f}% de la variabilidad
   • Error promedio (MAE): {mae_test:.4f} → {mae_test*100:.1f}% de error en probabilidades
   • Error cuadrático (RMSE): {rmse_test:.4f}

2. GENERALIZACIÓN (Overfitting):
   • Diferencia R² (Train-Test): {r2_train - r2_test:+.4f}
   • {'⚠️  Posible overfitting' if (r2_train - r2_test) > 0.1 else '✅ Buena generalización'}

3. INTERPRETACIÓN PARA MALWARE:
   • El modelo predice probabilidades de malware entre 0 y 1
   • Con un MAE de {mae_test:.3f}, tiene un error promedio del {mae_test*100:.1f}%
   • Con un R² de {r2_test:.3f}, {'explica poca variabilidad' if r2_test < 0.1 else 'explica variabilidad moderada'}

4. RECOMENDACIONES:
   • {'Considerar reducir complejidad del modelo' if (r2_train - r2_test) > 0.1 else 'El modelo generaliza adecuadamente'}
   • {'Probar más variables o features' if r2_test < 0.1 else 'El modelo tiene poder predictivo aceptable'}
   • Para Windows Defender: Usar umbral de {0.6 if mae_test < 0.3 else 0.7} para alertas

📊 MÉTRICAS NUMÉRICAS PARA INCLUIR EN TABLAS:
   MAE_train = {mae_train:.4f}     MAE_test = {mae_test:.4f}
   MSE_train = {mse_train:.4f}     MSE_test = {mse_test:.4f}
   RMSE_train = {rmse_train:.4f}   RMSE_test = {rmse_test:.4f}
   R²_train = {r2_train:.4f}       R²_test = {r2_test:.4f}
""")

print("\n" + "=" * 60)
print("✅ ANÁLISIS DE MÉTRICAS DE REGRESIÓN COMPLETADO")
print("=" * 60)

# %% colab={"base_uri": "https://localhost:8080/"} id="HP9tnmd--MOO" outputId="c34c9926-7ee0-42de-ad6c-83a96ca540aa"
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# --- 1. Cargar los Datos (Asumiendo que ya están cargados en 'datos') ---
# Asegúrate de que tu archivo 'datos' esté cargado. Por ejemplo:
# datos = pd.read_csv('tu_archivo_de_datos.csv', low_memory=False)

# ----------------------------------------------------------------------
# --- 2. Preparación de Datos y Selección de Variables Numéricas ---
# ----------------------------------------------------------------------

# 1. Identificar el Target
TARGET = 'HasDetections'

# 2. Identificar Columnas Numéricas (¡Cambiado a 'datos'!)
# Se excluyen 'MachineIdentifier' y el TARGET.
numeric_cols = datos.select_dtypes(include=['number']).columns.tolist()
numeric_features = [col for col in numeric_cols if col != 'MachineIdentifier' and col != TARGET]

print(f"Número de características numéricas seleccionadas: {len(numeric_features)}")
print("Características Numéricas:")
print(numeric_features)

# 3. Manejo de Valores Perdidos (Imputación Simple)
# Usaremos la mediana para las características numéricas en el DataFrame 'datos'.
for col in numeric_features:
    datos[col].fillna(datos[col].median(), inplace=True)

# 4. Separar Features (X) y Target (y)
X = datos[numeric_features]
y = datos[TARGET]

# ----------------------------------------------------------------------
# --- 3. División de Datos ---
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nConjunto de Entrenamiento (Filas): {X_train.shape[0]}")
print(f"Conjunto de Prueba (Filas): {X_test.shape[0]}")

# ----------------------------------------------------------------------
# --- 4. Entrenamiento del Modelo Random Forest ---
# ----------------------------------------------------------------------

# Inicializar el clasificador Random Forest
# Usamos un max_depth bajo (10) y 100 árboles para un buen balance entre rendimiento y velocidad.
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    verbose=1 # Para ver el progreso del entrenamiento
)


print("\n--- Iniciando Entrenamiento de Random Forest (solo con variables numéricas) ---")
rf_model.fit(X_train, y_train)
print("--- Entrenamiento Finalizado ---")

# ----------------------------------------------------------------------
# --- 5. Evaluación del Modelo ---
# ----------------------------------------------------------------------

# 1. Realizar predicciones
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]

# 2. Reporte de Clasificación
print("\n### Reporte de Clasificación ###")
print(classification_report(y_test, y_pred))

# 3. AUC-ROC Score (Métrica clave)
auc_roc = roc_auc_score(y_test, y_proba)
print(f"AUC-ROC Score en el conjunto de prueba: **{auc_roc:.4f}**")

# 4. Matriz de Confusión
print("\n### Matriz de Confusión ###")
print(confusion_matrix(y_test, y_pred))

# ----------------------------------------------------------------------
# --- 6. Análisis Adicional: Importancia de las Características ---
# ----------------------------------------------------------------------

# Extraer la importancia de las características
feature_importances = pd.Series(
    rf_model.feature_importances_,
    index=numeric_features
).sort_values(ascending=False)

print("\n### Top 10 de Importancia de Características Numéricas ###")
print(feature_importances.head(10))

# %% colab={"base_uri": "https://localhost:8080/"} id="cfjGm3Ny-j3K" outputId="4398c216-849e-4bfc-800f-c36a97f744c7"
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Asumiendo que las siguientes variables están disponibles tras el entrenamiento anterior:
# rf_model (El modelo Random Forest entrenado)
# X_train, X_test, y_train, y_test

# --- 1. Obtener Predicciones de Probabilidad ---
# Usamos predict_proba para obtener la probabilidad de la clase positiva (detección=1).

# Predicciones de probabilidad para TRAIN
y_train_proba = rf_model.predict_proba(X_train)[:, 1]

# Predicciones de probabilidad para TEST
y_test_proba = rf_model.predict_proba(X_test)[:, 1]

print("--- Predicciones Generadas ---")
print(f"Probabilidades de TRAIN (primeros 5): {y_train_proba[:5]}")
print(f"Probabilidades de TEST (primeros 5): {y_test_proba[:5]}")

# ----------------------------------------------------------------------
# --- 2. Cálculo de Métricas de Regresión (usando las probabilidades) ---
# ----------------------------------------------------------------------
# Nota: Aquí se está tratando el TARGET binario (0 o 1) como si fuera un valor continuo
# y las probabilidades (0 a 1) como las predicciones de regresión.

results = {}

# --- A. Métricas para el conjunto de TRAIN ---
mae_train = mean_absolute_error(y_train, y_train_proba)
mse_train = mean_squared_error(y_train, y_train_proba)
r2_train = r2_score(y_train, y_train_proba)

results['TRAIN'] = {
    'MAE': mae_train,
    'MSE': mse_train,
    'R2': r2_train
}

# --- B. Métricas para el conjunto de TEST ---
mae_test = mean_absolute_error(y_test, y_test_proba)
mse_test = mean_squared_error(y_test, y_test_proba)
r2_test = r2_score(y_test, y_test_proba)

results['TEST'] = {
    'MAE': mae_test,
    'MSE': mse_test,
    'R2': r2_test
}

# ----------------------------------------------------------------------
# --- 3. Mostrar Resultados en una Tabla ---
# ----------------------------------------------------------------------

metrics_df = pd.DataFrame(results).T # Transponer para tener TRAIN y TEST como filas

print("\n### 📊 Resultados de Métricas de Regresión (Usando Probabilidades) ###")
print(metrics_df.to_markdown(floatfmt=".4f"))

# ----------------------------------------------------------------------
# --- 4. Interpretación Rápida ---
# ----------------------------------------------------------------------

print("\n--- Interpretación (Importante para Clasificación) ---")
print(f"La diferencia MAE (TRAIN - TEST): {mae_train - mae_test:.4f}")

if abs(mae_train - mae_test) > 0.05:
    print("⚠️ Posible Overfitting: El rendimiento en TRAIN es significativamente mejor que en TEST. Considera reducir la profundidad del árbol (`max_depth`) o usar regularización.")
else:
    print("✅ Balance Aceptable: El error es consistente entre los conjuntos de TRAIN y TEST.")
