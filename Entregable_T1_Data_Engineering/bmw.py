# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: penv
#     language: python
#     name: python3
# ---

# %% [markdown] id="1704791e"
# ## Dependencias
# # Ejecutar este comando para instalar las librerías necesarias
# ```
# penv/Scripts/activate
# pip install pandas numpy scikit-learn matplotlib seaborn jinja2
# ```

# %% executionInfo={"elapsed": 2423, "status": "ok", "timestamp": 1764867642695, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="f40eb448"
import pandas as pd # Librería para la manipulación y el análisis de datos
import numpy as np # Librería para la manipulación de datos y para la ejecución de operaciones matemáticas
import matplotlib.pyplot as plt # Librería para la visualización de datos
import seaborn as sns # Librería para la visualización de datos
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder, OneHotEncoder # Librería para crear modelos de ML
import sys
import os
import importlib

# Obtiene la ruta absoluta del directorio actual del script
# current_dir = os.path.dirname(os.path.abspath(__file__))  # python
current_dir = os.getcwd()                                   # jupyter notebook

# Obtiene la ruta al directorio 'src' (subiendo un nivel y entrando a src)
src_path = os.path.join(current_dir, '..', 'src')

# Lo añade al path de Python
sys.path.append(src_path)

# Ahora ya puedes importar
import utils
importlib.reload(utils)  # Forzar recarga si ya se había importado antes

# %% executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1764867642709, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="06c52e51"
# df_inicial = pd.read_csv("/content/bmw_pricing_v3.csv")

# %%
df_inicial = pd.read_csv("Datasets/bmw_pricing_v3.csv")

# %% colab={"base_uri": "https://localhost:8080/", "height": 245} executionInfo={"elapsed": 147, "status": "ok", "timestamp": 1764867642872, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="11f98c1a" outputId="9907d390-a1c8-4baa-e8b8-7c830105829d"
print("Shape:", df_inicial.shape)
df_inicial.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 648} executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1764867642890, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="0bec4738" outputId="99f5628c-e58b-490d-9cad-1c25d9bba44c"
df_inicial.isnull().sum()

# %% colab={"base_uri": "https://localhost:8080/", "height": 648} executionInfo={"elapsed": 40, "status": "ok", "timestamp": 1764867642964, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="701d2c60" outputId="f445e262-01c1-40bc-f53c-048ab0dd1626"
df_inicial.isnull().any()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 22, "status": "ok", "timestamp": 1764867642982, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="a92ccb8c" outputId="ede8783c-fbf3-4028-9710-578029166d3e"
df_inicial.info()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 32, "status": "ok", "timestamp": 1764867643017, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="b670a026" outputId="f19712b1-96c0-453c-f383-d96e3e98d62e"
# Conversión con pd.to_datetime de los atributos que tienen fecha y son float
df_inicial_test = df_inicial.copy()
for i in df_inicial.columns:
    if str(i).upper().startswith('FECHA'):
        df_inicial_test[i] = pd.to_datetime(df_inicial[i])

df_inicial_test.info()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 21, "status": "ok", "timestamp": 1764867643039, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="97be129d" outputId="0d171fee-c514-4c1f-fb31-86fa7c3d7dcf"
df_inicial_test[df_inicial_test.duplicated(keep='first')].shape

# %% [markdown]
# ---

# %%
df_inicial_test.isnull().sum()

# %%
for i in df_inicial_test.columns:
    prctj = df_inicial_test[i].isnull().mean() * 100
    print(f'{prctj:.2f}% \tde nulos en {i}')

# %%
df_inicial_test_2 = df_inicial_test.copy()

# Borramos nulos en precio porque es el target principal
df_inicial_test_2 = df_inicial_test_2.dropna(subset=['precio'])
# Borramos columnas que no aportan valor
df_inicial_test_2 = df_inicial_test_2.drop(columns=['marca','asientos_traseros_plegables'])

# %% [markdown]
# ---

# %%
# Rellenamos color desconocido
df_inicial_test_2 = df_inicial_test_2.fillna({'color': 'desconocido'})
# Corregimos los 'Diesel' a 'diesel' para homogeneizar
df_inicial_test_2['tipo_gasolina'] = df_inicial_test_2['tipo_gasolina'].str.lower()
# Corregimos el error de tipo_coche 'van' en modelo X3
df_inicial_test_2.loc[(df_inicial_test_2['modelo'] == 'X3') & (df_inicial_test_2['tipo_coche'] == 'van'),'tipo_coche'] = 'suv'

# Imputación por KNN para los nulos restantes
df_inicial_test_knn = df_inicial_test_2.copy()

target_equipamiento = ['aire_acondicionado', 'camara_trasera', 'elevalunas_electrico', 
                       'bluetooth', 'alerta_lim_velocidad', 'volante_regulable', 'gps']
target_estructural = ['modelo', 'tipo_coche', 'tipo_gasolina']
target_fechas = ['fecha_registro', 'fecha_venta'] # A estos les aplicaremos mediana

config_strict = {'potencia': 0.01, 'km': 0.10}

for col in target_equipamiento:
    df_inicial_test_knn = utils.imputar_por_similitud(
        df_inicial_test_knn, target_col=col, numeric_cols=config_strict,
        cat_bool_cols=['modelo', 'tipo_gasolina', 'tipo_coche'],
        umbral_dominancia=0.70, estrategia='moda' # MODA para equipamiento
    )

for col in target_estructural:
    preds = [c for c in target_estructural if c != col]
    df_inicial_test_knn = utils.imputar_por_similitud(
        df_inicial_test_knn, target_col=col, numeric_cols=config_strict,
        cat_bool_cols=preds, umbral_dominancia=0.70, estrategia='moda' # MODA para categóricas
    )

config_relaxed = {'potencia': 0.01, 'km': 0.30}

# Hacemos dos pasadas para estructurales para rellenar recíprocamente
for _ in range(2): 
    for col in target_estructural:
        preds = [c for c in target_estructural if c != col]
        df_inicial_test_knn = utils.imputar_por_similitud(
            df_inicial_test_knn, target_col=col, numeric_cols=config_relaxed,
            cat_bool_cols=preds, umbral_dominancia=0.50, estrategia='moda'
        )

for col in target_equipamiento:
    df_inicial_test_knn = utils.imputar_por_similitud(
        df_inicial_test_knn, target_col=col, numeric_cols=config_relaxed,
        cat_bool_cols=['modelo', 'tipo_coche'], umbral_dominancia=0.55, estrategia='moda'
    )

# Aquí cambiamos a MEDIANA
config_fechas = {'potencia': 0.10, 'km': 0.10}

for col in target_fechas:
    df_inicial_test_knn = utils.imputar_por_similitud(
        df_inicial_test_knn, target_col=col,
        numeric_cols=config_fechas,
        cat_bool_cols=['modelo', 'tipo_coche'],
        usar_percentiles=True,
        estrategia='mediana'
    )

df_inicial_test_knn_2 = df_inicial_test_knn.copy()

# Buscamos solo por KM (desgaste) y aplicamos mediana
config_fechas_fallback = {'km': 0.05} 

for col in target_fechas:
    df_inicial_test_knn = utils.imputar_por_similitud(
        df_inicial_test_knn, target_col=col,
        numeric_cols=config_fechas_fallback,
        cat_bool_cols=[], # Sin filtros categóricos
        usar_percentiles=True,
        estrategia='mediana'
    )

# Si quedan KM o Potencia vacíos, también deberíamos usar Mediana en una pasada final
if df_inicial_test_knn['km'].isnull().sum() > 0:
     df_inicial_test_knn = utils.imputar_por_similitud(
        df_inicial_test_knn, target_col='km', numeric_cols={'potencia': 0.10}, 
        cat_bool_cols=['modelo', 'tipo_coche'], estrategia='mediana'
    )

# Si quedan Potencia, Modelo o tipo_coche vacíos, también deberíamos usar Mediana en una pasada final
df_inicial_test_knn = utils.imputar_por_similitud(
            df_inicial_test_knn, target_col='modelo', numeric_cols=config_relaxed,
            cat_bool_cols=preds, umbral_dominancia=0.30, estrategia='moda',
            usar_percentiles=True
        )

df_inicial_test_knn = utils.imputar_por_similitud(
            df_inicial_test_knn, target_col='tipo_coche', numeric_cols={'precio': 1.0},
            cat_bool_cols=['modelo'], umbral_dominancia=0.40, estrategia='moda',
            usar_percentiles=True
        )

# Equipamientos nulos a False
df_inicial_test_knn[target_equipamiento] = df_inicial_test_knn[target_equipamiento].fillna(False)

# última pasada a potencia por modelo
df_inicial_test_knn = utils.imputar_por_similitud(
            df_inicial_test_knn, target_col='potencia', numeric_cols={'precio': 1.0},
            cat_bool_cols=['modelo'], umbral_dominancia=0.40, estrategia='moda',
            usar_percentiles=False
        )

# Convertimos las fechas a primero de mes
for col in df_inicial_test_knn.columns:
    if str(col).upper().startswith('FECHA'):
        df_inicial_test_knn[col] = pd.to_datetime(df_inicial_test_knn[col], errors='coerce')
        df_inicial_test_knn[col] = df_inicial_test_knn[col].dt.to_period('M').dt.to_timestamp()

# Filtramos filas con más de 1 nulo restante
df_limpio = df_inicial_test_knn[df_inicial_test_knn.isnull().sum(axis=1) <= 1].copy()

# Finalmente, revisamos nulos que no deberían quedar
display(utils.auditoria_nulos(df_limpio))

# %% [markdown]
# ---

# %%
df_limpio_2 = df_limpio.copy()

# %%
df_limpio['fecha_registro'].hist(bins=50)

# %%
df_limpio[df_limpio['fecha_registro'] < pd.Timestamp('1999-01-01')]

# %%
df_inicial_test[df_inicial_test['fecha_registro'] < pd.Timestamp('1999-01-01')]

# %%
df_limpio[df_limpio['fecha_registro'] > df_limpio['fecha_venta']]

# %%
# Quitamos fechas incoherentes
df_limpio_2 = df_limpio_2[df_limpio_2['fecha_registro'] <= df_limpio_2['fecha_venta']]

# Calculamos antigüedad
df_limpio_2["antigüedad"] = df_limpio_2["fecha_venta"] - df_limpio_2["fecha_registro"]
df_limpio_2['antigüedad'] = df_limpio_2['antigüedad'].dt.days / 365.25
df_limpio_2 = df_limpio_2.drop(columns=['fecha_venta','fecha_registro'])

# %%
df_limpio_2['antigüedad'].hist(bins=50)

# %%
df_limpio_2[df_limpio_2['antigüedad'] < 15.0]['antigüedad'].hist(bins=20)

# %%
df_limpio_2[df_limpio_2['antigüedad'] > 15.0]['antigüedad'].hist(bins=20)

# %%
df_limpio_2[df_limpio_2['antigüedad'] > 15.0]['modelo'].value_counts()

# %%
df_limpio_2['km'].hist(bins=50)

# %%
df_limpio_2[df_limpio_2['km'] < 5000.0]

# %%
df_limpio_2[df_limpio_2['modelo'] == '640 Gran Coupé']['km'].median()

# %%
df_limpio_2[df_limpio_2['km'] < 0.0]

# %%
df_limpio_2['km'].hist(bins=50)

# %%
df_limpio_2[df_limpio_2['km'] < 0.0]['modelo'].iloc[0]

# %%
df_test = df_limpio_2.copy()
df_test.loc[df_test['km'] <= 0, 'km'] = df_test[df_test['modelo'] == df_test[df_test['km'] < 0.0]['modelo'].iloc[0]]['km'].median()
df_test.iloc[[2928]]


# %%
df_limpio_2[df_limpio_2['km'] > 500000.0]

# %%
df_test = df_test[df_test['km'] < 500000.0]

# %%
df_test['km'].hist(bins=50)

# %%
df_test[df_test['km'] > 500000.0]

# %%
df_test['potencia'].hist(bins=50)

# %%
df_test[df_test['potencia'] < 50]

# %%
df_test.iloc[[1786, 1915, 3755]]

# %%
df_limpio_2[(df_limpio_2['modelo'] == 'i3') & (df_limpio_2['precio'].between(15000.0, 25000.0))]['potencia'].value_counts()

# %%
df_test.loc[df_test['potencia'] < 50, 'potencia'] = None

# %%
df_test[df_test['potencia'].isnull()]

# %%
df_test[(df_test['modelo'] == 'i3') & (df_test['precio'].between(15000.0, 25000.0))]['potencia'].value_counts()

# %%
df_test[(df_test['modelo'] == 'X1') & (df_test['precio'].between(10000.0, 13000.0)) & (df_test['antigüedad'].between(2.2, 4.6))]['potencia'].value_counts(normalize=True)

# %%
df_test = utils.imputar_por_similitud(
            df_test, target_col='potencia', numeric_cols={'antigüedad': 0.35, 'precio': 0.1},
            cat_bool_cols=['modelo'], umbral_dominancia=0.50, estrategia='moda',
            usar_percentiles=False
        )

df_test = utils.imputar_por_similitud(
            df_test, target_col='potencia', numeric_cols={'antigüedad': 0.35},
            cat_bool_cols=['modelo'], umbral_dominancia=0.50, estrategia='moda',
            usar_percentiles=False
        )

# %%
df_test.iloc[[1786, 1915, 3755]]

# %%
df_test['potencia'].hist(bins=30)

# %%
df_limpio_2[df_limpio_2['potencia'] > 300]

# %%
df_test[df_test['precio'] < 50000]['precio'].hist(bins=60)
# df_test['precio'].hist(bins=60)

# %%
df_test[df_test['precio'] > 50000]['precio'].hist(bins=60)

# %%
df_test[df_test['precio'] > 80000]

# %%
df_test = df_test[df_test['precio'] < 100000]

# %%
df_test[df_test['precio'] > 80000]

# %%
df_test[df_test['precio'] > 50000]['precio'].hist(bins=60)

# %%
df_test[df_test['modelo'].map(df_test['modelo'].value_counts()) < 2]

# %%
df_test[(df_test['modelo'].str.endswith('Active Tourer'))]

# %%
df_test[(df_test['modelo'].str.endswith('Active Tourer')) & (df_test['antigüedad'].between(2.5, 4.5)) & (df_test['potencia'] == 100.0)]#['modelo'].value_counts()

# %%
df_test.loc[[4820]]

# %%
df_test.loc[df_test['modelo'] == ' Active Tourer', 'modelo'] = '218 Active Tourer'
df_test.loc[[4820]]

# %%
df_test['color'].hist()

# %%
df_test['tipo_gasolina'].hist()

# %%
plt.figure(figsize=(14, 10))
df_test['tipo_coche'].value_counts().plot(kind='bar', color='green')
plt.title('Tipos de coche más comunes en BMW', fontsize=35)
plt.xlabel('Tipo de coche', fontsize=30)
plt.ylabel('Cantidad de coches', fontsize=30)
plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.show()

# %%
for col in target_equipamiento:
    plt.figure()  # Crea un lienzo nuevo en cada vuelta
    df_test[col].value_counts().plot(kind='bar', title=f'Distribución de {col}')
    plt.show()    # Fuerza a que se pinte ahora mismo

# %% [markdown]
# Tiene más probabilidad de tener gps que un elevalunas eléctrico

# %% [markdown]
# ---

# %% [markdown]
# Correlaciones por tipo de dato

# %%
df_corr = df_test.copy()
for col in target_equipamiento:
    if col in df_corr.columns:
        # Convertimos 1.0 -> True, 0.0 -> False
        df_corr[col] = df_corr[col].astype(bool)

# %%
corr_bool = df_corr[target_equipamiento].corr(numeric_only = True)
corr_bool.style.background_gradient(cmap='cividis')

# %%
cols_numericas = ['precio', 'km', 'potencia', 'antigüedad']
corr_num = df_corr[cols_numericas].corr(numeric_only = True)
corr_num.style.background_gradient(cmap='cividis')

# %%
# 1. Creamos una copia para no dañar el dataset original
df_cat_numerica = df_corr[target_estructural].copy()

# 2. Convertimos texto a números (Factorize)
# Esto asigna 0, 1, 2... a cada categoría única
for col in target_estructural:
    df_cat_numerica[col] = pd.factorize(df_cat_numerica[col])[0]

# 3. Ahora sí podemos hacer la correlación (Usamos Spearman porque no son lineales)
corr_cat = df_cat_numerica.corr(method='spearman')

# 4. Visualización
plt.figure(figsize=(6, 5))
sns.heatmap(corr_cat, annot=True, cmap='cividis', fmt=".2f")
plt.title('Correlación entre Categóricas (Codificación Temporal)')
plt.show()

# %%
import scipy.stats as ss

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# Calcular la matriz
cols = target_estructural # ['modelo', 'tipo_coche', 'tipo_gasolina']
cramer_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

for col1 in cols:
    for col2 in cols:
        cramer_matrix.loc[col1, col2] = cramers_v(df_test[col1], df_test[col2])

# Visualizar
plt.figure(figsize=(6, 5))
sns.heatmap(cramer_matrix, annot=True, cmap='cividis', vmin=0, vmax=1)
plt.title('Asociación (V de Cramér) entre Categóricas')
plt.show()

# %% [markdown]
# ---

# %% [markdown]
# Correlación con target

# %%
df_analisis = df_test.copy() # O df_inicial_test_knn, el que estés usando

# Variables numéricas continuas
nums = ['km', 'potencia', 'antigüedad'] 

plt.figure(figsize=(15, 5))

for i, col in enumerate(nums):
    plt.figure()
    # Usamos regplot para ver la línea de tendencia automáticamente
    sns.regplot(x=col, y='precio', data=df_analisis, 
                scatter_kws={'alpha':0.3, 's':10}, line_kws={'color':'red'})
    plt.title(f'{col} vs Precio')
    plt.grid(True, alpha=0.3)
    plt.axis([0, max(df_analisis[col]), 0, max(df_analisis['precio'])])
    plt.show()

plt.tight_layout()
plt.show()

# %%
nums = ['km', 'potencia', 'antigüedad'] 

for col in nums:
    plt.figure(figsize=(8, 5))
    data_plot = df_analisis[df_analisis[col] > 0]
    sns.regplot(x=col, y='precio', data=data_plot, 
                logx=True,
                scatter_kws={'alpha':0.3, 's':10}, 
                line_kws={'color':'red'})
    plt.title(f'{col} vs Precio (Tendencia Logarítmica)')
    plt.grid(True, alpha=0.3)
    plt.show()

# %%
# Lista de equipamiento
cols_equip = target_equipamiento # ['aire_acondicionado', 'gps', etc.]

# Configuración del grid para pintar muchos gráficos
n_cols = 3
n_rows = (len(cols_equip) + n_cols - 1) // n_cols

plt.figure(figsize=(15, n_rows * 4))

for i, col in enumerate(cols_equip):
    plt.figure()
    # Boxplot: Compara la distribución de precios si TIENE o NO TIENE el extra
    sns.boxplot(x=col, y='precio', data=df_analisis, palette='Set2')
    plt.title(f'Impacto de {col}')
    plt.xlabel('') # Quitamos etiqueta x para limpiar
    plt.show()
    
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(8, 20))
top_n = df_analisis['modelo'].value_counts().index
data_to_plot = df_analisis[df_analisis['modelo'].isin(top_n)]
order = top_n
titulo = f'{'modelo'} vs Precio'
sns.violinplot(y='modelo', x='precio', data=data_to_plot, order=order, palette='Set2')
plt.title(titulo)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.show()

# %%
cols_cat = ['tipo_gasolina', 'tipo_coche', 'color']

for col in cols_cat:
    plt.figure(figsize=(12, 5))
    data_to_plot = df_analisis
    order = df_analisis[col].value_counts().index # Ordenar por frecuencia
    titulo = f'{col} vs Precio'
    sns.boxplot(x=col, y='precio', data=data_to_plot, order=order, palette='viridis')
    plt.title(titulo)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.show()

# %%
plt.figure(figsize=(10, 20))

# Usamos el mismo orden de precio
orden_precio = df_analisis.groupby('modelo')['precio'].median().sort_values(ascending=False).index

sns.stripplot(x='precio', y='modelo', data=df_analisis, 
              order=orden_precio, color='blue', 
              size=3, alpha=0.4, jitter=True)

plt.title('Precio Real de cada coche por Modelo', fontsize=15)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

# %% [markdown]
# ---

# %% [markdown]
# OHE para color, tipo_gasolina, tipo_coche y modelo

# %%
df_encoded = df_analisis.copy()

categories_ohe = ['tipo_coche', 'tipo_gasolina', 'color', 'modelo']
ohe_feature_names = []

for col in categories_ohe:
    df_encoded[col] = df_encoded[col].astype(str)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=bool)
    encoded_cols = ohe.fit_transform(df_encoded[[col]])
    current_col_names = [f"{col}_{cat}" for cat in ohe.categories_[0]]
    ohe_feature_names.extend(current_col_names)
    encoded_df = pd.DataFrame(encoded_cols, columns=[f"{col}_{cat}" for cat in ohe.categories_[0]], index=df_encoded.index)
    df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
    df_encoded_2 = df_encoded.drop(columns=[col]).copy()

df_encoded_2.shape

# %%
ohe_feature_names

# %%
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

def evaluar_modelo(X, y):
    """Función auxiliar para entrenar y evaluar rápidamente con Validación Cruzada"""
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    # Usamos cross_val_score para ser más honestos con el resultado (R2 score negativo = mal modelo)
    scores = cross_val_score(model, X, y, cv=3, scoring='r2')
    return scores.mean()

y = df_encoded_2['precio']
X = df_encoded_2[ohe_feature_names + ['km', 'potencia', 'antigüedad']]

resultado_r2 = evaluar_modelo(X, y)
print(f"R2 promedio del modelo: {resultado_r2:.4f}")

# %% [markdown]
# ---

# %% [markdown]
# Aplicamos MinMaxScaler en las variables numéricas

# %%
df_scaled = df_encoded_2.copy()

scaler = MinMaxScaler()
df_scaled[nums] = scaler.fit_transform(df_scaled[nums])
df_scaled[nums].describe()

# %%
df_scaled.columns.shape

# %%
corr_final = df_scaled.corr(numeric_only=True)

plt.figure(figsize=(500, 400)) 

sns.heatmap(corr_final, 
            annot=True,        # Escribe el número dentro del cuadro
            fmt=".2f",         # Solo 2 decimales para que ocupe menos
            cmap='RdBu_r',     # Red-Blue reverse (Rojo=Positivo, Azul=Negativo)
            center=0,          # IMPORTANTE: Esto fuerza al 0 a ser blanco
            vmin=-1, vmax=1,   # Límites para que los colores sean simétricos
            square=True,       # Cuadraditos perfectos
            linewidths=.5)     # Líneas blancas entre celdas para limpieza

plt.title('Matriz de Correlación', fontsize=10)
plt.show()

# %%
# 1. Configuración
target = 'precio'
umbral = 0.1  # Tu parámetro: Mínima correlación para aparecer en el mapa
# 2. Calcular la correlación de TODAS las variables contra el PRECIO
# Usamos abs() porque nos interesa la magnitud, no el signo (-0.8 es muy importante)
corrs_con_target = df_scaled.corr(numeric_only=True)[target]
features_importantes = corrs_con_target[abs(corrs_con_target) > umbral].index
# 3. Crear una nueva matriz de correlación SOLO con esas columnas seleccionadas
corr_final_filtrada = df_scaled[features_importantes].corr(numeric_only=True)
# 4. Graficar
# Ajustamos el tamaño dinámicamente según cuántas variables hayan sobrevivido
n_vars = len(features_importantes)
plt.figure(figsize=(n_vars * 0.8, n_vars * 0.7)) # Tamaño automático razonable
sns.heatmap(corr_final_filtrada,
            annot=True,
            fmt=".2f",
            cmap='RdBu_r',
            center=0,
            vmin=-1, vmax=1,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .8}) # Hace la barra de color un poco más pequeña
plt.title(f'Variables con correlación > {umbral} con {target}', fontsize=12)
plt.show()

# %%
# 1. Configuración
target = 'precio'
umbral = 0.2

# 2. Calcular la correlación de TODAS las variables contra el PRECIO
corrs_con_target = df_scaled.corr(numeric_only=True)[target]

# --- MODIFICACIÓN: EXCLUIR PRIMERAS 14 COLUMNAS ---
# Identificamos los nombres de las columnas 0 a 13
columnas_a_ignorar = df_scaled.columns[:14]

# Filtramos: Nos quedamos con la correlación SOLO si:
# A) La columna NO está en la lista de ignoradas  O
# B) La columna es el propio 'target' (para que salga en la gráfica)
corrs_con_target = corrs_con_target[
    (~corrs_con_target.index.isin(columnas_a_ignorar)) | 
    (corrs_con_target.index == target)
]
# --------------------------------------------------

# Aplicamos el filtro del umbral (Features importantes)
features_importantes = corrs_con_target[abs(corrs_con_target) > umbral].index

# 3. Crear una nueva matriz de correlación SOLO con esas columnas seleccionadas
corr_final_filtrada = df_scaled[features_importantes].corr(numeric_only=True)

# 4. Graficar
n_vars = len(features_importantes)

# Si tras filtrar quedan muy pocas variables, aseguramos un tamaño mínimo de gráfica
figsize_w = max(6, n_vars * 0.8)
figsize_h = max(5, n_vars * 0.7)
plt.figure(figsize=(figsize_w, figsize_h)) 

sns.heatmap(corr_final_filtrada, 
            annot=True, 
            fmt=".2f", 
            cmap='RdBu_r', 
            center=0, 
            vmin=-1, vmax=1, 
            square=True, 
            linewidths=.5,
            cbar_kws={"shrink": .8})

plt.title(f'Variables (excluyendo primeras 14) con correlación > {umbral}', fontsize=12)
plt.show()

# %% [markdown]
# ---

# %%
df_scaled.info(True)

# %%
df_excel = df_test.head(50).copy()

# %%
df_excel.to_excel("bmw.xlsx", index=False)

# %% [markdown]
# ---
