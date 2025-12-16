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
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder # Librería para crear modelos de ML
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
df_limpio_2[df_limpio_2['km'] < 5000.0]

# %%
df_limpio_2[df_limpio_2['modelo'] == '640 Gran Coupé']['km'].median()

# %%
df_limpio_2[df_limpio_2['km'] < 0.0]

# %%
df_limpio_2[df_limpio_2['km'] < 0.0]['modelo'].iloc[0]

# %%
df_test = df_limpio_2.copy()
df_test.loc[df_test['km'] <= 0, 'km'] = df_test[df_test['modelo'] == df_test[df_test['km'] < 0.0]['modelo'].iloc[0]]['km'].median()
df_test.iloc[[2928]]


# %%
df_limpio_2[df_limpio_2['potencia'] < 50]

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
for col in target_equipamiento:
    plt.figure()  # Crea un lienzo nuevo en cada vuelta
    df_test[col].value_counts().plot(kind='bar', title=f'Distribución de {col}')
    plt.show()    # Fuerza a que se pinte ahora mismo

# %% [markdown]
# Tiene más probabilidad de tener gps que un elevalunas eléctrico

# %% [markdown]
# ---

# %%

# %% [markdown]
# ---

# %%
# !pip install category_encoders
# import category_encoders as ce

# %%
import sys
import os

# Esta es la ruta que apareció en tu error como "Requirement satisfied"
path_libreria = r"c:\users\j\desktop\joaquín\clase\master\vsc\penv\lib\site-packages"

if path_libreria not in sys.path:
    sys.path.append(path_libreria)

# Ahora intenta importar
import category_encoders as ce

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor # Usamos un modelo robusto y genérico
from category_encoders import CatBoostEncoder

def comparar_encodings(df, col_cat, col_target, umbral_frecuencia=0.9):
    """
    Compara 3 estrategias de encoding:
    1. Vanilla One-Hot (Todas las categorías).
    2. Top X% One-Hot (Agrupa categorías raras como 'Otros').
    3. CatBoost Encoding (Encoding basado en el target).
    
    Args:
        df: Dataframe original.
        col_cat: Nombre de la columna categórica a codificar (ej: 'modelos').
        col_target: Nombre de la columna objetivo (target) para entrenar (ej: 'precio').
        umbral_frecuencia: Porcentaje (0.0 a 1.0) para el corte de frecuencia (def: 0.9).
    """
    
    # 1. Preparación de datos (Evitamos modificar el df original fuera de la función)
    df_temp = df.copy()
    
    # Aseguramos que no haya nulos para el ejemplo
    df_temp = df_temp.dropna(subset=[col_cat, col_target])
    
    # Separamos X (solo la columna a probar) e y
    X = df_temp[[col_cat]]
    y = df_temp[col_target]
    
    results = {}
    
    # ---------------------------------------------------------
    # MÉTODO 1: Vanilla One-Hot Encoding
    # ---------------------------------------------------------
    print(f"--- Procesando Método 1: Vanilla One-Hot ---")
    X_vanilla = pd.get_dummies(X[col_cat], prefix='OHE', drop_first=True)
    
    # Evaluamos
    score_vanilla = evaluar_modelo(X_vanilla, y)
    results['Vanilla OHE'] = {'R2_Score': score_vanilla, 'Num_Columnas': X_vanilla.shape[1]}
    
    # ---------------------------------------------------------
    # MÉTODO 2: One-Hot Top X% (Agrupando 'Otros')
    # ---------------------------------------------------------
    print(f"--- Procesando Método 2: Top {umbral_frecuencia*100}% One-Hot ---")
    
    # Calculamos frecuencias
    counts = df_temp[col_cat].value_counts(normalize=True)
    cumulative = counts.cumsum()
    
    # Filtramos las categorías que entran en el % acumulado
    top_categories = cumulative[cumulative < umbral_frecuencia].index
    
    # Aplicamos la transformación "Otros"
    X_top = X.copy()
    X_top['new_cat'] = X_top[col_cat].apply(lambda x: x if x in top_categories else 'Otros')
    
    # One Hot sobre la nueva columna reducida
    X_top_ohe = pd.get_dummies(X_top['new_cat'], prefix='Top_OHE', drop_first=True)
    
    # Evaluamos
    score_top = evaluar_modelo(X_top_ohe, y)
    results[f'Top {int(umbral_frecuencia*100)}% OHE'] = {'R2_Score': score_top, 'Num_Columnas': X_top_ohe.shape[1]}

    # ---------------------------------------------------------
    # MÉTODO 3: CatBoost Encoding
    # ---------------------------------------------------------
    print(f"--- Procesando Método 3: CatBoost Encoding ---")
    
    # El encoder de CatBoost necesita ver el target
    cbe = CatBoostEncoder()
    # Importante: CatBoostEncoder suele requerir fit_transform con y
    X_cbe = cbe.fit_transform(X[col_cat], y)
    
    # Evaluamos
    score_cbe = evaluar_modelo(X_cbe, y)
    results['CatBoost Enc'] = {'R2_Score': score_cbe, 'Num_Columnas': X_cbe.shape[1]}
    
    # ---------------------------------------------------------
    # Visualización de Resultados
    # ---------------------------------------------------------
    df_res = pd.DataFrame(results).T
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Gráfico de Barras para el Score (Rendimiento)
    color = 'tab:blue'
    ax1.set_xlabel('Método de Encoding')
    ax1.set_ylabel('R2 Score (Mayor es mejor)', color=color)
    bars = ax1.bar(df_res.index, df_res['R2_Score'], color=color, alpha=0.6, label='Precision Modelo')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1) # Asumiendo R2, ajusta si usas otra métrica

    # Eje secundario para el número de columnas (Dimensionalidad)
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Número de Columnas Generadas', color=color)
    line = ax2.plot(df_res.index, df_res['Num_Columnas'], color=color, marker='o', linewidth=2, linestyle='--', label='Dimensionalidad')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Comparativa de Encodings: Rendimiento vs Dimensionalidad (Umbral: {umbral_frecuencia})')
    fig.tight_layout()
    plt.show()
    
    return df_res

def evaluar_modelo(X, y):
    """Función auxiliar para entrenar y evaluar rápidamente con Validación Cruzada"""
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    # Usamos cross_val_score para ser más honestos con el resultado (R2 score negativo = mal modelo)
    scores = cross_val_score(model, X, y, cv=3, scoring='r2')
    return scores.mean()


# %%
df_resultados = comparar_encodings(df_inicial_test, 'modelo', 'precio', umbral_frecuencia=0.80)
print(df_resultados)

# %%
df_resultados = comparar_encodings(df_inicial_test, 'modelo', 'precio', umbral_frecuencia=0.90)
print(df_resultados)


# %%
def comparar_encodings_v2(df, col_cat, col_target, umbral_frecuencia=0.9):
    """
    Compara 3 estrategias de encoding:
    1. Vanilla One-Hot (Todas las categorías).
    2. Top X% One-Hot (Agrupa categorías raras como 'Otros').
    3. CatBoost Encoding (Encoding basado en el target).
    
    Args:
        df: Dataframe original.
        col_cat: Nombre de la columna categórica a codificar (ej: 'modelos').
        col_target: Nombre de la columna objetivo (target) para entrenar (ej: 'precio').
        umbral_frecuencia: Porcentaje (0.0 a 1.0) para el corte de frecuencia (def: 0.9).
    """
    
    # 1. Preparación de datos (Evitamos modificar el df original fuera de la función)
    df_temp = df.copy()
    
    # Aseguramos que no haya nulos para el ejemplo
    df_temp = df_temp.dropna(subset=[col_cat, col_target])
    
    # Separamos X (solo la columna a probar) e y
    X = df_temp[[col_cat]]
    y = df_temp[col_target]
    
    results = {}
    
    # ---------------------------------------------------------
    # MÉTODO 1: Vanilla One-Hot Encoding
    # ---------------------------------------------------------
    print(f"--- Procesando Método 1: Vanilla One-Hot ---")
    X_vanilla = pd.get_dummies(X[col_cat], prefix='OHE', drop_first=True)
    
    # Evaluamos
    score_vanilla = evaluar_modelo(X_vanilla, y)
    results['Vanilla OHE'] = {'R2_Score': score_vanilla, 'Num_Columnas': X_vanilla.shape[1]}
    
    # ---------------------------------------------------------
    # MÉTODO 2: One-Hot Top X% (Lógica Mejorada)
    # ---------------------------------------------------------
    print(f"--- Procesando Método 2: Top {umbral_frecuencia*100}% One-Hot ---")
    
    counts = df_temp[col_cat].value_counts(normalize=True)
    cumulative = counts.cumsum()
    
    top_categories = cumulative[cumulative <= (umbral_frecuencia + 0.05)].index 
    
    # Si la lista queda vacía (porque el primero ya supera el umbral), cogemos el top 1 al menos
    if len(top_categories) == 0:
        top_categories = counts.index[:1]

    print(f"   -> Categorías mantenidas: {len(top_categories)} | Categorías en 'Otros': {len(counts) - len(top_categories)}")

    X_top = X.copy()
    X_top['new_cat'] = X_top[col_cat].apply(lambda x: x if x in top_categories else 'Otros')
    
    X_top_ohe = pd.get_dummies(X_top['new_cat'], prefix='Top_OHE', drop_first=True)
    
    score_top = evaluar_modelo(X_top_ohe, y)
    results[f'Top {int(umbral_frecuencia*100)}% OHE'] = {'R2_Score': score_top, 'Num_Columnas': X_top_ohe.shape[1]}
    # ---------------------------------------------------------
    # MÉTODO 3: CatBoost Encoding
    # ---------------------------------------------------------
    print(f"--- Procesando Método 3: CatBoost Encoding ---")
    
    # El encoder de CatBoost necesita ver el target
    cbe = CatBoostEncoder()
    # Importante: CatBoostEncoder suele requerir fit_transform con y
    X_cbe = cbe.fit_transform(X[col_cat], y)
    
    # Evaluamos
    score_cbe = evaluar_modelo(X_cbe, y)
    results['CatBoost Enc'] = {'R2_Score': score_cbe, 'Num_Columnas': X_cbe.shape[1]}
    
    # ---------------------------------------------------------
    # Visualización de Resultados
    # ---------------------------------------------------------
    df_res = pd.DataFrame(results).T
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Gráfico de Barras para el Score (Rendimiento)
    color = 'tab:blue'
    ax1.set_xlabel('Método de Encoding')
    ax1.set_ylabel('R2 Score (Mayor es mejor)', color=color)
    bars = ax1.bar(df_res.index, df_res['R2_Score'], color=color, alpha=0.6, label='Precision Modelo')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1) # Asumiendo R2, ajusta si usas otra métrica

    # Eje secundario para el número de columnas (Dimensionalidad)
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Número de Columnas Generadas', color=color)
    line = ax2.plot(df_res.index, df_res['Num_Columnas'], color=color, marker='o', linewidth=2, linestyle='--', label='Dimensionalidad')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Comparativa de Encodings: Rendimiento vs Dimensionalidad (Umbral: {umbral_frecuencia})')
    fig.tight_layout()
    plt.show()
    
    return df_res



# %%
df_resultados = comparar_encodings_v2(df_inicial_test, 'modelo', 'precio', umbral_frecuencia=0.80)
print(df_resultados)
df_resultados = comparar_encodings_v2(df_inicial_test, 'modelo', 'precio', umbral_frecuencia=0.90)
print(df_resultados)


# %%
def comparar_encodings_completo(df, col_cat, col_target, cols_extra=[], umbral_frecuencia=0.9):
    """
    Compara encodings incluyendo variables numéricas adicionales para ver el impacto real.
    Args:
        cols_extra: Lista de nombres de columnas numéricas (ej: ['km', 'ano', 'cv'])
    """
    
    # 1. Preparación
    # Seleccionamos todas las columnas necesarias
    cols_to_use = [col_cat, col_target] + cols_extra
    df_temp = df[cols_to_use].copy()
    
    # Eliminamos nulos en cualquiera de las columnas seleccionadas
    df_temp = df_temp.dropna()
    
    # Separamos Target
    y = df_temp[col_target]
    
    # Separamos Bloque Categórico y Bloque Numérico (Extra)
    X_cat = df_temp[[col_cat]]
    X_num = df_temp[cols_extra] # Esto es un DataFrame con las numéricas
    
    results = {}
    
    # ---------------------------------------------------------
    # MÉTODO 1: Vanilla One-Hot + Numéricas
    # ---------------------------------------------------------
    print(f"--- Procesando Método 1: Vanilla One-Hot + {len(cols_extra)} vars extra ---")
    # Codificamos solo la categórica
    X_cat_vanilla = pd.get_dummies(X_cat[col_cat], prefix='OHE', drop_first=True)
    # Concatenamos con las numéricas (Index align es automático)
    X_final_vanilla = pd.concat([X_num, X_cat_vanilla], axis=1)
    
    score_vanilla = evaluar_modelo(X_final_vanilla, y)
    results['Vanilla OHE'] = {'R2_Score': score_vanilla, 'Num_Columnas': X_final_vanilla.shape[1]}
    
    # ---------------------------------------------------------
    # MÉTODO 2: Top X% + Numéricas (Lógica Inclusiva)
    # ---------------------------------------------------------
    print(f"--- Procesando Método 2: Top {umbral_frecuencia*100}% + {len(cols_extra)} vars extra ---")
    
    counts = df_temp[col_cat].value_counts(normalize=True)
    cumulative = counts.cumsum()
    
    # Lógica inclusiva (Soft Cut)
    top_categories = cumulative[cumulative <= (umbral_frecuencia + 0.05)].index 
    if len(top_categories) == 0: top_categories = counts.index[:1]

    X_cat_top = X_cat.copy()
    X_cat_top['new_cat'] = X_cat_top[col_cat].apply(lambda x: x if x in top_categories else 'Otros')
    
    X_cat_top_ohe = pd.get_dummies(X_cat_top['new_cat'], prefix='Top_OHE', drop_first=True)
    
    # Concatenamos
    X_final_top = pd.concat([X_num, X_cat_top_ohe], axis=1)
    
    score_top = evaluar_modelo(X_final_top, y)
    results[f'Top {int(umbral_frecuencia*100)}% OHE'] = {'R2_Score': score_top, 'Num_Columnas': X_final_top.shape[1]}

    # ---------------------------------------------------------
    # MÉTODO 3: CatBoost Encoding + Numéricas
    # ---------------------------------------------------------
    print(f"--- Procesando Método 3: CatBoost Encoding + {len(cols_extra)} vars extra ---")
    
    cbe = CatBoostEncoder()
    # Transformamos SOLO la columna categórica
    X_cat_cbe = cbe.fit_transform(X_cat[col_cat], y)
    
    # Concatenamos con las numéricas
    X_final_cbe = pd.concat([X_num, X_cat_cbe], axis=1)
    
    score_cbe = evaluar_modelo(X_final_cbe, y)
    results['CatBoost Enc'] = {'R2_Score': score_cbe, 'Num_Columnas': X_final_cbe.shape[1]}
    
    # ---------------------------------------------------------
    # Visualización
    # ---------------------------------------------------------
    df_res = pd.DataFrame(results).T
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Método de Encoding')
    ax1.set_ylabel('R2 Score (Con variables extra)', color=color)
    bars = ax1.bar(df_res.index, df_res['R2_Score'], color=color, alpha=0.6)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Ajustamos escala dinámica para ver diferencias pequeñas si el R2 es alto
    min_score = df_res['R2_Score'].min()
    max_score = df_res['R2_Score'].max()
    ax1.set_ylim(min_score - 0.05, max_score + 0.05)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Total Columnas (Features)', color=color)
    line = ax2.plot(df_res.index, df_res['Num_Columnas'], color=color, marker='o', linewidth=2, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Impacto Real del Encoding con Variables de Ayuda ({", ".join(cols_extra)})')
    fig.tight_layout()
    plt.show()
    
    return df_res

# --- EJEMPLO DE USO ---
# Define aquí tus columnas numéricas reales
columnas_numericas = ['km', 'potencia'] # <--- CAMBIA ESTO si tus columnas se llaman diferente

# Ejecutamos la comparación "Realista"
df_res_completo = comparar_encodings_completo(
    df_inicial_test, 
    'modelo', 
    'precio', 
    cols_extra=columnas_numericas, # Pasamos las columnas extra
    umbral_frecuencia=0.90
)

print(df_res_completo)

# %%
df_res_completo = comparar_encodings_completo(
    df_inicial_test, 
    'color', 
    'precio', 
    cols_extra=columnas_numericas, # Pasamos las columnas extra
    umbral_frecuencia=0.90
)

print(df_res_completo)

# %%
frecuencias = df_inicial_test['modelo'].value_counts(normalize=True, dropna=False)
acumulado = frecuencias.cumsum()
display(acumulado.head(20)*100)

display(df_inicial_test['modelo'].value_counts(dropna=False).head(20))

# %%
display(df_inicial_test[df_inicial_test['modelo'].isnull()])

# %%
df_inicial_test[(df_inicial_test['potencia'].between(100.0, 110.0)) & (df_inicial_test['tipo_coche'] == 'suv') & (df_inicial_test['precio'].between(13000.0, 15000.0)) & (df_inicial_test['km'].between(100000.0, 150000.0))]['modelo'].value_counts()

# %%
df_inicial_test[(df_inicial_test['potencia'].between(105.0, 110.0)) & (df_inicial_test['precio'].between(13000.0, 15000.0)) & (df_inicial_test['km'].between(100000.0, 150000.0))]['modelo'].value_counts()

# %%
# display(df_inicial_test_2[df_inicial_test_2['modelo'].isnull()])
# # display(df_inicial_test_2[(df_inicial_test_2['modelo'].isnull()) & (df_inicial_test_2['tipo_coche'].isnull())])
# display(df_inicial_test_2[(df_inicial_test_2['potencia'].between(105.0, 110.0)) & (df_inicial_test_2['precio'].between(13000.0, 15000.0)) & (df_inicial_test_2['km'].between(100000.0, 150000.0))]['modelo'].value_counts())
# display(df_inicial_test_2[(df_inicial_test_2['potencia'].between(100.0, 110.0)) & (df_inicial_test_2['tipo_coche'] == 'suv') & (df_inicial_test_2['precio'].between(13000.0, 15000.0)) & (df_inicial_test_2['km'].between(100000.0, 150000.0))]['modelo'].value_counts())

# %%
# df_inicial_test_knn_2.iloc[[172, 4760, 4796]]

# %%
df_inicial_test[(df_inicial_test['potencia'].between(105.0, 110.0)) & (df_inicial_test['km'].between(100000.0, 150000.0))]['modelo'].value_counts(normalize=True)
df_inicial_test[(df_inicial_test['potencia'].between(105.0, 110.0)) & (df_inicial_test['km'].between(100000.0, 150000.0))]['modelo'].mode()[0]

# %%
df_inicial_test[(df_inicial_test['potencia'].between(105.0, 110.0)) & (df_inicial_test['precio'] < 1000.0) & (df_inicial_test['km'].between(100000.0, 150000.0))]['modelo'].value_counts()

# %%
display(df_inicial_test[df_inicial_test['potencia'].isnull()])
# df_inicial_test[(df_inicial_test['modelo']== '525') & (df_inicial_test['km'].between(100000.0, 110000.0)) & (df_inicial_test['tipo_gasolina'] == diesel) & (df_inicial_test['tipo_coche']=='estate') & (df_inicial_test['precio'].between(15000.0, 20000.0))]['potencia'].value_counts()

# %%
df_inicial_test[(df_inicial_test['tipo_coche']== 'estate') & (df_inicial_test['modelo']== '525') & (df_inicial_test['precio'].between(15000.0, 20000.0)) & (df_inicial_test['km'].between(100000.0, 150000.0))]['potencia'].value_counts()

# %%
# df_limpio.iloc[[181]]

# %%
display(df_inicial_test[df_inicial_test['camara_trasera'].isnull()])
display(df_inicial_test[(df_inicial_test['modelo']== '520') & (df_inicial_test['km'].between(150000.0, 180000.0)) & (df_inicial_test['potencia'].between(125.0, 145.0)) & (df_inicial_test['tipo_gasolina'] == 'diesel') & (df_inicial_test['tipo_coche']=='estate') & (df_inicial_test['precio'].between(10000.0, 15000.0))]['camara_trasera'].value_counts())
display(df_inicial_test[(df_inicial_test['modelo']== '316') & (df_inicial_test['km'].between(150000.0, 180000.0)) & (df_inicial_test['potencia'].between(75.0, 95.0)) & (df_inicial_test['tipo_gasolina'] == 'diesel') & (df_inicial_test['tipo_coche']=='estate') & (df_inicial_test['precio'].between(10000.0, 15000.0))]['camara_trasera'].value_counts())

# %%
# df_limpio.iloc[[291, 409]]

# %%
display(df_inicial_test[df_inicial_test['elevalunas_electrico'].isnull()])
display(df_inicial_test[(df_inicial_test['modelo']== '640') & (df_inicial_test['km'].between(60000.0, 140000.0)) & (df_inicial_test['potencia'].between(220.0, 240.0)) & (df_inicial_test['tipo_gasolina'] == 'diesel') & (df_inicial_test['tipo_coche']=='coupe') & (df_inicial_test['precio'].between(30000.0, 405000.0))]['elevalunas_electrico'].value_counts())
display(df_inicial_test[(df_inicial_test['modelo']== '535') & (df_inicial_test['km'].between(100000.0, 190000.0)) & (df_inicial_test['potencia'].between(220.0, 240.0)) & (df_inicial_test['tipo_gasolina'] == 'diesel') & (df_inicial_test['tipo_coche']=='estate') & (df_inicial_test['precio'].between(20000.0, 35000.0))]['elevalunas_electrico'].value_counts())

# %%
# df_limpio.iloc[[63, 305]]

# %%
display(df_inicial_test[df_inicial_test['km'].isnull()])

# %%
display(df_inicial_test[df_inicial_test['tipo_gasolina'].isnull()])

# %%
display(df_inicial_test[df_inicial_test['volante_regulable'].isnull()])

# %%
frecuencias = df_inicial_test['tipo_coche'].value_counts(normalize=True)
acumulado = frecuencias.cumsum()
print(acumulado.head(20)*100)

display(df_inicial_test['tipo_coche'].value_counts().head(20))

# %% [markdown] id="8da2b150"
# No hay duplicados

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 20, "status": "ok", "timestamp": 1764867643053, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="1d785c55" outputId="a4f2637c-f12a-430b-ad81-2552e641b5c7"
# Porcentaje de nulos
for i in df_inicial_test.columns:
    prctj = df_inicial_test[i].isnull().mean() * 100
    print(f'{prctj:.0f}% \tde nulos en {i}')

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1764867643086, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="9duBlL40TmHg" outputId="deaa6854-474b-4eb4-c618-e4a6c07e17af"
# No es correcto, en todos los atributos hay nulos salvo en GPS, saca los decimales del procentaje
for i in df_inicial_test.columns:
    prctj = df_inicial_test[i].isnull().mean() * 100
    print(f'{prctj:.3f}%\tde nulos en {i}')

# %% executionInfo={"elapsed": 2, "status": "ok", "timestamp": 1764867643090, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="04OkbR_byHoe"
import sys
import os

# Obtiene la ruta absoluta del directorio actual del script
# current_dir = os.path.dirname(os.path.abspath(__file__))  # python
current_dir = os.getcwd()                                   # jupyter notebook

# Obtiene la ruta al directorio 'src' (subiendo un nivel y entrando a src)
src_path = os.path.join(current_dir, '..', 'src')

# Lo añade al path de Python
sys.path.append(src_path)

# Ahora ya puedes importar
import utils

# %%
null = utils.auditoria_nulos(df_inicial_test)

# %% colab={"base_uri": "https://localhost:8080/", "height": 178} executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1764867643117, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="15eb1261" outputId="a52a9b72-1bfb-4e8e-b703-65beef0f6952"
df_inicial_test["marca"].value_counts(dropna=False)

# %% [markdown] id="c19358b3"
# Borramos la columna marca debido a que es irrelevante al tratarse todo de BMW

# %% executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1764867643123, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="0ffb42b4"
df_inicial_test.drop(columns=['marca'], inplace=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 335} executionInfo={"elapsed": 19, "status": "ok", "timestamp": 1764867643148, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="48b98274" outputId="0371e53c-a960-421c-f369-c25f933e4711"
df_inicial_test["precio"].describe()

# %% colab={"base_uri": "https://localhost:8080/", "height": 467} executionInfo={"elapsed": 511, "status": "ok", "timestamp": 1764867643661, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="e93bf20c" outputId="196410d9-f15a-4411-b110-b852a1688443"
df_inicial_test.precio.hist(bins=50)
plt.xlabel('Precio')
plt.ylabel('Frecuencia')


# %% executionInfo={"elapsed": 44, "status": "ok", "timestamp": 1764867742032, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="f0c0619d"
def columnas_por_tipo(df, as_dict=False):
    """
    Agrupa las columnas de un DataFrame por tipo y devuelve las listas de nombres de columnas.

    Parámetros:
    - df: pandas.DataFrame
    - as_dict: bool (opcional). Si True devuelve un dict con las listas;
               si False (por defecto) devuelve una tupla con las listas en el
               siguiente orden: (numeric, bools, datetimes, objects, categories,
               timedeltas, complex_cols, ints, floats, others).

    Retorno:
    - Si as_dict=True: dict con claves 'all','numeric','int','float','bool',
      'datetime','object','category','timedelta','complex','others'.
    - Si as_dict=False: tupla de listas: (numeric, bools, datetimes, objects,
      categories, timedeltas, complex_cols, ints, floats, others).

    Ejemplos:
    - numeric, bools, datetimes, objects, categories, timedeltas, complex_cols,
      ints, floats, others = columnas_por_tipo(df)
    - grouped = columnas_por_tipo(df, as_dict=True)
    - df[numeric].head()  # muestra solo columnas numéricas si 'numeric' es la lista
    """
    # columnas por tipos básicos
    numeric = df.select_dtypes(include=['number']).columns.tolist()
    bools = df.select_dtypes(include=['bool']).columns.tolist()
    datetimes = df.select_dtypes(include=['datetime']).columns.tolist()
    objects = df.select_dtypes(include=['object']).columns.tolist()
    categories = df.select_dtypes(include=['category']).columns.tolist()
    timedeltas = df.select_dtypes(include=['timedelta']).columns.tolist()
    complex_cols = df.select_dtypes(include=['complex']).columns.tolist()

    # distinguir int / float dentro de numeric
    ints = [c for c in numeric if pd.api.types.is_integer_dtype(df[c])]
    floats = [c for c in numeric if pd.api.types.is_float_dtype(df[c])]

    grouped = {
        'all': df.columns.tolist(),
        'numeric': numeric,
        'int': ints,
        'float': floats,
        'bool': bools,
        'datetime': datetimes,
        'object': objects,
        'category': categories,
        'timedelta': timedeltas,
        'complex': complex_cols,
    }

    # columnas que no encajan en los grupos anteriores
    grouped_lists = [grouped[k] for k in ['numeric','int','float','bool','datetime','object','category','timedelta','complex']]
    used = set().union(*map(set, grouped_lists)) if grouped_lists else set()
    grouped['others'] = [c for c in df.columns if c not in used]

    if as_dict:
        return grouped

    # devolver las listas en un orden fijo para permitir unpacking directo
    return (numeric, bools, datetimes, objects, categories, timedeltas,
            complex_cols, ints, floats, grouped['others'])


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 36, "status": "ok", "timestamp": 1764867744848, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="538a035e" outputId="5c6be6eb-2cc7-42a7-b632-84f63787f918"
grouped_columns = columnas_por_tipo(df_inicial_test, as_dict=True)
for tipo, columnas in grouped_columns.items():
    print(f"{tipo}: {columnas}")

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"elapsed": 22, "status": "ok", "timestamp": 1764867643739, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="c5c96cf4" outputId="dfa61d66-f505-4433-ca47-8ef56d98b0ca"
numeric, bools, datetimes, objects, categories, timedeltas, complex_cols, ints, floats, others = columnas_por_tipo(df_inicial_test)
df_inicial_test[numeric].head()


# %%
def imputar_por_similitud(df, target_col, numeric_cols, cat_bool_cols, umbral_dominancia=0.75, margen_global=0.05, usar_percentiles=False, estrategia='moda'):
    """
        Imputa valores nulos en una columna objetivo basándose en la similitud con otros registros (vecinos).
        
        Esta función busca filas similares a aquellas que tienen datos faltantes en `target_col`. 
        La similitud se define mediante coincidencias exactas en variables categóricas/booleanas 
        y proximidad dentro de un margen en variables numéricas.

        Lógica de búsqueda de vecinos:
        1. Filtro estricto: Debe coincidir el valor en todas las columnas de `cat_bool_cols`.
        2. Filtro numérico: El valor debe estar dentro de un rango (margen) definido para `numeric_cols`.
        - Si `usar_percentiles=False`: Rango = valor ± (valor * margen).
        - Si `usar_percentiles=True`: Rango = percentiles correspondientes al (rank_actual ± margen).

        Args:
            df (pd.DataFrame): DataFrame original conteniendo los datos.
            target_col (str): Nombre de la columna que contiene los valores nulos a imputar.
            numeric_cols (list o dict): 
                - Si es lista: Columnas numéricas para comparar similitud usando `margen_global`.
                - Si es dict: Claves son columnas y valores son el margen específico para esa columna (ej. {'edad': 0.1}).
            cat_bool_cols (list): Lista de columnas categóricas o booleanas para coincidencia exacta (Hard matching).
            umbral_dominancia (float, opcional): Solo para estrategia 'moda'. Porcentaje mínimo de coincidencia 
                entre vecinos para aceptar la imputación (defecto 0.75).
            margen_global (float, opcional): Tolerancia por defecto para la comparación numérica (defecto 0.05 o 5%).
            usar_percentiles (bool, opcional): Define cómo se calcula el margen numérico. 
                False usa distancia relativa al valor; True usa distancia en la distribución (quantiles).
            estrategia (str, opcional): Método de agregación de los vecinos encontrados.
                - 'moda': Valor más frecuente (requiere cumplir `umbral_dominancia`).
                - 'mediana': Mediana de los vecinos (soporta números y fechas).
                - 'media': Promedio aritmético (solo números).

        Returns:
            pd.DataFrame: Una copia del DataFrame original con los valores imputados donde fue posible.
    """

    df_out = df.copy()
    
    if df_out[target_col].isnull().sum() == 0:
        return df_out

    filas_nulas = df_out[df_out[target_col].isnull()]
    print(f"Procesando {len(filas_nulas)} nulos en '{target_col}' | Estrategia: {estrategia} | Percentiles: {usar_percentiles}...")
    
    imputaciones = 0
    
    # Pre-procesamiento config numérica
    if isinstance(numeric_cols, list):
        numeric_config = {col: margen_global for col in numeric_cols}
    else:
        numeric_config = numeric_cols

    for idx, row in filas_nulas.iterrows():
        mask = pd.Series([True] * len(df), index=df.index)
        
        # 1. Filtros
        for col in cat_bool_cols:
            valor = row[col]
            if not pd.isna(valor):
                mask &= (df[col] == valor)
        
        for col, margen_especifico in numeric_config.items():
            valor = row[col]
            if not pd.isna(valor):
                if usar_percentiles:
                    # Lógica Percentil
                    rank_actual = (df[col] < valor).mean()
                    q_min = max(0, rank_actual - margen_especifico)
                    q_max = min(1, rank_actual + margen_especifico)
                    try:
                        lim_inf = df[col].quantile(q_min)
                        lim_sup = df[col].quantile(q_max)
                        mask &= (df[col].between(lim_inf, lim_sup))
                    except: pass
                else:
                    # Lógica Rango Fijo
                    try:
                        lim_inf = valor * (1 - margen_especifico)
                        lim_sup = valor * (1 + margen_especifico)
                        if valor < 0: lim_inf, lim_sup = lim_sup, lim_inf
                        mask &= (df[col].between(lim_inf, lim_sup))
                    except: pass

        mask &= (df.index != idx)
        vecinos = df.loc[mask, target_col]
        
        if vecinos.empty:
            continue
            
        # 2. DECISIÓN SEGÚN ESTRATEGIA
        valor_imputado = None
        
        if estrategia == 'moda':
            # Lógica original: Requiere consenso
            conteo = vecinos.value_counts(normalize=True)
            if not conteo.empty:
                if conteo.iloc[0] >= umbral_dominancia:
                    valor_imputado = conteo.index[0]
        
        elif estrategia == 'mediana':
            # Lógica numérica/fechas: Valor central
            # Para fechas, convertimos a numérico, calculamos mediana y reconvertimos
            if pd.api.types.is_datetime64_any_dtype(vecinos):
                mediana_num = vecinos.astype(np.int64).median()
                valor_imputado = pd.to_datetime(mediana_num)
            else:
                valor_imputado = vecinos.median()
                
        elif estrategia == 'media':
             if pd.api.types.is_numeric_dtype(vecinos):
                valor_imputado = vecinos.mean()

        # 3. Asignación
        if valor_imputado is not None:
            df_out.at[idx, target_col] = valor_imputado
            imputaciones += 1

    print(f"Terminado: Se imputaron {imputaciones} de {len(filas_nulas)} ({round(imputaciones/len(filas_nulas)*100, 2)}%).")
    return df_out


# %%
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

df_inicial_test_2 = df_inicial_test.copy()
df_inicial_test_2.info()
df_inicial_test_2 = df_inicial_test_2.dropna(subset=['precio'])
# df_inicial_test_2 = df_inicial_test_2.drop(columns=['marca','asientos_traseros_plegables'])
df_inicial_test_2 = df_inicial_test_2.fillna({'color': 'desconocido'})
df_inicial_test_2['tipo_gasolina'] = df_inicial_test_2['tipo_gasolina'].str.lower()
df_inicial_test_2.loc[(df_inicial_test_2['modelo'] == 'X3') & (df_inicial_test_2['tipo_coche'] == 'van'),'tipo_coche'] = 'suv'
# null = utils.auditoria_nulos(df_inicial_test_2)
df_inicial_test_knn = df_inicial_test_2.copy()

target_equipamiento = ['aire_acondicionado', 'camara_trasera', 'elevalunas_electrico', 
                       'bluetooth', 'alerta_lim_velocidad', 'volante_regulable']
target_estructural = ['modelo', 'tipo_coche', 'tipo_gasolina']
target_fechas = ['fecha_registro', 'fecha_venta'] # A estos les aplicaremos mediana

# config_strict = {'potencia': 0.05, 'km': 0.20, 'precio': 0.10}
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

df_inicial_test_knn[target_equipamiento] = df_inicial_test_knn[target_equipamiento].fillna(False)

df_inicial_test_knn = utils.imputar_por_similitud(
            df_inicial_test_knn, target_col='potencia', numeric_cols={'precio': 1.0},
            cat_bool_cols=['modelo'], umbral_dominancia=0.40, estrategia='moda',
            usar_percentiles=False
        )

for col in df_inicial_test_knn.columns:
    if str(col).upper().startswith('FECHA'):
        # 1. Aseguramos que sea datetime (ignora errores si ya hay NaTs)
        df_inicial_test_knn[col] = pd.to_datetime(df_inicial_test_knn[col], errors='coerce')
        # 2. Convertimos a timestamp mensual para evitar problemas de horas/minutos/segundos
        df_inicial_test_knn[col] = df_inicial_test_knn[col].dt.to_period('M').dt.to_timestamp()

     
df_inicial_test_knn_2 = df_inicial_test_knn.copy()

df_limpio = df_inicial_test_knn[df_inicial_test_knn.isnull().sum(axis=1) <= 1].copy()
display(utils.auditoria_nulos(df_limpio))

# %%
df_inicial.shape, df_limpio.shape

# %%
df_limpio[df_limpio['potencia'] <= 60.0]

# %%
df_limpio[df_limpio['modelo'] == 'i3']['potencia'].value_counts()

# %%
df_limpio[df_limpio['modelo'] == 'X1']['potencia'].value_counts(normalize=True)

# %%
df_limpio[df_limpio['modelo'] == 'X1']['potencia'].median()

# %%
df_test = df_limpio.copy()
df_test["DIF_TIEMPO"] = df_limpio["fecha_venta"] - df_limpio["fecha_registro"]
df_test["DIF_TIEMPO"].mean().days / 365.25

# %%
df_limpio['precio'].quantile(0.90)

# %% [markdown] id="qMzV9mTcwP4e"
# ## PREGUNTA 1

# %% executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1764867645537, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="eLwQsmuewvOw"
## 1. ¿Qué columnas eliminaron inicialmente del dataset y por qué?

# %% executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1764867646474, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="QavYpCoMwvAQ"
## Kike Se elimino la columna MARCA al ser todo el data set de la marca BMW

# %% executionInfo={"elapsed": 39, "status": "ok", "timestamp": 1764867647271, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="3djyoOHH3_qA"
## Kike defino target. TARGET = "precio"

# %% [markdown] id="v6lXYPvgwPp2"
# ## PREGUNTA 2

# %% executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1764867648706, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="ive941SBwwOw"
## 2. Manejo de nulos, explicar qué se hizo con los nulos por cada columna

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1764867649591, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="sBxB-oRfwwF4" outputId="9263c787-2d79-4e11-854f-1fdabbbdbeee"
# Kike Porcentaje de nulos
for i in df_inicial_test.columns:
    prctj = df_inicial_test[i].isnull().mean() * 100
    print(f'{prctj:.3f}% \tde nulos en {i}')

# %% colab={"base_uri": "https://localhost:8080/", "height": 699} executionInfo={"elapsed": 990, "status": "ok", "timestamp": 1764867652557, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="IN1flmUd6sh5" outputId="78a45f66-31f9-42ed-96ca-5b149ef6f884"
## Kike Vemos si fecha de registro incide en target
## Kike Este código es para transformar en años
df_inicial_test["fecha_registro"] = pd.to_datetime(
    df_inicial_test["fecha_registro"], errors='coerce'
)

df_inicial_test["anio_registro"] = df_inicial_test["fecha_registro"].dt.year


plt.figure(figsize=(12,6))
sns.violinplot(x="anio_registro", y="precio", data=df_inicial_test, palette="Wistia")
plt.xticks(rotation=45)
plt.tight_layout()


# %% executionInfo={"elapsed": 33, "status": "ok", "timestamp": 1764867666038, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="o0XKkhbA3G45"
## Kike Esta es la propuesta:
## 20.03% 	de nulos en marca. Eliminar variable
## 0.06% 	de nulos en modelo
## 0.04% 	de nulos en km
## 0.02% 	de nulos en potencia
## 50.03% 	de nulos en fecha_registro
## 0.10% 	de nulos en tipo_gasolina
## 9.19% 	de nulos en color
## 30.15% 	de nulos en tipo_coche
## 0.08% 	de nulos en volante_regulable
## 10.04% 	de nulos en aire_acondicionado
## 0.04% 	de nulos en camara_trasera
## 70.02% 	de nulos en asientos_traseros_plegables
## 0.04% 	de nulos en elevalunas_electrico
## 15.03% 	de nulos en bluetooth
## 0.00% 	de nulos en gps
## 15.03% 	de nulos en alerta_lim_velocidad
## 0.12% 	de nulos en precio
## 0.02% 	de nulos en fecha_venta
##Kike
## Marca, eliminar variable. Son todos BMW
## Fecha registro, muchos nulos, incide en target. Pondría 0 si conozco antigüedad y 1 : no la conozco
## Asientos traseros plegables, muchos nulos, poner que no tiene. bool:0
## Entre 10%-50%, objetc:"desconocido", bool:0
## <1%. No afectan modelo. objet: "desconocido", float:mediana, bool:0
## Target precio. No se imputan, se eliminan filas

# %% [markdown] id="8S9xk1TawPcd"
# ## PREGUNTA 3

# %% executionInfo={"elapsed": 49, "status": "ok", "timestamp": 1764867668405, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="HfB96SCxww7y"
## 3. Análisis univariable, explicar alguna información interesante encontrada

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 44, "status": "ok", "timestamp": 1764867669280, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="_aj8OzLZ7t-v" outputId="088fd32f-1457-490a-a681-f82055414ae0"
df_inicial_test.info()
## Kike
# No hay duplicados
# Se cambia a dates los atributos de fechas.
# Los nulos se contestaron en pregunta anterior.
# Outliers en precio (haría LOG), en potencia y km. Ver más abajo y habría que decidir qué hacer con ellos.
## Hay km negativos

# %% colab={"base_uri": "https://localhost:8080/", "height": 394} executionInfo={"elapsed": 120, "status": "ok", "timestamp": 1764867672072, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="9Tcgo-zB7utM" outputId="adb194db-a397-4668-d534-22833a8f3b76"
df_inicial_test.describe(include = ['object']).T

# %% colab={"base_uri": "https://localhost:8080/", "height": 269} executionInfo={"elapsed": 83, "status": "ok", "timestamp": 1764867674083, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="v9fOWYV-_pqC" outputId="2564652f-a232-4f38-d7bb-d87193c9fb84"
df_inicial_test.describe(exclude = ['object']).T

# %% colab={"base_uri": "https://localhost:8080/", "height": 467} executionInfo={"elapsed": 221, "status": "ok", "timestamp": 1764867676508, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="2_r2UrqYCr5H" outputId="fb5adaf3-c253-4428-a73d-b65f01d350b6"
## Kike El precio tiene 2 outliers. Quizás sea necesario un LOG

df_inicial_test.precio.hist(bins=50)
plt.xlabel('Precio')
plt.ylabel('Frecuencia')

# %% colab={"base_uri": "https://localhost:8080/", "height": 467} executionInfo={"elapsed": 209, "status": "ok", "timestamp": 1764867678939, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="z3SBBD0TCre1" outputId="c8058944-b6fb-4aa6-94cb-693425565a63"
sns.boxplot(x=df_inicial_test["precio"])

# %% colab={"base_uri": "https://localhost:8080/", "height": 467} executionInfo={"elapsed": 153, "status": "ok", "timestamp": 1764867681391, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="-mhO-rqwB-X6" outputId="5304982a-30a9-4e5e-c270-287f5f26926a"
## Kike Miro outliers en km y potencia según describe. Hay un outlier en km y dos en potencia

sns.boxplot(x=df_inicial_test["km"])


# %% colab={"base_uri": "https://localhost:8080/", "height": 467} executionInfo={"elapsed": 151, "status": "ok", "timestamp": 1764867683861, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="eoWI74RvCUlS" outputId="24c3ef3b-5349-42d8-9c3c-c395d4853bba"
sns.boxplot(x=df_inicial_test["potencia"])

# %% [markdown] id="eglFtNgxwY5o"
# ## PREGUNTA 4

# %% executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1764867686161, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="HoKuH-kNwxnI"
## 4. Análisis de correlación inicial, ¿Hay alguna variable correlacionada?

# %%
utils.analizar_correlaciones_productos(df_limpio, 'precio', agrupar=True, clave_agrupar='modelo')


# %%
def analizar_correlaciones(df, cols_analisis, agrupar=False, clave_agrupar=None, metodo_agrupacion='mean'):
    """
    Genera un Mapa de Calor (Heatmap) de correlaciones (Pearson) para columnas numéricas o booleanas.
    
    Parámetros:
    - df: DataFrame.
    - cols_analisis: Lista de las columnas a relacionar.
    - agrupar: (Bool) Si True, agrupa los datos antes de correlacionar.
    - clave_agrupar: (Opcional) Columna para agrupar (ej. 'cliente_id').
    - metodo_agrupacion: 'mean' (promedio, recomendado numéricas) o 'max' (binario estricto).
    """
    
    # Copia para no alterar original
    df_temp = df.copy()

    # 1. Gestión de Agrupación
    if agrupar:
        # Auto-detectar clave si no se pasa
        if clave_agrupar is None:
            # Busca primera columna que empiece por 'pk_' y no sea partition
            posibles_pks = [c for c in df_temp.columns if 'pk_' in c and c != 'pk_partition']
            if posibles_pks:
                clave_agrupar = posibles_pks[0]
            else:
                raise ValueError("No se encontró 'pk_' automática. Especifica 'clave_agrupar'.")
        
        print(f"Agrupando por: {clave_agrupar} usando método: {metodo_agrupacion}")
        
        # Agrupar
        if metodo_agrupacion == 'max':
            df_analisis = df_temp.groupby(clave_agrupar)[cols_analisis].max()
        else:
            df_analisis = df_temp.groupby(clave_agrupar)[cols_analisis].mean()
            
    else:
        # Sin agrupar: análisis directo de filas
        df_analisis = df_temp[cols_analisis]

    # 2. Cálculo de la Correlación
    corr_matrix = df_analisis.corr(method='pearson')
    
    # 3. Diseño del Gráfico
    plt.figure(figsize=(14, 12))
    
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap='RdBu',      # Rojo (inversa) - Blanco (neutro) - Azul (directa)
        vmin=-1, vmax=1,  # Rango completo para numéricas
        center=0, 
        linewidths=0.5, 
        cbar_kws={"shrink": .5}
    )
    
    titulo = f"Matriz de Correlación ({'Agrupada' if agrupar else 'Directa'})"
    plt.title(titulo, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.show()


# %%
def aplanar_lista(elementos):
    lista_plana = []
    for item in elementos:
        if isinstance(item, list):
            lista_plana.extend(item) # Si es lista, añade sus elementos
        else:
            lista_plana.append(item) # Si es texto, lo añade directamente
    return lista_plana

# 2. Creamos la lista pasando los elementos crudos (sin sumar con +)
lista_elementos = [columnas_numericas, target_equipamiento, 'precio']
cols_finales = aplanar_lista(lista_elementos)

# 3. Verificamos (opcional)
print("Columnas finales:", cols_finales)

# %%
analizar_correlaciones(df_limpio, cols_finales)

# %%
cols_categoricas = ['modelo', 'tipo_coche']
target = 'precio'

# 2. Creamos las variables dummy (0 y 1) para los textos
# drop_first=False para ver todas las categorías explícitamente
df_dummies = pd.get_dummies(df_limpio[cols_categoricas], prefix_sep=': ')

# 3. Añadimos el precio al dataframe de dummies
df_analisis = pd.concat([df_dummies, df_limpio[target]], axis=1)

# 4. Calculamos la correlación SOLO contra el precio (para no hacer una matriz gigante ilegible)
correlaciones_precio = df_analisis.corr(method='pearson')[[target]]

# 5. Ordenamos de mayor a menor impacto y quitamos la fila de 'precio' contra sí mismo
correlaciones_precio = correlaciones_precio.drop(index=target).sort_values(by=target, ascending=False)

# --- GRÁFICO ---
plt.figure(figsize=(10, len(correlaciones_precio) * 0.4)) # Altura dinámica según cantidad de modelos

# Usamos un heatmap de una sola columna para ver el impacto visualmente
sns.heatmap(correlaciones_precio, 
            annot=True, 
            cmap='RdBu', 
            vmin=-1, vmax=1, 
            fmt='.2f',
            linewidths=0.5)

plt.title('Impacto de Modelo y Tipo en el Precio', fontsize=14)
plt.show()

# %%
ols_categoricas = ['color']
target = 'precio'

# 2. Creamos las variables dummy (0 y 1) para los textos
# drop_first=False para ver todas las categorías explícitamente
df_dummies = pd.get_dummies(df_limpio[ols_categoricas], prefix_sep=': ')

# 3. Añadimos el precio al dataframe de dummies
df_analisis = pd.concat([df_dummies, df_limpio[target]], axis=1)

# 4. Calculamos la correlación SOLO contra el precio (para no hacer una matriz gigante ilegible)
correlaciones_precio = df_analisis.corr(method='pearson')[[target]]

# 5. Ordenamos de mayor a menor impacto y quitamos la fila de 'precio' contra sí mismo
correlaciones_precio = correlaciones_precio.drop(index=target).sort_values(by=target, ascending=False)

# --- GRÁFICO ---
plt.figure(figsize=(10, len(correlaciones_precio) * 0.4)) # Altura dinámica según cantidad de modelos

# Usamos un heatmap de una sola columna para ver el impacto visualmente
sns.heatmap(correlaciones_precio, 
            annot=True, 
            cmap='RdBu', 
            vmin=-1, vmax=1, 
            fmt='.2f',
            linewidths=0.5)

plt.title('Impacto de Color en el Precio', fontsize=14)
plt.show()

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"elapsed": 145, "status": "ok", "timestamp": 1764867687119, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="j2CNGJLVwxfQ" outputId="00ae6e8f-991a-4a96-dd7f-2cef871b1932"
## Kike Miramos correlaciones entre variables numericas. Nada a destacar, precio y km son casi opuestas
## Kike Habría que pasar las categóricas a numéricas, pero lo pide en la pregunta 6.

corr=df_inicial_test.corr(numeric_only=True)
corr.style.background_gradient(cmap='coolwarm')

# %% [markdown] id="sUJiwOPSwYuI"
# ## PREGUNTA 5

# %% executionInfo={"elapsed": 73, "status": "ok", "timestamp": 1764867689334, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="yOfXslvFwyQB"
## 5. Análisis variable vs target, ¿Hay algún insight interesante?

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 41, "status": "ok", "timestamp": 1764867690192, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="TRZoc7WXwyJb" outputId="722265eb-7d6f-42a3-81ce-39c9036bc4e6"
df_inicial_test.info()
## Kike annio está en float, debe pasarse a datetime64

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 76, "status": "ok", "timestamp": 1764868515912, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="hYF3IPBhVodO" outputId="3f41b690-9728-434b-da90-573ded5e1120"
## Kike Paso annio_registro a datetime
df_inicial_test["anio_registro"] = pd.to_datetime(df_inicial_test["anio_registro"])

df_inicial_test.info()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 63, "status": "ok", "timestamp": 1764868104853, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="Fe34sDbMVbLV" outputId="f0fafba6-81f3-4e5b-d476-2ed44a136223"
df_inicial_test["modelo"].unique()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 45, "status": "ok", "timestamp": 1764868556136, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="SGEwCQL1VKaF" outputId="0ad01ece-1c45-4fa6-9159-92b10042dc82"
## Kike listo toda las variables con sus valores únicos
for i in df_inicial_test:
  print(i,"\n\n",df_inicial_test[i].value_counts(),"\n\n")

# %% executionInfo={"elapsed": 44, "status": "ok", "timestamp": 1764868794038, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="p95HJuElX_54"
## Kike Defino target
target = ["precio"]

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 84, "status": "ok", "timestamp": 1764871935195, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="RUnXOW4obshE" outputId="15e5bb0a-1471-4a61-9c34-b76b5d8fc585"
## Kike Paso variable gps booleana a int
df_inicial_test["gps_bool"] = df_inicial_test["gps"].astype(int)
df_inicial_test.drop(columns=["gps"], inplace=True)
df_inicial_test.info()

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 827, "status": "ok", "timestamp": 1764870095718, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="bEmP1Ji2ZUMP" outputId="8f399fcf-1c20-441c-e320-8ff8fd8b48b8"
## Kike Variables numericas con target con scatterplot
## Kike Conclusion: a más km, menos precio y a más potencia, más precio. Con gps son más caros.
## Kike Seleccionar columnas numéricas

num_cols = df_inicial_test.select_dtypes(include=['int64','float64']).columns

for col in num_cols:
    if col == "precio":   # evitar comparar el target consigo mismo
        continue

    plt.figure(figsize=(7,5))
    sns.scatterplot(x=df_inicial_test[col], y=df_inicial_test["precio"], alpha=0.5)
    plt.title(f"Scatterplot: {col} vs precio")
    plt.xlabel(col)
    plt.ylabel("precio")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 7174, "status": "ok", "timestamp": 1764869261955, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="mn75HSyiZy22" outputId="40a3e239-3c31-4214-d51a-9b5ace83ed2a"
## Kike Variables categoricas con target usando violinplot
## Kike Conclusion: Asientos plegables traseros más caros. Naranja, cuope e Hybrid-petrol más caro
## Kike Seleccionar columnas categóricas (object o category)
cat_cols = df_inicial_test.select_dtypes(include=['object', 'category']).columns

for col in cat_cols:
    plt.figure(figsize=(10,5))
    sns.violinplot(
        x=df_inicial_test[col],
        y=df_inicial_test["precio"],
        palette="Wistia"
    )
    plt.title(f"Violin plot: {col} vs precio")
    plt.xlabel(col)
    plt.ylabel("precio")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# %% [markdown] id="4OteeAdRwYYO"
# ## PREGUNTA 6

# %% executionInfo={"elapsed": 44, "status": "ok", "timestamp": 1764867537577, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="viEM_19fwy7J"
## Kike 6. Transformación de categóricas a numéricas, ¿Qué variables van a transformar? ¿Que técnica se va usar?

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 100, "status": "ok", "timestamp": 1764871961806, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="3wVU8UiOwyyY" outputId="d8f6dafa-62e5-4a05-ca11-52e0063b319b"
## Kike Miramos en un info como están las variables y después hacemos un bucle que me cambie todas las categóricas a numéricas

df_inicial_test.info()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 62, "status": "ok", "timestamp": 1764871140595, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="O3sBMO0Denjg" outputId="ed91d2f3-95a4-409a-96d6-ef3a1f3f9837"
## Kike vuelvo a listar todas las variables con sus valores únicos
for i in df_inicial_test:
  print(i,"\n\n",df_inicial_test[i].value_counts(),"\n\n")

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 65, "status": "ok", "timestamp": 1764871978913, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="3I9Y9CF-e3P7" outputId="e2f09d6f-8745-4c56-b7ef-6b4543864f44"
for i in df_inicial_test:
    print(df_inicial_test[i].dtype.kind)


# %% executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1764870878283, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="3p4uq15lgBYH"
def obtener_lista_variables(dataset):

    lista_numericas=[]
    lista_boolean=[]
    lista_categoricas=[]

    for i in dataset:
        if    (dataset[i].dtype.kind=="f" or dataset[i].dtype.kind=="i") and len(dataset[i].unique())!=2 and (i not in target):
              lista_numericas.append(i)
        elif  (dataset[i].dtype.kind=="f" or dataset[i].dtype.kind=="i") and len(dataset[i].unique())==2 and (i not in target):
              lista_boolean.append(i)
        elif  (dataset[i].dtype.kind=="O") and i not in target:
              lista_categoricas.append(i)

    return lista_numericas, lista_boolean, lista_categoricas


# %% executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1764870942954, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="0nKBhix_gH4I"
## Kike Paso al dataset las listas
lista_numericas, lista_boolean, lista_categoricas = obtener_lista_variables(df_inicial_test)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 54, "status": "ok", "timestamp": 1764870956819, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="LoK02iXAgXsN" outputId="422e3ea5-890b-4fb0-a01b-11c440931521"
print (lista_categoricas)

# %% id="v_oc_x2mgzlJ"
## Kike Transformamos 11 variables categóricas
## Kike modelo (77), tipo gasolina (6), color (11), tipo coche(9), volante reg(T/F), Aire (T/F), Camara T(T/F), Asientos (T/F), elevalunas (T/F), bluetooth (T/F), Alerta (T/F)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 33, "status": "ok", "timestamp": 1764871544902, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="l1tOUXK7gyzV" outputId="6f46ad23-63bc-49cf-cac7-054f7ac5e209"
for col in lista_categoricas:
    print(f"Columna: {col}\n")
    print(df_inicial_test[col].value_counts(dropna=False))  # Incluye NaN
    print("\n" + "-"*50 + "\n")


# %% executionInfo={"elapsed": 42, "status": "ok", "timestamp": 1764873536371, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="QQlHh_Ejn4Zr"
## Kike Quiero pasar las categoricas a numericas.
## Kike Primero paso las 7 variables T/F. Nos quedan 4 variables categóricas

cols_bool = ["volante_regulable", "aire_acondicionado", "camara_trasera", "asientos_traseros_plegables", "elevalunas_electrico", "bluetooth","alerta_lim_velocidad"]

for col in cols_bool:
    df_inicial_test[col] = np.where(df_inicial_test[col] == "TRUE", 1, 0)


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 73, "status": "ok", "timestamp": 1764873610099, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="eMbb00kpqeTU" outputId="a10d4255-8033-4ac0-f135-72609a98c2ce"
## Kike Creo que no me ha pasado a lista_numericas las 7 variables anteriores
## Kike Se puede hacer un ordinal encoder a tipo_gasolina, color y tipo_coche.
## Kike Pendiente de decidir que se hace con los 77 modelos.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 62, "status": "ok", "timestamp": 1764873538901, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="nq1JJA-PpPo1" outputId="3220aff5-d192-4043-f3ad-58b3d286b76f"
df_inicial_test.info()

# %% [markdown] id="UhDx02LSwYLd"
# ## PREGUNTA 7

# %% executionInfo={"elapsed": 29, "status": "ok", "timestamp": 1764867547911, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="wgjjs1z1wz1Y"
## 7. Escalar variables (usando minmaxscaler) y luego aplicar la correlación final de variables ¿Hay alguna variable finalmente correlacionada?

# %% executionInfo={"elapsed": 2, "status": "ok", "timestamp": 1764867547917, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}, "user_tz": -60} id="zbo1Sp3JwztQ"

# %% [markdown] id="GgiUUdodwX81"
# ## PREGUNTA 8

# %% id="vII6Ckx-w0hz"
## 8. Subir un pantallazo con el nombre de TODAS las columnas que tiene el dataframe final final vfin y el tipo de dato que tiene, tip lo más sencillo es hacer un .info al dataframe y tomarle un pantallazo y subirlo, importante deben aparecer TODAS las columnas

# %% id="lPTSL4ypw0bg"
## Es hacer un pantallazo del data frame y del .info y subirlo al enlance.

# %% [markdown] id="-IcZDPWnwo1x"
# ##PREGUNTA 9

# %% id="0Zlg5Fcgw1Xc"
## 9. Exportar en un excel de las primeras 50 filas del dataset (df.to_excel(...)

# %% id="gAFBQ-fgw1Qg"
pd.to_excel(df_final,"/content/drive/MyDrive/Colab Notebooks/df_final")

# %% [markdown] id="XHGWvGglwoqw"
# ## PREGUNTA 10

# %% id="Y7g1yEJKw2IS"
## 10. Sube el programa (el archivo .ipynb, NO EL PICKLE de lo contrario me quedaré sin espacio...) y con esto hemos terminado ;)

# %% id="G2_pVKq5w2AC"
# En archivo voy a descargar .ipynb. Luego lo subo al enlace
