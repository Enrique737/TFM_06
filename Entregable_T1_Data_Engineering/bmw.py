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

# %% id="f40eb448" executionInfo={"status": "ok", "timestamp": 1764867642695, "user_tz": -60, "elapsed": 2423, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
import pandas as pd # Librería para la manipulación y el análisis de datos
import numpy as np # Librería para la manipulación de datos y para la ejecución de operaciones matemáticas
import matplotlib.pyplot as plt # Librería para la visualización de datos
import seaborn as sns # Librería para la visualización de datos
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder # Librería para crear modelos de ML

# %% id="06c52e51" executionInfo={"status": "ok", "timestamp": 1764867642709, "user_tz": -60, "elapsed": 3, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
df_inicial = pd.read_csv("/content/bmw_pricing_v3.csv")

# %% id="11f98c1a" outputId="9907d390-a1c8-4baa-e8b8-7c830105829d" colab={"base_uri": "https://localhost:8080/", "height": 245} executionInfo={"status": "ok", "timestamp": 1764867642872, "user_tz": -60, "elapsed": 147, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
print("Shape:", df_inicial.shape)
df_inicial.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 648} id="0bec4738" executionInfo={"status": "ok", "timestamp": 1764867642890, "user_tz": -60, "elapsed": 10, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="99f5628c-e58b-490d-9cad-1c25d9bba44c"
df_inicial.isnull().sum()

# %% colab={"base_uri": "https://localhost:8080/", "height": 648} id="701d2c60" executionInfo={"status": "ok", "timestamp": 1764867642964, "user_tz": -60, "elapsed": 40, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="f445e262-01c1-40bc-f53c-048ab0dd1626"
df_inicial.isnull().any()

# %% colab={"base_uri": "https://localhost:8080/"} id="a92ccb8c" executionInfo={"status": "ok", "timestamp": 1764867642982, "user_tz": -60, "elapsed": 22, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="ede8783c-fbf3-4028-9710-578029166d3e"
df_inicial.info()

# %% colab={"base_uri": "https://localhost:8080/"} id="b670a026" executionInfo={"status": "ok", "timestamp": 1764867643017, "user_tz": -60, "elapsed": 32, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="f19712b1-96c0-453c-f383-d96e3e98d62e"
# Conversión con pd.to_datetime de los atributos que tienen fecha y son float
df_inicial_test = df_inicial.copy()
for i in df_inicial.columns:
    if str(i).upper().startswith('FECHA'):
        df_inicial_test[i] = pd.to_datetime(df_inicial[i])

df_inicial_test.info()

# %% colab={"base_uri": "https://localhost:8080/"} id="97be129d" executionInfo={"status": "ok", "timestamp": 1764867643039, "user_tz": -60, "elapsed": 21, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="0d171fee-c514-4c1f-fb31-86fa7c3d7dcf"
df_inicial_test[df_inicial_test.duplicated(keep='first')].shape

# %% [markdown] id="8da2b150"
# No hay duplicados

# %% colab={"base_uri": "https://localhost:8080/"} id="1d785c55" executionInfo={"status": "ok", "timestamp": 1764867643053, "user_tz": -60, "elapsed": 20, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="a4f2637c-f12a-430b-ad81-2552e641b5c7"
# Porcentaje de nulos
for i in df_inicial_test.columns:
    prctj = df_inicial_test[i].isnull().mean() * 100
    print(f'{prctj:.0f}% \tde nulos en {i}')

# %% colab={"base_uri": "https://localhost:8080/"} id="9duBlL40TmHg" executionInfo={"status": "ok", "timestamp": 1764867643086, "user_tz": -60, "elapsed": 7, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="deaa6854-474b-4eb4-c618-e4a6c07e17af"
# No es correcto, en todos los atributos hay nulos salvo en GPS, saca los decimales del procentaje
for i in df_inicial_test.columns:
    prctj = df_inicial_test[i].isnull().mean() * 100
    print(f'{prctj:.2f}% \tde nulos en {i}')

# %% id="04OkbR_byHoe" executionInfo={"status": "ok", "timestamp": 1764867643090, "user_tz": -60, "elapsed": 2, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}

# %% colab={"base_uri": "https://localhost:8080/", "height": 178} id="15eb1261" executionInfo={"status": "ok", "timestamp": 1764867643117, "user_tz": -60, "elapsed": 7, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="a52a9b72-1bfb-4e8e-b703-65beef0f6952"
df_inicial_test["marca"].value_counts(dropna=False)

# %% [markdown] id="c19358b3"
# Borramos la columna marca debido a que es irrelevante al tratarse todo de BMW

# %% id="0ffb42b4" executionInfo={"status": "ok", "timestamp": 1764867643123, "user_tz": -60, "elapsed": 3, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
df_inicial_test.drop(columns=['marca'], inplace=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 335} id="48b98274" executionInfo={"status": "ok", "timestamp": 1764867643148, "user_tz": -60, "elapsed": 19, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="0371e53c-a960-421c-f369-c25f933e4711"
df_inicial_test["precio"].describe()

# %% colab={"base_uri": "https://localhost:8080/", "height": 467} id="e93bf20c" executionInfo={"status": "ok", "timestamp": 1764867643661, "user_tz": -60, "elapsed": 511, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="196410d9-f15a-4411-b110-b852a1688443"
df_inicial_test.precio.hist(bins=50)
plt.xlabel('Precio')
plt.ylabel('Frecuencia')


# %% id="f0c0619d" executionInfo={"status": "ok", "timestamp": 1764867742032, "user_tz": -60, "elapsed": 44, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
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


# %% colab={"base_uri": "https://localhost:8080/"} id="538a035e" executionInfo={"status": "ok", "timestamp": 1764867744848, "user_tz": -60, "elapsed": 36, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="5c6be6eb-2cc7-42a7-b632-84f63787f918"
grouped_columns = columnas_por_tipo(df_inicial_test, as_dict=True)
for tipo, columnas in grouped_columns.items():
    print(f"{tipo}: {columnas}")

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="c5c96cf4" executionInfo={"status": "ok", "timestamp": 1764867643739, "user_tz": -60, "elapsed": 22, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="dfa61d66-f505-4433-ca47-8ef56d98b0ca"
numeric, bools, datetimes, objects, categories, timedeltas, complex_cols, ints, floats, others = columnas_por_tipo(df_inicial_test)
df_inicial_test[numeric].head()

# %% [markdown] id="qMzV9mTcwP4e"
# ## PREGUNTA 1

# %% id="eLwQsmuewvOw" executionInfo={"status": "ok", "timestamp": 1764867645537, "user_tz": -60, "elapsed": 7, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
## 1. ¿Qué columnas eliminaron inicialmente del dataset y por qué?

# %% id="QavYpCoMwvAQ" executionInfo={"status": "ok", "timestamp": 1764867646474, "user_tz": -60, "elapsed": 7, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
## Kike Se elimino la columna MARCA al ser todo el data set de la marca BMW

# %% id="3djyoOHH3_qA" executionInfo={"status": "ok", "timestamp": 1764867647271, "user_tz": -60, "elapsed": 39, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
## Kike defino target. TARGET = "precio"

# %% [markdown] id="v6lXYPvgwPp2"
# ## PREGUNTA 2

# %% id="ive941SBwwOw" executionInfo={"status": "ok", "timestamp": 1764867648706, "user_tz": -60, "elapsed": 7, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
## 2. Manejo de nulos, explicar qué se hizo con los nulos por cada columna

# %% id="sBxB-oRfwwF4" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1764867649591, "user_tz": -60, "elapsed": 10, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="9263c787-2d79-4e11-854f-1fdabbbdbeee"
# Kike Porcentaje de nulos
for i in df_inicial_test.columns:
    prctj = df_inicial_test[i].isnull().mean() * 100
    print(f'{prctj:.2f}% \tde nulos en {i}')

# %% colab={"base_uri": "https://localhost:8080/", "height": 699} id="IN1flmUd6sh5" executionInfo={"status": "ok", "timestamp": 1764867652557, "user_tz": -60, "elapsed": 990, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="78a45f66-31f9-42ed-96ca-5b149ef6f884"
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


# %% id="o0XKkhbA3G45" executionInfo={"status": "ok", "timestamp": 1764867666038, "user_tz": -60, "elapsed": 33, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
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

# %% id="HfB96SCxww7y" executionInfo={"status": "ok", "timestamp": 1764867668405, "user_tz": -60, "elapsed": 49, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
## 3. Análisis univariable, explicar alguna información interesante encontrada

# %% colab={"base_uri": "https://localhost:8080/"} id="_aj8OzLZ7t-v" executionInfo={"status": "ok", "timestamp": 1764867669280, "user_tz": -60, "elapsed": 44, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="088fd32f-1457-490a-a681-f82055414ae0"
df_inicial_test.info()
## Kike
# No hay duplicados
# Se cambia a dates los atributos de fechas.
# Los nulos se contestaron en pregunta anterior.
# Outliers en precio (haría LOG), en potencia y km. Ver más abajo y habría que decidir qué hacer con ellos.
## Hay km negativos

# %% colab={"base_uri": "https://localhost:8080/", "height": 394} id="9Tcgo-zB7utM" executionInfo={"status": "ok", "timestamp": 1764867672072, "user_tz": -60, "elapsed": 120, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="adb194db-a397-4668-d534-22833a8f3b76"
df_inicial_test.describe(include = ['object']).T

# %% colab={"base_uri": "https://localhost:8080/", "height": 269} id="v9fOWYV-_pqC" executionInfo={"status": "ok", "timestamp": 1764867674083, "user_tz": -60, "elapsed": 83, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="2564652f-a232-4f38-d7bb-d87193c9fb84"
df_inicial_test.describe(exclude = ['object']).T

# %% colab={"base_uri": "https://localhost:8080/", "height": 467} id="2_r2UrqYCr5H" executionInfo={"status": "ok", "timestamp": 1764867676508, "user_tz": -60, "elapsed": 221, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="fb5adaf3-c253-4428-a73d-b65f01d350b6"
## Kike El precio tiene 2 outliers. Quizás sea necesario un LOG

df_inicial_test.precio.hist(bins=50)
plt.xlabel('Precio')
plt.ylabel('Frecuencia')

# %% colab={"base_uri": "https://localhost:8080/", "height": 467} id="z3SBBD0TCre1" executionInfo={"status": "ok", "timestamp": 1764867678939, "user_tz": -60, "elapsed": 209, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="c8058944-b6fb-4aa6-94cb-693425565a63"
sns.boxplot(x=df_inicial_test["precio"])

# %% colab={"base_uri": "https://localhost:8080/", "height": 467} id="-mhO-rqwB-X6" executionInfo={"status": "ok", "timestamp": 1764867681391, "user_tz": -60, "elapsed": 153, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="5304982a-30a9-4e5e-c270-287f5f26926a"
## Kike Miro outliers en km y potencia según describe. Hay un outlier en km y dos en potencia

sns.boxplot(x=df_inicial_test["km"])


# %% colab={"base_uri": "https://localhost:8080/", "height": 467} id="eoWI74RvCUlS" executionInfo={"status": "ok", "timestamp": 1764867683861, "user_tz": -60, "elapsed": 151, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="24c3ef3b-5349-42d8-9c3c-c395d4853bba"
sns.boxplot(x=df_inicial_test["potencia"])

# %% [markdown] id="eglFtNgxwY5o"
# ## PREGUNTA 4

# %% id="HoKuH-kNwxnI" executionInfo={"status": "ok", "timestamp": 1764867686161, "user_tz": -60, "elapsed": 7, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
## 4. Análisis de correlación inicial, ¿Hay alguna variable correlacionada?

# %% id="j2CNGJLVwxfQ" colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"status": "ok", "timestamp": 1764867687119, "user_tz": -60, "elapsed": 145, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="00ae6e8f-991a-4a96-dd7f-2cef871b1932"
## Kike Miramos correlaciones entre variables numericas. Nada a destacar, precio y km son casi opuestas
## Kike Habría que pasar las categóricas a numéricas, pero lo pide en la pregunta 6.

corr=df_inicial_test.corr(numeric_only=True)
corr.style.background_gradient(cmap='coolwarm')

# %% [markdown] id="sUJiwOPSwYuI"
# ## PREGUNTA 5

# %% id="yOfXslvFwyQB" executionInfo={"status": "ok", "timestamp": 1764867689334, "user_tz": -60, "elapsed": 73, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
## 5. Análisis variable vs target, ¿Hay algún insight interesante?

# %% id="TRZoc7WXwyJb" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1764867690192, "user_tz": -60, "elapsed": 41, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="722265eb-7d6f-42a3-81ce-39c9036bc4e6"
df_inicial_test.info()
## Kike annio está en float, debe pasarse a datetime64

# %% colab={"base_uri": "https://localhost:8080/"} id="hYF3IPBhVodO" executionInfo={"status": "ok", "timestamp": 1764868515912, "user_tz": -60, "elapsed": 76, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="3f41b690-9728-434b-da90-573ded5e1120"
## Kike Paso annio_registro a datetime
df_inicial_test["anio_registro"] = pd.to_datetime(df_inicial_test["anio_registro"])

df_inicial_test.info()

# %% colab={"base_uri": "https://localhost:8080/"} id="Fe34sDbMVbLV" executionInfo={"status": "ok", "timestamp": 1764868104853, "user_tz": -60, "elapsed": 63, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="f0fafba6-81f3-4e5b-d476-2ed44a136223"
df_inicial_test["modelo"].unique()

# %% colab={"base_uri": "https://localhost:8080/"} id="SGEwCQL1VKaF" executionInfo={"status": "ok", "timestamp": 1764868556136, "user_tz": -60, "elapsed": 45, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="0ad01ece-1c45-4fa6-9159-92b10042dc82"
## Kike listo toda las variables con sus valores únicos
for i in df_inicial_test:
  print(i,"\n\n",df_inicial_test[i].value_counts(),"\n\n")

# %% id="p95HJuElX_54" executionInfo={"status": "ok", "timestamp": 1764868794038, "user_tz": -60, "elapsed": 44, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
## Kike Defino target
target = ["precio"]

# %% colab={"base_uri": "https://localhost:8080/"} id="RUnXOW4obshE" executionInfo={"status": "ok", "timestamp": 1764871935195, "user_tz": -60, "elapsed": 84, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="15e5bb0a-1471-4a61-9c34-b76b5d8fc585"
## Kike Paso variable gps booleana a int
df_inicial_test["gps_bool"] = df_inicial_test["gps"].astype(int)
df_inicial_test.drop(columns=["gps"], inplace=True)
df_inicial_test.info()

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="bEmP1Ji2ZUMP" executionInfo={"status": "ok", "timestamp": 1764870095718, "user_tz": -60, "elapsed": 827, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="8f399fcf-1c20-441c-e320-8ff8fd8b48b8"
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


# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="mn75HSyiZy22" executionInfo={"status": "ok", "timestamp": 1764869261955, "user_tz": -60, "elapsed": 7174, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="40a3e239-3c31-4214-d51a-9b5ace83ed2a"
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

# %% id="viEM_19fwy7J" executionInfo={"status": "ok", "timestamp": 1764867537577, "user_tz": -60, "elapsed": 44, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
## Kike 6. Transformación de categóricas a numéricas, ¿Qué variables van a transformar? ¿Que técnica se va usar?

# %% id="3wVU8UiOwyyY" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1764871961806, "user_tz": -60, "elapsed": 100, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="d8f6dafa-62e5-4a05-ca11-52e0063b319b"
## Kike Miramos en un info como están las variables y después hacemos un bucle que me cambie todas las categóricas a numéricas

df_inicial_test.info()

# %% colab={"base_uri": "https://localhost:8080/"} id="O3sBMO0Denjg" executionInfo={"status": "ok", "timestamp": 1764871140595, "user_tz": -60, "elapsed": 62, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="ed91d2f3-95a4-409a-96d6-ef3a1f3f9837"
## Kike vuelvo a listar todas las variables con sus valores únicos
for i in df_inicial_test:
  print(i,"\n\n",df_inicial_test[i].value_counts(),"\n\n")

# %% colab={"base_uri": "https://localhost:8080/"} id="3I9Y9CF-e3P7" executionInfo={"status": "ok", "timestamp": 1764871978913, "user_tz": -60, "elapsed": 65, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="e2f09d6f-8745-4c56-b7ef-6b4543864f44"
for i in df_inicial_test:
    print(df_inicial_test[i].dtype.kind)


# %% id="3p4uq15lgBYH" executionInfo={"status": "ok", "timestamp": 1764870878283, "user_tz": -60, "elapsed": 7, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
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


# %% id="0nKBhix_gH4I" executionInfo={"status": "ok", "timestamp": 1764870942954, "user_tz": -60, "elapsed": 7, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
## Kike Paso al dataset las listas
lista_numericas, lista_boolean, lista_categoricas = obtener_lista_variables(df_inicial_test)

# %% colab={"base_uri": "https://localhost:8080/"} id="LoK02iXAgXsN" executionInfo={"status": "ok", "timestamp": 1764870956819, "user_tz": -60, "elapsed": 54, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="422e3ea5-890b-4fb0-a01b-11c440931521"
print (lista_categoricas)

# %% id="v_oc_x2mgzlJ"
## Kike Transformamos 11 variables categóricas
## Kike modelo (77), tipo gasolina (6), color (11), tipo coche(9), volante reg(T/F), Aire (T/F), Camara T(T/F), Asientos (T/F), elevalunas (T/F), bluetooth (T/F), Alerta (T/F)

# %% colab={"base_uri": "https://localhost:8080/"} id="l1tOUXK7gyzV" executionInfo={"status": "ok", "timestamp": 1764871544902, "user_tz": -60, "elapsed": 33, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="6f46ad23-63bc-49cf-cac7-054f7ac5e209"
for col in lista_categoricas:
    print(f"Columna: {col}\n")
    print(df_inicial_test[col].value_counts(dropna=False))  # Incluye NaN
    print("\n" + "-"*50 + "\n")


# %% id="QQlHh_Ejn4Zr" executionInfo={"status": "ok", "timestamp": 1764873536371, "user_tz": -60, "elapsed": 42, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
## Kike Quiero pasar las categoricas a numericas.
## Kike Primero paso las 7 variables T/F. Nos quedan 4 variables categóricas

cols_bool = ["volante_regulable", "aire_acondicionado", "camara_trasera", "asientos_traseros_plegables", "elevalunas_electrico", "bluetooth","alerta_lim_velocidad"]

for col in cols_bool:
    df_inicial_test[col] = np.where(df_inicial_test[col] == "TRUE", 1, 0)


# %% colab={"base_uri": "https://localhost:8080/"} id="eMbb00kpqeTU" executionInfo={"status": "ok", "timestamp": 1764873610099, "user_tz": -60, "elapsed": 73, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="a10d4255-8033-4ac0-f135-72609a98c2ce"
## Kike Creo que no me ha pasado a lista_numericas las 7 variables anteriores
## Kike Se puede hacer un ordinal encoder a tipo_gasolina, color y tipo_coche.
## Kike Pendiente de decidir que se hace con los 77 modelos.

# %% colab={"base_uri": "https://localhost:8080/"} id="nq1JJA-PpPo1" executionInfo={"status": "ok", "timestamp": 1764873538901, "user_tz": -60, "elapsed": 62, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} outputId="3220aff5-d192-4043-f3ad-58b3d286b76f"
df_inicial_test.info()

# %% [markdown] id="UhDx02LSwYLd"
# ## PREGUNTA 7

# %% id="wgjjs1z1wz1Y" executionInfo={"status": "ok", "timestamp": 1764867547911, "user_tz": -60, "elapsed": 29, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
## 7. Escalar variables (usando minmaxscaler) y luego aplicar la correlación final de variables ¿Hay alguna variable finalmente correlacionada?

# %% id="zbo1Sp3JwztQ" executionInfo={"status": "ok", "timestamp": 1764867547917, "user_tz": -60, "elapsed": 2, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}

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
