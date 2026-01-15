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

# %% [markdown] id="hIqU3MKw9HyD"
# # Master Data Science: Entregable 1

# %% [markdown] id="SLrJXOgy_CPy"
# ## Grupo: TFM 06

# %% id="lG2xfPpI9Sdk"
##  Christian Mandez
##  Joaquín Perticari
##  Enrique González
##  Juan Sánchez
##  Jordi Servat

# %% [markdown] id="u3WAdiVz-ytd"
# ## Data Engineering
# Este es el primer entregable que tenéis que realizar. En el campus está el fi chero con los datos, que contiene datos de coches de BMW.
# El objetivo es que realicéis la limpieza y el preprocesado de los datos con la fi nalidad de crear un modelo que pueda predecir el precio (¡ojo! no hay que hacer la predicción, pero es el objetivo a tener en cuenta).
#

# %% id="dfaI5Fwz-6c_"
## Os adjunto el formulario en el que están todas las preguntas y que tendréis que rellenar y mandar cuando acabéis:
## htt ps://docs.google.com/forms/d/e/1FAIpQLSfm1X41heoei-hnUWnfBucTPHQbt11txtqFFLljqF0s042kKg/viewform

# %% colab={"base_uri": "https://localhost:8080/"} id="odO6yAIxHDic" outputId="a4a51723-3fe0-4a0c-aea3-97ccde41921e" executionInfo={"status": "ok", "timestamp": 1764883757870, "user_tz": -60, "elapsed": 1110, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}}
from google.colab import drive
drive.mount('/content/drive')

# %% id="s6mjXoQtHtIu"
import pandas as pd # Librería para la manipulación y el análisis de datos
import numpy as np # Librería para la manipulación de datos y para la ejecución de operaciones matemáticas
import matplotlib.pyplot as plt # Librería para la visualización de datos
import seaborn as sns # Librería para la visualización de datos
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder # Librería para crear modelos de ML

# %% id="puRPDlKRHxJl"
bmw = pd.read_csv("/content/drive/MyDrive/bmw_pricing_v3.csv") # LEEMOS EL FICHERO

# %% colab={"base_uri": "https://localhost:8080/", "height": 226} id="dTIBUI0tH4Zr" outputId="0efbf7a9-2933-4495-b97f-3b53941f3668" executionInfo={"status": "ok", "timestamp": 1764883761800, "user_tz": -60, "elapsed": 191, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}}
bmw.head() #Hacemos head para ver las 5 primeras columnas empezando por el 0.

# %% colab={"base_uri": "https://localhost:8080/"} id="TSZacS4YIASk" outputId="b75bd1f3-132f-4598-ce1d-8e45cc82b5aa" executionInfo={"status": "ok", "timestamp": 1764883763770, "user_tz": -60, "elapsed": 27, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}}
bmw.info()  # Hacemos info, para ver el tipo de dato de cada columna

# %% colab={"base_uri": "https://localhost:8080/"} id="hDxAdXzmattk" outputId="64c9a101-8c53-4ee1-fd77-d1712282f359" executionInfo={"status": "ok", "timestamp": 1764883765782, "user_tz": -60, "elapsed": 9, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}}
bmw["tipo_gasolina"].unique() # Vemos que tantas observaciones tiene la columna Tipo_gasolina

# %% colab={"base_uri": "https://localhost:8080/", "height": 649} id="U4UjBNSpbG6M" outputId="be78fd12-4db7-41b9-d1de-378393c4b2a5" executionInfo={"status": "ok", "timestamp": 1764883767986, "user_tz": -60, "elapsed": 44, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}}
bmw.isnull().any() # Miramos en cada columna si tiene nulos o no.

# %% colab={"base_uri": "https://localhost:8080/", "height": 649} id="80w9Csv8I2u5" outputId="08bc9a04-6334-4739-92f4-966826aa0419" executionInfo={"status": "ok", "timestamp": 1764883770174, "user_tz": -60, "elapsed": 53, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}}
bmw.isnull().sum() # Miramos cuantos nulos tiene cada columna

# %% colab={"base_uri": "https://localhost:8080/"} id="hndbjhNydjOT" outputId="77134c52-a754-494e-fd8c-975ee5212c64" executionInfo={"status": "ok", "timestamp": 1764883772373, "user_tz": -60, "elapsed": 35, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}}
for i in bmw.columns: # Loop en las columnas del DataF
    print('\n',i,bmw[i].unique(),'\n\n') # Print nombre de la columna y valores únicos de ésta

# %% colab={"base_uri": "https://localhost:8080/", "height": 447} id="5fSc1o7afVc4" outputId="78a23eac-f4f8-49c2-b2fe-ef7c8f8184ff" executionInfo={"status": "ok", "timestamp": 1764883775486, "user_tz": -60, "elapsed": 227, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}}
bmw["marca"].hist() # Hacemos un Hist de marca para ver valores unicos agrupados,
# vemos que no existe otra marca. todo el documento es de la marca BMW por lo que es una columnaque no aporta valor al modelo


# %% [markdown] id="oPiXB-Wxhbgs"
# ##DUPLICADOS

# %% colab={"base_uri": "https://localhost:8080/", "height": 73} id="nbmntc-mhXGc" outputId="5413d031-123d-45b3-967e-822b4482b762" executionInfo={"status": "ok", "timestamp": 1764883777624, "user_tz": -60, "elapsed": 124, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}}
bmw[bmw.duplicated(keep=False)] ## miramos cuantos duplicados tenemos. No se aprecian observaciones duplicadas.

# %% [markdown] id="CTxLR86diGyj"
#
#

# %% [markdown] id="FVx4mLArDSsW"
# # Ejercicio 1.

# %% [markdown] id="KwQyudBHDWRJ"
# ### ¿Qué columnas eliminaron inicialmente del dataset y por qué?

# %% id="C1EGovKiDml2"
# ELIMINAR COLUMNAS: Marca + Asientos_traseros_plegables
# Marca: Es eliminado porque todo el DataSet pertenecen a la marca de BMW,
# Asientos_traseros_plegables: El 75% de las observaciones son valores nulos y no son suficientes para el modelo,
# Nos quedamos con 16 columnas.

# %% id="9GJhCcl9isAa"
bmw1 = bmw.copy()   # hacemos copia del DataF original.

# %% id="r_GnWT_li288"
del(bmw1["marca"]) # Eliminamos Marca.

# %% id="fWgA_sd7i-_M"
del(bmw1["asientos_traseros_plegables"]) # Eliminamos asientos_traseros_plegables.

# %% colab={"base_uri": "https://localhost:8080/"} id="iA34wzTF6RWK" executionInfo={"status": "ok", "timestamp": 1764883784445, "user_tz": -60, "elapsed": 48, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="8ea1d59d-e22c-43dd-a32b-47626306d27d"
bmw.shape

# %% colab={"base_uri": "https://localhost:8080/"} id="_crqVImp6ec6" executionInfo={"status": "ok", "timestamp": 1764883785684, "user_tz": -60, "elapsed": 6, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="10fb010a-c47d-4c6e-92af-acbe0b896447"
bmw1.shape

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="zCWTvAyvjkiD" outputId="38d1fb96-555e-4b61-d1c3-5e6590df2355" executionInfo={"status": "ok", "timestamp": 1764883787608, "user_tz": -60, "elapsed": 4, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}}
bmw1["modelo"].value_counts() # Vemos la proporcion de modelos.

# %% colab={"base_uri": "https://localhost:8080/", "height": 586} id="I-gCYq5Nkbve" outputId="22613c69-c99a-4b81-a571-72df4af0a78d" executionInfo={"status": "ok", "timestamp": 1764883948364, "user_tz": -60, "elapsed": 13, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}}
bmw1.isnull().sum() # Miramos cuantos nulos tiene cada columna

# %% colab={"base_uri": "https://localhost:8080/", "height": 163} id="ZVJpKGSmmMW_" outputId="c4c407ce-010b-4fce-db30-694b415ad004" executionInfo={"status": "ok", "timestamp": 1764883793508, "user_tz": -60, "elapsed": 60, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}}
bmw1[bmw1["modelo"].isnull()] # Miramos para la columna de Modelo, todos los registros en null. son tres registros.

# %% id="e9QMeooTmkwM" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1764883795016, "user_tz": -60, "elapsed": 8, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="9303255a-a779-4b5a-9725-81645c64e4df"
# ✅ CORRECTO - Obtener el primer valor de la Series mode
moda = bmw1['modelo'].mode()[0]  # [0] para obtener el valor, no la Series
bmw1['modelo'].fillna(moda, inplace=True) # Substituimos las columnas en null por la moda y no la mediana. ( sumamos los nulos a la moda )

# %% colab={"base_uri": "https://localhost:8080/", "height": 73} id="xHH4J6gMobhs" outputId="2c0ca0f8-3301-467e-a026-a14b5403993f" executionInfo={"status": "ok", "timestamp": 1764883796597, "user_tz": -60, "elapsed": 39, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}}
bmw1[bmw1["modelo"].isnull()] # Miramos para la columna de Modelo, todos los registros en null. Ya no tenemos porque han sido substituidos por la moda.

# %% colab={"base_uri": "https://localhost:8080/"} id="kxg5spNIoh6r" outputId="444fbd6e-dfd0-48dd-9df6-918a3d2ae989" executionInfo={"status": "ok", "timestamp": 1764883797347, "user_tz": -60, "elapsed": 40, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}}
print((bmw1.isnull().sum() / len(bmw1)) * 100) # Miramos la canitdad % nulos en cada columna.

# %% id="vZteHmqnxeU6"
bmw2=bmw1.copy()

# %% colab={"base_uri": "https://localhost:8080/", "height": 300} id="0yKD6tzqiA1E" executionInfo={"status": "ok", "timestamp": 1764883799895, "user_tz": -60, "elapsed": 88, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="6e70e6a3-7f10-4971-97cf-052a2d581eb2"
bmw2.describe()

# %% colab={"base_uri": "https://localhost:8080/", "height": 300} id="Bhh1CVuF7X-C" executionInfo={"status": "ok", "timestamp": 1764883801142, "user_tz": -60, "elapsed": 119, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="417b08b5-5b88-4791-861d-37fbca6183d7"
bmw2.describe(exclude=['object', 'bool'])

# %% [markdown] id="OJUp9wFY8sSg"
# Al hacer describe para las variables numéricas, surgen varios problemas:
# * el mínimo de km es -64. Valor negativo no puede ser! El máx es kms es de 1.000.000 de kms. Es demasiado alto!
# * La potencia mínima es 0, ¿Coche con 0 de potencia? Mientras que el máximo es de 423 (posible outlier)
# * El máximo de precio es de 178.500. Es demasiado alto!

# %% colab={"base_uri": "https://localhost:8080/"} id="zSiDyLbt9j0w" executionInfo={"status": "ok", "timestamp": 1764883803618, "user_tz": -60, "elapsed": 42, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="8334aa2e-c0e7-41c9-d61a-38fad84d44aa"
# Para ver valores más bajos - COLOCA UNA COLUMNA A LA VEZ
print("VALORES MÁS BAJOS DE KM:")
print(bmw2['km'].nsmallest(5))

print("\nVALORES MÁS BAJOS DE POTENCIA:")
print(bmw2['potencia'].nsmallest(5))

print("\nVALORES MÁS BAJOS DE PRECIO:")
print(bmw2['precio'].nsmallest(5))

print("\nVALORES MÁS ALTOS DE KM:")
print(bmw2['km'].nlargest(5))

print("\nVALORES MÁS ALTOS DE POTENCIA:")
print(bmw2['potencia'].nlargest(5))

print("\nVALORES MÁS ALTOS DE PRECIO:")
print(bmw2['precio'].nlargest(5))

# %% colab={"base_uri": "https://localhost:8080/"} id="iI3STKII_k_h" executionInfo={"status": "ok", "timestamp": 1764883808465, "user_tz": -60, "elapsed": 43, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="ad06113f-e80e-4bdf-a7a9-2427f92a5e41"
# Versión menos estricta
bmw2_suave = bmw2.copy()

# Solo eliminar lo OBVIAMENTE incorrecto
bmw2_suave = bmw2_suave[
    (bmw2_suave['km'] >= 0) &           # Eliminar km negativos
    (bmw2_suave['km'] <= 500000) &      # KM máximo más alto
    (bmw2_suave['potencia'] > 0) &      # Eliminar potencia 0
    (bmw2_suave['potencia'] <= 450) &   # Potencia máxima más alta
    (bmw2_suave['precio'] >= 1000) &    # Precio mínimo bajo
    (bmw2_suave['precio'] <= 150000)    # Precio máximo alto
]

print(f"Versión suave: {len(bmw2_suave)} coches válidos")

# %% id="-vSjm_IaqqQl"
#Convertimos las fechas a formato datetime.
for i in ["fecha_registro","fecha_venta"]:
    bmw2[i] = pd.to_datetime(bmw2[i])

# %% colab={"base_uri": "https://localhost:8080/"} id="LnAazI58rM5K" outputId="42030542-3b66-4d11-8afa-4f997ba777ab" executionInfo={"status": "ok", "timestamp": 1764883811109, "user_tz": -60, "elapsed": 41, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}}
bmw2_suave.info() # En el info podemos ver que; fecha_registro y fecha_venta, tienen ahora un formato Datatime ns64.

# %% colab={"base_uri": "https://localhost:8080/"} id="KaOcrpeYBOrL" executionInfo={"status": "ok", "timestamp": 1764883812758, "user_tz": -60, "elapsed": 72, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="bf09b8e6-5591-41bf-a7d7-2b137dade4e9"
# 1. PRIMERO convertir las columnas a datetime
bmw2_suave["fecha_registro"] = pd.to_datetime(bmw2_suave["fecha_registro"], errors='coerce')
bmw2_suave["fecha_venta"] = pd.to_datetime(bmw2_suave["fecha_venta"], errors='coerce')

# 2. AHORA calcular la diferencia
bmw2_suave["DIF_TIEMPO"] = bmw2_suave["fecha_venta"] - bmw2_suave["fecha_registro"]

# 3. Verificar resultado
print(bmw2_suave["DIF_TIEMPO"].head())

# %% colab={"base_uri": "https://localhost:8080/", "height": 583} id="grVXa1qvsAvg" outputId="f08b3a57-3be9-4b6c-fc45-81550a8bfda6" executionInfo={"status": "ok", "timestamp": 1764883814649, "user_tz": -60, "elapsed": 91, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}}
bmw2_suave # observamos que ya tenemos la columna agregada al DataF.

# %% id="B51EMMREAWg6"
bmw2_suave = bmw2_suave.drop(['fecha_registro', 'fecha_venta'], axis=1)

# %% id="jZxiV6HntfKj"
#Tipo_Gasolina, Remplazar "diesel" y "Diesel"
# Preguntar el Miercoles en clase: FechaRegistro: 2423, 50% nulos, Porcentaje minimo para passar a media o mediana.

# Punto1
# Marca, se elimina porque solo hai un tipo de observacion y nulos. por lo que es una columna que no aporta nada al modelo
#Asientos_Traseros_plegables, Falta de observacines, 3391 nulos. no aporta valor al modelo.

# El DataSet: NO tiene duplicados:  bmw[bmw.duplicated(keep=False)]

#Punto 3, hacer Histograma Scatter Plot, Violin Plot, Graphics.

#En las variables numericas, haremos describe.

#Al final hacer una correlacion de variables con el precio, para ver la fiabilidad del modelo.

#Análisis univarible
#- Describe() -- análisis de outliers, extremos ,
#- Hist() -- de cada uno de ellos con plt y un for columna in lista_variable_numericas

# %% colab={"base_uri": "https://localhost:8080/", "height": 486} id="B5AY1Vj3i1td" executionInfo={"status": "ok", "timestamp": 1764883822081, "user_tz": -60, "elapsed": 136, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="59c4440a-3b16-4d97-cb03-3e17526f2202"
bmw2_suave.groupby (['modelo']).agg ({'precio': 'describe'}).sort_values(by=('precio','mean'))

# %% colab={"base_uri": "https://localhost:8080/", "height": 455} executionInfo={"status": "ok", "timestamp": 1764883824349, "user_tz": -60, "elapsed": 195, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="295437cb-dc2a-481b-8560-f06e58478ed2" id="mfk7pgnMB-OK"
# Porcentaje de cada color dentro de cada modelo
porcentaje_colores = (bmw2_suave.groupby(['modelo', 'color']).size() /
                      bmw2_suave.groupby('modelo').size() * 100).unstack(fill_value=0)
porcentaje_colores.round(2)

# %% colab={"base_uri": "https://localhost:8080/"} id="oAD0HtQwC5WP" executionInfo={"status": "ok", "timestamp": 1764883825559, "user_tz": -60, "elapsed": 42, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="a6203e31-1947-490b-e6ce-1d82b1c74019"
# Esta forma siempre funciona
bmw2_suave['color'] = bmw2_suave.groupby('modelo')['color'].transform(
    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Desconocido')
)

print(f"Listo. Nulls: {bmw2_suave['color'].isnull().sum()}")

# %% colab={"base_uri": "https://localhost:8080/"} id="uou1YpuvoVoz" executionInfo={"status": "ok", "timestamp": 1764878739527, "user_tz": -60, "elapsed": 8, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="4b68edc9-0e8d-4717-9a1a-20b4acbcf229"
#Desnormalizar.
np.exp(13-1)

# %% id="2miTiXrDobH0"
#TRANSFORM

# %% id="CWFBPS_JTH3x" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1764883828282, "user_tz": -60, "elapsed": 8, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="1fe95567-0bde-40af-c353-2f4821f810e0"
bmw2_suave["tipo_gasolina"].fillna(bmw2_suave["tipo_gasolina"].mode()[0], inplace=True)

# %% id="jAPMuhYh1SXu" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1764883829295, "user_tz": -60, "elapsed": 114, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="3dd040b6-60df-4923-d01f-909fc068fc87"
bmw2_suave["volante_regulable"].fillna(bmw2_suave["volante_regulable"].mode()[0], inplace=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 554} id="qJnomBsEESKE" executionInfo={"status": "ok", "timestamp": 1764883830507, "user_tz": -60, "elapsed": 5, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="2ba010f0-3980-4586-cad0-9cd9aa9f6782"
bmw2_suave.isnull().sum()

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1764883832551, "user_tz": -60, "elapsed": 7, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="1a9be0a6-df6b-42d3-afba-cf7e8e1f5987" id="elGfLRdL1Z30"
bmw2_suave["camara_trasera"].fillna(bmw2_suave["camara_trasera"].mode()[0], inplace=True)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1764883833946, "user_tz": -60, "elapsed": 111, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="3050df06-ff41-4187-f1c5-b7076f18c01e" id="xRhY2gt51hJM"
bmw2_suave["elevalunas_electrico"].fillna(bmw2_suave["elevalunas_electrico"].mode()[0], inplace=True)

# %% id="EJGc2gr-1zMk"
bmw2_suave['km'] = pd.to_numeric(bmw2_suave['km'], errors='coerce')

# %% colab={"base_uri": "https://localhost:8080/"} id="byoz-Ow43mGs" executionInfo={"status": "ok", "timestamp": 1764582844235, "user_tz": -60, "elapsed": 11, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="ddaad0f0-79d0-4c38-8e9f-9d5d0a16a76b"
#Sin ejecutar esta función, al quitar outliers, también se me han limpiado los valores de kms nulls.


media_km = bmw2['km'].mean()
bmw2['km'].fillna(media_km, inplace=True)


# %% id="i28Bm3T_4Aiy"
bmw2_suave['potencia'] = pd.to_numeric(bmw2_suave['potencia'], errors='coerce')

# %% colab={"base_uri": "https://localhost:8080/"} id="CNHhR2YX4E2x" executionInfo={"status": "ok", "timestamp": 1764582847461, "user_tz": -60, "elapsed": 27, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="b235e69c-dbcf-4e28-a422-c18d110f86bc"
#Sin ejecutar esta función, al quitar outliers, también se me han limpiado los valores de kms nulls.


media_km = bmw2['potencia'].mean()
bmw2['potencia'].fillna(media_km, inplace=True)


# %% id="9qyS-uX0DBW1" colab={"base_uri": "https://localhost:8080/", "height": 159} executionInfo={"status": "error", "timestamp": 1764763125580, "user_tz": -60, "elapsed": 56, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="5159c006-2d8d-409c-d769-0ad7c57b6cd0"
df['Postal_code.Mean_price']=df.groupby(['Postal Code'])['Price'].transform('mean')

# %% id="YTO2_MXJDXEA" colab={"base_uri": "https://localhost:8080/", "height": 211} executionInfo={"status": "error", "timestamp": 1764763127631, "user_tz": -60, "elapsed": 34, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="f9a7b4f9-3b3c-4237-82d8-cf4dc7644bc2"
df['Price']=np.where(
    df['Price'].isna(),
    df['Postal Code Mean Price'],
    df['Price']
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 554} id="MW3XoqAo05p8" executionInfo={"status": "ok", "timestamp": 1764883840674, "user_tz": -60, "elapsed": 56, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="fcc574d4-a2b9-4333-b185-0250d5b34cfa"
bmw2_suave.isnull().sum()

# %% colab={"base_uri": "https://localhost:8080/", "height": 447} id="fMZAUsb21Jbq" executionInfo={"status": "ok", "timestamp": 1764878772458, "user_tz": -60, "elapsed": 347, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="d7861570-7ef4-4e4f-b013-6bc64734494e"
bmw["color"].hist()

# %% colab={"base_uri": "https://localhost:8080/", "height": 178} id="XVKkzxMUXMv_" executionInfo={"status": "ok", "timestamp": 1764883844399, "user_tz": -60, "elapsed": 38, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="72f1ef75-4b7e-4ad7-cf35-bb73833f9a0d"
bmw2_suave["alerta_lim_velocidad"].value_counts()

# %% colab={"base_uri": "https://localhost:8080/", "height": 366} id="pbwYOAKvYrIQ" executionInfo={"status": "ok", "timestamp": 1764883845333, "user_tz": -60, "elapsed": 177, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="633c621a-db28-49fa-bfc7-13b758fb5b03"
bmw2_suave["tipo_coche"].value_counts()

# %% id="ZcaXJKjKR9U_"
# Para tipo_coche, como no hay una distribución clara, rellenamos los nulls con el valor "Desconocido"

bmw2_suave['tipo_coche'] = bmw2_suave['tipo_coche'].fillna('Desconocido')

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="fcBHLl7xSM9N" executionInfo={"status": "ok", "timestamp": 1764883851259, "user_tz": -60, "elapsed": 60, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="0d9b6ca3-b8d5-471c-fa33-4fb2aa9b83c9"
bmw2_suave.value_counts()

# %% colab={"base_uri": "https://localhost:8080/", "height": 178} id="khyKQ2zzGRb_" executionInfo={"status": "ok", "timestamp": 1764883858667, "user_tz": -60, "elapsed": 8, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="ad9830be-b614-4a51-ec54-cb0386382272"
bmw2_suave["aire_acondicionado"].value_counts()

# %% colab={"base_uri": "https://localhost:8080/"} id="Fnv4GsxgMmY8" executionInfo={"status": "ok", "timestamp": 1764884015571, "user_tz": -60, "elapsed": 16, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="6a934494-d8c2-465e-8809-013ebe948ac1"
# Porcentaje de coches CON aire acondicionado por modelo
porcentaje_aire = bmw2_suave.groupby('modelo')['aire_acondicionado'].mean() * 100
print("Porcentaje con aire acondicionado por modelo:")
print(porcentaje_aire.sort_values(ascending=False))

# %% colab={"base_uri": "https://localhost:8080/"} id="9167Mve4Rg4D" executionInfo={"status": "ok", "timestamp": 1764884020340, "user_tz": -60, "elapsed": 46, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="d10e210d-1b2f-4884-d5cc-9629b86f0592"
# Código fácil para transformar aire_acondicionado

# 1. Primero mira qué valores tienes
print("Valores que hay ahora:")
print(bmw2_suave['aire_acondicionado'].unique())

# 2. Transforma todo
bmw2_suave['aire_acondicionado'] = bmw2_suave['aire_acondicionado'].map({
    True: 1,
    False: 0
})

# 3. Cambia los que están vacíos por -1
bmw2_suave['aire_acondicionado'] = bmw2_suave['aire_acondicionado'].fillna(-1)

# 5. Verifica que quedó bien
print("\nDespués del cambio:")
print(bmw2_suave['aire_acondicionado'].value_counts())
print("\n1 = True, 0 = False, -1 = Desconocido")

# %% colab={"base_uri": "https://localhost:8080/"} id="0hNjEctVNJ6j" executionInfo={"status": "ok", "timestamp": 1764883867016, "user_tz": -60, "elapsed": 128, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="d8da0533-42b9-405d-a267-afd884cc49de"
# Porcentaje de coches CON Bluetooth por modelo
porcentaje_bluetooth = bmw2_suave.groupby('modelo')['bluetooth'].mean() * 100
print("Porcentaje con bluetooth por modelo:")
print(porcentaje_bluetooth.sort_values(ascending=False))

# %% colab={"base_uri": "https://localhost:8080/", "height": 694} id="URA9NwsNOK8F" executionInfo={"status": "ok", "timestamp": 1764883872968, "user_tz": -60, "elapsed": 3986, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="1b75f5cf-893e-4160-af63-d29ab2a7403b"
plt.figure(figsize=(14, 6))
sns.violinplot(x='modelo', y='bluetooth', data=bmw2, inner='stick', palette='Set2')

# Mejorar visualización
plt.title('Distribución de Bluetooth por modelo BMW', fontsize=14)
plt.xlabel('Modelo', fontsize=12)
plt.ylabel('Bluetooth (0=No, 1=Sí)', fontsize=12)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="DLrFN746Ni4-" executionInfo={"status": "ok", "timestamp": 1764883873068, "user_tz": -60, "elapsed": 98, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="2ae4c83e-9c09-42ed-9b59-6d8a9358e54f"
# Rellenar NaN con el valor más frecuente de cada modelo
bmw2_suave['bluetooth'] = bmw2_suave.groupby('modelo')['bluetooth'].transform(
    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else False)
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 178} id="s6mnBiffGuFW" executionInfo={"status": "ok", "timestamp": 1764883874405, "user_tz": -60, "elapsed": 40, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="1a864df2-e997-4c3b-f596-cb635ad55c2a"
bmw2_suave["bluetooth"].value_counts()

# %% colab={"base_uri": "https://localhost:8080/", "height": 210} id="N7p12TeUHABC" executionInfo={"status": "ok", "timestamp": 1764883876629, "user_tz": -60, "elapsed": 32, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="9e943d9a-6364-451c-cdcc-54543d6c4635"
bmw2_suave["alerta_lim_velocidad"].value_counts(dropna=False)

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="kone-QhuH7Fo" executionInfo={"status": "ok", "timestamp": 1764883877692, "user_tz": -60, "elapsed": 58, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="57778336-da8e-42fc-9653-4bcdafe5c2fe"
bmw2_suave.groupby('alerta_lim_velocidad', dropna=False).agg({'precio': 'describe'})

# %% colab={"base_uri": "https://localhost:8080/", "height": 694} id="eznEZbe_PJJO" executionInfo={"status": "ok", "timestamp": 1764883882342, "user_tz": -60, "elapsed": 2636, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="25c2133b-7a15-4b20-e6e3-725cddc4fff4"
plt.figure(figsize=(14, 6))
sns.violinplot(x='modelo', y='alerta_lim_velocidad', data=bmw2, inner='stick', palette='Set2')

# Mejorar visualización
plt.title('Distribución de Alerta_límite_velocidad por modelo BMW', fontsize=14)
plt.xlabel('Modelo', fontsize=12)
plt.ylabel('Alerta_lim_velocidad (0=No, 1=Sí)', fontsize=12)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# %% colab={"base_uri": "https://localhost:8080/"} id="l6xRtTvEM8GI" executionInfo={"status": "ok", "timestamp": 1764883882357, "user_tz": -60, "elapsed": 13, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="0c1d0cf3-3464-462a-9a87-23f8901f080a"
# Código fácil para transformar alerta_lim_velocidad

# 1. Primero mira qué valores tienes
print("Valores que hay ahora:")
print(bmw2_suave['alerta_lim_velocidad'].unique())

# 2. Transforma todo
bmw2_suave['alerta_lim_velocidad'] = bmw2_suave['alerta_lim_velocidad'].map({
    True: 1,
    False: 0
})

# 3. Cambia los que están vacíos por -1
bmw2_suave['alerta_lim_velocidad'] = bmw2_suave['alerta_lim_velocidad'].fillna(-1)

# 5. Verifica que quedó bien
print("\nDespués del cambio:")
print(bmw2_suave['alerta_lim_velocidad'].value_counts())
print("\n1 = True, 0 = False, -1 = Desconocido")

# %% colab={"base_uri": "https://localhost:8080/", "height": 447} id="7E4jBPu-BJQO" executionInfo={"status": "ok", "timestamp": 1764883882840, "user_tz": -60, "elapsed": 186, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="ff8c4438-60da-43df-d8d3-be999ad276c4"
bmw2["tipo_coche"].hist()

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="I5_cSwPIBVEU" executionInfo={"status": "ok", "timestamp": 1764883884290, "user_tz": -60, "elapsed": 129, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="510d8615-6366-455f-d4ee-5b4d03973737"
bmw2_suave.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 394} executionInfo={"status": "ok", "timestamp": 1764883886010, "user_tz": -60, "elapsed": 58, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="7c3940da-639c-45bc-99ba-e20e5ca0474e" id="O2DOnMKxDmtE"
bmw2_suave.groupby (['tipo_coche']).agg ({'precio': 'describe'}).sort_values(by=('precio','mean'))

# %% id="3Fxm8MzWVvxH" colab={"base_uri": "https://localhost:8080/", "height": 547} executionInfo={"status": "error", "timestamp": 1764883887381, "user_tz": -60, "elapsed": 13, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="8c7533a2-f21d-4471-e6f7-dd67705a88ac"
# Le imputamos la el promedio de antigüedad a la variable DIF_TIEMPO, y eliminamos las otras dos variables.
del (bmw2_suave['fecha_registro'])

# %% id="dUV7LJogWa84" executionInfo={"status": "error", "timestamp": 1764883889815, "user_tz": -60, "elapsed": 34, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} colab={"base_uri": "https://localhost:8080/", "height": 547} outputId="f84cd747-98e7-485c-dfb3-f4b96d631070"
del (bmw2_suave['fecha_venta'])

# %% colab={"base_uri": "https://localhost:8080/"} id="g4QL-pZnWfp1" executionInfo={"status": "ok", "timestamp": 1764883891420, "user_tz": -60, "elapsed": 15, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="98e67436-57c7-4579-ff85-355bfe706789"
bmw2_suave['DIF_TIEMPO'].fillna(bmw2_suave['DIF_TIEMPO'].mean(), inplace=True)

# %% id="OsMKRpTmacQH"
#Después de haber limpiado todos los valores nulls de nuestra tabla, vamos a proceder a hacer el análisis univariable.



# %% [markdown] id="VpzLdXdTaqnE"
# # **3. Análisis univariable, explicar alguna información interesante encontrada**
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="L1CkL4WtViSV" executionInfo={"status": "ok", "timestamp": 1764878997931, "user_tz": -60, "elapsed": 36, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="06c57b58-46fa-4c1e-c160-3ded3a5fc328"
bmw2_suave.describe()

# %% colab={"base_uri": "https://localhost:8080/", "height": 429} id="_hSvFmkcVpni" executionInfo={"status": "ok", "timestamp": 1764883896187, "user_tz": -60, "elapsed": 94, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="f9bb751c-cee0-4cc7-8c48-34cd4d765f64"
bmw2_suave['km'].quantile(np.arange(0, 1.1, 0.1))

# %% colab={"base_uri": "https://localhost:8080/", "height": 429} id="56v8jZ50WIim" executionInfo={"status": "ok", "timestamp": 1764883898164, "user_tz": -60, "elapsed": 81, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="84cec6c4-763e-4ea1-9cb5-9cd7f40fa14b"
bmw2_suave['potencia'].quantile(np.arange(0, 1.1, 0.1))

# %% colab={"base_uri": "https://localhost:8080/", "height": 429} id="8EFr02JXWLbk" executionInfo={"status": "ok", "timestamp": 1764883900375, "user_tz": -60, "elapsed": 99, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="a7961a46-cf1d-4f77-e47b-a90fd45c86d1"
bmw2_suave['precio'].quantile(np.arange(0, 1.1, 0.1))

# %% colab={"base_uri": "https://localhost:8080/", "height": 241} id="EQ1l2VCZWMeC" executionInfo={"status": "ok", "timestamp": 1764883901990, "user_tz": -60, "elapsed": 85, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="0e378b57-a941-4cf3-9b0c-bf3e3556abc3"
bmw2_suave.DIF_TIEMPO.nsmallest(5)

# %% id="oJi265pNXLiZ" executionInfo={"status": "ok", "timestamp": 1764883903811, "user_tz": -60, "elapsed": 126, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} colab={"base_uri": "https://localhost:8080/"} outputId="a8454c97-e510-469a-d5d0-601897a6a42c"
# Convertir todos los valores negativos a positivos
bmw2_suave['DIF_TIEMPO'] = bmw2_suave['DIF_TIEMPO'].abs()

print("✅ Aplicado: Valor absoluto a DIF_TIEMPO")
print(f"Mínimo después: {bmw2_suave['DIF_TIEMPO'].min()}")

# %% id="-BInoK8xb842"
# Después de hacer describe, han salido varios valores para las variables numéricas, y podemos ver que el dataset presenta outliers, que son valores muy diferenciados de resto de valores como por ejemplo:
# - El precio para el máximo es casi 12 veces la media, y el minimo es 15.000 veces más bajo.
# - La diferencia de tiempo es 5 veces la media, mientras que el mínimo presenta -2000 días, da un resultado negativo, y eso no puede ser.
# - La potencia minima de 0.00, valor no posible dado que sino el coche no funcionaría. El máximo no tiene un valor exagerado, entendiendo que los coches BMW pueden tener potencias altas.
# - El kilometraje no puede presentar valores negativos. En su minimo los presenta.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="MLNHKfLNavhD" executionInfo={"status": "ok", "timestamp": 1764883907003, "user_tz": -60, "elapsed": 1590, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="a3b67b49-47b2-4b06-ed5c-f8aa42f05cf7"
# Saco los valores por cada columna con value_counts con un bucle
for i in bmw2_suave:
  if bmw2_suave[i].dtype.kind=="O":
    #imprime value_counts de variables categóricas
    print("\n",bmw2_suave[i].value_counts(),"\n")
  elif (bmw2_suave[i].dtype.kind=="f") or (bmw2_suave[i].dtype.kind=="i"):
    #imprime histograma de variables numericas
    print("\n",bmw2_suave.hist(i),"\n")


# %% id="6VUDEIAISeUs"
# Para las variables categóricas haremos gráficos de barras para ver su distribución.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="SmYekzs5TRLz" executionInfo={"status": "ok", "timestamp": 1764883908986, "user_tz": -60, "elapsed": 847, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="09b6ecb4-536f-4ade-b258-fbf393b67d22"
# Para COLOR
plt.figure(figsize=(12, 6))
bmw2_suave['color'].value_counts().head(10).plot(kind='bar', color='skyblue')
plt.title('Top 10 Colores más comunes en BMW', fontsize=14)
plt.xlabel('Color')
plt.ylabel('Cantidad de coches')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Para MODELO
plt.figure(figsize=(14, 6))
bmw2_suave['modelo'].value_counts().head(15).plot(kind='bar', color='lightcoral')
plt.title('Top 15 Modelos más comunes de BMW', fontsize=14)
plt.xlabel('Modelo')
plt.ylabel('Cantidad de coches')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Para TIPO_GASOLINA
plt.figure(figsize=(10, 6))
bmw2_suave['tipo_gasolina'].value_counts().plot(kind='bar', color='lightgreen')
plt.title('Distribución de Tipo de Gasolina', fontsize=14)
plt.xlabel('Tipo de Gasolina')
plt.ylabel('Cantidad de coches')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% id="PAXJqplYU-gi"

# %% [markdown] id="befa5CVygM62"
# # **4. Análisis de correlación inicial, ¿Hay alguna variable correlacionada?**

# %% colab={"base_uri": "https://localhost:8080/", "height": 363} id="wKA-5Hw6gaOk" executionInfo={"status": "ok", "timestamp": 1764883911582, "user_tz": -60, "elapsed": 46, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="56de9e95-fe36-46b4-8bc9-eb51e68fb00e"
corr = bmw2_suave.corr(numeric_only = True)
corr

# %% colab={"base_uri": "https://localhost:8080/", "height": 432} id="CRri_zV1iPX-" executionInfo={"status": "ok", "timestamp": 1764884215194, "user_tz": -60, "elapsed": 63, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="af7a1957-374d-461f-cff9-f1d97f62a240"
corr.style.background_gradient(cmap='coolwarm')

# %% id="xw8nt6NUcSuQ"
# Lo que podemos comprobar en el siguiente mapa de correlación y analizando sobretodo las variables numéricas, lo que podemos comprobar es que los kilometros presentan una correlación negativa con
# respecto al precio, es decir, que cuando el coche aumenta los kilométros recorridos, el precio de éste disminuye. Por otro lado, la potencia tiene una correlación bastante alta con el precio,
# entendiendóse que contra más potencia tiene el coche, más alto será su precio. Tenemos también algunas de las variables booleanas que nos han salido con una correlación con el precio de
# aproximadamente 0,25; lo que, nos da a entender que tienen una pequeña correlación, y que, cuando disponemos de algunos de estos artilugios en el coche, su precio será mayor.

# %% id="YzrYpaYpWUSa"
# A continuación hacemos gráficos scatterplot con las variables más relacionadas que tenemos en el mapa de correlación:
# Por un lado km con precio y por el otro potencia con precio.

# %% colab={"base_uri": "https://localhost:8080/", "height": 564} id="yYXT4qeAYyPw" executionInfo={"status": "ok", "timestamp": 1764879081970, "user_tz": -60, "elapsed": 272, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="b46cec0c-8115-4676-fdd7-74945f8ea29a"
# Gráfico 1
plt.figure(figsize=(10, 6))
plt.scatter(bmw2_suave['km'], bmw2_suave['precio'], alpha=0.5, s=20)
plt.title('Precio vs Kilómetros')
plt.xlabel('Kilómetros')
plt.ylabel('Precio (€)')
plt.grid(True, alpha=0.3)
plt.show()

# %% id="jxlllr52ZAAd"
# En este caso, podemos ver una clara correlación negativa, cuando tenemos menos kilómetros hechos es cuando el precio más aumenta.
# Sin embargo, cuando tenemos más kilómetros recorridos, es cuando podemos encontrar un precio más bajo.

# %% colab={"base_uri": "https://localhost:8080/", "height": 564} id="7SpTZBvWY6iu" executionInfo={"status": "ok", "timestamp": 1764852218325, "user_tz": -60, "elapsed": 457, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="f1d56409-041f-4eb7-c5dc-09eb0fdcfca4"
# Gráfico 2
plt.figure(figsize=(10, 6))
plt.scatter(bmw2_suave['potencia'], bmw2_suave['precio'], alpha=0.5, s=20, color='red')
plt.title('Potencia vs Precio')
plt.xlabel('Potencia (CV)')
plt.ylabel('Precio (€)')
plt.grid(True, alpha=0.3)
plt.show()

# %% id="oJmfOgEMZPqT"
# En este caso, podemos comprobar como al tener de más potencia, tenemos un precio más elevado. Garantizamos que hay presente
# una correlación positiva entre potencia y precio.

# %% id="q13COBH5ihra"
# Las demás variables no son numéricas, sino booleanas, sin embargo, las pasamos a categoricas para hacer el mapa de correlación.

# %% colab={"base_uri": "https://localhost:8080/"} id="MzpvXCoyde_h" executionInfo={"status": "ok", "timestamp": 1764879087791, "user_tz": -60, "elapsed": 10, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="dbd4a23b-f867-4408-d1d7-d1a0f3977277"
# Lista de todas las variables booleanas
variables_booleanas = ['volante_regulable', 'camara_trasera', 'elevalunas_electrico',
                       'bluetooth', 'gps', 'alerta_lim_velocidad']

# Análisis para cada variable
for variable in variables_booleanas:
    print(f"\n📊 ANÁLISIS DE {variable.upper()}")
    print("="*40)

    # 1. Filtrar solo True (1) y False (0)
    if variable == 'alerta_lim_velocidad':
        datos_filtrados = bmw2_suave[bmw2_suave[variable].isin([0, 1])]
    else:
        # Para otras booleanas, True=1, False=0
        datos_filtrados = bmw2_suave

    # 2. Contar cuántos hay de cada
    print(f"\nCantidad de coches:")
    print(f"False (0): {(datos_filtrados[variable] == 0).sum()}")
    print(f"True (1): {(datos_filtrados[variable] == 1).sum()}")

    # 3. Porcentajes
    total = len(datos_filtrados)
    print(f"\nPorcentajes:")
    print(f"False: {(datos_filtrados[variable] == 0).sum()/total*100:.1f}%")
    print(f"True: {(datos_filtrados[variable] == 1).sum()/total*100:.1f}%")

    # 4. Ver precio promedio
    print(f"\n💰 Precio promedio:")
    print(f"False: €{datos_filtrados[datos_filtrados[variable] == 0]['precio'].mean():,.0f}")
    print(f"True: €{datos_filtrados[datos_filtrados[variable] == 1]['precio'].mean():,.0f}")

    print("-" * 40)

# %% colab={"base_uri": "https://localhost:8080/", "height": 143} id="gUm9UjthfMkh" executionInfo={"status": "ok", "timestamp": 1764879755336, "user_tz": -60, "elapsed": 114, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="cf014f8e-b321-461c-999f-135e19bf4088"
bmw2_suave.loc[bmw2_suave['volante_regulable'].isin([0,1])].groupby(['volante_regulable'])['precio'].describe()

# %% colab={"base_uri": "https://localhost:8080/", "height": 278} id="1CMj2GSjqgNZ" executionInfo={"status": "ok", "timestamp": 1764838141253, "user_tz": -60, "elapsed": 54, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="873a7d57-6319-496d-8725-b11cd72b2227"
bmw2_suave.head()

# %% colab={"base_uri": "https://localhost:8080/"} id="Zt7hTn17aSG3" executionInfo={"status": "ok", "timestamp": 1764838143980, "user_tz": -60, "elapsed": 35, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="949f2c20-bb29-4f51-cbf0-ecb036f93a2e"
bmw2_suave.info()

# %% colab={"base_uri": "https://localhost:8080/"} id="EWIhTHjUhd3_" executionInfo={"status": "ok", "timestamp": 1764879751568, "user_tz": -60, "elapsed": 8, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="886ed3f3-4e39-4b5f-931f-aa0cda1931f5"
print("=== DISTRIBUCIÓN DE ALERTA_LIM_VELOCIDAD ===")
distribucion = bmw2_suave['alerta_lim_velocidad'].value_counts().sort_index()
porcentajes = bmw2_suave['alerta_lim_velocidad'].value_counts(normalize=True).sort_index() * 100

print("\nValor | Cantidad | Porcentaje")
print("-" * 35)
for valor in distribucion.index:
    print(f"{valor:5} | {distribucion[valor]:8} | {porcentajes[valor]:.1f}%")

print(f"\nTotal registros: {len(bmw2_suave)}")

# %% id="S-_rapwUdMOv" colab={"base_uri": "https://localhost:8080/", "height": 394} executionInfo={"status": "ok", "timestamp": 1764838287780, "user_tz": -60, "elapsed": 84, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="16aaa225-4526-4da4-e516-88f894e0ece5"
bmw2_suave.groupby (['tipo_coche']).agg ({'precio': 'describe'}).sort_values(by=('precio','mean'))

# %% colab={"base_uri": "https://localhost:8080/", "height": 457} executionInfo={"status": "ok", "timestamp": 1764849956160, "user_tz": -60, "elapsed": 96, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="a685f1c6-c1d4-476d-f918-9941237828e7" id="elFyH63Wj2cI"
bmw2_suave.groupby (['color']).agg ({'precio': 'describe'}).sort_values(by=('precio','mean'))

# %% colab={"base_uri": "https://localhost:8080/", "height": 269} executionInfo={"status": "ok", "timestamp": 1764879746542, "user_tz": -60, "elapsed": 46, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="1d80265b-9989-4e36-bfa7-f65b30376254" id="H9V1TCKbj834"
bmw2_suave.groupby (['tipo_gasolina']).agg ({'precio': 'describe'}).sort_values(by=('precio','mean'))

# %% id="LAHZHYL7kHRz"
# Habiendo sacado el tipo de gasolina, podemos ver que Diesel (en el que solamente hay 5 registros) y diesel son lo mismo, con lo cual, las agrupamos.

# %% colab={"base_uri": "https://localhost:8080/"} id="wTM98n5Do0eI" executionInfo={"status": "ok", "timestamp": 1764839610208, "user_tz": -60, "elapsed": 47, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="cb263f73-6537-4255-973e-2992199aebd4"
# Convertir TODO a minúsculas
bmw2_suave['tipo_gasolina'] = bmw2_suave['tipo_gasolina'].str.lower().str.strip()

print("✅ Convertido todo a minúsculas")
print(bmw2_suave['tipo_gasolina'].value_counts())

# %% colab={"base_uri": "https://localhost:8080/", "height": 226} id="3L-i7HldyVPG" executionInfo={"status": "ok", "timestamp": 1764881455507, "user_tz": -60, "elapsed": 72, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="cca29ced-745b-4e03-c5df-c5e195e9693d"
bmw2.head(5)

# %% [markdown] id="PwjBt_NNornG"
# # **5. Análisis variable vs target, ¿Hay algún insight interesante?**
#

# %% colab={"base_uri": "https://localhost:8080/"} id="5AIF1Laoo63s" executionInfo={"status": "ok", "timestamp": 1764884049071, "user_tz": -60, "elapsed": 43, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="d90abe79-802c-4389-bfb8-838ba5b7eb40"
print("💰 ¿QUÉ HACE CARO UN BMW?")
print("=" * 30)

# 1. MÁS POTENCIA = MÁS CARO
precio_alta_potencia = bmw2_suave[bmw2_suave['potencia'] > 200]['precio'].mean()
precio_baja_potencia = bmw2_suave[bmw2_suave['potencia'] < 150]['precio'].mean()

print("\n1. POTENCIA:")
print(f"   Alta potencia: €{precio_alta_potencia:,.0f}")
print(f"   Baja potencia: €{precio_baja_potencia:,.0f}")
print(f"   → Más potencia = Más caro")

# 2. MENOS KILÓMETROS = MÁS CARO
precio_pocos_km = bmw2_suave[bmw2_suave['km'] < 50000]['precio'].mean()
precio_muchos_km = bmw2_suave[bmw2_suave['km'] > 150000]['precio'].mean()

print("\n2. KILÓMETROS:")
print(f"   Pocos km: €{precio_pocos_km:,.0f}")
print(f"   Muchos km: €{precio_muchos_km:,.0f}")
print(f"   → Menos km = Más caro")

# 3. TODOS LOS EXTRAS
print("\n3. EXTRAS:")

extras = [
    ('GPS', 'gps'),
    ('Bluetooth', 'bluetooth'),
    ('Volante regulable', 'volante_regulable'),
    ('Cámara trasera', 'camara_trasera'),
    ('Elevalunas eléctrico', 'elevalunas_electrico'),
    ('Alerta límite velocidad', 'alerta_lim_velocidad')
]

for nombre_extra, columna_extra in extras:
    precio_con = bmw2_suave[bmw2_suave[columna_extra] == True]['precio'].mean()
    precio_sin = bmw2_suave[bmw2_suave[columna_extra] == False]['precio'].mean()

    print(f"\n   Con {nombre_extra}: €{precio_con:,.0f}")
    print(f"   Sin {nombre_extra}: €{precio_sin:,.0f}")

    if precio_con > precio_sin:
        print(f"   → Tener {nombre_extra} = Más caro")
    else:
        print(f"   → Tener {nombre_extra} = Más barato")

# RESUMEN
print("\n" + "=" * 30)
print("📌 RESUMEN:")
print("=" * 30)
print("Un BMW es más caro si:")
print("✅ Tiene más potencia")
print("✅ Tiene menos kilómetros")
print("✅ Tiene extras (todos los mencionados)")

# %% id="6pcmydOPq1b4"
# Podemos llegar a la conclusión que un BMW será más caro contra más potencia tenga, contra menos kilometraje tenga, y
# contra más extras tenga, a excepción del GPS.

# %% [markdown] id="s9KyQG9QrY2Z"
# # **6. Transformación de categóricas a numéricas, ¿Qué variables van a transformar? ¿Que técnica se va usar?**
#
#

# %% id="vhCfEAoFFzfs"
# Haremos normalize con las funciones que aparezcan como tipo object y bool en nuestro dataset. Estas serán las que trataremos como categóricas.

# %% id="LfNeOvyEKUDS"
bmw3_suave=bmw2_suave.copy()

# %% colab={"base_uri": "https://localhost:8080/"} id="Z3_NTKytFVlq" executionInfo={"status": "ok", "timestamp": 1764884064880, "user_tz": -60, "elapsed": 37, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="da61db66-c718-4a1f-fc06-e3c3feea7b14"
bmw3_suave.info()


# %% colab={"base_uri": "https://localhost:8080/", "height": 272} id="PPCzkSUAFL3D" executionInfo={"status": "ok", "timestamp": 1764884069238, "user_tz": -60, "elapsed": 8, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="a14b3008-b57f-4f5b-ee10-b401b80570f3"
bmw3_suave.tipo_gasolina.value_counts(normalize=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 460} id="2y7CzMwdFyec" executionInfo={"status": "ok", "timestamp": 1764884071257, "user_tz": -60, "elapsed": 5, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="19108f7e-e6a8-4feb-e4c6-32c85f2b4948"
bmw3_suave.color.value_counts(normalize=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 398} id="zPOhfc17GC7x" executionInfo={"status": "ok", "timestamp": 1764884072778, "user_tz": -60, "elapsed": 42, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="a40f62d2-26c6-43c0-9f05-3ccf75ba575e"
bmw3_suave.tipo_coche.value_counts(normalize=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 147} id="hg3BT67pGMkg" executionInfo={"status": "ok", "timestamp": 1764882055716, "user_tz": -60, "elapsed": 43, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="72c02f73-3fe6-423c-9d84-cf6aa7fe254d"
bmw3_suave.volante_regulable.value_counts(normalize=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 178} id="-3dRQ3CpGQE-" executionInfo={"status": "ok", "timestamp": 1764884074541, "user_tz": -60, "elapsed": 5, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="b2021c40-808c-42e2-818a-de1e87af2693"
bmw3_suave.camara_trasera.value_counts(normalize=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 178} id="-JBGajOyG9XQ" executionInfo={"status": "ok", "timestamp": 1764884076916, "user_tz": -60, "elapsed": 5, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="bf075e36-3c8b-4ed0-880d-d8f79288a7cf"
bmw3_suave.elevalunas_electrico.value_counts(normalize=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 178} id="jSm29mF-HCaH" executionInfo={"status": "ok", "timestamp": 1764884078562, "user_tz": -60, "elapsed": 6, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="36cceab7-0832-4321-8f27-27753fc4d8bc"
bmw3_suave.bluetooth.value_counts(normalize=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 178} id="QSYorigSHGJ-" executionInfo={"status": "ok", "timestamp": 1764884080295, "user_tz": -60, "elapsed": 105, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="5bb24570-4fea-4247-9156-f267d130515e"
bmw3_suave.gps.value_counts(normalize=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 258} id="6PSAwlGFIM_K" executionInfo={"status": "ok", "timestamp": 1764884081569, "user_tz": -60, "elapsed": 181, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="fabf2f99-9f7f-4b1d-eff9-b4edcfec0151"
bmw3_suave.head(5)

# %% id="yC_tIQLnIzYN"
# Convertir booleano a entero (True=1, False=0)
bmw3_suave['volante_regulable'] = bmw2_suave['volante_regulable'].astype(int)

# O con np.where
bmw3_suave['volante_regulable'] = np.where(
    bmw2_suave['volante_regulable'] == True,  # Sin comillas
    1,
    0
)

# %% id="3_5z9_QWJVOM"
# Convertir booleano a entero (True=1, False=0)
bmw3_suave['camara_trasera'] = bmw2_suave['camara_trasera'].astype(int)

# O con np.where
bmw3_suave['camara_trasera'] = np.where(
    bmw2_suave['camara_trasera'] == True,  # Sin comillas
    1,
    0
)

# %% id="qjNxgLtcJjO6"
# Convertir booleano a entero (True=1, False=0)
bmw3_suave['elevalunas_electrico'] = bmw3_suave['elevalunas_electrico'].astype(int)

# O con np.where
bmw3_suave['elevalunas_electrico'] = np.where(
    bmw3_suave['elevalunas_electrico'] == True,  # Sin comillas
    1,
    0
)

# %% id="ApGw5eHSJwKh"
# Convertir booleano a entero (True=1, False=0)
bmw3_suave['bluetooth'] = bmw3_suave['bluetooth'].astype(int)

# O con np.where
bmw3_suave['bluetooth'] = np.where(
    bmw3_suave['bluetooth'] == True,  # Sin comillas
    1,
    0
)

# %% id="6Jp9aO34J73m"
# Convertir booleano a entero (True=1, False=0)
bmw3_suave['gps'] = bmw3_suave['gps'].astype(int)

# O con np.where
bmw3_suave['gps'] = np.where(
    bmw3_suave['gps'] == True,  # Sin comillas
    1,
    0
)

# %% id="YdnmKbq1K3S2"
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
color_encoded = ohe.fit_transform(bmw3_suave[['color']])
color_df = pd.DataFrame(color_encoded, columns=ohe.get_feature_names_out(['color']))
bmw3_suave = pd.concat([bmw3_suave, color_df], axis=1)

# %% colab={"base_uri": "https://localhost:8080/", "height": 444} id="BiO2OX0LM9qT" executionInfo={"status": "ok", "timestamp": 1764884096989, "user_tz": -60, "elapsed": 56, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="f8eefd0b-6cf0-46a4-8e1f-37ffaf598f89"
bmw3_suave

# %% id="jBo8UMwiNEac"
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
tipo_coche_encoded = ohe.fit_transform(bmw3_suave[['tipo_coche']])
tipo_coche_df = pd.DataFrame(tipo_coche_encoded, columns=ohe.get_feature_names_out(['tipo_coche']))
bmw3_suave = pd.concat([bmw3_suave, tipo_coche_df], axis=1)

# %% id="k7oWjhLpNH-b"
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
tipo_gasolina_encoded = ohe.fit_transform(bmw3_suave[['tipo_gasolina']])
tipo_gasolina_df = pd.DataFrame(tipo_gasolina_encoded, columns=ohe.get_feature_names_out(['tipo_gasolina']))
bmw3_suave = pd.concat([bmw3_suave, tipo_gasolina_df], axis=1)

# %% [markdown] id="8swZPCwmNf1w"
# # **7. Escalar variables (usando minmaxscaler) y luego aplicar la correlación final de variables ¿Hay alguna variable finalmente correlacionada?**
#

# %% id="7VLMIhC5Qsj1"
# Escalamos solo variables numéricas y que no sea el target.

# %% id="TBWrKOASNkLv"
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
bmw3_suave['Potencia_escalada'] = scaler.fit_transform(bmw3_suave[['potencia']])

# %% id="DMc9YBERQIz7"
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
bmw3_suave['Kilometros_escalados'] = scaler.fit_transform(bmw3_suave[['km']])

# %% colab={"base_uri": "https://localhost:8080/", "height": 444} id="5xa5doc2Qp9g" executionInfo={"status": "ok", "timestamp": 1764884120489, "user_tz": -60, "elapsed": 63, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="3b55554a-3e95-437e-b098-bb98d2955e0f"
bmw3_suave

# %% id="TPOi9rMZQ49r" executionInfo={"status": "ok", "timestamp": 1764884125229, "user_tz": -60, "elapsed": 4, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="fc6b5baa-6bf4-4ad2-9db0-d37e476394e8" colab={"base_uri": "https://localhost:8080/", "height": 1000}
bmw3_suave.isnull().sum()

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="c4POCpkLTCyi" executionInfo={"status": "ok", "timestamp": 1764884885721, "user_tz": -60, "elapsed": 582, "user": {"displayName": "Christian M\u00e9ndez", "userId": "08527219713841474419"}} outputId="e6f8a5df-92f7-4998-9f08-22b5fd5eb267"
corr2 = bmw3_suave.corr(numeric_only=True)  # ✅
plt.figure(figsize=(1, 1))
corr2.style.background_gradient(cmap='coolwarm')
