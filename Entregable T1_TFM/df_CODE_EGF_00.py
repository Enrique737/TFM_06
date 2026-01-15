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

# %% id="jU9NvteqrwUc" executionInfo={"status": "ok", "timestamp": 1763999486461, "user_tz": -60, "elapsed": 1529, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
#Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,MinMaxScaler, OrdinalEncoder

# %% [markdown] id="lYsdTeWm9wvP"
# ## TABLA SOCIODEMOGRAPHICS

# %% id="XNd0RTytHeoG"
# Lectura del archivo excel de la tabla SD
df_SD= pd.read_excel('/content/customer_sociodemographics 11.11.25.xlsx')

# %% id="NrgAEZHBJwVK" executionInfo={"status": "error", "timestamp": 1763999745912, "user_tz": -60, "elapsed": 94, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} colab={"base_uri": "https://localhost:8080/", "height": 319} outputId="d6314b2b-24eb-4240-f804-339c72146158"
# Lectura del archivo csv de la tabla SD
df_SD_csv= pd.read_csv('/content/customer_sociodemographics 11.11.25.csv', encoding='latin-1')


# %% id="0t_V_W3bKrjk"
# El dataframe aparece como el _csv, es decir, archivo separado por comas.
df_SD_csv

# %% [markdown] id="pqc9yrzmGySz"
# ###ANALISIS PRELIMINAR

# %% id="wjORO6GmH_Lm"
# Tabla SD en python
df_SD


# %% id="LxjsQ0UDIcLQ" executionInfo={"status": "aborted", "timestamp": 1763999692788, "user_tz": -60, "elapsed": 77524, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Información número de filas y columnas
df_SD.shape

# %% id="2nAQJ0pGKYiF" executionInfo={"status": "aborted", "timestamp": 1763999692794, "user_tz": -60, "elapsed": 77525, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Información tipos en columnas y nulos
df_SD.info()

# %% id="s7bUscLcK6DO" executionInfo={"status": "aborted", "timestamp": 1763999692836, "user_tz": -60, "elapsed": 77565, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Análisis de variables estadísticas
df_SD.describe()
# Valores extraños en columna salary
# me faltan columnas categóricas

# %% id="SeboxC4NZS3x" executionInfo={"status": "aborted", "timestamp": 1763999692839, "user_tz": -60, "elapsed": 77567, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
df_SD.dtypes


# %% id="5sJ-iwZM04F5" executionInfo={"status": "aborted", "timestamp": 1763999692841, "user_tz": -60, "elapsed": 77567, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Saco los valores por cada columna con value_counts y nulls con un bucle
# la pk_cid tiene duplicados
for i in df_SD:
  if df_SD[i].dtype.kind=="O":
    #imprime value_counts de variables categóricas
    print("\n",df_SD[i].value_counts(),"\n")
  elif (df_SD[i].dtype.kind=="f") or (df_SD[i].dtype.kind=="i"):
    #imprime histograma de variables numericas
    print("\n",df_SD.hist(i),"\n")

# %% id="0JmAgM4-Lu0t" executionInfo={"status": "aborted", "timestamp": 1763999692843, "user_tz": -60, "elapsed": 77564, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Suma de nulos por columna
df_SD.isnull().sum()

# %% id="EaLT6K4ZL5gu" executionInfo={"status": "aborted", "timestamp": 1763999692846, "user_tz": -60, "elapsed": 77566, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Quiero las dos útlimas filas de la tabla
df_SD.tail(2)

# %% id="SU6dKadE8qD1" executionInfo={"status": "aborted", "timestamp": 1763999692849, "user_tz": -60, "elapsed": 77563, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Hago un boxplot a salary para ver outliers
sns.boxplot(x=df_SD["salary"])

# %% id="uJJQj_ahxG0c" executionInfo={"status": "aborted", "timestamp": 1763999692851, "user_tz": -60, "elapsed": 77564, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Quiero ver cuantos salarios son por debajo de 100.000 € y cuantos por encima
print(df_SD[df_SD["salary"]<20000].shape) #1
print(df_SD[(df_SD["salary"] >=20000) & (df_SD["salary"] <=50000)].shape) #2
print(df_SD[(df_SD["salary"] >50000) & (df_SD["salary"] <=100000)].shape) #3
print(df_SD[df_SD["salary"]<100000].shape) #4
print(df_SD[df_SD["salary"]>100000].shape )#5
print(df_SD[(df_SD["salary"] >100000) & (df_SD["salary"] <=200000)].shape) #6
print(df_SD[(df_SD["salary"] >200000) & (df_SD["salary"] <=500000)].shape) #7
print(df_SD[df_SD["salary"]>500000].shape) #8
print(df_SD[(df_SD["salary"] >500000) & (df_SD["salary"] <=1000000)].shape) #9
print(df_SD[df_SD["salary"]>1000000].shape) #10
# No cuadra los 120494 pos por encima 1Mio con el bloxplot.

# %% id="qT4knpC6_sG6" executionInfo={"status": "aborted", "timestamp": 1763999692853, "user_tz": -60, "elapsed": 77564, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Hago un scatter_plot para ver distribución de valores
sns.scatterplot(x="pk_cid", y="salary", hue="age", data=df_SD, palette="coolwarm")

# %% id="k0RJEfB_Pkho" executionInfo={"status": "aborted", "timestamp": 1763999692856, "user_tz": -60, "elapsed": 77566, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# DE region code quiero ver sus valores
df_SD["region_code"].value_counts()

# %% id="CHqIYq9wCsCa" executionInfo={"status": "aborted", "timestamp": 1763999692858, "user_tz": -60, "elapsed": 77564, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Cuantos clientes unicos tenemos? #249689 clientes únicos
df_cp["pk_cid"].nunique()

# %% [markdown] id="Rtnaa_eiMcXY"
# ### DUPLICADOS
#

# %% id="-gOLc-KRMeSr" executionInfo={"status": "aborted", "timestamp": 1763999692860, "user_tz": -60, "elapsed": 77564, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Copia 2 de seguridad
df_SD_2 =df_SD.copy()
df_SD_2.shape

# %% id="b82EUp9GPMVW" executionInfo={"status": "aborted", "timestamp": 1763999692862, "user_tz": -60, "elapsed": 77565, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Inicio de análisis de duplicados en pk_city
df_SD_2.duplicated(subset=["pk_cid"],keep=False)


# %% id="sgkxnyz2PgF1" executionInfo={"status": "aborted", "timestamp": 1763999692864, "user_tz": -60, "elapsed": 77561, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Analisis de duplicados en la columna pk_cid
##df_SD_2["pk_cid"].value_counts() ##Duplicados
##df_SD[df_SD.duplicated(subset=["pk_cid"], keep=False)]
##Cuenta todos los duplicados
##df_SD[df_SD.duplicated(subset=["pk_cid"], keep=False)].shape[0]
## Lista duplicados contando número de repeticiones
## df_SD['pk_cid'].value_counts()[df_SD['pk_cid'].value_counts() > 1]
## Filtro por valos pk_cid 1136935
df_SD[df_SD["pk_cid"]==1136935]

##df_SD_2["pk_partition"].value_counts() No duplicados
##df_SD_2["country_id"].value_counts() No veo nada extraño
##df_SD_2["region_code"].value_counts() No veo nada extraño
##df_SD_2["gender"].value_counts() No veo nada extraño
##df_SD_2["age"].value_counts() No veo nada extraño
#df_SD_2["deceased"].value_counts() No veo nada extraño
## df_SD_2["salary"].value_counts() ## Valor e+16 fuera de rango

# %% id="FIPsKGnyAxkH" executionInfo={"status": "aborted", "timestamp": 1763999692866, "user_tz": -60, "elapsed": 77562, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
## df = df.drop_duplicates(subset=["comando"], keep="first")  # deja el primero
## df = df.drop_duplicates(subset=["comando"], keep="last")   # deja el último

# %% [markdown] id="tBpk4hZkPkSe"
# ### NULOS

# %% id="XyOzTYtBGojK" executionInfo={"status": "aborted", "timestamp": 1763999692868, "user_tz": -60, "elapsed": 77562, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Suma de nulos por columna
df_SD.isnull().sum()

# %% id="OUHJn5ICPmaU" executionInfo={"status": "aborted", "timestamp": 1763999692870, "user_tz": -60, "elapsed": 77563, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
df_SD.isnull().any()

# %% [markdown] id="YmsAIy9s94_3"
# ## TABLA CUSTOMER_COMMERCIAL_ACTIVITY

# %% id="PNgutVrP9-7e" executionInfo={"status": "aborted", "timestamp": 1763999692872, "user_tz": -60, "elapsed": 76445, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Lectura de archivo excel de la tabla customer commercial activity
df_cca = pd.read_excel('/content/customer_commercial_activity 11.11.25.xlsx')

# %% id="-hC7dJrmGNYU" executionInfo={"status": "aborted", "timestamp": 1763999692874, "user_tz": -60, "elapsed": 76431, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
df_cca

# %% [markdown] id="lZi1eocvMh0U"
# ### ANALISIS PRELIMINAR

# %% id="zMMKc0U-L3c_" executionInfo={"status": "aborted", "timestamp": 1763999692876, "user_tz": -60, "elapsed": 76417, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
df_cca.shape

# %% id="Yvfhf1rML7tp" executionInfo={"status": "aborted", "timestamp": 1763999692878, "user_tz": -60, "elapsed": 76407, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
df_cca.info()

# %% id="uGHWaRxVL-7o" executionInfo={"status": "aborted", "timestamp": 1763999692879, "user_tz": -60, "elapsed": 76400, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
df_cca.describe()

# %% id="5RgfG1lPMzmG" executionInfo={"status": "aborted", "timestamp": 1763999692881, "user_tz": -60, "elapsed": 76395, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
df_cca.dtypes

# %% id="X0qP81n7M4Ae" executionInfo={"status": "aborted", "timestamp": 1763999692883, "user_tz": -60, "elapsed": 76373, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Saco los valores por cada columna con value_counts con un bucle
for i in df_cca:
  if df_cca[i].dtype.kind=="O":
    #imprime value_counts de variables categóricas
    print("\n",df_cca[i].value_counts(),"\n")
  elif (df_cca[i].dtype.kind=="f") or (df_cca[i].dtype.kind=="i"):
    #imprime histograma de variables numericas
    print("\n",df_cca.hist(i),"\n")

# %% id="J4FsyVO_Namm" executionInfo={"status": "aborted", "timestamp": 1763999692884, "user_tz": -60, "elapsed": 76371, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Hago un boxplot a net_margin para ver outliers
sns.boxplot(x=df_cca["entry_date"])

# %% id="d6h49vUB25hL" executionInfo={"status": "aborted", "timestamp": 1763999692889, "user_tz": -60, "elapsed": 76365, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Quiero hacer un % por la variable segment

(df_cca["segment"].value_counts() / df_cca.shape[0] * 100).round(2).astype(str) + " %"

# %% id="Nv_cwAa74fsb" executionInfo={"status": "aborted", "timestamp": 1763999692891, "user_tz": -60, "elapsed": 76352, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
#Quiero contar los datos de la columna active_customer
(df_cca["active_customer"].value_counts() / (df_cca.shape[0])*100).round(2).astype(str) + " %"

# %% [markdown] id="e4M57_T5PYqa"
# ### NULOS

# %% id="GeT2ilizPXWi" executionInfo={"status": "aborted", "timestamp": 1763999692893, "user_tz": -60, "elapsed": 76351, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Suma de nulos por columna
df_cca.isnull().sum()

# %% [markdown] id="TtvGVSZjPcp6"
# ### DUPLICADOS

# %% id="TtlzK7wDPjIa" executionInfo={"status": "aborted", "timestamp": 1763999692895, "user_tz": -60, "elapsed": 76352, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Analisis de duplicados en todas las columnas
df_cca[df_cca.duplicated(keep=False)]
# CCA no tiene duplicados

# %% [markdown] id="IDEzgGND-GZE"
# ## TABLA SALES

# %% id="f7OI7lZ5FNyz" executionInfo={"status": "aborted", "timestamp": 1763999692900, "user_tz": -60, "elapsed": 75241, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Lectura del archivo excel de la tabla sales
df_sales= pd.read_excel('/content/sales 11.11.25.xlsx')

# %% id="JkqpAhlhFncP" executionInfo={"status": "aborted", "timestamp": 1763999692902, "user_tz": -60, "elapsed": 75213, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
df_sales

# %% [markdown] id="ZWYv6H84FwSx"
# ### ANALISIS PRELIMINAR

# %% id="lsEK_7ZcF0N_" executionInfo={"status": "aborted", "timestamp": 1763999692904, "user_tz": -60, "elapsed": 75200, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
df_sales.shape

# %% id="VI7S-U6zGbKF" executionInfo={"status": "aborted", "timestamp": 1763999692906, "user_tz": -60, "elapsed": 75198, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
df_sales.info()

# %% id="H7uA-Lb9Gi17" executionInfo={"status": "aborted", "timestamp": 1763999692908, "user_tz": -60, "elapsed": 75192, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
df_sales.describe()

# %% id="8EYOEd0PGrhT" executionInfo={"status": "aborted", "timestamp": 1763999692910, "user_tz": -60, "elapsed": 75183, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
df_sales.dtypes

# %% executionInfo={"status": "aborted", "timestamp": 1763999692911, "user_tz": -60, "elapsed": 75169, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} id="2tEmPqkFGuyz"
# Saco los valores por cada columna con value_counts con un bucle
for i in df_sales:
  if df_sales[i].dtype.kind=="O":
    #imprime value_counts de variables categóricas
    print("\n",df_sales[i].value_counts(),"\n")
  elif (df_sales[i].dtype.kind=="f") or (df_sales[i].dtype.kind=="i"):
    #imprime histograma de variables numericas
    print("\n",df_sales.hist(i),"\n")

# %% id="CZx8-TYIHhUp" executionInfo={"status": "aborted", "timestamp": 1763999692926, "user_tz": -60, "elapsed": 75174, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Hago un boxplot a net_margin para ver outliers
sns.boxplot(x=df_sales["net_margin"])

# %% id="6TYE75NN6CX2" executionInfo={"status": "aborted", "timestamp": 1763999692928, "user_tz": -60, "elapsed": 75169, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Quiero un porcentaje de los datos de net_margin por debajo de 100 €
## (df_sales["net_margin"].value_counts() / (df_sales.shape[0])*100).round(2).astype(str) + " %"
print(df_sales[(df_sales["net_margin"] >=0) & (df_sales["net_margin"] <=100)].shape)
print(df_sales.shape)

margen_menos_100 = (df_sales[(df_sales["net_margin"] >= 0) & (df_sales["net_margin"] <= 100)].shape[0]
       / df_sales.shape[0]) * 100

print(str(round(margen_menos_100,2)) + " %")


# %% [markdown] id="ZrfbAD4gIgRH"
# ### NULOS

# %% id="EPathKhJIimG" executionInfo={"status": "aborted", "timestamp": 1763999692930, "user_tz": -60, "elapsed": 75165, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Suma de nulos por columna
df_sales.isnull().sum()

# %% [markdown] id="gd7IGKOtIwKX"
# ### DUPLICADOS

# %% executionInfo={"status": "aborted", "timestamp": 1763999692931, "user_tz": -60, "elapsed": 75158, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}} id="oC2UU0-0JArq"
# Analisis de duplicados en todas las columnas
df_sales[df_sales.duplicated(keep=False)]
## Sales no tiene duplicados

# %% [markdown] id="h9_GJ1QQf1dY"
# ##TABLA CUSTOMER_PRODUCTS

# %% id="HRHhtH9Jf71H" executionInfo={"status": "aborted", "timestamp": 1763999692935, "user_tz": -60, "elapsed": 73704, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Lectura del archivo excel de la tabla C_P
df_cp= pd.read_excel('/content/customer_products 13.11.2025.xlsx')


# %% id="8PrxKDMVhZNF" executionInfo={"status": "aborted", "timestamp": 1763999692937, "user_tz": -60, "elapsed": 73692, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
df_cp

# %% [markdown] id="CntyjQ8lhZpW"
# ###ANALISIS PRELIMINAR

# %% id="7K3sJFtqhdrT" executionInfo={"status": "aborted", "timestamp": 1763999692940, "user_tz": -60, "elapsed": 73691, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
df_cp.shape

# %% id="k9-oP0rj-PgV" executionInfo={"status": "aborted", "timestamp": 1763999692943, "user_tz": -60, "elapsed": 73686, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
df_cp.info()

# %% id="D746p4qB-SPD" executionInfo={"status": "aborted", "timestamp": 1763999692945, "user_tz": -60, "elapsed": 73678, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
df_cp.describe()

# %% id="BBd7HtRY-bvC" executionInfo={"status": "aborted", "timestamp": 1763999692946, "user_tz": -60, "elapsed": 73666, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
df_cp.dtypes

# %% id="EeV65fQC-k2l" executionInfo={"status": "aborted", "timestamp": 1763999692949, "user_tz": -60, "elapsed": 73658, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Saco los valores por cada columna con value_counts con un bucle
for i in df_cp:
  if df_cp[i].dtype.kind=="O":
    #imprime value_counts de variables categóricas
    print("\n",df_cp[i].value_counts(),"\n")
  elif (df_cp[i].dtype.kind=="f") or (df_cp[i].dtype.kind=="i"):
    #imprime histograma de variables numericas
    print("\n",df_cp.hist(i),"\n")

    # Practicamente no contratan productos

# %% id="bbyXY5Gd-jtb" executionInfo={"status": "aborted", "timestamp": 1763999692952, "user_tz": -60, "elapsed": 73656, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Quiero hacer una suma de valores de las siguientes columnas credit_card, debit_card, em_account_p,em_account_pp,emc_account,funds,loans,long_term_deposit,mortgage,payroll,payroll_account,pension_plan,securities,short_term_deposit
cols=["credit_card","debit_card","em_account_p","em_account_pp","emc_account","funds","loans","long_term_deposit","mortgage","payroll","payroll_account","pension_plan","securities","short_term_deposit"]
# Con esto saco el número de productos contratados
print("\n","Productos_Contratados")
print(df_cp[cols].sum())

#quiero el porcentaje de cada suma por producto
print("\n","Número total de filas")
print(df_cp.shape[0])
print("\n","Porcentaje de clientes que contratan cada producto")
print((df_cp[cols].sum()/249689*100).round(2).astype(str) + " %")
#Con esto saco el número de productos contratados


# %% id="QtVVJo9jBr1n" executionInfo={"status": "aborted", "timestamp": 1763999692955, "user_tz": -60, "elapsed": 73658, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Cuantos clientes unicos tenemos? #249689 clientes únicos
df_cp["pk_cid"].nunique()
df_cp_1=df_cp.copy()

# %% id="hW4M9ZefEQAW" executionInfo={"status": "aborted", "timestamp": 1763999692958, "user_tz": -60, "elapsed": 73652, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Dame las filas duplicadas de pk_cid?
df_cp[df_cp.duplicated(subset=["pk_cid"], keep=False)]
df_cp_1.drop_duplicates(subset=["pk_cid"], keep="first", inplace=True)
df_cp_1.shape

# %% [markdown] id="ZWZNZRTZheOn"
# ###NULOS

# %% id="GcCLQrhRhgXq" executionInfo={"status": "aborted", "timestamp": 1763999692964, "user_tz": -60, "elapsed": 73657, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Suma de nulos por columna
df_cp.isnull().sum()

# %% [markdown] id="T0kkoYIbhhBR"
# ###DUPLICADOS

# %% id="KGkNI4Mlhj6w" executionInfo={"status": "aborted", "timestamp": 1763999692968, "user_tz": -60, "elapsed": 73661, "user": {"displayName": "Enrique Gonzalez", "userId": "06466324871652609959"}}
# Analisis de duplicados en todas las columnas
df_cp[df_cp.duplicated(keep=False)]
## Customer_Products no tiene duplicados
