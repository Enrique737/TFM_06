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

# %% [markdown]
# ## Dependencias
# # Ejecutar este comando para instalar las librerías necesarias
# ```
# penv/Scripts/activate
# pip install pandas numpy scikit-learn matplotlib seaborn jinja2
# ```

# %%
from sklearn.preprocessing import LabelEncoder
import pandas as pd # Librería para la manipulación y el análisis de datos
import numpy as np # Librería para la manipulación de datos y para la ejecución de operaciones matemáticas
import matplotlib.pyplot as plt # Librería para la visualización de datos
import seaborn as sns # Librería para la visualización de datosr, MinMaxScaler, OrdinalEncoder # Librería para crear modelos de ML

# %%

# df_inicial1 = pd.read_csv("DatasetsTFM/customer_commercial_activity.csv")
# df_sample1 = df_inicial1.sample(10000).copy(deep=True)
# df_sample1.to_csv("DatasetsTFM/customer_commercial_activity_sample.csv", index=False)

# %%
# df_inicial2 = pd.read_csv("DatasetsTFM/customer_products.csv")
# df_sample2 = df_inicial2.sample(10000).copy(deep=True)
# df_sample2.to_csv("DatasetsTFM/customer_products_sample.csv", index=False)

# %%
# df_inicial3 = pd.read_csv("DatasetsTFM/customer_sociodemographics.csv")
# df_sample3 = df_inicial3.sample(10000).copy(deep=True)
# df_sample3.to_csv("DatasetsTFM/customer_sociodemographics_sample.csv", index=False)

# %%
# df_inicial4 = pd.read_csv("DatasetsTFM/product_description.csv")
# df_sample4 = df_inicial4.sample(13).copy(deep=True)
# df_sample4.to_csv("DatasetsTFM/product_description_sample.csv", index=False)

# %%
# df_inicial5 = pd.read_csv("DatasetsTFM/sales.csv")
# df_sample5 = df_inicial5.sample(10000).copy(deep=True)
# df_sample5.to_csv("DatasetsTFM/sales_sample.csv", index=False)

# %%
# 1. Cargar los datasets
df_prod = pd.read_csv("../DatasetsTFM/customer_products.csv")
df_socio = pd.read_csv("../DatasetsTFM/customer_sociodemographics.csv")
df_comm = pd.read_csv("../DatasetsTFM/customer_commercial_activity.csv")
df_sales = pd.read_csv("../DatasetsTFM/sales.csv")
df_prod_desc = pd.read_csv("../DatasetsTFM/product_description.csv")

# %%
duplicados = df_socio[df_socio.duplicated(subset=['pk_cid'], keep=False)]
duplicados.sort_values('pk_cid')

# %%
# Agrupamos por cliente y contamos cuántos salarios DIFERENTES ha tenido
# dropna=False es importante: cuenta cambiar de NaN a un valor numérico como un cambio
conteo_cambios = df_socio.groupby('pk_cid')['salary'].nunique(dropna=False)

# Filtramos solo los que tienen más de 1 valor distinto
clientes_con_cambios = conteo_cambios[conteo_cambios > 1]

print(f"Hay {len(clientes_con_cambios)} clientes que han cambiado de salario (o de NaN a dato).")

# %%
# Cogemos, por ejemplo, los primeros 5 clientes que sabemos que han cambiado
ids_ejemplo = clientes_con_cambios.head(5).index

# Filtramos el dataframe original solo con esos IDs y ordenamos para ver la historia
historial_cambios = df_socio[df_socio['pk_cid'].isin(ids_ejemplo)].sort_values(['pk_cid', 'pk_partition'])

# Mostramos las columnas relevantes
cols_interes = ['pk_cid', 'pk_partition', 'salary']
historial_cambios[cols_interes]

# %%
conteo_cambios = df_comm.groupby('pk_cid')['segment'].nunique(dropna=False)

# Filtramos solo los que tienen más de 1 valor distinto
clientes_con_cambios = conteo_cambios[conteo_cambios > 1]

print(f"Hay {len(clientes_con_cambios)} clientes que han cambiado de segmento (o de NaN a dato).")

# %%
# Cogemos, por ejemplo, los primeros 5 clientes que sabemos que han cambiado
ids_ejemplo = clientes_con_cambios.head(5).index

# Filtramos el dataframe original solo con esos IDs y ordenamos para ver la historia
historial_cambios = df_comm[df_comm['pk_cid'].isin(ids_ejemplo)].sort_values(['pk_cid', 'pk_partition'])

# Mostramos las columnas relevantes
cols_interes = ['pk_cid', 'pk_partition', 'segment']
historial_cambios[cols_interes]

# %%
for i in df_comm.columns:
    prctj = df_comm[i].isnull().mean() * 100
    print(f'{prctj:.3f}% \tde nulos en {i}')

# %%
for i in df_socio.columns:
    prctj = df_socio[i].isnull().mean() * 100
    print(f'{prctj:.3f}% \tde nulos en {i}')

# %%
print(df_socio['pk_partition'].dtype)
print(df_comm['pk_partition'].dtype)
print(df_socio['pk_cid'].dtype)
print(df_comm['pk_cid'].dtype)


# %%
def limpiar_columnas_basura(df):
    cols_to_drop = [c for c in df.columns if 'Unnamed' in c]
    if cols_to_drop:
        print(f"🗑️ Eliminando columnas basura: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)
    return df


# %%
df_prod = limpiar_columnas_basura(df_prod)
df_socio = limpiar_columnas_basura(df_socio)
df_comm = limpiar_columnas_basura(df_comm)
df_sales = limpiar_columnas_basura(df_sales)
df_prod_desc = limpiar_columnas_basura(df_prod_desc)

# %%
df_prod_desc.head(13)

# %%
# SQL Equivalente:
# SELECT s.*, d.product_desc 
# FROM sales s 
# LEFT JOIN product_description d ON s.product_ID = d.pk_product_ID;

# En Pandas:
df_ventas_enriquecida = pd.merge(
    df_sales, 
    df_prod_desc, 
    left_on='product_ID', 
    right_on='pk_product_ID', 
    how='left'
)
# df_ventas_enriquecida = limpiar_columnas_basura(df_ventas_enriquecida)
df_ventas_enriquecida.head(20)
#df_ventas_enriquecida["Unnamed: 0_y"].value_counts()

# %%
import polars as pl
import pyarrow

# 1. Convertir a Polars (requiere pyarrow instalado)
df_socio_pl = pl.from_pandas(df_socio)
df_comm_pl = pl.from_pandas(df_comm) 

# 2. Ejecutar el join usando las variables de Polars (_pl)
df_master_pl = df_socio_pl.join(
    df_comm_pl,
    on=['pk_cid', 'pk_partition'],
    how='inner',
    validate='1:1'
)

# Opcional: Si necesitas volver a pandas al final
# df_master = df_master_pl.to_pandas()

# %%
# SQL Equivalente:
# SELECT * FROM sociodemo s
# JOIN commercial c ON s.pk_cid = c.pk_cid AND s.pk_partition = c.pk_partition

# En Pandas:
df_master = pd.merge(
    df_socio,
    df_comm,
    on=['pk_cid', 'pk_partition'],
    how='inner',
    validate='1:1'
)
# df_master = limpiar_columnas_basura(df_master)
df_master.head(20)
# df_master["em_acount"].value_counts()
print(f"Hay {len(df_master)} tantos inner")
print(f"Hay {len(df_socio)} tantos socio")
print(f"Hay {len(df_comm)} tantos customer comercial activity")

# %%
df_master.columns

# %%
target = 'salary'
cols_group = ['segment', 'gender']

stats = df_master.groupby(cols_group)[target].agg(['mean', 'count'])
stats_filtered = stats[stats['count'] > 1000].sort_values(by='mean', ascending=False)
stats_filtered

# %%
df_master

# %%
df_test = df_master.copy()
df_test.loc[df_test['country_id'] != 'ES', 'country_id'] = 'international'
df_test = df_test[df_test['country_id'] == 'ES']
for i in df_test.columns:
    prctj = df_test[i].isnull().sum()
    print(f'{prctj} \tde nulos en {i}')

# %%
for i in df_socio.columns:
    prctj = df_socio[i].isnull().mean()*100
    print(f'{prctj} \tde nulos en {i}')

# %%
for i in df_master.columns:
    prctj = df_master[i].isnull().sum()
    print(f'{prctj} \tde nulos en {i}')

# %%
for i in df_comm.columns:
    prctj = df_comm[i].isnull().sum()
    print(f'{prctj} \tde nulos en {i}')

# %%
df_prod_desc['family_product'].value_counts()

# %%
df_master[(df_master['country_id'] == 'ES')]['region_code'].value_counts().shape

# %%
df_master.head(1)

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

df_inicial_test_2 = df_prod.copy()

null = utils.auditoria_nulos(df_inicial_test_2)

df_inicial_test_knn = df_inicial_test_2.copy()

target_equipamiento = ['aire_acondicionado', 'camara_trasera', 'elevalunas_electrico', 
                       'bluetooth', 'alerta_lim_velocidad', 'volante_regulable']
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




# df_limpio = df_inicial_test_knn[df_inicial_test_knn.isnull().sum(axis=1) <= 1].copy()
display(utils.auditoria_nulos(df_inicial_test_knn))

# %%
for i in df_master.columns:
    prctj = df_master[i].isnull().mean() * 100
    print(f'{prctj:.3f}% \tde nulos en {i}')


# %%
def calcular_nuevas_ventas(df_completo, columna_producto):
    """
    Calcula las ventas (cambio de 0 a 1) asegurando que comparamos
    al MISMO cliente en meses CONSECUTIVOS.
    """
    # 1. Ordenamos para garantizar el orden cronológico
    df_sorted = df_completo.sort_values(by=['pk_cid', 'pk_partition'])
    
    # 2. Agrupamos por cliente antes de hacer el shift (EL PASO CLAVE QUE FALTABA)
    # Esto evita que compares el último mes del Cliente A con el primero del Cliente B
    df_sorted['estado_anterior'] = df_sorted.groupby('pk_cid')[columna_producto].shift(1)
    
    # 3. Rellenamos los nulos (el primer mes de cada cliente) con 0
    df_sorted['estado_anterior'] = df_sorted['estado_anterior'].fillna(0)
    
    # 4. Calculamos la venta: Tiene ahora (1) Y NO tenía antes (0)
    # OJO: Convertimos a entero para evitar problemas de float
    mascara_venta = (df_sorted[columna_producto] == 1) & (df_sorted['estado_anterior'] == 0)
    
    # 5. Filtramos solo las ventas
    ventas = df_sorted[mascara_venta]
    
    return ventas

# PRUEBA FINAL CON EL DATASET COMPLETO (No con la muestra, que tiene huecos)
df_ventas_tarjetas = calcular_nuevas_ventas(df_prod, 'credit_card')
df_ventas_tarjetas = limpiar_columnas_basura(df_ventas_tarjetas)
print(f"Ventas detectadas con tarjeta de crédito: {len(df_ventas_tarjetas)}")
df_ventas_tarjetas.head(10)

# %%
df_prod.sample(20)
#df_prod["short_term_deposit"].value_counts()
#for col in df_prod.columns:
#    print(f"{col}: {df_prod[col].nunique()} valores únicos")

# %%
df_prod.info()

# %%
df_comm.head(20)

# %%
df_socio.head(20)
#df_socio[""].value_counts()

# %%
# 2. Convertir la partición a fecha real para poder ordenar correctamente
# Esto es CRÍTICO: Sin orden temporal, no podemos calcular ventas
for df in [df_prod, df_socio, df_comm]:
    df['pk_partition'] = pd.to_datetime(df['pk_partition'])

print("Datos cargados. Rango de fechas en la muestra:")
print(f"Desde {df_prod['pk_partition'].min()} hasta {df_prod['pk_partition'].max()}")

# %%
# Seleccionamos un cliente que tenga bastante historia en la muestra
# (Buscamos el ID que más veces se repite en el dataset de productos)
cliente_ejemplo = df_prod['pk_cid'].value_counts().idxmax()

print(f"Analizando al cliente ID: {cliente_ejemplo}")

# Filtramos su historia en la tabla de productos
historia_cliente = df_prod[df_prod['pk_cid'] == cliente_ejemplo].sort_values('pk_partition')

# Mostramos solo las columnas clave para no saturar
cols_visualizar = ['pk_cid', 'pk_partition', 'loans', 'credit_card', 'payroll']
display(historia_cliente[cols_visualizar])

# %%
df_socio.head()

# %%
df_socio["deceased"].value_counts()

# %%
df_socio["country_id"].value_counts()

# %%
df_comm.head()

# %%
# Vamos a unir la info comercial y sociodemográfica a nuestra tabla de productos
# La llave de unión es ['pk_cid', 'pk_partition']

vista_360 = pd.merge(
    historia_cliente,
    df_socio[['pk_cid', 'pk_partition', 'age', 'salary']], # Solo traemos edad y salario
    on=['pk_cid', 'pk_partition'],
    how='left'
)

vista_360 = pd.merge(
    vista_360,
    df_comm[['pk_cid', 'pk_partition', 'segment', 'active_customer']], # Traemos segmento y actividad
    on=['pk_cid', 'pk_partition'],
    how='left'
)

# Visualizamos cómo cambia el cliente mes a mes en todas sus dimensiones
cols_360 = ['pk_partition', 'loans', 'age', 'salary', 'segment', 'active_customer']
display(vista_360[cols_360])

# %%
# Vamos a enfocarnos en un producto, por ejemplo 'credit_card' (o cambia por 'loans' o 'payroll')
producto_analizar = 'credit_card' # Puedes cambiar esto por otro producto que tenga el cliente

# 1. Estado Actual (Lo que tiene en el mes T)
df_analisis = historia_cliente[['pk_cid', 'pk_partition', producto_analizar]].copy()
df_analisis = df_analisis.sort_values('pk_partition')

# 2. Estado Anterior (Lo que tenía en el mes T-1)
# Usamos .shift(1) para "bajar" los datos una fila
df_analisis['estado_mes_anterior'] = df_analisis[producto_analizar].shift(1).fillna(0)

# 3. Cálculo de la DIFERENCIA (Venta vs Baja vs Mantenimiento)
# 1 - 0 =  1 (Alta/Venta) -> ESTO ES LO QUE BUSCA CAROL
# 1 - 1 =  0 (Mantiene)
# 0 - 1 = -1 (Baja/Churn)
# 0 - 0 =  0 (Nada)
df_analisis['diferencia'] = df_analisis[producto_analizar] - df_analisis['estado_mes_anterior']

# 4. Etiquetamos para que sea legible a humanos
def etiquetar_venta(valor):
    if valor == 1: return "💰 NUEVA VENTA"
    if valor == -1: return "📉 BAJA"
    if valor == 0: return "Mantiene"
    
df_analisis['que_paso'] = df_analisis['diferencia'].apply(etiquetar_venta)

# Mostramos el resultado visual
# Busca las filas donde aparezca "NUEVA VENTA"
display(df_analisis)
# Filtramos solo las filas con nuevas ventas
nuevas_ventas = df_analisis[df_analisis['diferencia'] == 1]
print(f"Nuevas ventas encontradas: {len(nuevas_ventas)}")
display(nuevas_ventas)

# %%
if 'pk_product_ID' in df_ventas_enriquecida.columns:
    df_ventas_enriquecida.drop(columns=['pk_product_ID'], inplace=True)

# %%
filas_con_nulos = df_ventas_enriquecida[df_ventas_enriquecida['product_desc'].isnull()]
if len(filas_con_nulos) > 0:
    print(f"⚠️ ALERTA: Hay {len(filas_con_nulos)} ventas con IDs de producto desconocidos.")
    print("IDs de producto que no tienen descripción:")
    print(filas_con_nulos['product_ID'].unique())
    display(filas_con_nulos.head())
else:
    print("✅ Integridad perfecta: Todas las ventas tienen una descripción de producto asociada.")

# %%
cols_productos = [col for col in df_prod.columns if col not in ['pk_cid', 'pk_partition']]


# %%
def auditoria_nulos_inteligente(df):
    nulos_por_col = df.isnull().sum()
    cols_con_nulos = nulos_por_col[nulos_por_col > 0]
    
    if cols_con_nulos.empty:
        print("No hay nulos en este DataFrame.")
        return
    
    print(f"Columnas afectadas:\n{cols_con_nulos.sort_values(ascending=False)}")
    
    # Creamos una máscara para ver solo las filas que tienen AL MENOS un nulo
    filas_nulas = df[df.isnull().any(axis=1)].copy()
    
    # ¿Falta solo un dato o faltan 5 a la vez?
    filas_nulas['n_nulos'] = filas_nulas.isnull().sum(axis=1)
    
    print(f"\nTotal de filas con algún dato faltante: {len(filas_nulas)} ({(len(filas_nulas)/len(df))*100:.2f}% del total)")
    print("\nDistribución de gravedad (¿Cuántos datos faltan por fila?):")
    display(filas_nulas['n_nulos'].value_counts().sort_index())
    
    print("\nEjemplo de filas con múltiples nulos (Top 5):")
    display(filas_nulas.sort_values('n_nulos', ascending=False).head(5))
    
    return filas_nulas


# %%
df_nulos_master = auditoria_nulos_inteligente(df_master)

# %%
df_nulos_master = auditoria_nulos_inteligente(df_prod)

# %%
df_nulos_master = auditoria_nulos_inteligente(df_ventas_enriquecida)

# %% [markdown]
# El salario puede estar asociado a la actividad (si no es activo no nos interesa y podemos poner 0), también al segmento (sacamos el promedio por segmento, país, región y ventanas de edad) y si está vivo o muerto.
# El segmento lo ponemos como desconocido (podría ser moda)
# El canal de entrada Desconocido (pueden tener correlación o ser datos migrados de bases de datos más viejas)
# La región la podemos poner por moda según segmento (los universitarios tienden a estar en las mismas ciudades), edad y país
# El género en sí mismo no sabría como rellenarlo, quizá como Desconocido o por moda si es muy claro, si son casos muy típicos quizá sería posible hasta eliminarlos

# %%

# %%
df_nulos_ventas = auditoria_nulos_inteligente(df_ventas_enriquecida)

# %%
df_nulos_prod = auditoria_nulos_inteligente(df_prod)

# %%
df_prod.head(20)

# %%
df_prod.info()

# %%
df_ventas_enriquecida.info()

# %%
df_master.info()


# %%
def auditoria_agrupada(df, columna_objetivo, columnas_agrupacion, ordenar_por_volumen=True):
    """
    Analiza la distribución de nulos agrupando por columnas.
    
    Parámetros:
    - df: DataFrame.
    - columna_objetivo: Columna a auditar (nulos).
    - columnas_agrupacion: Lista de columnas para agrupar.
    - ordenar_por_volumen: 
        True -> Ordena por 'total_filas' descendente (Rompe grupos para mostrar lo más grande primero).
        False -> Ordena jerárquicamente por 'columnas_agrupacion' (Mantiene el orden lógico del índice).
    """
    print(f"📊 Analizando '{columna_objetivo}' agrupado por {columnas_agrupacion}...")
    
    # Agrupamos (dropna=False para ver nulos en la agrupación)
    resumen = df.groupby(columnas_agrupacion, dropna=False).agg(
        total_filas=(columna_objetivo, 'size'),
        nulos_objetivo=(columna_objetivo, lambda x: x.isnull().sum())
    )
    
    # Métricas
    resumen['pct_nulos'] = (resumen['nulos_objetivo'] / resumen['total_filas'] * 100).round(2)
    resumen['peso_grupo_%'] = (resumen['total_filas'] / len(df) * 100).round(2)
    
    # Lógica de Ordenamiento
    if ordenar_por_volumen:
        # Prioriza el volumen total (Rompe la jerarquía visual de regiones)
        return resumen.sort_values('total_filas', ascending=False)
    else:
        # Prioriza la jerarquía de las columnas de agrupación
        # Además, dentro de cada grupo, ordenamos por volumen descendente para que quede bonito
        return resumen.sort_values(columnas_agrupacion + ['total_filas'], ascending=[True]*len(columnas_agrupacion) + [False])


# %%
df_paises = auditoria_agrupada(df_master, "salary", ["country_id","segment"])
df_paises.head(50)

# %%
df_paises = auditoria_agrupada(df_master, "salary", ["country_id","region_code","segment"])
df_paises.head(50)


# %%
def agrupador(df, columna, valores_a_mantener, group_as="Other_countries"):
    """
    Agrupa los valores de una columna que NO estén en la lista 'valores_a_mantener'
    bajo una nueva etiqueta.
    
    Mejoras:
    - Admite tanto lista ['A', 'B'] como string único 'A'.
    - Retorna una copia para no dañar el dataset original.
    - Imprime resumen del cambio.
    """
    # 1. Protección: Si el usuario pasa un string suelto, lo convertimos a lista
    if isinstance(valores_a_mantener, str):
        valores_a_mantener = [valores_a_mantener]
        
    print(f"\n   Procesando columna '{columna}'...")
    print(f"   Manteniendo intactos: {valores_a_mantener}")
    print(f"   Agrupando el resto en: '{group_as}'")
    
    # 2. Seguridad: Trabajamos sobre una copia
    df_out = df.copy() 
    
    # 3. Lógica de Agrupación (Vectorizada)
    # Seleccionamos todo lo que NO (~) esté en la lista blanca
    mask = ~df_out[columna].isin(valores_a_mantener)
    num_afectados = mask.sum()
    
    # Aplicamos el cambio
    df_out.loc[mask, columna] = group_as
    
    # 4. Reporte final
    print(f"   ¡Hecho! Se han agrupado {num_afectados} filas.")
    print(f"   Distribución final:\n{df_out[columna].value_counts().head()}")
    
    return df_out


# %%
# 1. Aplicamos la agrupación a Países (Manteniendo solo España)
df_master_grouped = agrupador(df_master, "country_id", ['ES'], group_as="Other_countries")

# 2. Verificamos el resultado usando tu función de auditoría anterior
# Deberías ver solo 2 filas: 'ES' y 'Other_countries'
auditoria_agrupada(df_master_grouped, "salary", ["country_id","segment"])

# %%
display(df_master_grouped['country_id'].value_counts())

# %%
auditoria_agrupada(df_master, "salary", ["country_id"])

# %%
auditoria_agrupada(df_master_grouped[(df_master_grouped["country_id"]=='ES')], "salary", ["region_code","segment"])

# %%
df_master_spain = df_master_grouped[df_master_grouped["country_id"] == 'ES'].copy()
df_master_spain = df_master_spain[df_master_spain["region_code"] == 28.0].copy()
top_regiones = df_master_spain['region_code'].value_counts().head(10).index.tolist()
df_spain_top = df_master_spain[df_master_spain['region_code'].isin(top_regiones)]
auditoria_agrupada(df_spain_top, "salary", ["region_code", "segment"], ordenar_por_volumen=False)

# %%
auditoria_agrupada(df_master_spain, "salary", ["region_code", "segment"], ordenar_por_volumen=False)


# %%
def visualizar_correlacion(df, col_x, col_y, hue=None, tipo_grafico='auto'):
    """
    Visualiza la relación entre dos variables eligiendo el gráfico adecuado.
    
    Parámetros:
    - df: DataFrame.
    - col_x: Variable eje X.
    - col_y: Variable eje Y.
    - hue: (Opcional) Variable para segmentar por colores (ej. 'active_customer').
    - tipo_grafico: 'auto', 'scatter', 'box', 'heatmap', 'bar'.
    """
    
    # Preparar lienzo
    plt.figure(figsize=(10, 6))
    
    # Detectar tipos de datos
    es_numerica_x = pd.api.types.is_numeric_dtype(df[col_x])
    es_numerica_y = pd.api.types.is_numeric_dtype(df[col_y])
    
    # Lógica de decisión automática
    if tipo_grafico == 'auto':
        if es_numerica_x and es_numerica_y:
            tipo_grafico = 'scatter' # Num vs Num (ej. Edad vs Salario)
            # Si hay demasiados puntos (>10k), mejor un hexbin o samplear
            if len(df) > 10000: tipo_grafico = 'hexbin' 
            
        elif (not es_numerica_x) and es_numerica_y:
            tipo_grafico = 'box' # Cat vs Num (ej. Segmento vs Salario)
            
        elif es_numerica_x and (not es_numerica_y):
            # Intercambiamos ejes para que la categoría quede en X (mejor lectura)
            col_x, col_y = col_y, col_x
            tipo_grafico = 'box'
            
        else:
            tipo_grafico = 'heatmap' # Cat vs Cat (ej. Región vs Segmento)

    print(f" Generando gráfico tipo: {tipo_grafico.upper()} para {col_x} vs {col_y}")

    # --- GENERACIÓN DE GRÁFICOS ---
    
    if tipo_grafico == 'scatter':
        sns.scatterplot(data=df, x=col_x, y=col_y, hue=hue, alpha=0.6)
        plt.title(f"Correlación: {col_x} vs {col_y}")
        
    elif tipo_grafico == 'hexbin':
        # Hexbin es mejor para millones de datos que Scatter
        plt.hexbin(df[col_x], df[col_y], gridsize=50, cmap='Blues')
        plt.colorbar(label='Cantidad de Clientes')
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        plt.title(f"Densidad: {col_x} vs {col_y}")
        
    elif tipo_grafico == 'box':
        sns.boxplot(data=df, x=col_x, y=col_y, hue=hue)
        plt.title(f"Distribución de {col_y} por {col_x}")
        plt.xticks(rotation=45)
        
    elif tipo_grafico == 'heatmap':
        # Tabla de contingencia (Crosstab)
        ct = pd.crosstab(df[col_x], df[col_y], normalize='index') # % por fila
        sns.heatmap(ct, annot=True, cmap="YlGnBu", fmt=".2%")
        plt.title(f"Relación Categorica: {col_x} vs {col_y}")
        
    elif tipo_grafico == 'bar':
        # Útil para ver la media de una variable Y por cada categoría de X
        sns.barplot(data=df, x=col_x, y=col_y, hue=hue, errorbar=None)
        plt.title(f"Media de {col_y} por {col_x}")
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# %%
visualizar_correlacion(df_prod, "emc_account", "credit_card", tipo_grafico='heatmap')

# %%
visualizar_correlacion(df_prod, "em_account_p", "credit_card", tipo_grafico='heatmap')

# %%
visualizar_correlacion(df_prod, "emc_account", "mortgage", tipo_grafico='heatmap')

# %%
visualizar_correlacion(df_prod, "em_account_p", "mortgage", tipo_grafico='heatmap')

# %%
visualizar_correlacion(df_prod, "emc_account", "short_term_deposit", tipo_grafico='heatmap')

# %%
visualizar_correlacion(df_prod, "em_account_p", "short_term_deposit", tipo_grafico='heatmap')

# %%
visualizar_correlacion(df_prod, "emc_account", "long_term_deposit", tipo_grafico='heatmap')

# %%
visualizar_correlacion(df_prod, "em_account_p", "long_term_deposit", tipo_grafico='heatmap')

# %%
df_prod.columns

# %%
display(df_prod['long_term_deposit'].value_counts())


# %%
def analizar_correlaciones_productos(df, cols_productos, agrupar=False, clave_agrupar=None):
    """
    Genera un Mapa de Calor (Heatmap) para múltiples columnas booleanas en base a una agrupación.
    
    Parámetros:
    - df: DataFrame.
    - cols_productos: Lista de las columnas boleanas o binarias a relacionar.
    - agrupar:

        * False (Default): agrupa por el primer pk_ o columna que detecte

        * True: Analiza el perfil general (¿El cliente que contrata X suele contratar Y?)

    - clave_agrupar: (Opcional) Columna específica para agrupar. Si no se proporciona, se busca la primera columna 'pk_'.
    """
    
    # Preparamos los datos
    if agrupar:
        if clave_agrupar is None:
            for col in df.columns:
                if 'pk_' in col and col != 'pk_partition':
                    clave_agrupar = col
                    break
                else:
                    raise ValueError("No se encontró una columna 'pk_' para agrupar. Por favor, especifica 'clave_agrupar'.")
            
            df_analisis = df.groupby(clave_agrupar)[cols_productos].max()
        else:
            df_analisis = df.groupby(clave_agrupar)[cols_productos].max()
        print(f"Agrupando por {clave_agrupar}")
    else:
        df_analisis = df[cols_productos]

    # Cálculo de la Correlación
    # Para variables binarias (0/1), Pearson funciona como el coeficiente Phi
    corr_matrix = df_analisis.corr(method='pearson')
    
    # Diseño del Gráfico
    plt.figure(figsize=(14, 12))
    
    # Heatmap
    # cmap='RdBu_r': Rojo para inversa (si tengo X no tengo Y), Azul para directa (van juntos)
    sns.heatmap(
        corr_matrix, 
        annot=True,     # Muestra el número
        fmt=".2f",      # 2 decimales
        cmap='RdBu',  # Escala de colores Rojo-Azul
        vmin=-0.2, vmax=0.6, # Ajustamos el rango para resaltar correlaciones sutiles (pocas pasan de 0.6)
        center=0,       # El blanco es correlación cero
        linewidths=0.5, # Líneas entre celdas
        cbar_kws={"shrink": .5} # Barra de leyenda más pequeña
    )
    
    plt.title(f"Correlación de booleanos por {clave_agrupar if agrupar else 'relación general'}", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.show()



# %%
# Columnas de productos a analizar
cols_productos_reales = [
    'loans', 'mortgage', 'funds', 'securities', 'long_term_deposit', 
    'em_account_pp', 'credit_card', 'payroll', 'pension_plan', 
    'payroll_account', 'emc_account', 'debit_card', 'em_account_p', 'em_acount'
]

# %%
# Opción A: ¿Qué productos se consumen JUNTOS en el mismo mes?
analizar_correlaciones_productos(df_prod, cols_productos_reales, agrupar=False)

# %%
# Opción B: ¿Qué productos suele tener EL MISMO CLIENTE (aunque sea en meses distintos)?
# Esta responde a tu pregunta: "¿El que tiene emc_account tiende a long_term_deposit?"
analizar_correlaciones_productos(df_prod, cols_productos_reales, agrupar=True, clave_agrupar='pk_cid')

# %%
cols_productos = [
    'loans', 'mortgage', 'funds', 'securities', 'long_term_deposit', 
    'credit_card', 'payroll', 'pension_plan', 'debit_card'
]
analizar_correlaciones_productos(df_prod, cols_productos, agrupar=True, clave_agrupar='emc_acount')
analizar_correlaciones_productos(df_prod, cols_productos, agrupar=False)

# %%
# Esta responde a tu pregunta: Hay algún periodo donde se compre más algún producto?"
display(df_prod.groupby('pk_partition')[cols_productos_reales].sum())


# %%
def visualizar_evolucion_temporal(df, cols_productos, col_tiempo='pk_partition', modo='linea', normalizar=False):
    """
    Visualiza cómo evoluciona la tenencia o venta de productos a lo largo del tiempo.
    
    Parámetros:
    - df: DataFrame con los datos.
    - cols_productos: Lista de columnas de productos.
    - col_tiempo: Nombre de la columna temporal (default: 'pk_partition').
    - modo: 
        * 'linea': Gráfico de líneas (ideal para ver tendencias).
        * 'area': Gráfico de áreas apiladas (ideal para ver volumen total acumulado).
        * 'heatmap': Mapa de calor (ideal para comparar intensidades).
    - normalizar: 
        * False: Muestra valores absolutos (Cantidad de contratos).
        * True: Muestra valores relativos (Porcentaje del total de ese mes).
    """
    
    # 1. Preparación de datos
    df_temp = df.copy()
    
    # Aseguramos que la fecha sea datetime para que se ordene bien
    if df_temp[col_tiempo].dtype == 'object':
        df_temp[col_tiempo] = pd.to_datetime(df_temp[col_tiempo])
        
    # Agrupamos por fecha y sumamos (para ver volumen de productos activos)
    df_agrupado = df_temp.groupby(col_tiempo)[cols_productos].sum()
    
    # Título dinámico
    titulo = "Evolución Temporal de Productos"
    ylabel = "Cantidad de Contratos"
    
    # 2. Normalización (Opcional)
    if normalizar:
        # Dividimos cada fila por la suma total de esa fila (para obtener %)
        df_agrupado = df_agrupado.div(df_agrupado.sum(axis=1), axis=0) * 100
        titulo += " (Normalizado %)"
        ylabel = "Porcentaje del Portfolio (%)"

    # 3. Visualización
    plt.figure(figsize=(15, 8))
    
    if modo == 'linea':
        # Usamos plot de pandas que gestiona muy bien las fechas en el eje X
        # Iteramos para dar estilos si son muchos productos
        for col in df_agrupado.columns:
            plt.plot(df_agrupado.index, df_agrupado[col], marker='.', label=col, linewidth=2)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.ylabel(ylabel)
        
    elif modo == 'area':
        df_agrupado.plot.area(figsize=(15, 8), alpha=0.7, stacked=True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.ylabel(ylabel)

    elif modo == 'heatmap':
        # Transponemos para que los Productos estén en el eje Y y el Tiempo en el X
        # Es más fácil de leer así
        sns.heatmap(
            df_agrupado.T, 
            cmap='YlGnBu', # Amarillo a Verde a Azul (bueno para volumen)
            annot=False,   # False para no saturar si hay muchos meses
            linewidths=0.5,
            fmt=".1f" if normalizar else ".0f"
        )
        plt.ylabel("Productos")
        plt.xlabel("Fecha (Partition)")
    
    plt.title(titulo, fontsize=16)
    plt.tight_layout()
    plt.show()



# %%
# --- EJEMPLOS DE USO ---

# %%
# 1. Ver la tendencia absoluta (¿Suben o bajan las ventas?)
visualizar_evolucion_temporal(df_prod, cols_productos_reales, modo='linea', normalizar=False)

# %%
# 2. Ver el "Share" de productos (¿Qué productos dominan la cartera?)
# El modo 'area' normalizado es genial para ver cómo un producto "come terreno" a otro
visualizar_evolucion_temporal(df_prod, cols_productos_reales, modo='area', normalizar=True)

# %%
# 3. Mapa de calor (Tu petición específica de colormap)
# Aquí verás claramente cuándo se lanzó un producto (pasa de claro a oscuro)
visualizar_evolucion_temporal(df_prod, cols_productos_reales, modo='heatmap', normalizar=False)

# %%
# 3. Mapa de calor (Tu petición específica de colormap)
# Aquí verás claramente cuándo se lanzó un producto (pasa de claro a oscuro)
visualizar_evolucion_temporal(df_prod, cols_productos_reales, modo='heatmap', normalizar=True)

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

def visualizar_evolucion_premium(df, cols_productos, col_tiempo='pk_partition', modo='linea', normalizar=False):
    """
    Visualización mejorada con estética 'Business Intelligence'.
    """
    
    # 1. Preparación de datos
    df_temp = df.copy()
    if df_temp[col_tiempo].dtype == 'object':
        df_temp[col_tiempo] = pd.to_datetime(df_temp[col_tiempo])
        
    # Agrupamos y ordenamos
    df_agrupado = df_temp.groupby(col_tiempo)[cols_productos].sum()
    
    # Formato de fechas para que se lean bien (Año-Mes)
    df_agrupado.index = df_agrupado.index.strftime('%Y-%m')
    
    # Configuración de Etiquetas y Títulos
    titulo = "Evolución de Contratación"
    unidad_cbar = "Cantidad"
    
    # 2. Normalización (Cálculo del Share %)
    if normalizar:
        # Dividimos por el total del mes para ver la cuota de cada producto
        df_agrupado = df_agrupado.div(df_agrupado.sum(axis=1), axis=0) * 100
        titulo += " (Share de Cartera %)"
        unidad_cbar = "Porcentaje (%)"

    # 3. Visualización
    plt.figure(figsize=(16, 9)) # Formato panorámico
    
    if modo == 'heatmap':
        # Transponemos: Productos en filas (Y), Fechas en columnas (X)
        data_plot = df_agrupado.T
        
        # Paleta: 'Blues' o 'YlGnBu' para absoluto, 'RdYlGn' (semáforo) o 'viridis' para %
        cmap = 'viridis' if normalizar else 'YlGnBu'
        
        # Formato de texto dentro de las celdas
        fmt = ".1f" if normalizar else ",.0f" # 1 decimal o enteros con comas
        
        # DIBUJO DEL HEATMAP
        ax = sns.heatmap(
            data_plot, 
            cmap=cmap, 
            annot=True,             # Mostrar números
            fmt=fmt,                # Formato numérico base
            linewidths=1,           # Líneas blancas separadoras (Estilo Waffle)
            linecolor='white',      
            cbar_kws={'label': unidad_cbar, 'shrink': 0.8},
            annot_kws={"size": 9}   # Tamaño de letra cómodo
        )
        
        # TRUCO: Añadir el símbolo '%' manualmente si está normalizado
        if normalizar:
            for t in ax.texts:
                t.set_text(t.get_text() + "%")

        # AJUSTE DE ETIQUETAS (Lo que pediste)
        plt.yticks(rotation=0, fontsize=11)  # Productos rectos
        plt.xticks(rotation=45, ha='right')  # Fechas inclinadas ligeramente
        plt.xlabel("") # Quitamos la etiqueta 'pk_partition' que sobra
        plt.ylabel("") # Quitamos 'Productos' que es obvio
        
    elif modo == 'linea':
        # Estilo de líneas limpio
        sns.set_style("whitegrid")
        sns.lineplot(data=df_agrupado, dashes=False, palette="tab10", linewidth=2.5)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Productos")
        plt.ylabel(unidad_cbar)
        plt.xticks(rotation=45)

    elif modo == 'area':
        # Área apilada
        df_agrupado.plot.area(figsize=(16, 9), alpha=0.8, stacked=True, colormap="tab20")
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Productos")
        plt.ylabel(unidad_cbar)
    
    plt.title(titulo, fontsize=18, pad=20, fontweight='bold', color='#333333')
    plt.tight_layout()
    plt.show()

# --- PRUEBA LA DIFERENCIA ---

# Opción A: Absoluto (Limpio y con separadores)
visualizar_evolucion_premium(df_prod, cols_productos_reales, modo='heatmap', normalizar=False)

# Opción B: Tu petición (Normalizado, con % y etiquetas rectas)
visualizar_evolucion_premium(df_prod, cols_productos_reales, modo='heatmap', normalizar=True)

# %% [markdown]
# Entrenar el modelo balanceando las clases. Cuando separemos quien ha comprado algo un mes prepararemos un modelo que separe la misma cantidad de compradores y misma cantidad de no compradores. Esto se podrá preparar por cada producto o por todos los productos
