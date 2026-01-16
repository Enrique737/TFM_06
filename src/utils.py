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

# %%
from sklearn.preprocessing import LabelEncoder
import pandas as pd # Librería para la manipulación y el análisis de datos
import numpy as np # Librería para la manipulación de datos y para la ejecución de operaciones matemáticas
import matplotlib.pyplot as plt # Librería para la visualización de datos
import seaborn as sns # Librería para la visualización de datosr, MinMaxScaler, OrdinalEncoder # Librería para crear modelos de ML


# %%
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


# %%
def porcentaje_nulos(df):
    """
    Calcula el porcentaje de valores nulos en cada columna de un DataFrame.

    Parámetros:
    - df: pandas.DataFrame

    Retorno:
    - pandas.Series con el porcentaje de nulos por columna.
    """
    for i in df.columns:
        prctj = df[i].isnull().mean() * 100
        print(f'{prctj:.3f}% \tde nulos en {i}')
    return



# %%
def auditoria_nulos(df):
    """
    Realiza una auditoría de los valores nulos en un DataFrame.
    Parámetros:
    - df: pandas.DataFrame
    Retorno:
    - pandas.DataFrame con las filas que contienen al menos un valor nulo,
      incluyendo una columna adicional 'n_nulos' que indica el número de
      valores nulos en cada fila.
    """
    print("Auditoría de nulos en el DataFrame:")
    
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
        
        vecinos = vecinos.dropna()
        
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
def detectar_outliers_iqr(df, columnas_numericas):
    outliers_dict = {}
    for col in columnas_numericas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lim_inf = Q1 - 1.5 * IQR
        lim_sup = Q3 + 1.5 * IQR
        
        # Filtramos los valores que caen fuera
        mask_outliers = (df[col] < lim_inf) | (df[col] > lim_sup)
        n_outliers = mask_outliers.sum()
        
        if n_outliers > 0:
            outliers_dict[col] = {
                'n_outliers': n_outliers,
                'lim_inf': lim_inf, 
                'lim_sup': lim_sup,
                '%_dataset': (n_outliers / len(df)) * 100
            }
            print(f"{col}: {n_outliers} outliers detectados ({outliers_dict[col]['%_dataset']:.2f}%)")
            
    return outliers_dict
