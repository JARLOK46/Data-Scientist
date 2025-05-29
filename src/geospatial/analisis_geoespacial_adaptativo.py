"""
Módulo de análisis geoespacial adaptativo.

Este script permite realizar un análisis geoespacial flexible y robusto sobre un archivo CSV con datos de eventos o registros georreferenciados. El flujo incluye:

- Detección automática del delimitador del archivo.
- Carga y limpieza de datos, eliminando duplicados y valores inválidos.
- Identificación automática de columnas de latitud y longitud.
- Limpieza de filas con coordenadas inválidas o desconocidas.
- Visualización de la distribución geográfica de los puntos (scatter).
- Histogramas de latitud y longitud para entender la dispersión espacial.
- Clustering espacial con K-Means para identificar agrupaciones naturales.
- Gráfico de barras de cantidad de puntos por cluster.
- Mapa interactivo con Folium para explorar los datos geoespaciales.
- Generación de un informe Markdown con el resumen y las visualizaciones generadas.

Dependencias principales: pandas, numpy, matplotlib, seaborn, folium, scikit-learn.

Estructura del archivo:
- Funciones utilitarias para carga, limpieza y visualización.
- Un flujo principal (main) que orquesta todo el análisis.

Autor: Anderson Zapata
Fecha: 2025
"""

import os
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
import folium
from folium.plugins import MarkerCluster, HeatMap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import urllib.request
from PIL import Image

# Suprimir advertencias
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Verificar e importar bibliotecas opcionales
def check_optional_imports():
    """
    Verifica e intenta importar bibliotecas opcionales para análisis geoespacial avanzado.

    Esta función intenta importar paquetes como `folium`, `geopandas`, `shapely`, `sklearn.cluster` y `scipy.spatial`.
    Devuelve un diccionario con el estado de disponibilidad de cada paquete y una lista de los que faltan.

    :return: Una tupla con (diccionario de paquetes disponibles, lista de paquetes faltantes).
    :rtype: tuple(dict, list)
    :example:
        >>> available, missing = check_optional_imports()
        >>> print(available['folium'])
        True
        >>> print(missing)
        ['geopandas (Para análisis geoespacial)']
    :note:
        Esta función es útil para advertir al usuario sobre dependencias opcionales antes de ejecutar análisis avanzados.
    """
    missing_packages = []
    optional_packages = {
        'folium': 'Para mapas interactivos',
        'geopandas': 'Para análisis geoespacial',
        'shapely': 'Para geometrías espaciales',
        'sklearn.cluster': 'Para clustering espacial',
        'scipy.spatial': 'Para análisis espacial avanzado'
    }
    
    available_packages = {}
    
    # Intentar importar cada paquete
    for package, description in optional_packages.items():
        try:
            if package == 'sklearn.cluster':
                from sklearn.cluster import DBSCAN, KMeans
                from sklearn.preprocessing import StandardScaler
                available_packages[package] = True
            elif package == 'scipy.spatial':
                from scipy.spatial import ConvexHull
                available_packages[package] = True
            elif package == 'folium':
                import folium
                from folium.plugins import MarkerCluster, HeatMap
                available_packages[package] = True
            elif package == 'geopandas':
                import geopandas as gpd
                available_packages[package] = True
            elif package == 'shapely':
                from shapely.geometry import Point
                available_packages[package] = True
        except ImportError:
            missing_packages.append(f"{package} ({description})")
            available_packages[package] = False
    
    return available_packages, missing_packages

# Crear directorio para resultados si no existe
def create_results_dir():
    """
    Crea el directorio de resultados 'output' en la misma carpeta que el script si no existe.

    :return: Ruta absoluta al directorio de resultados creado o existente.
    :rtype: str
    :example:
        >>> results_dir = create_results_dir()
        >>> print(results_dir)
        '/ruta/al/proyecto/src/geospatial/output'
    :note:
        Si el directorio ya existe, no lo borra ni lo modifica.
    """
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def detect_delimiter(file_path, num_lines=5):
    """
    Detecta automáticamente el delimitador utilizado en un archivo CSV.

    Lee las primeras líneas del archivo y cuenta la frecuencia de los delimitadores más comunes
    (tabulación, coma, punto y coma, barra vertical). Devuelve el delimitador más frecuente.

    :param file_path: Ruta al archivo CSV.
    :type file_path: str
    :param num_lines: Número de líneas a analizar para la detección.
    :type num_lines: int
    :return: El delimitador detectado ('\t', ',', ';', '|').
    :rtype: str
    :example:
        >>> detect_delimiter('datos.csv')
        ','
    """
    # Abrimos el archivo en modo lectura, ignorando errores de codificación
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        # Leemos las primeras 'num_lines' líneas para tener una muestra representativa
        sample = ''.join([file.readline() for _ in range(num_lines)])
    # Inicializamos un diccionario para contar la frecuencia de cada delimitador
    delimiters = {'\t': 0, ',': 0, ';': 0, '|': 0}
    for delimiter in delimiters:
        delimiters[delimiter] = sample.count(delimiter)
    # Devolvemos el delimitador que más veces aparece en la muestra
    return max(delimiters, key=delimiters.get)

def load_and_clean_data(file_path):
    """
    Carga y limpia un archivo CSV para análisis geoespacial.

    - Detecta el delimitador automáticamente.
    - Intenta cargar el archivo con diferentes codificaciones.
    - Normaliza los nombres de las columnas.
    - Elimina filas duplicadas.

    :param file_path: Ruta al archivo CSV.
    :type file_path: str
    :return: DataFrame limpio y sin duplicados.
    :rtype: pandas.DataFrame
    """
    print(f"Cargando datos desde: {file_path}")
    delimiter = detect_delimiter(file_path)
    print(f"Delimitador detectado: '{delimiter}'")
    try:
        df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8', low_memory=False)
    except Exception as e:
        print(f"Error con encoding utf-8: {e}")
        try:
            df = pd.read_csv(file_path, delimiter=delimiter, encoding='latin1', low_memory=False)
        except Exception as e:
            print(f"Error con encoding latin1: {e}")
            return None
    print(f"Datos cargados. Dimensiones: {df.shape}")
    # Normalizamos los nombres de las columnas para evitar problemas posteriores
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    # Eliminamos duplicados para evitar sesgos en el análisis
    df = df.drop_duplicates()
    return df

def identify_geo_columns(df):
    """
    Identifica automáticamente las columnas de latitud y longitud en un DataFrame.

    Busca primero columnas con nombres típicos ('lat', 'latitude', 'lon', 'lng', 'longitude').
    Si no las encuentra, busca columnas numéricas cuyos valores estén en los rangos válidos de latitud y longitud.

    :param df: DataFrame de entrada.
    :type df: pandas.DataFrame
    :return: Nombres de las columnas de latitud y longitud.
    :rtype: tuple(str, str) o (None, None)
    """
    # Buscamos columnas con nombres típicos de latitud y longitud
    lat_candidates = [col for col in df.columns if any(term in col.lower() for term in ['lat', 'latitude'])]
    lon_candidates = [col for col in df.columns if any(term in col.lower() for term in ['lon', 'lng', 'longitude'])]
    if lat_candidates and lon_candidates:
        return lat_candidates[0], lon_candidates[0]
    # Si no se encuentran, buscamos columnas numéricas en los rangos válidos
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        vals = df[col].dropna()
        if vals.min() >= -90 and vals.max() <= 90:
            lat_candidates.append(col)
        if vals.min() >= -180 and vals.max() <= 180:
            lon_candidates.append(col)
    if lat_candidates and lon_candidates:
        return lat_candidates[0], lon_candidates[0]
    return None, None

def clean_geo_data(df, lat_col, lon_col):
    """
    Filtra el DataFrame para conservar solo filas con coordenadas válidas.

    - Elimina filas con valores nulos o 'Desconocido' en latitud/longitud.
    - Filtra coordenadas fuera de los rangos válidos.

    :param df: DataFrame de entrada.
    :type df: pandas.DataFrame
    :param lat_col: Nombre de la columna de latitud.
    :type lat_col: str
    :param lon_col: Nombre de la columna de longitud.
    :type lon_col: str
    :return: DataFrame filtrado con coordenadas válidas.
    :rtype: pandas.DataFrame
    """
    print("\nLimpiando datos geoespaciales...")
    # Eliminamos filas con valores nulos o 'Desconocido'
    geo_df = df.dropna(subset=[lat_col, lon_col]).copy()
    geo_df = geo_df[(geo_df[lat_col].astype(str).str.lower() != 'desconocido') & (geo_df[lon_col].astype(str).str.lower() != 'desconocido')]
    # Filtramos coordenadas fuera de rango
    geo_df = geo_df[(geo_df[lat_col] >= -90) & (geo_df[lat_col] <= 90) & (geo_df[lon_col] >= -180) & (geo_df[lon_col] <= 180)]
    print(f"  Datos geoespaciales válidos: {len(geo_df)} de {len(df)} filas")
    return geo_df

def plot_geo_scatter(geo_df, lat_col, lon_col, results_dir):
    """
    Genera un mapa de dispersión de los puntos geográficos.

    Cada punto representa un evento o registro con coordenadas válidas.
    El gráfico se guarda como imagen PNG en el directorio de resultados.

    :param geo_df: DataFrame con datos geoespaciales válidos.
    :type geo_df: pandas.DataFrame
    :param lat_col: Nombre de la columna de latitud.
    :type lat_col: str
    :param lon_col: Nombre de la columna de longitud.
    :type lon_col: str
    """
    print("\nGenerando visualización: Mapa de dispersión...")
    plt.figure(figsize=(12, 10))
    plt.scatter(geo_df[lon_col], geo_df[lat_col], alpha=0.5, s=10)
    plt.title(f'Distribución Geográfica de Puntos ({lat_col} vs {lon_col})')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, f'distribucion_geografica_{lat_col}_{lon_col}.png')
    plt.savefig(path)
    plt.close()
    print(f"Gráfico guardado: {path}")

def plot_histograms(geo_df, lat_col, lon_col, results_dir):
    """
    Genera histogramas de latitud y longitud para analizar la dispersión espacial.

    Permite identificar concentraciones o vacíos geográficos en los datos.
    El gráfico se guarda como imagen PNG en el directorio de resultados.

    :param geo_df: DataFrame con datos geoespaciales válidos.
    :type geo_df: pandas.DataFrame
    :param lat_col: Nombre de la columna de latitud.
    :type lat_col: str
    :param lon_col: Nombre de la columna de longitud.
    :type lon_col: str
    """
    print("\nGenerando visualización: Histogramas de coordenadas...")
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(geo_df[lat_col], kde=True)
    plt.title(f'Distribución de {lat_col.capitalize()} (Latitud)')
    plt.xlabel('Latitud')
    plt.subplot(1, 2, 2)
    sns.histplot(geo_df[lon_col], kde=True)
    plt.title(f'Distribución de {lon_col.capitalize()} (Longitud)')
    plt.xlabel('Longitud')
    plt.tight_layout()
    path = os.path.join(results_dir, f'histogramas_{lat_col}_{lon_col}.png')
    plt.savefig(path)
    plt.close()
    print(f"Gráfico guardado: {path}")

def plot_kmeans_clusters(geo_df, lat_col, lon_col, results_dir):
    """
    Realiza clustering espacial con K-Means y genera el gráfico correspondiente.

    - Normaliza las coordenadas para evitar sesgos por escala.
    - Determina el número de clusters según la cantidad de datos.
    - Asigna cada punto a un cluster y lo visualiza por color.
    - El gráfico se guarda como imagen PNG en el directorio de resultados.

    :param geo_df: DataFrame con datos geoespaciales válidos.
    :type geo_df: pandas.DataFrame
    :param lat_col: Nombre de la columna de latitud.
    :type lat_col: str
    :param lon_col: Nombre de la columna de longitud.
    :type lon_col: str
    :return: DataFrame con columna 'kmeans_cluster'.
    :rtype: pandas.DataFrame
    """
    print("\nRealizando clustering K-Means...")
    if len(geo_df) < 10:
        print("No hay suficientes datos para clustering.")
        return None
    coords = geo_df[[lat_col, lon_col]].values
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    n_clusters = min(8, len(geo_df) // 10) if len(geo_df) > 10 else 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    geo_df['kmeans_cluster'] = kmeans.fit_predict(coords_scaled)
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(geo_df[lon_col], geo_df[lat_col], c=geo_df['kmeans_cluster'], cmap='tab10', s=30, alpha=0.7, edgecolors='w')
    plt.colorbar(scatter, label='Cluster K-Means')
    plt.title(f'Clustering Espacial K-Means ({lat_col} vs {lon_col})')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, f'clusters_kmeans_{lat_col}_{lon_col}.png')
    plt.savefig(path)
    plt.close()
    print(f"Gráfico guardado: {path}")
    return geo_df

def plot_bar_by_cluster(geo_df, results_dir):
    """
    Genera un gráfico de barras con la cantidad de puntos por cluster K-Means.

    Permite visualizar la distribución de los datos entre los diferentes clusters.
    El gráfico se guarda como imagen PNG en el directorio de resultados.

    :param geo_df: DataFrame con columna 'kmeans_cluster'.
    :type geo_df: pandas.DataFrame
    """
    print("\nGenerando visualización: Gráfico de barras por cluster...")
    if 'kmeans_cluster' not in geo_df.columns:
        return
    counts = geo_df['kmeans_cluster'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.index, y=counts.values, palette='viridis')
    plt.title('Cantidad de Puntos por Cluster K-Means')
    plt.xlabel('Cluster')
    plt.ylabel('Cantidad de puntos')
    plt.tight_layout()
    path = os.path.join(results_dir, 'puntos_por_cluster_kmeans.png')
    plt.savefig(path)
    plt.close()
    print(f"Gráfico guardado: {path}")

def create_interactive_map(geo_df, lat_col, lon_col, results_dir):
    """
    Genera un mapa interactivo con Folium y clusters de puntos.

    - Muestra los puntos en un mapa base OpenStreetMap.
    - Agrupa los puntos en clusters interactivos.
    - Añade un mapa de calor para visualizar densidad.
    - El mapa se guarda como archivo HTML en el directorio de resultados.

    :param geo_df: DataFrame con datos geoespaciales válidos.
    :type geo_df: pandas.DataFrame
    :param lat_col: Nombre de la columna de latitud.
    :type lat_col: str
    :param lon_col: Nombre de la columna de longitud.
    :type lon_col: str
    """
    print("\nGenerando visualización: Mapa interactivo...")
    center_lat = geo_df[lat_col].mean()
    center_lon = geo_df[lon_col].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=2, tiles='OpenStreetMap')
    marker_cluster = MarkerCluster().add_to(m)
    # Para no sobrecargar el mapa, limitamos a 1000 puntos
    sample_df = geo_df.sample(min(1000, len(geo_df))) if len(geo_df) > 1000 else geo_df
    for idx, row in sample_df.iterrows():
        popup_text = f"<b>Coordenadas:</b> {row[lat_col]:.4f}, {row[lon_col]:.4f}<br>"
        if 'kmeans_cluster' in row:
            popup_text += f"<b>Cluster:</b> {row['kmeans_cluster']}<br>"
        folium.Marker(
            location=[row[lat_col], row[lon_col]],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(marker_cluster)
    # Añadimos un mapa de calor para visualizar densidad
    heat_data = [[row[lat_col], row[lon_col]] for idx, row in sample_df.iterrows()]
    HeatMap(heat_data, radius=15).add_to(m)
    map_path = os.path.join(results_dir, 'mapa_interactivo.html')
    m.save(map_path)
    print(f"Mapa interactivo guardado en: {map_path}")

def generate_report(geo_df, lat_col, lon_col, results_dir):
    """
    Genera un informe en formato Markdown con el resumen del análisis geoespacial adaptativo.

    El informe incluye:
    - Resumen de puntos válidos y rangos de coordenadas.
    - Resultados del clustering K-Means (si aplica).
    - Listado de visualizaciones generadas.
    - Conclusiones y recomendaciones.

    :param geo_df: DataFrame con datos geoespaciales válidos.
    :type geo_df: pandas.DataFrame
    :param lat_col: Nombre de la columna de latitud.
    :type lat_col: str
    :param lon_col: Nombre de la columna de longitud.
    :type lon_col: str
    """
    print("\nGenerando informe de análisis geoespacial...")
    md_report = f"# Informe de Análisis Geoespacial Adaptativo\n\n"
    md_report += f"**Fecha de análisis:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_report += f"## Resumen\n\n"
    md_report += f"- Total de puntos con coordenadas válidas: **{len(geo_df)}**\n"
    md_report += f"- Rango de latitudes: **{geo_df[lat_col].min():.4f}** a **{geo_df[lat_col].max():.4f}**\n"
    md_report += f"- Rango de longitudes: **{geo_df[lon_col].min():.4f}** a **{geo_df[lon_col].max():.4f}**\n\n"
    if 'kmeans_cluster' in geo_df.columns:
        md_report += f"## Clustering K-Means\n\n"
        kmeans_counts = geo_df['kmeans_cluster'].value_counts()
        md_report += f"- Número de clusters: **{len(kmeans_counts)}**\n"
        md_report += f"- Distribución de puntos por cluster:\n"
        for cluster, count in kmeans_counts.items():
            md_report += f"  - Cluster {cluster}: **{count}** puntos ({count/len(geo_df)*100:.1f}%)\n"
    md_report += f"\n## Visualizaciones Generadas\n\n"
    md_report += f"- ![Distribución Geográfica de Puntos]({os.path.join(results_dir, f'distribucion_geografica_{lat_col}_{lon_col}.png')})\n"
    md_report += f"  - Archivo: distribucion_geografica_{lat_col}_{lon_col}.png\n"
    md_report += f"- ![Histogramas de Latitud y Longitud]({os.path.join(results_dir, f'histogramas_{lat_col}_{lon_col}.png')})\n"
    md_report += f"  - Archivo: histogramas_{lat_col}_{lon_col}.png\n"
    if 'kmeans_cluster' in geo_df.columns:
        md_report += f"- ![Clustering Espacial K-Means]({os.path.join(results_dir, f'clusters_kmeans_{lat_col}_{lon_col}.png')})\n"
        md_report += f"  - Archivo: clusters_kmeans_{lat_col}_{lon_col}.png\n"
        md_report += f"- ![Barras de cantidad de puntos por cluster]({os.path.join(results_dir, 'puntos_por_cluster_kmeans.png')})\n"
        md_report += f"  - Archivo: puntos_por_cluster_kmeans.png\n"
    md_report += f"- Mapa interactivo (HTML): mapa_interactivo.html\n"
    md_report += f"\n## Conclusiones\n\n"
    md_report += f"- Los datos muestran una distribución geográfica coherente.\n"
    if 'kmeans_cluster' in geo_df.columns:
        md_report += f"- El clustering permite identificar agrupaciones espaciales relevantes.\n"
    md_report_path = os.path.join(results_dir, 'informe_analisis_geoespacial_adaptativo.md')
    with open(md_report_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    print(f"Informe en formato Markdown guardado en: {md_report_path}")

def detect_country_column(df):
    """
    Detecta automáticamente la columna que representa el país en un DataFrame.

    Busca en los nombres de las columnas términos comunes relacionados con países,
    como 'pais', 'country', 'codigo_pais', 'pais_codigo'. Retorna el nombre de la
    primera columna válida encontrada.

    :param df: DataFrame de entrada donde se buscará la columna de país.
    :type df: pandas.DataFrame
    :return: Nombre de la columna que representa el país, o None si no se encuentra.
    :rtype: str or None
    :example:
        >>> detect_country_column(df)
        'pais_codigo'
    :note:
        Es útil para automatizar análisis geográficos cuando los nombres de columnas pueden variar entre archivos.
    """
    country_candidates = [col for col in df.columns if any(term in col.lower() for term in ['pais', 'country', 'codigo_pais', 'pais_codigo'])]
    return country_candidates[0] if country_candidates else None

def detect_year_column(df):
    """
    Detecta automáticamente la columna que representa el año o la fecha en un DataFrame.

    Busca en los nombres de las columnas términos comunes relacionados con año o fecha,
    como 'año', 'anio', 'year', 'fecha', 'date'. Intenta convertir los valores a fechas
    y retorna el nombre de la primera columna válida encontrada.

    :param df: DataFrame de entrada donde se buscará la columna de año o fecha.
    :type df: pandas.DataFrame
    :return: Nombre de la columna que representa el año o fecha, o None si no se encuentra.
    :rtype: str or None
    :example:
        >>> detect_year_column(df)
        'año_evento'
    :note:
        Es útil para automatizar análisis temporales cuando los nombres de columnas pueden variar entre archivos.
    """
    year_candidates = [col for col in df.columns if any(term in col.lower() for term in ['año', 'anio', 'year', 'fecha', 'date'])]
    for col in year_candidates:
        try:
            vals = pd.to_datetime(df[col], errors='coerce')
            if vals.notnull().sum() > 0:
                return col
        except:
            continue
    return None

def detect_type_column(df):
    """
    Detecta automáticamente la columna que representa el tipo, categoría o clase de suceso en un DataFrame.

    Busca en los nombres de las columnas términos comunes como 'tipo', 'categoria', 'clase', 'evento', 'suceso',
    'incident', 'event', 'category'. Retorna el nombre de la primera columna encontrada.

    :param df: DataFrame de entrada donde se buscará la columna de tipo/categoría.
    :type df: pandas.DataFrame
    :return: Nombre de la columna que representa el tipo/categoría, o None si no se encuentra.
    :rtype: str or None
    :example:
        >>> detect_type_column(df)
        'tipo_evento'
    :note:
        Es útil para automatizar análisis categóricos cuando los nombres de columnas pueden variar entre archivos.
    """
    type_candidates = [col for col in df.columns if any(term in col.lower() for term in ['tipo', 'categoria', 'clase', 'evento', 'suceso', 'incident', 'event', 'category'])]
    return type_candidates[0] if type_candidates else None

def plot_top_countries(df, country_col, results_dir):
    """
    Genera un gráfico de barras con los 20 países que tienen más sucesos registrados.

    :param df: DataFrame que contiene los datos a graficar
    :type df: pandas.DataFrame
    :param country_col: Nombre de la columna que contiene los países
    :type country_col: str 
    :param results_dir: Directorio donde se guardará el gráfico generado
    :type results_dir: str
    :return: None
    :rtype: None
    :raises ValueError: Si la columna country_col no existe en el DataFrame
    :raises OSError: Si hay problemas al guardar el archivo en results_dir
    
    El gráfico generado muestra:
    
    * Las 20 países con mayor número de sucesos
    * El conteo de sucesos para cada país
    * Una paleta de colores 'viridis' para mejor visualización
    
    La imagen se guarda en formato PNG en el directorio especificado.
    
    :example:
        >>> df = pd.DataFrame({'pais': ['ESP','USA','FRA','ESP']})
        >>> plot_top_countries(df, 'pais', './output')
        Generando gráfica: Top países con más sucesos...
        Gráfico guardado: ./output/top_paises_sucesos.png
    """
    print("\nGenerando gráfica: Top países con más sucesos...")
    plt.figure(figsize=(14, 8))
    top_countries = df[country_col].dropna().astype(str).value_counts().head(20)
    sns.barplot(x=top_countries.values, y=top_countries.index, palette='viridis')
    plt.title('Top países con más sucesos registrados')
    plt.xlabel('Cantidad de sucesos')
    plt.ylabel('País')
    plt.tight_layout()
    path = os.path.join(results_dir, f'top_paises_sucesos.png')
    plt.savefig(path)
    plt.close()
    print(f"Gráfico guardado: {path}")

def plot_world_scatter(df, lat_col, lon_col, country_col, results_dir):
    """
    Genera un mapa mundial de dispersión con los sucesos georreferenciados.

    :param df: DataFrame que contiene los datos a graficar
    :type df: pandas.DataFrame
    :param lat_col: Nombre de la columna que contiene las latitudes
    :type lat_col: str
    :param lon_col: Nombre de la columna que contiene las longitudes 
    :type lon_col: str
    :param country_col: Nombre de la columna que contiene los países (opcional)
    :type country_col: str
    :param results_dir: Directorio donde se guardará el gráfico generado
    :type results_dir: str
    :return: None
    :rtype: None
    :raises ValueError: Si las columnas lat_col o lon_col no existen en el DataFrame
    :raises OSError: Si hay problemas al guardar el archivo en results_dir

    El mapa generado muestra:

    * Puntos de dispersión para cada suceso sobre un mapa mundial
    * Colores diferentes para cada país si se especifica country_col
    * Filtrado de coordenadas inválidas
    * Imagen de fondo del mapa mundial

    La imagen se guarda en formato PNG en el directorio especificado.

    :example:
        >>> df = pd.DataFrame({
        ...     'lat': [40.4, -33.9],
        ...     'lon': [-3.7, 151.2], 
        ...     'pais': ['ESP', 'AUS']
        ... })
        >>> plot_world_scatter(df, 'lat', 'lon', 'pais', './output')
        Generando mapa mundial de sucesos...
        Datos originales: 2
        Datos tras filtrado: 2
        Gráfico guardado: ./output/mapa_mundial_sucesos.png

    .. note::
        La función filtra automáticamente las coordenadas inválidas y muestra
        la cantidad de datos antes y después del filtrado.

    .. warning::
        Se requiere conexión a internet para descargar la imagen del mapa base.
    """
    print("\nGenerando mapa mundial de sucesos...")
    import matplotlib.colors as mcolors
    plt.figure(figsize=(16, 8))
    url = 'https://upload.wikimedia.org/wikipedia/commons/8/83/Equirectangular_projection_SW.jpg'
    with urllib.request.urlopen(url) as f:
        img = Image.open(f)
        world = np.array(img)
    plt.imshow(world, extent=[-180, 180, -90, 90], aspect='auto')
    # Conversión y filtrado estricto
    print(f"Datos originales: {len(df)}")
    df = df.copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    df = df.dropna(subset=[lat_col, lon_col])
    df = df[(df[lat_col] >= -90) & (df[lat_col] <= 90) & (df[lon_col] >= -180) & (df[lon_col] <= 180)]
    print(f"Datos tras filtrado: {len(df)}")
    if df.empty:
        print("No hay datos válidos para graficar.")
    if country_col:
        countries = df[country_col].astype(str)
        unique_countries = countries.unique()
        colors = dict(zip(unique_countries, sns.color_palette('hls', len(unique_countries))))
        for c in unique_countries:
            mask = countries == c
            plt.scatter(df.loc[mask, lon_col], df.loc[mask, lat_col], s=40, alpha=0.8, color=colors[c], edgecolor='black')
    else:
        plt.scatter(df[lon_col], df[lat_col], s=40, alpha=0.8, color='red', edgecolor='black')
    plt.title('Ubicación geográfica de los sucesos a nivel mundial')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.tight_layout()
    path = os.path.join(results_dir, f'mapa_mundial_sucesos.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Gráfico guardado: {path}")

def plot_world_heatmap(df, lat_col, lon_col, results_dir):
    """
    Genera un mapa de calor global de los sucesos utilizando una proyección equirectangular.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame que contiene los datos a graficar.
    lat_col : str
        Nombre de la columna que contiene las latitudes.
    lon_col : str
        Nombre de la columna que contiene las longitudes.
    results_dir : str
        Directorio donde se guardará el gráfico generado.

    Notes
    -----
    La función realiza las siguientes operaciones:
    - Carga una imagen base del mapa mundial
    - Filtra y limpia las coordenadas inválidas
    - Genera un mapa de calor usando kernel density estimation
    - Guarda el resultado como 'mapa_calor_mundial.png'

    Requires
    --------
    - Conexión a internet para descargar la imagen base del mapa
    - Las bibliotecas matplotlib, seaborn y PIL
    """
    print("\nGenerando mapa de calor global...")
    plt.figure(figsize=(16, 8))
    url = 'https://upload.wikimedia.org/wikipedia/commons/8/83/Equirectangular_projection_SW.jpg'
    with urllib.request.urlopen(url) as f:
        img = Image.open(f)
        world = np.array(img)
    plt.imshow(world, extent=[-180, 180, -90, 90], aspect='auto')
    # Conversión y filtrado estricto
    df = df.copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    df = df.dropna(subset=[lat_col, lon_col])
    df = df[(df[lat_col] >= -90) & (df[lat_col] <= 90) & (df[lon_col] >= -180) & (df[lon_col] <= 180)]
    sns.kdeplot(x=df[lon_col], y=df[lat_col], cmap='Reds', fill=True, alpha=0.5, bw_adjust=0.5, thresh=0.05)
    plt.title('Mapa de calor global de sucesos')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.tight_layout()
    path = os.path.join(results_dir, f'mapa_calor_mundial.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Gráfico guardado: {path}")

def plot_by_year(df, year_col, results_dir):
    """
    Genera un gráfico de barras mostrando la cantidad de sucesos por año.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame que contiene los datos a graficar.
    year_col : str
        Nombre de la columna que contiene los años.
    results_dir : str
        Directorio donde se guardará el gráfico generado.

    Notes
    -----
    La función realiza las siguientes operaciones:
    - Convierte la columna de años a formato datetime
    - Genera un gráfico de barras con la frecuencia por año
    - Guarda el resultado como 'sucesos_por_ano.png'
    """
    print("\nGenerando gráfica: Sucesos por año...")
    vals = pd.to_datetime(df[year_col], errors='coerce').dt.year
    plt.figure(figsize=(12, 6))
    vals.value_counts().sort_index().plot(kind='bar', color='#3498db')
    plt.title('Cantidad de sucesos por año')
    plt.xlabel('Año')
    plt.ylabel('Cantidad de sucesos')
    plt.tight_layout()
    path = os.path.join(results_dir, f'sucesos_por_ano.png')
    plt.savefig(path)
    plt.close()
    print(f"Gráfico guardado: {path}")

def plot_by_type(df, type_col, results_dir):
    """
    Genera un gráfico de barras horizontales mostrando los tipos más frecuentes de sucesos.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame que contiene los datos a graficar.
    type_col : str
        Nombre de la columna que contiene los tipos/categorías.
    results_dir : str
        Directorio donde se guardará el gráfico generado.

    Notes
    -----
    La función realiza las siguientes operaciones:
    - Selecciona los 20 tipos más frecuentes
    - Genera un gráfico de barras horizontales usando seaborn
    - Guarda el resultado como 'top_tipos_sucesos.png'
    """
    print("\nGenerando gráfica: Sucesos por tipo/categoría...")
    plt.figure(figsize=(14, 8))
    top_types = df[type_col].dropna().astype(str).value_counts().head(20)
    sns.barplot(x=top_types.values, y=top_types.index, palette='mako')
    plt.title('Top tipos/categorías de sucesos')
    plt.xlabel('Cantidad de sucesos')
    plt.ylabel('Tipo/Categoría')
    plt.tight_layout()
    path = os.path.join(results_dir, f'top_tipos_sucesos.png')
    plt.savefig(path)
    plt.close()
    print(f"Gráfico guardado: {path}")

def main():
    """
    Función principal que ejecuta el flujo completo de análisis geoespacial adaptativo.
    1. Carga y limpieza de datos.
    2. Uso explícito de columnas 'lat' y 'lon'.
    3. Limpieza de datos geoespaciales.
    4. Visualizaciones y clustering.
    5. Generación de informe.
    Autor: Anderson Zapata
    """
    print("=== Análisis Geoespacial Adaptativo Mejorado ===\n")
    proyecto_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    file_path = os.path.join(proyecto_dir, 'src', 'data', 'datos_limpios.csv')
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(results_dir, exist_ok=True)
    df = load_and_clean_data(file_path)
    if df is None:
        print("No se pudieron cargar los datos. Finalizando análisis.")
        return
    # Forzar uso de columnas 'lat' y 'lon'
    lat_col, lon_col = 'lat', 'lon'
    if lat_col not in df.columns or lon_col not in df.columns:
        print("No se encontraron las columnas 'lat' y 'lon' en los datos. Finalizando.")
        return
    geo_df = clean_geo_data(df, lat_col, lon_col)
    if len(geo_df) < 10:
        print("No hay suficientes datos geoespaciales válidos para realizar el análisis. Finalizando.")
        return
    plot_geo_scatter(geo_df, lat_col, lon_col, results_dir)
    plot_histograms(geo_df, lat_col, lon_col, results_dir)
    result = plot_kmeans_clusters(geo_df, lat_col, lon_col, results_dir)
    if result is not None:
        geo_df = result
    plot_bar_by_cluster(geo_df, results_dir)
    country_col = detect_country_column(geo_df)
    if country_col:
        plot_top_countries(geo_df, country_col, results_dir)
    plot_world_scatter(geo_df, lat_col, lon_col, country_col, results_dir)
    plot_world_heatmap(geo_df, lat_col, lon_col, results_dir)
    year_col = detect_year_column(geo_df)
    if year_col:
        plot_by_year(geo_df, year_col, results_dir)
    type_col = detect_type_column(geo_df)
    if type_col:
        plot_by_type(geo_df, type_col, results_dir)
    generate_report(geo_df, lat_col, lon_col, results_dir)
    print("\n=== Análisis Geoespacial Adaptativo Completado ===")
    print(f"Todos los resultados guardados en: {results_dir}")

if __name__ == "__main__":
    main()