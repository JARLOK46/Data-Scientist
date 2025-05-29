"""
Módulo de análisis geoespacial avanzado.

Este script permite cargar, limpiar y analizar datos geoespaciales, generando visualizaciones coherentes y un informe en Markdown. Incluye clustering espacial, mapas de dispersión, histogramas y mapas interactivos.

Autor: Anderson Zapata
Fecha: 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster, HeatMap
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Directorio de resultados
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(results_dir, exist_ok=True)

def detect_delimiter(file_path, num_lines=5):
    """
    Detecta automáticamente el delimitador utilizado en un archivo CSV.
    
    :param file_path: Ruta al archivo CSV.
    :type file_path: str
    :param num_lines: Número de líneas a analizar para la detección.
    :type num_lines: int
    :return: El delimitador detectado ('\t', ',', ';', '|').
    :rtype: str
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        sample = ''.join([file.readline() for _ in range(num_lines)])
    delimiters = {'\t': 0, ',': 0, ';': 0, '|': 0}
    for delimiter in delimiters:
        delimiters[delimiter] = sample.count(delimiter)
    return max(delimiters, key=delimiters.get)

def load_and_clean_data(file_path):
    """
    Carga y limpia un archivo CSV para análisis geoespacial.
    
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
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    df = df.drop_duplicates()
    return df

def identify_geo_columns(df):
    """
    Identifica las columnas de latitud y longitud en un DataFrame.

    :param df: DataFrame de entrada.
    :type df: pandas.DataFrame
    :return: Nombres de las columnas de latitud y longitud.
    :rtype: tuple(str, str) o (None, None)
    """
    lat_candidates = [col for col in df.columns if any(term in col.lower() for term in ['lat', 'latitude'])]
    lon_candidates = [col for col in df.columns if any(term in col.lower() for term in ['lon', 'lng', 'longitude'])]
    if lat_candidates and lon_candidates:
        return lat_candidates[0], lon_candidates[0]
    # Buscar columnas numéricas posibles
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
    Limpia el DataFrame para conservar solo filas con coordenadas válidas.

    :param df: DataFrame de entrada.
    :type df: pandas.DataFrame
    :param lat_col: Nombre de la columna de latitud.
    :type lat_col: str
    :param lon_col: Nombre de la columna de longitud.
    :type lon_col: str
    :return: DataFrame filtrado con coordenadas válidas.
    :rtype: pandas.DataFrame
    """
    geo_df = df.dropna(subset=[lat_col, lon_col]).copy()
    geo_df = geo_df[(geo_df[lat_col].astype(str).str.lower() != 'desconocido') & (geo_df[lon_col].astype(str).str.lower() != 'desconocido')]
    geo_df = geo_df[(geo_df[lat_col] >= -90) & (geo_df[lat_col] <= 90) & (geo_df[lon_col] >= -180) & (geo_df[lon_col] <= 180)]
    print(f"Datos geoespaciales válidos: {len(geo_df)} de {len(df)} filas")
    return geo_df

def plot_geo_scatter(geo_df, lat_col, lon_col):
    """
    Genera un mapa de dispersión de los puntos geográficos.

    :param geo_df: DataFrame con datos geoespaciales válidos.
    :type geo_df: pandas.DataFrame
    :param lat_col: Nombre de la columna de latitud.
    :type lat_col: str
    :param lon_col: Nombre de la columna de longitud.
    :type lon_col: str
    """
    plt.figure(figsize=(12, 10))
    plt.scatter(geo_df[lon_col], geo_df[lat_col], alpha=0.5, s=10)
    plt.title('Distribución Geográfica de Puntos')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, 'distribucion_geografica.png')
    plt.savefig(path)
    plt.close()
    print(f"Gráfico guardado: {path}")

def plot_histograms(geo_df, lat_col, lon_col):
    """
    Genera histogramas de latitud y longitud.

    :param geo_df: DataFrame con datos geoespaciales válidos.
    :type geo_df: pandas.DataFrame
    :param lat_col: Nombre de la columna de latitud.
    :type lat_col: str
    :param lon_col: Nombre de la columna de longitud.
    :type lon_col: str
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(geo_df[lat_col], kde=True)
    plt.title('Distribución de Latitudes')
    plt.xlabel('Latitud')
    plt.subplot(1, 2, 2)
    sns.histplot(geo_df[lon_col], kde=True)
    plt.title('Distribución de Longitudes')
    plt.xlabel('Longitud')
    plt.tight_layout()
    path = os.path.join(results_dir, 'histogramas_coordenadas.png')
    plt.savefig(path)
    plt.close()
    print(f"Gráfico guardado: {path}")

def plot_kmeans_clusters(geo_df, lat_col, lon_col):
    """
    Realiza clustering espacial con K-Means y genera el gráfico correspondiente.

    :param geo_df: DataFrame con datos geoespaciales válidos.
    :type geo_df: pandas.DataFrame
    :param lat_col: Nombre de la columna de latitud.
    :type lat_col: str
    :param lon_col: Nombre de la columna de longitud.
    :type lon_col: str
    :return: DataFrame con columna 'kmeans_cluster'.
    :rtype: pandas.DataFrame
    """
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
    plt.title('Clustering Espacial con K-Means')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, 'clusters_kmeans.png')
    plt.savefig(path)
    plt.close()
    print(f"Gráfico guardado: {path}")
    return geo_df

def plot_bar_by_cluster(geo_df):
    """
    Genera un gráfico de barras con la cantidad de puntos por cluster K-Means.

    :param geo_df: DataFrame con columna 'kmeans_cluster'.
    :type geo_df: pandas.DataFrame
    """
    if 'kmeans_cluster' not in geo_df.columns:
        return
    counts = geo_df['kmeans_cluster'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.index, y=counts.values, palette='viridis')
    plt.title('Cantidad de Puntos por Cluster K-Means')
    plt.xlabel('Cluster')
    plt.ylabel('Cantidad de puntos')
    plt.tight_layout()
    path = os.path.join(results_dir, 'puntos_por_cluster.png')
    plt.savefig(path)
    plt.close()
    print(f"Gráfico guardado: {path}")

def create_interactive_map(geo_df, lat_col, lon_col):
    """
    Genera un mapa interactivo con Folium y clusters de puntos.

    :param geo_df: DataFrame con datos geoespaciales válidos.
    :type geo_df: pandas.DataFrame
    :param lat_col: Nombre de la columna de latitud.
    :type lat_col: str
    :param lon_col: Nombre de la columna de longitud.
    :type lon_col: str
    """
    center_lat = geo_df[lat_col].mean()
    center_lon = geo_df[lon_col].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=2, tiles='OpenStreetMap')
    marker_cluster = MarkerCluster().add_to(m)
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
    heat_data = [[row[lat_col], row[lon_col]] for idx, row in sample_df.iterrows()]
    HeatMap(heat_data, radius=15).add_to(m)
    map_path = os.path.join(results_dir, 'mapa_interactivo.html')
    m.save(map_path)
    print(f"Mapa interactivo guardado en: {map_path}")
    
def generate_report(geo_df, lat_col, lon_col):
    """
    Genera un informe en formato Markdown con el resumen del análisis geoespacial.

    :param geo_df: DataFrame con datos geoespaciales válidos.
    :type geo_df: pandas.DataFrame
    :param lat_col: Nombre de la columna de latitud.
    :type lat_col: str
    :param lon_col: Nombre de la columna de longitud.
    :type lon_col: str
    """
    md_report = f"# Informe de Análisis Geoespacial\n\n"
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
    md_report += f"- Mapa de dispersión de puntos\n"
    md_report += f"- Histogramas de latitud y longitud\n"
    if 'kmeans_cluster' in geo_df.columns:
        md_report += f"- Clustering espacial con K-Means\n"
        md_report += f"- Barras de cantidad de puntos por cluster\n"
    md_report += f"- Mapa interactivo (HTML)\n"
    md_report += f"\n## Conclusiones\n\n"
    md_report += f"- Los datos muestran una distribución geográfica coherente.\n"
    if 'kmeans_cluster' in geo_df.columns:
        md_report += f"- El clustering permite identificar agrupaciones espaciales relevantes.\n"
    md_report_path = os.path.join(results_dir, 'informe_analisis_geoespacial.md')
    with open(md_report_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    print(f"Informe en formato Markdown guardado en: {md_report_path}")
    
def main():
    """
    Función principal que ejecuta el flujo completo de análisis geoespacial.

    1. Carga y limpieza de datos.
    2. Identificación de columnas geográficas.
    3. Limpieza de datos geoespaciales.
    4. Visualizaciones y clustering.
    5. Generación de informe.
    """
    print("=== Análisis Geoespacial Mejorado ===\n")
    proyecto_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    file_path = os.path.join(proyecto_dir, 'src', 'data', '20250525.export.CSV')
    df = load_and_clean_data(file_path)
    if df is None:
        print("No se pudieron cargar los datos. Finalizando análisis.")
        return
    lat_col, lon_col = identify_geo_columns(df)
    if lat_col is None or lon_col is None:
        print("No se pudieron identificar columnas geográficas. Finalizando análisis.")
        return
    geo_df = clean_geo_data(df, lat_col, lon_col)
    if len(geo_df) < 10:
        print("No hay suficientes datos geoespaciales válidos para realizar el análisis. Finalizando.")
        return
    plot_geo_scatter(geo_df, lat_col, lon_col)
    plot_histograms(geo_df, lat_col, lon_col)
    result = plot_kmeans_clusters(geo_df, lat_col, lon_col)
    if result is not None:
        geo_df = result
    plot_bar_by_cluster(geo_df)
    create_interactive_map(geo_df, lat_col, lon_col)
    generate_report(geo_df, lat_col, lon_col)
    print("\n=== Análisis Geoespacial Completado ===")
    print(f"Todos los resultados guardados en: {results_dir}")

if __name__ == "__main__":
    main()