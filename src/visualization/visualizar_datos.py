import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from urllib.parse import urlparse
import re
import sys

# Añadir el directorio raíz del proyecto al path para poder importar módulos
proyecto_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, proyecto_dir)

"""
Módulo para visualización de datos limpios.

Este script genera visualizaciones estáticas y descriptivas a partir de un DataFrame limpio, listas para ser usadas en informes o dashboards.

Autor: Tu Nombre
Fecha: 2025
"""

def visualizar_datos():
    """
    Visualizaciones coherentes usando solo columnas limpias y relevantes.
    - Elimina filas con datos faltantes o 'Desconocido'.
    - Solo usa columnas como fecha_evento, region, pais_codigo, latitud, longitud, fuente_url.
    - Genera mapas de dispersión, barras por país/región/año, tendencias temporales.
    - Etiquetas claras en ejes y títulos.
    """
    # Rutas de archivos
    datos_limpios_path = os.path.join(proyecto_dir, 'src', 'data', 'datos_limpios.csv')
    output_dir = os.path.join(proyecto_dir, 'src', 'visualization', 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(datos_limpios_path):
        print('No se encontró el archivo de datos limpios.')
        return
    df = pd.read_csv(datos_limpios_path)

    # Filtrar solo columnas relevantes si existen
    columnas_relevantes = [
        'fecha_evento', 'region', 'pais_codigo', 'latitud', 'longitud', 'fuente_url'
    ]
    columnas_relevantes = [col for col in columnas_relevantes if col in df.columns]
    df = df[columnas_relevantes]

    # Eliminar filas con datos faltantes o 'Desconocido'
    for col in columnas_relevantes:
        df = df[df[col].notnull() & (df[col].astype(str).str.lower() != 'desconocido')]

    print(f"\nDatos limpios para graficar: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(df.head())

    # Gráfico 1: Mapa de dispersión de eventos (si hay latitud y longitud)
    if 'latitud' in df.columns and 'longitud' in df.columns:
        plt.figure(figsize=(12, 8))
        plt.scatter(df['longitud'], df['latitud'], alpha=0.5, c='red', edgecolor='k')
        plt.title('Mapa de eventos geolocalizados')
        plt.xlabel('Longitud')
        plt.ylabel('Latitud')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mapa_eventos_geolocalizados.png'))
        print(f"Mapa de eventos guardado: {os.path.join(output_dir, 'mapa_eventos_geolocalizados.png')}")

    # Gráfico 2: Barras de eventos por región
    if 'region' in df.columns:
        conteo_region = df['region'].value_counts().head(15)
        plt.figure(figsize=(12, 6))
        conteo_region.plot(kind='bar', color='skyblue')
        plt.title('Eventos por Región')
        plt.xlabel('Región')
        plt.ylabel('Cantidad de eventos')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'eventos_por_region.png'))
        print(f"Gráfico de barras por región guardado: {os.path.join(output_dir, 'eventos_por_region.png')}")

    # Gráfico 3: Barras de eventos por país
    if 'pais_codigo' in df.columns:
        conteo_pais = df['pais_codigo'].value_counts().head(15)
        plt.figure(figsize=(12, 6))
        conteo_pais.plot(kind='bar', color='orange')
        plt.title('Eventos por País')
        plt.xlabel('País')
        plt.ylabel('Cantidad de eventos')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'eventos_por_pais.png'))
        print(f"Gráfico de barras por país guardado: {os.path.join(output_dir, 'eventos_por_pais.png')}")

    # Gráfico 4: Tendencia temporal de eventos por año (si hay fecha)
    if 'fecha_evento' in df.columns:
        df['año'] = pd.to_datetime(df['fecha_evento'], errors='coerce').dt.year
        conteo_anual = df['año'].value_counts().sort_index()
        plt.figure(figsize=(10, 6))
        conteo_anual.plot(kind='bar', color='green')
        plt.title('Eventos por Año')
        plt.xlabel('Año')
        plt.ylabel('Cantidad de eventos')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'eventos_por_año.png'))
        print(f"Gráfico de barras por año guardado: {os.path.join(output_dir, 'eventos_por_año.png')}")

    # Gráfico 5: Top 10 dominios de fuentes (si hay fuente_url)
    if 'fuente_url' in df.columns:
        df['dominio'] = df['fuente_url'].astype(str).apply(lambda x: urlparse(x).netloc if 'http' in x else '')
        dominios = df['dominio'].value_counts().head(10)
        plt.figure(figsize=(12, 6))
        dominios.plot(kind='bar', color='purple')
        plt.title('Top 10 dominios de fuentes de eventos')
        plt.xlabel('Dominio')
        plt.ylabel('Cantidad de eventos')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_dominios_fuente.png'))
        print(f"Gráfico de dominios guardado: {os.path.join(output_dir, 'top_dominios_fuente.png')}")

    print("\n=== Visualizaciones coherentes generadas correctamente ===")

if __name__ == '__main__':
    print('=== VISUALIZACIÓN DE DATOS ===\n')
    visualizar_datos()