import pandas as pd
import os
import csv
import re
import sys

# Añadir el directorio raíz del proyecto al path para poder importar módulos
proyecto_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, proyecto_dir)

"""
Módulo para limpieza avanzada de archivos CSV.

Este script permite limpiar, renombrar y filtrar datos de archivos CSV para análisis coherente y visualización.

Autor: Tu Nombre
Fecha: 2025
"""

def limpiar_analizar_csv():
    """
    Limpieza avanzada: asigna nombres claros, elimina columnas/filas innecesarias,
    y deja el DataFrame listo para análisis geográfico, temporal y de eventos.
    """
    file_path = os.path.join(proyecto_dir, 'src', 'data', '20250525.export.CSV')
    output_path = os.path.join(proyecto_dir, 'src', 'data', 'datos_limpios.csv')

    # 1. Detectar delimitador
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        contenido = f.read(5000)
        tabs = contenido.count('\t')
        comas = contenido.count(',')
        puntoycoma = contenido.count(';')
        if tabs > comas and tabs > puntoycoma:
            delimitador = '\t'
        elif comas > tabs and comas > puntoycoma:
            delimitador = ','
        elif puntoycoma > tabs and puntoycoma > comas:
            delimitador = ';'
        else:
            delimitador = '\t'

    # 2. Leer el archivo sin cabecera
    df = pd.read_csv(file_path, sep=delimitador, header=None, on_bad_lines='skip', low_memory=False)

    # 3. Asignar nombres claros a las columnas principales (ajustar según tus datos)
    columnas = [
        'id_evento', 'fecha_evento', 'mes_evento', 'año_evento', 'año_decimal',
        'codigo_region', 'region', 'subregion', 'col8', 'col9', 'col10', 'col11', 'col12', 'col13', 'col14', 'col15', 'col16', 'col17', 'col18', 'col19', 'col20', 'col21', 'col22', 'col23', 'col24',
        'valor1', 'valor2', 'valor3', 'valor4', 'valor5', 'valor6', 'valor7', 'valor8', 'valor9', 'latitud', 'longitud', 'ubicacion', 'pais_codigo', 'region_codigo', 'lat', 'lon', 'col40',
        'col41', 'col42', 'col43', 'col44', 'col45', 'col46', 'col47', 'col48', 'ubicacion2', 'pais_codigo2', 'region_codigo2', 'lat2', 'lon2', 'col54', 'fecha_evento2', 'fuente_url'
    ]
    # Ajustar el número de columnas si es necesario
    if len(df.columns) > len(columnas):
        columnas += [f'col_extra_{i}' for i in range(len(df.columns)-len(columnas))]
    df.columns = columnas[:len(df.columns)]

    # 4. Eliminar columnas innecesarias o automáticas
    columnas_utiles = [
        'id_evento', 'fecha_evento', 'año_evento', 'region', 'subregion',
        'latitud', 'longitud', 'ubicacion', 'pais_codigo', 'region_codigo',
        'lat', 'lon', 'ubicacion2', 'pais_codigo2', 'region_codigo2',
        'fecha_evento2', 'fuente_url'
    ]
    columnas_utiles = [col for col in columnas_utiles if col in df.columns]
    df = df[columnas_utiles]

    # 5. Eliminar filas con datos clave nulos o 'Desconocido'
    claves = ['fecha_evento', 'region', 'latitud', 'longitud', 'ubicacion', 'pais_codigo', 'fuente_url']
    claves = [col for col in claves if col in df.columns]
    for col in claves:
        df = df[df[col].notnull() & (df[col].astype(str).str.lower() != 'desconocido')]

    # 6. Convertir tipos de datos
    if 'fecha_evento' in df.columns:
        df['fecha_evento'] = pd.to_datetime(df['fecha_evento'], errors='coerce', format='%Y%m%d')
    for col in ['latitud', 'longitud', 'lat', 'lon']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 7. Eliminar filas con coordenadas nulas
    if 'latitud' in df.columns and 'longitud' in df.columns:
        df = df[df['latitud'].notnull() & df['longitud'].notnull()]

    # 8. Guardar datos limpios
    df.to_csv(output_path, index=False)
    print(f"\nDatos limpios y organizados guardados en: {output_path}")
    print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
    print(f"Columnas finales: {list(df.columns)}")
    return df

if __name__ == '__main__':
    print('=== LIMPIEZA Y ANÁLISIS DE DATOS CSV (COHERENTE) ===\n')
    df = limpiar_analizar_csv()
    if df is not None:
        print("\n=== El DataFrame está listo para análisis y visualización coherente ===")