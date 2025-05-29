"""
Módulo para análisis y limpieza básica de archivos CSV.

Este script permite cargar, explorar y analizar archivos CSV, mostrando información sobre el delimitador, estructura, tipos de datos y valores nulos. Es útil para entender la estructura inicial de un archivo antes de una limpieza profunda.

Autor: Anderson Zapata
Fecha: 2025
"""
import pandas as pd
import os

def analizar_csv():
    """
    Realiza un análisis exploratorio básico de un archivo CSV.

    - Verifica la existencia y tamaño del archivo.
    - Lee y muestra las primeras líneas del archivo en modo texto crudo.
    - Detecta el delimitador más probable (tabulación o coma).
    - Intenta leer el archivo con pandas usando varios delimitadores.
    - Muestra información básica: dimensiones, columnas, tipos de datos y valores nulos.

    :return: None
    """
    file_path = 'c:\programacion\UN PASO AL EXITO\20250525.export.CSV'
    
    # Verificar si el archivo existe y mostrar su tamaño
    print(f'Archivo existe: {os.path.exists(file_path)}')
    print(f'Tamaño del archivo: {os.path.getsize(file_path)} bytes')
    
    # Intentar leer las primeras líneas del archivo raw
    try:
        with open(file_path, 'r') as f:
            print('\nPrimeras 3 líneas (raw):')
            for i in range(3):
                line = f.readline()
                print(f'Línea {i+1}: {line[:100]}...')
                # Solo mostramos los primeros 100 caracteres para no saturar la salida
                
            # Analizar la estructura de la primera línea para detectar delimitador
            f.seek(0)
            first_line = f.readline()
            if '\t' in first_line:
                delimiter = '\t'
                print('\nDelimitador detectado: Tab')
            elif ',' in first_line:
                delimiter = ','
                print('\nDelimitador detectado: Coma')
            else:
                delimiter = None
                print('\nDelimitador detectado: Desconocido')
    except Exception as e:
        print(f'Error al leer archivo: {e}')
    
    # Intentar leer con pandas usando diferentes delimitadores
    try:
        # Probar con diferentes delimitadores comunes
        for sep in ['\t', ',', ';', '|']:
            try:
                if sep == '\t':
                    sep_name = 'Tab'
                else:
                    sep_name = sep
                print(f'\nIntentando leer CSV con delimitador: {sep_name}')
                df = pd.read_csv(file_path, sep=sep, nrows=5, on_bad_lines='skip')
                print('Éxito!')
                print(f'Dimensiones: {df.shape}')
                print(f'Columnas: {df.columns.tolist()}')
                print('\nPrimeras 2 filas:')
                print(df.head(2))
                
                # Análisis básico de tipos de datos
                print('\nTipos de datos:')
                print(df.dtypes)
                
                # Conteo de valores nulos por columna
                print('\nValores nulos por columna:')
                print(df.isnull().sum())
                
                # Si se pudo leer correctamente, no intentamos con otros delimitadores
                return
            except Exception as e:
                print(f'Error con delimitador {sep_name}: {e}')
    except Exception as e:
        print(f'Error general al leer CSV: {e}')

if __name__ == '__main__':
    print('=== ANÁLISIS DE ARCHIVO CSV ===')
    analizar_csv()