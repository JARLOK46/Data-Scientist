"""
Módulo de dashboard adaptativo para análisis de datos.

Este script permite generar visualizaciones estáticas, informes HTML y recomendaciones a partir de datos limpios, facilitando el análisis exploratorio y descriptivo de grandes volúmenes de información.

Flujo principal:
- Carga y limpieza de datos desde un archivo CSV.
- Generación de visualizaciones estáticas (histogramas, correlaciones, mapas, barras, etc.).
- Creación de un informe HTML con tablas, gráficos y recomendaciones automáticas.
- Opcionalmente, puede integrarse con dashboards interactivos.

Dependencias principales: pandas, numpy, matplotlib, seaborn, os, sys, urllib.parse, re.

Estructura del archivo:
- Funciones utilitarias para carga, limpieza y visualización.
- Función principal (main) que orquesta todo el proceso.

Autor: Anderson Zapata
Fecha: 2025
"""
import os
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
import webbrowser
warnings.filterwarnings('ignore')

# Añadir el directorio raíz del proyecto al path para poder importar módulos
proyecto_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, proyecto_dir)

# Crear directorios para visualizaciones
dashboard_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard_estatico')
os.makedirs(dashboard_dir, exist_ok=True)

def detect_delimiter(file_path, num_lines=5):
    """
    Detecta automáticamente el delimitador utilizado en un archivo CSV.

    Parámetros
    ----------
    file_path : str
        Ruta al archivo CSV que se desea analizar.
    num_lines : int, opcional
        Número de líneas a analizar para detectar el delimitador (por defecto 5).

    Retorna
    -------
    str
        El delimitador detectado (puede ser ',', ';', '\t' o '|').

    Advertencias
    ------------
    - Si el archivo tiene pocos datos o un formato inusual, la detección puede fallar.
    - Solo analiza las primeras líneas, por lo que si el delimitador cambia en el archivo, puede no detectarse correctamente.

    Ejemplo
    -------
    >>> detect_delimiter('datos.csv')
    ','

    Lógica Interna
    --------------
    1. Lee las primeras líneas del archivo.
    2. Cuenta cuántas veces aparece cada delimitador común.
    3. Retorna el que más aparece.

    Autor: Anderson Zapata
    Fecha: 2025
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        sample = ''.join([file.readline() for _ in range(num_lines)])
    
    delimiters = {'\t': 0, ',': 0, ';': 0, '|': 0}
    for delimiter in delimiters:
        delimiters[delimiter] = sample.count(delimiter)
    
    return max(delimiters, key=delimiters.get)

def load_and_clean_data(file_path):
    """
    Carga y limpia un archivo CSV, dejando los datos listos para análisis y visualización.

    Parámetros
    ----------
    file_path : str
        Ruta al archivo CSV.

    Retorna
    -------
    df_cleaned : pandas.DataFrame
        DataFrame limpio y listo para análisis.
    geo_cols : dict
        Diccionario con listas de nombres de columnas geográficas detectadas ('lat', 'lon').

    Advertencias
    ------------
    - Si el archivo tiene problemas de codificación, intenta con 'utf-8' y luego con 'latin1'.
    - Si no se puede cargar, retorna (None, None).
    - Si hay columnas con nombres poco claros, pueden no ser detectadas como geográficas.

    Ejemplo
    -------
    >>> df, geo_cols = load_and_clean_data('datos.csv')

    Lógica Interna
    --------------
    1. Detecta el delimitador.
    2. Intenta cargar el archivo con diferentes codificaciones.
    3. Limpia nombres de columnas y elimina duplicados.
    4. Rellena valores nulos en columnas numéricas y de texto.
    5. Extrae dominios de URLs si existen.
    6. Detecta columnas de latitud y longitud.

    Autor: Anderson Zapata
    Fecha: 2025
    """
    print(f"Cargando datos desde: {file_path}")
    
    # Detectar delimitador
    delimiter = detect_delimiter(file_path)
    print(f"Delimitador detectado: '{delimiter}'")
    
    # Cargar datos
    try:
        df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8', low_memory=False)
    except Exception as e:
        print(f"Error con encoding utf-8: {e}")
        try:
            df = pd.read_csv(file_path, delimiter=delimiter, encoding='latin1', low_memory=False)
        except Exception as e:
            print(f"Error con encoding latin1: {e}")
            return None, None
    
    print(f"Datos cargados. Dimensiones: {df.shape}")
    print(f"Nombres de columnas detectadas: {list(df.columns)}")
    
    # Limpiar nombres de columnas
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    
    # Eliminar duplicados
    df_cleaned = df.drop_duplicates()
    print(f"Duplicados eliminados: {df.shape[0] - df_cleaned.shape[0]}")
    
    # Identificar columnas numéricas y rellenar valores nulos
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
    
    # Identificar columnas de texto y rellenar valores nulos
    text_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in text_cols:
        df_cleaned[col].fillna("Desconocido", inplace=True)
    
    # Extraer dominios de URLs
    url_cols = [col for col in text_cols if any(url_term in col for url_term in ['url', 'link', 'web'])]
    for col in url_cols:
        try:
            df_cleaned[f'{col}_domain'] = df_cleaned[col].apply(
                lambda x: urlparse(x).netloc if pd.notna(x) and isinstance(x, str) and x.startswith('http') else "")
        except:
            print(f"No se pudo extraer dominio de la columna {col}")
    
    # Identificar columnas geográficas
    geo_cols = {
        'lat': [col for col in df_cleaned.columns if any(term in col.lower() for term in ['lat', 'latitude'])],
        'lon': [col for col in df_cleaned.columns if any(term in col.lower() for term in ['lon', 'lng', 'longitude'])]
    }
    
    if geo_cols['lat'] and geo_cols['lon']:
        print(f"Columnas geográficas detectadas: {geo_cols}")
    
    return df_cleaned, geo_cols

def generate_static_visualizations(df, geo_cols, output_dir):
    """
    Genera visualizaciones estáticas y estadísticas descriptivas a partir de un DataFrame limpio.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame limpio.
    geo_cols : dict
        Diccionario con listas de columnas geográficas ('lat', 'lon').
    output_dir : str
        Directorio donde se guardarán las imágenes y archivos generados.

    Retorna
    -------
    bool
        True si se generaron correctamente las visualizaciones.

    Advertencias
    ------------
    - Si no hay columnas numéricas o geográficas, se omiten ciertos gráficos.
    - El directorio de salida debe existir y ser escribible.
    - Si hay muchas columnas, solo se grafican las más relevantes para evitar sobrecarga visual.

    Ejemplo
    -------
    >>> generate_static_visualizations(df, geo_cols, 'output/')

    Lógica Interna
    --------------
    1. Genera histogramas de variables numéricas.
    2. Genera matriz de correlación.
    3. Genera gráficos de dispersión.
    4. Genera gráficos de barras para variables categóricas o dominios.
    5. Si hay coordenadas, genera un mapa de dispersión.
    6. Guarda estadísticas descriptivas en CSV.
    7. Llama a la función para generar el informe HTML.

    Autor: Anderson Zapata
    Fecha: 2025
    """
    print("Generando visualizaciones estáticas...")
    
    # Configurar estilo de las visualizaciones
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("viridis")
    
    # Filtrar columnas relevantes
    irrelevantes = ['id', 'fecha', 'unknown', 'desconocido']
    def es_relevante(col):
        if any(term in col.lower() for term in irrelevantes):
            return False
        if df[col].isnull().mean() > 0.5:
            return False
        return True
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if es_relevante(col)]
    text_cols = [col for col in df.select_dtypes(include=['object']).columns if es_relevante(col)]
    
    # Usar columnas explícitas según datos_limpios.csv
    year_col = 'año_evento'
    country_col = 'pais_codigo'
    region_col = 'region'
    # 1. Gráfica de barras: sucesos por año
    if year_col in df.columns:
        print('Generando gráfica de sucesos por año...')
        years = pd.to_numeric(df[year_col], errors='coerce').dropna().astype(int)
        year_counts = years.value_counts().sort_index()
        plt.figure(figsize=(12, 6))
        sns.barplot(x=year_counts.index, y=year_counts.values, palette='crest')
        plt.title('Cantidad de sucesos políticos por año', fontsize=18, fontweight='bold')
        plt.xlabel('Año', fontsize=14)
        plt.ylabel('Cantidad de sucesos', fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        path_ano = os.path.join(dashboard_dir, 'sucesos_por_ano.png')
        plt.savefig(path_ano, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico de sucesos por año guardado en: {os.path.abspath(path_ano)}")

        # 3. Tasa de crecimiento anual (%)
        print('Generando gráfica de tasa de crecimiento anual...')
        tasa = year_counts.pct_change().fillna(0) * 100
        plt.figure(figsize=(12, 6))
        sns.barplot(x=year_counts.index, y=tasa.values, palette='flare')
        plt.title('Tasa de crecimiento anual de sucesos políticos (%)', fontsize=18, fontweight='bold')
        plt.xlabel('Año', fontsize=14)
        plt.ylabel('Tasa de crecimiento (%)', fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.tight_layout()
        path_tasa = os.path.join(dashboard_dir, 'tasa_crecimiento_anual.png')
        plt.savefig(path_tasa, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico de tasa de crecimiento anual guardado en: {os.path.abspath(path_tasa)}")

    # 2. Gráfica de torta: proporción por país
    if country_col in df.columns:
        print('Generando gráfica de torta de proporción por país...')
        pie_data = df[country_col].dropna().astype(str)
        pie_data = pie_data[pie_data.str.lower() != 'desconocido']
        pie_counts = pie_data.value_counts().head(10)
        plt.figure(figsize=(10, 10))
        plt.pie(pie_counts.values, labels=pie_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
        plt.title('Distribución porcentual de sucesos políticos por país', fontsize=18, fontweight='bold')
        plt.tight_layout()
        path_pie = os.path.join(dashboard_dir, 'proporcion_sucesos_pais.png')
        plt.savefig(path_pie, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico de torta guardado en: {os.path.abspath(path_pie)}")

    # 7. Generar estadísticas descriptivas
    print("Generando estadísticas descriptivas...")
    stats = df.describe(include='all').transpose()
    stats_path = os.path.join(output_dir, 'estadisticas_descriptivas.csv')
    stats.to_csv(stats_path)
    print(f"Estadísticas descriptivas guardadas en: {stats_path}")
    
    # 8. Generar informe HTML
    generate_html_report(df, output_dir, geo_cols)
    
    return True

def generate_html_report(df, output_dir, geo_cols):
    """
    Genera un informe HTML con visualizaciones, estadísticas y recomendaciones automáticas.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame limpio y analizado.
    output_dir : str
        Directorio donde se guardarán las imágenes y el informe HTML.
    geo_cols : dict
        Diccionario con listas de columnas geográficas ('lat', 'lon').

    Retorna
    -------
    str
        Ruta al archivo HTML generado.

    Advertencias
    ------------
    - El informe solo enlaza imágenes PNG generadas previamente en el directorio de salida.
    - Si no existen imágenes, la sección de visualizaciones puede quedar vacía.

    Ejemplo
    -------
    >>> generate_html_report(df, 'output/', geo_cols)

    Lógica Interna
    --------------
    1. Busca imágenes PNG en el directorio de salida.
    2. Construye el HTML con resumen, visualizaciones, estadísticas y recomendaciones.
    3. Escribe el HTML en disco y retorna la ruta.

    Autor: Anderson Zapata
    Fecha: 2025
    """
    print("Generando informe HTML...")
    
    # Obtener listas de imágenes generadas
    image_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    
    # Crear contenido HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Informe de Análisis de Datos</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
            .section {{ margin-bottom: 30px; background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            .image-container {{ margin: 20px 0; text-align: center; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px 15px; border-bottom: 1px solid #ddd; text-align: left; }}
            th {{ background-color: #f8f9fa; }}
            tr:hover {{ background-color: #f1f1f1; }}
            .footer {{ text-align: center; margin-top: 30px; color: #7f8c8d; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Informe de Análisis de Datos</h1>
                <p>Generado automáticamente a partir del archivo CSV</p>
            </div>
            
            <div class="section">
                <h2>Resumen del Conjunto de Datos</h2>
                <p>Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas</p>
                <p>Columnas numéricas: {len(df.select_dtypes(include=[np.number]).columns)}</p>
                <p>Columnas categóricas: {len(df.select_dtypes(include=['object']).columns)}</p>
                <p>Valores nulos totales: {df.isna().sum().sum()}</p>
                
                <h3>Primeras filas del conjunto de datos:</h3>
                <div style="overflow-x: auto;">
                    {df.head(5).to_html()}
                </div>
            </div>
    """
    
    # Sección de visualizaciones
    html_content += """
            <div class="section">
                <h2>Visualizaciones</h2>
    """
    
    # Agregar imágenes al informe
    for img_file in image_files:
        img_path = os.path.join('dashboard_estatico', img_file)
        img_title = ' '.join(img_file.replace('.png', '').split('_')).title()
        html_content += f"""
                <div class="image-container">
                    <h3>{img_title}</h3>
                    <img src="{img_path}" alt="{img_title}">
                </div>
        """
    
    html_content += """
            </div>
    """
    
    # Sección de estadísticas
    html_content += """
            <div class="section">
                <h2>Estadísticas Descriptivas</h2>
                <div style="overflow-x: auto;">
                    {}
                </div>
            </div>
    """.format(df.describe().to_html())
    
    # Sección de recomendaciones
    html_content += """
            <div class="section">
                <h2>Recomendaciones</h2>
                <ul>
    """
    
    # Generar recomendaciones basadas en el análisis
    recommendations = []
    
    # Recomendación sobre valores nulos
    null_cols = df.columns[df.isna().any()].tolist()
    if null_cols:
        recommendations.append(f"<li>Se detectaron valores nulos en {len(null_cols)} columnas. Considere técnicas de imputación más avanzadas para mejorar la calidad de los datos.</li>")
    
    # Recomendación sobre correlaciones
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        high_corr = [(i, j, corr_matrix.loc[i, j]) 
                    for i in corr_matrix.columns 
                    for j in corr_matrix.columns 
                    if i < j and abs(corr_matrix.loc[i, j]) > 0.7]
        if high_corr:
            recommendations.append(f"<li>Se detectaron {len(high_corr)} pares de variables con alta correlación (>0.7). Considere técnicas de reducción de dimensionalidad como PCA.</li>")
    
    # Recomendación sobre datos geográficos
    if geo_cols['lat'] and geo_cols['lon']:
        recommendations.append("<li>Los datos contienen información geográfica. Considere realizar análisis de clustering espacial o visualizaciones en mapas interactivos.</li>")
    
    # Recomendación sobre dominios de URL
    domain_cols = [col for col in df.columns if 'domain' in col]
    if domain_cols:
        recommendations.append("<li>Se han extraído dominios de URLs. Considere un análisis más profundo del contenido web relacionado con estos dominios.</li>")
    
    # Recomendación sobre dashboard interactivo
    recommendations.append("<li>Para un análisis más interactivo, instale las dependencias necesarias ejecutando: <code>pip install -r requirements_dashboard.txt</code> y luego ejecute el dashboard interactivo.</li>")
    
    # Agregar recomendaciones al HTML
    for rec in recommendations:
        html_content += f"                {rec}\n"
    
    # Cerrar la lista de recomendaciones y finalizar el HTML
    html_content += """
                </ul>
            </div>
            
            <div class="footer">
                <p>Informe generado automáticamente - © 2025 Análisis de Datos Avanzado</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Guardar el informe HTML
    html_path = os.path.join(output_dir, 'informe_analisis.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Informe HTML guardado en: {html_path}")
    return html_path

def plot_barras_coherentes(df, col, path):
    """
    Genera y guarda un gráfico de barras para una columna categórica, omitiendo valores nulos o desconocidos.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame limpio.
    col : str
        Nombre de la columna categórica a graficar.
    path : str
        Ruta donde se guardará la imagen PNG generada.

    Advertencias
    ------------
    - Solo se grafican los 15 valores más frecuentes.
    - Si la columna tiene pocos valores únicos, el gráfico puede ser poco informativo.

    Ejemplo
    -------
    >>> plot_barras_coherentes(df, 'ciudad', 'output/ciudad.png')

    Lógica Interna
    --------------
    1. Filtra valores nulos o 'desconocido'.
    2. Cuenta los valores más frecuentes.
    3. Genera y guarda el gráfico de barras.

    Autor: Anderson Zapata
    Fecha: 2025
    """
    df = df[df[col].notnull() & (df[col].astype(str).str.lower() != 'desconocido')]
    conteo = df[col].value_counts().head(15)
    plt.figure(figsize=(12, 6))
    conteo.plot(kind='bar', color='skyblue')
    plt.title(f'Eventos por {col.capitalize()}')
    plt.xlabel(col.capitalize())
    plt.ylabel('Cantidad de eventos')
    plt.tight_layout()
    plt.savefig(path)
    print(f"Gráfico de barras por {col} guardado: {path}")

def main():
    """
    Función principal que orquesta la generación del dashboard adaptativo.

    Este método ejecuta el flujo completo: carga y limpieza de datos, generación de visualizaciones y creación del informe HTML.

    Retorna
    -------
    bool
        True si el proceso se completó correctamente, False si hubo errores.

    Advertencias
    ------------
    - El archivo de datos debe existir y estar en la ruta esperada.
    - Si la carga de datos falla, el proceso se detiene.
    - El directorio de salida debe tener permisos de escritura.

    Ejemplo
    -------
    >>> main()

    Lógica Interna
    --------------
    1. Carga y limpia los datos desde un archivo CSV.
    2. Genera visualizaciones estáticas y estadísticas.
    3. Crea un informe HTML con los resultados.
    4. Informa al usuario sobre la ubicación de los archivos generados.

    Autor: Anderson Zapata
    Fecha: 2025
    """
    print("=== Dashboard Adaptativo de Análisis de Datos ===")
    print("Este script generará visualizaciones estáticas y un informe HTML.")
    print("Para un dashboard interactivo, instale las dependencias con: pip install -r requirements_dashboard.txt")
    print("\nIniciando análisis...")
    
    # Cargar datos
    file_path = os.path.join(proyecto_dir, 'src', 'data', 'datos_limpios.csv')
    df, geo_cols = load_and_clean_data(file_path)
    
    if df is None:
        print("Error: No se pudieron cargar los datos. Verifique el formato y la ruta del archivo.")
        return False
    
    # Generar visualizaciones estáticas
    success = generate_static_visualizations(df, geo_cols, dashboard_dir)
    
    if success:
        print("\n=== Proceso completado con éxito ===")
        print(f"Se han generado visualizaciones estáticas en: {dashboard_dir}")
        informe_path = os.path.join(dashboard_dir, 'informe_analisis.html')
        print(f"Abra el informe HTML para ver los resultados: {informe_path}")
        print("\nPara un dashboard interactivo, instale las dependencias con: pip install -r requirements_dashboard.txt")
        print("y luego ejecute: python dashboard_interactivo.py")
        # Preguntar al usuario si desea abrir el informe HTML
        respuesta = input("¿Desea abrir el informe HTML en su navegador ahora? (s/n): ").strip().lower()
        if respuesta in ['s', 'si', 'y', 'yes']:
            webbrowser.open(f'file://{os.path.abspath(informe_path)}')
    else:
        print("\nError: No se pudieron generar todas las visualizaciones.")
    
    return success

# Ejecutar el script
if __name__ == '__main__':
    main()