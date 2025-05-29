"""
Módulo de dashboard interactivo para análisis de datos.

Este script permite visualizar y explorar datos de manera interactiva usando Dash y Plotly. Ofrece filtros, gráficos dinámicos, mapas y tablas estadísticas para análisis exploratorio avanzado.

Flujo principal:
- Carga y limpieza de datos desde un archivo CSV.
- Inicialización de la aplicación Dash.
- Definición de callbacks para actualizar gráficos en tiempo real según los filtros del usuario.
- Visualización de histogramas, dispersión, barras, mapas y estadísticas descriptivas.

Dependencias principales: dash, plotly, pandas, numpy, os, sys, urllib.parse, re.

Estructura del archivo:
- Funciones utilitarias para carga y limpieza de datos.
- Definición de la interfaz y callbacks de Dash.
- Ejecución del servidor web local.

Autor: Anderson Zapata
Fecha: 2025
"""
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from urllib.parse import urlparse
import os
import csv
import re
import sys

# Añadir el directorio raíz del proyecto al path para poder importar módulos
proyecto_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, proyecto_dir)

# Crear directorio para el dashboard si no existe
dashboard_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard')
os.makedirs(dashboard_dir, exist_ok=True)

# Función para detectar el delimitador
def detect_delimiter(file_path, num_lines=5):
    """
    Detecta automáticamente el delimitador utilizado en un archivo CSV.

    :param str file_path: Ruta completa al archivo CSV a analizar.
    :param int num_lines: Número de líneas a analizar para la detección. Por defecto es 5.
    :return: El delimitador detectado ('\t', ',', ';', '|') que aparece con mayor frecuencia.
    :rtype: str

    .. warning::
        Si el archivo tiene pocos datos o un formato inusual, la detección puede fallar.
        Solo analiza las primeras líneas, por lo que si el delimitador cambia en el archivo, puede no detectarse correctamente.

    :Example:
        >>> delimitador = detect_delimiter('datos.csv')
        >>> print(f"El delimitador detectado es: {delimitador}")

    :author: Anderson Zapata
    :date: 2025
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        sample = ''.join([file.readline() for _ in range(num_lines)])
    delimiters = {'\t': 0, ',': 0, ';': 0, '|': 0}
    for delimiter in delimiters:
        delimiters[delimiter] = sample.count(delimiter)
    return max(delimiters, key=delimiters.get)

# Cargar y limpiar datos
def load_and_clean_data(file_path):
    """
    Carga un archivo CSV y realiza un proceso de limpieza básica de los datos.

    :param str file_path: Ruta completa al archivo CSV a cargar y limpiar.
    :return: Una tupla con el DataFrame limpio y un diccionario con listas de columnas geográficas ('lat', 'lon').
    :rtype: tuple[pandas.DataFrame, dict]

    .. warning::
        Si el archivo tiene problemas de codificación, intenta con 'utf-8' y luego con 'latin1'.
        Si no se puede cargar, retorna None.
        Si hay columnas con nombres poco claros, pueden no ser detectadas como geográficas.

    :Example:
        >>> df, geo_cols = load_and_clean_data('datos.csv')

    :author: Anderson Zapata
    :date: 2025
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

# Inicializar la aplicación Dash
app = dash.Dash(__name__, title="Dashboard Interactivo de Análisis de Datos")
server = app.server

# Cargar datos
file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'datos_limpios.csv')
df, geo_cols = load_and_clean_data(file_path)

# Filtrar columnas relevantes
irrelevantes = ['id', 'fecha', 'unknown', 'desconocido']
def es_relevante(col):
    """
    Determina si una columna es relevante para el análisis.

    Esta función evalúa si una columna debe ser incluida en el análisis basándose en dos criterios:
    1. Si el nombre de la columna contiene términos irrelevantes predefinidos
    2. Si la columna tiene más del 50% de valores nulos

    Args:
        col (str): Nombre de la columna a evaluar

    Returns:
        bool: True si la columna es relevante, False en caso contrario

    Nota:
        Utiliza la variable global 'irrelevantes' que contiene términos a excluir
        y la variable global 'df' que es el DataFrame con los datos
    """
    if any(term in col.lower() for term in irrelevantes):
        return False
    if df[col].isnull().mean() > 0.5:
        return False
    return True

numeric_columns = [col for col in df.select_dtypes(include=[np.number]).columns if es_relevante(col)]
text_columns = [col for col in df.select_dtypes(include=['object']).columns if es_relevante(col)]
all_columns = numeric_columns + text_columns

# Filtros únicos
region_options = sorted(df['region'].dropna().unique()) if 'region' in df.columns else []
subregion_options = sorted(df['subregion'].dropna().unique()) if 'subregion' in df.columns else []
ubicacion_options = sorted(df['ubicacion'].dropna().unique()) if 'ubicacion' in df.columns else []

# Verificar si se cargaron los datos correctamente
if df is None:
    app.layout = html.Div([
        html.H1("Error al cargar los datos"),
        html.P("No se pudieron cargar los datos del archivo CSV. Verifique el formato y la ruta del archivo.")
    ])
else:
    # Diseño del dashboard
    app.layout = html.Div([
        html.H1("Dashboard Interactivo de Análisis de Datos", style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
        html.Div([
            html.Div([
                html.H3("Filtros y Controles", style={'color': '#2c3e50'}),
                html.Label("Filtrar por Región:"),
                dcc.Dropdown(id='region-filter', options=[{'label': r, 'value': r} for r in region_options], value=None, placeholder="Todas"),
                html.Label("Filtrar por Subregión:"),
                dcc.Dropdown(id='subregion-filter', options=[{'label': s, 'value': s} for s in subregion_options], value=None, placeholder="Todas"),
                html.Label("Filtrar por Ubicación:"),
                dcc.Dropdown(id='ubicacion-filter', options=[{'label': u, 'value': u} for u in ubicacion_options], value=None, placeholder="Todas"),
                html.Br(),
                html.Label("Seleccione columna para histograma:"),
                dcc.Dropdown(id='histogram-column', options=[{'label': col, 'value': col} for col in numeric_columns], value=numeric_columns[0] if numeric_columns else None),
                html.Br(),
                html.Label("Seleccione columnas para gráfico de dispersión:"),
                html.Label("Eje X:"),
                dcc.Dropdown(id='scatter-x-column', options=[{'label': col, 'value': col} for col in numeric_columns], value=numeric_columns[0] if numeric_columns else None),
                html.Label("Eje Y:"),
                dcc.Dropdown(id='scatter-y-column', options=[{'label': col, 'value': col} for col in numeric_columns], value=numeric_columns[1] if len(numeric_columns) > 1 else numeric_columns[0] if numeric_columns else None),
                html.Label("Color (opcional):"),
                dcc.Dropdown(id='scatter-color-column', options=[{'label': col, 'value': col} for col in all_columns], value=None),
                html.Br(),
                html.Label("Seleccione columna para gráfico de barras:"),
                dcc.Dropdown(id='bar-column', options=[{'label': col, 'value': col} for col in text_columns], value=text_columns[0] if text_columns else None),
                html.Label("Número de categorías a mostrar:"),
                dcc.Slider(id='bar-top-n', min=5, max=20, step=1, value=10, marks={i: str(i) for i in range(5, 21, 5)})
            ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),
            html.Div([
                html.Div([
                    html.H3("Distribución de Variables", style={'color': '#2c3e50', 'textAlign': 'center'}),
                    dcc.Graph(id='histogram-plot')
                ], style={'marginBottom': '20px', 'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '10px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.08)'}),
                html.Div([
                    html.H3("Relación entre Variables", style={'color': '#2c3e50', 'textAlign': 'center'}),
                    dcc.Graph(id='scatter-plot')
                ], style={'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '10px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.08)'})
            ], style={'width': '73%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ], style={'width': '100%', 'display': 'flex', 'flexDirection': 'row', 'gap': '2%'}),
        html.Div([
            html.Div([
                html.H3("Frecuencia de Categorías", style={'color': '#2c3e50', 'textAlign': 'center'}),
                dcc.Graph(id='bar-plot')
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '10px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.08)'}),
            html.Div([
                html.H3("Distribución Geográfica", style={'color': '#2c3e50', 'textAlign': 'center'}),
                dcc.Graph(id='map-plot')
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '10px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.08)', 'marginLeft': '4%'})
        ], style={'marginTop': '30px'}),
        html.Div([
            html.H3("Estadísticas Descriptivas", style={'color': '#2c3e50', 'textAlign': 'center'}),
            html.Div(id='stats-table')
        ], style={'marginTop': '30px', 'backgroundColor': 'white', 'padding': '15px', 'borderRadius': '10px', 'boxShadow': '0px 0px 10px rgba(0,0,0,0.08)'})
    ], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#ecf0f1'})

    # Callbacks para filtrar el DataFrame según los filtros seleccionados
    def filter_df(region, subregion, ubicacion):
        """
        Filtra el DataFrame global según los valores seleccionados de región, subregión y ubicación.

        :param region: Valor de la región seleccionada o None.
        :type region: str or None
        :param subregion: Valor de la subregión seleccionada o None.
        :type subregion: str or None
        :param ubicacion: Valor de la ubicación seleccionada o None.
        :type ubicacion: str or None
        :return: DataFrame filtrado según los parámetros.
        :rtype: pandas.DataFrame
        """
        dff = df.copy()
        if region:
            dff = dff[dff['region'] == region]
        if subregion:
            dff = dff[dff['subregion'] == subregion]
        if ubicacion:
            dff = dff[dff['ubicacion'] == ubicacion]
        return dff

    # Modificar todos los callbacks para usar el DataFrame filtrado
    @app.callback(
        Output('histogram-plot', 'figure'),
        [Input('histogram-column', 'value'),
         Input('region-filter', 'value'),
         Input('subregion-filter', 'value'),
         Input('ubicacion-filter', 'value')]
    )
    def update_histogram(column, region, subregion, ubicacion):
        """
        Actualiza el histograma según la columna y los filtros seleccionados.

        :param column: Nombre de la columna numérica a graficar.
        :type column: str
        :param region: Región seleccionada o None.
        :type region: str or None
        :param subregion: Subregión seleccionada o None.
        :type subregion: str or None
        :param ubicacion: Ubicación seleccionada o None.
        :type ubicacion: str or None
        :return: Figura de Plotly con el histograma.
        :rtype: plotly.graph_objs._figure.Figure
        """
        dff = filter_df(region, subregion, ubicacion)
        if column is None or column not in dff.columns:
            return go.Figure()
        fig = px.histogram(
            dff, x=column,
            title=f'Distribución de {column}' + (f' en {region}' if region else ''),
            color_discrete_sequence=['#3498db'],
            opacity=0.8
        )
        fig.update_layout(
            xaxis_title=column,
            yaxis_title="Frecuencia",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50')
        )
        return fig

    @app.callback(
        Output('scatter-plot', 'figure'),
        [Input('scatter-x-column', 'value'),
         Input('scatter-y-column', 'value'),
         Input('scatter-color-column', 'value'),
         Input('region-filter', 'value'),
         Input('subregion-filter', 'value'),
         Input('ubicacion-filter', 'value')]
    )
    def update_scatter(x_column, y_column, color_column, region, subregion, ubicacion):
        """
        Actualiza el gráfico de dispersión según las columnas y filtros seleccionados.

        :param x_column: Columna para el eje X.
        :type x_column: str
        :param y_column: Columna para el eje Y.
        :type y_column: str
        :param color_column: Columna para el color (opcional).
        :type color_column: str or None
        :param region: Región seleccionada o None.
        :type region: str or None
        :param subregion: Subregión seleccionada o None.
        :type subregion: str or None
        :param ubicacion: Ubicación seleccionada o None.
        :type ubicacion: str or None
        :return: Figura de Plotly con el gráfico de dispersión.
        :rtype: plotly.graph_objs._figure.Figure
        """
        dff = filter_df(region, subregion, ubicacion)
        if x_column is None or y_column is None or x_column not in dff.columns or y_column not in dff.columns:
            return go.Figure()
        if color_column and color_column in dff.columns:
            fig = px.scatter(
                dff, x=x_column, y=y_column, color=color_column,
                title=f'Relación entre {x_column} y {y_column}' + (f' en {region}' if region else ''),
                opacity=0.7
            )
        else:
            fig = px.scatter(
                dff, x=x_column, y=y_column,
                title=f'Relación entre {x_column} y {y_column}' + (f' en {region}' if region else ''),
                color_discrete_sequence=['#3498db'],
                opacity=0.7
            )
        fig.update_layout(
            xaxis_title=x_column,
            yaxis_title=y_column,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50')
        )
        return fig

    @app.callback(
        Output('bar-plot', 'figure'),
        [Input('bar-column', 'value'),
         Input('bar-top-n', 'value'),
         Input('region-filter', 'value'),
         Input('subregion-filter', 'value'),
         Input('ubicacion-filter', 'value')]
    )
    def update_bar(column, top_n, region, subregion, ubicacion):
        """
        Actualiza el gráfico de barras según la columna categórica y los filtros seleccionados.

        :param column: Columna categórica a graficar.
        :type column: str
        :param top_n: Número de categorías a mostrar.
        :type top_n: int
        :param region: Región seleccionada o None.
        :type region: str or None
        :param subregion: Subregión seleccionada o None.
        :type subregion: str or None
        :param ubicacion: Ubicación seleccionada o None.
        :type ubicacion: str or None
        :return: Figura de Plotly con el gráfico de barras.
        :rtype: plotly.graph_objs._figure.Figure
        """
        dff = filter_df(region, subregion, ubicacion)
        if column is None or column not in dff.columns:
            return go.Figure()
        value_counts = dff[dff[column].notnull() & (dff[column].astype(str).str.lower() != 'desconocido')][column].value_counts().reset_index()
        value_counts.columns = ['value', 'count']
        top_values = value_counts.head(top_n)
        filtro = ''
        if region:
            filtro += f' en {region}'
        if subregion:
            filtro += f', subregión {subregion}'
        if ubicacion:
            filtro += f', ubicación {ubicacion}'
        fig = px.bar(
            top_values, x='value', y='count',
            title=f'Top {top_n} valores más frecuentes de {column}{filtro}',
            color='count',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            xaxis_title=column.capitalize(),
            yaxis_title="Frecuencia",
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50')
        )
        return fig

    @app.callback(
        Output('map-plot', 'figure'),
        [Input('region-filter', 'value'),
         Input('subregion-filter', 'value'),
         Input('ubicacion-filter', 'value')]
    )
    def update_map(region, subregion, ubicacion):
        """
        Actualiza el mapa de distribución geográfica según los filtros seleccionados.

        :param region: Región seleccionada o None.
        :type region: str or None
        :param subregion: Subregión seleccionada o None.
        :type subregion: str or None
        :param ubicacion: Ubicación seleccionada o None.
        :type ubicacion: str or None
        :return: Figura de Plotly con el mapa.
        :rtype: plotly.graph_objs._figure.Figure
        """
        dff = filter_df(region, subregion, ubicacion)
        lat_col = 'lat'
        lon_col = 'lon'
        if lat_col not in dff.columns or lon_col not in dff.columns:
            fig = go.Figure()
            fig.update_layout(
                title="No se detectaron columnas geográficas",
                annotations=[
                    dict(
                        text="No se encontraron columnas 'lat' y 'lon' en los datos",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.5, y=0.5
                    )
                ]
            )
            return fig
        # Conversión explícita a numérico
        dff[lat_col] = pd.to_numeric(dff[lat_col], errors='coerce')
        dff[lon_col] = pd.to_numeric(dff[lon_col], errors='coerce')
        # Filtrar datos válidos
        map_data = dff.dropna(subset=[lat_col, lon_col])
        map_data = map_data[(map_data[lat_col].astype(str).str.lower() != 'desconocido') & (map_data[lon_col].astype(str).str.lower() != 'desconocido')]
        map_data = map_data[(map_data[lat_col] >= -90) & (map_data[lat_col] <= 90) & (map_data[lon_col] >= -180) & (map_data[lon_col] <= 180)]
        if map_data.empty:
            fig = go.Figure()
            fig.update_layout(
                title="No hay datos geográficos válidos para el filtro seleccionado",
                annotations=[
                    dict(
                        text="No hay datos válidos de latitud y longitud",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.5, y=0.5
                    )
                ]
            )
            return fig
        fig = px.scatter_mapbox(
            map_data, lat=lat_col, lon=lon_col,
            zoom=2, height=500,
            title="Distribución Geográfica de los Datos",
            color_discrete_sequence=['#3498db']
        )
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r":0,"t":50,"l":0,"b":0},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50')
        )
        return fig

    @app.callback(
        Output('stats-table', 'children'),
        [Input('histogram-column', 'value')]
    )
    def update_stats(column):
        """
        Genera la tabla de estadísticas descriptivas para las columnas numéricas del DataFrame.

        :param column: Columna seleccionada para el histograma (no afecta la tabla, es solo para trigger del callback).
        :type column: str
        :return: Tabla HTML con las estadísticas descriptivas.
        :rtype: dash.html.Table
        """
        # Mostrar estadísticas para todas las columnas numéricas
        numeric_df = df.select_dtypes(include=[np.number])
        stats = numeric_df.describe().transpose().reset_index()
        stats.columns = ['Variable', 'Conteo', 'Media', 'Desv. Estándar', 'Mínimo', '25%', '50%', '75%', 'Máximo']
        
        # Formatear las estadísticas para mostrarlas en una tabla HTML
        table_header = [html.Tr([html.Th(col) for col in stats.columns])]
        table_rows = []
        for i, row in stats.iterrows():
            formatted_row = []
            for j, val in enumerate(row):
                if j == 0:  # Nombre de la variable
                    formatted_row.append(html.Td(val))
                else:  # Valores numéricos
                    formatted_val = f"{val:.2f}" if isinstance(val, (int, float)) else val
                    formatted_row.append(html.Td(formatted_val))
            table_rows.append(html.Tr(formatted_row))
        
        return html.Table(
            table_header + table_rows,
            style={
                'width': '100%',
                'border': '1px solid #ddd',
                'borderCollapse': 'collapse',
                'textAlign': 'center'
            }
        )

# Ejecutar la aplicación
if __name__ == '__main__':
    print("Iniciando el dashboard en http://127.0.0.1:8050/")
    app.run_server(debug=True, port=8050)
    print("Dashboard iniciado. Acceda a http://127.0.0.1:8050/ en su navegador.")