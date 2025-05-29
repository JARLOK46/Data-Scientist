# Referencia Técnica

Este documento proporciona una referencia detallada de las APIs, funciones y clases principales del proyecto. Está diseñado como un recurso técnico para desarrolladores que necesiten utilizar o extender la funcionalidad existente.

## Módulo de Análisis

### `src.analysis.limpiar_analizar_csv`

#### Funciones Principales

##### `detect_delimiter(file_path)`

Detecta automáticamente el delimitador utilizado en un archivo CSV.

**Parámetros:**
- `file_path` (str): Ruta al archivo CSV.

**Retorna:**
- str: Delimitador detectado (',' o ';').

**Ejemplo:**
```python
from src.analysis.limpiar_analizar_csv import detect_delimiter

delimiter = detect_delimiter('datos/mi_archivo.csv')
print(f"El delimitador detectado es: {delimiter}")
```

##### `load_and_prepare_data(file_path, delimiter=None)`

Carga un archivo CSV y realiza preparación básica de los datos.

**Parámetros:**
- `file_path` (str): Ruta al archivo CSV.
- `delimiter` (str, opcional): Delimitador a utilizar. Si es None, se detecta automáticamente.

**Retorna:**
- pandas.DataFrame: DataFrame con los datos cargados y preparados.

**Ejemplo:**
```python
from src.analysis.limpiar_analizar_csv import load_and_prepare_data

df = load_and_prepare_data('datos/mi_archivo.csv')
```

##### `analyze_csv(file_path, delimiter=None, output_dir=None)`

Realiza un análisis completo de un archivo CSV, generando visualizaciones y estadísticas.

**Parámetros:**
- `file_path` (str): Ruta al archivo CSV.
- `delimiter` (str, opcional): Delimitador a utilizar. Si es None, se detecta automáticamente.
- `output_dir` (str, opcional): Directorio donde guardar los resultados. Si es None, se crea un directorio basado en el nombre del archivo.

**Retorna:**
- dict: Diccionario con resultados del análisis y rutas a los archivos generados.

**Ejemplo:**
```python
from src.analysis.limpiar_analizar_csv import analyze_csv

resultados = analyze_csv('datos/mi_archivo.csv', output_dir='resultados/analisis1')
```

### `src.analysis.analisis_predictivo`

#### Funciones Principales

##### `train_test_model(df, target_column, features=None, test_size=0.2, random_state=42)`

Entrena un modelo predictivo utilizando los datos proporcionados.

**Parámetros:**
- `df` (pandas.DataFrame): DataFrame con los datos.
- `target_column` (str): Nombre de la columna objetivo.
- `features` (list, opcional): Lista de columnas a utilizar como características. Si es None, se utilizan todas las columnas numéricas excepto la objetivo.
- `test_size` (float, opcional): Proporción de datos a utilizar para pruebas.
- `random_state` (int, opcional): Semilla para reproducibilidad.

**Retorna:**
- tuple: (modelo entrenado, precisión en conjunto de prueba, características utilizadas)

**Ejemplo:**
```python
from src.analysis.analisis_predictivo import train_test_model

modelo, precision, features = train_test_model(df, 'target_column')
print(f"Precisión del modelo: {precision:.4f}")
```

##### `predict_with_model(model, data, features)`

Realiza predicciones utilizando un modelo entrenado.

**Parámetros:**
- `model`: Modelo entrenado.
- `data` (pandas.DataFrame): Datos para los que realizar predicciones.
- `features` (list): Lista de columnas a utilizar como características.

**Retorna:**
- numpy.ndarray: Array con las predicciones.

**Ejemplo:**
```python
from src.analysis.analisis_predictivo import predict_with_model

predicciones = predict_with_model(modelo, nuevos_datos, features)
```

##### `run_predictive_analysis(file_path, target_column, delimiter=None, output_dir=None)`

Realiza un análisis predictivo completo, incluyendo entrenamiento, evaluación y visualizaciones.

**Parámetros:**
- `file_path` (str): Ruta al archivo CSV.
- `target_column` (str): Nombre de la columna objetivo.
- `delimiter` (str, opcional): Delimitador a utilizar. Si es None, se detecta automáticamente.
- `output_dir` (str, opcional): Directorio donde guardar los resultados.

**Retorna:**
- dict: Diccionario con resultados del análisis y rutas a los archivos generados.

**Ejemplo:**
```python
from src.analysis.analisis_predictivo import run_predictive_analysis

resultados = run_predictive_analysis('datos/mi_archivo.csv', 'target_column')
```

## Módulo Geoespacial

### `src.geospatial.analisis_geoespacial`

#### Funciones Principales

##### `create_map(df, lat_column, lon_column, popup_columns=None, cluster=True)`

Crea un mapa interactivo con los puntos especificados.

**Parámetros:**
- `df` (pandas.DataFrame): DataFrame con los datos.
- `lat_column` (str): Nombre de la columna con latitudes.
- `lon_column` (str): Nombre de la columna con longitudes.
- `popup_columns` (list, opcional): Columnas a mostrar en el popup al hacer clic en un punto.
- `cluster` (bool, opcional): Si se deben agrupar los puntos cercanos.

**Retorna:**
- folium.Map: Mapa interactivo.

**Ejemplo:**
```python
from src.geospatial.analisis_geoespacial import create_map

mapa = create_map(df, 'latitud', 'longitud', popup_columns=['nombre', 'valor'])
mapa.save('mapa.html')
```

##### `spatial_clustering(df, lat_column, lon_column, n_clusters=5)`

Realiza clustering espacial de los puntos.

**Parámetros:**
- `df` (pandas.DataFrame): DataFrame con los datos.
- `lat_column` (str): Nombre de la columna con latitudes.
- `lon_column` (str): Nombre de la columna con longitudes.
- `n_clusters` (int, opcional): Número de clusters a crear.

**Retorna:**
- pandas.DataFrame: DataFrame original con una columna adicional 'cluster' indicando el cluster asignado.

**Ejemplo:**
```python
from src.geospatial.analisis_geoespacial import spatial_clustering

df_clustered = spatial_clustering(df, 'latitud', 'longitud', n_clusters=8)
```

##### `run_geospatial_analysis(file_path, lat_column, lon_column, delimiter=None, output_dir=None)`

Realiza un análisis geoespacial completo, incluyendo mapas y clustering.

**Parámetros:**
- `file_path` (str): Ruta al archivo CSV.
- `lat_column` (str): Nombre de la columna con latitudes.
- `lon_column` (str): Nombre de la columna con longitudes.
- `delimiter` (str, opcional): Delimitador a utilizar. Si es None, se detecta automáticamente.
- `output_dir` (str, opcional): Directorio donde guardar los resultados.

**Retorna:**
- dict: Diccionario con resultados del análisis y rutas a los archivos generados.

**Ejemplo:**
```python
from src.geospatial.analisis_geoespacial import run_geospatial_analysis

resultados = run_geospatial_analysis('datos/ubicaciones.csv', 'lat', 'lon')
```

## Módulo Web

### `src.web.analisis_contenido_web`

#### Funciones Principales

##### `extract_domains(df, url_column)`

Extrae dominios de URLs en una columna específica.

**Parámetros:**
- `df` (pandas.DataFrame): DataFrame con los datos.
- `url_column` (str): Nombre de la columna con URLs.

**Retorna:**
- pandas.DataFrame: DataFrame con una columna adicional 'domain' con los dominios extraídos.

**Ejemplo:**
```python
from src.web.analisis_contenido_web import extract_domains

df_with_domains = extract_domains(df, 'url')
```

##### `analyze_url_content(url, timeout=10)`

Analiza el contenido de una URL, extrayendo texto y metadatos.

**Parámetros:**
- `url` (str): URL a analizar.
- `timeout` (int, opcional): Tiempo máximo de espera para la respuesta en segundos.

**Retorna:**
- dict: Diccionario con título, descripción, texto principal y metadatos.

**Ejemplo:**
```python
from src.web.analisis_contenido_web import analyze_url_content

content = analyze_url_content('https://ejemplo.com')
print(f"Título: {content['title']}")
```

##### `run_web_content_analysis(file_path, url_column, sample_size=100, delimiter=None, output_dir=None)`

Realiza un análisis completo de contenido web para URLs en un archivo CSV.

**Parámetros:**
- `file_path` (str): Ruta al archivo CSV.
- `url_column` (str): Nombre de la columna con URLs.
- `sample_size` (int, opcional): Número de URLs a analizar (para conjuntos grandes).
- `delimiter` (str, opcional): Delimitador a utilizar. Si es None, se detecta automáticamente.
- `output_dir` (str, opcional): Directorio donde guardar los resultados.

**Retorna:**
- dict: Diccionario con resultados del análisis y rutas a los archivos generados.

**Ejemplo:**
```python
from src.web.analisis_contenido_web import run_web_content_analysis

resultados = run_web_content_analysis('datos/urls.csv', 'url_column')
```

## Módulo de Dashboard

### `src.dashboard.dashboard_interactivo`

#### Funciones Principales

##### `create_dashboard(df, title="Dashboard Interactivo")`

Crea una aplicación Dash con un dashboard interactivo.

**Parámetros:**
- `df` (pandas.DataFrame): DataFrame con los datos.
- `title` (str, opcional): Título del dashboard.

**Retorna:**
- dash.Dash: Aplicación Dash configurada.

**Ejemplo:**
```python
from src.dashboard.dashboard_interactivo import create_dashboard

app = create_dashboard(df, title="Mi Dashboard")
app.run_server(debug=True)
```

##### `run_dashboard(file_path, delimiter=None, port=8050)`

Carga datos de un archivo CSV y ejecuta un dashboard interactivo.

**Parámetros:**
- `file_path` (str): Ruta al archivo CSV.
- `delimiter` (str, opcional): Delimitador a utilizar. Si es None, se detecta automáticamente.
- `port` (int, opcional): Puerto en el que ejecutar el servidor.

**Retorna:**
- None: La función inicia el servidor y bloquea hasta que se cierre.

**Ejemplo:**
```python
from src.dashboard.dashboard_interactivo import run_dashboard

run_dashboard('datos/mi_archivo.csv', port=8051)
```

### `src.dashboard.dashboard_adaptativo`

#### Funciones Principales

##### `generate_static_dashboard(df, output_file)`

Genera un dashboard estático en formato HTML.

**Parámetros:**
- `df` (pandas.DataFrame): DataFrame con los datos.
- `output_file` (str): Ruta donde guardar el archivo HTML.

**Retorna:**
- str: Ruta al archivo HTML generado.

**Ejemplo:**
```python
from src.dashboard.dashboard_adaptativo import generate_static_dashboard

html_path = generate_static_dashboard(df, 'dashboard.html')
```

## Módulo de Visualización

### `src.visualization.visualizar_datos`

#### Funciones Principales

##### `plot_numeric_distribution(df, column, bins=30, figsize=(10, 6))`

Crea un histograma y un boxplot para una columna numérica.

**Parámetros:**
- `df` (pandas.DataFrame): DataFrame con los datos.
- `column` (str): Nombre de la columna a visualizar.
- `bins` (int, opcional): Número de bins para el histograma.
- `figsize` (tuple, opcional): Tamaño de la figura.

**Retorna:**
- matplotlib.figure.Figure: Figura con las visualizaciones.

**Ejemplo:**
```python
from src.visualization.visualizar_datos import plot_numeric_distribution

fig = plot_numeric_distribution(df, 'edad', bins=20)
fig.savefig('distribucion_edad.png')
```

##### `plot_correlation_matrix(df, figsize=(12, 10))`

Crea un mapa de calor de la matriz de correlación.

**Parámetros:**
- `df` (pandas.DataFrame): DataFrame con los datos.
- `figsize` (tuple, opcional): Tamaño de la figura.

**Retorna:**
- matplotlib.figure.Figure: Figura con el mapa de calor.

**Ejemplo:**
```python
from src.visualization.visualizar_datos import plot_correlation_matrix

fig = plot_correlation_matrix(df)
fig.savefig('correlaciones.png')
```

##### `create_visualization_report(df, output_dir)`

Genera un conjunto completo de visualizaciones para todas las columnas.

**Parámetros:**
- `df` (pandas.DataFrame): DataFrame con los datos.
- `output_dir` (str): Directorio donde guardar las visualizaciones.

**Retorna:**
- dict: Diccionario con rutas a las visualizaciones generadas.

**Ejemplo:**
```python
from src.visualization.visualizar_datos import create_visualization_report

visualizaciones = create_visualization_report(df, 'resultados/visualizaciones')
```

## Módulo de Utilidades

### `src.utils.ejecutar_analisis`

#### Funciones Principales

##### `run_all_analyses(file_path, output_dir=None, delimiter=None)`

Ejecuta todos los tipos de análisis disponibles en el proyecto.

**Parámetros:**
- `file_path` (str): Ruta al archivo CSV.
- `output_dir` (str, opcional): Directorio base donde guardar todos los resultados.
- `delimiter` (str, opcional): Delimitador a utilizar. Si es None, se detecta automáticamente.

**Retorna:**
- dict: Diccionario con resultados de todos los análisis y rutas a los archivos generados.

**Ejemplo:**
```python
from src.utils.ejecutar_analisis import run_all_analyses

resultados = run_all_analyses('datos/mi_archivo.csv', output_dir='resultados/completo')
```

## Extensión del Proyecto

### Añadir un Nuevo Tipo de Análisis

Para añadir un nuevo tipo de análisis al proyecto, sigue estos pasos:

1. **Crea un nuevo módulo** en el directorio apropiado (por ejemplo, `src/new_analysis/`).

2. **Implementa las funciones principales** siguiendo el patrón establecido:
   - Funciones de utilidad específicas para el análisis
   - Una función principal `run_*_analysis()` que integre todo el flujo

3. **Integra con el módulo de utilidades** añadiendo tu análisis a `run_all_analyses()` en `src.utils.ejecutar_analisis`.

4. **Actualiza la documentación** para incluir tu nuevo módulo.

### Ejemplo de Integración

```python
# En src/new_analysis/mi_nuevo_analisis.py

def analyze_specific_aspect(df, column):
    # Implementación específica
    pass

def run_new_analysis(file_path, specific_column, delimiter=None, output_dir=None):
    # Implementación del flujo completo
    pass

# En src/utils/ejecutar_analisis.py

from src.new_analysis.mi_nuevo_analisis import run_new_analysis

def run_all_analyses(file_path, output_dir=None, delimiter=None):
    # Código existente...
    
    # Añadir el nuevo análisis
    new_results = run_new_analysis(file_path, 'columna_especifica', 
                                  delimiter=delimiter,
                                  output_dir=os.path.join(output_dir, 'nuevo_analisis'))
    results['nuevo_analisis'] = new_results
    
    return results
```

## Consideraciones de Rendimiento

### Manejo de Conjuntos de Datos Grandes

Para conjuntos de datos que no caben en memoria:

1. **Utiliza procesamiento por lotes**:
   ```python
   for chunk in pd.read_csv(file_path, chunksize=10000):
       # Procesar cada fragmento
       process_chunk(chunk)
   ```

2. **Considera utilizar Dask** para procesamiento paralelo:
   ```python
   import dask.dataframe as dd
   
   ddf = dd.read_csv(file_path)
   result = ddf.map_partitions(process_function).compute()
   ```

3. **Optimiza tipos de datos** para reducir el uso de memoria:
   ```python
   df = pd.read_csv(file_path, dtype={'id': 'int32', 'value': 'float32'})
   ```

## Solución de Problemas

### Errores Comunes

1. **Problemas de codificación de archivos CSV**:
   - Utiliza `encoding='latin1'` o `encoding='utf-8-sig'` si encuentras errores de codificación.

2. **Memoria insuficiente**:
   - Utiliza procesamiento por lotes como se describió anteriormente.
   - Considera utilizar versiones adaptativas de los módulos que requieren menos recursos.

3. **Dependencias faltantes**:
   - Asegúrate de instalar todas las dependencias listadas en `requirements.txt`.
   - Para funcionalidades geoespaciales, puede ser necesario instalar dependencias adicionales.

### Depuración

Para facilitar la depuración, puedes utilizar el modo de depuración en las funciones principales:

```python
from src.analysis.limpiar_analizar_csv import analyze_csv

resultados = analyze_csv('datos/mi_archivo.csv', debug=True)
```

Esto proporcionará información adicional durante la ejecución.

## Contribuciones al Proyecto

Si deseas contribuir al proyecto:

1. **Sigue las convenciones de estilo** descritas en la guía de estilo.

2. **Añade pruebas unitarias** para cualquier nueva funcionalidad.

3. **Documenta tu código** utilizando docstrings en formato Google o NumPy.

4. **Actualiza la documentación** para reflejar los cambios realizados.

5. **Envía un pull request** con una descripción clara de los cambios y su propósito.

Esta referencia técnica proporciona la información necesaria para utilizar y extender las funcionalidades del proyecto. Para más detalles sobre cada módulo, consulta la documentación generada por Sphinx.