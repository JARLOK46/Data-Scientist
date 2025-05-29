# Casos de Uso

Este documento presenta escenarios prácticos que ilustran cómo utilizar el sistema de análisis de datos para resolver problemas específicos.

## Caso 1: Análisis Exploratorio de Datos para Identificar Patrones

### Escenario

Un analista necesita comprender rápidamente la estructura y patrones principales en un conjunto de datos CSV recién recibido.

### Solución

1. **Ejecutar el análisis exploratorio básico:**

```bash
python src/utils/ejecutar_analisis.py
```

2. Seleccionar la opción 1: "Análisis Exploratorio y Visualizaciones"

3. **Resultados obtenidos:**
   - Estadísticas descriptivas de todas las variables
   - Identificación de valores faltantes
   - Distribuciones de variables numéricas
   - Correlaciones entre variables
   - Visualizaciones automáticas guardadas en el directorio `visualizaciones/`

### Código de ejemplo para uso programático

```python
from src.analysis.limpiar_analizar_csv import analizar_csv
from src.visualization.visualizar_datos import visualizar_datos

# Ejecutar análisis exploratorio
df = analizar_csv('src/data/20250525.export.CSV')

# Generar visualizaciones
visualizar_datos()

# Análisis personalizado adicional
print("\nVariables con más valores faltantes:")
print(df.isnull().sum().sort_values(ascending=False).head())

print("\nEstadísticas de las variables numéricas:")
print(df.describe())
```

## Caso 2: Creación de un Dashboard para Presentación a Directivos

### Escenario

Un gerente de proyecto necesita presentar los resultados del análisis de datos a directivos que no tienen conocimientos técnicos.

### Solución

#### Opción 1: Dashboard Estático (sin dependencias adicionales)

1. **Ejecutar el dashboard adaptativo:**

```bash
python src/dashboard/dashboard_adaptativo.py
```

2. **Resultados obtenidos:**
   - Informe HTML generado en `dashboard_estatico/informe_analisis.html`
   - Visualizaciones estáticas en formato PNG
   - Tablas de resumen con estadísticas clave

3. **Presentación:**
   - Compartir el archivo HTML con los directivos
   - Incorporar las imágenes PNG en presentaciones de PowerPoint

#### Opción 2: Dashboard Interactivo (requiere dependencias adicionales)

1. **Instalar dependencias y ejecutar el dashboard interactivo:**

```bash
python src/dashboard/setup_dashboard.py
python src/dashboard/dashboard_interactivo.py
```

2. **Resultados obtenidos:**
   - Dashboard interactivo accesible en http://127.0.0.1:8050/
   - Filtros dinámicos para explorar los datos
   - Gráficos interactivos que responden a selecciones del usuario

3. **Presentación:**
   - Realizar una demostración en vivo durante la reunión
   - Permitir a los directivos explorar los datos según sus intereses

### Código de ejemplo para personalizar el dashboard

```python
# Modificar el archivo dashboard_interactivo.py para añadir nuevas visualizaciones

# Añadir un nuevo gráfico al layout
nuevo_grafico = dcc.Graph(
    id='grafico-personalizado',
    figure=px.scatter(df, x='variable_x', y='variable_y', 
                     color='categoria', size='valor',
                     hover_name='identificador',
                     title='Gráfico Personalizado para Directivos')
)

# Añadir el gráfico al layout
app.layout = html.Div([
    # ... componentes existentes ...
    html.Div([
        html.H3("Análisis Personalizado"),
        nuevo_grafico
    ], className='row')
])
```

## Caso 3: Análisis Geoespacial para Identificar Clusters Regionales

### Escenario

Un investigador necesita identificar patrones geográficos y clusters en los datos para entender la distribución regional.

### Solución

1. **Ejecutar el análisis geoespacial adaptativo (sin dependencias especiales):**

```bash
python src/geospatial/analisis_geoespacial_adaptativo.py
```

2. **Alternativamente, para análisis más avanzado (requiere dependencias adicionales):**

```bash
pip install -r config/requirements_geoespacial.txt
python src/geospatial/analisis_geoespacial.py
```

3. **Resultados obtenidos:**
   - Mapas de distribución geográfica
   - Identificación de clusters mediante K-means
   - Estadísticas por región
   - Visualizaciones guardadas en `geospatial/output/`

### Código de ejemplo para análisis personalizado

```python
from src.geospatial.analisis_geoespacial import cargar_datos, analizar_clusters
import folium

# Cargar datos con coordenadas
df = cargar_datos('src/data/20250525.export.CSV')

# Realizar clustering personalizado
df_clusters = analizar_clusters(df, n_clusters=5, variables=['latitud', 'longitud', 'variable_interes'])

# Crear mapa personalizado
mapa = folium.Map(location=[df['latitud'].mean(), df['longitud'].mean()], zoom_start=4)

# Añadir marcadores por cluster
for cluster in range(5):
    datos_cluster = df_clusters[df_clusters['cluster'] == cluster]
    for _, row in datos_cluster.iterrows():
        folium.CircleMarker(
            location=[row['latitud'], row['longitud']],
            radius=5,
            color=['red', 'blue', 'green', 'purple', 'orange'][cluster],
            fill=True,
            popup=f"Cluster: {cluster}<br>Valor: {row['variable_interes']}"
        ).add_to(mapa)

# Guardar mapa personalizado
mapa.save('mapa_clusters_personalizado.html')
```

## Caso 4: Análisis Predictivo para Estimar Valores Futuros

### Escenario

Un analista de datos necesita desarrollar un modelo predictivo para estimar valores futuros basados en datos históricos.

### Solución

1. **Ejecutar el análisis predictivo:**

```bash
python src/analysis/analisis_predictivo.py
```

2. **Resultados obtenidos:**
   - Modelos de regresión entrenados (Random Forest, Gradient Boosting, etc.)
   - Evaluación de rendimiento de los modelos
   - Importancia de características
   - Modelos guardados para uso futuro

### Código de ejemplo para predicciones personalizadas

```python
from src.analysis.analisis_predictivo import load_and_prepare_data, train_models, evaluate_models
import joblib
import pandas as pd

# Cargar y preparar datos
df = load_and_prepare_data('src/data/20250525.export.CSV')

# Definir variables predictoras y objetivo
X = df[['variable1', 'variable2', 'variable3']]
y = df['variable_objetivo']

# Entrenar modelos personalizados
modelos = train_models(X, y, modelos=['random_forest', 'gradient_boosting'])

# Evaluar modelos
resultados = evaluate_models(modelos, X, y)
print("Resultados de evaluación:")
for nombre, metricas in resultados.items():
    print(f"{nombre}: R² = {metricas['r2']:.4f}, RMSE = {metricas['rmse']:.4f}")

# Guardar el mejor modelo
mejor_modelo = modelos['gradient_boosting']
joblib.dump(mejor_modelo, 'mejor_modelo_predictivo.pkl')

# Usar el modelo para nuevas predicciones
df_nuevos = pd.DataFrame({
    'variable1': [valor1, valor2],
    'variable2': [valor3, valor4],
    'variable3': [valor5, valor6]
})

predicciones = mejor_modelo.predict(df_nuevos)
print("Predicciones para nuevos datos:")
print(predicciones)
```

## Caso 5: Análisis de Contenido Web para Identificar Dominios Relevantes

### Escenario

Un especialista en marketing digital necesita analizar las URLs presentes en los datos para identificar dominios relevantes y patrones de contenido.

### Solución

1. **Ejecutar el análisis de contenido web:**

```bash
python src/web/analisis_contenido_web.py
```

2. **Resultados obtenidos:**
   - Distribución de dominios
   - Análisis de palabras clave en URLs
   - Clasificación de contenido
   - Visualizaciones de nubes de palabras

### Código de ejemplo para análisis personalizado

```python
from src.web.analisis_contenido_web import extraer_dominios, analizar_contenido
from urllib.parse import urlparse
import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos con URLs
df = pd.read_csv('src/data/datos_limpios.csv')

# Extraer y analizar dominios
df['dominio'] = df['url'].apply(lambda x: urlparse(x).netloc if isinstance(x, str) else '')

# Contar dominios más frecuentes
top_dominios = df['dominio'].value_counts().head(15)

# Visualizar dominios más frecuentes
plt.figure(figsize=(12, 6))
top_dominios.plot(kind='bar')
plt.title('Dominios más frecuentes')
plt.xlabel('Dominio')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('top_dominios_personalizado.png')

# Análisis de palabras clave en URLs
palabras_clave = ['producto', 'servicio', 'oferta', 'descuento', 'promocion']
for palabra in palabras_clave:
    df[f'contiene_{palabra}'] = df['url'].str.contains(palabra, case=False)

# Resumen de URLs que contienen palabras clave
resumen_palabras = {palabra: df[f'contiene_{palabra}'].sum() for palabra in palabras_clave}
print("URLs que contienen palabras clave de marketing:")
for palabra, cantidad in resumen_palabras.items():
    print(f"{palabra}: {cantidad} URLs")
```

## Caso 6: Automatización de Análisis Periódicos

### Escenario

Un equipo necesita ejecutar análisis de forma periódica (diaria/semanal) sobre nuevos datos que se reciben regularmente.

### Solución

1. **Crear un script de automatización:**

Guardar el siguiente código como `automatizar_analisis.py`:

```python
import os
import sys
import schedule
import time
from datetime import datetime

# Añadir el directorio raíz del proyecto al path
proyecto_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, proyecto_dir)

def ejecutar_analisis_completo():
    print(f"\n[{datetime.now()}] Iniciando análisis automático...")
    
    # Definir rutas
    archivo_datos = os.path.join(proyecto_dir, 'src/data/20250525.export.CSV')
    
    # Importar funciones necesarias
    from src.analysis.limpiar_analizar_csv import limpiar_datos, analizar_csv
    from src.analysis.analisis_predictivo import ejecutar_analisis as ejecutar_predictivo
    from src.visualization.visualizar_datos import visualizar_datos
    from src.dashboard.dashboard_adaptativo import generar_dashboard
    
    # Ejecutar pipeline de análisis
    print("1. Limpieza y análisis exploratorio...")
    df = limpiar_datos(archivo_datos)
    analizar_csv(archivo_datos)
    
    print("2. Generando visualizaciones...")
    visualizar_datos()
    
    print("3. Ejecutando análisis predictivo...")
    ejecutar_predictivo()
    
    print("4. Generando dashboard estático...")
    generar_dashboard()
    
    print(f"[{datetime.now()}] Análisis completado. Resultados guardados en los directorios correspondientes.")
    
    # Opcional: Enviar notificación por email
    # enviar_notificacion("Análisis completado", "El análisis automático ha finalizado correctamente.")

# Programar ejecución diaria a las 2:00 AM
schedule.every().day.at("02:00").do(ejecutar_analisis_completo)

# Alternativamente, para ejecución semanal los lunes
# schedule.every().monday.at("02:00").do(ejecutar_analisis_completo)

print("Servicio de análisis automático iniciado. Presione Ctrl+C para detener.")

# Ejecutar inmediatamente la primera vez
ejecutar_analisis_completo()

# Mantener el script en ejecución
while True:
    schedule.run_pending()
    time.sleep(60)  # Verificar cada minuto
```

2. **Ejecutar el script de automatización:**

```bash
pip install schedule
python automatizar_analisis.py
```

3. **Para entornos de producción:**
   - Configurar como servicio del sistema o tarea programada
   - Implementar registro de eventos y manejo de errores robusto
   - Añadir notificaciones por email o mensajería

## Caso 7: Integración con Sistemas Existentes

### Escenario

Una empresa necesita integrar el sistema de análisis con su infraestructura existente (base de datos, sistema de BI, etc.).

### Solución

1. **Exportación a base de datos:**

```python
import pandas as pd
import sqlalchemy
import os

# Configurar conexión a la base de datos
db_user = os.environ.get('DB_USER', 'usuario')
db_pass = os.environ.get('DB_PASS', 'contraseña')
db_host = os.environ.get('DB_HOST', 'localhost')
db_name = os.environ.get('DB_NAME', 'analisis_datos')

# Crear string de conexión (ejemplo para PostgreSQL)
connection_string = f"postgresql://{db_user}:{db_pass}@{db_host}/{db_name}"
engine = sqlalchemy.create_engine(connection_string)

# Cargar resultados del análisis
from src.analysis.analisis_predictivo import load_and_prepare_data
df = load_and_prepare_data('src/data/20250525.export.CSV')

# Exportar a la base de datos
def exportar_resultados():
    # Tabla principal de datos
    df.to_sql('datos_analizados', engine, if_exists='replace', index=False)
    
    # Tabla de estadísticas
    estadisticas = pd.DataFrame({
        'variable': df.columns,
        'media': df.mean(),
        'mediana': df.median(),
        'desviacion': df.std(),
        'minimo': df.min(),
        'maximo': df.max()
    })
    estadisticas.to_sql('estadisticas', engine, if_exists='replace', index=False)
    
    print("Datos exportados correctamente a la base de datos.")

# Ejecutar exportación
exportar_resultados()
```

2. **Integración con API REST:**

```python
import requests
import json
import pandas as pd

# Cargar resultados del análisis
from src.analysis.analisis_predictivo import load_and_prepare_data
df = load_and_prepare_data('src/data/20250525.export.CSV')

# Configurar endpoint de la API
api_url = "https://api.empresa.com/datos/actualizar"
api_key = "su_clave_api"

# Preparar datos para envío
datos_json = df.to_json(orient='records')

# Enviar datos a la API
response = requests.post(
    api_url,
    headers={
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    },
    data=json.dumps({
        'datos': json.loads(datos_json),
        'fecha_analisis': pd.Timestamp.now().isoformat(),
        'version': '1.0'
    })
)

# Verificar respuesta
if response.status_code == 200:
    print("Datos enviados correctamente a la API.")
    print(f"Respuesta: {response.json()}")
else:
    print(f"Error al enviar datos: {response.status_code}")
    print(f"Detalle: {response.text}")
```

---

Estos casos de uso ilustran diferentes escenarios prácticos donde el sistema de análisis de datos puede aplicarse para resolver problemas específicos. Cada caso incluye instrucciones paso a paso y ejemplos de código que pueden adaptarse a necesidades particulares.