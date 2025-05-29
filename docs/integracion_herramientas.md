# Guía de Integración de Herramientas

Este documento describe cómo integrar el sistema de análisis de datos con herramientas externas y otros sistemas.

## Integración con Herramientas de Análisis

### Exportación a Jupyter Notebook

Los datos procesados y los resultados de análisis pueden ser fácilmente exportados para su uso en Jupyter Notebooks:

```python
# En un Jupyter Notebook
import sys
import os

# Añadir la ruta del proyecto al path
proyecto_dir = "ruta/a/UN PASO AL EXITO"
sys.path.insert(0, proyecto_dir)

# Importar módulos del proyecto
from src.analysis.analisis_predictivo import load_and_prepare_data, train_models
from src.visualization.visualizar_datos import generar_visualizaciones

# Cargar datos
df = load_and_prepare_data(os.path.join(proyecto_dir, 'src/data/20250525.export.CSV'))

# Realizar análisis adicionales
# ...
```

### Integración con Power BI

Para visualizar los resultados en Power BI:

1. Exportar los datos procesados a CSV:
   ```python
   # En cualquier script del proyecto
   df_resultados.to_csv('resultados_para_powerbi.csv', index=False)
   ```

2. En Power BI:
   - Usar "Obtener datos" > "Texto/CSV"
   - Seleccionar el archivo CSV exportado
   - Configurar los tipos de datos según sea necesario

## Integración con Sistemas de Bases de Datos

### Exportación a SQL

Para almacenar los resultados en una base de datos SQL:

```python
import pandas as pd
import sqlalchemy

# Crear conexión a la base de datos
engine = sqlalchemy.create_engine('sqlite:///mi_base_datos.db')
# Para MySQL: 'mysql+pymysql://usuario:contraseña@localhost/nombre_db'
# Para PostgreSQL: 'postgresql://usuario:contraseña@localhost/nombre_db'

# Exportar DataFrame a tabla SQL
def exportar_a_sql(df, nombre_tabla):
    df.to_sql(nombre_tabla, engine, if_exists='replace', index=False)
    print(f"Datos exportados a tabla '{nombre_tabla}'")

# Ejemplo de uso
from src.analysis.analisis_predictivo import load_and_prepare_data
df = load_and_prepare_data('ruta/al/archivo.csv')
exportar_a_sql(df, 'datos_procesados')
```

### Importación desde SQL

Para utilizar datos desde una base de datos SQL:

```python
import pandas as pd
import sqlalchemy

# Crear conexión a la base de datos
engine = sqlalchemy.create_engine('sqlite:///mi_base_datos.db')

# Importar datos desde SQL
def importar_desde_sql(query, conexion):
    return pd.read_sql(query, conexion)

# Ejemplo de uso
df = importar_desde_sql("SELECT * FROM datos_procesados", engine)
print(f"Datos importados: {df.shape[0]} filas x {df.shape[1]} columnas")
```

## Integración con APIs Web

### Consumo de APIs Externas

Para enriquecer los datos con información de APIs externas:

```python
import requests
import pandas as pd

def obtener_datos_api(url, params=None):
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error al obtener datos: {response.status_code}")
        return None

# Ejemplo: Enriquecer datos con información geográfica
def enriquecer_con_datos_geograficos(df, columna_ciudad):
    df_resultado = df.copy()
    df_resultado['info_geografica'] = None
    
    for idx, ciudad in enumerate(df[columna_ciudad]):
        # Llamada a API de ejemplo (OpenStreetMap Nominatim)
        datos = obtener_datos_api(
            "https://nominatim.openstreetmap.org/search",
            {"q": ciudad, "format": "json"}
        )
        if datos and len(datos) > 0:
            df_resultado.loc[idx, 'latitud'] = datos[0]['lat']
            df_resultado.loc[idx, 'longitud'] = datos[0]['lon']
    
    return df_resultado
```

### Exposición de Resultados como API

Para exponer los resultados del análisis como una API REST:

```python
from flask import Flask, jsonify, request
import pandas as pd
import os

app = Flask(__name__)

# Cargar datos (en una aplicación real, esto podría estar en una base de datos)
@app.before_first_request
def cargar_datos():
    global df
    proyecto_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    df = pd.read_csv(os.path.join(proyecto_dir, 'src/data/datos_limpios.csv'))

# Endpoint para obtener estadísticas descriptivas
@app.route('/api/estadisticas', methods=['GET'])
def obtener_estadisticas():
    return jsonify({
        'descripcion': df.describe().to_dict(),
        'columnas': list(df.columns),
        'filas': len(df)
    })

# Endpoint para filtrar datos
@app.route('/api/datos', methods=['GET'])
def obtener_datos():
    # Obtener parámetros de consulta
    filtros = request.args.to_dict()
    
    # Aplicar filtros si existen
    datos_filtrados = df
    for columna, valor in filtros.items():
        if columna in df.columns:
            datos_filtrados = datos_filtrados[datos_filtrados[columna] == valor]
    
    # Convertir a lista de diccionarios para JSON
    return jsonify(datos_filtrados.head(100).to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
```

## Integración con Herramientas de Visualización

### Exportación a Tableau

Para visualizar los datos en Tableau:

1. Exportar los datos a CSV:
   ```python
   df_resultados.to_csv('datos_para_tableau.csv', index=False)
   ```

2. En Tableau:
   - Conectar a un archivo de texto
   - Seleccionar el archivo CSV exportado
   - Configurar los tipos de datos según sea necesario

### Integración con Grafana

Para monitorizar los resultados en tiempo real con Grafana:

1. Configurar una base de datos compatible (como InfluxDB o PostgreSQL)
2. Exportar periódicamente los resultados a la base de datos
3. Configurar Grafana para conectarse a la base de datos
4. Crear dashboards en Grafana para visualizar los datos

## Automatización e Integración Continua

### Programación de Tareas con Airflow

Para automatizar la ejecución de análisis con Apache Airflow:

```python
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

# Definir argumentos por defecto
default_args = {
    'owner': 'un_paso_al_exito',
    'depends_on_past': False,
    'start_date': datetime(2025, 5, 25),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Crear DAG
dag = DAG(
    'analisis_datos_diario',
    default_args=default_args,
    description='Ejecuta análisis de datos diariamente',
    schedule_interval=timedelta(days=1),
)

# Definir funciones para las tareas
def ejecutar_limpieza():
    import sys
    import os
    sys.path.insert(0, '/ruta/a/UN PASO AL EXITO')
    from src.analysis.limpiar_analizar_csv import limpiar_datos
    limpiar_datos('/ruta/a/datos/20250525.export.CSV')

def ejecutar_analisis_predictivo():
    import sys
    import os
    sys.path.insert(0, '/ruta/a/UN PASO AL EXITO')
    from src.analysis.analisis_predictivo import ejecutar_analisis
    ejecutar_analisis()

# Crear tareas
tarea_limpieza = PythonOperator(
    task_id='limpiar_datos',
    python_callable=ejecutar_limpieza,
    dag=dag,
)

tarea_analisis = PythonOperator(
    task_id='analisis_predictivo',
    python_callable=ejecutar_analisis_predictivo,
    dag=dag,
)

# Definir dependencias
tarea_limpieza >> tarea_analisis
```

## Integración con Sistemas de Notificación

### Envío de Alertas por Email

Para enviar notificaciones por email cuando se completen los análisis:

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def enviar_email(destinatario, asunto, cuerpo, remitente='analisis@unpaso.com', servidor='smtp.servidor.com'):
    # Crear mensaje
    mensaje = MIMEMultipart()
    mensaje['From'] = remitente
    mensaje['To'] = destinatario
    mensaje['Subject'] = asunto
    
    # Añadir cuerpo del mensaje
    mensaje.attach(MIMEText(cuerpo, 'html'))
    
    # Conectar al servidor SMTP y enviar
    try:
        servidor_smtp = smtplib.SMTP(servidor, 587)
        servidor_smtp.starttls()
        servidor_smtp.login(remitente, 'contraseña')
        texto = mensaje.as_string()
        servidor_smtp.sendmail(remitente, destinatario, texto)
        servidor_smtp.quit()
        print(f"Email enviado a {destinatario}")
        return True
    except Exception as e:
        print(f"Error al enviar email: {e}")
        return False

# Ejemplo de uso
def notificar_analisis_completado(resultados):
    cuerpo_html = f"""
    <html>
    <body>
        <h2>Análisis Completado</h2>
        <p>El análisis se ha completado con éxito.</p>
        <h3>Resultados:</h3>
        <ul>
            <li>Precisión del modelo: {resultados['precision']:.2f}</li>
            <li>Tiempo de ejecución: {resultados['tiempo']} segundos</li>
        </ul>
        <p>Puede ver el informe completo <a href="http://servidor/informe.html">aquí</a>.</p>
    </body>
    </html>
    """
    
    enviar_email(
        'analista@empresa.com',
        'Análisis de Datos Completado',
        cuerpo_html
    )
```

## Consideraciones de Seguridad

Al integrar el sistema con herramientas externas, tenga en cuenta las siguientes consideraciones de seguridad:

1. **Credenciales**: Nunca incluya credenciales directamente en el código. Utilice variables de entorno o sistemas de gestión de secretos.

2. **Validación de Datos**: Valide siempre los datos recibidos de fuentes externas antes de procesarlos.

3. **Control de Acceso**: Implemente autenticación y autorización adecuadas para las APIs expuestas.

4. **Cifrado**: Utilice HTTPS/TLS para todas las comunicaciones en red.

5. **Auditoría**: Mantenga registros de todas las integraciones y accesos para fines de auditoría.

## Resolución de Problemas Comunes

### Problemas de Conexión a Bases de Datos

- Verifique que la base de datos esté en ejecución y sea accesible desde la máquina donde se ejecuta el script.
- Compruebe que las credenciales sean correctas.
- Asegúrese de que el usuario tenga los permisos necesarios.

### Errores en Llamadas a APIs

- Verifique la URL y los parámetros de la solicitud.
- Compruebe si la API requiere autenticación.
- Verifique los límites de tasa de la API.
- Implemente reintentos con retroceso exponencial para manejar fallos temporales.

### Problemas de Formato de Datos

- Asegúrese de que los datos exportados tengan el formato esperado por la herramienta de destino.
- Verifique la codificación de caracteres (UTF-8 es generalmente la mejor opción).
- Para archivos CSV, preste atención a los delimitadores y caracteres de escape.

---

Esta guía proporciona una base para integrar el sistema de análisis de datos con diversas herramientas y sistemas externos. Adapte las soluciones según sus necesidades específicas y las herramientas particulares que utilice en su entorno.