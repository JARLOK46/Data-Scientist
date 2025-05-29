# Preguntas Frecuentes (FAQ)

Este documento responde a las preguntas más comunes sobre el uso del proyecto de Análisis de Datos y Visualización. Si no encuentras respuesta a tu pregunta aquí, considera revisar la documentación técnica o crear un issue en el repositorio.

## Índice

1. [Preguntas Generales](#preguntas-generales)
2. [Instalación y Configuración](#instalación-y-configuración)
3. [Carga y Procesamiento de Datos](#carga-y-procesamiento-de-datos)
4. [Análisis de Datos](#análisis-de-datos)
5. [Visualización y Dashboards](#visualización-y-dashboards)
6. [Análisis Geoespacial](#análisis-geoespacial)
7. [Análisis Web](#análisis-web)
8. [Solución de Problemas](#solución-de-problemas)

## Preguntas Generales

### ¿Qué es este proyecto y para qué sirve?

Este proyecto es una suite de herramientas para análisis de datos y visualización, diseñada para facilitar el procesamiento, análisis y presentación de datos en diversos formatos. Incluye capacidades para análisis exploratorio, predictivo, geoespacial, análisis de contenido web y creación de dashboards interactivos.

### ¿Qué tipos de datos puedo analizar con esta herramienta?

La herramienta está diseñada principalmente para trabajar con datos tabulares en formato CSV, pero puede adaptarse para otros formatos. Es especialmente útil para datos que incluyen:
- Información numérica y categórica para análisis estadístico
- Coordenadas geográficas para análisis espacial
- URLs y contenido web para análisis de dominios y texto
- Series temporales para análisis de tendencias

### ¿Necesito conocimientos avanzados de programación para usar este proyecto?

No necesariamente. El proyecto está diseñado con diferentes niveles de acceso:
- **Usuarios básicos**: Pueden utilizar los scripts principales y el dashboard interactivo sin modificar código
- **Usuarios intermedios**: Pueden ajustar parámetros y combinar diferentes módulos
- **Usuarios avanzados**: Pueden extender la funcionalidad y personalizar los análisis

## Instalación y Configuración

### ¿Cuáles son los requisitos mínimos del sistema?

- Python 3.7 o superior
- 4GB de RAM (8GB recomendado para conjuntos de datos grandes)
- Espacio en disco: 500MB para la instalación básica, más espacio adicional para datos y resultados

### ¿Cómo instalo todas las dependencias necesarias?

Puedes instalar todas las dependencias utilizando pip y el archivo requirements.txt:

```bash
pip install -r requirements.txt
```

Para una instalación mínima con funcionalidades básicas:

```bash
pip install -r requirements-minimal.txt
```

### ¿Cómo configuro el proyecto para mi caso de uso específico?

El proyecto utiliza archivos de configuración en el directorio `config/` que puedes modificar según tus necesidades:

1. `config/analysis_config.json`: Configuración para módulos de análisis
2. `config/visualization_config.json`: Parámetros para visualizaciones
3. `config/dashboard_config.json`: Configuración del dashboard interactivo

Puedes editar estos archivos para ajustar parámetros como colores, tamaños de gráficos, algoritmos utilizados, etc.

## Carga y Procesamiento de Datos

### ¿Qué formatos de archivo CSV son compatibles?

El proyecto puede manejar archivos CSV con diferentes configuraciones:
- Delimitadores: coma (,) o punto y coma (;) - detectados automáticamente
- Codificaciones: UTF-8, Latin-1 (ISO-8859-1), y otras codificaciones comunes
- Con o sin encabezados
- Con o sin comillas en los campos

### ¿Cómo maneja el proyecto los valores faltantes o incorrectos?

El proyecto incluye varias estrategias para manejar datos incompletos o incorrectos:

1. **Detección automática**: Identifica valores faltantes, atípicos y formatos incorrectos
2. **Limpieza**: Ofrece opciones para eliminar o imputar valores faltantes
3. **Transformación**: Normaliza formatos inconsistentes (fechas, números, texto)
4. **Registro**: Documenta todas las transformaciones aplicadas para transparencia

### ¿Puedo procesar archivos CSV muy grandes?

Sí, el proyecto incluye opciones para manejar archivos grandes:

1. **Procesamiento por lotes**: Carga y procesa el archivo en fragmentos
2. **Muestreo**: Analiza una muestra representativa para conjuntos muy grandes
3. **Versiones adaptativas**: Módulos con menor consumo de memoria para entornos limitados

Para archivos extremadamente grandes (>1GB), se recomienda utilizar las funciones con el parámetro `low_memory=True` o considerar herramientas complementarias como Dask.

## Análisis de Datos

### ¿Qué tipos de análisis puedo realizar?

El proyecto soporta diversos tipos de análisis:

1. **Análisis exploratorio**: Estadísticas descriptivas, distribuciones, correlaciones
2. **Análisis predictivo**: Modelos de regresión, clasificación y clustering
3. **Análisis geoespacial**: Mapas, clustering espacial, análisis de densidad
4. **Análisis de contenido web**: Extracción de dominios, análisis de texto
5. **Análisis temporal**: Tendencias, estacionalidad, detección de anomalías

### ¿Cómo interpreto los resultados del análisis predictivo?

Los resultados del análisis predictivo incluyen:

1. **Métricas de rendimiento**: Precisión, recall, F1-score para clasificación; RMSE, MAE, R² para regresión
2. **Importancia de características**: Qué variables tienen mayor impacto en las predicciones
3. **Visualizaciones**: Matrices de confusión, curvas ROC, gráficos de residuos
4. **Informes**: Resúmenes en formato HTML y PDF con interpretaciones

Consulta la documentación específica en `docs/mejores_practicas_analisis.md` para guías detalladas sobre interpretación.

### ¿Puedo personalizar los algoritmos utilizados?

Sí, puedes personalizar los algoritmos de varias maneras:

1. **Parámetros en línea de comandos**: Muchos scripts aceptan parámetros para ajustar algoritmos
2. **Archivos de configuración**: Modifica `config/analysis_config.json` para cambios permanentes
3. **Modificación de código**: Extiende las clases existentes para implementar algoritmos personalizados

## Visualización y Dashboards

### ¿Cómo puedo personalizar las visualizaciones?

Las visualizaciones se pueden personalizar de varias formas:

1. **Parámetros de función**: La mayoría de funciones de visualización aceptan parámetros como colores, tamaños, títulos
2. **Archivo de configuración**: Edita `config/visualization_config.json` para cambios globales
3. **Estilos de matplotlib**: Utiliza `plt.style.use()` para aplicar estilos predefinidos o personalizados
4. **Post-procesamiento**: Las funciones devuelven objetos de figura que puedes modificar antes de guardar

### ¿Cómo comparto mi dashboard con otros usuarios?

Hay varias opciones para compartir dashboards:

1. **Dashboard interactivo**:
   - Ejecuta localmente y comparte la URL (http://localhost:8050 por defecto)
   - Despliega en servicios como Heroku o PythonAnywhere

2. **Dashboard estático**:
   - Genera un archivo HTML autónomo con `dashboard_adaptativo.py`
   - Comparte el archivo HTML por correo o servicios de almacenamiento

3. **Informes**:
   - Genera informes en PDF con visualizaciones y análisis
   - Exporta visualizaciones individuales como imágenes PNG o SVG

### ¿El dashboard funciona en dispositivos móviles?

Sí, el dashboard interactivo está diseñado con un diseño responsivo que se adapta a diferentes tamaños de pantalla. Sin embargo, para una mejor experiencia en dispositivos móviles, considera:

1. Utilizar la versión adaptativa (`dashboard_adaptativo.py`) que está optimizada para pantallas pequeñas
2. Limitar el número de gráficos mostrados simultáneamente
3. Preferir visualizaciones simples sobre complejas para pantallas pequeñas

## Análisis Geoespacial

### ¿Qué formato deben tener mis datos para el análisis geoespacial?

Para el análisis geoespacial básico, tus datos deben incluir:

1. Columnas con coordenadas geográficas:
   - Latitud (valores entre -90 y 90)
   - Longitud (valores entre -180 y 180)

2. Opcionalmente, columnas adicionales para análisis y visualización:
   - Categorías para colorear puntos
   - Valores numéricos para tamaños o colores graduados
   - Texto para etiquetas o popups

Para análisis más avanzados, también se aceptan formatos GeoJSON, Shapefile y otros formatos geoespaciales estándar.

### ¿Necesito conexión a internet para los mapas interactivos?

Sí, los mapas interactivos generados con Folium utilizan bibliotecas JavaScript y tiles de mapas que requieren conexión a internet para cargarse correctamente. Sin embargo, una vez cargado, el mapa puede ser utilizado sin conexión.

Para uso completamente offline, el proyecto incluye opciones para generar mapas estáticos que no requieren conexión a internet después de ser generados.

### ¿Cómo interpreto los resultados del clustering espacial?

El clustering espacial agrupa puntos geográficamente cercanos. Para interpretar los resultados:

1. **Asignación de clusters**: Cada punto se asigna a un cluster (número o color)
2. **Centros de clusters**: Representan la ubicación promedio de cada grupo
3. **Métricas**: Incluyen cohesión interna (qué tan cercanos están los puntos dentro de un cluster) y separación (qué tan distintos son los clusters)

Los mapas generados muestran los puntos coloreados por cluster y, opcionalmente, los centros de cada cluster.

## Análisis Web

### ¿Qué información puedo extraer de URLs en mis datos?

El módulo de análisis web puede extraer y analizar:

1. **Componentes de URL**: Dominio, subdominio, ruta, parámetros
2. **Metadatos**: Título, descripción, palabras clave (cuando está disponible)
3. **Contenido**: Texto principal, imágenes, enlaces
4. **Estadísticas**: Longitud, densidad de palabras clave, estructura

### ¿El análisis web requiere conexión a internet?

Sí, para analizar el contenido actual de las URLs se requiere conexión a internet. Sin embargo:

1. Los resultados se almacenan en caché para análisis posteriores
2. Puedes limitar el análisis solo a los componentes de URL (sin descargar contenido)
3. Para conjuntos grandes, puedes analizar solo una muestra de URLs

### ¿Cómo maneja el proyecto URLs que ya no están disponibles?

El proyecto implementa varias estrategias:

1. **Timeouts configurables**: Evita esperas largas por URLs no disponibles
2. **Reintentos**: Intenta múltiples veces con intervalos crecientes
3. **Registro**: Documenta URLs fallidas para análisis posterior
4. **Modo robusto**: Continúa el análisis incluso cuando algunas URLs fallan

## Solución de Problemas

### El programa se cierra al procesar archivos grandes

Esto generalmente indica un problema de memoria. Soluciones:

1. Utiliza procesamiento por lotes:
   ```python
   from src.analysis.limpiar_analizar_csv import analyze_csv_in_batches
   analyze_csv_in_batches('archivo_grande.csv', batch_size=10000)
   ```

2. Utiliza la versión adaptativa con menor consumo de memoria:
   ```python
   from src.analysis.limpiar_analizar_csv import analyze_csv_low_memory
   analyze_csv_low_memory('archivo_grande.csv')
   ```

3. Reduce el alcance del análisis especificando solo las columnas necesarias:
   ```python
   from src.analysis.limpiar_analizar_csv import analyze_csv
   analyze_csv('archivo.csv', columns=['col1', 'col2', 'col3'])
   ```

### Obtengo errores de codificación al cargar archivos CSV

Los problemas de codificación son comunes con archivos CSV. Soluciones:

1. Especifica la codificación manualmente:
   ```python
   from src.analysis.limpiar_analizar_csv import analyze_csv
   analyze_csv('archivo.csv', encoding='latin1')  # o 'utf-8-sig', 'cp1252', etc.
   ```

2. Utiliza la función de detección automática de codificación:
   ```python
   from src.utils.helpers import detect_encoding
   encoding = detect_encoding('archivo.csv')
   print(f"La codificación detectada es: {encoding}")
   ```

3. Pre-procesa el archivo para convertirlo a UTF-8:
   ```python
   from src.utils.helpers import convert_to_utf8
   convert_to_utf8('archivo_original.csv', 'archivo_utf8.csv')
   ```

### El dashboard interactivo no muestra gráficos o muestra errores

Problemas comunes con el dashboard:

1. **Puerto en uso**: Cambia el puerto predeterminado:
   ```python
   from src.dashboard.dashboard_interactivo import run_dashboard
   run_dashboard('datos.csv', port=8051)  # en lugar del predeterminado 8050
   ```

2. **Tipos de datos incompatibles**: Asegúrate de que las columnas tienen los tipos correctos:
   ```python
   from src.dashboard.dashboard_interactivo import run_dashboard
   run_dashboard('datos.csv', dtype={'columna_numerica': float, 'columna_fecha': str})
   ```

3. **Memoria insuficiente**: Utiliza la versión ligera del dashboard:
   ```python
   from src.dashboard.dashboard_adaptativo import run_lightweight_dashboard
   run_lightweight_dashboard('datos.csv')
   ```

### Los mapas geoespaciales no se cargan o muestran correctamente

Problemas comunes con mapas:

1. **Coordenadas inválidas**: Verifica y limpia las coordenadas:
   ```python
   from src.geospatial.analisis_geoespacial import clean_coordinates
   df_cleaned = clean_coordinates(df, 'latitud', 'longitud')
   ```

2. **Bibliotecas faltantes**: Instala dependencias adicionales:
   ```bash
   pip install folium geopandas rtree shapely
   ```

3. **Problemas de renderizado**: Utiliza mapas estáticos en lugar de interactivos:
   ```python
   from src.geospatial.analisis_geoespacial_adaptativo import create_static_map
   create_static_map(df, 'latitud', 'longitud', 'mapa.png')
   ```

### ¿Dónde puedo encontrar más ayuda?

Si sigues teniendo problemas:

1. Consulta la documentación detallada en el directorio `docs/`
2. Revisa los ejemplos en el directorio `examples/`
3. Busca problemas similares en la sección de issues del repositorio
4. Crea un nuevo issue con una descripción detallada del problema, incluyendo:
   - Pasos para reproducir
   - Mensajes de error completos
   - Información del sistema (versión de Python, sistema operativo)
   - Muestra de datos (si es posible, sin información sensible)

---

Esperamos que estas preguntas frecuentes te ayuden a resolver los problemas más comunes. La documentación se actualiza regularmente con nuevas preguntas y soluciones.