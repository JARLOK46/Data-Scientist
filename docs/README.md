# Análisis de Datos y Visualización

Este proyecto proporciona herramientas para el análisis, limpieza y visualización de datos del archivo CSV `20250525.export.CSV`. Incluye scripts para análisis exploratorio, predictivo, geoespacial y visualización interactiva.

## Contenido del Proyecto

### Scripts Principales

- **limpiar_analizar_csv.py**: Limpieza inicial y análisis exploratorio de datos.
- **visualizar_datos.py**: Generación de visualizaciones estáticas básicas.
- **analisis_predictivo.py**: Modelos predictivos utilizando técnicas de machine learning.
- **analisis_contenido_web.py**: Análisis de URLs y contenido web en los datos.
- **analisis_geoespacial.py**: Análisis geoespacial completo (requiere bibliotecas adicionales).
- **analisis_geoespacial_adaptativo.py**: Versión adaptativa del análisis geoespacial que funciona con dependencias mínimas.
- **dashboard_interactivo.py**: Dashboard interactivo con Dash y Plotly (requiere instalación de dependencias).
- **dashboard_adaptativo.py**: Generación de visualizaciones estáticas e informe HTML sin dependencias adicionales.

### Archivos de Configuración

- **requirements_dashboard.txt**: Dependencias necesarias para el dashboard interactivo.
- **requirements_geoespacial.txt**: Dependencias para el análisis geoespacial completo.
- **setup_dashboard.py**: Script para verificar e instalar las dependencias del dashboard interactivo.

### Informes y Resultados

- **informe_analisis_datos.md**: Informe general del análisis de datos.
- **dashboard_estatico/**: Directorio con visualizaciones estáticas e informe HTML.
- **visualizaciones/**: Directorio con gráficos generados por el análisis exploratorio.
- **analisis_geoespacial/**: Directorio con resultados del análisis geoespacial.

## Guía de Uso

### 1. Análisis Básico y Visualizaciones Estáticas

Para un análisis rápido sin dependencias adicionales:

```bash
python dashboard_adaptativo.py
```

Este script generará visualizaciones estáticas y un informe HTML en el directorio `dashboard_estatico/`.

### 2. Dashboard Interactivo

Para utilizar el dashboard interactivo, primero debe instalar las dependencias necesarias:

**Opción 1**: Usar el script de configuración automática:

```bash
python setup_dashboard.py
```

Este script verificará las dependencias, las instalará si es necesario y ofrecerá ejecutar el dashboard.

**Opción 2**: Instalar manualmente las dependencias:

```bash
pip install -r requirements_dashboard.txt
python dashboard_interactivo.py
```

Una vez iniciado, acceda al dashboard en su navegador: http://127.0.0.1:8050/

### 3. Análisis Geoespacial

Para ejecutar el análisis geoespacial adaptativo (funciona con dependencias mínimas):

```bash
python analisis_geoespacial_adaptativo.py
```

Para el análisis geoespacial completo (requiere bibliotecas adicionales):

```bash
pip install -r requirements_geoespacial.txt
python analisis_geoespacial.py
```

### 4. Análisis Predictivo y de Contenido Web

Para ejecutar el análisis predictivo:

```bash
python analisis_predictivo.py
```

Para el análisis de contenido web:

```bash
python analisis_contenido_web.py
```

## Requisitos del Sistema

- Python 3.7 o superior
- Bibliotecas básicas: pandas, numpy, matplotlib, seaborn
- Para el dashboard interactivo: dash, plotly
- Para análisis geoespacial completo: folium, geopandas, scikit-learn

## Solución de Problemas

### Error "No module named 'dash'"

Este error indica que las dependencias del dashboard interactivo no están instaladas. Ejecute:

```bash
python setup_dashboard.py
```

o instale manualmente las dependencias:

```bash
pip install -r requirements_dashboard.txt
```

### Error con bibliotecas geoespaciales

Si encuentra errores relacionados con bibliotecas geoespaciales, utilice la versión adaptativa:

```bash
python analisis_geoespacial_adaptativo.py
```

o instale las dependencias completas:

```bash
pip install -r requirements_geoespacial.txt
```

## Notas Adicionales

- Los informes generados se guardan en formato Markdown (.md) y HTML para fácil visualización.
- Las visualizaciones estáticas se guardan como archivos PNG en sus respectivos directorios.
- El dashboard interactivo permite exploración dinámica de los datos con filtros personalizables.#   D a t a - S c i e n t i s t  
 