# Diagrama de Flujo de Datos

Este documento presenta el flujo de datos a través de los diferentes módulos del sistema de análisis de datos.

## Visión General del Flujo de Datos

```mermaid
graph TD
    A[Datos CSV Originales] --> B[Módulo de Limpieza]
    B --> C[Datos Limpios]
    
    C --> D[Análisis Exploratorio]
    C --> E[Análisis Predictivo]
    C --> F[Análisis Geoespacial]
    C --> G[Análisis de Contenido Web]
    
    D --> H[Visualizaciones Estáticas]
    E --> I[Modelos Predictivos]
    F --> J[Mapas y Clusters Geoespaciales]
    G --> K[Análisis de Dominios y Contenido]
    
    H --> L[Dashboard Estático]
    I --> L
    J --> L
    K --> L
    
    H --> M[Dashboard Interactivo]
    I --> M
    J --> M
    K --> M
```

## Descripción de los Componentes

### Entrada de Datos

- **Datos CSV Originales**: Archivo `20250525.export.CSV` con los datos brutos.

### Procesamiento de Datos

- **Módulo de Limpieza**: Implementado en `limpiar_analizar_csv.py`, realiza la limpieza inicial de los datos.
- **Datos Limpios**: Almacenados en `datos_limpios.csv`, representan la versión procesada lista para análisis.

### Módulos de Análisis

- **Análisis Exploratorio**: Estadísticas descriptivas y análisis básico de los datos.
- **Análisis Predictivo**: Modelos de machine learning para predicciones.
- **Análisis Geoespacial**: Análisis basado en coordenadas geográficas.
- **Análisis de Contenido Web**: Análisis de URLs y dominios en los datos.

### Resultados

- **Visualizaciones Estáticas**: Gráficos y figuras generados por los módulos de análisis.
- **Modelos Predictivos**: Modelos entrenados y sus predicciones.
- **Mapas y Clusters Geoespaciales**: Visualizaciones geográficas y agrupaciones.
- **Análisis de Dominios y Contenido**: Resultados del análisis de contenido web.

### Presentación

- **Dashboard Estático**: Informe HTML con visualizaciones estáticas.
- **Dashboard Interactivo**: Aplicación Dash para exploración interactiva de los datos.

## Flujos Específicos

### Flujo de Análisis Predictivo

```mermaid
sequenceDiagram
    participant CSV as Datos CSV
    participant Prep as Preparación de Datos
    participant Train as Entrenamiento de Modelos
    participant Eval as Evaluación
    participant Vis as Visualización
    
    CSV->>Prep: Carga y limpieza
    Prep->>Prep: Transformación de características
    Prep->>Train: División train/test
    Train->>Train: Entrenamiento de modelos
    Train->>Eval: Modelos entrenados
    Eval->>Eval: Métricas de rendimiento
    Eval->>Vis: Resultados de evaluación
    Vis->>Vis: Generación de gráficos
```

### Flujo de Análisis Geoespacial

```mermaid
sequenceDiagram
    participant CSV as Datos CSV
    participant Geo as Procesamiento Geoespacial
    participant Clust as Clustering
    participant Map as Generación de Mapas
    participant Vis as Visualización
    
    CSV->>Geo: Carga y limpieza
    Geo->>Geo: Extracción de coordenadas
    Geo->>Clust: Datos geoespaciales
    Clust->>Clust: Algoritmos de clustering
    Clust->>Map: Clusters identificados
    Map->>Map: Creación de mapas interactivos
    Map->>Vis: Mapas y visualizaciones
```

## Integración con el Sistema

El flujo de datos está orquestado por el módulo `ejecutar_analisis.py`, que proporciona una interfaz unificada para ejecutar los diferentes tipos de análisis. Los resultados de todos los análisis se pueden visualizar a través de los dashboards estático e interactivo.

## Consideraciones de Rendimiento

- Los datos limpios se almacenan en disco para evitar reprocesamiento.
- Los modelos predictivos se serializan para su reutilización.
- El dashboard adaptativo proporciona una alternativa ligera cuando no se pueden instalar todas las dependencias.

## Extensibilidad

El diseño modular permite añadir nuevos tipos de análisis o visualizaciones sin modificar el flujo principal de datos. Cada módulo de análisis es independiente y puede ser mejorado o reemplazado según sea necesario.