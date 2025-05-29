# Guía de Análisis Avanzado

Este documento proporciona técnicas y metodologías avanzadas para maximizar el valor de los análisis realizados con el sistema "UN PASO AL EXITO". Está dirigido a usuarios que ya están familiarizados con las funcionalidades básicas del sistema.

## Índice

1. [Preparación de Datos Avanzada](#preparación-de-datos-avanzada)
2. [Técnicas de Visualización Avanzada](#técnicas-de-visualización-avanzada)
3. [Análisis Predictivo Optimizado](#análisis-predictivo-optimizado)
4. [Análisis Geoespacial Avanzado](#análisis-geoespacial-avanzado)
5. [Análisis de Texto y Web Profundo](#análisis-de-texto-y-web-profundo)
6. [Automatización de Análisis](#automatización-de-análisis)
7. [Integración con Otras Herramientas](#integración-con-otras-herramientas)

## Preparación de Datos Avanzada

### Estrategias para Datos Desbalanceados

Cuando trabajas con datos desbalanceados (por ejemplo, en clasificación donde una clase es mucho más frecuente que otras):

```python
# Importar módulos necesarios
from src.analysis.analisis_predictivo import balance_dataset
from src.utils.helpers import load_data

# Cargar datos
df = load_data('datos_desbalanceados.csv')

# Aplicar técnicas de balanceo
df_balanced = balance_dataset(df, target_column='clase', method='smote')
# Alternativas: 'undersample', 'oversample', 'smote', 'adasyn'
```

### Detección y Tratamiento de Valores Atípicos

Los valores atípicos pueden distorsionar significativamente los análisis:

```python
from src.analysis.limpiar_analizar_csv import detect_outliers, handle_outliers

# Detectar valores atípicos usando múltiples métodos
outliers_info = detect_outliers(df, columns=['columna_numerica'], 
                               methods=['zscore', 'iqr', 'isolation_forest'])

# Visualizar valores atípicos
from src.visualization.visualizar_datos import plot_outliers
plot_outliers(df, outliers_info, save_path='outliers_analysis.png')

# Tratar valores atípicos
df_cleaned = handle_outliers(df, outliers_info, 
                           strategy='winsorize', # Alternativas: 'remove', 'cap', 'transform'
                           percentile=0.05)
```

### Ingeniería de Características Avanzada

La creación de características relevantes puede mejorar significativamente el rendimiento de los modelos:

```python
from src.analysis.analisis_predictivo import feature_engineering

# Aplicar transformaciones automáticas basadas en el tipo de datos
df_engineered = feature_engineering(df, 
                                   date_columns=['fecha'], 
                                   categorical_columns=['categoria'], 
                                   text_columns=['descripcion'],
                                   numeric_columns=['valor'],
                                   interactions=True, # Crear interacciones entre variables
                                   polynomial=2)     # Características polinómicas
```

## Técnicas de Visualización Avanzada

### Visualizaciones Multidimensionales

Para explorar relaciones complejas entre múltiples variables:

```python
from src.visualization.visualizar_datos import create_advanced_visualization

# Crear matriz de dispersión con múltiples variables
create_advanced_visualization(df, 
                             plot_type='pairplot', 
                             vars=['var1', 'var2', 'var3', 'var4'],
                             hue='categoria',
                             save_path='analisis_multivariable.png')

# Crear visualización de coordenadas paralelas
create_advanced_visualization(df, 
                             plot_type='parallel_coordinates', 
                             vars=['var1', 'var2', 'var3', 'var4'],
                             class_column='categoria',
                             save_path='coordenadas_paralelas.png')
```

### Visualizaciones Interactivas Personalizadas

Para crear visualizaciones interactivas más allá del dashboard estándar:

```python
from src.dashboard.dashboard_components import create_custom_visualization

# Crear visualización interactiva personalizada
fig = create_custom_visualization(
    df,
    x='variable_x',
    y='variable_y',
    size='variable_tamaño',
    color='variable_color',
    hover_data=['info1', 'info2'],
    animation_frame='fecha',  # Para gráficos animados
    visualization_type='scatter'  # Alternativas: 'bar', 'line', 'heatmap', etc.
)

# Guardar como HTML interactivo
from src.utils.export import save_interactive_plot
save_interactive_plot(fig, 'visualizacion_personalizada.html')
```

## Análisis Predictivo Optimizado

### Selección Automática de Modelos

Para encontrar el mejor modelo para tus datos:

```python
from src.analysis.analisis_predictivo import auto_model_selection

# Configurar y ejecutar selección automática de modelos
best_model, performance_report = auto_model_selection(
    df,
    target_column='objetivo',
    feature_columns=['var1', 'var2', 'var3'],
    problem_type='classification',  # Alternativas: 'regression', 'clustering'
    evaluation_metric='f1',         # Métrica para optimizar
    cv_folds=5,                     # Validación cruzada
    search_iterations=50            # Iteraciones para búsqueda de hiperparámetros
)

# Guardar reporte de rendimiento
from src.utils.export import save_model_report
save_model_report(performance_report, 'seleccion_modelo.html')
```

### Interpretabilidad de Modelos

Para entender y explicar las predicciones de tus modelos:

```python
from src.analysis.model_interpretation import explain_model

# Generar explicaciones para el modelo
explanations = explain_model(
    model=best_model,
    X=df[['var1', 'var2', 'var3']],
    method='shap',  # Alternativas: 'lime', 'eli5', 'partial_dependence'
    n_samples=100   # Número de muestras para explicaciones
)

# Visualizar explicaciones
from src.visualization.model_viz import plot_feature_importance, plot_explanations
plot_feature_importance(explanations, save_path='importancia_caracteristicas.png')
plot_explanations(explanations, sample_idx=0, save_path='explicacion_prediccion.png')
```

## Análisis Geoespacial Avanzado

### Análisis de Patrones Espaciales

Para identificar patrones y relaciones espaciales en tus datos:

```python
from src.geospatial.analisis_geoespacial import spatial_pattern_analysis

# Realizar análisis de patrones espaciales
pattern_results = spatial_pattern_analysis(
    df,
    lat_column='latitud',
    lon_column='longitud',
    value_column='variable_interes',
    methods=['hotspot', 'autocorrelation', 'density'],
    significance_level=0.05
)

# Visualizar resultados
from src.geospatial.geo_visualization import plot_spatial_analysis
plot_spatial_analysis(pattern_results, analysis_type='hotspot', 
                     save_path='analisis_hotspot.html')
```

### Integración con Datos Externos Geoespaciales

Para enriquecer tu análisis con datos geoespaciales externos:

```python
from src.geospatial.geo_enrichment import enrich_with_external_data

# Enriquecer datos con información externa
df_enriched = enrich_with_external_data(
    df,
    lat_column='latitud',
    lon_column='longitud',
    data_sources=['osm', 'census', 'elevation'],  # Fuentes de datos
    buffer_distance=1000  # Distancia en metros para búsqueda
)

# Analizar correlaciones con nuevas variables
from src.analysis.limpiar_analizar_csv import correlation_analysis
correlation_analysis(df_enriched, 
                    target='variable_objetivo',
                    save_path='correlaciones_variables_geograficas.png')
```

## Análisis de Texto y Web Profundo

### Análisis de Sentimiento Avanzado

Para un análisis más profundo del sentimiento en textos:

```python
from src.web.text_analysis import advanced_sentiment_analysis

# Realizar análisis de sentimiento avanzado
sentiment_results = advanced_sentiment_analysis(
    df,
    text_column='texto',
    methods=['vader', 'textblob', 'transformers'],
    aspects=True,  # Análisis de sentimiento basado en aspectos
    emotions=True  # Detección de emociones específicas
)

# Visualizar distribución de sentimientos
from src.visualization.text_viz import plot_sentiment_distribution
plot_sentiment_distribution(sentiment_results, save_path='distribucion_sentimientos.png')
```

### Extracción y Análisis de Entidades

Para identificar y analizar entidades mencionadas en textos:

```python
from src.web.entity_analysis import extract_and_analyze_entities

# Extraer y analizar entidades
entity_analysis = extract_and_analyze_entities(
    df,
    text_column='texto',
    entity_types=['PERSON', 'ORG', 'LOC', 'DATE'],
    link_entities=True,  # Vincular entidades con bases de conocimiento
    co_occurrence=True   # Analizar co-ocurrencias
)

# Visualizar red de entidades
from src.visualization.network_viz import plot_entity_network
plot_entity_network(entity_analysis, 
                   min_weight=2,  # Filtrar conexiones débiles
                   save_path='red_entidades.html')
```

## Automatización de Análisis

### Creación de Flujos de Trabajo Personalizados

Para automatizar secuencias completas de análisis:

```python
from src.utils.workflow import create_analysis_workflow

# Definir y ejecutar flujo de trabajo personalizado
workflow = create_analysis_workflow([
    {'step': 'load_data', 'params': {'file_path': 'datos.csv'}},
    {'step': 'clean_data', 'params': {'handle_missing': True, 'handle_outliers': True}},
    {'step': 'feature_engineering', 'params': {'date_columns': ['fecha']}},
    {'step': 'train_model', 'params': {'target': 'objetivo', 'model_type': 'random_forest'}},
    {'step': 'evaluate_model', 'params': {'cv_folds': 5}},
    {'step': 'generate_report', 'params': {'output_path': 'reporte_analisis.html'}}
])

# Ejecutar el flujo de trabajo
results = workflow.run()

# Programar ejecución periódica
from src.utils.scheduler import schedule_workflow
schedule_workflow(workflow, 
                 schedule_type='daily',  # Alternativas: 'weekly', 'monthly', 'custom'
                 time='02:00',          # Hora de ejecución
                 output_dir='reportes_automaticos/')
```

### Monitoreo de Modelos en Producción

Para supervisar el rendimiento de modelos desplegados:

```python
from src.utils.monitoring import setup_model_monitoring

# Configurar monitoreo de modelo
monitoring = setup_model_monitoring(
    model_path='modelos/modelo_produccion.pkl',
    reference_data='datos_referencia.csv',
    metrics=['drift', 'performance', 'predictions'],
    alert_thresholds={'drift': 0.05, 'performance_drop': 0.1},
    notification_channels=['email', 'log']
)

# Ejecutar verificación manual
monitoring_report = monitoring.check_model('nuevos_datos.csv')

# Visualizar resultados del monitoreo
from src.visualization.monitoring_viz import plot_monitoring_results
plot_monitoring_results(monitoring_report, save_path='monitoreo_modelo.html')
```

## Integración con Otras Herramientas

### Exportación Avanzada a Herramientas de BI

Para integrar tus análisis con plataformas de Business Intelligence:

```python
from src.utils.export import export_to_bi_tool

# Exportar resultados a herramienta de BI
export_to_bi_tool(
    data=results_df,
    visualizations=generated_visualizations,
    tool='power_bi',  # Alternativas: 'tableau', 'looker', 'qlik'
    connection_params={
        'workspace_id': 'tu_workspace',
        'dataset_name': 'Análisis Avanzado',
        'refresh_schedule': 'daily'
    },
    create_dashboard=True,
    dashboard_template='template_ejecutivo.json'
)
```

### Integración con Sistemas de Bases de Datos

Para trabajar directamente con bases de datos en lugar de archivos CSV:

```python
from src.utils.database import setup_database_connection, query_to_dataframe

# Configurar conexión a base de datos
connection = setup_database_connection(
    db_type='postgresql',  # Alternativas: 'mysql', 'sqlserver', 'oracle', 'sqlite'
    connection_params={
        'host': 'localhost',
        'database': 'nombre_bd',
        'user': 'usuario',
        'password': 'contraseña'
    }
)

# Ejecutar consulta y obtener DataFrame
query = """
    SELECT columna1, columna2, fecha, valor 
    FROM tabla 
    WHERE fecha > '2023-01-01' 
    ORDER BY fecha
"""
df = query_to_dataframe(connection, query)

# Guardar resultados en la base de datos
from src.utils.database import save_results_to_database
save_results_to_database(
    connection,
    df_results,
    table_name='resultados_analisis',
    if_exists='replace'  # Alternativas: 'append', 'fail'
)
```

## Conclusión

Las técnicas avanzadas presentadas en esta guía te permitirán aprovechar al máximo las capacidades del sistema "UN PASO AL EXITO". Recuerda que el análisis efectivo combina el conocimiento del dominio, la selección adecuada de técnicas y la interpretación cuidadosa de los resultados.

Para obtener más información sobre implementaciones específicas, consulta los ejemplos en el directorio `examples/advanced/` y la documentación de API completa generada con Sphinx.

---

**Nota**: Los ejemplos de código en esta guía asumen que las funciones mencionadas están implementadas en el sistema. Si alguna funcionalidad específica no está disponible, consulta la sección de extensibilidad en la documentación para implementar tus propias soluciones personalizadas.