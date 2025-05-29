"""
Módulo de análisis predictivo.

Este script permite preparar datos, entrenar y evaluar modelos de regresión, y generar visualizaciones de resultados.

Autor: Anderson Zapata
Fecha: 2025
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from urllib.parse import urlparse
import os
import sys
import joblib
from datetime import datetime

# Añadir el directorio raíz del proyecto al path para poder importar módulos
proyecto_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, proyecto_dir)

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Crear directorio para modelos si no existe
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelos')
os.makedirs(models_dir, exist_ok=True)

# Función para detectar el delimitador
def detect_delimiter(file_path, num_lines=5):
    """
    Detecta automáticamente el delimitador utilizado en un archivo CSV.
    
    Args:
        file_path (str): Ruta al archivo CSV.
        num_lines (int, optional): Número de líneas a analizar. Por defecto es 5.
    
    Returns:
        str: El delimitador detectado (tab, coma, punto y coma o pipe).
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        sample = ''.join([file.readline() for _ in range(num_lines)])
    
    delimiters = {'\t': 0, ',': 0, ';': 0, '|': 0}
    for delimiter in delimiters:
        delimiters[delimiter] = sample.count(delimiter)
    
    return max(delimiters, key=delimiters.get)

# Cargar y preparar datos
def load_and_prepare_data(file_path):
    """
    Carga y prepara los datos desde un archivo CSV para el análisis predictivo.
    
    Args:
        file_path (str): Ruta al archivo CSV con los datos.
    
    Returns:
        tuple: Contiene el DataFrame completo, las características numéricas y categóricas,
               y la variable objetivo si se encuentra.
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
            return None
    
    print(f"Datos cargados. Dimensiones: {df.shape}")
    
    # Limpiar nombres de columnas
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    
    # Eliminar duplicados
    df_cleaned = df.drop_duplicates()
    print(f"Duplicados eliminados: {df.shape[0] - df_cleaned.shape[0]}")
    
    # Identificar columnas numéricas y rellenar valores nulos
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
    
    # Identificar columnas de texto
    text_cols = df_cleaned.select_dtypes(include=['object']).columns.tolist()
    
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
        # Crear una columna de región basada en coordenadas (clustering)
        geo_data = df_cleaned[[geo_cols['lat'][0], geo_cols['lon'][0]]].dropna()
        if len(geo_data) > 10:  # Asegurar que hay suficientes datos para clustering
            kmeans = KMeans(n_clusters=min(5, len(geo_data) // 2), random_state=42)
            df_cleaned.loc[geo_data.index, 'region_cluster'] = kmeans.fit_predict(
                geo_data[[geo_cols['lat'][0], geo_cols['lon'][0]]])
            print("Clustering geográfico aplicado para crear regiones")
    
    return df_cleaned, numeric_cols, text_cols

# Seleccionar características y objetivo para el modelo
def select_features_target(df, numeric_cols):
    """
    Selecciona automáticamente las características y la variable objetivo para el modelo predictivo.
    
    Esta función analiza las correlaciones entre variables numéricas para identificar la variable
    objetivo más adecuada (aquella con mayor correlación con otras variables) y selecciona las
    características numéricas y categóricas relevantes para el modelo.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos preparados.
        numeric_cols (list): Lista de nombres de columnas numéricas en el DataFrame.
    
    Returns:
        tuple: Contiene tres elementos:
            - feature_cols (list): Lista de columnas numéricas seleccionadas como características.
            - categorical_cols (list): Lista de columnas categóricas relevantes (con menos de 20 categorías).
            - target_col (str): Nombre de la columna seleccionada como variable objetivo.
    
    Nota:
        Si solo hay una variable numérica disponible, esta será seleccionada como objetivo.
        Las columnas categóricas seleccionadas son aquellas con entre 2 y 19 categorías únicas.
    """
    # Seleccionar la variable objetivo (la que tenga mayor correlación con otras variables)
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        # Sumar las correlaciones para cada variable
        corr_sum = corr_matrix.sum() - 1  # Restar 1 para eliminar la correlación consigo misma
        # Seleccionar la variable con mayor suma de correlaciones
        target_col = corr_sum.idxmax()
        print(f"Variable objetivo seleccionada automáticamente: {target_col}")
    else:
        target_col = numeric_cols[0]
        print(f"Solo hay una variable numérica, usando {target_col} como objetivo")
    
    # Seleccionar características (todas las demás columnas numéricas)
    feature_cols = [col for col in numeric_cols if col != target_col]
    
    # Añadir columnas categóricas relevantes
    categorical_cols = []
    for col in df.columns:
        if col not in numeric_cols and df[col].nunique() < 20 and df[col].nunique() > 1:
            categorical_cols.append(col)
    
    print(f"Características numéricas: {feature_cols}")
    print(f"Características categóricas: {categorical_cols}")
    
    return feature_cols, categorical_cols, target_col

# Preparar pipeline de preprocesamiento
def create_preprocessing_pipeline(numeric_features, categorical_features):
    """
    Crea un pipeline de preprocesamiento para características numéricas y categóricas.
    
    Esta función construye un pipeline de transformación de datos que incluye:
    - Para variables numéricas: imputación de valores faltantes con la mediana y estandarización.
    - Para variables categóricas: imputación de valores faltantes con 'missing' y codificación one-hot.
    
    Args:
        numeric_features (list): Lista de nombres de columnas numéricas.
        categorical_features (list): Lista de nombres de columnas categóricas.
    
    Returns:
        ColumnTransformer: Un objeto ColumnTransformer configurado con los transformadores
                          para características numéricas y categóricas.
    
    Nota:
        El preprocesador resultante puede ser utilizado en un pipeline de scikit-learn
        junto con un modelo de aprendizaje automático.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

# Entrenar y evaluar modelos
def train_evaluate_models(X, y):
    """
    Entrena y evalúa múltiples modelos de regresión para seleccionar el mejor.
    
    Esta función entrena varios modelos de regresión (Lineal, Ridge, Lasso, Random Forest y Gradient Boosting)
    sobre los datos proporcionados, evalúa su rendimiento utilizando métricas como RMSE, MAE y R², 
    y selecciona el modelo con mejor coeficiente de determinación (R²).
    
    Args:
        X (pandas.DataFrame): Características de entrada para el modelo.
        y (pandas.Series): Variable objetivo a predecir.
    
    Returns:
        tuple: Contiene dos elementos:
            - results (dict): Diccionario con los resultados de cada modelo, incluyendo el modelo entrenado
                             y sus métricas de rendimiento (RMSE, MAE, R²).
            - best_model (str): Nombre del modelo con mejor rendimiento según R².
    
    Nota:
        La función divide automáticamente los datos en conjuntos de entrenamiento (80%) y prueba (20%)
        utilizando una semilla aleatoria fija para garantizar reproducibilidad.
    """
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Datos de entrenamiento: {X_train.shape}, Datos de prueba: {X_test.shape}")
    
    # Definir modelos a probar
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    results = {}
    best_model = None
    best_score = -np.inf
    
    for name, model in models.items():
        print(f"\nEntrenando modelo: {name}")
        model.fit(X_train, y_train)
        
        # Predicciones
        y_pred = model.predict(X_test)
        
        # Métricas
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        
        results[name] = {
            'model': model,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        # Actualizar mejor modelo
        if r2 > best_score:
            best_score = r2
            best_model = name
    
    print(f"\nMejor modelo: {best_model} con R² = {results[best_model]['r2']:.4f}")
    return results, best_model

# Optimizar hiperparámetros del mejor modelo
def optimize_best_model(best_model_name, X, y, preprocessor):
    """
    Optimiza los hiperparámetros del mejor modelo seleccionado mediante validación cruzada.
    
    Esta función toma el nombre del mejor modelo identificado previamente y realiza una búsqueda
    de hiperparámetros utilizando GridSearchCV para encontrar la configuración óptima. Cada tipo
    de modelo tiene su propio conjunto de hiperparámetros a optimizar.
    
    Args:
        best_model_name (str): Nombre del mejor modelo seleccionado ('Linear Regression', 
                              'Ridge Regression', 'Lasso Regression', 'Random Forest' o
                              'Gradient Boosting').
        X (pandas.DataFrame): Características de entrada para el modelo.
        y (pandas.Series): Variable objetivo a predecir.
        preprocessor (ColumnTransformer): Preprocesador para transformar las características.
    
    Returns:
        Pipeline o GridSearchCV: Si hay hiperparámetros para optimizar, devuelve un objeto GridSearchCV
                               con el mejor modelo encontrado. Si no hay hiperparámetros para ajustar
                               (como en Regresión Lineal), devuelve un Pipeline entrenado.
    
    Nota:
        La función utiliza validación cruzada con 5 pliegues y optimiza según la métrica R².
        Para modelos sin hiperparámetros ajustables (como Regresión Lineal), simplemente entrena
        el modelo con el preprocesador.
    """
    # Definir parámetros a optimizar según el modelo
    if best_model_name == 'Linear Regression':
        model = LinearRegression()
        param_grid = {}
    elif best_model_name == 'Ridge Regression':
        model = Ridge()
        param_grid = {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        }
    elif best_model_name == 'Lasso Regression':
        model = Lasso()
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
        }
    elif best_model_name == 'Random Forest':
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    elif best_model_name == 'Gradient Boosting':
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    
    # Crear pipeline completo
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Optimizar hiperparámetros si hay parámetros para ajustar
    if param_grid:
        print(f"\nOptimizando hiperparámetros para {best_model_name}...")
        # Ajustar prefijos a los parámetros
        param_grid = {f'model__{key}': val for key, val in param_grid.items()}
        
        # Usar validación cruzada para encontrar los mejores hiperparámetros
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X, y)
        
        print(f"Mejores parámetros: {grid_search.best_params_}")
        print(f"Mejor puntuación R²: {grid_search.best_score_:.4f}")
        
        return grid_search
    else:
        # Si no hay parámetros para ajustar, simplemente entrenar el pipeline
        pipeline.fit(X, y)
        return pipeline

# Visualizar resultados
def visualize_results(results, y_test, y_pred, target_col):
    """
    Genera y guarda visualizaciones de los resultados del modelo predictivo.
    
    Esta función crea tres tipos de visualizaciones para evaluar el rendimiento del modelo:
    1. Gráfico de comparación de modelos: muestra R² y RMSE para cada modelo evaluado.
    2. Gráfico de predicciones vs valores reales: compara los valores predichos con los reales.
    3. Histograma de errores: muestra la distribución de los errores de predicción.
    
    Args:
        results (dict): Diccionario con los resultados de cada modelo, incluyendo métricas de rendimiento.
        y_test (pandas.Series): Valores reales de la variable objetivo en el conjunto de prueba.
        y_pred (numpy.ndarray): Predicciones del modelo para el conjunto de prueba.
        target_col (str): Nombre de la columna objetivo que se está prediciendo.
    
    Returns:
        None: La función guarda las visualizaciones en el directorio 'visualizaciones_predictivas'
              y muestra mensajes de confirmación.
    
    Nota:
        Las visualizaciones se guardan en formato PNG en un subdirectorio 'visualizaciones_predictivas'
        dentro del directorio del script.
    """
    # Crear directorio para visualizaciones si no existe
    vis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualizaciones_predictivas')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Comparar modelos
    plt.figure(figsize=(12, 6))
    models = list(results.keys())
    r2_scores = [results[model]['r2'] for model in models]
    rmse_scores = [results[model]['rmse'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    bars1 = ax1.bar(x - width/2, r2_scores, width, label='R²', color='#3498db')
    bars2 = ax2.bar(x + width/2, rmse_scores, width, label='RMSE', color='#e74c3c')
    
    ax1.set_xlabel('Modelos')
    ax1.set_ylabel('R² (mayor es mejor)', color='#3498db')
    ax2.set_ylabel('RMSE (menor es mejor)', color='#e74c3c')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    
    ax1.tick_params(axis='y', colors='#3498db')
    ax2.tick_params(axis='y', colors='#e74c3c')
    
    plt.title('Comparación de Modelos: R² vs RMSE')
    plt.tight_layout()
    
    # Añadir leyenda combinada
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    plt.savefig(os.path.join(vis_dir, 'comparacion_modelos.png'))
    print(f"Gráfico guardado: {os.path.join(vis_dir, 'comparacion_modelos.png')}")
    
    # Gráfico de predicciones vs valores reales
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel(f'Valores Reales de {target_col}')
    plt.ylabel(f'Predicciones de {target_col}')
    plt.title('Predicciones vs Valores Reales')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'predicciones_vs_reales.png'))
    print(f"Gráfico guardado: {os.path.join(vis_dir, 'predicciones_vs_reales.png')}")
    
    # Histograma de errores
    errors = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7, color='#2ecc71')
    plt.xlabel('Error de Predicción')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Errores de Predicción')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'distribucion_errores.png'))
    print(f"Gráfico guardado: {os.path.join(vis_dir, 'distribucion_errores.png')}")

# Guardar modelo
def save_model(model, feature_cols, categorical_cols, target_col):
    """
    Guarda el modelo entrenado y sus metadatos asociados.
    
    Esta función serializa el modelo utilizando joblib y guarda información relevante
    sobre las características, columnas categóricas y variable objetivo en un archivo JSON
    de metadatos. Ambos archivos se guardan con una marca de tiempo para facilitar su
    identificación y seguimiento.
    
    Args:
        model: Modelo entrenado (puede ser un Pipeline, GridSearchCV u otro objeto de scikit-learn).
        feature_cols (list): Lista de nombres de columnas numéricas utilizadas como características.
        categorical_cols (list): Lista de nombres de columnas categóricas utilizadas.
        target_col (str): Nombre de la columna objetivo que se predice.
    
    Returns:
        tuple: Contiene dos elementos:
            - model_filename (str): Ruta al archivo donde se guardó el modelo serializado.
            - metadata_filename (str): Ruta al archivo donde se guardaron los metadatos.
    
    Nota:
        Los archivos se guardan en el directorio 'modelos' dentro del directorio del script,
        con nombres que incluyen una marca de tiempo para evitar sobrescrituras.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = os.path.join(models_dir, f'modelo_predictivo_{timestamp}.pkl')
    
    # Guardar el modelo
    joblib.dump(model, model_filename)
    
    # Guardar metadatos del modelo
    metadata = {
        'feature_columns': feature_cols,
        'categorical_columns': categorical_cols,
        'target_column': target_col,
        'timestamp': timestamp,
        'model_type': type(model).__name__
    }
    
    metadata_filename = os.path.join(models_dir, f'metadata_modelo_{timestamp}.json')
    with open(metadata_filename, 'w') as f:
        import json
        json.dump(metadata, f, indent=4)
    
    print(f"\nModelo guardado en: {model_filename}")
    print(f"Metadatos guardados en: {metadata_filename}")
    
    return model_filename, metadata_filename

# Función principal
def main():
    """
    Función principal que ejecuta el flujo completo de análisis predictivo.
    
    Esta función coordina todo el proceso de análisis predictivo, incluyendo:
    1. Carga y preparación de datos desde un archivo CSV.
    2. Selección automática de características y variable objetivo.
    3. Entrenamiento y evaluación de múltiples modelos de regresión.
    4. Optimización de hiperparámetros del mejor modelo.
    5. Visualización de resultados mediante gráficos comparativos.
    6. Almacenamiento del modelo optimizado y sus metadatos.
    
    La función utiliza un archivo CSV predefinido ubicado en el directorio de datos del proyecto
    y ejecuta todas las etapas del análisis de forma secuencial, mostrando información relevante
    en cada paso del proceso.
    
    Returns:
        None: La función no devuelve ningún valor, pero genera archivos de modelo, metadatos
              y visualizaciones en los directorios correspondientes.
    """
    print("=== ANÁLISIS PREDICTIVO ===\n")
    
    # Cargar y preparar datos
    file_path = os.path.join(proyecto_dir, 'src', 'data', '20250525.export.CSV')
    df, numeric_cols, text_cols = load_and_prepare_data(file_path)
    
    if df is None or len(numeric_cols) < 2:
        print("No hay suficientes datos numéricos para realizar análisis predictivo.")
        return
    
    # Seleccionar características y objetivo
    feature_cols, categorical_cols, target_col = select_features_target(df, numeric_cols)
    
    if not feature_cols:
        print("No hay suficientes características para entrenar un modelo.")
        return
    
    # Preparar datos para el modelo
    X = df[feature_cols + categorical_cols].copy()
    y = df[target_col].copy()
    
    # Eliminar filas con valores nulos en la variable objetivo
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    if len(X) < 10:
        print("No hay suficientes datos para entrenar un modelo después de eliminar valores nulos.")
        return
    
    print(f"\nEntrenando modelos para predecir: {target_col}")
    print(f"Número de muestras: {len(X)}")
    
    # Crear pipeline de preprocesamiento
    preprocessor = create_preprocessing_pipeline(feature_cols, categorical_cols)
    
    # Preprocesar datos
    X_processed = preprocessor.fit_transform(X)
    
    # Entrenar y evaluar modelos
    results, best_model_name = train_evaluate_models(X_processed, y)
    
    # Optimizar el mejor modelo
    optimized_model = optimize_best_model(best_model_name, X, y, preprocessor)
    
    # Evaluar modelo optimizado
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = optimized_model.predict(X_test)
    
    # Calcular métricas finales
    final_mse = mean_squared_error(y_test, y_pred)
    final_rmse = np.sqrt(final_mse)
    final_mae = mean_absolute_error(y_test, y_pred)
    final_r2 = r2_score(y_test, y_pred)
    
    print("\n=== Resultados del Modelo Optimizado ===")
    print(f"RMSE: {final_rmse:.4f}")
    print(f"MAE: {final_mae:.4f}")
    print(f"R²: {final_r2:.4f}")
    
    # Visualizar resultados
    visualize_results(results, y_test, y_pred, target_col)
    
    # Guardar modelo
    save_model(optimized_model, feature_cols, categorical_cols, target_col)
    
    print("\n=== Análisis Predictivo Completado ===")

if __name__ == "__main__":
    main()