"""
Módulo de análisis de contenido web.

Este script permite analizar columnas de URLs, extraer dominios, limpiar y visualizar información relevante de contenido web en un DataFrame.

Autor: Anderson Zapata
Fecha: 2025
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
import re
import os
import json
import sys
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import warnings

# Añadir el directorio raíz del proyecto al path para poder importar módulos
proyecto_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, proyecto_dir)

# Suprimir advertencias
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Crear directorio para resultados si no existe
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analisis_web')
os.makedirs(results_dir, exist_ok=True)

# Descargar recursos de NLTK si no están disponibles
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Función para detectar el delimitador
def detect_delimiter(file_path, num_lines=5):
    """
    Detecta automáticamente el delimitador utilizado en un archivo CSV.
    
    Esta función analiza las primeras líneas de un archivo para determinar
    qué delimitador se utiliza con mayor frecuencia entre tabulaciones, comas,
    puntos y comas, y barras verticales.
    
    Args:
        file_path (str): Ruta al archivo CSV a analizar.
        num_lines (int, optional): Número de líneas a analizar. Por defecto es 5.
    
    Returns:
        str: El delimitador detectado (tab, coma, punto y coma o pipe).
    
    Nota:
        La función maneja errores de codificación utilizando 'ignore' para evitar
        problemas con caracteres especiales.
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        sample = ''.join([file.readline() for _ in range(num_lines)])
    
    delimiters = {'\t': 0, ',': 0, ';': 0, '|': 0}
    for delimiter in delimiters:
        delimiters[delimiter] = sample.count(delimiter)
    
    return max(delimiters, key=delimiters.get)

# Cargar datos
def load_data(file_path):
    """
    Carga datos desde un archivo CSV para análisis de contenido web.
    
    Esta función detecta automáticamente el delimitador del archivo, intenta cargarlo
    con diferentes codificaciones (utf-8 y latin1) y realiza una limpieza básica de los
    nombres de columnas.
    
    Args:
        file_path (str): Ruta al archivo CSV con los datos.
    
    Returns:
        pandas.DataFrame: DataFrame con los datos cargados y nombres de columnas normalizados.
                         Retorna None si no se puede cargar el archivo.
    
    Nota:
        Los nombres de columnas se convierten a minúsculas y los espacios se reemplazan por guiones bajos.
        La función intenta diferentes codificaciones para manejar archivos con caracteres especiales.
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
    
    return df

# Extraer y analizar URLs
def analyze_urls(df):
    """
    Identifica y analiza URLs presentes en un DataFrame.
    
    Esta función examina todas las columnas de texto en el DataFrame para detectar aquellas
    que contienen URLs. Para cada columna identificada, extrae y analiza información como:
    dominios, extensiones de archivo y parámetros de consulta.
    
    Args:
        df (pandas.DataFrame): DataFrame que contiene los datos a analizar.
    
    Returns:
        tuple: Una tupla con dos elementos:
            - list: Lista de nombres de columnas que contienen URLs.
            - dict: Diccionario con los datos extraídos de las URLs, organizado por columna.
                    Incluye URLs válidas, dominios, extensiones y parámetros de consulta.
                    Retorna (None, None) si no se encuentran columnas con URLs.
    
    Nota:
        La función considera que una columna contiene URLs si al menos el 10% de las muestras
        analizadas contienen patrones de URL válidos. Se analizan hasta 100 muestras por columna.
    """
    print("\nAnalizando URLs en el dataset...")
    
    # Identificar columnas que podrían contener URLs
    text_cols = df.select_dtypes(include=['object']).columns
    url_cols = []
    
    for col in text_cols:
        # Verificar si la columna contiene URLs
        serie = df[col].dropna().astype(str)
        sample = serie.sample(min(100, len(serie)), replace=False).tolist()
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        
        url_count = sum(1 for s in sample if url_pattern.search(s))
        if url_count > len(sample) * 0.1:  # Si más del 10% de las muestras contienen URLs
            url_cols.append(col)
            print(f"  Columna con URLs detectada: {col}")
    
    if not url_cols:
        print("  No se detectaron columnas con URLs.")
        return None, None
    
    # Extraer información de las URLs
    url_data = {}
    
    for col in url_cols:
        print(f"\nAnalizando URLs en columna: {col}")
        
        # Extraer URLs válidas
        urls = df[col].dropna().astype(str).tolist()
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        valid_urls = [url_pattern.search(url).group(0) for url in urls if url_pattern.search(url)]
        
        if not valid_urls:
            print(f"  No se encontraron URLs válidas en la columna {col}")
            continue
        
        print(f"  URLs válidas encontradas: {len(valid_urls)}")
        
        # Extraer dominios
        domains = [urlparse(url).netloc for url in valid_urls]
        domain_counts = Counter(domains)
        top_domains = domain_counts.most_common(10)
        
        print("  Top 10 dominios:")
        for domain, count in top_domains:
            print(f"    {domain}: {count}")
        
        # Extraer extensiones de archivo
        extensions = [os.path.splitext(urlparse(url).path)[1] for url in valid_urls]
        extension_counts = Counter([ext for ext in extensions if ext])
        top_extensions = extension_counts.most_common(5)
        
        if top_extensions:
            print("  Top 5 extensiones de archivo:")
            for ext, count in top_extensions:
                print(f"    {ext}: {count}")
        
        # Extraer parámetros de consulta
        query_params = []
        for url in valid_urls:
            parsed_url = urlparse(url)
            if parsed_url.query:
                params = parsed_url.query.split('&')
                for param in params:
                    if '=' in param:
                        key = param.split('=')[0]
                        query_params.append(key)
        
        param_counts = Counter(query_params)
        top_params = param_counts.most_common(5)
        
        if top_params:
            print("  Top 5 parámetros de consulta:")
            for param, count in top_params:
                print(f"    {param}: {count}")
        
        # Guardar datos para visualización
        url_data[col] = {
            'valid_urls': valid_urls,
            'domains': domains,
            'domain_counts': dict(domain_counts),
            'extensions': extensions,
            'extension_counts': dict(extension_counts),
            'query_params': query_params,
            'param_counts': dict(param_counts)
        }
    
    return url_cols, url_data

# Extraer y analizar texto de URLs (simulado)
def analyze_url_content(url_data):
    """
    Simula la extracción y análisis de contenido textual de URLs.
    
    Esta función crea un corpus de texto simulado basado en los dominios de las URLs
    identificadas previamente. En un entorno real, esta función realizaría web scraping
    para extraer el contenido real de las páginas web.
    
    Args:
        url_data (dict): Diccionario con datos de URLs organizados por columna,
                         generado por la función analyze_urls.
    
    Returns:
        dict: Diccionario con el corpus simulado para cada columna que contiene URLs.
              Cada entrada contiene:
              - 'corpus': Lista de textos simulados.
              - 'domains': Lista de dominios correspondientes a cada texto.
    
    Nota:
        Esta es una función de simulación. En un entorno de producción, se reemplazaría
        con código real de web scraping utilizando bibliotecas como requests y BeautifulSoup.
    """
    print("\nAnalizando contenido de URLs (simulado)...")
    print("Nota: En un entorno real, este proceso implicaría scraping web para extraer el contenido real de las URLs.")
    
    # Simular extracción de texto de URLs
    # En un entorno real, aquí se utilizaría requests, BeautifulSoup, etc. para extraer contenido
    
    # Crear datos simulados basados en los dominios
    content_data = {}
    
    for col, data in url_data.items():
        print(f"\nAnalizando contenido para URLs en columna: {col}")
        
        # Crear un corpus simulado basado en los dominios
        corpus = []
        for domain in data['domains']:
            # Generar texto simulado basado en el dominio
            domain_parts = domain.split('.')
            base_domain = domain_parts[0] if domain_parts[0] != 'www' else domain_parts[1]
            
            # Simular texto relacionado con el dominio
            simulated_text = f"Este es un contenido simulado para {domain}. "
            simulated_text += f"El sitio {base_domain} contiene información relevante sobre tecnología, "
            simulated_text += f"noticias, datos, análisis y contenido web. "
            simulated_text += f"Los usuarios de {domain} pueden encontrar recursos útiles y actualizaciones."
            
            corpus.append(simulated_text)
        
        # Guardar corpus simulado
        content_data[col] = {
            'corpus': corpus,
            'domains': data['domains']
        }
        
        print(f"  Corpus simulado creado con {len(corpus)} documentos")
    
    return content_data

# Preprocesar texto para análisis
def preprocess_text(corpus):
    """
    Preprocesa un corpus de textos para análisis de lenguaje natural.
    
    Esta función realiza varias operaciones de limpieza y normalización en cada documento:
    - Conversión a minúsculas
    - Eliminación de caracteres especiales y números
    - Tokenización
    - Eliminación de stopwords (en inglés y español)
    - Lematización
    - Filtrado de tokens cortos (menos de 3 caracteres)
    
    Args:
        corpus (list): Lista de documentos de texto a preprocesar.
    
    Returns:
        list: Lista de documentos preprocesados, donde cada documento es una cadena
              de tokens lematizados separados por espacios.
    
    Nota:
        Esta función utiliza NLTK para la tokenización, eliminación de stopwords y lematización.
        Requiere que los recursos de NLTK 'punkt', 'stopwords' y 'wordnet' estén descargados.
    """
    # Inicializar lematizador y lista de stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english') + stopwords.words('spanish'))
    
    processed_corpus = []
    
    for doc in tqdm(corpus, desc="Preprocesando textos"):
        # Convertir a minúsculas
        doc = doc.lower()
        
        # Eliminar caracteres especiales y números
        doc = re.sub(r'[^\w\s]', '', doc)
        doc = re.sub(r'\d+', '', doc)
        
        # Tokenizar
        tokens = word_tokenize(doc)
        
        # Eliminar stopwords y lematizar
        filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
        
        # Unir tokens nuevamente
        processed_doc = ' '.join(filtered_tokens)
        processed_corpus.append(processed_doc)
    
    return processed_corpus

# Realizar análisis de temas con LDA
def topic_modeling(corpus, n_topics=5):
    """
    Realiza modelado de temas en un corpus de texto utilizando Latent Dirichlet Allocation (LDA).
    
    Esta función vectoriza el corpus utilizando CountVectorizer y luego aplica
    el algoritmo LDA para identificar temas latentes en los documentos. Para cada tema,
    extrae las 10 palabras más representativas junto con sus pesos.
    
    Args:
        corpus (list): Lista de documentos de texto preprocesados.
        n_topics (int, optional): Número de temas a identificar. Por defecto es 5.
    
    Returns:
        tuple: Una tupla con tres elementos:
            - list: Lista de diccionarios, cada uno representando un tema con sus palabras
                   y pesos asociados.
            - LatentDirichletAllocation: El modelo LDA entrenado.
            - CountVectorizer: El vectorizador utilizado para transformar el corpus.
    
    Nota:
        La función utiliza parámetros predeterminados para el vectorizador:
        - max_df=0.95: Ignora términos que aparecen en más del 95% de los documentos
        - min_df=2: Ignora términos que aparecen en menos de 2 documentos
        - max_features=1000: Limita el vocabulario a 1000 términos
    """
    # Vectorizar el corpus
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000)
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    
    # Aplicar LDA
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    
    # Extraer temas
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-11:-1]  # Top 10 palabras
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append({
            'id': topic_idx,
            'words': top_words,
            'weights': topic[top_words_idx].tolist()
        })
    
    return topics, lda, vectorizer

# Función para sanear nombres de archivo
def sanitize_filename(name):
    """
    Sanea un nombre de archivo eliminando caracteres no permitidos.

    Esta función reemplaza cualquier carácter que no sea alfanumérico, guión bajo o guión
    con un guión bajo, generando un nombre de archivo seguro para usar en cualquier sistema
    de archivos.

    Parameters
    ----------
    name : str
        El nombre de archivo a sanear.

    Returns
    -------
    str
        El nombre de archivo saneado, conteniendo solo caracteres alfanuméricos,
        guiones bajos y guiones.

    Examples
    --------
    >>> sanitize_filename("my file (1).txt")
    'my_file__1_.txt'
    >>> sanitize_filename("data$analysis#2023")
    'data_analysis_2023'
    """
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)

# Visualizar resultados del análisis de URLs
def visualize_url_analysis(url_cols, url_data):
    """
    Genera y guarda visualizaciones para el análisis de URLs.
    
    Esta función crea diferentes tipos de visualizaciones para cada columna que contiene URLs:
    - Gráfico de barras de los 15 dominios más comunes
    - Nube de palabras de dominios
    - Gráfico de barras de las extensiones de archivo más comunes (si hay suficientes)
    - Gráfico de barras de los parámetros de consulta más comunes (si hay suficientes)
    
    Args:
        url_cols (list): Lista de nombres de columnas que contienen URLs.
        url_data (dict): Diccionario con datos de URLs organizados por columna,
                         generado por la función analyze_urls.
    
    Returns:
        None: Las visualizaciones se guardan como archivos PNG en el directorio de resultados.
    
    Nota:
        Las visualizaciones se guardan en el directorio especificado por la variable global
        'results_dir'. Los nombres de archivo incluyen el nombre de la columna con espacios
        reemplazados por guiones bajos.
    """
    print("\nGenerando visualizaciones del análisis de URLs...")
    
    for col in url_cols:
        data = url_data[col]
        safe_col = sanitize_filename(col)
        
        # Visualizar dominios más comunes
        plt.figure(figsize=(12, 6))
        domain_df = pd.DataFrame(list(data['domain_counts'].items()), columns=['Domain', 'Count'])
        domain_df = domain_df.sort_values('Count', ascending=False).head(15)
        
        sns.barplot(x='Count', y='Domain', data=domain_df)
        plt.title(f'Top 15 Dominios en {col}')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'top_domains_{safe_col}.png'))
        print(f"  Gráfico guardado: {os.path.join(results_dir, f'top_domains_{safe_col}.png')}")
        
        # Crear nube de palabras de dominios
        if data['domain_counts']:
            plt.figure(figsize=(10, 8))
            wordcloud = WordCloud(width=800, height=600, background_color='white', 
                                 colormap='viridis', max_words=100).generate_from_frequencies(data['domain_counts'])
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Nube de Dominios en {col}')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'domain_wordcloud_{safe_col}.png'))
            print(f"  Gráfico guardado: {os.path.join(results_dir, f'domain_wordcloud_{safe_col}.png')}")
        
        # Visualizar extensiones de archivo si hay suficientes
        if len(data['extension_counts']) > 1:
            plt.figure(figsize=(10, 6))
            ext_df = pd.DataFrame(list(data['extension_counts'].items()), columns=['Extension', 'Count'])
            ext_df = ext_df.sort_values('Count', ascending=False).head(10)
            
            sns.barplot(x='Count', y='Extension', data=ext_df)
            plt.title(f'Extensiones de Archivo más Comunes en {col}')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'file_extensions_{safe_col}.png'))
            print(f"  Gráfico guardado: {os.path.join(results_dir, f'file_extensions_{safe_col}.png')}")
        
        # Visualizar parámetros de consulta si hay suficientes
        if len(data['param_counts']) > 1:
            plt.figure(figsize=(10, 6))
            param_df = pd.DataFrame(list(data['param_counts'].items()), columns=['Parameter', 'Count'])
            param_df = param_df.sort_values('Count', ascending=False).head(10)
            
            sns.barplot(x='Count', y='Parameter', data=param_df)
            plt.title(f'Parámetros de Consulta más Comunes en {col}')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'query_params_{safe_col}.png'))
            print(f"  Gráfico guardado: {os.path.join(results_dir, f'query_params_{safe_col}.png')}")

# Visualizar resultados del análisis de contenido
def visualize_content_analysis(content_data, topics_data):
    """
    Genera y guarda visualizaciones para el análisis de contenido de URLs.
    
    Esta función crea visualizaciones para representar los temas identificados
    en el contenido de las URLs mediante modelado de temas. Para cada columna,
    genera un gráfico de barras horizontales que muestra las palabras más relevantes
    para cada tema y sus pesos asociados.
    
    Args:
        content_data (dict): Diccionario con el corpus simulado para cada columna,
                            generado por la función analyze_url_content.
        topics_data (dict): Diccionario con los temas identificados para cada columna,
                           generado por la función topic_modeling.
    
    Returns:
        None: Las visualizaciones se guardan como archivos PNG en el directorio de resultados.
    
    Nota:
        La función está configurada para mostrar hasta 5 temas por columna en un único gráfico.
        Las visualizaciones se guardan en el directorio especificado por la variable global
        'results_dir'.
    """
    print("\nGenerando visualizaciones del análisis de contenido...")
    
    for col, topics in topics_data.items():
        safe_col = sanitize_filename(col)
        # Visualizar temas
        plt.figure(figsize=(12, 8))
        for i, topic in enumerate(topics):
            plt.subplot(3, 2, i+1 if i < 5 else 5)
            y_pos = np.arange(len(topic['words']))
            plt.barh(y_pos, topic['weights'], align='center')
            plt.yticks(y_pos, topic['words'])
            plt.title(f'Tema {i+1}')
            plt.tight_layout()
        plt.suptitle(f'Temas Identificados en {col}', fontsize=16)
        plt.subplots_adjust(top=0.9)
        plt.savefig(os.path.join(results_dir, f'topics_{safe_col}.png'))
        print(f"  Gráfico guardado: {os.path.join(results_dir, f'topics_{safe_col}.png')}")

# Generar informe de análisis
def generate_report(url_cols, url_data, topics_data):
    """
    Genera un informe detallado del análisis de contenido web en formatos JSON y Markdown.
    
    Esta función recopila los resultados del análisis de URLs y del modelado de temas
    para crear informes estructurados. El informe incluye estadísticas sobre dominios,
    extensiones de archivo, parámetros de consulta y temas identificados en el contenido.
    
    Args:
        url_cols (list): Lista de nombres de columnas que contienen URLs.
        url_data (dict): Diccionario con datos de URLs organizados por columna,
                         generado por la función analyze_urls.
        topics_data (dict): Diccionario con los temas identificados para cada columna,
                           generado por la función topic_modeling.
    
    Returns:
        tuple: Una tupla con dos elementos:
            - str: Ruta al archivo JSON del informe.
            - str: Ruta al archivo Markdown del informe.
    
    Nota:
        El informe en formato Markdown incluye secciones de resumen, análisis de contenido,
        conclusiones y recomendaciones. Ambos informes se guardan en el directorio especificado
        por la variable global 'results_dir'.
    """
    print("\nGenerando informe de análisis de contenido web...")
    
    report = {
        'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'url_columns': url_cols,
        'url_analysis': {},
        'content_analysis': {}
    }
    
    # Añadir resultados del análisis de URLs
    for col in url_cols:
        report['url_analysis'][col] = {
            'total_urls': len(url_data[col]['valid_urls']),
            'unique_domains': len(set(url_data[col]['domains'])),
            'top_domains': dict(Counter(url_data[col]['domains']).most_common(10)),
            'file_extensions': dict(Counter([ext for ext in url_data[col]['extensions'] if ext]).most_common(5)),
            'query_parameters': dict(Counter(url_data[col]['query_params']).most_common(5))
        }
    
    # Añadir resultados del análisis de contenido
    for col, topics in topics_data.items():
        report['content_analysis'][col] = {
            'num_topics': len(topics),
            'topics': topics
        }
    
    # Guardar informe como JSON
    report_path = os.path.join(results_dir, 'informe_analisis_web.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    
    print(f"Informe guardado en: {report_path}")
    
    # Generar informe en formato Markdown
    md_report = f"# Informe de Análisis de Contenido Web\n\n"
    md_report += f"**Fecha de análisis:** {report['analysis_date']}\n\n"
    
    md_report += f"## 1. Resumen del Análisis\n\n"
    md_report += f"Se analizaron URLs en {len(url_cols)} columnas del dataset.\n\n"
    
    for col in url_cols:
        md_report += f"### 1.1 Columna: {col}\n\n"
        md_report += f"- Total de URLs analizadas: {report['url_analysis'][col]['total_urls']}\n"
        md_report += f"- Dominios únicos: {report['url_analysis'][col]['unique_domains']}\n\n"
        
        md_report += f"#### Top 10 Dominios:\n\n"
        for domain, count in report['url_analysis'][col]['top_domains'].items():
            md_report += f"- {domain}: {count}\n"
        
        md_report += f"\n#### Extensiones de Archivo más Comunes:\n\n"
        for ext, count in report['url_analysis'][col]['file_extensions'].items():
            md_report += f"- {ext}: {count}\n"
        
        md_report += f"\n#### Parámetros de Consulta más Comunes:\n\n"
        for param, count in report['url_analysis'][col]['query_parameters'].items():
            md_report += f"- {param}: {count}\n"
    
    md_report += f"\n## 2. Análisis de Contenido\n\n"
    md_report += f"Se realizó un análisis de temas (topic modeling) sobre el contenido simulado de las URLs.\n\n"
    
    for col, content in report['content_analysis'].items():
        md_report += f"### 2.1 Temas en {col}\n\n"
        
        for i, topic in enumerate(content['topics']):
            md_report += f"#### Tema {i+1}:\n\n"
            md_report += f"Palabras clave: {', '.join(topic['words'][:10])}\n\n"
    
    md_report += f"\n## 3. Conclusiones y Recomendaciones\n\n"
    md_report += f"### Conclusiones:\n\n"
    md_report += f"- El análisis de URLs revela una diversidad de dominios, lo que sugiere que los datos provienen de múltiples fuentes.\n"
    md_report += f"- Los patrones en los parámetros de consulta indican posibles sistemas de seguimiento o categorización en las URLs.\n"
    md_report += f"- El análisis de temas muestra clusters de contenido relacionado que podrían indicar categorías temáticas en los datos.\n\n"
    
    md_report += f"### Recomendaciones:\n\n"
    md_report += f"- Implementar un sistema de scraping web para extraer el contenido real de las URLs y realizar un análisis más preciso.\n"
    md_report += f"- Utilizar técnicas de procesamiento de lenguaje natural más avanzadas para clasificar automáticamente el contenido.\n"
    md_report += f"- Desarrollar un sistema de monitoreo para detectar cambios en el contenido de las URLs a lo largo del tiempo.\n"
    md_report += f"- Integrar el análisis de contenido web con el análisis geoespacial para identificar patrones regionales en los temas.\n"
    
    # Guardar informe en formato Markdown
    md_report_path = os.path.join(results_dir, 'informe_analisis_web.md')
    with open(md_report_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    
    print(f"Informe en formato Markdown guardado en: {md_report_path}")
    
    return report_path, md_report_path

# Función principal
def main():
    """
    Función principal que coordina el flujo de trabajo completo del análisis de contenido web.
    
    Esta función ejecuta secuencialmente todas las etapas del análisis:
    1. Carga de datos desde el archivo CSV
    2. Análisis de URLs en el DataFrame
    3. Visualización de los resultados del análisis de URLs
    4. Análisis simulado del contenido de las URLs
    5. Preprocesamiento de texto y modelado de temas
    6. Visualización de los resultados del análisis de contenido
    7. Generación de informes en formatos JSON y Markdown
    
    Returns:
        None: Los resultados se guardan como archivos en el directorio especificado
              por la variable global 'results_dir'.
    
    Nota:
        Esta función imprime mensajes de estado durante la ejecución para informar
        sobre el progreso del análisis y la ubicación de los archivos generados.
    """
    print("=== Análisis de Contenido Web ===\n")
    
    # Cargar datos
    file_path = os.path.join(proyecto_dir, 'src', 'data', '20250525.export.CSV')
    df = load_data(file_path)
    
    if df is None:
        print("No se pudieron cargar los datos. Finalizando análisis.")
        return
    
    # Analizar URLs
    url_cols, url_data = analyze_urls(df)
    
    if not url_cols or not url_data:
        print("No se encontraron URLs para analizar. Finalizando análisis.")
        return
    
    # Visualizar resultados del análisis de URLs
    visualize_url_analysis(url_cols, url_data)
    
    # Analizar contenido de URLs (simulado)
    content_data = analyze_url_content(url_data)
    
    # Realizar análisis de temas
    topics_data = {}
    for col, data in content_data.items():
        print(f"\nRealizando análisis de temas para URLs en columna: {col}")
        
        # Preprocesar corpus
        processed_corpus = preprocess_text(data['corpus'])
        
        # Modelado de temas
        topics, lda_model, vectorizer = topic_modeling(processed_corpus)
        topics_data[col] = topics
        
        print(f"  Se identificaron {len(topics)} temas principales")
        for i, topic in enumerate(topics):
            print(f"    Tema {i+1}: {', '.join(topic['words'][:5])}...")
    
    # Visualizar resultados del análisis de contenido
    visualize_content_analysis(content_data, topics_data)
    
    # Generar informe
    report_path, md_report_path = generate_report(url_cols, url_data, topics_data)
    
    print("\n=== Análisis de Contenido Web Completado ===")
    print(f"Todos los resultados guardados en: {results_dir}")

if __name__ == "__main__":
    main()