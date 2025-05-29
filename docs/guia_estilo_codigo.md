# Guía de Estilo de Código

Esta guía establece las convenciones de estilo para el código del proyecto de Análisis de Datos y Visualización. Seguir estas pautas ayudará a mantener la coherencia y legibilidad del código.

## Convenciones Generales

### Formato de Código

- Utiliza **4 espacios** para la indentación (no tabulaciones).
- Limita las líneas a un máximo de **88 caracteres** (compatible con Black).
- Utiliza líneas en blanco para separar funciones y clases, así como bloques lógicos dentro de funciones.
- Utiliza espacios alrededor de operadores y después de comas.
- Utiliza codificación UTF-8 para todos los archivos Python.

### Convenciones de Nomenclatura

- **Funciones y Variables**: Utiliza `snake_case` (minúsculas con guiones bajos).
- **Clases**: Utiliza `CamelCase` (cada palabra comienza con mayúscula, sin guiones bajos).
- **Constantes**: Utiliza `MAYUSCULAS_CON_GUIONES_BAJOS`.
- **Nombres de archivos**: Utiliza `snake_case` para los nombres de archivos Python.

### Importaciones

- Organiza las importaciones en bloques separados:
  1. Bibliotecas estándar de Python
  2. Bibliotecas de terceros
  3. Módulos locales del proyecto
- Dentro de cada bloque, ordena las importaciones alfabéticamente.

```python
# Ejemplo de organización de importaciones
# Bibliotecas estándar
import os
import sys
from datetime import datetime

# Bibliotecas de terceros
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Módulos locales
from src.utils import helpers
```

## Documentación

### Docstrings

Utiliza el formato de Google para los docstrings:

```python
def funcion_ejemplo(param1, param2):
    """
    Descripción breve de la función.
    
    Descripción más detallada si es necesario.
    
    Args:
        param1 (tipo): Descripción del parámetro 1.
        param2 (tipo): Descripción del parámetro 2.
    
    Returns:
        tipo: Descripción del valor de retorno.
    
    Raises:
        ExcepcionTipo: Descripción de cuándo se lanza esta excepción.
    
    Examples:
        >>> funcion_ejemplo(1, 2)
        3
    """
    # Código de la función
```

### Comentarios

- Utiliza comentarios para explicar el "por qué" y no el "qué" o "cómo".
- Mantén los comentarios actualizados cuando el código cambie.
- Escribe comentarios completos en forma de frases con la primera letra en mayúscula y un punto al final.

## Prácticas de Codificación

### Manejo de Errores

- Utiliza bloques try/except para manejar excepciones esperadas.
- Especifica las excepciones que esperas capturar en lugar de usar un bloque except genérico.
- Proporciona mensajes de error informativos.

```python
try:
    df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv(file_path, delimiter=delimiter, encoding='latin1')
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None
```

### Funciones

- Las funciones deben hacer una sola cosa y hacerla bien.
- Limita la longitud de las funciones a 50 líneas cuando sea posible.
- Utiliza nombres descriptivos que indiquen lo que hace la función.

### Variables

- Utiliza nombres descriptivos para las variables.
- Evita nombres de una sola letra excepto para contadores o iteradores.
- Utiliza nombres en español para mantener la coherencia con el resto del proyecto.

## Análisis de Datos y Visualización

### Pandas

- Utiliza la notación de punto para acceder a columnas (`df.columna`) cuando sea posible.
- Utiliza la notación de corchetes (`df['columna']`) cuando los nombres de columnas tengan espacios o caracteres especiales.
- Encadena operaciones de pandas cuando sea apropiado para mejorar la legibilidad.

### Visualizaciones

- Incluye siempre títulos, etiquetas de ejes y leyendas en las visualizaciones.
- Utiliza una paleta de colores coherente en todo el proyecto.
- Guarda las visualizaciones con nombres descriptivos y en formatos de alta calidad (PNG o SVG).

```python
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='columna_numerica', kde=True)
plt.title('Distribución de Valores')
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.savefig('distribucion_valores.png', dpi=300, bbox_inches='tight')
```

## Control de Versiones

### Mensajes de Commit

- Escribe mensajes de commit claros y descriptivos.
- Utiliza el imperativo presente: "Añade función" en lugar de "Añadida función".
- Estructura los mensajes con una línea de resumen (50 caracteres o menos) seguida de una línea en blanco y una descripción más detallada si es necesario.

## Herramientas Recomendadas

- **Formateo**: Black
- **Linting**: Flake8
- **Type Checking**: mypy
- **Documentación**: Sphinx

Seguir estas pautas ayudará a mantener un código limpio, coherente y fácil de mantener para todos los colaboradores del proyecto.