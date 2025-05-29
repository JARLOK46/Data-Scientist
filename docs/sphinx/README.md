# Documentación con Sphinx

Este directorio contiene la documentación generada con Sphinx para el proyecto de Análisis de Datos y Visualización.

## Estructura

- `source/`: Contiene los archivos fuente de la documentación
  - `conf.py`: Archivo de configuración de Sphinx
  - `index.rst`: Página principal de la documentación
  - `modules/`: Documentación manual de los módulos
  - `api/`: Documentación generada automáticamente con sphinx-apidoc
- `build/`: Contiene la documentación generada
  - `html/`: Documentación en formato HTML

## Cómo actualizar la documentación

1. Asegúrate de tener Sphinx instalado:
   ```
   pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
   ```

2. Actualiza los docstrings en los archivos Python del proyecto siguiendo el formato de Google o NumPy.

3. Regenera la documentación automática:
   ```
   cd docs/sphinx
   sphinx-apidoc -o source/api ../../src -f
   ```

4. Genera la documentación HTML:
   ```
   cd docs/sphinx
   .\make html
   ```

5. Abre `build/html/index.html` en un navegador para ver la documentación.

## Mejores prácticas para los docstrings

Para que la documentación generada sea útil, sigue estas pautas al escribir docstrings:

- Usa el formato de Google o NumPy para los docstrings.
- Documenta todos los parámetros, valores de retorno y excepciones.
- Incluye ejemplos de uso cuando sea apropiado.
- Añade type hints a las funciones y métodos.

Ejemplo de docstring en formato Google:

```python
def mi_funcion(param1, param2):
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
        >>> mi_funcion(1, 2)
        3
    """
    # Código de la función
```