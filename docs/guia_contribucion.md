# Guía de Contribución

Gracias por tu interés en contribuir al proyecto de Análisis de Datos y Visualización. Esta guía te ayudará a entender el proceso de contribución y los estándares que seguimos para mantener la calidad del código.

## Tabla de Contenidos

1. [Código de Conducta](#código-de-conducta)
2. [Cómo Empezar](#cómo-empezar)
3. [Flujo de Trabajo de Desarrollo](#flujo-de-trabajo-de-desarrollo)
4. [Estándares de Código](#estándares-de-código)
5. [Pruebas](#pruebas)
6. [Documentación](#documentación)
7. [Proceso de Revisión](#proceso-de-revisión)
8. [Informar Problemas](#informar-problemas)

## Código de Conducta

Este proyecto y todos sus participantes están regidos por un código de conducta que promueve un entorno abierto, respetuoso e inclusivo. Al participar, se espera que respetes este código.

Puntos clave:
- Utiliza lenguaje acogedor e inclusivo
- Respeta diferentes puntos de vista y experiencias
- Acepta críticas constructivas con gracia
- Enfócate en lo que es mejor para la comunidad
- Muestra empatía hacia otros miembros de la comunidad

## Cómo Empezar

### Configuración del Entorno

1. **Clona el repositorio**:
   ```bash
   git clone https://github.com/tu-usuario/un-paso-al-exito.git
   cd un-paso-al-exito
   ```

2. **Crea un entorno virtual**:
   ```bash
   python -m venv venv
   # En Windows
   venv\Scripts\activate
   # En macOS/Linux
   source venv/bin/activate
   ```

3. **Instala las dependencias**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Dependencias para desarrollo
   ```

4. **Configura las herramientas de pre-commit** (opcional pero recomendado):
   ```bash
   pre-commit install
   ```

### Estructura del Proyecto

Familiarízate con la estructura del proyecto:

```
UN PASO AL EXITO/
├── config/                  # Archivos de configuración
├── docs/                    # Documentación
├── src/                     # Código fuente
│   ├── analysis/            # Módulos de análisis
│   ├── dashboard/           # Componentes de dashboard
│   ├── geospatial/          # Análisis geoespacial
│   ├── utils/               # Utilidades
│   ├── visualization/       # Visualizaciones
│   └── web/                 # Análisis web
└── tests/                   # Pruebas
```

## Flujo de Trabajo de Desarrollo

### Ramas

Utilizamos un flujo de trabajo basado en ramas:

- `main`: Rama principal, siempre estable y lista para producción
- `develop`: Rama de desarrollo, integra nuevas características
- `feature/nombre-caracteristica`: Ramas para nuevas características
- `bugfix/nombre-error`: Ramas para correcciones de errores
- `docs/nombre-documentacion`: Ramas para actualizaciones de documentación

### Proceso de Contribución

1. **Crea una nueva rama** desde `develop`:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/mi-nueva-caracteristica
   ```

2. **Realiza tus cambios** siguiendo los estándares de código.

3. **Ejecuta las pruebas** para asegurarte de que todo funciona correctamente:
   ```bash
   pytest
   ```

4. **Actualiza la documentación** si es necesario.

5. **Haz commit de tus cambios** con mensajes descriptivos:
   ```bash
   git add .
   git commit -m "Añade funcionalidad para análisis de tendencias temporales"
   ```

6. **Envía tu rama** al repositorio remoto:
   ```bash
   git push origin feature/mi-nueva-caracteristica
   ```

7. **Crea un Pull Request** hacia la rama `develop`.

## Estándares de Código

Seguimos estándares específicos para mantener la coherencia y calidad del código:

### Estilo de Código

- Seguimos la [Guía de Estilo de Código](guia_estilo_codigo.md) del proyecto.
- Utilizamos [Black](https://black.readthedocs.io/) para formateo automático.
- Seguimos [PEP 8](https://www.python.org/dev/peps/pep-0008/) para convenciones de estilo Python.

### Convenciones de Nomenclatura

- **Funciones y Variables**: `snake_case`
- **Clases**: `CamelCase`
- **Constantes**: `MAYUSCULAS_CON_GUIONES_BAJOS`
- **Archivos**: `snake_case.py`

### Importaciones

Organiza las importaciones en bloques:

```python
# Bibliotecas estándar
import os
import sys

# Bibliotecas de terceros
import pandas as pd
import numpy as np

# Módulos locales
from src.utils import helpers
```

## Pruebas

Todas las contribuciones deben incluir pruebas adecuadas:

### Escribir Pruebas

Utilizamos [pytest](https://docs.pytest.org/) para las pruebas:

```python
# tests/analysis/test_limpiar_analizar_csv.py

import pytest
from src.analysis.limpiar_analizar_csv import detect_delimiter

def test_detect_delimiter_comma():
    # Configuración
    test_file = "tests/data/sample_comma.csv"
    
    # Ejecución
    result = detect_delimiter(test_file)
    
    # Verificación
    assert result == ','
```

### Ejecutar Pruebas

```bash
# Ejecutar todas las pruebas
pytest

# Ejecutar pruebas con cobertura
pytest --cov=src

# Ejecutar pruebas para un módulo específico
pytest tests/analysis/
```

## Documentación

La documentación es crucial para el proyecto:

### Docstrings

Utilizamos el formato de Google para los docstrings:

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

### Documentación de Sphinx

Para actualizar la documentación generada por Sphinx:

1. Asegúrate de que tus docstrings estén correctamente formateados.

2. Regenera la documentación automática:
   ```bash
   cd docs/sphinx
   sphinx-apidoc -o source/api ../src -f
   ```

3. Genera el HTML:
   ```bash
   make html
   ```

4. Verifica los resultados en `build/html/index.html`.

## Proceso de Revisión

Todas las contribuciones pasan por un proceso de revisión:

1. **Revisión Automatizada**: Verificación de estilo, pruebas y cobertura.

2. **Revisión de Código**: Un mantenedor revisará tu código para:
   - Funcionalidad correcta
   - Adherencia a los estándares de código
   - Calidad y claridad del código
   - Documentación adecuada
   - Pruebas suficientes

3. **Iteración**: Es posible que se soliciten cambios antes de la aprobación.

4. **Fusión**: Una vez aprobado, un mantenedor fusionará tu Pull Request.

## Informar Problemas

Si encuentras un problema o tienes una sugerencia:

1. **Verifica** si el problema ya ha sido reportado.

2. **Crea un nuevo issue** con la siguiente información:
   - Descripción clara y concisa
   - Pasos para reproducir (si es un error)
   - Comportamiento esperado vs. actual
   - Capturas de pantalla si aplica
   - Entorno (sistema operativo, versión de Python, etc.)

3. **Etiqueta** el issue apropiadamente (bug, enhancement, documentation, etc.).

### Plantilla para Reportar Problemas

```
## Descripción
[Descripción clara y concisa del problema]

## Pasos para Reproducir
1. [Primer paso]
2. [Segundo paso]
3. [Y así sucesivamente...]

## Comportamiento Esperado
[Descripción clara y concisa de lo que esperabas que sucediera]

## Comportamiento Actual
[Descripción clara y concisa de lo que sucede actualmente]

## Capturas de Pantalla
[Si aplica, añade capturas de pantalla para ayudar a explicar tu problema]

## Entorno
 - Sistema Operativo: [e.g. Windows 10, Ubuntu 20.04]
 - Versión de Python: [e.g. 3.8.5]
 - Versiones de Paquetes Relevantes: [e.g. pandas 1.3.0, numpy 1.20.3]
```

---

Gracias por contribuir a este proyecto. Tu esfuerzo ayuda a mejorar esta herramienta para todos los usuarios.