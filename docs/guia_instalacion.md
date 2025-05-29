# Guía de Instalación

Este documento proporciona instrucciones detalladas para instalar y configurar el entorno necesario para ejecutar el sistema de análisis de datos.

## Requisitos del Sistema

### Requisitos Mínimos

- **Sistema Operativo**: Windows 10/11, macOS 10.14+, o Linux (Ubuntu 18.04+, CentOS 7+)
- **Procesador**: Intel Core i3 / AMD Ryzen 3 o superior
- **Memoria RAM**: 8 GB (16 GB recomendado para análisis de conjuntos de datos grandes)
- **Espacio en Disco**: 2 GB para la instalación básica + espacio para datos
- **Python**: Versión 3.7 o superior

### Requisitos Recomendados

- **Procesador**: Intel Core i5/i7 o AMD Ryzen 5/7
- **Memoria RAM**: 16 GB o más
- **Espacio en Disco**: 5 GB + espacio para datos
- **Tarjeta Gráfica**: No es esencial, pero puede acelerar ciertos análisis predictivos

## Instalación de Python

### Windows

1. **Descargar Python**:
   - Visita [python.org](https://www.python.org/downloads/)
   - Descarga la última versión de Python 3.x (3.7 o superior)

2. **Instalar Python**:
   - Ejecuta el instalador descargado
   - **¡IMPORTANTE!** Marca la casilla "Add Python to PATH"
   - Selecciona "Install Now" para una instalación estándar

3. **Verificar la instalación**:
   - Abre PowerShell o Símbolo del sistema
   - Ejecuta: `python --version`
   - Deberías ver la versión de Python instalada

### macOS

1. **Usando Homebrew** (recomendado):
   ```bash
   # Instalar Homebrew si no está instalado
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Instalar Python
   brew install python
   ```

2. **Desde python.org**:
   - Visita [python.org](https://www.python.org/downloads/)
   - Descarga e instala la última versión de Python 3.x

3. **Verificar la instalación**:
   ```bash
   python3 --version
   ```

### Linux

1. **Ubuntu/Debian**:
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-venv
   ```

2. **CentOS/RHEL**:
   ```bash
   sudo yum install python3 python3-pip
   ```

3. **Verificar la instalación**:
   ```bash
   python3 --version
   ```

## Configuración del Entorno Virtual

Se recomienda utilizar entornos virtuales para aislar las dependencias del proyecto.

### Windows

```powershell
# Navegar al directorio del proyecto
cd "c:\ruta\a\UN PASO AL EXITO"

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
.\venv\Scripts\activate
```

### macOS/Linux

```bash
# Navegar al directorio del proyecto
cd /ruta/a/UN\ PASO\ AL\ EXITO

# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
source venv/bin/activate
```

## Instalación de Dependencias Básicas

Una vez activado el entorno virtual, instala las dependencias básicas:

```bash
# Actualizar pip
pip install --upgrade pip

# Instalar dependencias básicas
pip install pandas numpy matplotlib seaborn
```

## Instalación de Dependencias Específicas

### Para el Dashboard Interactivo

```bash
pip install -r config/requirements_dashboard.txt
```

O ejecutar el script de configuración automática:

```bash
python src/dashboard/setup_dashboard.py
```

### Para el Análisis Geoespacial

```bash
pip install -r config/requirements_geoespacial.txt
```

## Instalación de Sphinx para Documentación

Para generar o actualizar la documentación:

```bash
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
```

## Verificación de la Instalación

Para verificar que todo está correctamente instalado y configurado:

```bash
# Ejecutar el script de utilidades
python src/utils/ejecutar_analisis.py
```

Deberías ver el menú principal del sistema de análisis de datos.

## Solución de Problemas Comunes

### Python no es reconocido como un comando

**Windows**:
1. Verifica que Python esté en el PATH:
   - Busca "Variables de entorno" en el menú de inicio
   - En Variables de sistema, edita la variable PATH
   - Asegúrate de que la ruta a Python (ej. C:\Python39) esté incluida

**macOS/Linux**:
1. Verifica la ubicación de Python:
   ```bash
   which python3
   ```
2. Añade la ruta a tu archivo .bashrc o .zshrc si es necesario

### Error al instalar dependencias

1. Asegúrate de que pip esté actualizado:
   ```bash
   pip install --upgrade pip
   ```

2. Si hay errores con paquetes que requieren compilación:
   - **Windows**: Instala Visual C++ Build Tools
   - **macOS**: Instala Xcode Command Line Tools
   - **Linux**: Instala paquetes de desarrollo esenciales
     ```bash
     # Ubuntu/Debian
     sudo apt install build-essential python3-dev
     ```

### Problemas con el Dashboard Interactivo

1. Verifica que todas las dependencias estén instaladas:
   ```bash
   pip install dash plotly pandas
   ```

2. Comprueba si hay puertos en uso:
   - El dashboard utiliza el puerto 8050 por defecto
   - Si está en uso, modifica el código en `dashboard_interactivo.py` para usar otro puerto

### Problemas con el Análisis Geoespacial

1. Para problemas con geopandas:
   ```bash
   # Windows (usando conda, recomendado para geopandas)
   conda install -c conda-forge geopandas
   
   # Linux
   sudo apt install libgdal-dev
   pip install geopandas
   ```

## Configuración Avanzada

### Configuración para Grandes Conjuntos de Datos

Si trabajas con conjuntos de datos muy grandes, considera estas optimizaciones:

1. **Aumentar la memoria disponible para Python**:
   - En Windows, crea un archivo `.bat` con:
     ```
     @echo off
     set PYTHONMEMORY=4096
     python src/utils/ejecutar_analisis.py
     ```

2. **Usar procesamiento en paralelo**:
   - Instala bibliotecas para procesamiento paralelo:
     ```bash
     pip install joblib dask[complete]
     ```

### Configuración para Desarrollo

Si vas a contribuir al desarrollo del proyecto:

1. **Instalar herramientas de desarrollo**:
   ```bash
   pip install pytest flake8 black isort mypy
   ```

2. **Configurar pre-commit hooks** (opcional):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Próximos Pasos

Una vez completada la instalación, consulta los siguientes recursos:

- [Guía de Inicio Rápido](docs/README.md) - Para comenzar a usar el sistema
- [Casos de Uso](docs/casos_uso.md) - Para ejemplos prácticos
- [Arquitectura del Proyecto](docs/arquitectura_proyecto.md) - Para entender la estructura

---

Si encuentras problemas durante la instalación, consulta la sección de [Preguntas Frecuentes](docs/preguntas_frecuentes.md) o crea un issue en el repositorio del proyecto.