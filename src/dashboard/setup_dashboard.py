"""
Módulo de configuración del dashboard interactivo.

Este script instala y verifica las dependencias necesarias para ejecutar el dashboard interactivo del proyecto. Permite comprobar la versión de Python, la presencia de pip y la instalación de los paquetes requeridos, además de lanzar el dashboard si el usuario lo desea.

Flujo principal:
- Verificación de versión de Python y pip.
- Comprobación e instalación de dependencias desde requirements_dashboard.txt.
- Opción para ejecutar el dashboard interactivo tras la instalación.

Dependencias principales: os, sys, subprocess, platform.

Estructura del archivo:
- Funciones utilitarias para verificación e instalación de dependencias.
- Función principal (main) que orquesta todo el proceso.

Autor: Anderson Zapata
Fecha: 2025
"""
import os
import sys
import subprocess
import platform

def check_python_version():
    """
    Verifica que la versión de Python sea compatible con el dashboard interactivo.

    Retorna
    -------
    bool
        True si la versión es compatible, False en caso contrario.

    Advertencias
    ------------
    - Requiere Python 3.7 o superior.
    - Si la versión es inferior, el proceso se detiene.

    Ejemplo
    -------
    >>> check_python_version()
    True

    Lógica Interna
    --------------
    1. Obtiene la versión actual de Python.
    2. Verifica si cumple con el requisito mínimo.
    3. Imprime mensajes informativos y retorna el resultado.

    Autor: Anderson Zapata
    Fecha: 2025
    """
    print("Verificando versión de Python...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"Error: Se requiere Python 3.7 o superior. Versión actual: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✓ Versión de Python compatible: {version.major}.{version.minor}.{version.micro}")
    return True

def check_pip():
    """
    Verifica que pip esté instalado en el entorno actual de Python.

    Retorna
    -------
    bool
        True si pip está instalado, False en caso contrario.

    Advertencias
    ------------
    - Si pip no está instalado o no está en el PATH, el proceso se detiene.

    Ejemplo
    -------
    >>> check_pip()
    True

    Lógica Interna
    --------------
    1. Ejecuta 'python -m pip --version' para comprobar la instalación.
    2. Retorna True si no hay errores, False si falla.

    Autor: Anderson Zapata
    Fecha: 2025
    """
    print("Verificando instalación de pip...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✓ pip está instalado")
        return True
    except subprocess.CalledProcessError:
        print("Error: pip no está instalado o no está en el PATH")
        return False

def install_requirements(requirements_file):
    """
    Instala las dependencias desde un archivo requirements.txt.

    Parámetros
    ----------
    requirements_file : str
        Ruta al archivo requirements.txt que contiene las dependencias.

    Retorna
    -------
    bool
        True si la instalación fue exitosa, False si hubo errores.

    Advertencias
    ------------
    - El archivo debe existir y ser accesible.
    - Si alguna dependencia falla, se informa al usuario.

    Ejemplo
    -------
    >>> install_requirements('requirements_dashboard.txt')
    True

    Lógica Interna
    --------------
    1. Verifica la existencia del archivo.
    2. Ejecuta 'pip install -r requirements.txt'.
    3. Retorna el resultado de la instalación.

    Autor: Anderson Zapata
    Fecha: 2025
    """
    if not os.path.exists(requirements_file):
        print(f"Error: No se encontró el archivo {requirements_file}")
        return False
    
    print(f"Instalando dependencias desde {requirements_file}...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_file], check=True)
        print("✓ Dependencias instaladas correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error al instalar dependencias: {e}")
        return False

def check_dependencies(dependencies):
    """
    Verifica si las dependencias especificadas están instaladas en el entorno actual.

    Parámetros
    ----------
    dependencies : list of str
        Lista de nombres de paquetes a verificar.

    Retorna
    -------
    list
        Lista de dependencias que no están instaladas.

    Advertencias
    ------------
    - Si una dependencia no está instalada, se informa al usuario.

    Ejemplo
    -------
    >>> check_dependencies(['dash', 'plotly'])
    []

    Lógica Interna
    --------------
    1. Intenta importar cada dependencia.
    2. Si falla, la agrega a la lista de faltantes.
    3. Retorna la lista de dependencias faltantes.

    Autor: Anderson Zapata
    Fecha: 2025
    """
    print("Verificando dependencias instaladas...")
    missing = []
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep} está instalado")
        except ImportError:
            print(f"✗ {dep} no está instalado")
            missing.append(dep)
    return missing

def main():
    """
    Función principal que orquesta la configuración del dashboard interactivo.

    Retorna
    -------
    bool
        True si la configuración se completó correctamente, False si hubo errores.

    Advertencias
    ------------
    - El usuario debe tener permisos para instalar paquetes.
    - Si la versión de Python o pip no es compatible, el proceso se detiene.
    - Si faltan dependencias, se ofrece instalarlas.

    Ejemplo
    -------
    >>> main()

    Lógica Interna
    --------------
    1. Verifica versión de Python y pip.
    2. Comprueba e instala dependencias si es necesario.
    3. Ofrece ejecutar el dashboard interactivo.
    4. Informa al usuario sobre el estado de la configuración.

    Autor: Anderson Zapata
    Fecha: 2025
    """
    print("=== Configuración del Dashboard Interactivo ===")
    
    # Verificar versión de Python
    if not check_python_version():
        print("Por favor, actualice Python a la versión 3.7 o superior.")
        return False
    
    # Verificar pip
    if not check_pip():
        print("Por favor, instale pip para continuar.")
        return False
    
    # Verificar dependencias existentes
    dependencies = ["dash", "plotly", "pandas", "numpy"]
    missing_deps = check_dependencies(dependencies)
    
    # Instalar dependencias si faltan
    if missing_deps:
        print(f"\nSe necesitan instalar {len(missing_deps)} dependencias: {', '.join(missing_deps)}")
        requirements_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements_dashboard.txt")
        
        user_input = input("¿Desea instalar las dependencias ahora? (s/n): ").lower()
        if user_input == 's':
            if install_requirements(requirements_file):
                print("\nTodas las dependencias están instaladas correctamente.")
            else:
                print("\nNo se pudieron instalar todas las dependencias.")
                print("Intente instalarlas manualmente con: pip install -r requirements_dashboard.txt")
                return False
        else:
            print("\nPuede instalar las dependencias más tarde con: pip install -r requirements_dashboard.txt")
            return False
    else:
        print("\nTodas las dependencias necesarias ya están instaladas.")
    
    # Ofrecer ejecutar el dashboard
    print("\nConfiguración completada. Ahora puede ejecutar el dashboard interactivo.")
    user_input = input("¿Desea ejecutar el dashboard ahora? (s/n): ").lower()
    if user_input == 's':
        dashboard_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_interactivo.py")
        print("\nIniciando el dashboard interactivo...")
        print("Acceda a http://127.0.0.1:8050/ en su navegador para ver el dashboard.")
        print("Presione Ctrl+C para detener el dashboard cuando termine.")
        try:
            subprocess.run([sys.executable, dashboard_file])
        except KeyboardInterrupt:
            print("\nDashboard detenido.")
    else:
        print("\nPuede ejecutar el dashboard más tarde con: python dashboard_interactivo.py")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n=== Configuración completada con éxito ===")
        else:
            print("\n=== La configuración no se completó correctamente ===")
    except Exception as e:
        print(f"\nError inesperado: {e}")
        print("La configuración no se completó correctamente.")
    
    # Mantener la ventana abierta en Windows
    if platform.system() == "Windows":
        input("\nPresione Enter para salir...")