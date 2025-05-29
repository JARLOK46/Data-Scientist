import os
import shutil
import sys

def crear_carpeta(ruta):
    """Crea una carpeta si no existe."""
    if not os.path.exists(ruta):
        print(f"Creando carpeta: {ruta}")
        os.makedirs(ruta, exist_ok=True)
    else:
        print(f"La carpeta {ruta} ya existe")

def mover_archivo(origen, destino):
    """Mueve un archivo a una carpeta destino."""
    if os.path.exists(origen):
        print(f"Moviendo {origen} a {destino}")
        shutil.move(origen, destino)
    else:
        print(f"El archivo {origen} no existe")

def mover_carpeta(origen, destino):
    """Mueve el contenido de una carpeta a otra y elimina la original."""
    if os.path.exists(origen):
        print(f"Moviendo carpeta {origen} a {destino}")
        # Crear la carpeta de destino si no existe
        crear_carpeta(destino)
        # Mover el contenido
        for item in os.listdir(origen):
            s = os.path.join(origen, item)
            d = os.path.join(destino, item)
            shutil.move(s, d)
        # Eliminar la carpeta vacía original
        os.rmdir(origen)
    else:
        print(f"La carpeta {origen} no existe")

def main():
    # Directorio base del proyecto
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    
    print(f"Reorganizando proyecto en: {base_dir}")
    
    # 1. Crear estructura de carpetas
    carpetas = [
        "src",
        "src/data",
        "src/analysis",
        "src/visualization",
        "src/dashboard",
        "src/utils",
        "src/geospatial",
        "src/web",
        "src/predictive",
        "tests",
        "docs",
        "config"
    ]
    
    for carpeta in carpetas:
        crear_carpeta(carpeta)
    
    # 2. Mover archivos según su función
    
    # Archivos de datos
    archivos_datos = [
        "20250525.export.CSV",
        "datos_limpios.csv"
    ]
    
    for archivo in archivos_datos:
        mover_archivo(archivo, "src/data/")
    
    # Archivos de análisis
    archivos_analisis = [
        "analizar_csv.py",
        "limpiar_analizar_csv.py",
        "analisis_predictivo.py"
    ]
    
    for archivo in archivos_analisis:
        mover_archivo(archivo, "src/analysis/")
    
    # Archivos de visualización
    archivos_visualizacion = [
        "visualizar_datos.py"
    ]
    
    for archivo in archivos_visualizacion:
        mover_archivo(archivo, "src/visualization/")
    
    # Archivos de dashboard
    archivos_dashboard = [
        "dashboard_adaptativo.py",
        "dashboard_interactivo.py",
        "setup_dashboard.py"
    ]
    
    for archivo in archivos_dashboard:
        mover_archivo(archivo, "src/dashboard/")
    
    # Archivos geoespaciales
    archivos_geoespacial = [
        "analisis_geoespacial.py",
        "analisis_geoespacial_adaptativo.py"
    ]
    
    for archivo in archivos_geoespacial:
        mover_archivo(archivo, "src/geospatial/")
    
    # Archivos de análisis web
    archivos_web = [
        "analisis_contenido_web.py"
    ]
    
    for archivo in archivos_web:
        mover_archivo(archivo, "src/web/")
    
    # Archivos de utilidades
    archivos_utils = [
        "ejecutar_analisis.py"
    ]
    
    for archivo in archivos_utils:
        mover_archivo(archivo, "src/utils/")
    
    # Mover carpetas existentes
    carpetas_a_mover = [
        ("visualizaciones", "src/visualization/output"),
        ("dashboard", "src/dashboard/dashboard_files"),
        ("dashboard_estatico", "src/dashboard/static"),
        ("analisis_geoespacial", "src/geospatial/output")
    ]
    
    for origen, destino in carpetas_a_mover:
        mover_carpeta(origen, destino)
    
    # Archivos de configuración
    archivos_config = [
        "requirements_dashboard.txt",
        "requirements_geoespacial.txt"
    ]
    
    for archivo in archivos_config:
        mover_archivo(archivo, "config/")
    
    # Archivos de documentación
    archivos_docs = [
        "README.md",
        "explicacion_detallada.md",
        "informe_analisis_datos.md",
        "mejoras.txt"
    ]
    
    for archivo in archivos_docs:
        mover_archivo(archivo, "docs/")
    
    print("\nReorganización completada. La nueva estructura del proyecto es:\n")
    
    # Mostrar la nueva estructura
    for root, dirs, files in os.walk("."):
        level = root.replace(".", "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = " " * 4 * (level + 1)
        for f in files:
            print(f"{sub_indent}{f}")

if __name__ == "__main__":
    main()