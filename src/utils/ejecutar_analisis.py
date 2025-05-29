"""
Módulo de utilidades y menú interactivo para análisis de datos.

Este script permite ejecutar, desde un menú interactivo en consola, los distintos análisis, visualizaciones y dashboards del proyecto. Facilita la navegación y ejecución de scripts de análisis, limpieza, visualización, dashboards y generación de informes.

Flujo principal:
- Menú principal con opciones para cada tipo de análisis.
- Submenús para análisis exploratorio, dashboards, análisis geoespacial, predictivo y de contenido web.
- Ejecución de scripts externos y apertura de archivos de resultados.

Dependencias principales: os, sys, subprocess, platform, matplotlib.

Estructura del archivo:
- Funciones utilitarias para ejecución de scripts y apertura de archivos.
- Funciones de menú para cada tipo de análisis.
- Función principal (main) que orquesta la interacción con el usuario.

Autor: Anderson Zapata
Fecha: 2025
"""
import os
import sys
import subprocess
import platform
import matplotlib.pyplot as plt

# Añadir el directorio raíz del proyecto al path para poder importar módulos
proyecto_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, proyecto_dir)

def clear_screen():
    """Limpia la pantalla de la consola según el sistema operativo."""
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')

def print_header():
    """Imprime el encabezado del programa."""
    clear_screen()
    print("="*80)
    print("                     SISTEMA DE ANÁLISIS DE DATOS")
    print("                        UN PASO AL ÉXITO")
    print("="*80)
    print()

def print_menu():
    """Imprime el menú principal."""
    print("MENÚ PRINCIPAL:\n")
    print("1. Análisis Exploratorio y Visualizaciones")
    print("2. Dashboard Interactivo")
    print("3. Análisis Geoespacial")
    print("4. Análisis Predictivo")
    print("5. Análisis de Contenido Web")
    print("6. Configurar Entorno")
    print("7. Ver Informes Generados")
    print("8. Salir")
    print()

def run_script(script_path, wait=True):
    """Ejecuta un script de Python."""
    try:
        if wait:
            subprocess.run([sys.executable, script_path], check=True)
        else:
            # Para scripts que inician servidores, ejecutar sin esperar
            if platform.system() == "Windows":
                subprocess.Popen([sys.executable, script_path], 
                                creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.Popen([sys.executable, script_path])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el script: {e}")
        return False
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {script_path}")
        return False

def open_file(file_path):
    """Abre un archivo con la aplicación predeterminada."""
    try:
        if platform.system() == "Windows":
            os.startfile(file_path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", file_path], check=True)
        else:  # Linux
            subprocess.run(["xdg-open", file_path], check=True)
        return True
    except Exception as e:
        print(f"Error al abrir el archivo: {e}")
        return False

def exploratory_analysis_menu():
    """Menú para análisis exploratorio y visualizaciones."""
    while True:
        print_header()
        print("ANÁLISIS EXPLORATORIO Y VISUALIZACIONES:\n")
        print("1. Ejecutar limpieza y análisis básico")
        print("2. Generar visualizaciones estáticas")
        print("3. Generar dashboard estático con informe HTML")
        print("4. Volver al menú principal")
        print()
        
        choice = input("Seleccione una opción (1-4): ")
        
        if choice == "1":
            print("\nEjecutando limpieza y análisis básico...")
            run_script(os.path.join(proyecto_dir, "src", "analysis", "limpiar_analizar_csv.py"))
            input("\nPresione Enter para continuar...")
        
        elif choice == "2":
            print("\nGenerando visualizaciones estáticas...")
            run_script(os.path.join(proyecto_dir, "src", "visualization", "visualizar_datos.py"))
            input("\nPresione Enter para continuar...")
        
        elif choice == "3":
            print("\nGenerando dashboard estático con informe HTML...")
            run_script(os.path.join(proyecto_dir, "src", "dashboard", "dashboard_adaptativo.py"))
            
            # Preguntar si desea abrir el informe HTML
            html_path = os.path.join(proyecto_dir, "src", "dashboard", "static", "informe_analisis.html")
            if os.path.exists(html_path):
                open_html = input("\n¿Desea abrir el informe HTML generado? (s/n): ").lower()
                if open_html == "s":
                    open_file(html_path)
            
            input("\nPresione Enter para continuar...")
        
        elif choice == "4":
            break
        
        else:
            print("\nOpción no válida. Intente de nuevo.")
            input("\nPresione Enter para continuar...")

def dashboard_menu():
    """Menú para el dashboard interactivo."""
    while True:
        print_header()
        print("DASHBOARD INTERACTIVO:\n")
        print("1. Verificar e instalar dependencias")
        print("2. Ejecutar dashboard interactivo")
        print("3. Volver al menú principal")
        print()
        
        choice = input("Seleccione una opción (1-3): ")
        
        if choice == "1":
            print("\nVerificando e instalando dependencias...")
            run_script(os.path.join(proyecto_dir, "src", "dashboard", "setup_dashboard.py"))
            input("\nPresione Enter para continuar...")
        
        elif choice == "2":
            print("\nEjecutando dashboard interactivo...")
            print("Acceda a http://127.0.0.1:8050/ en su navegador para ver el dashboard.")
            print("El dashboard se ejecutará en una nueva ventana. Cierre esa ventana para detenerlo.")
            input("Presione Enter para iniciar el dashboard...")
            
            # Ejecutar sin esperar para que no bloquee este script
            run_script(os.path.join(proyecto_dir, "src", "dashboard", "dashboard_interactivo.py"), wait=False)
            input("\nPresione Enter para volver al menú...")
        
        elif choice == "3":
            break
        
        else:
            print("\nOpción no válida. Intente de nuevo.")
            input("\nPresione Enter para continuar...")

def geospatial_analysis_menu():
    """Menú para análisis geoespacial."""
    while True:
        print_header()
        print("ANÁLISIS GEOESPACIAL:\n")
        print("1. Ejecutar análisis geoespacial adaptativo (dependencias mínimas)")
        print("2. Instalar dependencias para análisis geoespacial completo")
        print("3. Ejecutar análisis geoespacial completo")
        print("4. Ver informe de análisis geoespacial")
        print("5. Volver al menú principal")
        print()
        
        choice = input("Seleccione una opción (1-5): ")
        
        if choice == "1":
            print("\nEjecutando análisis geoespacial adaptativo...")
            run_script(os.path.join(proyecto_dir, "src", "geospatial", "analisis_geoespacial_adaptativo.py"))
            input("\nPresione Enter para continuar...")
        
        elif choice == "2":
            print("\nInstalando dependencias para análisis geoespacial...")
            req_path = os.path.join(proyecto_dir, "config", "requirements_geoespacial.txt")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_path], check=True)
                print("\nDependencias instaladas correctamente.")
            except subprocess.CalledProcessError as e:
                print(f"\nError al instalar dependencias: {e}")
            
            input("\nPresione Enter para continuar...")
        
        elif choice == "3":
            print("\nEjecutando análisis geoespacial completo...")
            run_script(os.path.join(proyecto_dir, "src", "geospatial", "analisis_geoespacial.py"))
            input("\nPresione Enter para continuar...")
        
        elif choice == "4":
            report_path = os.path.join(proyecto_dir, "src", "geospatial", "output", "informe_analisis_geoespacial.md")
            if os.path.exists(report_path):
                print("\nAbriendo informe de análisis geoespacial...")
                open_file(report_path)
            else:
                print("\nEl informe no existe. Ejecute primero el análisis geoespacial.")
            
            input("\nPresione Enter para continuar...")
        
        elif choice == "5":
            break
        
        else:
            print("\nOpción no válida. Intente de nuevo.")
            input("\nPresione Enter para continuar...")

def view_reports_menu():
    """Menú para ver informes generados."""
    while True:
        print_header()
        print("INFORMES GENERADOS:\n")
        print("1. Informe general de análisis de datos")
        print("2. Informe de análisis geoespacial")
        print("3. Dashboard estático (HTML)")
        print("4. Visualizaciones generadas")
        print("5. Volver al menú principal")
        print()
        
        choice = input("Seleccione una opción (1-5): ")
        
        if choice == "1":
            report_path = os.path.join(proyecto_dir, "docs", "informe_analisis_datos.md")
            if os.path.exists(report_path):
                print("\nAbriendo informe general de análisis de datos...")
                open_file(report_path)
            else:
                print("\nEl informe no existe.")
            
            input("\nPresione Enter para continuar...")
        
        elif choice == "2":
            report_path = os.path.join(proyecto_dir, "src", "geospatial", "output", "informe_analisis_geoespacial.md")
            if os.path.exists(report_path):
                print("\nAbriendo informe de análisis geoespacial...")
                open_file(report_path)
            else:
                print("\nEl informe no existe. Ejecute primero el análisis geoespacial.")
            
            input("\nPresione Enter para continuar...")
        
        elif choice == "3":
            html_path = os.path.join(proyecto_dir, "src", "dashboard", "static", "informe_analisis.html")
            if os.path.exists(html_path):
                print("\nAbriendo dashboard estático (HTML)...")
                open_file(html_path)
            else:
                print("\nEl dashboard estático no existe. Ejecute primero el dashboard adaptativo.")
            
            input("\nPresione Enter para continuar...")
        
        elif choice == "4":
            # Mostrar submenú de visualizaciones
            while True:
                print_header()
                print("VISUALIZACIONES GENERADAS:\n")
                print("1. Visualizaciones básicas")
                print("2. Visualizaciones del dashboard estático")
                print("3. Visualizaciones geoespaciales")
                print("4. Volver al menú de informes")
                print()
                
                viz_choice = input("Seleccione una opción (1-4): ")
                
                if viz_choice == "1":
                    viz_dir = os.path.join(proyecto_dir, "src", "visualization", "output")
                    if os.path.exists(viz_dir) and os.listdir(viz_dir):
                        print("\nVisualizaciones disponibles:")
                        for i, file in enumerate(os.listdir(viz_dir), 1):
                            if file.endswith(".png"):
                                print(f"{i}. {file}")
                        
                        file_num = input("\nIngrese el número de la visualización que desea ver (0 para volver): ")
                        if file_num.isdigit() and 1 <= int(file_num) <= len(os.listdir(viz_dir)):
                            file_path = os.path.join(viz_dir, os.listdir(viz_dir)[int(file_num)-1])
                            open_file(file_path)
                    else:
                        print("\nNo hay visualizaciones básicas disponibles.")
                    
                    input("\nPresione Enter para continuar...")
                
                elif viz_choice == "2":
                    viz_dir = os.path.join(proyecto_dir, "src", "dashboard", "static")
                    if os.path.exists(viz_dir) and os.listdir(viz_dir):
                        print("\nVisualizaciones disponibles:")
                        png_files = [f for f in os.listdir(viz_dir) if f.endswith(".png")]
                        for i, file in enumerate(png_files, 1):
                            print(f"{i}. {file}")
                        
                        if png_files:
                            file_num = input("\nIngrese el número de la visualización que desea ver (0 para volver): ")
                            if file_num.isdigit() and 1 <= int(file_num) <= len(png_files):
                                file_path = os.path.join(viz_dir, png_files[int(file_num)-1])
                                open_file(file_path)
                        else:
                            print("No hay archivos PNG en el directorio.")
                    else:
                        print("\nNo hay visualizaciones del dashboard estático disponibles.")
                    
                    input("\nPresione Enter para continuar...")
                
                elif viz_choice == "3":
                    viz_dir = os.path.join(proyecto_dir, "src", "geospatial", "output")
                    if os.path.exists(viz_dir) and os.listdir(viz_dir):
                        print("\nVisualizaciones disponibles:")
                        png_files = [f for f in os.listdir(viz_dir) if f.endswith(".png")]
                        for i, file in enumerate(png_files, 1):
                            print(f"{i}. {file}")
                        
                        if png_files:
                            file_num = input("\nIngrese el número de la visualización que desea ver (0 para volver): ")
                            if file_num.isdigit() and 1 <= int(file_num) <= len(png_files):
                                file_path = os.path.join(viz_dir, png_files[int(file_num)-1])
                                open_file(file_path)
                        else:
                            print("No hay archivos PNG en el directorio.")
                    else:
                        print("\nNo hay visualizaciones geoespaciales disponibles.")
                    
                    input("\nPresione Enter para continuar...")
                
                elif viz_choice == "4":
                    break
                
                else:
                    print("\nOpción no válida. Intente de nuevo.")
                    input("\nPresione Enter para continuar...")
        
        elif choice == "5":
            break
        
        else:
            print("\nOpción no válida. Intente de nuevo.")
            input("\nPresione Enter para continuar...")

def plot_histograma_coherente(df, col, path):
    df = df[df[col].notnull() & (df[col].astype(str).str.lower() != 'desconocido')]
    plt.figure(figsize=(10, 6))
    plt.hist(df[col], bins=30, color='green', alpha=0.7)
    plt.title(f'Histograma de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.savefig(path)
    print(f"Histograma guardado: {path}")

def main():
    """Función principal que muestra el menú y maneja las opciones."""
    while True:
        print_header()
        print_menu()
        
        choice = input("Seleccione una opción (1-8): ")
        
        if choice == "1":
            exploratory_analysis_menu()
        
        elif choice == "2":
            dashboard_menu()
        
        elif choice == "3":
            geospatial_analysis_menu()
        
        elif choice == "4":
            print("\nEjecutando análisis predictivo...")
            run_script(os.path.join(proyecto_dir, "src", "analysis", "analisis_predictivo.py"))
            input("\nPresione Enter para continuar...")
        
        elif choice == "5":
            print("\nEjecutando análisis de contenido web...")
            run_script(os.path.join(proyecto_dir, "src", "web", "analisis_contenido_web.py"))
            input("\nPresione Enter para continuar...")
        
        elif choice == "6":
            print("\nConfigurando entorno...")
            run_script(os.path.join(proyecto_dir, "src", "dashboard", "setup_dashboard.py"))
            input("\nPresione Enter para continuar...")
        
        elif choice == "7":
            view_reports_menu()
        
        elif choice == "8":
            print("\nSaliendo del programa...")
            break
        
        else:
            print("\nOpción no válida. Intente de nuevo.")
            input("\nPresione Enter para continuar...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPrograma interrumpido por el usuario.")
    except Exception as e:
        print(f"\n\nError inesperado: {e}")
    
    # En Windows, mantener la ventana abierta
    if platform.system() == "Windows":
        input("\nPresione Enter para salir...")