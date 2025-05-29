# Script para reorganizar la estructura del proyecto

# Crear estructura de carpetas
$carpetas = @(
    "src",
    "src\data",
    "src\analysis",
    "src\visualization",
    "src\dashboard",
    "src\utils",
    "src\geospatial",
    "src\web",
    "src\predictive",
    "tests",
    "docs",
    "config"
)

foreach ($carpeta in $carpetas) {
    if (-not (Test-Path $carpeta)) {
        Write-Host "Creando carpeta: $carpeta"
        New-Item -ItemType Directory -Path $carpeta -Force | Out-Null
    } else {
        Write-Host "La carpeta $carpeta ya existe"
    }
}

# Mover archivos según su función

# 1. Archivos de datos
$archivos_datos = @(
    "20250525.export.CSV",
    "datos_limpios.csv"
)

foreach ($archivo in $archivos_datos) {
    if (Test-Path $archivo) {
        Write-Host "Moviendo $archivo a src\data"
        Move-Item -Path $archivo -Destination "src\data\" -Force
    }
}

# 2. Archivos de análisis
$archivos_analisis = @(
    "analizar_csv.py",
    "limpiar_analizar_csv.py",
    "analisis_predictivo.py"
)

foreach ($archivo in $archivos_analisis) {
    if (Test-Path $archivo) {
        Write-Host "Moviendo $archivo a src\analysis"
        Move-Item -Path $archivo -Destination "src\analysis\" -Force
    }
}

# 3. Archivos de visualización
$archivos_visualizacion = @(
    "visualizar_datos.py"
)

foreach ($archivo in $archivos_visualizacion) {
    if (Test-Path $archivo) {
        Write-Host "Moviendo $archivo a src\visualization"
        Move-Item -Path $archivo -Destination "src\visualization\" -Force
    }
}

# 4. Archivos de dashboard
$archivos_dashboard = @(
    "dashboard_adaptativo.py",
    "dashboard_interactivo.py",
    "setup_dashboard.py"
)

foreach ($archivo in $archivos_dashboard) {
    if (Test-Path $archivo) {
        Write-Host "Moviendo $archivo a src\dashboard"
        Move-Item -Path $archivo -Destination "src\dashboard\" -Force
    }
}

# 5. Archivos geoespaciales
$archivos_geoespacial = @(
    "analisis_geoespacial.py",
    "analisis_geoespacial_adaptativo.py"
)

foreach ($archivo in $archivos_geoespacial) {
    if (Test-Path $archivo) {
        Write-Host "Moviendo $archivo a src\geospatial"
        Move-Item -Path $archivo -Destination "src\geospatial\" -Force
    }
}

# 6. Archivos de análisis web
$archivos_web = @(
    "analisis_contenido_web.py"
)

foreach ($archivo in $archivos_web) {
    if (Test-Path $archivo) {
        Write-Host "Moviendo $archivo a src\web"
        Move-Item -Path $archivo -Destination "src\web\" -Force
    }
}

# 7. Archivos de utilidades
$archivos_utils = @(
    "ejecutar_analisis.py"
)

foreach ($archivo in $archivos_utils) {
    if (Test-Path $archivo) {
        Write-Host "Moviendo $archivo a src\utils"
        Move-Item -Path $archivo -Destination "src\utils\" -Force
    }
}

# 8. Mover carpetas existentes
$carpetas_a_mover = @(
    @{"origen" = "visualizaciones"; "destino" = "src\visualization"},
    @{"origen" = "dashboard"; "destino" = "src\dashboard\dashboard_files"},
    @{"origen" = "dashboard_estatico"; "destino" = "src\dashboard\static"},
    @{"origen" = "analisis_geoespacial"; "destino" = "src\geospatial\output"}
)

foreach ($carpeta in $carpetas_a_mover) {
    if (Test-Path $carpeta.origen) {
        Write-Host "Moviendo carpeta $($carpeta.origen) a $($carpeta.destino)"
        # Crear la carpeta de destino si no existe
        if (-not (Test-Path $carpeta.destino)) {
            New-Item -ItemType Directory -Path $carpeta.destino -Force | Out-Null
        }
        # Mover el contenido
        Get-ChildItem -Path $carpeta.origen | Move-Item -Destination $carpeta.destino -Force
        # Eliminar la carpeta vacía original
        Remove-Item -Path $carpeta.origen -Force
    }
}

# 9. Mover archivos de configuración
$archivos_config = @(
    "requirements_dashboard.txt",
    "requirements_geoespacial.txt"
)

foreach ($archivo in $archivos_config) {
    if (Test-Path $archivo) {
        Write-Host "Moviendo $archivo a config"
        Move-Item -Path $archivo -Destination "config\" -Force
    }
}

# 10. Mover archivos de documentación
$archivos_docs = @(
    "README.md",
    "explicacion_detallada.md",
    "informe_analisis_datos.md",
    "mejoras.txt"
)

foreach ($archivo in $archivos_docs) {
    if (Test-Path $archivo) {
        Write-Host "Moviendo $archivo a docs"
        Move-Item -Path $archivo -Destination "docs\" -Force
    }
}

Write-Host "\nReorganización completada. La nueva estructura del proyecto es:\n"

# Mostrar la nueva estructura
Get-ChildItem -Recurse -Directory | Select-Object FullName | Format-Table -HideTableHeaders