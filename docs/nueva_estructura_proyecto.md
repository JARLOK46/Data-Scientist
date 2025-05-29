# Nueva Estructura del Proyecto

Este documento describe la nueva organización del proyecto siguiendo el principio de responsabilidad única y separación por carpetas según tipo y función.

## Estructura de Carpetas

```
UN PASO AL EXITO/
├── src/                      # Código fuente principal
│   ├── data/                 # Datos y archivos CSV
│   ├── analysis/             # Scripts de análisis de datos
│   ├── visualization/        # Visualizaciones estáticas
│   │   └── output/           # Gráficos generados
│   ├── dashboard/            # Dashboards interactivos y estáticos
│   │   ├── dashboard_files/  # Archivos del dashboard interactivo
│   │   └── static/           # Archivos del dashboard estático
│   ├── utils/                # Utilidades y herramientas comunes
│   ├── geospatial/           # Análisis geoespacial
│   │   └── output/           # Resultados del análisis geoespacial
│   ├── web/                  # Análisis de contenido web
│   └── predictive/           # Modelos predictivos
├── tests/                    # Pruebas unitarias y de integración
├── docs/                     # Documentación del proyecto
└── config/                   # Archivos de configuración
```

## Organización por Responsabilidad

### 1. Módulo de Datos (`src/data/`)

**Responsabilidad**: Almacenamiento y gestión de datos.

**Archivos**:
- `20250525.export.CSV`: Datos originales
- `datos_limpios.csv`: Datos procesados y limpios

### 2. Módulo de Análisis (`src/analysis/`)

**Responsabilidad**: Análisis exploratorio y procesamiento de datos.

**Archivos**:
- `analizar_csv.py`: Análisis inicial del CSV
- `limpiar_analizar_csv.py`: Limpieza y análisis básico
- `analisis_predictivo.py`: Análisis predictivo

### 3. Módulo de Visualización (`src/visualization/`)

**Responsabilidad**: Generación de visualizaciones estáticas.

**Archivos**:
- `visualizar_datos.py`: Generación de gráficos y visualizaciones
- `output/`: Directorio con gráficos generados

### 4. Módulo de Dashboard (`src/dashboard/`)

**Responsabilidad**: Creación y gestión de dashboards interactivos y estáticos.

**Archivos**:
- `dashboard_adaptativo.py`: Dashboard estático adaptativo
- `dashboard_interactivo.py`: Dashboard interactivo con Dash
- `setup_dashboard.py`: Configuración del entorno para dashboards
- `dashboard_files/`: Archivos del dashboard interactivo
- `static/`: Archivos del dashboard estático

### 5. Módulo de Utilidades (`src/utils/`)

**Responsabilidad**: Funciones y herramientas comunes utilizadas por otros módulos.

**Archivos**:
- `ejecutar_analisis.py`: Script principal para ejecutar análisis

### 6. Módulo Geoespacial (`src/geospatial/`)

**Responsabilidad**: Análisis y visualización de datos geoespaciales.

**Archivos**:
- `analisis_geoespacial.py`: Análisis geoespacial completo
- `analisis_geoespacial_adaptativo.py`: Versión adaptativa del análisis
- `output/`: Resultados del análisis geoespacial

### 7. Módulo de Análisis Web (`src/web/`)

**Responsabilidad**: Análisis de URLs y contenido web en los datos.

**Archivos**:
- `analisis_contenido_web.py`: Análisis de contenido web

### 8. Módulo Predictivo (`src/predictive/`)

**Responsabilidad**: Modelos y análisis predictivos avanzados.

**Archivos**:
- (Pendiente de implementación)

### 9. Pruebas (`tests/`)

**Responsabilidad**: Pruebas unitarias y de integración.

**Archivos**:
- (Pendiente de implementación)

### 10. Documentación (`docs/`)

**Responsabilidad**: Documentación del proyecto.

**Archivos**:
- `README.md`: Documentación principal
- `explicacion_detallada.md`: Explicación detallada del proyecto
- `informe_analisis_datos.md`: Informe de análisis
- `mejoras.txt`: Lista de mejoras propuestas
- `nueva_estructura_proyecto.md`: Este documento

### 11. Configuración (`config/`)

**Responsabilidad**: Archivos de configuración del proyecto.

**Archivos**:
- `requirements_dashboard.txt`: Dependencias para el dashboard
- `requirements_geoespacial.txt`: Dependencias para análisis geoespacial

## Beneficios de la Nueva Estructura

1. **Separación de Responsabilidades**: Cada módulo tiene una responsabilidad única y bien definida.

2. **Mejor Mantenibilidad**: Es más fácil encontrar y modificar código cuando está organizado por función.

3. **Escalabilidad**: Facilita la adición de nuevas funcionalidades sin afectar a las existentes.

4. **Reutilización de Código**: Promueve la creación de componentes reutilizables.

5. **Documentación Clara**: La estructura del proyecto es autoexplicativa y facilita la comprensión del sistema.

## Próximos Pasos

1. **Refactorización de Código**: Adaptar el código existente para que funcione con la nueva estructura de carpetas.

2. **Implementación de Pruebas**: Crear pruebas unitarias para cada módulo.

3. **Documentación con Sphinx**: Implementar documentación detallada con Sphinx.

4. **Creación de Paquetes**: Convertir los módulos en paquetes Python adecuados con archivos `__init__.py`.

5. **Actualización de Importaciones**: Actualizar las importaciones en todos los archivos para reflejar la nueva estructura.