"""Este módulo contiene herramientas para el análisis de contenido web."""

# Importar funciones y clases principales para facilitar su uso
try:
    from .analisis_contenido_web import (
        detect_delimiter,
        load_data,
        analyze_urls,
        analyze_url_content,
        preprocess_text,
        topic_modeling,
        visualize_url_analysis,
        visualize_content_analysis,
        generate_report,
        main
    )
except ImportError:
    # Manejo de errores para cuando las dependencias no están disponibles
    # (útil para la generación de documentación)
    pass