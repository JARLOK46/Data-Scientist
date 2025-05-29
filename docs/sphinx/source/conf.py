# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Analisis de Datos y Visualizacion'
copyright = '2025, UN PASO AL EXITO'
author = 'Anderson Zapata Calle'

version = '1.0'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []

language = 'es'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Configuración adicional para mejorar la documentación
autoclass_content = 'both'  # Incluir tanto la docstring de la clase como del __init__
autodoc_member_order = 'bysource'  # Ordenar miembros como aparecen en el código
autodoc_typehints = 'description'  # Mostrar type hints en la descripción

# Añadir el directorio del proyecto al path de Python
import os
import sys
sys.path.insert(0, os.path.abspath('../../..'))

# Configuración para myst_parser
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]
