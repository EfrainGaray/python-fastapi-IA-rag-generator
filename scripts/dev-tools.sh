#!/bin/sh

# Ejecutar pre-commit manualmente
pre-commit run --all-files

# Ejecutar black para formatear el código
black .

# Ejecutar isort para ordenar los imports
isort .
