#!/bin/bash

# Script para limpiar __pycache__
find . -type d -name "__pycache__" -exec rm -r {} +
