FROM python:3.9-slim

WORKDIR /app

# Instalar git y dependencias del sistema
RUN apt-get update && apt-get install -y git netcat-openbsd

# Instalar dependencias
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . /app

# Crear directorios de logs y archivo de estado
RUN mkdir -p /app/logs && touch /app/processed_files.txt

# Hacer que los scripts sean ejecutables
RUN chmod +x /app/scripts/dev-tools.sh /app/scripts/wait-for-it.sh

# Generar archivos __pycache__
RUN python -m compileall .

# Instalar pre-commit y configurar hooks
RUN if [ "$ENV" = "development" ]; then pre-commit install; fi

EXPOSE 5000

# Iniciar Uvicorn
CMD ["/app/scripts/wait-for-it.sh", "elasticsearch", "9200", "--", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000", "--reload"]
