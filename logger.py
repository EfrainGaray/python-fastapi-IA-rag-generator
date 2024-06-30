from loguru import logger
import sys

# Configuración básica de loguru
logger.remove()
logger.add(sys.stdout, level="DEBUG", format="{time} {level} {message}", enqueue=True)

# Agregar un archivo de log
logger.add("logs/app.log", rotation="500 MB", level="INFO", format="{time} {level} {message}")

# Configuración para eliminar logs antiguos y mantener el archivo organizado
logger.add("logs/error.log", rotation="1 week", retention="10 days", level="ERROR", format="{time} {level} {message}")

# Función para configurar loguru en diferentes partes de la aplicación
def get_logger():
    return logger
