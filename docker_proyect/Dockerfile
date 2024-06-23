# Usa una imagen base de Python
FROM python:3.10-slim

# Configura el directorio de trabajo
WORKDIR /app

# Copia los archivos requeridos para el proyecto
COPY requirements.txt .
COPY inferencia.py .

# Instala las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Instala cualquier otra dependencia que no sea de Python
# Ejemplo, si necesitas compiladores o librerías específicas del sistema
# RUN apt-get update && apt-get install -y libjpeg-dev zlib1g-dev etc.

# Puerto para exponer (opcional, dependiendo de tu aplicación)
EXPOSE 8080

# Comando para ejecutar al iniciar el contenedor
CMD ["python", "inferencia.py"]
