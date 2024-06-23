# Utilizar una imagen base oficial de Python
FROM python:3.10-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo de dependencias primero, para aprovechar la cache de capas de Docker
COPY requirements.txt .

# Instalar AWS CLI y boto3
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install boto3 && \
    apt-get update && \
    apt-get install -y awscli

# Copiar el resto del código fuente del proyecto al contenedor
COPY . .

# Configurar el puerto (opcional, dependiendo si tu aplicación necesita un puerto)
EXPOSE 8080

# Comando para ejecutar la aplicación Python
CMD ["python", "inferencia.py"]

