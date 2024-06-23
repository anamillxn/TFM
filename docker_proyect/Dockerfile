# Utilizar una imagen base oficial de Python
FROM python:3.10-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copiar el archivo de dependencias primero, para aprovechar la cache de capas de Docker
COPY requirements.txt .

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install boto3  # Asegúrate de incluir boto3 si no está en requirements.txt

# Copiar el resto del código fuente del proyecto al contenedor
COPY inferencia.py .

# Comando para ejecutar la aplicación Python
CMD ["python", "inferencia.py", "s3_bucket_name", "image_key", "yolo_model_key", "rtdetr_model_key"]


