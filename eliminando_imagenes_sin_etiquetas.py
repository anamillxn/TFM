import os

# Definir las rutas a las carpetas de imágenes y etiquetas
ruta_etiquetas = '/home/ana/TFM/datasets/desperfectos_moviles_reduccion_clases.v3i.yolov8/valid/labels'
ruta_imagenes = '/home/ana/TFM/datasets/desperfectos_moviles_reduccion_clases.v3i.yolov8/valid/images'

# Listar todos los archivos en la carpeta de etiquetas
archivos_etiquetas = os.listdir(ruta_etiquetas)

# Contar los archivos eliminados
archivos_eliminados = 0

# Procesar cada archivo de etiqueta
for archivo in archivos_etiquetas:
    ruta_archivo_etiqueta = os.path.join(ruta_etiquetas, archivo)
    
    # Verificar si el archivo de etiqueta está vacío
    if os.stat(ruta_archivo_etiqueta).st_size == 0:
        # Construir el nombre del archivo de imagen correspondiente
        nombre_imagen = archivo.replace('.txt', '.jpg')
        ruta_archivo_imagen = os.path.join(ruta_imagenes, nombre_imagen)
        
        # Eliminar el archivo de etiqueta
        os.remove(ruta_archivo_etiqueta)
        
        # Verificar si la imagen correspondiente existe y eliminarla
        if os.path.exists(ruta_archivo_imagen):
            os.remove(ruta_archivo_imagen)
            archivos_eliminados += 1

print(f'Se eliminaron {archivos_eliminados} pares de archivos.')
