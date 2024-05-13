import os
import shutil

# Define las rutas a las carpetas
carpeta_txt = '/home/ana/JSON2YOLO/new_dir/labels/train_128_sliced_coco.json_coco'
carpeta_imagenes = '/home/ana/TFM/datasets/data_sliced_casero_128_128'
carpeta_destino = '/home/ana/JSON2YOLO/new_dir/images'

# Crea la carpeta destino si no existe
os.makedirs(carpeta_destino, exist_ok=True)

# Recorre todos los archivos .txt en la carpeta de etiquetas YOLO
for archivo_txt in os.listdir(carpeta_txt):
    if archivo_txt.endswith('.txt'):
        nombre_base = archivo_txt[:-4]  # Elimina la extensión .txt para obtener el nombre base

        # Intenta encontrar y mover la imagen correspondiente
        for extension in ['.jpg', '.jpeg', '.png']:  # Asegúrate de incluir todas las extensiones posibles de imágenes
            imagen_path = os.path.join(carpeta_imagenes, nombre_base + extension)
            if os.path.exists(imagen_path):
                shutil.move(imagen_path, carpeta_destino)
                print(f"Imagen movida: {imagen_path} -> {carpeta_destino}")
                break

print("Proceso completado.")
