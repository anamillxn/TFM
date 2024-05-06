import os
import shutil

# Rutas de las carpetas
ruta_original = '/home/ana/TFM/datasets/desperfectos_moviles_reduccion_clases.v1i.coco/valid/sliced'
ruta_destino = '/home/ana/Desktop/dataset_sliced_casero/valid'

# Asegúrate de que la carpeta destino existe, si no, créala
if not os.path.exists(ruta_destino):
    os.makedirs(ruta_destino)

# Lista todos los archivos en la carpeta original
for archivo in os.listdir(ruta_original):
    # Comprueba si el archivo es una imagen PNG
    if archivo.endswith('.png'):
        # Construye el nombre completo del archivo y su posible pareja TXT
        nombre_base = archivo[:-4]  # Elimina la extensión '.png'
        archivo_png = os.path.join(ruta_original, archivo)
        archivo_txt = os.path.join(ruta_original, nombre_base + '.txt')

        # Si existe el archivo TXT correspondiente
        if os.path.exists(archivo_txt):
            # Mueve el archivo PNG a la carpeta destino
            shutil.move(archivo_png, os.path.join(ruta_destino, archivo))
            # Opcionalmente, también puedes mover el archivo TXT
            shutil.move(archivo_txt, os.path.join(ruta_destino, nombre_base + '.txt'))

print("Archivos movidos correctamente.")
