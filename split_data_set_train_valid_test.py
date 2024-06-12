import os
import shutil
from random import shuffle

# Rutas a las carpetas donde se encuentran todas las imágenes y etiquetas
carpeta_imagenes = '/home/ana/TFM/datasets/dataset_precrop_sahi_coco'
carpeta_labels = '/home/ana/JSON2YOLO/new_dir/labels/crop_cell_128_sliced_coco.json_coco'

# Crea las carpetas para los conjuntos de datos
sets = ['train', 'valid', 'test']
for set_name in sets:
    os.makedirs(f'/home/ana/TFM/datasets/dataset_precrop_sahi_yolo/{set_name}/images', exist_ok=True)
    os.makedirs(f'/home/ana/TFM/datasets/dataset_precrop_sahi_yolo/{set_name}/labels', exist_ok=True)

# Lista todas las imágenes y etiquetas
all_images = [f for f in os.listdir(carpeta_imagenes) if f.endswith(('.jpg', '.jpeg', '.png'))]
all_labels = [f for f in os.listdir(carpeta_labels) if f.endswith('.txt')]

# Asegúrate de que cada imagen tenga su correspondiente archivo de etiqueta
all_files = [(img, img.replace('.png', '.txt')) for img in all_images if img.replace('.png', '.txt') in all_labels]

# Baraja aleatoriamente los archivos
shuffle(all_files)

# Asigna archivos a cada conjunto
train_files = all_files[:10000]
valid_files = all_files[10000:12000]
test_files = all_files[12000:14000]

# Función para mover los archivos
def move_files(files, set_name):
    for img, label in files:
        # Mover imágenes
        shutil.move(os.path.join(carpeta_imagenes, img), f'/home/ana/TFM/datasets/dataset_precrop_sahi_yolo/{set_name}/images/{img}')
        # Mover etiquetas
        shutil.move(os.path.join(carpeta_labels, label), f'/home/ana/TFM/datasets/dataset_precrop_sahi_yolo/{set_name}/labels/{label}')

# Mueve los archivos a las carpetas correspondientes
move_files(train_files, 'train')
move_files(valid_files, 'valid')
move_files(test_files, 'test')

print("Los archivos han sido distribuidos correctamente en las carpetas de train, valid y test.")
