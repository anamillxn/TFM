import json

# Cargar los archivos JSON
with open('/home/ana/TFM/datasets/dataset_cropped/dataset_precrop_coco/test/annotations.json', 'r') as file:
    annotations_data = json.load(file)

with open('/home/ana/TFM/datasets/dataset_cropped/dataset_precrop_coco/test/predict.json', 'r') as file:
    predict_data = json.load(file)


# Preparar el diccionario para el nuevo JSON
new_annotations = {
    'images': annotations_data['images'],
    'annotations': []
}

# Crear un mapeo de image_id a detalles de imagen para acceso rápido
image_details = {img['id']: img for img in annotations_data['images']}

# Recorrer las predicciones y transformarlas al formato de annotations
for image_predictions in predict_data:
    for pred in image_predictions:
        new_annotation = {
            'image_id': pred['image_id'],
            'category_id': pred['category_id'],
            'bbox': pred['bbox'],
            'area': pred['area'],
            'iscrowd': pred['iscrowd'],
            'segmentation': pred.get('segmentation', [])
        }
        new_annotations['annotations'].append(new_annotation)

# Guardar el nuevo archivo JSON
with open('/home/ana/TFM/datasets/dataset_cropped/dataset_precrop_coco/test/new_annotations.json', 'w') as f:
    json.dump(new_annotations, f, indent=4)

print("Nuevo archivo 'new_annotations.json' creado con éxito.")

