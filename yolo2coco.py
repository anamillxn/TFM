import json
import os
import shutil
from PIL import Image

# Set the paths for the input directories and the output directory
images_dir = '/home/ana/TFM/datasets/dataset_sliced_casero/valid/images'
labels_dir = '/home/ana/TFM/datasets/dataset_sliced_casero/valid/labels'
output_dir = '/home/ana/TFM/datasets/dataset_sliced_casero_coco/valid'
output_images_dir = os.path.join(output_dir, 'images')  # Folder for the copied images

# Ensure the output directories exist
os.makedirs(output_images_dir, exist_ok=True)

# Define the categories for the COCO dataset
categories = [{"id": 0, "name": "Chasis leve"}, {"id": 1, "name": "Chasis profundo"}, {"id": 2, "name": "Pantalla leve"}]

# Define the COCO dataset dictionary
coco_dataset = {
    "info": {},
    "licenses": [],
    "categories": categories,
    "images": [],
    "annotations": []
}

# Initialize a counter for unique image IDs
image_id_counter = 0

# Loop through the images in the images directory
for image_file in os.listdir(images_dir):
    image_path = os.path.join(images_dir, image_file)
    
    # Check if the path is a file and not a directory
    if os.path.isfile(image_path):
        # Load the image and get its dimensions
        image = Image.open(image_path)
        width, height = image.size
        
        # Add the image to the COCO dataset with a unique ID
        image_dict = {
            "id": image_id_counter,
            "width": width,
            "height": height,
            "file_name": image_file
        }
        coco_dataset["images"].append(image_dict)
        image_id_counter += 1  # Increment the counter for the next image
        
        # Copy the image to the output images directory
        output_image_path = os.path.join(output_images_dir, image_file)
        shutil.copy(image_path, output_image_path)
        
        # Construct the path for the corresponding annotation file
        annotation_file = os.path.join(labels_dir, f'{image_file.split(".")[0]}.txt')
        
        # Check if the annotation file exists and is a file
        if os.path.isfile(annotation_file):
            with open(annotation_file) as f:
                annotations = f.readlines()
            
            # Loop through the annotations and add them to the COCO dataset
            for ann in annotations:
                x, y, w, h = map(float, ann.strip().split()[1:])
                x_min, y_min = int((x - w / 2) * width), int((y - h / 2) * height)
                x_max, y_max = int((x + w / 2) * width), int((y + h / 2) * height)
                ann_dict = {
                    "id": len(coco_dataset["annotations"]),
                    "image_id": image_dict["id"],  # Reference the image ID
                    "category_id": 0,  # Adjust based on the data read
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "area": (x_max - x_min) * (y_max - y_min),
                    "iscrowd": 0
                }
                coco_dataset["annotations"].append(ann_dict)

# Save the COCO dataset to a JSON file
with open(os.path.join(output_dir, 'annotations.json'), 'w') as f:
    json.dump(coco_dataset, f)

