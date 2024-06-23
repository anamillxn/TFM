from ultralytics import YOLO, RTDETR
from PIL import Image
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

class MobileDamageDetection:
    def __init__(self, yolov9_model_path, rtdetr_model_path, device='cuda:0'):
        # Carga del modelo YOLO
        self.yolo_model = YOLO(yolov9_model_path)
        # Carga del modelo RTDETR
        self.rtdetr_model = AutoDetectionModel.from_pretrained(
            model_type='rtdetr',
            model_path=rtdetr_model_path,
            confidence_threshold=0.3,
            device=device
        )

    def load_image(self, image_path):
        # Carga de la imagen
        return Image.open(image_path)

    def predict_yolo(self, image):
        # Inferencia del modelo YOLO sobre la imagen
        results = self.yolo_model(image, verbose=False)
        for result in results:
            boxes = result.boxes.cpu().numpy().xyxy
            classes = result.boxes.cpu().numpy().cls
            names = result.names
            for box, cls in zip(boxes, classes):
                if names[cls] == "cell phone":
                    # Recorte y redimensionamiento de la imagen del teléfono celular
                    crop = image.crop((box[0], box[1], box[2], box[3]))
                    new_size = (int(crop.size[0] / 3), int(crop.size[1] / 3))
                    return crop.resize(new_size)
        return None

    def predict_rtdetr(self, image_array):
        # Inferencia del modelo RTDETR sobre la imagen recortada
        return get_sliced_prediction(
            image_array,
            self.rtdetr_model,
            slice_height=128,
            slice_width=128,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )

    def save_results(self, result, export_dir="demo_data/"):
        # Exportar resultados visuales
        result.export_visuals(export_dir=export_dir)
        return f"{export_dir}/prediction_visual.png"

# Uso de la clase
detector = MobileDamageDetection(
    yolov9_model_path='/home/ana/TFM/modelos/pre-crop-cell-phone/weights/best.pt',
    rtdetr_model_path='/home/ana/TFM/modelos/experimento_4_ampliado/weights/best.pt'
)

image = detector.load_image('/home/ana/TFM/datasets/dataset_previa/desperfectos_moviles_reduccion_clases.v1i.yolov9/test/images/IMG_0001-3-_png.rf.929b8b12404c6e6dfc6a0bc00dcd0bb7.jpg')
cropped_image = detector.predict_yolo(image)
if cropped_image:
    image_array = np.array(cropped_image)
    rtdetr_result = detector.predict_rtdetr(image_array)
    visual_path = detector.save_results(rtdetr_result)
    print("Visuals saved at:", visual_path)
