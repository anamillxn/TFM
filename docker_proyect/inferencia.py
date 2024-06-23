import argparse
import io
import boto3
import numpy as np
from PIL import Image
from ultralytics import YOLO, RTDETR
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

class MobileDamageDetection:
    def __init__(self, bucket, yolo_model_key, rtdetr_model_key, device='cuda:0'):
        self.bucket = bucket
        self.yolo_model_key = yolo_model_key
        self.rtdetr_model_key = rtdetr_model_key
        self.device = device
        # Carga de modelos
        self.yolo_model = self.load_model_from_s3(self.yolo_model_key, YOLO)
        self.rtdetr_model = self.load_model_from_s3(self.rtdetr_model_key, AutoDetectionModel)

    def load_model_from_s3(self, model_key, model_class):
        """Carga un modelo desde un bucket de S3"""
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=self.bucket, Key=model_key)
        model_data = response['Body'].read()
        model_path = f"/tmp/{model_key}"
        with open(model_path, 'wb') as f:
            f.write(model_data)
        if model_class == YOLO:
            return model_class(model_path)
        else:
            return model_class.from_pretrained(
                model_type='rtdetr',
                model_path=model_path,
                confidence_threshold=0.3,
                device=self.device
            )

    def load_image_from_s3(self, bucket, key):
        """Carga una imagen desde un bucket de S3"""
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_content = response['Body'].read()
        image = Image.open(io.BytesIO(image_content))
        return image

    def predict_yolo(self, image):
        results = self.yolo_model(image, verbose=False)
        for result in results:
            boxes = result.boxes.cpu().numpy().xyxy
            classes = result.boxes.cpu().numpy().cls
            names = result.names
            for box, cls in zip(boxes, classes):
                if names[cls] == "cell phone":
                    crop = image.crop((box[0], box[1], box[2], box[3]))
                    new_size = (int(crop.size[0] / 3), int(crop.size[1] / 3))
                    return crop.resize(new_size)
        return None

    def predict_rtdetr(self, image_array):
        return get_sliced_prediction(
            image_array,
            self.rtdetr_model,
            slice_height=128,
            slice_width=128,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )

    def save_results(self, result, export_dir="demo_data/"):
        result.export_visuals(export_dir=export_dir)
        return f"{export_dir}/prediction_visual.png"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('s3_bucket', type=str, help='The S3 bucket where the models and image are stored')
    parser.add_argument('image_key', type=str, help='The S3 key for the image')
    parser.add_argument('yolo_model_key', type=str, help='S3 key for YOLO model weights')
    parser.add_argument('rtdetr_model_key', type=str, help='S3 key for RTDETR model weights')
    args = parser.parse_args()

    detector = MobileDamageDetection(
        bucket=args.s3_bucket,
        yolo_model_key=args.yolo_model_key,
        rtdetr_model_key=args.rtdetr_model_key
    )

    image = detector.load_image_from_s3(args.s3_bucket, args.image_key)
    cropped_image = detector.predict_yolo(image)
    if cropped_image:
        image_array = np.array(cropped_image)
        rtdetr_result = detector.predict_rtdetr(image_array)
        visual_path = detector.save_results(rtdetr_result)
        print("Visuals saved at:", visual_path)


