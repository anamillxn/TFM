"""
####################################################################################
# Copyright (C) Airvant Mediciones Ambientales - All Rights Reserved               #
# Unauthorized copying of this file, via any medium is strictly prohibited         #
# Proprietary and confidential                                                     #
# Written by Airvant Mediciones Ambientales <administracion@airvant.es>, June 2021 #
####################################################################################

@file    accuracy_functions.py
@brief   Functions for computing accuracy metrics for classification
@author  Ana Millán Serrano , ana.millan@airvant.es
@date    10/2023
"""
import ultralytics
import numpy as np
import io
import supervision as sv
import matplotlib.pyplot as plt
import cv2
import random

class model:
    def __init__(self,weigth_path, type):
        """
            Initialize the model class with a specified weight path.
            Args:
                weight_path (str): The path to the YOLO model weights.

            Returns:
                None
        """
        if type == 'YOLO':
            self.model=ultralytics.YOLO(weigth_path)
        else:
            self.model=ultralytics.RTDETR(weigth_path)

    def detected_labels_model(self,image_path,format="xyxy"):
        """
            Perform object detection using the model model on the given image.

            Args:
                image_path (str): The path to the input image for object detection.
                format (Default="xyxy") (str): Desired format of bounding boxes | xywh | xywhn | xyxy | xyxyn |

            Returns:
                labels_detected (numpy.ndarray): An array containing detected class IDs and bounding boxes.
        """
         
        detection=self.model(image_path, verbose = False) #Prediction on image

        # Extract class IDs from the detection results.
        class_id=detection[0].boxes.cpu().numpy().cls

        #Extract confidence from the detection results
        conf=detection[0].boxes.cpu().numpy().conf

        names=detection[0].names

        #Extract bounding boxes with desired format from the detection results.
        if format == "xywh":
            bboxes=detection[0].boxes.cpu().numpy().xywh
        elif format == "xywhn":
            bboxes=detection[0].boxes.cpu().numpy().xywhn
        elif format == "xyxy":
            bboxes=detection[0].boxes.cpu().numpy().xyxy
        elif format == "xyxyn":
            bboxes=detection[0].boxes.cpu().numpy().xyxyn
        else:
            print("The 'format' value is not valid; we take the default format: | xyxy |")
            bboxes=detection[0].boxes.cpu().numpy().xyxy
        

        # Combine class IDs and bounding boxes into a single array.
        labels_detected=np.concatenate((class_id.reshape(-1,1),bboxes,conf.reshape(-1,1)), axis=1)
        return labels_detected, names
    
    def save_predicted_boxes_to_txt(self,predicted_boxes, txt_file_path, names_path,height, width):
        """
            Save predicted bounding boxes to a TXT file.

            Args:
                predicted_boxes (list of dicts): List of dictionaries containing predicted bounding boxes.
                txt_file_path (str): The path to the output TXT file.
                names_path (str): The path to a file containing class names.
                height (int): Height of the original image.
                width (int): Width of the original image.

            Returns:
                None
        """
        # Get class names.
        with open(names_path, 'r') as f:
            names = f.readlines()
        names = [name.strip() for name in names]

        # Convert bounding boxes to TXT format.
        boxes_txt = []
        for box in predicted_boxes:
            key=list(box.keys())
            class_id = names.index(key[0])
            x1, y1 = box[key[0]][0][0]/width, box[key[0]][0][1]/height           
            x2, y2 = box[key[0]][1][0]/width, box[key[0]][1][1]/height

            w = x2 - x1
            h = y2 - y1
            x = x1  + w / 2
            y = y1 + h / 2
            
            boxes_txt.append(f"{class_id} {x:.4f} {y:.4f} {w:.4f} {h:.4f}")

        # Save the results to a TXT file.
        with io.open(txt_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(boxes_txt))

    def confusion_matrix(self,image_path,labels_path,data_class_path):
        """
            Generate a confusion matrix for object detection.

            Args:
                image_path (str): Path to the directory containing images.
                labels_path (str): Path to the directory containing labels for the images.
                data_class_path (str): Path to the file containing class names.

            Returns:
                sv.ConfusionMatrix: The confusion matrix generated from the object detection results.
        """
        # Create a dataset from YOLO format labels
        dataset = sv.DetectionDataset.from_yolo(image_path,labels_path,data_class_path)

        
        def callback(image: np.ndarray) -> sv.Detections:
            # Perform object detection on the image using the model model
            result = self.model(image)[0]
            return sv.Detections.from_ultralytics(result)

        # Generate the confusion matrix
        confusion_matrix = sv.ConfusionMatrix.benchmark(
        dataset = dataset,
        callback = callback
        )

        # Plot and save the confusion matrix
        confusion_matrix.plot(normalize=True)
        plt.savefig('confusion_matrix.jpg', format='jpg')
        return confusion_matrix
    
    def plot_detected_labels(self,img_path):
        """
            Perform object detection on an image and plot the detected objects.

            Args:
                img_path (str): Path to the input image for object detection.

            Returns:
                np.ndarray: An image with detected objects plotted.
        """
        # Perform object detection on the image using the model model
        detection=self.model(img_path,iou=0)
        
        # Plot the detected objects on the image
        img_detected=detection[0].plot()
        return img_detected
        
    def detected_labels_filter_depth(self,img_path,img_path_depth,threshold):

        def generate_fixed_colors(num_colors):
            fixed_colors = [
                (125, 120, 20),  
                (255, 160, 180),  
                (100, 200, 100),  
                (120, 160, 180),  
                (200, 100, 100),  
                (180, 100, 200),  
                (255, 90, 90),    
                (100, 200, 100),   
                (120, 160, 180),   
                (255, 210, 180),   
                (255, 160, 180),   
                (150, 150, 220)    
            ]

            if num_colors <= len(fixed_colors):
                return fixed_colors[:num_colors]
            else:
                additional_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_colors - len(fixed_colors))]
                return fixed_colors + additional_colors

        # Detect objects in the image using model
        detections, names = self.detected_labels_model(img_path, "xyxy")

        # Read the depth image
        cv_depth = cv2.imread(img_path_depth, cv2.IMREAD_UNCHANGED)

        # Read the RGB image
        img = cv2.imread(img_path)

        # Convert the depth image to grayscale and apply a colormap
        depth_gray = cv2.convertScaleAbs(cv_depth, alpha=255 / (2.1 * 1000))
        depth_colored = cv2.applyColorMap(depth_gray, cv2.COLORMAP_JET)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to the depth image
        _, ib = cv2.threshold(depth_colored, threshold, 255, cv2.THRESH_BINARY)

        # Generate a list of fixed colors for the detected labels
        class_colors = generate_fixed_colors(len(names))

        # Extract the class labels, confidences, and bounding boxes from the detections
        cls = detections[:, 0].astype(int)
        conf = detections[:, 5]
        boxes = detections[:, 1:5].astype(int)

        # Create a list to store the indices of the detected labels that fall within the filtered depth region
        nearest = []

        # Iterate over the detected labels
        for i in range(len(boxes)):
            # Get the bounding box coordinates of the current label
            x_min, y_min, x_max, y_max = boxes[i]

            # Extract the region of interest (ROI) from the filtered depth image
            roi = ib[y_min:y_max, x_min:x_max]

            # Calculate the mean of the ROI
            roi_mean = np.mean(roi)

            # If the mean of the ROI is greater than the threshold, append the index of the current label to the nearest list
            if roi_mean > threshold:
                nearest.append(i)

        # Create an annotator to draw the detected labels on the RGB image
        annotator = ultralytics.utils.plotting.Annotator(img, False)

        depth_detections=detections[nearest]

        # Iterate over the detected labels that fall within the filtered depth region
        for i in nearest:
            # Get the label name and confidence of the current label
            label=names[cls[i]]+" "+str(round(conf[i],2))

            # Draw the bounding box and label of the current label
            annotator.box_label(boxes[i], label=label, color=class_colors[int(cls[i])])

            
        return img, depth_detections

    def detected_labels_tracking(self,frame,format="xyxy"):

        tracking_detections=self.model.track(frame, persist=True)

        # Extract class IDs from the detection results.
        class_id=tracking_detections[0].boxes.cpu().numpy().cls

        #Extract tracking IDs from the detection results
        track_id=tracking_detections[0].boxes.cpu().numpy().id

        #Extract confidence from the detection results
        conf=tracking_detections[0].boxes.cpu().numpy().conf

        names=tracking_detections[0].names

        #Extract bounding boxes with desired format from the detection results.
        if format == "xywh":
            bboxes=tracking_detections[0].boxes.cpu().numpy().xywh
        elif format == "xywhn":
            bboxes=tracking_detections[0].boxes.cpu().numpy().xywhn
        elif format == "xyxy":
            bboxes=tracking_detections[0].boxes.cpu().numpy().xyxy
        elif format == "xyxyn":
            bboxes=tracking_detections[0].boxes.cpu().numpy().xyxyn
        else:
            print("The 'format' value is not valid; we take the default format: | xyxy |")
            bboxes=tracking_detections[0].boxes.cpu().numpy().xyxy
        

        # Combine class IDs and bounding boxes into a single array.
        labels_detected=np.concatenate((class_id.reshape(-1,1),track_id.reshape(-1,1),bboxes,conf.reshape(-1,1)), axis=1)
        return labels_detected, names
        