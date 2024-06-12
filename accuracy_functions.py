import cv2
import random

class ULTRALYTICSMetrics:
    
    def get_boxes_image(self,image_path, labels, names_path,draw=False):
        """
            This function extracts bounding boxes from detection labels and optionally draws them on the image.

            Args:
            image_path (str): The path to the image.
            labels (list): A list of detection labels in the format [class_id, x, y, w, h].
            names_path (str): The path to the file containing class names.
            draw (bool): If True, it draws labels on the image; if False, it returns the original image.

            Returns:
            true_boxes: A list of true boxes, where each box is represented as a dictionary with the format {'class_name': ((x1, y1), (x2, y2))}.
            image_with_labels (optional): The image with labels and bounding boxes drawn (only if draw=True).
        """
        # Reads the class names from the training.names file.
        with open(names_path, "r") as f:
            class_names = f.readlines()
            class_names = [c.strip() for c in class_names]
        class_names = {i: name for i, name in enumerate(class_names)}

        # Get the width and height of the image.
        image = cv2.imread(image_path)
        width = image.shape[1]
        height = image.shape[0]
        
        true_boxes = []

        '''
        # Function to generate colors
        def generate_colors(num_colors):
            color_dict = {}
            for i in range(num_colors):
                red = random.randint(0, 255)
                green = random.randint(0, 255)
                blue = random.randint(0, 255)
                color_dict[i] = (red, green, blue)
            return color_dict

        class_colors = generate_colors(len(class_names))
        '''

        def generate_fixed_colors(num_colors):
            fixed_colors = [
                (255, 255, 0), # Amarillo
                (255, 0, 0),   # Rojo
                (0, 255, 0),   # Verde
                (0, 0, 255),   # Azul                
                (255, 0, 255), # Magenta
                (0, 255, 255), # Cian
                (128, 0, 0),   # Rojo oscuro
                (0, 128, 0),   # Verde oscuro
                (0, 0, 128),   # Azul oscuro
                (128, 128, 0), # Amarillo oscuro
                (128, 0, 128), # Magenta oscuro
                (0, 128, 128)  # Cian oscuro
            ]

            if num_colors <= len(fixed_colors):
                return fixed_colors[:num_colors]
            else:
                additional_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_colors - len(fixed_colors))]
                return fixed_colors + additional_colors

        class_colors = generate_fixed_colors(len(class_names))

        for label in labels:
            class_id, x, y, w, h, _ = label
        
            # Convert coordinates to pixels
            x, y, w, h = float(x) * width, float(y) * height, float(w) * width, float(h) * height
            x, y, w, h = round(x), round(y), round(w), round(h)

            # Calculate the coordinates of the upper left corner of the rectangle
            x1 = int((x - w / 2) )
            y1 = int((y - h / 2) )

            # Calculate the coordinates of the lower right corner of the rectangle
            x2 = int((x + w / 2) )
            y2 = int((y + h / 2) )

            true_boxes.append({class_names[int(class_id)]: ((x1, y1), (x2, y2))})

            if draw==True:
                image = cv2.rectangle(image, (x1, y1), (x2, y2), class_colors[int(class_id)], 3)
                cv2.putText(image, str(class_names[int(class_id)]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, class_colors[int(class_id)], 2)

        return image, true_boxes

    def ground_truth_labels(self, labels_path):
        # Reads the tags from the text file.
        with open(labels_path, "r") as f:
            labels = f.readlines()        
        
        #format
        labels=[cadena.rstrip('\n') for cadena in labels]
        sublabels=[cadena.split() for cadena in labels]
        labels = [[float(sublabels) for sublabels in sublabels] for sublabels in sublabels]
        
        return labels
    
    def calculate_iou(self,box1, box2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        This function computes the IoU, a measure of overlap, between two bounding boxes provided as pairs of coordinates (x1, y1) and (x2, y2). It calculates the area of intersection and the area of union between the boxes and returns the IoU score.
        Args:
            box1: A pair of coordinates (x1, y1) and (x2, y2) defining the first bounding box.
            box2: A pair of coordinates (x1, y1) and (x2, y2) defining the second bounding box.
        Returns:
            The IoU score, a value between 0.0 (no overlap) and 1.0 (complete overlap).
        """

        # Calculate the intersection (common area)
        x1 = max(box1[0][0], box2[0][0])
        y1 = max(box1[0][1], box2[0][1])
        x2 = min(box1[1][0], box2[1][0])
        y2 = min(box1[1][1], box2[1][1])

        # Calculate the intersection area
        intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

        # Calculate area of bounding boxes
        box1_area = (box1[1][0] - box1[0][0] + 1) * (box1[1][1] - box1[0][1] + 1)
        box2_area = (box2[1][0] - box2[0][0] + 1) * (box2[1][1] - box2[0][1] + 1)

        # Calculate the union (total area - area of intersection)
        union_area = box1_area + box2_area - intersection_area

        # Calculate the index of overlap (IoU)
        iou = intersection_area / union_area

        return iou
    def match_labels(self,predictions, ground_truth,iou_threshold=0.5):
        """
        Match ULTRALYTICS Labels in Predictions with Ground Truth Labels.

        This function compares ULTRALYTICS-style label predictions with ground truth labels and returns a list of matched labels, where each match contains the prediction and the corresponding ground truth label if the Intersection over Union (IoU) exceeds a specified threshold.

        Args:
            predictions: List of ULTRALYTICS-style label predictions.
            ground_truth: List of ULTRALYTICS-style ground truth labels.
            iou_threshold: IoU threshold for matching labels (default is 0.5).

        Returns:
            A list of matched labels, each containing 'prediction' and 'ground_truth' information.
        """

        matched_labels = []

        for prediction in predictions:
            match = None
            max_iou = 0

            for gt_label in ground_truth:
                for gt_class, gt_bbox in gt_label.items():
                    if gt_class not in prediction:
                        continue

                    iou = self.calculate_iou(prediction[gt_class], gt_bbox)
                    if iou > max_iou and iou >= iou_threshold:
                        max_iou = iou                    
                        match = {'prediction': {gt_class: prediction[gt_class]}, 'ground_truth': gt_label}

            if match:
                matched_labels.append(match)
        
        return matched_labels

    def calculate_total_image_accuracy(self,ground_truth, predictions,print_):
        """
        Calculate the Total Image Accuracy by Matching Predictions with Ground Truth Labels.

        This function calculates the total image accuracy by comparing ULTRALYTICS-style label predictions with ground truth labels. It computes the accuracy as the ratio of correctly matched labels to the total number of ground truth labels, with an optional ability to print the matched labels.

        Args:
            ground_truth: List of ULTRALYTICS-style ground truth labels.
            predictions: List of ULTRALYTICS-style label predictions.
            print_: Boolean flag to control whether to print matched labels (True for printing, False for no printing).

        Returns:
            The total image accuracy, a value between 0.0 (no correct matches) and 1.0 (all matches are correct).
        """

        total_labels = 0
        correctly_matched_labels = 0

        # Calculate the total number of actual labels.
        for gt_label in ground_truth:
            total_labels += len(gt_label)

        matched_labels = self.match_labels(predictions, ground_truth,iou_threshold=0.5)

        if print_==True:
            for match in matched_labels:
                print(f"Prediction: {match['prediction']} matches Ground Truth: {match['ground_truth']}")
        # Calculate the number of correctly related tags.
        for match in matched_labels:
            correctly_matched_labels += len(match['ground_truth'])

        # Calculate the "total accuracy" of the image.
        total_accuracy = correctly_matched_labels / total_labels

        return total_accuracy

    def calculate_label_accuracy(self,ground_truth, predictions, class_id, iou_threshold=0.5):
        """
        Calculate Label Accuracy for a Specific Class in ULTRALYTICS-style Object Detection.

        This function calculates the accuracy of a specific class in ULTRALYTICS-style object detection. It counts the number of correctly matched labels for the given class and computes the accuracy as the ratio of correct matches to the total number of ground truth labels for that class.

        Args:
            ground_truth: List of ULTRALYTICS-style ground truth labels.
            predictions: List of ULTRALYTICS-style label predictions.
            class_id: The ID of the specific class for which accuracy is calculated.
            iou_threshold: IoU threshold for matching labels (default is 0.5).

        Returns:
            The accuracy for the specific class, a value between 0.0 (no correct matches) and 1.0 (all matches for the class are correct).
        """

        total_labels = 0
        correctly_matched_labels = 0

        for gt_labels in ground_truth:
            for obj in gt_labels:
                if obj == class_id:
                    total_labels += 1  #get the number of actual tags of the class being evaluated

        matched_labels = self.match_labels(predictions, ground_truth, iou_threshold)

        for match in matched_labels:
            for obj in match['ground_truth']:
                if obj == class_id:
                    correctly_matched_labels += 1

        # Calculate the accuracy of the specific label
        label_accuracy = min(correctly_matched_labels / total_labels, 1) if total_labels > 0 else 0

        return label_accuracy

    def calculate_class_accuracies(self,true_boxes, predicted_boxes, names_path, iou_threshold=0.5):
        """
        Calculate Class Accuracies in ULTRALYTICS-style Object Detection.

        This function calculates the accuracies for each class in ULTRALYTICS-style object detection. It reads class names from a file, computes the accuracy for each class by comparing true boxes with predicted boxes, and returns a dictionary of class accuracies.

        Args:
            true_boxes: List of ULTRALYTICS-style ground truth labels.
            predicted_boxes: List of ULTRALYTICS-style label predictions.
            names_path: Path to a file containing class names.
            iou_threshold: IoU threshold for matching labels (default is 0.5).

        Returns:
            A dictionary where keys are class names and values are the accuracies for each class, with values between 0.0 (no correct matches) and 1.0 (all matches for the class are correct).
        """

        with open(names_path, "r") as f:
            class_names = f.readlines()
            class_names = [c.strip() for c in class_names]
            class_names={i: nombre for i, nombre in enumerate(class_names)}

        # A dictionary to store the results
        class_accuracies = {}

        for key, class_name in class_names.items():
            accuracy = self.calculate_label_accuracy(true_boxes, predicted_boxes, class_name, iou_threshold=0.5)
            class_accuracies[class_name] = accuracy

        return class_accuracies
    

