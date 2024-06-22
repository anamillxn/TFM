import sys
import json
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QFileDialog, QLabel
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage
from PyQt5.QtCore import Qt

class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pixmap = None
        self.annotations = []
        self.setScaledContents(False)

    def set_image_with_annotations(self, image_path, annotations):
        self.original_pixmap = QPixmap(image_path)
        if self.original_pixmap.isNull():
            raise IOError("Cannot load the specified image.")
        self.annotations = annotations
        self.update_pixmap()

    def update_pixmap(self):
        if self.original_pixmap:
            pixmap = QPixmap(self.original_pixmap.size())
            pixmap.fill(Qt.transparent)  # Fondo transparente

            painter = QPainter(pixmap)
            painter.drawPixmap(0, 0, self.original_pixmap)

            for annotation in self.annotations:
                # Determina el color basado en la clase de la anotación
                if annotation['category_id'] == 1:
                    pen = QPen(Qt.red, 3, Qt.SolidLine)
                elif annotation['category_id'] == 2:
                    pen = QPen(Qt.blue, 3, Qt.SolidLine)
                elif annotation['category_id'] == 3:
                    pen = QPen(Qt.green, 3, Qt.SolidLine)
                else:
                    pen = QPen(Qt.yellow, 3, Qt.SolidLine)  # Un color por defecto

                painter.setPen(pen)
                x, y, w, h = [int(dim) for dim in annotation['bbox']]
                painter.drawRect(x, y, w, h)

            painter.end()
            scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        self.update_pixmap()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer")
        self.resize(800, 600)

        self.image_index = 0
        self.image_keys = []
        self.layout = QVBoxLayout(self)

        self.image_label = ImageLabel(self)
        self.layout.addWidget(self.image_label)

        self.button_layout = QHBoxLayout()
        self.btn_prev = QPushButton("Previous Image")
        self.btn_prev.clicked.connect(self.prev_image)
        self.button_layout.addWidget(self.btn_prev)

        self.btn_next = QPushButton("Next Image")
        self.btn_next.clicked.connect(self.next_image)
        self.button_layout.addWidget(self.btn_next)

        self.layout.addLayout(self.button_layout)
        self.setLayout(self.layout)
        self.annotations = {}
        self.image_folder = ''
        
        self.load_annotations()

    def load_annotations(self):
        file_name = QFileDialog.getOpenFileName(self, "Open Annotations File", "", "JSON Files (*.json);;All Files (*)")[0]
        if file_name:
            self.image_folder = os.path.dirname(file_name)
            with open(file_name, 'r') as file:
                data = json.load(file)
                image_data = {img['id']: img['file_name'] for img in data['images']}
                for annotation in data['annotations']:
                    image_id = annotation['image_id']
                    if image_id in image_data:
                        file_name = image_data[image_id]
                        if file_name not in self.annotations:
                            self.annotations[file_name] = []
                        self.annotations[file_name].append(annotation)
            self.image_keys = list(self.annotations.keys())
            self.show_image()

    def show_image(self):
        if self.image_keys:
            file_name = self.image_keys[self.image_index]
            image_path = os.path.join(self.image_folder, file_name)
            self.image_label.set_image_with_annotations(image_path, self.annotations[file_name])

    def next_image(self):
        if self.image_index < len(self.image_keys) - 1:
            self.image_index += 1
            self.show_image()

    def prev_image(self):
        if self.image_index > 0:
            self.image_index -= 1
            self.show_image()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
