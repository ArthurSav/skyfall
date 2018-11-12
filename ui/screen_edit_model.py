#!/usr/local/bin/python3

import os

import cv2
from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QLabel, QFileDialog, QMessageBox, QInputDialog, QLineEdit, QDialog, \
    QPushButton, QVBoxLayout

from engine.contours import ContourFinder
from ui.widgets import ImageWidget
from utils.utils_camera import CameraManager

# load ui file
dir_path = os.path.dirname(os.path.realpath(__file__))
screen_edit_model_ui = uic.loadUiType(dir_path + '/screen_edit_model.ui')[0]


class ScreenEditModel(QMainWindow, screen_edit_model_ui):
    """
    UI manager for screen 'Add or Edit model'
    """

    camera_manager = CameraManager()
    finder = ContourFinder()

    is_processing_enabled = False
    scaled_crop_image = 150
    video_fps = 5

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)

        self.setup_categories()

        # self.window_width = 940
        # self.window_height = 1500
        self.window_width = self.widgetCamera.frameSize().width()
        self.window_height = self.widgetCamera.frameSize().height()
        # print("Height:{}, Width:{}".format(self.widgetCamera.frameSize().height(), self.widgetCamera.frameSize().width()))
        self.widgetCamera = ImageWidget(self.widgetCamera)

        self.btnLive.clicked.connect(self.__on_click_recording)
        self.btnPicture.clicked.connect(self.__on_click_picture)

        self.btnComponentAdd.clicked.connect(self.showdialog)

        # start recording by default
        self.open_camera()

    def __on_click_recording(self):
        """
        Opens camera and displays video
        """
        QtWidgets.QApplication.processEvents()

        self.open_camera()

        # start/stop video analysis
        if self.is_processing_enabled:
            self.btnLive.setText("Start recording")
            self.is_processing_enabled = False
        else:
            self.btnLive.setText("Stop recording")
            self.is_processing_enabled = True

    def __on_click_picture(self):
        """
        Opens file picker + loads image into imageview
        """
        QtWidgets.QApplication.processEvents()

        path = self.open_filename_dialog()
        image = cv2.imread(path)

        if image is not None:
            self.is_processing_enabled = True
            self.close_camera()
            self.update_frame(image)
        else:
            print("Could not load image")

    def showdialog(self):
        form = Form()
        form.show()

    def update_frame(self, frame):

        img_height, img_width, img_colors = frame.shape
        scale_w = float(self.window_width) / float(img_width)
        scale_h = float(self.window_height) / float(img_height)
        scale = min(scale_w, scale_h)

        if scale == 0:
            scale = 1

        image = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, bpc = image.shape
        bpl = bpc * width

        if self.is_processing_enabled:
            finder = self.finder
            finder.load_image(image)
            image, cropped, metadata = finder.draw_external_contours(verbose=True, crop=True)
            self.show_cropped_images(cropped)

        img = QImage(image.data, width, height, bpl, QImage.Format_RGB888)
        self.widgetCamera.setImage(img)

    def show_cropped_images(self, cropped, columns=5):
        self.clearLayout(self.gridLayout_2)

        if cropped is None or not cropped:
            return

        images_size = len(cropped)
        rows = round((images_size / (columns * 1.0)))

        counter_total = 0
        for i in range(rows):
            for j in range(columns):

                if counter_total >= images_size:
                    break

                image = cropped[counter_total]
                counter_total += 1

                try:
                    height, width = image.shape
                except AttributeError:
                    continue

                # print("width: {}, height: {}".format(width, height))
                img = QImage(image.data, width, height, width, QImage.Format_Grayscale8)
                label = QLabel(self)
                pixmap = QPixmap.fromImage(img)
                pixmap = pixmap.scaled(self.scaled_crop_image, self.scaled_crop_image, QtCore.Qt.KeepAspectRatio)
                label.setPixmap(pixmap)
                self.gridLayout_2.addWidget(label, i, j)

    def setup_categories(self):
        self.listWidget.show()

        ls = []
        for i in range(100):
            ls.append("row {}".format(i))

        self.listWidget.addItems(ls)

    def open_camera(self):
        if not self.camera_manager.is_camera_open():
            self.camera_manager.open_camera(self, self.update_frame, fps=self.video_fps)

    def close_camera(self):
        self.camera_manager.close_camera()

    def open_filename_dialog(self):
        options = QFileDialog.Options()
        # fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Image (*.png)", options=options)
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "Image (*.png *.jpg *.jpeg)", options=options)
        return fileName

    def clearLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget() is not None:
                child.widget().deleteLater()
            elif child.layout() is not None:
                self.clearLayout(child.layout())

    def closeEvent(self, event):
        self.camera_manager.close_camera()

class Form(QDialog):

    def __init__(self, parent=None):
        super(Form, self).__init__(parent)
        # Create widgets
        self.edit = QLineEdit("Write my name here")
        self.button = QPushButton("Show Greetings")
        # Create layout and add widgets
        layout = QVBoxLayout()
        layout.addWidget(self.edit)
        layout.addWidget(self.button)
        # Set dialog layout
        self.setLayout(layout)
        # Add button signal to greetings slot
        self.button.clicked.connect(self.greetings)

    # Greets the user
    def greetings(self):
        print ("Hello %s" % self.edit.text())
