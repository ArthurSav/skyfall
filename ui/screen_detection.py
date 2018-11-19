#!/usr/local/bin/python3
import os

import cv2
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QImage, QMovie
from PyQt5.QtWidgets import QMainWindow, QFileDialog

from engine.contours import ContourFinder
from engine.training import ModelCreator
from models.model_utils import ContourType
from ui.widgets import ImageWidget, ImageGridLayout
from utils.utils_camera import CameraManager

dir_path = os.path.dirname(os.path.realpath(__file__))
screen_detection_ui = uic.loadUiType(dir_path + '/screen_detection.ui')[0]


class ScreenDetection(QMainWindow, screen_detection_ui):
    camera_manager = CameraManager()
    finder = ContourFinder()

    is_processing_enabled = False
    # cropped images displayed size
    scale_dimen = 150
    video_fps = 5

    creator = ModelCreator('data')

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)

        self.window_width = self.widgetCamera.frameSize().width()
        self.window_height = self.widgetCamera.frameSize().height()
        self.widgetCamera = ImageWidget(self.widgetCamera)
        self.gridLayout_2 = ImageGridLayout(self.gridLayout_2, columns=5)
        self.__setup_loader()

        self.btnGenerate.clicked.connect(self.show_loader)
        self.btnLive.clicked.connect(self.__on_click_recording)
        self.btnPicture.clicked.connect(self.__on_click_picture)

        self.invalidate_displayed_models()

        # start recording by default
        self.open_camera()

    def update_frame(self, frame):
        """
        Use it to display either a video feed or just a regular image
        :param frame: image to be displayed
        """

        img_height, img_width, img_colors = frame.shape
        scale_w = float(self.window_width) / float(img_width)
        scale_h = float(self.window_height) / float(img_height)
        scale = min(scale_w, scale_h)

        if scale == 0:
            scale = 1

        if self.is_processing_enabled:
            finder = self.finder
            finder.load_image(frame)

            image, cropped, metadata = finder.draw_and_crop_contours(ContourType.MOBILE, verbose=True, crop=True)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            self.gridLayout_2.add_images(cropped, scale_width=self.scale_dimen, scale_height=self.scale_dimen,
                                         replace=True, is_checkable=False)
            self.on_cropped_images_updated(cropped, metadata)
        else:
            image = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, bpc = image.shape
        bpl = bpc * width

        img = QImage(image.data, width, height, bpl, QImage.Format_RGB888)
        self.widgetCamera.setImage(img)

    def __setup_loader(self):
        widget_movie = QMovie("assets/loader.gif")
        widget_movie.setScaledSize(QSize(64, 64))
        self.widget_progress = widget_movie
        self.label.setMovie(widget_movie)
        self.label.show()

        widget_movie.start()
        widget_movie.stop()

    def show_loader(self):
        if self.widget_progress.state() == self.widget_progress.Running:
            self.widget_progress.stop()
        else:
            self.widget_progress.start()


    def on_cropped_images_updated(self, images, metadata):
        pass

    def invalidate_displayed_models(self):
        self.listWidget.show()
        self.listWidget.clear()

        folders, prefixed = self.creator.list_model_folders()

        # folder names
        names = [os.path.splitext(os.path.basename(name))[0] for name in folders]
        self.listWidget.addItems(names)

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

    def open_filename_dialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "Image (*.png *.jpg *.jpeg)", options=options)
        return fileName

    def open_camera(self):
        if not self.camera_manager.is_camera_open():
            self.camera_manager.open_camera(self, self.update_frame, fps=self.video_fps)

    def close_camera(self):
        self.camera_manager.close_camera()

    def closeEvent(self, event):
        self.camera_manager.close_camera()
