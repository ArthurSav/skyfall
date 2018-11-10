#!/usr/local/bin/python3

import cv2
import os
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QMainWindow

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

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)

        self.window_width = 700
        self.window_height = 700
        # self.window_width = self.widgetCamera.frameSize().width()
        # self.window_height = self.widgetCamera.frameSize().height()
        # print("Height:{}, Width:{}".format(self.widgetCamera.frameSize().height(), self.widgetCamera.frameSize().width()))
        self.widgetCamera = ImageWidget(self.widgetCamera)

        self.btnLive.clicked.connect(self.__on_click_recording)
        self.btnPicture.clicked.connect(self.__on_click_picture)

        # start recording by default
        self.__on_click_recording()

    def __on_click_recording(self):
        """
        Opens camera and displays video
        """
        QtWidgets.QApplication.processEvents()

        # close if open and vice versa
        if self.camera_manager.is_camera_open():
            self.btnLive.setText("Start recording")
            self.camera_manager.close_camera()
        else:
            self.btnLive.setText("Stop recording")
            self.camera_manager.open_camera(self, self.update_frame, fps=5)

    def __on_click_picture(self):
        """
        Opens file picker + loads image into imageview
        """
        QtWidgets.QApplication.processEvents()

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

        finder = self.finder
        finder.load_image(image)

        # image = finder.draw_external_contours(verbose=True, crop=True)
        image, cropped, metadata = finder.draw_external_contours(verbose=True, crop=True)

        img = QImage(image.data, width, height, bpl, QImage.Format_RGB888)
        self.widgetCamera.setImage(img)

    def closeEvent(self, event):
        self.camera_manager.close_camera()
