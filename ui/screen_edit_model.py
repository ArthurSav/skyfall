#!/usr/local/bin/python3

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtGui import QPainter, QImage
from utils.utils_camera import CameraManager
from ui.widgets import ImageWidget
from engine.contours import ContourFinder
from models.model_utils import CropType

import numpy as np
import cv2, os

# load ui file
dir_path = os.path.dirname(os.path.realpath(__file__))
screen_edit_model_ui = uic.loadUiType(dir_path + '/screen_edit_model.ui')[0]

class ScreenEditModel(QMainWindow, screen_edit_model_ui):
    """
    UI manager for screen 'Add or Edit model'
    """

    camera_manager = CameraManager()
    finder = ContourFinder()


    def __init__(self, parent = None):
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

    #############################     

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
            self.camera_manager.open_camera(self, self.update_frame_v2)

    def __on_click_picture(self):
        """
        Opens file picker + loads image into imageview
        """
        QtWidgets.QApplication.processEvents()

    #############################

    def update_frame_v2(self, frame):
        finder = self.finder

        finder.load_image(frame)
        finder.crop(CropType.TRAINING, verbose = True)

        image = finder.image_with_contours

        img_height, img_width, img_colors = image.shape
        scale_w = float(self.window_width) / float(img_width)
        scale_h = float(self.window_height) / float(img_height)
        scale = min(scale_w, scale_h)

        if scale == 0:
            scale = 1

        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, bpc = image.shape
        bpl = bpc * width
        img = QImage(image.data, width, height, bpl, QImage.Format_RGB888)
        self.widgetCamera.setImage(img)



    def update_frame(self, image = None):

        if image is None:
            return
        
        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.drawContours(image, contours, -1, (0,255,0), 3)

        for c in contours:
            # get the bounding rect
            x, y, w, h = cv2.boundingRect(c)
            # draw a green rectangle to visualize the bounding rect

            if w < 80 and h < 80:
                continue

            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        img_height, img_width, img_colors = image.shape
        scale_w = float(self.window_width) / float(img_width)
        scale_h = float(self.window_height) / float(img_height)
        scale = min(scale_w, scale_h)

        if scale == 0:
            scale = 1

        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, bpc = image.shape
        bpl = bpc * width
        img = QImage(image.data, width, height, bpl, QImage.Format_RGB888)
        self.widgetCamera.setImage(img)

    def closeEvent(self, event):
        self.camera_manager.close_camera()
            
        

        