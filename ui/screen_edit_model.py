#!/usr/local/bin/python3

import os

import cv2
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QDialog

from engine.contours import ContourFinder
from engine.data_training import TrainingModelCreator
from ui.widgets import ImageWidget, ImageGridLayout
from utils.utils_camera import CameraManager

# load ui file

dir_path = os.path.dirname(os.path.realpath(__file__))
screen_edit_model_ui = uic.loadUiType(dir_path + '/screen_edit_model.ui')[0]
dialog_add_model_ui = uic.loadUiType(dir_path + "/dialog_add_component.ui")[0]


class ScreenEditModel(QMainWindow, screen_edit_model_ui):
    """
    UI manager for screen 'Add or Edit model'
    """

    camera_manager = CameraManager()
    finder = ContourFinder()

    is_processing_enabled = False

    # cropped images displayed size
    scale_dimen = 150
    video_fps = 5

    cropped_images = None

    creator = TrainingModelCreator('data')

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)

        self.setup_categories()

        # self.window_width = 940
        # self.window_height = 1500
        self.window_width = self.widgetCamera.frameSize().width()
        self.window_height = self.widgetCamera.frameSize().height()
        self.widgetCamera = ImageWidget(self.widgetCamera)
        self.gridLayout_2 = ImageGridLayout(self.gridLayout_2, columns=5)

        self.btnLive.clicked.connect(self.__on_click_recording)
        self.btnPicture.clicked.connect(self.__on_click_picture)

        self.btnComponentAdd.clicked.connect(self.show_dialog_add_component)

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

    def save_component_images(self, name, images):
        self.creator.save_component(name, images, replace=True, verbose=True)

    def show_dialog_add_component(self):
        form = DialogAddComponent(self.save_component_images)
        form.show()
        form.show_images(self.cropped_images)
        form.exec_()

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
            self.cropped_images = cropped

            self.gridLayout_2.add_images(cropped, scale_width=self.scale_dimen, scale_height=self.scale_dimen,
                                         replace=True, is_checkable=False)

        img = QImage(image.data, width, height, bpl, QImage.Format_RGB888)
        self.widgetCamera.setImage(img)

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
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "Image (*.png *.jpg *.jpeg)", options=options)
        return fileName

    def closeEvent(self, event):
        self.camera_manager.close_camera()


class DialogAddComponent(QDialog, dialog_add_model_ui):

    # displayed image dimension
    scale_dimen = 100
    displayed_images = None

    def __init__(self, callback_save, parent=None):
        super(DialogAddComponent, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Add component")

        self.buttonBox.accepted.connect(lambda: self.save(callback_save))
        self.buttonBox.rejected.connect(self.reject)

        self.gridLayout = ImageGridLayout(self.gridLayout, columns=5)

    def save(self, callback_save):

        name = self.lineEditName.text()
        images = self.displayed_images

        if self.displayed_images is None:
            error_dialog = QtWidgets.QErrorMessage()
            error_dialog.showMessage('No images to save')
            error_dialog.show()
            error_dialog.exec_()
            return
        if not name:
            error_dialog = QtWidgets.QErrorMessage()
            error_dialog.showMessage('Please set a component name')
            error_dialog.show()
            error_dialog.exec_()
            return

        callback_save(name, images)
        self.close()


    def show_images(self, images):
        self.displayed_images = images
        self.gridLayout.add_images(images, scale_width=self.scale_dimen, scale_height=self.scale_dimen, replace=True,
                                   is_checkable=True, is_preselected=True)
