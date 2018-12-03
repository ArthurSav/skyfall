#!/usr/local/bin/python3

import os
import queue as Queue

from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QDialog

from engine.contours import ContourType
from engine.model_manager import ModelCreator, DataLoader
from ui import PATH_MODEL_EXPORT
from ui.widgets import ImageWidget, ImageGridLayout

# load ui file
from utils.utils_camera import CameraHelper

dir_path = os.path.dirname(os.path.realpath(__file__))
screen_edit_model_ui = uic.loadUiType(dir_path + '/screen_edit_model.ui')[0]
dialog_add_model_ui = uic.loadUiType(dir_path + "/dialog_add_component.ui")[0]


class ScreenEditModel(QMainWindow, screen_edit_model_ui):
    """
    UI manager for screen 'Add or Edit model'
    """

    camera_helper = None
    cropped_images = None

    creator = ModelCreator(PATH_MODEL_EXPORT)

    # cropped images displayed size
    scale_dimen = 150

    loader = DataLoader()

    cropped_queue = Queue.Queue()

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)

        self.creator.create_model_dir()
        self.invalidate_displayed_components()

        self.window_width = self.widgetCamera.frameSize().width()
        self.window_height = self.widgetCamera.frameSize().height()
        self.widgetCamera = ImageWidget(self.widgetCamera)
        self.gridLayout_2 = ImageGridLayout(self.gridLayout_2, columns=5)

        self.btnLive.clicked.connect(self.__on_click_recording)
        self.btnPicture.clicked.connect(self.__on_click_picture)

        self.btnComponentAdd.clicked.connect(self.show_dialog_add_component)
        self.btnComponentRemove.clicked.connect(self.remove_selected_component)

        self.btnSave.clicked.connect(self.train_model)

        self.checkBoxContours.stateChanged \
            .connect(lambda: self.__on_checkbox_contours_changed(self.checkBoxContours.isChecked()))
        self.checkBoxCrop.stateChanged \
            .connect(lambda: self.__on_checkbox_cropped_changed(self.checkBoxCrop.isChecked()))

        # start recording by default
        self.camera_helper = CameraHelper(self.window_width, self.window_height,
                                          self.__on_image_updated,
                                          self.__on_image_cropped)

        self.camera_helper.set_contour_type(ContourType.TRAINING)
        self.camera_helper.set_contours_enabled(self.checkBoxContours.isChecked())
        self.camera_helper.set_contours_cropping_enabled(self.checkBoxCrop.isChecked())
        self.checkBoxCrop.setEnabled(self.checkBoxContours.isChecked())

        self.__start_cropped_queue_timer()
        self.camera_helper.open_camera()

    def __start_cropped_queue_timer(self):
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.__process_cropped)
        self.timer.start(1)

    def __on_image_updated(self, qImage):
        self.widgetCamera.setImage(qImage)

    def __on_image_cropped(self, cropped, metadata):
        if cropped:
            self.cropped_queue.put(cropped)

    def __process_cropped(self):
        if not self.cropped_queue.empty():
            cropped = self.cropped_queue.get()
            self.cropped_images = cropped
            self.gridLayout_2.add_images(cropped, scale_width=self.scale_dimen, scale_height=self.scale_dimen,
                                         replace=True, is_checkable=False)

    def __on_checkbox_contours_changed(self, is_checked):
        self.camera_helper.set_contours_enabled(is_checked)
        self.checkBoxCrop.setEnabled(is_checked)

    def __on_checkbox_cropped_changed(self, is_checked):
        self.camera_helper.set_contours_cropping_enabled(is_checked)

    def __on_click_recording(self):
        self.camera_helper.open_camera()

    def __on_click_picture(self):
        """
        Opens file picker + loads image into imageview
        """
        path = self.open_filename_dialog()
        if path:
            self.camera_helper.load_image(path)

    def save_component_images(self, name, images):
        self.creator.save_component(name, images, replace=True, verbose=True)
        self.invalidate_displayed_components()

    def show_dialog_add_component(self):
        form = DialogAddComponent(self.save_component_images)
        form.show()
        form.show_images(self.cropped_images)
        form.exec_()

    def invalidate_displayed_components(self):
        """
        Refreshes displayed component list
        """

        self.listWidget.show()
        self.listWidget.clear()
        components = self.creator.list_model_components()
        ls = []
        for component in components:
            ls.append("{} ({})".format(component['name'], len(component['images'])))

        self.listWidget.addItems(ls)

    def remove_selected_component(self):
        """
        Removes currently selected component
        """
        selected_item = self.listWidget.currentItem()
        if selected_item is None:
            return

        component_name = selected_item.text()
        is_removed = self.creator.remove_component(component_name)
        if is_removed:
            print("Removed component: {}".format(component_name))
        self.invalidate_displayed_components()

    def open_filename_dialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "Image (*.png *.jpg *.jpeg)", options=options)
        return fileName

    def train_model(self):
        name = self.lineEdit.text()
        if not name:
            print("Please provide a valid model name")
            return
        self.creator.train(name)

    def closeEvent(self, event):
        self.camera_helper.close_camera()


class DialogAddComponent(QDialog, dialog_add_model_ui):
    """
    A dialog to add currently cropped images into the component list
    """

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
        """
        Saves displayed images
        :param callback_save: actual function that stores images
        """

        name = self.lineEditName.text()
        images = self.displayed_images

        if self.gridLayout.is_checkable():
            selected_positions = self.gridLayout.get_selected()
            images = [image for i, image in enumerate(images) if
                      any(i == selected for selected in selected_positions)]

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
        """
        Shows images into dialog grid
        :param images: a list of numpy images
        """
        self.displayed_images = images
        self.gridLayout.add_images(images, scale_width=self.scale_dimen, scale_height=self.scale_dimen, replace=True,
                                   is_checkable=True, is_preselected=True)
