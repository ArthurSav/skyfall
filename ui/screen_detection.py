#!/usr/local/bin/python3
import os
import queue as Queue

from PyQt5 import uic, QtCore
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QMainWindow, QFileDialog

from skyfall.engine.contours import ContourType
from skyfall.engine.model_manager import ModelCreator
from skyfall.utils.utils_templates import TemplateGeneratorHelper
from ui import PATH_MODEL_EXPORT
from ui.widgets import ImageWidget, ImageGridLayout
from skyfall.utils.utils_camera import CameraHelper

dir_path = os.path.dirname(os.path.realpath(__file__))
screen_detection_ui = uic.loadUiType(dir_path + '/screen_detection.ui')[0]


class ScreenDetection(QMainWindow, screen_detection_ui):
    camera_helper = None

    creator = ModelCreator(PATH_MODEL_EXPORT)
    generatorHelper = None

    cropped_queue = Queue.Queue()

    # cropped images displayed size
    scale_dimen = 150

    STATE_AUTOMATIC = 0
    STATE_MANUAL = 1

    state_current = STATE_AUTOMATIC

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)

        self.window_width = self.widgetCamera.frameSize().width()
        self.window_height = self.widgetCamera.frameSize().height()
        self.widgetCamera = ImageWidget(self.widgetCamera)
        self.gridLayout_2 = ImageGridLayout(self.gridLayout_2, columns=5)

        self.radioAutomatic.toggled.connect(lambda: self.__set_code_generation_state(self.STATE_AUTOMATIC))
        self.radioManual.toggled.connect(lambda: self.__set_code_generation_state(self.STATE_MANUAL))

        # self.btnGenerate.clicked.connect(self.set_loading_state)
        self.btnLive.clicked.connect(self.__on_click_recording)
        self.btnPicture.clicked.connect(self.__on_click_picture)
        self.listWidget.itemSelectionChanged.connect(self.on_list_item_select)

        self.checkBoxContours.stateChanged \
            .connect(lambda: self.__on_checkbox_contours_changed(self.checkBoxContours.isChecked()))
        self.checkBoxCrop.stateChanged \
            .connect(lambda: self.__on_checkbox_cropped_changed(self.checkBoxCrop.isChecked()))

        self.__setup_component_list()
        self.__setup_code_generation_progress_loader()
        self.__set_code_generation_state(self.STATE_AUTOMATIC)

        self.generatorHelper = TemplateGeneratorHelper(self.creator)

        # start recording by default
        self.camera_helper = CameraHelper(self.window_width, self.window_height,
                                          self.__on_image_updated,
                                          self.__on_image_cropped)

        self.camera_helper.set_contour_type(ContourType.MOBILE)
        self.camera_helper.set_contours_enabled(self.checkBoxContours.isChecked())
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
            self.cropped_queue.put({'cropped': cropped, 'metadata': metadata})

    def __process_cropped(self):
        if not self.cropped_queue.empty():
            item = self.cropped_queue.get()
            metadata = item['metadata']
            cropped = item['cropped']
            self.gridLayout_2.add_images(cropped, scale_width=self.scale_dimen, scale_height=self.scale_dimen,
                                         replace=True, is_checkable=False)
            self.generatorHelper.load_images(cropped, metadata)

    def __setup_code_generation_progress_loader(self):

        widget_movie = QMovie("assets/loader.gif")
        widget_movie.setScaledSize(QSize(64, 64))

        self.widget_progress = widget_movie
        self.label.setMovie(widget_movie)
        self.label.show()

        widget_movie.start()
        widget_movie.stop()

    def __setup_component_list(self):
        self.invalidate_displayed_models()

        # preselect last model available
        preselected_position = 0
        if self.listWidget.count() > 0:
            preselected_position = self.listWidget.count() - 1
        self.listWidget.setCurrentRow(preselected_position)

    def __set_loading_indicator_state(self, is_loading):
        if is_loading and not self.__is_loading_indicator_active():
            self.widget_progress.start()

        elif not is_loading and self.__is_loading_indicator_active():
            self.widget_progress.stop()

    def __is_loading_indicator_active(self):
        """
        :return: true if progress widget is running
        """
        return self.widget_progress.state() == self.widget_progress.Running

    def __on_click_recording(self):
        self.camera_helper.open_camera()

    def __on_click_picture(self):
        """
        Opens file picker + loads image into imageview
        """
        path = self.open_filename_dialog()
        if path:
            self.camera_helper.load_image(path)

    def __on_checkbox_contours_changed(self, is_checked):
        self.camera_helper.set_contours_enabled(is_checked)
        self.checkBoxCrop.setEnabled(is_checked)

    def __on_checkbox_cropped_changed(self, is_checked):
        self.camera_helper.set_contours_cropping_enabled(is_checked)

    def open_filename_dialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "Image (*.png *.jpg *.jpeg)", options=options)
        return fileName

    def on_list_item_select(self):
        selected_item = self.listWidget.currentItem()
        if selected_item is None:
            return
        model_name = selected_item.text()
        self.creator.load_model(model_name)

    def invalidate_displayed_models(self):
        self.listWidget.show()
        self.listWidget.clear()

        folders, prefixed = self.creator.list_model_folders()

        # folder names
        names = [os.path.splitext(os.path.basename(name))[0] for name in folders]
        self.listWidget.addItems(names)

    def __set_code_generation_state(self, state):
        self.code_gen_state = state

        if state == self.STATE_AUTOMATIC:
            self.btnGenerate.setEnabled(False)
        elif state == self.STATE_MANUAL:
            self.btnGenerate.setEnabled(True)

    def closeEvent(self, event):
        self.camera_helper.close_camera()
        self.generatorHelper.close_thread()
