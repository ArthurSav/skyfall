#!/usr/local/bin/python3
import os
import queue as Queue
import threading
import time

import cv2
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QImage, QMovie
from PyQt5.QtWidgets import QMainWindow, QFileDialog

from converters.converter import Converter
from engine.contours import ContourFinder
from engine.contours import ContourType
from engine.model_manager import ModelCreator
from ui import PATH_MODEL_EXPORT
from ui.widgets import ImageWidget, ImageGridLayout
from utils.utils_camera import CameraManager

dir_path = os.path.dirname(os.path.realpath(__file__))
screen_detection_ui = uic.loadUiType(dir_path + '/screen_detection.ui')[0]


class CameraHelper:
    camera = CameraManager()
    finder = ContourFinder()

    is_contour_detection_enabled = False
    is_contour_cropping_enabled = False

    callback_on_image_updated = None
    callback_on_image_cropped = None

    image_queue = Queue.Queue()

    fps = 5

    def __init__(self, window_width, window_height, callback_on_image_updated, callback_on_image_cropped):
        self.window_width = window_width
        self.window_height = window_height
        self.callback_on_image_updated = callback_on_image_updated
        self.callback_on_image_cropped = callback_on_image_cropped

        self.__processor_thread = threading.Thread(target=self.__queue_processor, args=())
        self.__processor_thread.daemon = True
        self.__processor_thread.start()

    def __queue_processor(self):
        while True:
            frame = None

            # camera frames
            if self.camera.is_camera_open():
                if self.camera.has_frames():
                    frame = self.camera.get_next_frame()

            # other sources
            elif not self.image_queue.empty():
                frame = self.image_queue.get()

            if frame is not None:
                self.__process_frame(frame)

    def __process_frame(self, frame):

        if self.is_contour_detection_enabled:
            frame, cropped, metadata = self.__apply_contours(frame)
            self.callback_on_image_cropped(cropped, metadata)

        frame = self.__resize_to_window(frame)
        qimage = self.export_to_qimage(frame)

        self.callback_on_image_updated(qimage)

    def __apply_contours(self, frame):
        finder = self.finder
        finder.load_image(frame)
        image, cropped, metadata = self.finder.draw_and_crop_contours(ContourType.MOBILE, verbose=True,
                                                                      crop=self.is_contour_cropping_enabled)
        return image, cropped, metadata

    def __resize_to_window(self, frame):
        img_height, img_width, img_colors = frame.shape
        scale_w = float(self.window_width) / float(img_width)
        scale_h = float(self.window_height) / float(img_height)
        scale = min(scale_w, scale_h)

        if scale == 0:
            scale = 1

        image = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def clear_queue(self):
        with self.queue.mutex:
            self.queue.queue.clear()

    def is_contours_enabled(self):
        return self.is_contour_detection_enabled

    def set_contours_enabled(self, is_enabled):
        self.is_contour_detection_enabled = is_enabled

    def set_contours_cropping_enabled(self, is_enabled):
        self.is_contour_cropping_enabled = is_enabled

    def load_image(self, filepath):

        if not os.path.isfile(filepath):
            raise Exception("Could not load image with path {}".format(filepath))
        self.close_camera()

        image = cv2.imread(filepath)
        self.image_queue.put(image)

    def open_camera(self):

        if self.camera.is_camera_open():
            return

        self.camera.open_camera(fps=self.fps)

    def close_camera(self):
        self.camera.close_camera()

    @staticmethod
    def export_to_qimage(frame):
        height, width, bpc = frame.shape
        bpl = bpc * width
        return QImage(frame.data, width, height, bpl, QImage.Format_RGB888)


class TemplateGeneratorHelper:
    image_queue = Queue.Queue()
    is_processing_enabled = False

    processing_thread = None

    converter = Converter()
    creator = None

    PROCESSING_INTERVAL = 2

    def __init__(self, creator):
        """
        :param creator: ModelCreator
        """
        self.creator = creator
        self.converter.set_ouput("Test.js")

        self.__start_processing_thread()

    def __start_processing_thread(self):
        self.is_processing_enabled = True
        self.processing_thread = threading.Thread(target=self.__process_queue, args=())
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def load_images(self, images, metadata):

        if images is None or not images:
            return

        # clear queue
        if self.image_queue.qsize() >= 2:
            self.__clear_queue()

        self.image_queue.put({'cropped': images, 'metadata': metadata})

    def __process_queue(self):
        while self.is_processing_enabled:
            if not self.image_queue.empty():
                data = self.image_queue.get()
                images = data['cropped']
                metadata = data['metadata']

                # needed when using different threads
                with self.creator.graph.as_default():
                    # identify components
                    results = self.creator.predict(images, metadata)

                # generate code
                self.converter.convert(results)

            time.sleep(self.PROCESSING_INTERVAL)

    def __clear_queue(self):
        with self.image_queue.mutex:
            self.image_queue.queue.clear()

    def set_output(self, filepath):
        self.converter.set_ouput(filepath)

    def close_thread(self):
        self.is_processing_enabled = False


class ScreenDetection(QMainWindow, screen_detection_ui):
    camera_helper = None

    creator = ModelCreator(PATH_MODEL_EXPORT)
    generatorHelper = None

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
        self.camera_helper.open_camera()

    def __on_image_updated(self, qImage):
        self.widgetCamera.setImage(qImage)

    def __on_image_cropped(self, cropped, metadata):
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
        """
        Opens camera and displays video
        """

        QtWidgets.QApplication.processEvents()

        self.camera_helper.open_camera()

        # start/stop video analysis
        if self.is_contour_processing_enabled:
            self.btnLive.setText("Start recording")
            self.is_contour_processing_enabled = False
        else:
            self.btnLive.setText("Stop recording")
            self.is_contour_processing_enabled = True

    def __on_click_picture(self):
        """
        Opens file picker + loads image into imageview
        """
        path = self.open_filename_dialog()
        self.camera_helper.load_image(path)

    def __on_checkbox_contours_changed(self, is_checked):
        self.camera_helper.set_contours_enabled(is_checked)
        self.checkBoxCrop.setEnabled(is_checked)
        if is_checked:
            self.checkBoxCrop.setChecked(is_checked)

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
