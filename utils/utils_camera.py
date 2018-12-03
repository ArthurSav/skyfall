import os
import queue as Queue
import threading
import time

import cv2
from PyQt5.QtGui import QImage

from engine.contours import ContourFinder, ContourType


class CameraManager:
    """
    Handles camera actions in an async thread for smooth experience.

    To capture camera frames we use a new thread which then fills a queue. 
    We periodically get new frames from the queue
    """

    __queue = Queue.Queue()
    __capture = None
    __is_running = False
    __video_thread = None

    camera_id = 0

    def open_camera(self, width=1920, height=1080, fps=10):
        """
        camera operations will run in a new thread

        ui: anything that extends QMainWindow (runs QTimer)
        callback(frame): a callback to be notified when a frame is available. Accepts only one argument which is the frame
        width: camera view width
        height: camera view height
        fps: camera fps
        """

        self.__video_thread = threading.Thread(target=self.__open_camera, args=(width, height, fps))
        self.__video_thread.start()

    def __open_camera(self, width, height, fps, flip=True):
        """
        Simple camera operation with openCV
        note: don't forget to close the camera with close_camera()
        """

        self.__is_running = True

        self.__capture = cv2.VideoCapture(self.camera_id)
        self.__capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.__capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.__capture.set(cv2.CAP_PROP_FPS, fps)

        self.clear_queue()

        while self.__is_running:
            ret, frame = self.__capture.read()

            # hold a limited number of frames in queue
            if self.__queue.qsize() >= 10:
                self.clear_queue()

            if flip:
                frame = cv2.flip(frame, 1)

            self.__queue.put(frame)

        print("Closing camera...")
        self.__capture.release()
        cv2.destroyAllWindows()

    def restart_camera(self):
        self.close_camera()
        self.open_camera()

    def set_camera_id(self, camera_id):
        self.camera_id = camera_id

    def get_queue(self):
        return self.__queue

    def has_frames(self):
        return not self.__queue.empty()

    def get_next_frame(self):
        return self.__queue.get()

    def close_camera(self):
        self.__is_running = False
        self.clear_queue()

    def clear_queue(self):
        with self.__queue.mutex:
            self.__queue.queue.clear()

    def is_camera_open(self):
        return self.__is_running


class CameraHelper:
    camera = CameraManager()
    finder = ContourFinder()

    is_contour_detection_enabled = False
    is_contour_cropping_enabled = False

    callback_on_image_updated = None
    callback_on_image_cropped = None

    image_queue = Queue.Queue()
    cropped_queue = Queue.Queue()

    contour_type = None

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

            time.sleep(0.01)

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

        if self.contour_type == ContourType.MOBILE:
            image, cropped, metadata = self.finder.draw_and_crop_contours(ContourType.MOBILE, verbose=True,
                                                                          crop=self.is_contour_cropping_enabled)
        else:
            image, cropped, metadata = self.finder.draw_and_crop_contours(ContourType.TRAINING, verbose=True,
                                                                          crop=self.is_contour_cropping_enabled)
        return image, cropped, metadata

    def __resize_to_window(self, frame):

        if len(frame.shape) == 3:
            img_height, img_width, img_colors = frame.shape
        else:
            img_height, img_width = frame.shape

        scale_w = float(self.window_width) / float(img_width)
        scale_h = float(self.window_height) / float(img_height)
        scale = min(scale_w, scale_h)

        if scale == 0:
            scale = 1

        image = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def set_contour_type(self, contour_type):
        """
        :param contour_type: ContourType
        """
        self.contour_type = contour_type

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
        bpc = 1
        if len(frame.shape) == 3:
            height, width, bpc = frame.shape
        else:
            height, width = frame.shape
        bpl = bpc * width
        return QImage(frame.data, width, height, bpl, QImage.Format_RGB888)


class ImageHelper:

    @staticmethod
    def apply_contours(image):
        # convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 10, 250)

        # apply closing function
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        # find contours
        img_contour, contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            # rectangle coordinates
            x, y, w, h = cv2.boundingRect(c)

            # draw contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)

        return image
