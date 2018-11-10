import queue as Queue

import cv2
import threading
from PyQt5 import QtCore


class CameraManager:
    """
    Handles camera actions in an async thread for smooth experience.

    To capture camera frames we use a new thread which then fills a queue. 
    We periodically get new frames from the queue
    """

    queue = Queue.Queue()
    capture = None
    is_running = False

    video_thread = None
    timer = None

    camera_id = 1

    view = None

    def __init__(self):
        pass

    def open_camera(self, ui, callback, width=1920, height=1080, fps=10):
        """
        camera operations will run in a new thread

        ui: anything that extends QMainWindow (runs QTimer)
        callback(frame): a callback to be notified when a frame is available. Accepts only one argument which is the frame
        width: camera view width
        height: camera view height
        fps: camera fps
        """

        self.video_thread = threading.Thread(target=self.__open_camera, args=(width, height, fps))

        # timer that updates frames from queue periodically
        # todo - find a replacement for this timer since it requires the QT ui
        self.timer = QtCore.QTimer(ui)
        self.timer.timeout.connect(lambda: self.__on_frame_update(callback))
        self.timer.start(1)

        self.video_thread.start()

    def __open_camera(self, width, height, fps, flip=True):
        """
        Simple camera operation with openCV
        note: don't forget to close the camera with close_camera()
        """

        self.is_running = True

        self.capture = cv2.VideoCapture(self.camera_id)

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)

        while (self.is_running):
            ret, frame = self.capture.read()

            # hold a limited number of frames in queue
            if self.queue.qsize() >= 10:
                self.__clear_queue()

            if flip:
                frame = cv2.flip(frame, 1)

            self.queue.put(frame)

        print("Closing camera...")
        self.capture.release()
        cv2.destroyAllWindows()

    def __on_frame_update(self, callback):
        if self.queue.empty():
            return

        frame = self.queue.get()
        if frame is not None:
            callback(frame)

    def close_camera(self):

        # stop video
        self.is_running = False

        self.__clear_queue()

        # stop updating frames
        if self.timer:
            self.timer.stop()
            self.timer = None

    def __clear_queue(self):
        with self.queue.mutex:
            self.queue.queue.clear()

    def is_camera_open(self):
        return self.is_running
