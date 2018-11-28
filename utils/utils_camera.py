import queue as Queue
import threading

import cv2


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
