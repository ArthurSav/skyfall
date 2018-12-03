import queue as Queue
import threading
import time

from converters.converter import Converter


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
        self.converter.set_ouput("/Users/artursaveljev/lab/my-expo-project/screens/GeneratedScreen.js")

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