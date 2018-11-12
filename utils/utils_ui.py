from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel
from PyQt5 import QtCore


class LayoutUtils:
    @staticmethod
    def remove_children(self, layout):
        """
        Removes children from parent layout
        :param layout Qwidget
        """
        while layout.count():
            child = layout.takeAt(0)
            if child.widget() is not None:
                child.widget().deleteLater()
            elif child.layout() is not None:
                self.clear_layout(child.layout())

    @staticmethod
    def add_images_to_gridlayout(self, gridlayout, images, columns=4, scale_width=None, scale_height=None,
                                 replace=True):
        """
        Adds images into a gridlayout
        :param gridlayout:
        :param images: numpy image list
        :param columns: grid columns
        :param scale_width: width to rescale images
        :param scale_height: height to rescale images
        :param replace: if true, it will remove previous images
        :return:
        """

        if replace:
            LayoutUtils.remove_children(self, gridlayout)

        if images is None or not images:
            return

        images_size = len(images)
        rows = round((images_size / (columns * 1.0)))

        counter_total = 0
        for i in range(rows):
            for j in range(columns):
                if counter_total >= images_size:
                    break

                image = images[counter_total]
                counter_total += 1

                try:
                    height, width = image.shape
                except AttributeError:
                    continue

                # print("width: {}, height: {}".format(width, height))
                img = QImage(image.data, width, height, width, QImage.Format_Grayscale8)
                label = QLabel(self)
                pixmap = QPixmap.fromImage(img)

                # rescale image
                if scale_width is not None and scale_height is not None:
                    pixmap = pixmap.scaled(scale_width, scale_height, QtCore.Qt.KeepAspectRatio)

                label.setPixmap(pixmap)

                gridlayout.addWidget(label, i, j)
