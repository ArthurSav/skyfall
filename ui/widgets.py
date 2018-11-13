#!/usr/local/bin/python3

from PyQt5 import QtCore
from PyQt5.QtGui import QPainter, QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QGridLayout, QVBoxLayout, QCheckBox, QLabel


class ImageWidget(QWidget):
    def __init__(self, parent=None):
        super(ImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()


class ImageGridLayout:

    """
    Makes it easy to add images to a gridlayout
    p.s couldn't extend the actual gridlayout so we just use a reference of it
    """

    gridLayout = None

    def __init__(self, parent = None, columns = 4):
        self.gridLayout = parent
        self.columns = columns

    def add_images(self, images, scale_width=None, scale_height=None, replace=False, is_checkable=False, is_preselected = True):

        if replace:
            self.remove_children()

        if images is None or not images:
            return

        columns = self.columns
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
                label = QLabel()
                pixmap = QPixmap.fromImage(img)

                # rescale image
                if scale_width is not None and scale_height is not None:
                    pixmap = pixmap.scaled(scale_width, scale_height, QtCore.Qt.KeepAspectRatio)

                label.setPixmap(pixmap)

                if is_checkable:
                    checked = CheckedImage()
                    checked.add_image(label, is_preselected)
                    gridWidget = checked
                else:
                    gridWidget = label

                self.gridLayout.addWidget(gridWidget, i, j, QtCore.Qt.AlignTop)

    def remove_children(self):
        while self.gridLayout.count():
            child = self.gridLayout.takeAt(0)
            if child.widget() is not None:
                child.widget().deleteLater()
            elif child.layout() is not None:
                self.remove_children(child.layout())


class CheckedImage(QWidget):
    checkbox = None
    imageWidget = None

    def __init__(self, parent=None):
        QWidget.__init__(self, parent=parent)
        self.lay = QVBoxLayout(self)

    def add_image(self, widget, is_checked=False):
        self.lay.addWidget(widget)
        checkbox = QCheckBox(self)
        checkbox.setChecked(is_checked)
        self.lay.addWidget(checkbox)

        self.imageWidget = widget
        self.checkbox = checkbox

    def is_checked(self):
        if self.checkbox is not None:
            return self.checkbox.isChecked()
        return False
