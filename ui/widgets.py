#!/usr/local/bin/python3

from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QImage
from PyQt5 import QtCore

class ImageWidget(QWidget):
    def __init__(self, parent = None):
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