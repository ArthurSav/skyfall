#!/usr/local/bin/python3

from PyQt5 import QtWidgets

from ui.screen_detection import ScreenDetection

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = ScreenDetection(None)
    window.setWindowTitle('Mobile screen detection')
    window.show()
    sys.exit(app.exec_())
