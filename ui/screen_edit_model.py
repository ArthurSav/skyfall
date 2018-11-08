#!/usr/local/bin/python3

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow
import os 

# load ui file
dir_path = os.path.dirname(os.path.realpath(__file__))
screen_edit_model_ui = uic.loadUiType(dir_path + '/screen_edit_model.ui')[0]

class ScreenEditModel(QMainWindow, screen_edit_model_ui):

     def __init__(self, parent = None):
         QMainWindow.__init__(self, parent)
         self.setupUi(self)