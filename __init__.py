
""" Project Skyfall - Mobile interface segmentation """

__version__ = '0.0.1'
__author__ = 'Arthur Saveliev'

__all__ = ['utils', 'Loader', 'ContourFinder', 'CropType']

from skyfall import data_loading
from skyfall import contours
from skyfall import utils

from data_loading import Loader
from contours import ContourFinder
from contours import CropType