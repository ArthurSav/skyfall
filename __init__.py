
""" Project Skyfall - Mobile interface segmentation """

__version__ = '0.0.1'
__author__ = 'Arthur Saveliev'

__all__ = ['Loader', 'ContourFinder', 'CropType', 'DisplaySize', 'utils_train', 'utils_image']

from skyfall.utils_image import DisplaySize
from skyfall import utils_image
from skyfall import utils_train

from skyfall.data_loading import *
from skyfall.contours import *