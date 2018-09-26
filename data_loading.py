import cv2
from glob import glob
import pandas as pd
import numpy as np
import os
from skimage.transform import rescale, resize, downscale_local_mean, pyramid_reduce
from tensorflow import keras

from skyfall import utils

import matplotlib.pyplot as plt

class Loader:
  
  IMAGE_TYPE = 'png'
  
  def __init__(self, path):
    self.path = path
  
  def load_data(self, split_ratio = 0.2, size = 64, normalize = True, verbose = True, max_images_per_class = None):
    """
    Loads folders from a source directory.
    Folder names become 'class' names.
    
    split_ratio: Accepts 0-1 values. Splits training and testing data
    size: Resizes the image in width=size and height=size (square) while maintaining aspect ratio

    return class names, training data with labels, testing data with labels
    """
    
    classes = self.__load_from_root(self.path)
    
    # class names (should be unique)
    names = []

    # train/test image paths
    train_paths = []
    test_paths = []
    
    y_train = np.empty([0], dtype=int)
    y_test = np.empty([0], dtype=int)
    
    if verbose:
      print("Loading path data...")
      print("Found {} classes".format(len(classes)))
      if max_images_per_class:
        print("Limits to {} images per class".format(max_images_per_class))
    
    for idx, class_data in enumerate(classes):
      class_name = class_data[0]
      class_images = class_data[1]
      
      if max_images_per_class:
        class_images = class_images[:min(max_images_per_class, len(class_images))]

      print("name: {}, images: {}".format(class_name, len(class_images)))
      
      # split into train, test
      image_size = len(class_images)
      vfold_size = int((image_size / 100.0) * (split_ratio * 100.0))
      
      labels = np.full(image_size, idx)
      names.append(class_name)
      
      train = class_images[vfold_size:image_size]
      train_labels = labels[vfold_size: image_size]
      test = class_images[0: vfold_size]
      test_labels = labels[0: vfold_size]
      
      train_paths = np.concatenate((train_paths, train))
      test_paths = np.concatenate((test_paths, test))
      y_train = np.concatenate((y_train, train_labels))
      y_test = np.concatenate((y_test, test_labels))

    number_of_classes = len(names)
      
    # load images into numpy array
    x_train = self.__load_paths_as_array(train_paths, size)
    x_test = self.__load_paths_as_array(test_paths, size)

    # shuffle images to even out distribution during training
    x_train, y_train = self.__shuffle(x_train, y_train)
    x_test, y_test = self.__shuffle(x_test, y_test)
    
    # reahape into [num_of_images, width, height, num_of_color_channels]
    x_train = x_train.reshape(x_train.shape[0], size, size, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], size, size, 1).astype('float32')
    
    if normalize:
      x_train, y_train, x_test, y_test = self.__normalize(x_train, y_train, x_test, y_test, number_of_classes)
 
    print("Exporting size: {}x{}".format(size, size))
    print("x_train: {}, x_test: {}".format(len(x_train), len(x_test))) 
    
    return names, (x_train, y_train), (x_test, y_test)
  
  def __load_paths_as_array(self, paths, size):
    """
    paths: a list of image paths i.e /train/images/hello.jpg ...
    size: image size

    return a numpy array of loaded square images of shape i.e (num, size, size, 1)
    """

    x = np.empty([0, size, size])

    for path in paths:
      image = self.__load_square_image(path, size)
      image = np.invert(image) # white background, black content
      x = np.concatenate((x, image.reshape(1, size, size)))

    return x

  def __shuffle(self, x, y):
    """
    Random shuffle images and their labels
    """

    size = x.shape[0]
    permutation = np.random.permutation(size)

    x = x[permutation, :, :]
    y = y[permutation]

    return x, y
  

  def __normalize(self, x_train, y_train, x_test, y_test, number_of_classes):

    if x_train.dtype != 'float32':
      x_train = x_train..astype('float32')

    if x_test.dtype != 'float32':
      x_test = x_test.astype('float32')

    # normalize images [0, 1]
    x_train = (x_train - np.min(x_train)) / np.ptp(x_train)
    x_test = (x_test - np.min(x_test)) / np.ptp(x_test)

    # normalize labels to binary class matrix
    y_train = keras.utils.to_categorical(y_train, number_of_classes)
    y_test = keras.utils.to_categorical(y_test, number_of_classes)

    return x_train, y_train, x_test, y_test
 
  def __load_square_image(self, path, size):
    """
    Load image from path, resize to square (maintains aspect ratio)
    """
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = utils.convert_to_square(image, size) 
    return image
  
  def __load_from_root(self, path):
    """
    path: path to source directory where all classes are located
    return a list of all folder names and their files i.e ['dataset_test', ['/dataset_test/image1.png'...]]
    """
  
    folders = []
    classes = []

    # list all non hidden folders
    for root, dirs, files in os.walk(path):  
      for dir_ in dirs:
        if "." in dir_:
          continue
        folders.append(os.path.join(root, dir_))

    # load images    
    for folder in folders:
      name, files = self.__load_from_folder(folder, file_type = self.IMAGE_TYPE)
      if len(files) > 0:
        classes.append([name, files])

    return classes
  
  def __load_from_folder(self, path, file_type, verbose = False):
    """
    path: path to folder where the files are located
    return folder name, a list of file paths located in the folder
    """

    # find all files with defined extension
    extension = '*.{}'.format(file_type)
    files = glob(os.path.join(path, extension))

    # folder name becomes class name  
    name = os.path.splitext(os.path.basename(path))[0]

    if verbose:
      print("name: {}, files: {}".format(name, len(files)))

    return name, files
    