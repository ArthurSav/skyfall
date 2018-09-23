from skyfall import utils
import cv2
from glob import glob
import pandas as pd
import numpy as np
import os
from skimage.transform import rescale, resize, downscale_local_mean, pyramid_reduce
from tensorflow import keras

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
    
    names = []
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
      
    # todo - make sure provided size in no bigger than image size  
    x_train = np.empty([0, size, size])
    x_test = np.empty([0, size, size])
      
    # get real train images
    for path in train_paths:
      image = self.__load_square_image(path, size)
      image = np.invert(image) # white background, black content
      x_train = np.concatenate((x_train, image.reshape(1, size, size)))
      
    # get real test images  
    for path in test_paths:
      image = self.__load_square_image(path, size)
      image = np.invert(image) # white background, black content
      x_test = np.concatenate((x_test, image.reshape(1, size, size)))
    
    # shuffle images to even out distribution during training
    permutation_train = np.random.permutation(len(train_paths))
    permutation_test = np.random.permutation(len(test_paths))

    x_train = x_train[permutation_train, :, :]
    y_train = y_train[permutation_train]
    x_test = x_test[permutation_test, :, :]
    y_test = y_test[permutation_test]
    
    # reahape into [num_of_images, width, height, num_of_color_channels]
    x_train = x_train.reshape(x_train.shape[0], size, size, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], size, size, 1).astype('float32')
    
    if normalize:
      
      # normalize images
      x_train /= 255.0
      x_test /= 255.0

      # normalize labels
      number_of_classes = len(names)
      y_train = keras.utils.to_categorical(y_train, number_of_classes)
      y_test = keras.utils.to_categorical(y_test, number_of_classes)
 
    print("Exporting size: {}x{}".format(size, size))
    print("x_train: {}, x_test: {}".format(len(x_train), len(x_test))) 
    
    return names, (x_train, y_train), (x_test, y_test)  
 
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
    