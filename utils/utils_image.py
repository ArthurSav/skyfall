import os
import cv2
import numpy as np
import matplotlib.pylab as plt
from skimage.transform import rescale, resize, downscale_local_mean, pyramid_reduce

from skyfall.models.model_utils import DisplaySize

"""
Image manipulation utils
"""

def convert_to_square(image, size, retain_aspect_ratio = False):
    """
    Resizes an image while maintaing aspect ratio

    image: image as numpy array
    size: resize size for square
    retain_aspect_ratio: if true, it will try to resize the image while maintaining it's aspect ratio.

    Note: retain_aspect_ratio is not playing well with image normalization and seems to alter the original image quite a bit
    """

    mask = None

    if retain_aspect_ratio:

      height, width = image.shape
      if(height > width):
        differ = height
      else:
        differ = width
      differ += 4

      # square filler
      background_filler = image[0][0]
      mask = np.zeros((differ, differ), dtype = "uint8")
      mask.fill(background_filler)

      x_pos = int((differ - width) / 2)
      y_pos = int((differ - height) / 2)

      # center image inside the square
      mask[y_pos: y_pos + height, x_pos: x_pos + width] = image[0: height, 0: width]

      # downscale if needed
      if differ / size > 1:
        mask = pyramid_reduce(mask, differ / size)

    else:
      mask = image

    return cv2.resize(mask, (size, size), interpolation = cv2.INTER_AREA)

def show_image_list(images = None, columns = 10, size = None, display = DisplaySize.SMALL):
    """
    Displays all images in a list
    
    images: a list of images
    columns: number of images per row
    size: limit number of images to show
    folder: load images from folder
    """
      
    if images is None or len(images) == 0:
      print("Nothing to show")
      return
    
    img_size = len(images)
    
    if size == None or size > img_size:
      size = img_size
      
    if columns > size:
      columns = size
      print("Setting column size to {}".format(columns))
      
    # calculate how many rows we need  
    rows = round((size/ (columns * 1.0)) + 0.5)
    
    x_ratio = size / rows
    y_ratio = size / columns
    
    # controllable ratio by user
    if display == DisplaySize.BIG:
      ratio = 12
    elif display == DisplaySize.MEDIUM:
      ratio = 8
    else:
      ratio = 4
        
    fig = plt.figure(figsize=(x_ratio + ratio, y_ratio + ratio))
    
    for i in range(0, size):
      image = images[i]
      plt.subplot(rows, columns, i + 1)
      plt.grid('off')
      plt.xticks([])
      plt.yticks([])
      plt.imshow(image)
    fig.tight_layout()
    plt.show()

def show_prediction_list(images, class_names, predictions, size = None, labels = None, columns = None, display = DisplaySize.SMALL):
  """
  Show eval predictions alongside images

  images: numpy array of images
  class_names: names corresponding to predictions ['toolbar', 'button' etc..]
  predictions: predictions for provided images
  labels: used when image predictions are known
  size: limit the number of shown images
  """
 
  img_size = images.shape[0]
  
  if size == None or size > img_size:
    size = img_size
    
  if columns is None and size <= 10:
    columns = 2
  elif columns is None:
    columns = 10

  if columns > size:
    size = columns
  
    
  # calculate how many rows we need  
  rows = round((size/ (columns * 1.0)) + 0.5)
  
  x_ratio = size / rows
  y_ratio = size / columns
  
  # controllable ratio by user
  if display == DisplaySize.BIG:
    ratio = 12
  elif display == DisplaySize.MEDIUM:
    ratio = 8
  else:
    ratio = 4
      
  fig = plt.figure(figsize=(x_ratio + ratio, y_ratio + ratio))
  
  for i in range(0, size):
    image = images[i]
    plt.subplot(rows, columns, i + 1)
    plt.grid('off')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image.squeeze())
    
    predicted_label = np.argmax(predictions[i])
    score = predictions[i][predicted_label]
    
    color = 'blue'
    if labels is not None:
      true_label = np.argmax(labels[i])
      if predicted_label == true_label:
        color = 'green'
      else:
        color = 'red'
      
    label_name = class_names[predicted_label]
    if labels is not None:
      title = "{} {:0.2f} ({})".format(label_name, score, class_names[true_label])
    else:
      title = "{} {:0.2f}".format(label_name, score)
    plt.xlabel(title, color=color)
    
  fig.tight_layout()
  plt.show()