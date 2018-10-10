import cv2
import matplotlib.pylab as plt
import math
import os

from enum import Enum

from skyfall.models.model_utils import CropType
from skyfall.models.model_p1 import MetadataContour

class ContourFinder:

  EXTENSION = '.png'
  class_name = None
  image = None
  cropped = None
  metadata = None
  image_with_contours = None
  
  def __init__(self, image_path, class_name = None):
    """
    image_path: i.e hello/image.png
    class_name: i.e hello (used to name the parent folder when saving images)
    """

    if image_path is None:
      print("Please provide a valid image path")
      return

    if class_name is not None:
      self.class_name = class_name
    else:
      # image name becomes class name
      self.class_name = image_path.replace(self.EXTENSION, '')

    self.image = cv2.imread(image_path)
    if self.image is None:
      print("No image found")
    
  def __find_mobile_screen_contours(self, min_contour_dimen = 50, max_contour_percentage = 75, min_layer =1, max_layer = 3, verbose = False):
    """
    contours: cv2 contours of an image
    hierarchy: contour hierarchy [Next, Previous, First_Child, Parent]
    min_contour_dimen: contours that do not meet the minumum set dimen will be skipped
    max_contour_dimen_percentage: Max allowed element dimen compared to parent screen. 
    Used to filter out noise when processing screens i.e an element that covers 85% of the entire screen is probably the screen itself
    min_layer: Min layer to get contours
    max_layer: Max layer to get contours (anything deeper is usually noise)
    
    Note: Expects ONLY one screen as parent. Will not work if more that one parent is present
    for contours and hierarchy see here: https://docs.opencv.org/3.4.0/d9/d8b/tutorial_py_contours_hierarchy.html
    """
    
    contours, hierarchy = self.__find(cv2.RETR_TREE)
    hierarchy = hierarchy[0]
    
    # sort contours & hierarchy based on layer and element hierarchy
    # contours = [x for _,x in sorted(zip(hierarchy, contours), key=lambda x: (x[0][3], x[0][1]))]
    # hierarchy = sorted(hierarchy, key=lambda x: (x[3], x[1]))
        
    filtered_contours = []
    filtered_hierarchy = []
    
    # holds x,y coordinates of a bounding rectangle. Used for sorting purposes
    filtered_xy = []
    
    w_parent = 0
    h_parent = 0
    
    for idx, item in enumerate(hierarchy):
      
      parent_num = item[3]
      contour = contours[idx]
      x, y, w, h = cv2.boundingRect(contour)
            
      # screen contour
      if parent_num == -1:
        w_parent = w
        h_parent = h
      
      # filter out elements depending on layer
      if parent_num >= min_layer and parent_num <= max_layer:
        
        # skip contours that are too small
        if w < min_contour_dimen and h < min_contour_dimen:
          continue
        
        # skip contours that are too big (usually it's the screen itself)
        if self.__calculate_percentage_area(w_parent, h_parent, w, h) > max_contour_percentage:
          continue
        
        filtered_xy.append([x, y])
        filtered_contours.append(contour)
        filtered_hierarchy.append(item)
    
    filtered_contours = [x for _,x in sorted(zip(filtered_xy, filtered_contours), key=lambda x: x[0][1])]
    filtered_hierarchy = [x for _,x in sorted(zip(filtered_xy, filtered_hierarchy), key=lambda x: x[0][1])]
    filtered_xy = sorted(filtered_xy, key = lambda x: x[1])
    
    if verbose:
       for idx, item in enumerate(filtered_xy):
          h = filtered_hierarchy[idx]
          print("Added element from parent layer: {}".format(h[3]))
          print("x: {}, y: {}".format(item[0], item[1]))
        
    return filtered_contours, filtered_hierarchy

  def __calculate_percentage_area(self,w1, h1, w2, h2):
    """
    Compares 2 surface areas
    return the percentage of area2 compared to area1 (i.e area 2 is 80% of area 1)
    """
    
    if w1 != 0 and h1 != 0 and w2 != 0 and h2 != 0:
      a1 = 2 * (w1 + h1)
      a2 = 2 * (w2 + h2)
      return a2 * 100 / a1
      
    return 0
  
  def __find_training_elements_contours(self, min_contour_dimen = 50):
    """
    Element separation happens at a parent level only
    
    min_contour_dimen: contours that do not meet the minumum set dimen will be skipped
    """
    contours, hierarchy = self.__find(cv2.RETR_EXTERNAL)
    
    filtered_contours = []
    filtered_hierarchy = []
    
    for idx, item in enumerate(hierarchy[0]):
      
      contour = contours[idx]
      x, y, w, h = cv2.boundingRect(contour)
      
      # skip contours that are too small
      if w < min_contour_dimen and h < min_contour_dimen:
        continue
        
      filtered_contours.append(contour)
      filtered_hierarchy.append(item)
      
    return filtered_contours, filtered_hierarchy
    
  def __find(self, hierarchy_type = cv2.RETR_TREE):
    """
    hierarchy_type: RETR_LIST, RETR_EXTERNAL, RETR_CCOMP, RETR_TREE
    
    see more: https://docs.opencv.org/3.4.0/d9/d8b/tutorial_py_contours_hierarchy.html
    """
    
    image = self.image
    
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 10, 250)
    
    # apply closing function 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    
    # find contours 
    img_contour, contours, hierarchy = cv2.findContours(closed.copy(), hierarchy_type, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, hierarchy

  def crop(self, mode, crop_padding = 10, verbose = False):
    """
    mode: MOBILE, TRAINING
    min_contour_dimen: ignores anything below specified dimention when applying contours
    crop_padding: padding to apply when cropping contours
    crop: if true, it will crop contours

    return: image with applied contours, cropped contours
    """
    
    image = self.image
    image_with_contours = image.copy()
    
    contours, hierarchy = [], []
    
    if mode == CropType.MOBILE:
      contours, hierarchy = self.__find_mobile_screen_contours(verbose = True, min_layer = -1, max_layer = 3, max_contour_percentage = 60)
    elif mode == CropType.TRAINING:
      contours, hierarchy = self.__find_training_elements_contours()
    else:
      print("Please provide croping type: {}, {}".format(CropType.MOBILE.name, CropType.TRAINING.name))
      return
      
    if verbose:
        print("Cropping... mode: {} contours found: {}".format(mode.name, len(contours)))  
 
    cropped, metadata = [], []
    
    for c in contours:
      
      # rectangle coordinates
      x, y, w, h = cv2.boundingRect(c)
    
      # cropping metadata
      metadata_contour = MetadataContour(x, y, w, h)
      metadata.append(metadata_contour)

      # crop contour & convert to grayscale
      cropped_image = image[y - crop_padding: y + h + (crop_padding * 2) ,x - crop_padding: x + w + (crop_padding * 2)]
      cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
      cropped.append(cropped_image)
        
      # draw contour
      peri = cv2.arcLength(c, True)
      approx = cv2.approxPolyDP(c, 0.02 * peri, True)
      cv2.drawContours(image_with_contours, [approx], -1, (0, 255, 0), 2)

    print("Cropped images: {}".format(len(cropped)))

    self.image_with_contours = image_with_contours
    self.cropped = cropped
    self.metadata = metadata
    
  def save_images(self, images = None, class_name = None, override_existing = False, verbose = False, destination = 'data'):

    """
    Saves a list of images that belong to a class into the specified directory

    images: numpy array with images to save
    class_name: class name for images
    override_existing: if true, it will delete any previous images that belong to the specidied class
    destination: parent dir for class images
    """

    # use cropped
    if images is None and self.cropped is not None:
      images = self.cropped  
    elif images is None:
      print("Nothing to save")
      return

    if class_name is None and self.class_name is not None:
      class_name = self.class_name
    elif class_name is None:
      print("Class name is missing, cannot save")
      return
  
    path = "{}/{}".format(destination, class_name) # data/toolbar/
    
    extension = self.EXTENSION
    removed_num = 0
    added_num = 0
      
    # create dirs, if needed
    if not os.path.exists(destination):
      os.makedirs(destination)
    if not os.path.exists(path):
      os.makedirs(path)
      if verbose:
        print("Created {}".format(path))
      
    # get existing images in class folder, if any
    current_images = [file for file in os.listdir(path) if file.endswith(extension)]
    current_images_num = len(current_images)
    
    # removes previous images
    if override_existing and current_images_num > 0:
      for img in current_images:
        os.remove(path + "/" + img)
        removed_num += 1
        
    # get existing images in class folder, if any
    current_images = [file for file in os.listdir(path) if file.endswith(extension)]
    current_images_num = len(current_images)
    
    # save
    for idx,image in enumerate(images):
      dir_filename = "{}/{}_{}{}".format(path, class_name, idx + current_images_num, extension) # data/toolbar/toolbar_01.png
      cv2.imwrite(dir_filename, image)
      added_num += 1
      
    # get existing images in class folder, if any
    current_images = [file for file in os.listdir(path) if file.endswith(extension)]
    current_images_num = len(current_images)
      
    if verbose:
      print("deleted: {}, added: {}, total: {} ({})".format(removed_num, added_num, current_images_num, path))