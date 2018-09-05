import os
import cv2
import matplotlib.pylab as plt

MAIN_FOLDER = 'data'

def save_images(images, class_name, override_existing = False, verbose = False):
  """
  images: images to save
  class_name: training set class name
  override_existing: if true, it will delete any existing image in the class dir
  """
  
  dir_data = MAIN_FOLDER
  dir_data_class = "{}/{}".format(dir_data, class_name)
  extension = '.png'
  
  removed_num = 0
  added_num = 0
    
  # create dirs, if needed
  if not os.path.exists(dir_data):
    os.makedirs(dir_data)
  if not os.path.exists(dir_data_class):
    os.makedirs(dir_data_class)
    if verbose:
      print("Created {}".format(dir_data_class))
    
  # get existing images in class folder, if any
  current_images = [file for file in os.listdir(dir_data_class) if file.endswith(extension)]
  current_images_num = len(current_images)
  
  # removes previous images
  if override_existing and current_images_num > 0:
    for img in current_images:
      os.remove(dir_data_class + "/" + img)
      removed_num += 1
      
  # get existing images in class folder, if any
  current_images = [file for file in os.listdir(dir_data_class) if file.endswith(extension)]
  current_images_num = len(current_images)
  
  # save
  for idx,image in enumerate(images):
    dir_filename = "{}/{}_{}.png".format(dir_data_class, class_name, idx + current_images_num)
    cv2.imwrite(dir_filename, image)
    added_num += 1
    
  # get existing images in class folder, if any
  current_images = [file for file in os.listdir(dir_data_class) if file.endswith(extension)]
  current_images_num = len(current_images)
    
  if verbose:
    print("deleted: {}, added: {}, total: {} ({})".format(removed_num, added_num, current_images_num, dir_data_class))


def show_images(images, label = None, labels = None, size = 10):
  plt.figure(figsize=(size, size))
  for i in range(len(images)):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.tight_layout()
    
    l = ""
    if label is not None:
      l = label
    elif labels is not None:
      l = labels[i]
      
    plt.imshow(images[i], cmap='gray')
    plt.xlabel(l)