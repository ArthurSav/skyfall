import glob
import os
import shutil

import Augmentor
import cv2
import numpy as np

from skyfall.engine import contours
from skyfall.engine.contours import ContourFinder, ContourType
from skyfall.engine.model_manager import DataLoader, ModelTrain
from skyfall.utils import utils_image

IMAGES = [
    # 'mobile_screen_1.png',
    # 'mobile_screen_2.png',
    # 'mobile_screen_3.png',
    # 'mobile_screen_4.png',
    # 'mobile_screen_5.png',
    'contours_test1.png',
    # 'mobile_photo.jpg'
]


def filter_contours(contours, hierarchy):
    filtered_c = []

    # contains all parent elements
    # hierarchy = hierarchy[0]
    #
    # for idx, item in enumerate(hierarchy):
    #
    #     # find all parents
    #     if item[3] == -1:
    #         filtered_c.append(contours[idx])

    for c in contours:
        if cv2.contourArea(c) > 500:
            filtered_c.append(c)

    filtered_c = sorted(filtered_c, key=cv2.contourArea, reverse=True)[:1]  # get largest five contour area

    return filtered_c


def draw_contours(original, filtered):
    img_contour, contours, hierarchy = cv2.findContours(filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # get largest five contour area

    contours = filter_contours(contours, hierarchy)

    for c in contours:
        # rectangle coordinates

        # draw contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)

        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 10)


def test_contours():
    finder = ContourFinder()
    finder.load_image_path("/Users/artursaveljev/github/skyfall/datasets/mobile_screen_4.png")

    image_with_contours, cropped, metadata = finder.draw_and_crop_contours(ContourType.MOBILE, crop=True, verbose=True,
                                                                           crop_padding=10)

    utils_image.plot_image(image_with_contours)
    utils_image.plot_image_list(cropped)


def test_training():
    loader = DataLoader()
    trainer = ModelTrain()

    names, (x_train, y_train), (x_test, y_test) = loader.load_training_data(
        "/Users/artursaveljev/github/skyfall/models/xmodel_1")

    utils_image.plot_image_list(x_train, convert_to_list=True)

    trainer.train(x_train, y_train)
    trainer.eval(x_test, y_test)

    trainer.save("/Users/artursaveljev/github/skyfall/models/xmodel_1/xmodel_1.h5")


def image_resize():
    image = cv2.imread("/Users/artursaveljev/github/skyfall/tests/image_1.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    squared = utils_image.convert_to_square(image, 128, True)
    squared = np.round(squared, 1)

    utils_image.plot_image(squared)

    try:
        np.invert(squared)
    except:
        print("Could not convert")

    image = (image - np.min(image)) / np.ptp(image)
    squared = (squared - np.min(squared)) / np.ptp(squared)

    utils_image.plot_image(squared)


def resize_all():
    root_directory = "/Users/artursaveljev/github/skyfall/models/xmodel_1/*"
    folders = []
    for f in glob.glob(root_directory):
        if os.path.isdir(f):
            folders.append(os.path.abspath(f))

    for folder in folders:
        print(folder)
        for file in os.listdir(folder):
            if file.endswith(".png"):
                ipath = os.path.join(folder, file)
                image = cv2.imread(ipath)
                image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
                cv2.imwrite(ipath, image)


def augment_data():
    root_directory = "/Users/artursaveljev/github/skyfall/models/xmodel_1/*"
    folders = []

    for f in glob.glob(root_directory):
        if os.path.isdir(f):
            folders.append(os.path.abspath(f))

    for folder in folders:
        print(folder)

        # remove previous outputs
        path_output = os.path.join(folder, "output")
        if os.path.isdir(path_output):
            shutil.rmtree(path_output)

        p = Augmentor.Pipeline(folder)
        p.random_distortion(probability=1, grid_width=6, grid_height=6, magnitude=5)
        p.sample(100)


# test_contours()
path = "/Users/artursaveljev/github/skyfall/datasets/"
image = cv2.imread(path)
for img in IMAGES:
    p = path + img
    i = cv2.imread(p)
    filtered = contours.apply_filters(i)
    draw_contours(i, filtered)
    utils_image.plot_image(i)
