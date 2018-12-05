import cv2
import numpy as np

from skyfall.engine.model_manager import DataLoader, ModelTrain
from skyfall.utils import utils_image


def test_training():
    loader = DataLoader()
    trainer = ModelTrain()

    names, (x_train, y_train), (x_test, y_test) = loader.load_training_data(
        "/Users/artursaveljev/github/skyfall/models/xmodel_1")

    utils_image.plot_image_list(x_train, convert_to_list=True)

    trainer.train(x_train, y_train)
    trainer.eval(x_test, y_test)

    trainer.save("model.h5")


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

def test_mnist():
    pass

# image_resize()
test_training()
