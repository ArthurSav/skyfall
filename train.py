from tensorflow.keras import layers
from tensorflow import keras 
import tensorflow as tf
import matplotlib.pylab as plt

from enum import Enum

from skyfall import utils
from utils import TrainUtils as util_train
from utils import ImageUtils as util_image

class Train:

    __MODEL = None

    __MODEL_SAVE_NAME = 'weights.h5'

    def __init__(self):
        pass

    def __build_model(self, shape, outcomes, verbose = False):

        """
        Builds base model for greyscale (single channel) images

        shape: image shape (h, w, channel) i.e (28, 28, 1)
        outputs: possible numeric outcomes/labels i.e [0, 1, 2, 3, 5, 6]

        return new model
        """

        # define model
        model = keras.Sequential()
        model.add(layers.Convolution2D(16, (3, 3),
                                padding='same',
                                input_shape=shape, activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Convolution2D(32, (3, 3), padding='same', activation= 'relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Convolution2D(64, (3, 3), padding='same', activation= 'relu'))
        model.add(layers.MaxPooling2D(pool_size =(2,2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(outcomes, activation='softmax')) 

        # compile
        adam = tf.train.AdamOptimizer()
        model.compile(loss='categorical_crossentropy',
                        optimizer=adam,
                        metrics=['top_k_categorical_accuracy'])

        if verbose:
            print(model.summary())

        return model
    
    def train(self, x_train, y_train, x_test, y_test):

        shape = x_train.shape[1:]
        outcomes = y_train.shape[1]

        model = self.__build_model(shape = shape, outcomes = outcomes)
        model.fit(x = x_train, y = y_train, validation_split=0.1, batch_size=256, verbose=2, epochs=5)
        
        self.__MODEL = model

        return model

    def evaluate(self, x_test, y_test, model = __MODEL):

        """
        Evaluates model accuracy
        """

        if model is None:
            print("Nothing to evaluate")
            return

        predictions = model.predict(x_test)
        util_train.calculate_accuracy(predictions, y_test)

    def save(self, model = __MODEL, name = __MODEL_SAVE_NAME):
        """
        model: trained keras model
        name: save name
        """

        if model is None:
            print("Nothing to save")
            return

        model.save_weights(name)
