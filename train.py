from tensorflow.keras import layers
from tensorflow import keras 
import tensorflow as tf
import matplotlib.pylab as plt

from enum import Enum

from skyfall import utils_image
from skyfall import utils_train

class Train:

    model = None
    model_save_name = 'weights.h5'

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
        
        self.model = model

        return model

    def evaluate(self, x_test, y_test, model = None):
        """
        Evaluates model accuracy
        """

        if model is None and self.model is not None:
            model = self.model
        elif model is None:
            print("Model not found")
            return

        predictions = model.predict(x_test)
        utils_train.calculate_accuracy(predictions, y_test)

    def save(self, model = None, name = None, path = None):
        """
        model: trained keras model
        name: save name
        path: i.e folder/folder01/ Leave blank to save in current dir
        """

        if model is None and self.model is not None:
            model = self.model
        elif model is None:
            model = self.model
            print("Nothing to save")
            return

        if name is None and self.model_save_name is not None:
            name = self.model_save_name
        elif self.model_save_name is None:
            print("Missing model save name")
            return

        if path:
            name = path + name

        print("Saving model: {}".format(name))
        model.save_weights(name)