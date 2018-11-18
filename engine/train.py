from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from utils import utils_train


class Train:
    model = None
    model_save_name = 'model.h5'

    def __init__(self):
        pass

    def __build_model(self, shape, outcomes, verbose=False):

        """
        Builds base model for greyscale (single channel) images

        shape: image shape (h, w, channel) i.e (28, 28, 1)
        outcomes: possible numeric outcomes/labels i.e [0, 1, 2, 3, 5, 6] (normalized)

        :return: new model
        """

        # define model
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(3, 3), padding='same', input_shape=shape, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(outcomes, activation='softmax'))

        # compile
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['top_k_categorical_accuracy'])

        if verbose:
            print(model.summary())

        return model

    def train(self, x_train, y_train, x_test, y_test, verbose=2, epochs=5):

        shape = x_train.shape[1:]
        outcomes = y_train.shape[1]

        model = self.__build_model(shape=shape, outcomes=outcomes)
        model.fit(x=x_train, y=y_train, validation_split=0.1, batch_size=256, verbose=verbose, epochs=epochs)

        self.model = model

        return model

    def evaluate(self, x_test, y_test, model=None):
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

    def save(self, model=None, name=None, path=None):
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
        model.save(name)
