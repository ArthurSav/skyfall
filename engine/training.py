import os
import shutil

import cv2
import numpy as np

from keras import utils
from utils import utils_image, utils_train
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential


def list_images(path, extensions=['.png', '.jpg', '.jpeg']):
    """
    :param path: i.e /users/hello/project
    :return: a list of image paths located in provided path
    """
    images = [os.path.join(path, file) for file in os.listdir(path) if
              any(file.endswith(ext) for ext in extensions)]
    size = len(images)
    return images, size


def list_folders_with_images(path):
    """
    :param path: root path where folders are located
    :return: list of folder names and images located in each folder [{'folder_name', [image1.png, image2.png]}...]
    """

    folders = []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path) and not folder.startswith("."):
            images, size = list_images(folder_path)
            folders.append({"name": folder, "images": images})
    return folders


def create_folder_if_needed(path, name, verbose=False):
    folder = os.path.join(path, name)
    if not os.path.exists(folder):
        os.makedirs(folder)
        if verbose:
            print("Created dir: {}".format(folder))

    return folder


class DataLoader:

    def __init__(self):
        pass

    def load_training_data(self, path, split_ratio=0.2, size=64, normalize=True, invert=True, verbose=True,
                           max_images_per_class=None):
        """
        Loads folders from a source directory.
        Folder names become 'class' names.

        split_ratio: Accepts 0-1 values. Splits training and testing data
        size: Resizes the image in width=size and height=size (square) while maintaining aspect ratio

        return class names, training data with labels, testing data with labels
        """

        classes = list_folders_with_images(path)

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
            class_name = class_data['name']
            class_images = class_data['images']
            names.append(class_name)

            # limit images per class
            if max_images_per_class:
                class_images = class_images[:min(max_images_per_class, len(class_images))]

            print("name: {}, images: {}".format(class_name, len(class_images)))

            # split into train, test
            train, train_labels, test, test_labels = self.__split(class_images, idx, split=split_ratio)

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

        print("Exported size: {}x{}".format(size, size))
        print("x_train: {}, x_test: {}".format(len(x_train), len(x_test)))

        return names, (x_train, y_train), (x_test, y_test)

    def __load_paths_as_array(self, paths, size):
        """
        Convert a list of image paths into a numpy array of resized images

        paths: a list of image paths i.e /train/images/hello.jpg ...
        size: image size to resize to

        return a numpy array of loaded square images of shape i.e (num, size, size, 1)
        """

        x = np.empty([0, size, size])

        for path in paths:
            image = self.__load_square_image(path, size)
            image = np.invert(image)
            x = np.concatenate((x, image.reshape(1, size, size)))

        return x

    @staticmethod
    def __normalize(x_train, y_train, x_test, y_test, number_of_classes):
        """
        Normalize images to a range of [0, 1] and image labels to binary matrixes
        """

        if x_train.dtype != 'float32':
            x_train = x_train.astype('float32')

        if x_test.dtype != 'float32':
            x_test = x_test.astype('float32')

        # normalize images [0, 1]
        x_train = (x_train - np.min(x_train)) / np.ptp(x_train)
        x_test = (x_test - np.min(x_test)) / np.ptp(x_test)

        # normalize labels to binary class matrix
        y_train = utils.to_categorical(y_train, number_of_classes)
        y_test = utils.to_categorical(y_test, number_of_classes)

        return x_train, y_train, x_test, y_test

    @staticmethod
    def __shuffle(x, y):
        """
        Random shuffle images and their labels
        """

        size = x.shape[0]
        permutation = np.random.permutation(size)

        x = x[permutation, :, :]
        y = y[permutation]

        return x, y

    @staticmethod
    def __load_square_image(path, size):
        """
        Load image from path, resize to square (maintains aspect ratio)
        """
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = utils_image.convert_to_square(image, size)
        return image

    @staticmethod
    def __split(images, class_id, split=0.2):
        """
        Split images of a single class into training and test sets based on provided ratio
        """

        image_size = len(images)
        labels = np.full(image_size, class_id)

        vfold_size = int((image_size / 100.0) * (split * 100.0))

        train = images[vfold_size:image_size]
        train_labels = labels[vfold_size: image_size]
        test = images[0: vfold_size]
        test_labels = labels[0: vfold_size]

        return train, train_labels, test, test_labels


class ModelTrain:

    model = None
    model_name = "model.h5"

    def train(self, x_train, y_train, verbose=2, epochs=5, validation_split=0.1, batch_size=256):

        shape = x_train.shape[1:]
        outcomes = y_train.shape[1]

        model = self.__build_model(shape=shape, outcomes=outcomes)
        model.fit(x=x_train, y=y_train, validation_split=validation_split, batch_size=batch_size,
                  verbose=verbose, epochs=epochs)

        self.model = model

        return model

    def evaluate(self, x_test, y_test, model=None):

        if model is None and self.model is not None:
            model = self.model
        elif model is None:
            print("Model not found")
            return

        predictions = model.predict(x_test)
        accuracy = utils_train.calculate_accuracy(predictions, y_test)

        return accuracy

    def save(self, name=None, path=None, model=None):

        if model is None and self.model is not None:
            model = self.model
        elif model is None:
            print("Missing model, nothing to save")
            return

        if name is None and self.model_name is not None:
            name = self.model_name
        elif self.model_name is None:
            print("Please provide a valid model name to save")
            return

        # add extension
        if not name.endswith(".h5"):
            name += ".h5"

        if not path:
            path = name
        else:
            path = os.path.join(path, name)

        print("Saved model: {}".format(path))
        model.save(path)

    def get_model(self):
        return self.model

    @staticmethod
    def __build_model(shape, outcomes, verbose=False):

        """
        Builds base model for greyscale (single channel) images

        shape: image shape (h, w, channel) i.e (28, 28, 1)
        outcomes: possible numeric outcomes/labels i.e [0, 1, 2, 3, 5, 6] (normalized)

        :return: compiled model
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


class ModelCreator:
    path = None
    model_name = None
    model_prefix = "model_"
    exts = ['.png', '.jpg', '.jpeg']
    save_extension = ".png"

    loader = DataLoader()
    trainer = ModelTrain()

    def __init__(self, path, model_name=None):
        self.path = os.path.abspath(path)
        self.__set_model_folder_name_or_default(model_name)

    def train(self, name=None):

        path = self.get_model_path()
        names, (x_train, y_train), (x_test, y_test) = self.loader.load_training_data(path)

        self.trainer.train(x_train, y_train)
        self.trainer.evaluate(x_test, y_test)
        self.trainer.save(name=name, path=path)

    def save_component(self, name, images, replace=True, verbose=False):

        """
        Save component images in a folder named after the component.

        :param name: component/folder name
        :param images: component images
        :param replace: if true, it will replace images from an existing folder
        :param verbose: prints debug details
        :return:
        """

        if not name:
            print("Provided component name is not valid")
            return
        if images is None:
            print("No images provided")
            return

        removed = 0
        added = 0

        path = create_folder_if_needed(self.get_model_path(), name)
        current_images, current_images_size = list_images(path)

        # remove previous
        if replace and current_images_size > 0:
            for image in current_images:
                os.remove(image)
                removed += 1
            current_images, current_images_size = list_images(path)

        for idx, image in enumerate(images):
            image_name = "{}_{}{}".format(name, idx + current_images_size, self.save_extension)
            image_path = os.path.join(path, image_name)

            cv2.imwrite(image_path, image)
            added += 1

        current_images, current_images_size = list_images(path)

        if verbose:
            print("deleted: {}, added: {}, total: {} ({})".format(removed, added, current_images_size, path))

    def remove_component(self, name):
        """
        Deletes component folder and it's content
        :param name: component name
        """

        path = self.get_model_path()
        component_path = os.path.join(path, name)
        if os.path.exists(component_path):
            shutil.rmtree(component_path)
            return True
        return False

    def get_model_path(self):
        return os.path.join(self.path, self.model_name)

    def list_model_components(self):
        return list_folders_with_images(self.get_model_path())

    def __set_model_folder_name_or_default(self, name=None):
        """
        Creates component folder. If no name is provided and one is generated
        :param name: component folder name
        """
        if not name:
            folders, prefixed = self.__list_model_folders()
            self.model_name = "{}{}".format(self.model_prefix, len(prefixed))
        else:
            self.model_name = name

        create_folder_if_needed(self.path, self.model_name)

    def __list_model_folders(self):
        """
        :return: a list of folders located in project root
        """
        folders = []
        prefixed_folders = []
        root = self.path
        for item in os.listdir(root):
            if os.path.isdir(os.path.join(root, item)) and not item.startswith("."):
                if item.startswith(self.model_prefix):
                    prefixed_folders.append(item)
                folders.append(os.path.join(root, item))

        return folders, prefixed_folders


