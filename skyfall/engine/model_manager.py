import os
import shutil

import Augmentor
import cv2
import numpy as np
import tensorflow as tf
from keras import utils
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential, load_model

from skyfall.engine import IMAGE_SIZE_EXPORT, AUGMENTED_FOLDER_OUPUT

from skyfall.utils import utils_image, utils_train


def list_images(path, extensions=['.png', '.jpg', '.jpeg']):
    """
    :param path: i.e /users/hello/project
    :return: a list of image paths located in provided path
    """
    images = [os.path.join(path, file) for file in os.listdir(path) if
              any(file.endswith(ext) for ext in extensions)]
    size = len(images)
    return images, size


def list_folders_with_images(path, list_augmented=False):
    """
    :param path: root path where folders are located
    :param list_augmented: if true, it will list augmented images(if any)
    :return: list of folder names and images located in each folder [{'folder_name', [image1.png, image2.png]}...]
    """

    folders = []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path) and not folder.startswith("."):
            images = []
            normal_images, normal_size = list_images(folder_path)
            images.extend(normal_images)

            if list_augmented:
                path_augmented = os.path.join(folder_path, AUGMENTED_FOLDER_OUPUT)
                if os.path.isdir(path_augmented):
                    aug_images, aug_size = list_images(path_augmented)
                    if aug_size > 0:
                        images.extend(aug_images)

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
    __invert_images = True
    __retain_aspect_ratio = False

    def __init__(self):
        pass

    def load_training_data(self, path, split_ratio=0.2, size=64, normalize=True, verbose=True,
                           max_images_per_class=None):
        """
        Loads folders from a source directory.
        Folder names become 'class' names.

        split_ratio: Accepts 0-1 values. Splits training and testing data
        size: Resizes the image in width=size and height=size (square) while maintaining aspect ratio

        return class names, training data with labels, testing data with labels
        """

        classes = list_folders_with_images(path, True)

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
        x_train = self.__load_paths_as_array(train_paths, size, invert=self.__invert_images,
                                             retain_aspect_ratio=self.__retain_aspect_ratio)
        x_test = self.__load_paths_as_array(test_paths, size, invert=self.__invert_images,
                                            retain_aspect_ratio=self.__retain_aspect_ratio)

        # shuffle images to even out distribution during training
        x_train, y_train = self.__shuffle(x_train, y_train)
        x_test, y_test = self.__shuffle(x_test, y_test)

        # reahape into [num_of_images, width, height, num_of_color_channels]
        x_train = x_train.reshape(x_train.shape[0], size, size, 1).astype('float32')
        x_test = x_test.reshape(x_test.shape[0], size, size, 1).astype('float32')

        if normalize:
            x_train, y_train, x_test, y_test = self.__normalize_batch(x_train, y_train, x_test, y_test,
                                                                      number_of_classes)

        print("Exported size: {}x{}".format(size, size))
        print("x_train: {}, x_test: {}".format(len(x_train), len(x_test)))

        return names, (x_train, y_train), (x_test, y_test)

    def load_evaluation_data(self, path=None, images=None, size=64, normalize=True):
        if path is not None:
            images, num = list_images(path)
            images = self.__load_paths_as_array(images, size, invert=self.__invert_images,
                                                retain_aspect_ratio=self.__retain_aspect_ratio)
        elif images is not None and isinstance(images, (list,)):
            images = self.__load_list_as_array(images, size, invert=self.__invert_images,
                                               retain_aspect_ratio=self.__retain_aspect_ratio)
        else:
            raise ValueError('You must provide either a path or images to load')

        print("Loaded {} images for evaluation...".format(len(images)))

        images = images.reshape(images.shape[0], size, size, 1).astype('float32')

        if normalize:
            images = self.__normalize(images)

        print('Exported size: {}x{}'.format(size, size))
        return images

    @staticmethod
    def __load_paths_as_array(paths, size, grayscale=True, invert=True, retain_aspect_ratio=False):
        """
        :param paths: list of image paths to load
        :param size: image size (i.e 64x64)
        :param grayscale: if true, images will be converted to grayscale
        :param invert: if true, it will invert the colors.
         We use it to convert black images into white images since they seem to perform better when training/eval
        :return: numpy array of squared+grayscaled images
        """
        converted = np.empty([0, size, size])
        for path in paths:
            image = cv2.imread(path)
            if image is None:
                print("Could not convert load image")
                continue
            if grayscale:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if invert:
                image = np.invert(image)
            image = utils_image.convert_to_square(image, size, retain_aspect_ratio)
            converted = np.concatenate((converted, image.reshape(1, size, size)))
        return converted

    @staticmethod
    def __load_list_as_array(images, size, grayscale=True, invert=True, retain_aspect_ratio=False):
        """
        :param images: images to convert
        :param size: image size (i.e 64x64)
        :param grayscale: if true, images will be converted to grayscale
        :param invert: if true, it will invert the colors.
         We use it to convert black images into white images since they seem to perform better when training/eval
        :return: numpy array of squared+grayscaled images
        """
        converted = np.empty([0, size, size])
        for image in images:
            if len(image.shape) == 3 and grayscale:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if invert:
                image = np.invert(image)
            image = utils_image.convert_to_square(image, size, retain_aspect_ratio)
            converted = np.concatenate((converted, image.reshape(1, size, size)))
        return converted

    @staticmethod
    def __normalize_batch(x_train, y_train, x_test, y_test, number_of_classes):
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

        # limit decimal point numbers
        x_train = np.round(x_train, 1)
        x_test = np.round(x_test, 1)

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

    @staticmethod
    def __normalize(images):
        if images.dtype != 'float32':
            images = images.astype('float32')
        images = (images - np.min(images)) / np.ptp(images)

        # limit decimal point numbers
        images = np.round(images, 1)

        return images


class ModelPredict:
    model = None
    names = None

    def load_model(self, model, names):
        """
        :param model: compiled & trained model
        :param names: label names
        """
        self.model = model
        self.names = names

        self.__check_missing_data()

    def predict(self, images, metadata):
        self.__check_missing_data()

        model = self.model
        names = self.names

        results = []
        predictions = model.predict(images)

        for idx, prediction in enumerate(predictions):

            # label with highest score
            predicted_label = np.argmax(prediction)

            # score of highest predicted label
            score = prediction[predicted_label]

            # predicted label name
            predicted_label_name = names[predicted_label]

            result = {'label': predicted_label_name,
                      'label_id': predicted_label,
                      'score': score}

            if metadata is not None:
                contours = metadata[idx]
                result['contours'] = contours

            results.append(result)

        return results

    def __check_missing_data(self):
        if self.model is None:
            raise Exception("Please provide a valid and compiled model")
        if self.names is None:
            raise Exception("Please provide valid label names")


class ModelTrain:
    model = None
    model_name = None
    model_extension = ".h5"

    def train(self, x_train, y_train, verbose=2, epochs=5, validation_split=0.1, batch_size=256):

        shape = x_train.shape[1:]
        outcomes = y_train.shape[1]

        model = self.__build_model(shape=shape, outcomes=outcomes)
        model.fit(x=x_train, y=y_train, validation_split=validation_split, batch_size=batch_size,
                  verbose=verbose, epochs=epochs)

        self.model = model

        return model

    def eval(self, x_test, y_test, model=None):
        if model is None and self.model is not None:
            model = self.model
        elif model is None:
            print("Model not found")
            return

        predictions = model.predict(x_test)
        accuracy = utils_train.calculate_accuracy(predictions, y_test)

        return accuracy

    def save(self, name, path=None, model=None):
        """
        :param name: model name
        :param path: path where to save (default is current dir)
        :param model: actual model
        """

        if model is None and self.model is not None:
            model = self.model
        elif model is None:
            raise Exception("Missing model, nothing to save")

        if name is None:
            raise Exception("Please provide a valid model name to save")

        # add extension
        if not name.endswith(self.model_extension):
            name += self.model_extension

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
    model_extension = ".h5"

    model = None
    model_path = None

    graph = tf.get_default_graph()

    loader = DataLoader()
    trainer = ModelTrain()
    predictor = ModelPredict()

    def __init__(self, path):
        """
        Creates a new model directory
        :param path: root directory where all models are located
        """

        if path is None:
            raise Exception("Please provide a valid root path")

        self.path = os.path.abspath(path)

    def train(self, name):
        """
        Trains a new model on the components saved
        :param name: model name
        """

        self.__rename_current_model_dir(name)
        path = self.get_model_dir_path()

        # load & preprocess images
        names, (x_train, y_train), (x_test, y_test) = self.loader.load_training_data(path)

        z = []
        for c in x_train:
            z.append(c.squeeze())
        utils_image.plot_image_list(images=z)

        self.trainer.train(x_train, y_train)
        self.trainer.eval(x_test, y_test)
        self.trainer.save(name=name, path=path)

    def predict(self, images, metadata):
        results = []
        if images:
            self.__check_model_exists()

            # preprocess images
            images = self.loader.load_evaluation_data(images=images)

            model = self.model

            # todo - consider saving component names into a var instead of fethcing them everying time

            # use folder names (components) as label names
            components = self.list_model_components()
            names = []
            for component in components:
                names.append(component['name'])

            # load model info
            self.predictor.load_model(model, names)

            # predict labels
            results = self.predictor.predict(images, metadata)

        print("Predicted results... {}".format(len(results)))
        for result in results:
            print("Label: {}, score: {}".format(result['label'], result['score']))

        return results

    def load_model(self, name):
        """
        Loads model from model dir
        :param name: model name (folder name and model filename must be the same)
        """
        self.model_name = name
        self.model_path = self.get_model_dir_path()

        # create mode filepath
        filepath = os.path.join(self.model_path, "{}{}".format(name, self.model_extension))

        if not os.path.isfile(filepath):
            raise Exception("Model file '{}' not found".format(filepath))

        self.model = load_model(filepath)
        self.__check_model_exists()

        print("Loaded model {}".format(filepath))

    def create_model_dir(self, name=None):
        if not name:
            folders, prefixed = self.list_model_folders()
            name = "{}{}".format(self.model_prefix, len(prefixed))

        self.model_name = name
        create_folder_if_needed(self.path, name)

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

        path = create_folder_if_needed(self.get_model_dir_path(), name)
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

            # resize
            image = cv2.resize(image, (IMAGE_SIZE_EXPORT, IMAGE_SIZE_EXPORT), interpolation=cv2.INTER_AREA)

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

        path = self.get_model_dir_path()
        component_path = os.path.join(path, name)
        if os.path.exists(component_path):
            shutil.rmtree(component_path)
            return True
        return False

    def get_model_dir_path(self):
        if self.model_name is None:
            raise Exception("Model name is missing")
        return os.path.join(self.path, self.model_name)

    def list_model_components(self):
        return list_folders_with_images(self.get_model_dir_path())

    def list_model_folders(self):
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

    def get_model(self):
        return self.model

    def __check_model_exists(self):
        if not self.model_name:
            raise Exception("Model name is missing")
        if self.model is None:
            raise Exception("Model file is missing")
        if not self.model_path:
            raise Exception("Model directory path is missing")

    def __rename_current_model_dir(self, name):
        current_model_dir = self.get_model_dir_path()
        renamed_model_dir = os.path.join(self.path, name)
        os.rename(current_model_dir, renamed_model_dir)

        self.model_name = name
        self.model_path = renamed_model_dir
