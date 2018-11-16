import os
import shutil

import cv2


class TrainingModelCreator:
    path = None
    model_name = None
    model_prefix = "model_"
    exts = ['.png', '.jpg', '.jpeg']
    save_extension = ".png"

    def __init__(self, path, model_name=None):
        self.path = os.path.abspath(path)
        self.set_model_name_or_default(model_name)

    def save_component(self, name, images, replace=True, verbose=False):
        if not name:
            print("Provided component name is not valid")
            return
        if images is None:
            print("No images provided")
            return

        removed = 0
        added = 0

        path = self.__create_folder_if_needed(self.get_model_path(), name)
        current_images, current_images_size = self.__get_path_images(path)

        # remove previous
        if replace and current_images_size > 0:
            for image in current_images:
                os.remove(image)
                removed += 1
            current_images, current_images_size = self.__get_path_images(path)

        for idx, image in enumerate(images):
            image_name = "{}_{}{}".format(name, idx + current_images_size, self.save_extension)
            image_path = os.path.join(path, image_name)

            cv2.imwrite(image_path, image)
            added += 1

        current_images, current_images_size = self.__get_path_images(path)

        if verbose:
            print("deleted: {}, added: {}, total: {} ({})".format(removed, added, current_images_size, path))

    def remove_component(self, name):
        path = self.get_model_path()
        component_path = os.path.join(path, name)
        if os.path.exists(component_path):
            shutil.rmtree(component_path)
            return True
        return False

    def set_model_name_or_default(self, name=None):
        if not name:
            folders, prefixed = self.__list_model_folders()
            self.model_name = "{}{}".format(self.model_prefix, len(prefixed))
        else:
            self.model_name = name

        self.__create_folder_if_needed(self.path, self.model_name)

    @staticmethod
    def __create_folder_if_needed(path, name, verbose=False):
        folder = os.path.join(path, name)
        if not os.path.exists(folder):
            os.makedirs(folder)
            if verbose:
                print("Create dir: {}".format(folder))

        return folder

    def __list_model_folders(self):
        folders = []
        prefixed_folders = []
        root = self.path
        for item in os.listdir(root):
            if os.path.isdir(os.path.join(root, item)) and "." not in item:
                if item.startswith(self.model_prefix):
                    prefixed_folders.append(item)
                folders.append(os.path.join(root, item))

        return folders, prefixed_folders

    def get_model_path(self):
        return os.path.join(self.path, self.model_name)

    def list_model_components(self):
        components = []
        path = self.get_model_path()
        for component in os.listdir(path):
            if os.path.isdir(os.path.join(path, component)) and "." not in component:
                components.append(component)
        return components

    def __get_path_images(self, path):
        images = [os.path.join(path, file) for file in os.listdir(path) if
                  any(file.endswith(ext) for ext in self.exts)]
        size = len(images)
        return images, size
