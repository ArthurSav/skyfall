import os
from enum import Enum

from skyfall.converters.converter_react import ReactConverter


class ConverterType(Enum):
    REACT = 1


class Converter:
    filepath_output = None

    react = ReactConverter('templates/react_screen.js', 'templates/react_components.xml')

    def __init__(self):
        pass

    def set_ouput(self, filepath):
        self.filepath_output = filepath

    def convert(self, items):
        self.__check()

        if items is None or not items:
            print("Nothing to convert")
            return

        # use platform specific converter
        code = self.__convert_react(items)

        # export code
        self.__on_code_generated(code)

    def __on_code_generated(self, code):
        self.__inject_code_into_file(self.filepath_output, code)

    @staticmethod
    def __inject_code_into_file(filepath, code):

        # check a file with an extension is provided
        fileinfo = os.path.splitext(os.path.basename(filepath))
        if len(fileinfo) < 2:
            print('Could not load code into {}'.format(filepath))
            print('Please provide a valid file')
            return

        print('Injecting code into {}'.format(filepath))

        injectable_file = open(filepath, "w+")
        injectable_file.write(code)
        injectable_file.close()

        print('File updated!')

    def __check(self):
        if self.filepath_output is None or not os.path.isfile(self.filepath_output):
            raise Exception("Please provide a valid output file")

    """
     PLATFORM SPECIFIC CONVERTERS
    """

    def __convert_react(self, items):
        proccessed = []
        for item in items:
            contours = item['contours']
            proccessed.append({'name': item['label'],
                               'x': contours['x'],
                               'y': contours['y'],
                               'w': contours['w'],
                               'h': contours['h']})
        # sort on y axis
        proccessed = sorted(proccessed, key=lambda e: e['y'])

        # react class name
        name = os.path.splitext(os.path.basename(self.filepath_output))[0]

        return self.react.generate(name, proccessed)
