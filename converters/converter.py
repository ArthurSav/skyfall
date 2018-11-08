from skyfall.models.model_utils import ConverterType
import xml.etree.ElementTree as et


class Converter():

    LABELS = ['toolbar', 'toolbar_search', 'fab', 'list_item', 'image', 'checkbox']

    def __init__(self):
        pass

    def convert(self, converter_type, metadata):
        """
        type: ConverterType
        metadata: each type requires it's own set of metadata to proccess
        """

        if metadata == None:
            print('Nothing to do. No metadata provided.')
            return
        
        if converter_type == None or converter_type == ConverterType.NONE:
            print("Nothing to convert")
            return
        elif converter_type == ConverterType.REACT:
            self.__convert_from_mobile_screen_to_react_native(metadata)

    ####################################################################
    # REACT NATIVE                                                     #
    ####################################################################
    
    def __convert_from_mobile_screen_to_react_native(self, metadata):
        """
        Converts basic cropping and score metadata into react native code
        """

        metadata = self.__sort_by_xy(metadata)

        for component in metadata:
            pass
        

    def __sort_by_xy(self, lst):
        return lst

    def __create_component(self, name, x, y, w, h):

        # toolbar
        if name == self.LABELS[0]:
            pass
        # toolbar search
        elif name == self.LABELS[1]:
            pass
        # fab button
        elif name == self.LABELS[2]:
            pass
        # list item
        elif name == self.LABELS[3]:
            pass
        # image
        elif name == self.LABELS[4]:
            pass
        # checkbox
        elif name == self.LABELS[5]:
            pass

class ReactConverter():

    FILE = 'react.xml'
    generated_code = None

    file_data = None

    def __init__(self, file = None):

        if file is None:
            file = self.FILE;

        file_data = et.parse(file).getroot()

    def __load_components(self, components):
        """
        components: a sorted list that contains [x, y, w, h, name] for each component
        """
        pass

    def __update_file(self):
        """
        Loads components into target file
        """
        pass    

