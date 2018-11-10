import os
import xml.etree.ElementTree as et

import autopep8 as pep


class ReactConverter():
    FILE = 'react.xml'

    replace_var_filename = '{_FILENAME_}'
    replace_var_components = '{_COMPONENTS_}'

    generated_code = None

    injectable_file = None  # file to be modified after code is generated i.e /project/hello/ScreenComponent.js
    template_components = None
    template_file = None  # code of the template to generate
    filename_base = None

    def __init__(self, file=None):

        if file is None:
            file = self.FILE

        if not os.path.isfile(file):
            print('Could not load template with name "{}".'.format(file))
            return

        self.__load_file_data(file)

    def generate(self, components_to_generate, update_file=False):
        """
        components_to_generate: list of named components to inject
        update_file: if true, it will update the file provided in the template
        """

        # create a list of code to inject
        generated_components = self.__generate_components(self.template_components, components_to_generate)

        # inject code into template
        generated_template = self.__generate_template(self.filename_base, generated_components)

        # inject template into file
        if update_file:
            self.__load_code_into_file(self.injectable_file, generated_template)

        return generated_template

    def __load_code_into_file(self, filepath, code):

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

    def __generate_template(self, filename_base, generated_components):
        """
        Injects code into template

        filename_base: used as class name for template
        generated_components: list of components and code to inject i.e {'name': 'toolbar', 'code': '...'}
        """

        injectable_code_components = self.__generated_components_to_string(generated_components)

        template = self.template_file
        template = template.replace(self.replace_var_filename, filename_base)
        template = template.replace(self.replace_var_components, injectable_code_components)

        # pep8 code format
        template = pep.fix_code(template, options={'select': ['E1', 'W1']})

        return template

    def __generate_components(self, template_components, named_components):
        """        
        template_components: a list pre-defined component code
        named_components: a list that contains (at least) the names of the components we want to generate i.e [{x, y, w, h, name}]
        
        return a list of components and their code i.e [{'name': 'toolbar', 'code': '...'}]
        """

        g_components = []
        g_names = []

        for component in named_components:
            name = component['name']
            if name in template_components:
                g_components.append({'name': name, 'code': template_components[name]})
                g_names.append(name)
            else:
                print('Trying to generate component "{}" that has not been pre-defined in xml (ignored)'.format(name))

        print('Generating components...')
        print(g_names)

        return g_components

    def __generated_components_to_string(self, generated_components):
        """
        return a single string of all components code
        """
        str = ""
        for c in generated_components:
            str += c['code']

        return str

    def __load_file_data(self, xml_template_filepath):
        """
        Loads and defines xml elements
        root: xml root
        """

        # parse xml
        root = et.parse(xml_template_filepath).getroot()

        # xml project attributes
        path = root.attrib['path']
        components = {}
        component_names = []
        template = None

        # xml project children
        for child in root:

            if child.tag == 'component':
                name = child.attrib['name']
                components[name] = child.text
                component_names.append(name)

            elif child.tag == 'template':
                template = child.text

        # check stuff is not empty
        if path is None or not path:
            raise ValueError('Project path is required')
        if template is None or not template:
            raise ValueError('A project template is required')
        if components is None or not components:
            print('No components where found')

        print('Loading xml data from "{}"'.format(xml_template_filepath))
        print('Project file to modify: {}'.format(path))
        print('Template components: {}'.format(component_names))

        self.injectable_file = path
        self.template_components = components
        self.template_file = template

        self.filename_base = os.path.basename(path)
