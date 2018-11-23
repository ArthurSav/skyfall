import os
import xml.etree.ElementTree as et

import autopep8 as pep


class ReactConverter:

    replace_var_filename = '{_FILENAME_}'
    replace_var_components = '{_COMPONENTS_}'

    generated_code = None

    injectable_file = None  # file to be modified after code is generated i.e /project/hello/ScreenComponent.js
    template_components = None
    template_file = None  # code of the template to generate

    def __init__(self, template_filepath):

        template_filepath = os.path.abspath(template_filepath)

        if not os.path.isfile(template_filepath):
            raise Exception('Could not load template with name "{}".'.format(template_filepath))

        self.__load_file_data(template_filepath)

    def generate(self, components_to_generate, filepath):
        """
        :param components_to_generate:  [{name: 'toolbar', x: 1, y: 1, w: 1, h: 1}]
        :param filepath: used for the filename i.e 'HomeScreen.js' to inject into the template
        :return: generate code
        """

        # create a list of code to inject
        generated_components = self.__generate_components(self.template_components, components_to_generate)

        # inject code into template
        generated_template = self.__generate_template(filepath, generated_components)

        return generated_template

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

    def __load_file_data(self, xml_template_filepath):
        """
        Loads and defines xml elements
        root: xml root
        """

        # parse xml
        root = et.parse(xml_template_filepath).getroot()

        # xml project attributes
        project_name = root.attrib['name']

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
        if template is None or not template:
            raise ValueError('A project template is required')
        if components is None or not components:
            print('No components where found')
            return

        print('Loading xml data from "{}"'.format(xml_template_filepath))
        print('Template components: {}'.format(component_names))

        self.template_components = components
        self.template_file = template

    @staticmethod
    def __generate_components(template_components, named_components):
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

    @staticmethod
    def __generated_components_to_string(generated_components):
        """
        :return a single string of all components code
        """
        components = ""
        for c in generated_components:
            components += c['code']

        return components
