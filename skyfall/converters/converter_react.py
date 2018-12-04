import os
import xml.etree.ElementTree as et

import autopep8 as pep
from mako.template import Template


class ReactConverter:

    template = None
    template_components = None

    def __init__(self, filepath_template, filepath_template_components):
        """
        :param filepath_template: template filepath
        :param filepath_template_components: component template filepath
        """

        filepath_template = os.path.abspath(filepath_template)
        filepath_template_components = os.path.abspath(filepath_template_components)

        if not os.path.isfile(filepath_template):
            raise Exception('Could not load template with name "{}".'.format(filepath_template))

        if not os.path.isfile(filepath_template_components):
            raise Exception('Could not load template with name "{}".'.format(filepath_template_components))

        self.template = Template(filename=filepath_template)
        self.__load_template_components(filepath_template_components)

    def generate(self, cls_name, components_to_generate):
        """
        :param components_to_generate:  [{name: 'toolbar', x: 1, y: 1, w: 1, h: 1}]
        :param cls_name: class name to be injected
        :return: generate code
        """

        # create a list of code to inject
        generated_components = self.__parse_components(self.template_components, components_to_generate)

        # inject code into template
        generated_template = self.__generate_template(cls_name, generated_components)

        return generated_template

    def __generate_template(self, name, generated_components):
        """
         Injects code into template
        :param name: class name
        :param generated_components: list of injectable component code
        :return: rendered template
        """

        result = self.template.render(filename=name, components=generated_components)

        # pep8 code format
        result = pep.fix_code(result, options={'select': ['E1', 'W1']})

        return result

    def __load_template_components(self, filepath):
        """
        Load component elements from xml file
        :param filepath: i.e /projects/skyfall/template.xml
        """

        # parse xml
        root = et.parse(filepath).getroot()

        # xml project attributes
        project_name = root.attrib['name']

        components = {}
        component_names = []

        # xml project children
        for child in root:
            if child.tag == 'component':
                name = child.attrib['name']
                components[name] = child.text
                component_names.append(name)

        if components is None or not components:
            print('No components where found')
            return

        print('Loading xml data from "{}"'.format(filepath))
        print('Template components: {}'.format(component_names))

        self.template_components = components

    @staticmethod
    def __parse_components(template_components, named_components):
        """
        Matches a list of component names to template components
        :param template_components: [{'name': 'toolbar', 'code': '...'}']
        :param named_components: i.e ['toolbar', 'button'...]
        :return: a list of components and their code i.e [{'name': 'toolbar', 'code': '...'}]
        """

        g_components = []
        g_names = []

        for component in named_components:
            name = component['name']
            if name in template_components:
                code = template_components[name]
                g_names.append(name)
                g_components.append(code)
            else:
                print('Trying to generate component "{}" that has not been pre-defined in xml (ignored)'.format(name))

        print('Generating components...')
        print(g_names)

        return g_components
