class LayoutUtils:
    @staticmethod
    def remove_children(self, layout):
        """
        Removes children from parent layout
        :param layout Qwidget
        """
        while layout.count():
            child = layout.takeAt(0)
            if child.widget() is not None:
                child.widget().deleteLater()
            elif child.layout() is not None:
                self.clear_layout(child.layout())
