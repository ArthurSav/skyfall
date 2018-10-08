class MetadataContour():

    x = None
    y = None
    w = None # width
    h = None # height

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

class MetadataPrediction():

    metadata_contour = None
    predicted_label = None
    predicted_label_name = None
    score = None

    def __init__(self, metadata_contour, predicted_label, predicted_label_name, score):
        self.metadata_countour = metadata_contour
        self.predicted_label = predicted_label
        self.predicted_label_name = predicted_label_name
        self.score = score




        