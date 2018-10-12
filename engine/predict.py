from keras.models import load_model
from skyfall.utils import utils_image
from skyfall.models.model_p1 import MetadataContour, MetadataPrediction
import numpy as np

class Predict:

    model = None
    class_names = None

    def __init__(self, class_names, model = None, model_path = None):
        """
        model: pre-trained model
        model_path: load pre-trained model from path
        class_names: available class names to be used when showing predictions
        """

        self.class_names = class_names

        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = load_model(model_path)

        if self.model is None:
            print("Nothing to load")
            return
            
        if self.class_names is None:
            print("No class names provided")
            return
    
    def predict(self, images, metadata = None, show_predictions = False):
        """
        images: numpy array of images to predict (images should be normalized and resized)
        metadata: contour information for provided images [width, height, x, y]
        show_predictions: if true it will plot predictions alongside provided images

        Note: metadata should be in the same order as predictions i.e they belong to the same image
        """
        predictions = self.model.predict(images)
        metadata_predictions = []

        if metadata is not None:
            for idx, prediction in enumerate(predictions):
                predicted_label = np.argmax(prediction)
                score = prediction[predicted_label]
                predicted_label_name = self.class_names[predicted_label]
                metadata_contour = metadata[idx]
                metadata_prediction = MetadataPrediction(metadata_contour, predicted_label, predicted_label_name, score)

                metadata_predictions.append(metadata_prediction)


        # plot images and show their scores
        if show_predictions:
            utils_image.show_prediction_list(images = images, predictions = predictions, class_names = self.class_names)

        return metadata_predictions