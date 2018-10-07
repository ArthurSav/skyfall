from keras.models import load_model
from skyfall import utils_image

class Predict:

    model = None
    class_names = None

    def __init__(self, class_names, model_path = None, model = None):
        """
        model_path: path to pre-trained model
        class_names: class names corresponding to predictions
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
    
    def predict(self, images, show_predictions = False):
        """
        images: numpy array of images to predict (images should be normalized and resized)
        show_predictions: if true it will plot predictions alongside provided images
        """
        predictions = self.model.predict(images)
        
        # plot images and show their scores
        utils_image.show_prediction_list(images = images, predictions = predictions, class_names = self.class_names)