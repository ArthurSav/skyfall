from keras.models import load_model
from skyfall import utils_image

class Predict:

    model = None
    class_names = None

    def __init__(self, model_path, class_names):
        """
        model_path: path to pre-trained model
        class_names: class names corresponding to predictions
        """
        self.model = load_model(model_path)
        self.class_name = class_names
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

        columns = None
        if len(images) <= 10:
            columns = 2
        
        # plot images and show their scores
        utils_image.show_prediction_list(images = images, predictions = predictions, class_names = self.class_names, columns = columns)