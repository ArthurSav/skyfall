import numpy as np

"""
Model training utilities
"""


def calculate_accuracy(predictions, labels):
    """
    predictions: model predictions for a dataset
    labels: prediction labels (normalized)

    return accuracy
    """

    correct = 0
    for i in range(len(predictions)):
        predicted_label = np.argmax(predictions[i])
        true_label = np.argmax(labels[i])
        if predicted_label == true_label:
            correct += 1
    accuracy = correct / (len(predictions) * 1.0)

    print("Samples: {}, Correct: {}, Accuracy: {}".format(len(predictions), correct, accuracy))

    return accuracy
