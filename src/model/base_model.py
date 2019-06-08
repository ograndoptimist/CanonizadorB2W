import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


class ModelBaseline(object):
    """
        A model responsible to classify query strings provided by customers on our 4 brands.
    """

    def __init__(self):
        self._input_tensor = None
        self._output_tensor = None
        self._model = None
        self._history = None
        self._X_train = None
        self._Y_train = None
        self._X_test = None
        self._Y_test = None
        self._X_validation = None
        self._Y_validation = None
        self._Y_pred = None
        self._metric_confusion_matrix = None

    def load_data(self, path_input, path_output):
        """
            Loading the input e output tensors to the model.
        """
        assert self._input_tensor == None
        assert self._output_tensor == None

        self._input_tensor = pd.read_csv(path_input, compression='gzip')
        self._output_tensor = pd.read_csv(path_output, compression='gzip')

        self._split_dataset()

    def build_model(self):
        """
            Builds the models' architecture
        """
        pass

    def _split_dataset(self):
        """
            Split the dataset in Train, Validation and Test set.
        """
        assert self._X_train == None
        assert self._Y_train == None
        assert self._X_test == None
        assert self._Y_test == None
        assert self._X_validation == None
        assert self._Y_validation == None

        self._X_train_, self._X_test, self._Y_train_, self._Y_test = train_test_split(
            self._input_tensor, self._output_tensor, random_state=42)
        self._X_train, self._X_validation, self._Y_train, self._Y_validation = train_test_split(
            self._X_train_, self._Y_train_, random_state=42)

    def fit_model(self):
        """
            Train the model.
        """
        pass

    def plot_model_performance(self):
        """
            Using matpltotlib to plot models' performance.
        """
        pass

    def model_predict(self):
        """
            Model's prediction.
        """
        pass

    def save_model(self):
        pass

    def confusion_matrix(self):
        """
            Evaluating performance's model from accurater ways.
        """
        assert self._metric_confusion_matrix == None

        self._metric_confusion_matrix = confusion_matrix(self._Y_test, self._Y_pred,
                                                         labels=range(self._Y_train.shape[1]))
