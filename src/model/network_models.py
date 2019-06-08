from base_model import ModelBaseline

import matplotlib.pyplot as plt
import numpy as np


class NetWorkModels(ModelBaseline):
    def __init__(self):
        super().__init__()

    def fit_model(self, epochs=100, batch_size=1024, shuffle=True, verbose=True):
        """
            Train model.
        """
        assert self._model != None
        assert self._X_train.shape[0] != 0
        assert self._Y_train.shape[0] != 0
        assert self._X_test.shape[0] != 0
        assert self._Y_test.shape[0] != 0
        assert self._X_validation.shape[0] != 0
        assert self._Y_validation.shape[0] != 0

        self._history = self._model.fit(self._X_train, self._Y_train, epochs=epochs, batch_size=batch_size, shuffle=shuffle,
                                        validation_data=(self._X_validation, self._Y_validation), verbose=verbose)

    def predict_model(self):
        """
            A Numpy array containing the prediction's model.
        """
        assert self._model != None
        assert self._Y_test != None
        assert self._Y_pred == None

        self._Y_pred = np.array([self._model.predict(item)
                                 for item in self._Y_test])

    def plot_model(self):
        assert self.model != None

        plot_model(self.model, show_shapes=True,
                   to_file='../visualization/model.png')

    def plot_model_performance(self, path, mode_1='loss', mode_2='val_loss', label_1='Training loss',
                               label_2='Validation loss', xlabel='Epochs', ylabel='Loss',
                               title='Training and validation loss', savefig='Training and validation loss.png'):
        """
            Using matpltotlib to plot models' performance.
        """
        assert self._history != None

        metric_1 = self._history.history[mode_1]
        metric_2 = self._history.history[mode_2]

        epochs = range(1, len(mode_1) + 1)

        plt.plot(epochs, metric_1, 'bo', label=label_1)
        plt.plot(epochs, metric_2, 'b', label=label_2)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(savefig)
        plt.show()
        plt.clf()

    def save_model(self):
        pass
