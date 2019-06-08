from networks_models import NetWorkModels

from keras import Input
from keras.models import Model
from keras import layers
from keras import optimizers
from keras.utils import plot_model


class CanonizadorNetwork_1(NetWorkModels):
    def build_model(self, embedding_dimension=30, lstm_dimension=30, dense_units=256, optimizer=optimizers.RMSprop, learning_rate=0.001):
        """
            Builds the Neural networks' architecture
        """
        assert self._model == None

        input_data = Input(batch_shape=(None, self._input_tensor.shape[1]))
        x = layers.Embedding(
            self._input_tensor.shape[0], embedding_dimension)(input_data)
        x = layers.LSTM(lstm_dimension)(x)
        x = layers.Dense(dense_units, activation='relu')(x)
        x = layers.Dense(dense_units, activation='relu')(x)
        output = layers.Dense(
            self._output_tensor.shape[1], activation='softmax')(x)

        model = Model(input_data, output)
        optimizer = optimizer(lr=learning_rate)

        model.compile(optimizer=optimizer,
                      loss="categorical_crossentropy", metrics=['acc'])

        self._model = model


class CanonizadorNetwork_2(NetWorkModels):
    def build_model(self, embedding_dimension=30, lstm_dimension=30, dense_units=256, optimizer=optimizers.Adam):
        """
            Builds the Neural networks' architecture
        """
        assert self._model == None

        input_data = Input(batch_shape=(None, self._input_tensor.shape[1]))
        x = layers.Embedding(
            self._input_tensor.shape[0], embedding_dimension)(input_data)
        x = layers.LSTM(lstm_dimension)(x)
        x = layers.Dense(dense_units, activation='relu')(x)
        x = layers.Dense(dense_units, activation='relu')(x)
        x = layers.Dense(dense_units, activation='relu')(x)
        output = layers.Dense(
            self._output_tensor.shape[1], activation='softmax')(x)

        model = Model(input_data, output)
        optimizer = optimizer()

        model.compile(optimizer=optimizer,
                      loss="categorical_crossentropy", metrics=['acc'])

        self._model = model
