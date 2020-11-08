import tensorflow as tf
import keras
from keras import layers, models


class NeuralNet:

    def __init__(self, input_dim=(8, 8, 1)):
        self.model = self.create_model(input_dim)

    def create_model(self, input_dim=(8,  8, 1)):

        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_dim))
        # model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (2, 2), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])

        return model

    def fit(self, X, y, batch_size=64, epochs=100):
        # TODO
        pass

    def predict(self, X):
        # TODO
        pass

    def evaluate(self, X, y):
        # TODO
        pass
