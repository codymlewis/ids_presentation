"""
Federated learning client for the UNSW-NB15 task.
"""

import typing
import tensorflow as tf
import flwr as fl

import dataloader


def create_model(sample_shape: typing.Tuple[int]) -> tf.keras.Model:
    """
    Create a multilayered fully connected neural network.

    Arguments:
    - sample_shape: array shape of a single sample that the model will take as input
    """
    inputs = tf.keras.Input(sample_shape)
    x = tf.keras.layers.Dense(100, activation="relu")(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dense(50, activation="relu")(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"]
    )
    return model


class Client(fl.client.NumPyClient):
    def __init__(self):
        self.X_train, self.Y_train, self.X_test, self.Y_test = dataloader.load_data()
        self.model = create_model(self.X_train.shape[1:])

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        history = self.model.fit(self.X_train, self.Y_train, epochs=1, batch_size=64)
        return self.model.get_weights(), len(self.X_train), {k: v[-1] for k, v in history.history.items()}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_test, self.Y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}


if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=Client())
