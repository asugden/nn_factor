import numpy as np
import tensorflow as tf
from keras import callbacks, layers

from nn_factor.network_tools import transformer


class DefaultModel:
    def __init__(self):
        self.model = None

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        learning_rate: float = 0.0001,
        epochs: int = 100,
    ) -> None:
        """Train the neural network with early stopping.

        Args:
            features (np.ndarray): input features as array of shape
                (rows x embed_dim)
            labels (np.ndarray): training labels of shape (rows)
            learning_rate (float, optional): initial learning rate of
                the model. Defaults to 0.0001.
            epochs (int, optional): number of training epochs.
                Defaults to 100.
        """
        es = callbacks.EarlyStopping(
            monitor="loss",  # what to watch
            patience=10,  # how many epochs to wait
            min_delta=1e-4,  # any improvement smaller than this counts as no-op
            mode="auto",  # 'min' for loss, 'max' for accuracy, or 'auto' to infer
            restore_best_weights=True,  # after stopping, roll back to epoch with best monitored metric
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        self.model.fit(
            features,
            labels,
            epochs=epochs,
            batch_size=512,
            callbacks=[es],
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict labels from features

        Args:
            features (np.ndarray): input features for prediction of
                shape (rows, embed_dim)

        Returns:
            np.ndarray: probability of it being 1 from the training set
                of shape (rows)
        """
        return self.model.predict(features)

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Return the accuracy of the model from held-out features and
        labels.

        Args:
            features (np.ndarray): input features as array of shape
                (rows x embed_dim)
            labels (np.ndarray): testing labels of shape (rows)

        Returns:
            float: the accuracy of the model on held-out data
        """
        return self.model.evaluate(features, labels)[1]

    def summary(self):
        """Return the model summary from tensorflow."""
        return self.model.summary()

    def save(self, path: str):
        """Save the model into a folder at path."""
        self.model.save(path)
