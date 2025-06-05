import numpy as np
import tensorflow as tf
from keras import callbacks, layers

from nn_factor import transformer


class FactorModel:
    def __init__(
        self,
        feature_dim: int = 37,
        vocab_dim: int = 32,
        embed_dim: int = 128,
        num_heads: int = 8,
        ff_dim: int = 512,
    ):
        self.pos_encoding = self._create_positional_encoding(feature_dim, embed_dim)

        input = layers.Input(shape=(feature_dim,), dtype=np.int32)
        x = tf.cast(input, tf.float32)  # (batch, 37) → floats
        x = tf.expand_dims(x, axis=-1)

        x = layers.Dense(embed_dim, activation="relu")(x)
        x = layers.Lambda(lambda inp: inp + self.pos_encoding)(x)

        x = transformer.TransformerBlock(
            embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim
        )(x)

        # Feed-forward classification/regression head
        x = layers.GlobalAveragePooling1D()(x)

        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(16, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(8, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(2, activation="relu")(x)

        # Output layer
        output = layers.Dense(1, activation="sigmoid")(x)

        # Create model
        self.model = tf.keras.Model(inputs=input, outputs=output)

    def train(self, features, labels, learning_rate: float = 0.0001, epochs: int = 100):
        es = callbacks.EarlyStopping(
            monitor="loss",  # what to watch
            patience=3,  # how many epochs to wait
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
            # callbacks=[es],
        )

    def predict(self, features):
        return self.model.predict(features)

    def evaluate(self, features, labels):
        return self.model.evaluate(features, labels)

    def summary(self):
        return self.model.summary()

    def save(self, path: str):
        self.model.save(path)

    def _create_positional_encoding(self, seq_len, d_model):
        """Create sinusoidal positional encoding"""
        positions = np.arange(seq_len, dtype=np.float32)

        # Broadcast to embedding dimension
        pos_encoding = np.tile(positions.reshape(-1, 1), (1, d_model))

        # Normalize to prevent large values
        pos_encoding = pos_encoding / seq_len

        return tf.constant(pos_encoding, dtype=tf.float32)
