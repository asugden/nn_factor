import numpy as np
import tensorflow as tf
from keras import callbacks, layers

from nn_factor import transformer


class FactorModel:
    def __init__(
        self,
        feature_dim: int = 36,
        embed_dim: int = 128,
        num_heads: int = 8,
        ff_dim: int = 512,
    ):
        # Create reusable resources
        self.pos_half = self._create_positional_encoding(feature_dim // 2, embed_dim)
        self.pos_whole = self._create_positional_encoding(feature_dim, embed_dim)

        self.shared_block = transformer.TransformerBlock(
            embed_dim=embed_dim, num_heads=num_heads // 2, ff_dim=ff_dim
        )
        self.merge_block = transformer.TransformerBlock(
            embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim
        )

        # Inputs
        input = layers.Input(shape=(feature_dim,), dtype=np.int32)

        # Embed the inputs so that we can use the transformer
        x = tf.cast(input, tf.float32)  # (batch, 36) → floats
        x = tf.expand_dims(x, axis=-1)
        x = layers.Dense(embed_dim, activation="relu")(x)

        # Split into the two inputs
        x1, x2 = tf.split(x, [feature_dim // 2, feature_dim // 2], axis=1)

        # Add positional encoding and run through half transformer
        x1 = layers.Lambda(lambda v: v + self.pos_half)(x1)
        x2 = layers.Lambda(lambda v: v + self.pos_half)(x2)
        x1 = self.shared_block(x1)
        x2 = self.shared_block(x2)

        # Now pass through whole transformer
        x = layers.Concatenate(axis=1)([x1, x2])
        x = layers.Lambda(lambda v: v + self.pos_whole)(x)
        x = self.merge_block(x)

        # Feed-forward classification
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(16, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(4, activation="relu")(x)

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
