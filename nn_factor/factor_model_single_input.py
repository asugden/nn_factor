import numpy as np
import tensorflow as tf
from keras import layers

from nn_factor.network_tools import default_model, positional_encoding, transformer


class FactorModelSingleInput(default_model.DefaultModel):
    def __init__(
        self,
        feature_dim: int = 37,
        embed_dim: int = 128,
        num_heads: int = 8,
        ff_dim: int = 512,
    ):
        self.pos_encoding = positional_encoding.linear(feature_dim, embed_dim)

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
