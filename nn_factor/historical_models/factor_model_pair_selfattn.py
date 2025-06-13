import numpy as np
import tensorflow as tf
from keras import layers

from nn_factor.network_tools import default_model, positional_encoding, transformer


class FactorModelPairSelfAttn(default_model.DefaultModel):
    def __init__(
        self,
        feature_dim: int = 36,
        embed_dim: int = 128,
        num_heads: int = 8,
        ff_dim: int = 512,
    ):
        # Create reusable resources
        self.pos_half = positional_encoding.linear(feature_dim // 2, embed_dim)
        self.pos_whole = positional_encoding.linear(feature_dim, embed_dim)

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
