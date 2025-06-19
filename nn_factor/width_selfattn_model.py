from typing import Literal

import numpy as np
import tensorflow as tf
from keras import layers

from nn_factor.network_tools import default_model, positional_encoding, transformer


class WidthSelfAttnModel(default_model.DefaultModel):
    def __init__(
        self,
        w: int = 20,
        embed_dim: int = 64,
        num_heads: int = 8,
        activation_fn: str = "gelu",
        skip_mlp: bool = False,
        dropout_rate: float = 0.1,
    ):
        inputs = layers.Input(shape=(w,), dtype=np.int32, name="inputs")
        x = tf.cast(inputs, tf.float32, name="to_float")

        x = tf.expand_dims(x, -1)

        # Embed each of the integers of X
        x = layers.TimeDistributed(
            layers.Dense(embed_dim, activation=activation_fn),
            name="per_element_projection",
        )(x)

        # Add positional encoding
        self.pos = positional_encoding.sinusoidal(w, embed_dim)
        x = layers.Lambda(lambda v: v + self.pos, name="add_position")(x)

        # Self attention transformer
        # NOTE: It may be important to have sequential transformers
        x = transformer.TransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            activation_fn=activation_fn,
            skip_mlp=skip_mlp,
            name="self_attention",
        )(x)

        # Average pool the embeddings
        x = layers.GlobalAveragePooling1D()(x)

        # And do our traditional divide by 4 dense layer structure
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation=activation_fn)(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(16, activation=activation_fn)(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(4, activation=activation_fn)(x)

        # Output layer
        output = layers.Dense(1, activation="linear", name="prediction")(x)

        # Create model
        self.model = tf.keras.Model(inputs=inputs, outputs=output)
