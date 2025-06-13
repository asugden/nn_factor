from typing import Literal

import numpy as np
import tensorflow as tf
from keras import layers

from nn_factor.network_tools import default_model, positional_encoding, transformer


class KleshchevCNNModel(default_model.DefaultModel):
    def __init__(
        self,
        max_M: int = 5,
        max_N: int = 20,
        filter_dim: int = 32,
        activation_fn: str = "relu",
    ):
        # Inputs (in the form K, M*N)
        # Specifically, the K matrix, and then each of the partitions
        # padded out to N with 0s and max_N with -1s.
        # If M < maxM, pad K to max_M with -1 and pad M*N to
        # max_M*max_N with -1
        inputs = layers.Input(
            shape=(max_M + max_M * max_N,), dtype=np.int32, name="inputs"
        )
        x = tf.cast(inputs, tf.float32, name="to_float")

        # Split into the two inputs, K and x
        K_vec, x = tf.split(x, [max_M, max_M * max_N], axis=1)

        # Tile K to produce a second channel, and combine with x
        # We have to take into account the batch dimension
        K_mat = tf.tile(
            tf.expand_dims(K_vec, axis=1), [1, max_N, 1]
        )  # (batch, max_N, max_M)
        x = tf.reshape(x, (-1, max_N, max_M))  # (batch, max_N, max_M)

        # For a CNN, we will use the K vector as a channel of the matrix
        x = tf.stack([K_mat, x], axis=-1)  # (batch, max_N, max_M, 2)

        # NOTE: As of now, we are using a single additional channel.
        # If we would like, we can expand this to an embedding of 0 and
        # 1 ness.

        # For low values of max_M, there are very few computation layers
        # within the CNN. Traditionally you need multiple layers to get
        # higher-order representations. Therefore if max_M is less than
        # 6, I propose doubling up on computations.

        layer_count = (min(max_M, max_N) + 1) // 2 - 1
        # Remember, traditional Conv2D layers decrease each dimension by
        # 2 on each round. So 5 -> 3 -> 1 is 2 layers.

        # Also, relu outperforms gelu on CNNs traditionally
        for i in range(layer_count):
            if layer_count < 5:
                # Does not decrease the layer dimensions by 2
                x = layers.Conv2D(
                    filter_dim if i > 2 or layer_count - i < 2 else filter_dim // 2,
                    (3, 3),
                    activation="relu",
                    padding="same",
                    name=f"conv_same_{i+1}",
                )(x)
            x = layers.Conv2D(
                filter_dim if i > 2 or layer_count - i < 2 else filter_dim // 2,
                (3, 3),
                activation="relu",
                name=f"conv_dec_{i+1}",
            )(x)

        # Now we flatten
        x = layers.Flatten()(x)

        # And do our traditional divide by 4 dense layer structure
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation=activation_fn)(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(16, activation=activation_fn)(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(4, activation=activation_fn)(x)

        # Output layer
        output = layers.Dense(1, activation="sigmoid", name="prediction")(x)

        # Create model
        self.model = tf.keras.Model(inputs=inputs, outputs=output)

        # nonzero_mask = vec != 0

        # # Use tf.boolean_mask to remove elements that are zero
        # filtered_vec = tf.boolean_mask(vec, nonzero_mask)
