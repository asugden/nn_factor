from typing import Literal

import numpy as np
import tensorflow as tf
from keras import layers

from nn_factor.network_tools import default_model, positional_encoding, transformer


class KleshchevSelfAttnModel(default_model.DefaultModel):
    def __init__(
        self,
        max_M: int = 5,
        max_N: int = 20,
        embed_dim: int = 64,
        num_heads: int = 8,
        activation_fn: str = "gelu",
        k_segment_as_multiplicative: bool = False,
        k_segment_mult_dim: int = 8,
        skip_mlp: bool = False,
        dropout_rate: float = 0.1,
    ):
        # Inputs (in the form N, N, N)
        # The first N is the K values for each entry (to match the order
        # of the CNN input), padded to max_N with -1 if N < max_N
        # The second N is the concatenated non-zero values of the
        # partitions, padded to max_N with -1 if N < max_N
        # The third N is the segment/partition number labels, padded to
        # max_N with -1 if N < max_N
        # NOTE: This is DIFFERENT from the inputs to cnn or cross-attn
        # It is too much effort to compute the transformations within
        # tensorflow.
        inputs = layers.Input(shape=(max_N * 3,), dtype=np.int32, name="inputs")
        x = tf.cast(inputs, tf.float32, name="to_float")

        # Split into the two inputs, K and x
        K_mat, x, segment = tf.split(x, [max_N, max_N, max_N], axis=1)
        K_mat = tf.expand_dims(K_mat, -1)
        x = tf.expand_dims(x, -1)
        segment = tf.expand_dims(segment, -1)

        # For self-attention, we can create a vector of length max_N and
        # do all computations within it. It should also bring the -1s
        # with it
        # Because the multicharge is of a set of partitions that add up
        # to a fixed number, N, the longest it can be is a set of 1s of
        # length N.

        # Embed each of the integers of X
        x = layers.TimeDistributed(
            layers.Dense(embed_dim, activation=activation_fn),
            name="per_element_projection",
        )(x)

        # Now we have to either add our segment embeddings and k values
        # or concatenate them
        if not k_segment_as_multiplicative:
            # Via addition, they need to be embedded to the same
            # dimension as X
            segment = layers.TimeDistributed(
                layers.Dense(embed_dim, activation=activation_fn),
                name="per_segment_projection",
            )(segment)
            K_mat = layers.TimeDistributed(
                layers.Dense(embed_dim, activation=activation_fn),
                name="per_k_projection",
            )(K_mat)

            # Added dropout here to mimic the transformer architecture
            x = layers.Dropout(dropout_rate)(x)
            x = layers.LayerNormalization(epsilon=1e-6)(x + segment + K_mat)
        else:
            # If we concatenate them, then first we embed segment and
            # K_mat to smaller dimensions
            segment = layers.TimeDistributed(
                layers.Dense(k_segment_mult_dim, activation=activation_fn),
                name="per_segment_projection",
            )(segment)
            K_mat = layers.TimeDistributed(
                layers.Dense(k_segment_mult_dim, activation=activation_fn),
                name="per_k_projection",
            )(K_mat)

            # Then concatenate the vectors
            concat = layers.Concatenate(axis=-1, name="concat_k_seg")(
                [x, segment, K_mat]
            )

            # And re-project to embed_dim dimensions
            x = layers.Dense(
                embed_dim, activation=activation_fn, name="reproject_k_seg"
            )(concat)

            # Add dropout here to ensure consistency between labeling
            # techniques
            x = layers.Dropout(dropout_rate)(x)

        # Add positional encoding
        self.pos = positional_encoding.sinusoidal(max_N, embed_dim)
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
        output = layers.Dense(1, activation="sigmoid", name="prediction")(x)

        # Create model
        self.model = tf.keras.Model(inputs=inputs, outputs=output)
