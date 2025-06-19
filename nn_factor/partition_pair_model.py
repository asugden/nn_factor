from typing import Literal

import numpy as np
import tensorflow as tf
from keras import layers

from nn_factor.network_tools import default_model, positional_encoding, transformer


class PartitionPairModel(default_model.DefaultModel):
    def __init__(
        self,
        feature_dim: int = 36,
        embed_dim: int = 64,
        num_heads: int = 8,
        # ff_dim: int = 512,
        attention_type: Literal["cross", "self"] = "cross",
        skip_mlp: bool = False,
        activation_fn: str = "gelu",
    ):
        # Inputs
        inputs = layers.Input(shape=(feature_dim,), dtype=np.int32, name="inputs")
        x = tf.cast(inputs, tf.float32, name="to_float")  # (batch, 36) → floats

        # Split into the two inputs
        x1, x2 = tf.split(
            x, [feature_dim // 2, feature_dim // 2], axis=1, name="split_arrays"
        )

        # We can either do "global projection" or "per-element projection".
        # For a transformer, we need to have per-element projection.
        x1 = tf.expand_dims(x1, -1)
        x2 = tf.expand_dims(x2, -1)

        # Embed the inputs so that we can use the transformer
        self.embedder = layers.TimeDistributed(
            layers.Dense(embed_dim, activation=activation_fn),
            name="per_element_projection",
        )
        x1 = self.embedder(x1)
        x2 = self.embedder(x2)

        # Add positional encoding and run through self-attention
        # transformer
        self.pos = positional_encoding.sinusoidal(feature_dim // 2, embed_dim)
        self.add_pos = layers.Lambda(lambda v: v + self.pos, name="add_position")
        x1 = self.add_pos(x1)
        x2 = self.add_pos(x2)
        self.selfattn = transformer.TransformerBlock(
            embed_dim=embed_dim,
            key_dim=embed_dim // 2,
            # value_dim=embed_dim // 2,
            num_heads=num_heads // 2,
            # ff_dim=ff_dim,
            name="self_attention",
            activation_fn=activation_fn,
            skip_mlp=skip_mlp,
        )
        x1 = self.selfattn(x1)
        x2 = self.selfattn(x2)

        # Now run through cross-attention transformer
        # Cross attention is NOT symmetric. The key and query matrices
        # are different. If we were computing something like an
        # eigenvalue, which is consistent with transpose, I would
        # compute the cross attention of x1 and x2 and the concatenate
        # teh results with the cross attention of x2 and x1. In this
        # case, because the eigenvector of a matrix is not the same as
        # its transpose, we may not benefit from both computations.
        # NOTE: This is WRONG and we should do symmetric!

        if attention_type == "cross":
            # Now pass through cross-attention transformer
            x = transformer.TransformerBlock(
                embed_dim=embed_dim,
                key_dim=embed_dim,
                value_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                activation_fn=activation_fn,
                skip_mlp=skip_mlp,
                name="cross_attention",
            )(x1, cross_inputs=x2)
        else:
            # Now run through SELF-ATTENTION transformer
            # The issue is that we somehow need to encode the difference
            # between the two vectors. So, let's use two dense layers, one
            # for x1 and the other for x2. Then concatenate and pass through
            # transformer.
            # I tried using "labels" of a vector added to x1 and x2. That
            # was terrible.
            x1 = layers.Dense(embed_dim, activation="relu", name="mark_as_x1")(x1)
            x2 = layers.Dense(embed_dim, activation="relu", name="mark_as_x2")(x2)
            x = layers.Concatenate(axis=1)([x1, x2])
            x = transformer.TransformerBlock(
                embed_dim=embed_dim,
                key_dim=embed_dim,
                # value_dim=embed_dim,
                num_heads=num_heads,
                # ff_dim=ff_dim,
                activation_fn=activation_fn,
                skip_mlp=skip_mlp,
                name="combined_self_attention",
            )(x)

        # Feed-forward classification
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation=activation_fn)(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(16, activation=activation_fn)(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(4, activation=activation_fn)(x)

        # Output layer
        output = layers.Dense(1, activation="sigmoid")(x)

        # Create model
        self.model = tf.keras.Model(inputs=inputs, outputs=output)
