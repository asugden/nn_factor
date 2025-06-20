import numpy as np
import tensorflow as tf

from nn_factor.network_tools import default_model, positional_encoding, transformer


class FactorModelSelfAttn(default_model.DefaultModel):
    def __init__(
        self,
        feature_dim: int = 36,
        embed_dim: int = 64,
        num_heads: int = 8,
        ff_dim: int = 512,
    ):
        # Inputs
        inputs = tf.keras.layers.Input(
            shape=(feature_dim,), dtype=np.int32, name="inputs"
        )
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
        self.embedder = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(embed_dim, activation="relu"),
            name="per_element_projection",
        )
        x1 = self.embedder(x1)
        x2 = self.embedder(x2)

        # Add positional encoding and run through self-attention
        # transformer
        self.pos = positional_encoding.sinusoidal(feature_dim // 2, embed_dim)
        self.add_pos = tf.keras.layers.Lambda(
            lambda v: v + self.pos, name="add_position"
        )
        x1 = self.add_pos(x1)
        x2 = self.add_pos(x2)
        self.selfattn = transformer.TransformerBlock(
            embed_dim=embed_dim,
            key_dim=embed_dim // 2,
            value_dim=embed_dim // 2,
            num_heads=num_heads // 2,
            ff_dim=ff_dim,
            name="self_attention",
        )
        x1 = self.selfattn(x1)
        x2 = self.selfattn(x2)

        # Now run through SELF-ATTENTION transformer
        # The issue is that we somehow need to encode the difference
        # between the two vectors. So, let's use two dense tf.keras.layers, one
        # for x1 and the other for x2. Then concatenate and pass through
        # transformer.
        # I tried using "labels" of a vector added to x1 and x2. That
        # was terrible.
        x1 = tf.keras.layers.Dense(embed_dim, activation="relu", name="mark_as_x1")(x1)
        x2 = tf.keras.layers.Dense(embed_dim, activation="relu", name="mark_as_x2")(x2)
        x = tf.keras.layers.Concatenate(axis=1)([x1, x2])
        self.selfattn_comb = transformer.TransformerBlock(
            embed_dim=embed_dim,
            key_dim=embed_dim,
            value_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            name="combined_self_attention",
        )
        x = self.selfattn_comb(x)

        # Feed-forward classification
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(16, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(4, activation="relu")(x)

        # Output layer
        output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        # Create model
        self.model = tf.keras.Model(inputs=inputs, outputs=output)
