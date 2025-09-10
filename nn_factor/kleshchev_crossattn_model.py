from typing import Literal

import numpy as np
import tensorflow as tf

from nn_factor.network_tools import default_model, positional_encoding, transformer


class KleshchevCrossAttnModel(default_model.DefaultModel):
    def __init__(
        self,
        max_M: int = 10,
        max_N: int = 20,
        embed_dim: int = 64,
        num_heads: int = 8,
        activation_fn: str = "gelu",
        multiple_cross_attn: bool = False,
        k_segment_as_multiplicative: bool = False,
        k_segment_mult_dim: int = 8,
        skip_mlp: bool = False,
        dropout_rate: float = 0.1,
    ):
        # Inputs (in the form K, M*N)
        # Specifically, the K matrix, and then each of the partitions
        # padded out to N with 0s and max_N with -1s.
        # If M < maxM, pad K to max_M with -1 and pad M*N to
        # max_M*max_N with -1
        inputs = tf.keras.layers.Input(
            shape=(max_M + max_M * max_N,), dtype=np.int32, name="inputs"
        )
        x = tf.keras.layers.Lambda(lambda v: tf.cast(v, tf.float32, name="to_float"))(
            inputs
        )

        # Split into the two inputs, K and x
        K_vec, x = tf.keras.layers.Lambda(
            lambda v: tf.split(v, [max_M, max_M * max_N], axis=1),
            name="split_arrays",
        )(x)
        xs = tf.keras.layers.Lambda(
            lambda v: tf.split(v, [max_N for _ in range(max_M)], axis=1),
            name="split_partitions",
        )(x)

        # Embed each of the integers of each X
        self.embedder = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(embed_dim, activation=activation_fn),
            name=f"per_element_projection",
        )
        x = []
        for i in range(len(xs)):
            x_i = tf.keras.layers.Lambda(lambda v: tf.expand_dims(v, -1))(xs[i])
            x_i = self.embedder(x_i)
            x.append(x_i)
        xs = x

        # And do self attention on each
        self.pos = positional_encoding.sinusoidal(max_N, embed_dim)
        self.add_pos = tf.keras.layers.Lambda(
            lambda v: v + self.pos, name="add_position"
        )
        self.selfattn = transformer.TransformerBlock(
            embed_dim=embed_dim,
            key_dim=embed_dim // 2,
            num_heads=num_heads // 2,
            name="self_attention",
            activation_fn=activation_fn,
            skip_mlp=skip_mlp,
        )
        for i in range(len(xs)):
            xs[i] = self.add_pos(xs[i])
            xs[i] = self.selfattn(xs[i])

        # Now we have to either add our k values or concatenate them
        K_mat = tf.keras.layers.Lambda(lambda v: tf.expand_dims(v, -1))(K_vec)
        if not k_segment_as_multiplicative:
            # Via addition, they need to be embedded to the same
            # dimension as X
            K_mat = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(embed_dim, activation=activation_fn),
                name="per_k_projection",
            )(K_mat)

            for i in range(len(xs)):
                # Added dropout here to mimic the transformer architecture
                xs[i] = tf.keras.layers.Dropout(dropout_rate)(xs[i])
                xs[i] = tf.keras.layers.LayerNormalization(epsilon=1e-6)(
                    xs[i] + K_mat[:, i, :]
                )
        else:
            # If we concatenate them, then first we embed segment and
            # K_mat to smaller dimensions
            K_mat = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(k_segment_mult_dim, activation=activation_fn),
                name="per_k_projection",
            )(K_mat)

            for i in range(len(xs)):
                # Then concatenate the vectors
                tile_expand = tf.keras.layers.Lambda(
                    lambda x: tf.tile(tf.expand_dims(x[:, i, :], axis=1), [1, max_N, 1])
                )
                concat = tf.keras.layers.Concatenate(axis=-1, name=f"concat_k_seg_{i}")(
                    [xs[i], tile_expand(K_mat)]
                )

                # concat = tf.keras.layers.Concatenate(axis=-1, name=f"concat_k_seg_{i}")(
                #     [
                #         xs[i],
                #         tf.tile(tf.expand_dims(K_mat[:, i, :], axis=1), [1, max_N, 1]),
                #     ]
                # )

                # And re-project to embed_dim dimensions
                xs[i] = tf.keras.layers.Dense(
                    embed_dim, activation=activation_fn, name=f"reproject_k_seg_{i}"
                )(concat)

                # Add dropout here to ensure consistency between labeling
                # techniques
                xs[i] = tf.keras.layers.Dropout(dropout_rate)(xs[i])

        # Cross-attention transformer
        # # NOTE: It may be important to have sequential transformers
        self.crossattn = transformer.TransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            activation_fn=activation_fn,
            skip_mlp=skip_mlp,
            name="cross_attention_1",
        )

        for i in range(len(xs) - 1):
            xs[i] = self.crossattn(xs[i], cross_inputs=xs[i + 1])

        if multiple_cross_attn:
            self.crossattn_2 = transformer.TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                activation_fn=activation_fn,
                skip_mlp=skip_mlp,
                name="cross_attention_2",
            )
        else:
            self.crossattn_2 = self.crossattn

        for dist in range(2, len(xs) - 1):
            for i in range(len(xs) - 1, dist):
                xs[i] = self.crossattn(xs[i], cross_inputs=xs[i - dist])

        # Average pool the embeddings
        x = tf.keras.layers.GlobalAveragePooling1D()(tf.concat(xs, axis=1))

        # And do our traditional divide by 4 dense layer structure
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(64, activation=activation_fn)(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(16, activation=activation_fn)(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(4, activation=activation_fn)(x)

        # Output layer
        output = tf.keras.layers.Dense(1, activation="sigmoid", name="prediction")(x)

        # Create model
        self.model = tf.keras.Model(inputs=inputs, outputs=output)
