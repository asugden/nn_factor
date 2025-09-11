import numpy as np
import tensorflow as tf

from nn_factor.network_tools import default_model, positional_encoding, transformer


class WidthCNNModel(default_model.DefaultModel):
    def __init__(
        self,
        w: int = 20,
        filter_size: int = 3,
        filter_dim: int = 32,
        activation_fn: str = "relu",
        dropout_rate: float = 0.1,
    ):
        inputs = tf.keras.layers.Input(shape=(w,), dtype=np.int32, name="inputs")
        x = tf.keras.layers.Lambda(lambda v: tf.cast(v, tf.float32, name="to_float"))(
            inputs
        )
        x = tf.keras.layers.Lambda(lambda v: tf.expand_dims(v, -1))(x)

        layer_count = w // (filter_size - 1) - 1
        # Remember, traditional Conv layers decrease each dimension by
        # kernel - 1 dimensions on each round

        # Also, relu outperforms gelu on CNNs traditionally
        for i in range(layer_count):
            # Does not decrease the layer dimensions
            # x = layers.Conv1D(
            #     filter_dim if i > 2 or layer_count - i < 2 else filter_dim // 2,
            #     (filter_size),
            #     activation="relu",
            #     padding="same",
            #     name=f"conv_same_{i+1}",
            # )(x)
            x = tf.keras.layers.Conv1D(
                filter_dim if i > 2 or layer_count - i < 2 else filter_dim // 2,
                (filter_size),
                activation="relu",
                name=f"conv_dec_{i+1}",
            )(x)

        # Now we flatten
        x = tf.keras.layers.Flatten()(x)

        # And do our traditional divide by 4 dense layer structure
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(64, activation=activation_fn)(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(16, activation=activation_fn)(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(4, activation=activation_fn)(x)

        # Output layer
        output = tf.keras.layers.Dense(1, activation="linear", name="prediction")(x)

        # Create model
        self.model = tf.keras.Model(inputs=inputs, outputs=output)
