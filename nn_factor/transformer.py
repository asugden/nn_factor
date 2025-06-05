import unittest

import numpy as np
import tensorflow as tf
from tensorflow import keras


class TransformerBlock(keras.layers.Layer):
    """For computing blocks of transformer model"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int = None,
        key_dim: int = None,
        value_dim: int = None,
        rate=0.1,
        **kwargs
    ):
        """Initializes transformer block layer

        Args:
            embed_dim (int): Embedding dimension size of keys (AX embeddings)
            num_heads (int): Number of heads used in multi-head attention layer
            ff_dim (int, optional): Dimension of first layer of feed-forward neural network after applying self-attention. Defaults to size of key embeddings.
            key_dim (int, optional): Dimension that keys and queries are first projected to in each head of multi-head attention layer. Defaults to size of
                key embeddings divided by number of heads.
            value_dim (int, optional): Dimension that values are projected to in each head of multi-head attention layer. Defaults to size of
                key embeddings divided by number of heads.
            rate (float, optional): Dropout learning rate of dense layers. Defaults to 0.1.
        """
        super(TransformerBlock, self).__init__(**kwargs)
        if ff_dim is None:
            ff_dim = embed_dim
        if key_dim is None:
            key_dim = embed_dim // num_heads
        if value_dim is None:
            value_dim = embed_dim // num_heads
        self.att = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, value_dim=value_dim
        )
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(ff_dim, activation="relu"),
                keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training, attention_mask=None):
        attn_output = self.att(inputs, inputs, attention_mask=attention_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out = self.layernorm2(out1 + ffn_output)
        if attention_mask is not None:
            out = (
                tf.expand_dims(
                    tf.squeeze(tf.cast(attention_mask, tf.float32), axis=[1, 2]), axis=2
                )
                * out
            )
        return out


class TestTransformerBlock(unittest.TestCase):
    def test_attention_mask(self):
        """Check attention masking"""
        x = tf.random.uniform([1, 10, 48])
        x_6 = tf.concat(
            [tf.expand_dims(x[:, i, :], axis=1) for i in [1, 2, 4, 6, 8]], axis=1
        )
        tfblock = TransformerBlock(48, 4)
        y_6 = tfblock(tfblock(x_6)).numpy().sum(axis=2)
        mask = tf.constant(
            [False, True, True, False, True, False, True, False, True, False],
            dtype=tf.bool,
        )[tf.newaxis, tf.newaxis, tf.newaxis, :]
        y = (
            tfblock(tfblock(x, attention_mask=mask), attention_mask=mask)
            .numpy()
            .sum(axis=2)
        )
        y_full = tfblock(tfblock(x)).numpy().sum(axis=2)
        equals_without_keywords = np.all(y_6[y_6 != 0] == y[y != 0])
        equals_without_attention = False
        if len(y[y != 0]) == len(y_full[y_full != 0]):
            equals_without_attention = np.all(y[y != 0] == y_full[y_full != 0])
        self.assertTrue(equals_without_keywords and not equals_without_attention)


if __name__ == "__main__":
    unittest.main()
