import numpy as np
import tensorflow as tf


def linear(feature_dim, embed_dim):
    """Create linear positional encoding"""
    positions = np.arange(feature_dim, dtype=np.float32)

    # Broadcast to embedding dimension
    pos_encoding = np.tile(positions.reshape(-1, 1), (1, embed_dim))

    # Normalize to prevent large values
    pos_encoding = pos_encoding / feature_dim

    return tf.constant(pos_encoding, dtype=tf.float32)


def sinusoidal(feature_dim, embed_dim, wavelength: int = 10_000):
    """Create sinusoidal positional encoding"""
    P = np.zeros((feature_dim, embed_dim))
    for k in range(feature_dim):
        for i in np.arange(embed_dim // 2):
            denominator = np.power(wavelength, 2 * i / embed_dim)
            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)
    return tf.constant(P, dtype=tf.float32)
