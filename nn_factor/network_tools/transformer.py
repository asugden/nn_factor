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
        dropout_rate=0.1,
        **kwargs
    ):
        """Initializes transformer block layer

        Args:
            embed_dim (int): embedding vector dimension. The inputs must
                be embedded to allow adding context.
            num_heads (int): number of heads used in multi-head
                attention layer. Can be thought of as unique attention
                "questions"
            ff_dim (int, optional): dimension of first layer of feed-
                forward neural network after applying self-attention.
                Defaults to embed_dim.
            key_dim (int, optional): dimension that keys and queries are
                first projected to in each head of multi-head attention
                layer. Defaults to (embed_dim // num_heads).
            value_dim (int, optional): dimension that values are
                projected to in each head of multi-head attention layer.
                Defaults to (embed_dim // num_heads).
            dropout_rate (float, optional): Dropout learning rate of
                dense layers. Defaults to 0.1.
        """
        super(TransformerBlock, self).__init__(**kwargs)
        if ff_dim is None:
            ff_dim = embed_dim
        if key_dim is None:
            key_dim = embed_dim // num_heads
        if value_dim is None:
            value_dim = embed_dim // num_heads

        # Create attention layer
        self.att = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, value_dim=value_dim
        )

        # Create follow on MLP
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(ff_dim, activation="relu"),
                # Dense layer expands the input to do some computations
                keras.layers.Dense(embed_dim),  # Linear layer that
                # brings the model back to the embedding dimension
            ]
        )

        # Layer norm tries to transform mean to 0 and stdev to 1
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training: bool, cross_inputs=None, attention_mask=None):
        """Call is a function defined by superclass layers.Layer. It is
        run by __call__(). Specifically, this is a self-attention block.

        Args:
            inputs: input tensor
            training (bool): whether in training mode (for dropout)
            cross_inputs: inputs that would be used for cross-attention
            attention_mask (_type_, optional): an attention mask for
                language model training. Defaults to None.
        """
        # Transformer first passes data through attention block and
        # then dropout for training
        if cross_inputs is None:
            cross_inputs = inputs
        attn_output = self.att(inputs, cross_inputs, attention_mask=attention_mask)
        attn_output = self.dropout1(attn_output, training=training)
        # Attention produces "embedding deltas" to add to the input
        # embedding vectors, which are normalized
        out1 = self.layernorm1(inputs + attn_output)

        # Then, we hit the feedforward network/multilayer perceptron
        # which can be thought of as adding "facts" to the "context" of
        # attention (see Google research paper). This, too, is in the
        # form of a delta, which is added.
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out = self.layernorm2(out1 + ffn_output)

        # Attention masking is used for training language models. It is
        # ignored in mathematical models such as ours. I have left it in
        # in case it can be used in later problems.
        if attention_mask is not None:
            out = (
                tf.expand_dims(
                    tf.squeeze(tf.cast(attention_mask, tf.float32), axis=[1, 2]), axis=2
                )
                * out
            )
        return out
