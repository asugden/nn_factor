import tensorflow as tf
from tensorflow import keras


class TransformerBlock(keras.layers.Layer):
    """For computing blocks of transformer model"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        key_dim: int = None,
        dropout_rate: float = 0.1,
        activation_fn: str = "relu",
        skip_mlp: bool = False,
        **kwargs
    ):
        """Initializes transformer block layer

        Args:
            embed_dim (int): embedding vector dimension. The inputs must
                be embedded to allow adding context.
            num_heads (int): number of heads used in multi-head
                attention layer. Can be thought of as unique attention
                "questions"
            key_dim (int, optional): dimension that keys and queries are
                first projected to in each head of multi-head attention
                layer. Defaults to (embed_dim // num_heads).
            dropout_rate (float, optional): Dropout learning rate of
                dense layers. Defaults to 0.1.
            activation_fn (str, optional): Set the activation function
                of the ff layers. Defaults to 'relu'.
        """
        super(TransformerBlock, self).__init__(**kwargs)

        # Save inputs for loading model
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.key_dim = self.embed_dim // 2 if key_dim is None else key_dim
        self.ff_dim = 4 * self.key_dim  # Size of feedforward expansion layer
        self.dropout_rate = dropout_rate
        self.activation_fn = activation_fn
        self.skip_mlp = skip_mlp

        # Create attention layer
        # In the first case, the data is of shape (18, 64)
        self.att = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.key_dim
        )

        # Layer norm tries to transform mean to 0 and stdev to 1
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(self.dropout_rate)

        if not skip_mlp:
            # Create follow on feedforward network
            self.ffn = keras.Sequential(
                [
                    keras.layers.Dense(self.ff_dim, activation=self.activation_fn),
                    # Dense layer expands the input to do some computations
                    keras.layers.Dense(self.embed_dim),  # Linear layer that
                    # brings the model back to the embedding dimension
                ]
            )

            self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
            self.dropout2 = keras.layers.Dropout(self.dropout_rate)

    def call(
        self, inputs, cross_inputs=None, attention_mask=None, training: bool = None
    ):
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
        if not self.skip_mlp:
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            out = self.layernorm2(out1 + ffn_output)
        else:
            out = out1

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

    def get_config(self):
        """A necessary function for saving custom layers."""
        config = super(TransformerBlock, self).get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "ff_dim": self.ff_dim,
                "dropout_rate": self.dropout_rate,
                "activation_fn": self.activation_fn,
                "skip_mlp": self.skip_mlp,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """A necessary function for loading custom layers"""
        return cls(**config)
