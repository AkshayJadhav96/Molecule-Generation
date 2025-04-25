import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class GraphRelationConv(keras.layers.Layer):
    """
    Graph Convolutional Layer for Relational Networks.
    Facilitates message passing across molecular graph edges.
    """
    def __init__(
        self,
        output_units=128,
        activation_fn="relu",
        include_bias=False,
        weight_initializer="glorot_uniform",
        bias_initializer="zeros",
        weight_regularizer=None,
        bias_regularizer=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.output_units = output_units
        self.activation_fn = keras.activations.get(activation_fn)
        self.include_bias = include_bias
        self.weight_initializer = keras.initializers.get(weight_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.weight_regularizer = keras.regularizers.get(weight_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

    def build(self, input_shapes):
        edge_dim = input_shapes[0][1]
        node_dim = input_shapes[1][2]

        self.weight_matrix = self.add_weight(
            shape=(edge_dim, node_dim, self.output_units),
            initializer=self.weight_initializer,
            regularizer=self.weight_regularizer,
            trainable=True,
            name="weight",
            dtype=tf.float32,
        )

        if self.include_bias:
            self.bias_term = self.add_weight(
                shape=(edge_dim, 1, self.output_units),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                name="bias",
                dtype=tf.float32,
            )

        self.built = True

    def call(self, inputs, training=False):
        edge_tensor, node_features = inputs
        # Gather neighbor information
        aggregated = tf.matmul(edge_tensor, node_features[:, None, :, :])
        # Transform with weights
        transformed = tf.matmul(aggregated, self.weight_matrix)
        if self.include_bias:
            transformed += self.bias_term
        # Sum over edge types
        reduced = tf.reduce_sum(transformed, axis=1)
        # Apply activation
        return self.activation_fn(reduced)

class LatentSampler(layers.Layer):
    """
    Sampling layer for Variational Autoencoder.
    Generates latent vectors from mean and log-variance inputs.
    """
    def call(self, inputs):
        mean, log_variance = inputs
        batch_size = tf.shape(log_variance)[0]
        latent_dim = tf.shape(log_variance)[1]
        noise = tf.keras.backend.random_normal(shape=(batch_size, latent_dim))
        return mean + tf.exp(0.5 * log_variance) * noise