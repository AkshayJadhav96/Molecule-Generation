import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from VAE.models.layers import GraphRelationConv

def build_encoder(graph_units, latent_space_dim, edge_shape, node_shape, fully_connected_units, dropout_prob):
    """
    Constructs the encoder for the Variational Autoencoder.
    
    Parameters:
    -----------
    graph_units : list
        Number of units for each graph convolution layer
    latent_space_dim : int
        Size of the latent space
    edge_shape : tuple
        Dimensions of the edge tensor (BOND_DIM, NUM_ATOMS, NUM_ATOMS)
    node_shape : tuple
        Dimensions of the node feature matrix (NUM_ATOMS, ATOM_DIM)
    fully_connected_units : list
        Units for dense layers
    dropout_prob : float
        Probability for dropout regularization
    
    Returns:
    --------
    encoder_model : keras.Model
        Encoder mapping molecular graphs to latent distributions
    """
    # Define input tensors
    edge_input = keras.layers.Input(shape=edge_shape)
    node_input = keras.layers.Input(shape=node_shape)

    # Apply graph convolutions
    node_transformed = node_input
    for unit_count in graph_units:
        node_transformed = GraphRelationConv(unit_count)(
            [edge_input, node_transformed]
        )
    
    # Flatten to 1D representation
    flattened = keras.layers.GlobalAveragePooling1D()(node_transformed)

    # Process through dense layers
    dense_output = flattened
    for unit_count in fully_connected_units:
        dense_output = layers.Dense(unit_count, activation="relu")(dense_output)
        dense_output = layers.Dropout(dropout_prob)(dense_output)

    # Generate latent distribution parameters
    latent_mean = layers.Dense(latent_space_dim, dtype="float32", name="z_mean")(dense_output)  # Match original
    latent_log_var = layers.Dense(latent_space_dim, dtype="float32", name="z_log_var")(dense_output)  # Match original
    encoder_model = keras.Model([edge_input, node_input], [latent_mean, latent_log_var], name="encoder")  # Match original
    return encoder_model