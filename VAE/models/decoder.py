import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_decoder(fully_connected_units, dropout_prob, latent_space_dim, edge_shape, node_shape):
    """
    Constructs the decoder for the Variational Autoencoder.
    
    Parameters:
    -----------
    fully_connected_units : list
        Units for dense layers
    dropout_prob : float
        Probability for dropout regularization
    latent_space_dim : int
        Size of the latent space
    edge_shape : tuple
        Dimensions of the edge tensor (BOND_DIM, NUM_ATOMS, NUM_ATOMS)
    node_shape : tuple
        Dimensions of the node feature matrix (NUM_ATOMS, ATOM_DIM)
    
    Returns:
    --------
    decoder_model : keras.Model
        Decoder mapping latent vectors to molecular graph representations
    """
    # Define latent input
    latent_input = keras.Input(shape=(latent_space_dim,))
    
    # Apply dense layers
    decoded = latent_input
    for unit_count in fully_connected_units:
        decoded = layers.Dense(unit_count, activation="tanh")(decoded)
        decoded = layers.Dropout(dropout_prob)(decoded)
    
    # Compute output tensor sizes
    edge_elements = int(edge_shape[0] * edge_shape[1] * edge_shape[2])
    node_elements = int(node_shape[0] * node_shape[1])
    
    # Generate edge tensor
    edge_output = layers.Dense(edge_elements)(decoded)
    edge_output = layers.Reshape(edge_shape)(edge_output)
    
    # Symmetrize edge tensor
    def make_symmetric(tensor):
        transposed = keras.backend.permute_dimensions(tensor, (0, 1, 3, 2))
        return (tensor + transposed) / 2
    
    edge_output = layers.Lambda(make_symmetric)(edge_output)
    edge_output = layers.Softmax(axis=1)(edge_output)
    
    # Generate node feature tensor
    node_output = layers.Dense(node_elements)(decoded)
    node_output = layers.Reshape(node_shape)(node_output)
    node_output = layers.Softmax(axis=2)(node_output)
    
    # Construct decoder
    decoder_model = keras.Model(latent_input, outputs=[edge_output, node_output], name="decoder")
    
    return decoder_model