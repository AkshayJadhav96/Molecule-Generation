import ast

# Data parameters
MAX_MOLSIZE = 29  # Maximum number of atoms in QM9 molecules (including hydrogens)
NUM_ATOMS = 29    # Maximum number of atoms for graph representation

# Model hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
VAE_LR = 5e-4
LATENT_DIM = 256  # Size of the latent space, reduced since QM9 has smaller molecules

# Chemistry parameters - QM9 specific (C, N, O, F and H)
SMILE_CHARSET = '["C", "N", "O", "F", "H"]'
SMILE_CHARSET = ast.literal_eval(SMILE_CHARSET)

ATOM_DIM = len(SMILE_CHARSET)  # Number of atom types
BOND_DIM = 4 + 1  # Number of bond types (single, double, triple, aromatic + no bond)

# Mapping dictionaries
bond_mapping = {
    "SINGLE": 0,
    0: 0,  # Using just the integer to avoid RDKit dependency in config
    "DOUBLE": 1,
    1: 1,
    "TRIPLE": 2,
    2: 2,
    "AROMATIC": 3,
    3: 3,
}

# File paths
DATA_PATH = "data/qm9/dsgdb9nsd.xyz"  # Path to QM9 dataset
PROCESSED_DATA_PATH = "data/qm9/processed_qm9.csv"  # Path to processed QM9 data
MODEL_SAVE_PATH = "models/saved/qm9_vae_model"