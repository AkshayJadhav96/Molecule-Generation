import numpy as np
import pandas as pd
import tensorflow as tf
from rdkit import Chem
import tensorflow.keras as keras
from VAE.config import NUM_ATOMS, BOND_DIM, ATOM_DIM, SMILE_CHARSET, bond_mapping

# Define mappings for atom symbols
char_to_idx = dict((char, idx) for idx, char in enumerate(SMILE_CHARSET))
idx_to_char = dict((idx, char) for idx, char in enumerate(SMILE_CHARSET))
symbol_mapping = dict(char_to_idx)
symbol_mapping.update(idx_to_char)

def smiles_to_molecular_graph(smiles_string):
    """
    Transform a SMILES string into a molecular graph representation.
    
    Parameters:
    -----------
    smiles_string : str
        SMILES representation of a molecule
    
    Returns:
    --------
    edge_tensor : numpy.ndarray
        Edge tensor of shape (BOND_DIM, NUM_ATOMS, NUM_ATOMS)
    node_features : numpy.ndarray
        Node feature matrix of shape (NUM_ATOMS, ATOM_DIM)
    """
    # Parse SMILES into a molecule object
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        # Return zero tensors for invalid SMILES
        edge_tensor = np.zeros((BOND_DIM, NUM_ATOMS, NUM_ATOMS), dtype="float32")
        node_features = np.zeros((NUM_ATOMS, ATOM_DIM), dtype="float32")
        # Indicate "no bond" and "no atom" in respective channels
        edge_tensor[-1, :, :] = 1
        node_features[:, -1] = 1
        return edge_tensor, node_features

    # Initialize tensors for edges and nodes
    edge_tensor = np.zeros((BOND_DIM, NUM_ATOMS, NUM_ATOMS), dtype="float32")
    node_features = np.zeros((NUM_ATOMS, ATOM_DIM), dtype="float32")

    # Process each atom
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        if atom_idx >= NUM_ATOMS:  # Ignore atoms beyond maximum limit
            continue
            
        atom_symbol = atom.GetSymbol()
        if atom_symbol in symbol_mapping:
            atom_type = symbol_mapping[atom_symbol]
            node_features[atom_idx] = np.eye(ATOM_DIM)[atom_type]
            
            # Process neighboring atoms
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx >= NUM_ATOMS:  # Skip out-of-range neighbors
                    continue
                    
                bond = mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)
                bond_type = bond.GetBondType().name
                if bond_type in bond_mapping:
                    bond_idx = bond_mapping[bond_type]
                    edge_tensor[bond_idx, [atom_idx, neighbor_idx], [neighbor_idx, atom_idx]] = 1

    # Mark non-bonded positions
    edge_tensor[-1, np.sum(edge_tensor, axis=0) == 0] = 1

    # Mark non-atom positions
    node_features[np.where(np.sum(node_features, axis=1) == 0)[0], -1] = 1

    return edge_tensor, node_features

def molecular_graph_to_molecule(graph_data):
    """
    Convert a molecular graph back to an RDKit molecule object.
    
    Parameters:
    -----------
    graph_data : tuple
        Contains (edge_tensor, node_features) arrays
    
    Returns:
    --------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule object, or None if conversion is invalid
    """
    edge_tensor, node_features = graph_data
    mol = Chem.RWMol()

    # Check for valid atoms and bonds
    valid_indices = np.where(
        (np.argmax(node_features, axis=1) != ATOM_DIM - 1)
        & (np.sum(edge_tensor[:-1], axis=(0, 1)) != 0)
    )[0]
    if len(valid_indices) == 0:
        return None
        
    node_features = node_features[valid_indices]
    edge_tensor = edge_tensor[:, valid_indices, :][:, :, valid_indices]

    # Add atoms
    for atom_type_idx in np.argmax(node_features, axis=1):
        if atom_type_idx < len(SMILE_CHARSET):
            atom_symbol = symbol_mapping[atom_type_idx]
            mol.AddAtom(Chem.Atom(atom_symbol))
        # Skip invalid atom indices (match original behavior)

    # Add bonds
    (bond_indices, atom_i, atom_j) = np.where(np.triu(edge_tensor) == 1)
    for (bond_idx, i, j) in zip(bond_indices, atom_i, atom_j):
        if i == j or bond_idx == BOND_DIM - 1:
            continue
        if bond_idx in bond_mapping:
            if bond_idx == 0:
                bond_type = Chem.BondType.SINGLE
            elif bond_idx == 1:
                bond_type = Chem.BondType.DOUBLE
            elif bond_idx == 2:
                bond_type = Chem.BondType.TRIPLE
            elif bond_idx == 3:
                bond_type = Chem.BondType.AROMATIC
            else:
                continue  # Skip invalid bond indices (match original behavior)
            mol.AddBond(int(i), int(j), bond_type)
        # Skip invalid bond indices (match original behavior)

    try:
        Chem.SanitizeMol(mol)
        return mol
    except:
        return None

        
def generate_tf_dataset(dataframe, prop_column='qed', max_atoms=NUM_ATOMS, batch_size=32, shuffle_data=True):
    """
    Create a TensorFlow dataset from molecular data for training.
    
    Parameters:
    -----------
    dataframe : pd.DataFrame
        DataFrame with molecular data
    prop_column : str
        Column name for the target property
    max_atoms : int
        Maximum number of atoms per molecule
    batch_size : int
        Size of each batch
    shuffle_data : bool
        Whether to shuffle the dataset
    
    Returns:
    --------
    tf_dataset : tf.data.Dataset
        TensorFlow dataset for model training
    """
    def data_generator():
        indices = dataframe.index.tolist()
        if shuffle_data:
            np.random.shuffle(indices)
            
        for idx in indices:
            prop_value = dataframe.loc[idx][prop_column]
            edge_tensor, node_features = smiles_to_molecular_graph(dataframe.loc[idx]['smiles'])
            yield (edge_tensor, node_features), prop_value
    
    # Specify tensor shapes and types
    output_spec = (
        (
            tf.TensorSpec(shape=(BOND_DIM, max_atoms, max_atoms), dtype=tf.float32),
            tf.TensorSpec(shape=(max_atoms, ATOM_DIM), dtype=tf.float32)
        ),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )
    
    # Build dataset
    tf_dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=output_spec
    )
    
    # Apply batching and optimization
    tf_dataset = tf_dataset.batch(batch_size)
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
    
    return tf_dataset

class MolecularDataLoader(keras.utils.Sequence):
    """
    Batch data loader for molecular graph generation during training.
    """
    def __init__(self, dataframe, prop_column='qed', max_atoms=NUM_ATOMS, batch_size=32, shuffle_data=True, **kwargs):
        """
        Initialize the molecular data loader.
        """
        super().__init__(**kwargs)
        self.dataframe = dataframe
        self.indices = self.dataframe.index.tolist()
        self.prop_column = prop_column
        self.max_atoms = max_atoms
        self.batch_size = batch_size
        self.shuffle_data = shuffle_data
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        batch_end = (index + 1) * self.batch_size
        if batch_end > len(self.indices):
            current_batch_size = len(self.indices) - index * self.batch_size
        else:
            current_batch_size = self.batch_size
            
        batch_indices = self.indices[index * self.batch_size : batch_end]
        batch_features, batch_props = self.generate_batch(batch_indices, current_batch_size)
        return batch_features, batch_props

    def on_epoch_end(self):
        """Refresh indices after each epoch"""
        self.index = np.arange(len(self.indices))
        if self.shuffle_data:
            np.random.shuffle(self.index)

    def fetch_molecule(self, idx):
        """
        Retrieve molecular graph and property value from SMILES.
        """
        prop_value = self.dataframe.loc[idx][self.prop_column]
        edge_tensor, node_features = smiles_to_molecular_graph(self.dataframe.loc[idx]['smiles'])
        return edge_tensor, node_features, prop_value

    def generate_batch(self, batch_indices, batch_size):
        """Create a batch of molecular data"""
        edge_batch = np.empty((batch_size, BOND_DIM, self.max_atoms, self.max_atoms))
        node_batch = np.empty((batch_size, self.max_atoms, ATOM_DIM))
        prop_batch = np.empty((batch_size,))

        for i, idx in enumerate(batch_indices):
            edge_batch[i], node_batch[i], prop_batch[i] = self.fetch_molecule(idx)

        return [edge_batch, node_batch], prop_batch

def preprocess_qm9_dataset(file_path, target_prop='qed', test_split=0.25):
    """
    Load and preprocess QM9 dataset, splitting into training and test sets.
    
    Parameters:
    -----------
    file_path : str
        Path to QM9 CSV file
    target_prop : str
        Target property to predict ('qed', 'gap', 'homo', 'lumo')
    test_split : float
        Fraction of data for test set
    
    Returns:
    --------
    train_data : pandas.DataFrame
        Training dataset
    test_data : pandas.DataFrame
        Test dataset
    """
    # Read dataset
    data = pd.read_csv(file_path)
    
    # Filter valid SMILES
    valid_smiles = []
    for smile in data['smiles']:
        mol = Chem.MolFromSmiles(smile)
        if mol is not None and smile != '':
            valid_smiles.append(smile)
    
    data = data[data['smiles'].isin(valid_smiles)]
    
    # Scale target property
    if target_prop in data.columns:
        min_val = data[target_prop].min()
        max_val = data[target_prop].max()
        data[target_prop] = (data[target_prop] - min_val) / (max_val - min_val)
    
    # Split dataset
    train_data = data.sample(frac=1-test_split, random_state=42)
    test_data = data.drop(train_data.index)
    
    # Reset indices
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    
    print(f"Training set: {len(train_data)} molecules")
    print(f"Test set: {len(test_data)} molecules")
    
    return train_data, test_data