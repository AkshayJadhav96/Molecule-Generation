import os
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, QED, Descriptors
import matplotlib.pyplot as plt

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import *
from models.encoder import build_encoder
from models.decoder import build_decoder
from models.vae import MolecularVAE
from utils.data_utils import MolecularDataLoader, preprocess_qm9_dataset

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate molecular structures using VAE')
    parser.add_argument('--model_weights', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of molecules to generate')
    parser.add_argument('--target_prop', type=str, default='qed',
                        help='Property to optimize (qed, gap, homo, lumo)')
    parser.add_argument('--prop_value', type=float, default=0.9,
                        help='Target value for property optimization (0-1)')
    parser.add_argument('--sample_temp', type=float, default=1.0,
                        help='Temperature for sampling (higher for more diversity)')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to store generated outputs')
    parser.add_argument('--optimize_prop', action='store_true',
                        help='Enable property optimization')
    return parser.parse_args()

def evaluate_molecules(molecules, prop_key='qed'):
    """Evaluate properties of generated molecular structures"""
    valid_molecules = [mol for mol in molecules if mol is not None]
    valid_count = len(valid_molecules)
    total_count = len(molecules)
    
    # Compute validity metrics
    valid_ratio = valid_count / total_count if total_count > 0 else 0
    print(f"Produced {valid_count} valid molecules from {total_count} attempts")
    print(f"Valid molecule ratio: {valid_ratio:.4f}")

    if valid_count == 0:
        print("WARNING: No valid molecules generated. Returning empty properties.")
        return valid_molecules, {}
    
    # Calculate molecular properties
    mol_properties = {}
    
    if prop_key == 'qed':
        mol_properties['qed'] = [QED.qed(mol) for mol in valid_molecules]
        print(f"Mean QED score: {np.mean(mol_properties['qed']):.4f}")
    
    # Include additional properties
    mol_properties['logP'] = [Descriptors.MolLogP(mol) for mol in valid_molecules]
    mol_properties['MW'] = [Descriptors.MolWt(mol) for mol in valid_molecules]
    mol_properties['TPSA'] = [Descriptors.TPSA(mol) for mol in valid_molecules]
    mol_properties['HBA'] = [Descriptors.NumHAcceptors(mol) for mol in valid_molecules]
    mol_properties['HBD'] = [Descriptors.NumHDonors(mol) for mol in valid_molecules]
    mol_properties['RotBonds'] = [Descriptors.NumRotatableBonds(mol) for mol in valid_molecules]
    
    # Display property statistics
    for prop_name, values in mol_properties.items():
        if len(values) > 0:
            print(f"{prop_name}: Avg={np.mean(values):.4f}, Min={np.min(values):.4f}, Max={np.max(values):.4f}")
        else:
            print(f"{prop_name}: No values computed (empty list)")
    
    return valid_molecules, mol_properties

def store_molecules(molecules, save_dir, prefix='mol'):
    """Store molecules as images and SMILES strings"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate grid image
    if molecules:
        grid_img = Draw.MolsToGridImage(
            molecules[:min(100, len(molecules))], 
            molsPerRow=5,
            subImgSize=(300, 300),
            legends=[str(i+1) for i in range(min(100, len(molecules)))]
        )
        grid_img.save(os.path.join(save_dir, f"{prefix}_grid.png"))
    
    # Store individual molecules
    smiles_strings = []
    for idx, mol in enumerate(molecules):
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
            smiles_strings.append(smiles)
            if idx < 20:
                img = Draw.MolToImage(mol, size=(300, 300))
                img.save(os.path.join(save_dir, f"{prefix}_{idx+1}.png"))
    
    # Save SMILES data
    smiles_df = pd.DataFrame({'SMILES': smiles_strings})
    smiles_df.to_csv(os.path.join(save_dir, f"{prefix}_smiles.csv"), index=False)
    
    return smiles_strings

def visualize_properties(properties, save_dir, prefix='prop'):
    """Visualize molecular property distributions"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot property histograms
    for prop_name, values in properties.items():
        plt.figure(figsize=(8, 6))
        plt.hist(values, bins=20, alpha=0.8)
        plt.title(f"{prop_name} Distribution")
        plt.xlabel(prop_name)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_{prop_name}_hist.png"))
        plt.close()
    
    # Plot property correlations
    if len(properties) >= 2:
        key_props = ['qed', 'logP', 'MW'] if 'qed' in properties else list(properties.keys())[:3]
        for i, prop1 in enumerate(key_props):
            for prop2 in key_props[i+1:]:
                if prop1 in properties and prop2 in properties:
                    plt.figure(figsize=(8, 6))
                    plt.scatter(properties[prop1], properties[prop2], alpha=0.6)
                    plt.title(f"{prop1} vs {prop2}")
                    plt.xlabel(prop1)
                    plt.ylabel(prop2)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f"{prefix}_{prop1}_vs_{prop2}.png"))
                    plt.close()

def main():
    args = parse_arguments()
    
    # Ensure output directory exists
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Validate model weights file
    if not os.path.exists(args.model_weights):
        print(f"ERROR: Model weights file {args.model_weights} not found!")
        return
        
    try:
        file_size = os.path.getsize(args.model_weights)
        if file_size == 0:
            print(f"ERROR: Model weights file {args.model_weights} is empty!")
            return
        print(f"Model weights size: {file_size/1024/1024:.2f} MB")
    except Exception as e:
        print(f"ERROR: Unable to access model weights: {e}")
        return
    
    # Configure GPU memory
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Enabled memory growth for GPU: {gpu}")
    
    # Construct encoder
    print("Constructing encoder...")
    encoder = build_encoder(
        graph_units=[9],
        edge_shape=(BOND_DIM, NUM_ATOMS, NUM_ATOMS),
        node_shape=(NUM_ATOMS, ATOM_DIM),
        latent_space_dim=LATENT_DIM,
        fully_connected_units=[128],
        dropout_prob=0.2,
    )
    
    # Construct decoder
    print("Constructing decoder...")
    decoder = build_decoder(
        fully_connected_units=[32, 64, 128],
        dropout_prob=0.4,
        latent_space_dim=LATENT_DIM,
        edge_shape=(BOND_DIM, NUM_ATOMS, NUM_ATOMS),
        node_shape=(NUM_ATOMS, ATOM_DIM)
    )
    
    # Debug model architectures
    print("\nEncoder Architecture:")
    encoder.summary()
    
    print("\nDecoder Architecture:")
    decoder.summary()
    
    # Construct VAE
    print("Constructing VAE...")
    vae_model = MolecularVAE(
        encoder, 
        decoder, 
        max_atoms=NUM_ATOMS,
        kl_factor=5e-3,
        recon_factor=1.0,
        prop_factor=1.0,
        grad_factor=0.005
    )
    
    # Build model with dummy input
    dummy_edge = np.zeros((1, BOND_DIM, NUM_ATOMS, NUM_ATOMS))
    dummy_node = np.zeros((1, NUM_ATOMS, ATOM_DIM))
    _ = vae_model([dummy_edge, dummy_node])
    
    # Compile model
    vae_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    
    # Load weights
    print(f"Loading weights from {args.model_weights}...")
    try:
        vae_model.load_weights(args.model_weights)
        print("Weights loaded successfully!")
    except Exception as e:
        print(f"Failed to load weights: {e}")
        print("Trying with skip_mismatch...")
        try:
            vae_model.load_weights(args.model_weights, skip_mismatch=True)
            print("Weights loaded with skip_mismatch")
        except Exception as inner_e:
            print(f"Failed to load weights with skip_mismatch: {inner_e}")
            print("WARNING: Using uninitialized weights")
    
    # Generate molecules
    print(f"Generating {args.num_samples} molecules...")
    if args.optimize_prop:
        print(f"Optimizing for {args.target_prop} = {args.prop_value}")
        molecules, prop_scores = vae_model.optimize_molecules(
            num_samples=args.num_samples,
            target_prop=args.prop_value,
            opt_steps=100,
            step_lr=0.1
        )
        prefix = f"opt_{args.target_prop}_{args.prop_value}"
    else:
        # Debug decoder output
        latent_samples = tf.random.normal((args.num_samples, vae_model.encoder_model.output[0].shape[1]))
        edge_recon, node_recon = vae_model.decoder_model(latent_samples)
        print(f"Decoder output - Edge tensor shape: {edge_recon.shape}, Node features shape: {node_recon.shape}")
        print(f"Edge tensor softmax sum (sample): {tf.reduce_sum(edge_recon[0, :, 0, 0]).numpy():.4f}")
        print(f"Node features softmax sum (sample): {tf.reduce_sum(node_recon[0, 0, :]).numpy():.4f}")
        edge_indices = tf.argmax(edge_recon, axis=1)
        node_indices = tf.argmax(node_recon, axis=2)
        print(f"Edge indices valid: {(edge_indices < BOND_DIM).numpy().all()}")
        print(f"Node indices valid: {(node_indices < ATOM_DIM).numpy().all()}")
        
        molecules = vae_model.sample_molecules(args.num_samples, temp=args.sample_temp)
        none_count = sum(1 for mol in molecules if mol is None)
        print(f"Generated molecules: {len(molecules)}, None count: {none_count}")
        prefix = f"sample_temp_{args.sample_temp}"
    
    # Evaluate molecules
    valid_mols, properties = evaluate_molecules(molecules, prop_key=args.target_prop)
    
    # Store results
    smiles_list = store_molecules(valid_mols, args.save_dir, prefix=prefix)
    
    # Visualize properties
    visualize_properties(properties, args.save_dir, prefix=prefix)
    
    print(f"Outputs saved to {args.save_dir}")
    print(f"Generated {len(valid_mols)} valid molecules with mean {args.target_prop}: {np.mean(properties.get(args.target_prop, [0])):.4f}")

if __name__ == "__main__":
    main()