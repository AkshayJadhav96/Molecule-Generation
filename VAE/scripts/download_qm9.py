import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED
from tqdm import tqdm
from pathlib import Path

def preprocess_qm9():
    """Process QM9 dataset to extract SMILES and properties"""
    print("Processing QM9 dataset...")
    
    # Load the existing QM9 dataset
    qm9_file = "VAE/data/qm9/qm9.csv"
    
    if not os.path.exists(qm9_file):
        raise FileNotFoundError(f"QM9 dataset not found at {qm9_file}. Please check the file path.")
    
    # Read the CSV file
    raw_df = pd.read_csv(qm9_file)
    print(f"Loaded QM9 dataset with {len(raw_df)} molecules")
    
    # Process molecules and extract properties
    data = []
    for _, row in tqdm(raw_df.iterrows(), total=len(raw_df), desc="Processing molecules"):
        try:
            # Assuming the CSV has a 'smiles' column
            smiles = row['SMILES1']
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is not None:
                # Standardize SMILES representation
                canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                
                # Calculate QED (drug-likeness)
                qed_value = QED.qed(mol)
                
                # Extract other properties if they exist in your CSV
                # Adjust these based on your actual CSV columns
                properties = {
                    'smiles': canonical_smiles,
                    'qed': qed_value
                }
                
                # Add existing properties from the CSV if available
                for prop in ['homo', 'lumo', 'gap']:
                    if prop in row:
                        properties[prop] = row[prop]
                
                data.append(properties)
        except Exception as e:
            print(f"Error processing molecule: {e}")
    
    # Create DataFrame with processed data
    processed_df = pd.DataFrame(data)
    
    # Save processed data
    output_file = "VAE/data/qm9/processed_qm9.csv"
    processed_df.to_csv(output_file, index=False)
    
    print(f"Processed {len(processed_df)} molecules. Data saved to {output_file}")
    return processed_df

if __name__ == "__main__":
    df = preprocess_qm9()
    print("QM9 dataset preprocessing complete!")
    print(f"Dataset shape: {df.shape}")
    print(df.head())