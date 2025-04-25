from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import numpy as np
import pandas as pd

def compute_qm9_metrics(qm9_file='qm9.csv', max_molecules=10000):
    """Compute mean LogP, MW, QED for QM9 dataset"""
    try:
        # Load QM9 (assumes CSV with 'smiles' column)
        df = pd.read_csv(qm9_file)
        if max_molecules:
            df = df.sample(n=min(max_molecules, len(df)), random_state=42)
        
        logp_values = []
        mw_values = []
        qed_values = []
        
        for smiles in df['SMILES1']:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                logp_values.append(Descriptors.MolLogP(mol))
                mw_values.append(Descriptors.MolWt(mol))
                qed_values.append(QED.qed(mol))
            except Exception as e:
                print(f"Error processing SMILES {smiles}: {e}")
                continue
        
        return {
            "qm9_mean_logp": round(np.mean(logp_values), 2) if logp_values else 0.0,
            "qm9_mean_mw": round(np.mean(mw_values), 2) if mw_values else 0.0,
            "qm9_mean_qed": round(np.mean(qed_values), 2) if qed_values else 0.0,
            "count": len(logp_values)
        }
    except Exception as e:
        print(f"Error computing QM9 metrics: {e}")
        return None

if __name__ == "__main__":
    # Replace with path to your QM9 dataset
    metrics = compute_qm9_metrics(qm9_file='/home/u142201016/142201016/Molecule_Generation/qm9.csv')
    if metrics:
        print(metrics)