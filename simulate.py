import argparse
import os
import sys
# Ensure the parent directory is in the Python path to find the 'models' module
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import MolToSmiles, Atom, RWMol, BondType, Draw, AllChem
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from collections import deque
import time # For adding slight pauses if needed

# Assuming these modules exist in the parent directory as per sys.path insert
try:
    from models.hyperparams import Hyperparameters
    from models.model import MoFlow, rescale_adj
except ImportError:
    print("Error: Could not import MoFlow model components.")
    print(f"Looked for 'models' directory in: {parent_dir}")
    print("Make sure 'models/hyperparams.py' and 'models/model.py' exist relative to the script.")
    sys.exit(1)

# --- Model Loading (Robust version) ---
def load_model(snapshot_path, hyperparams_path, device='cpu', multigpu=False):
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")
    if not os.path.exists(hyperparams_path):
        raise FileNotFoundError(f"Hyperparameters file not found: {hyperparams_path}")

    print(f"Loading hyperparameters from {hyperparams_path}", flush=True)
    # Pass map_location='cpu' first to load onto CPU, then move to target device
    # This avoids potential issues if the model was saved on a different GPU setup
    model_params = Hyperparameters(path=hyperparams_path)
    model = MoFlow(model_params)

    print(f"Loading model state from {snapshot_path}", flush=True)
    state_dict = torch.load(snapshot_path, map_location='cpu') # Load to CPU first

    # Handle potential 'module.' prefix from DataParallel saving
    is_DataParallel = any(k.startswith('module.') for k in state_dict.keys())
    if is_DataParallel:
        print("Detected model saved with DataParallel. Removing 'module.' prefix.")
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    # Determine target device
    if isinstance(device, str):
        target_device = torch.device(device if torch.cuda.is_available() and device != 'cpu' else 'cpu')
    else: # Assume device is already a torch.device object
        target_device = device

    # Move model to target device
    model.to(target_device)
    model.eval() # Set to evaluation mode

    # Handle DataParallel wrapping if requested *after* loading state_dict
    # This part is tricky. Generally, you load the state dict *without* the prefix
    # and then wrap in DataParallel if needed.
    if multigpu and target_device != torch.device('cpu') and torch.cuda.device_count() > 1:
        print(f"Wrapping model in DataParallel for multi-GPU use on {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model) # Wrap the already loaded model
    elif multigpu:
         print("Warning: Multi-GPU requested but not available or target is CPU.")


    print(f"Model loaded successfully on {target_device}", flush=True)
    return model

# --- Molecule Generation (Corrected attribute access) ---
def generate_molecule_data(model, temp=0.7, seed=123, max_attempts=30):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    model.eval()
    # Use the device the model is actually on
    device = next(model.parameters()).device

    # Handle potential DataParallel wrapping when accessing attributes
    model_module = model.module if isinstance(model, torch.nn.DataParallel) else model

    z_dim = model_module.b_size + model_module.a_size
    num_atom_types_model = model_module.a_n_type # Correct attribute
    num_bond_types_model = model_module.b_n_type # Correct attribute

    # Atom mapping (adjust based on your dataset/model)
    default_atom_map = {0: 6, 1: 7, 2: 8, 3: 9} # QM9 default C, N, O, F
    atom_map = default_atom_map # Start with default
    print(f"Using default QM9 atom map initially: {atom_map}")
    # Example override from hyperparams (adjust key 'atom_types' if necessary)
    if hasattr(model_module.hyper_params, 'atom_types'):
        try:
            atom_syms = model_module.hyper_params.atom_types
            if isinstance(atom_syms, list) and all(isinstance(s, str) for s in atom_syms):
                atomic_num_list = [Chem.GetPeriodicTable().GetAtomicNumber(sym.capitalize()) for sym in atom_syms]
                atom_map = {i: num for i, num in enumerate(atomic_num_list)}
                print(f"Overriding atom map from hyperparameters: {atom_map}")
        except Exception as e:
            print(f"Warning: Could not parse atom_types from hyperparameters ({e}). Using default map.")

    # Add Hydrogen if model predicts more types than explicitly mapped
    if num_atom_types_model > len(atom_map):
        h_idx = len(atom_map) # Next available index
        if h_idx < num_atom_types_model: # Ensure we don't exceed model's output dimension
             atom_map[h_idx] = 1 # Atomic Number 1 for Hydrogen
             print(f"Adding Hydrogen (H) mapping at index {h_idx} as model predicts {num_atom_types_model} types.")
        else:
             print(f"Warning: Model predicts {num_atom_types_model} atom types, but map already has {len(atom_map)}. Not adding H automatically.")


    print(f"Final atom map being used: {atom_map}")
    print(f"Model expects {num_atom_types_model} atom types and {num_bond_types_model} bond types.")

    # Map RDKit bond types (CRITICAL: Check convention used by *your* MoFlow model)
    # Common MoFlow convention: 0=None, 1=Single, 2=Double, 3=Triple. Aromatic might be implicit or a 4th type.
    # RDKit: SINGLE, DOUBLE, TRIPLE, AROMATIC
    # **ADJUST THIS MAP BASED ON YOUR MODEL'S OUTPUT DEFINITION**
    model_bond_type_map = {
        1: BondType.SINGLE,
        2: BondType.DOUBLE,
        3: BondType.TRIPLE,
        # 4: BondType.AROMATIC, # Uncomment or add if your model explicitly predicts aromatic bonds as type 4
    }
    # Assuming bond type 0 is 'no bond'
    NO_BOND_INDEX = 0
    print(f"Using bond type map (Model Index -> RDKit Type): {model_bond_type_map}")
    print(f"Assuming Model Index {NO_BOND_INDEX} means 'no bond'.")


    for attempt in range(max_attempts):
        with torch.no_grad():
            # Sample latent vector z
            mu = np.zeros(z_dim)
            sigma_diag = np.ones(z_dim)
            if model_module.hyper_params.learn_dist:
                 if hasattr(model_module, 'ln_var') and model_module.ln_var is not None:
                    ln_var_cpu = model_module.ln_var.data.cpu().numpy()
                    sigma_diag = np.sqrt(np.exp(ln_var_cpu)) # Assuming ln_var matches z_dim directly
                    # Adjust logic here if ln_var has different structure (e.g., separate for atoms/bonds)
            sigma = temp * sigma_diag
            z = torch.from_numpy(np.random.normal(mu, sigma, (1, z_dim))).float().to(device)

            # Decode z using the model (potentially wrapped)
            # true_adj=None tells the reverse method to generate adj from z_adj part of z
            # Handle potential DataParallel wrapping
            model_module = model.module if isinstance(model, torch.nn.DataParallel) else model

            # Call the dedicated reverse method (assuming it exists and takes z)
            # Pass true_adj=None if the model should generate adj from z
            adj_logits, x_logits = model_module.reverse(z, true_adj=None) # <--- CORRECTED LINE

            # Post-process adjacency matrix
            adj_probs = torch.softmax(adj_logits, dim=1) # Softmax over bond types. Shape (1, num_bond_types, N, N)
            # Determine predicted bond type index (highest probability)
            pred_bond_type_idx = torch.argmax(adj_probs, dim=1) # Shape (1, N, N)
            # Get the probability of the predicted type
            pred_bond_prob = torch.max(adj_probs, dim=1).values # Shape (1, N, N)

            # Define criteria for accepting a bond:
            # 1. Predicted type is not the 'no bond' type.
            # 2. Probability of the predicted type exceeds a threshold (e.g., 0.5).
            # Need to unsqueeze pred_bond_type_idx to compare with NO_BOND_INDEX
            bond_exists = (pred_bond_type_idx != NO_BOND_INDEX) & (pred_bond_prob > 0.5) # Shape (1, N, N)

            # Post-process atom features
            x_probs = torch.softmax(x_logits, dim=2) # Softmax over atom types. Shape (1, N, num_atom_types)
            pred_atom_type_idx = torch.argmax(x_probs[0], dim=1).cpu().numpy() # Indices for our atom_map (Shape N)


            # Build RDKit molecule
            mol = RWMol()
            mol_atom_indices = {} # Map model node index -> RDKit atom index
            actual_atoms = [] # Store {'model_idx': i, 'rdkit_idx': rdkit_idx, 'atomic_num': num}

            num_nodes = model_module.a_n_node # Number of nodes model works with

            # Add atoms to RDKit Mol object
            for i in range(num_nodes):
                type_idx = pred_atom_type_idx[i]
                if type_idx in atom_map:
                    atomic_num = atom_map[type_idx]
                    # Crucial: Check if atomic_num is 0 (placeholder/padding)
                    # Modify this check if your 'no atom' type has a different index/atomic_num
                    if atomic_num != 0:
                         atom = Atom(atomic_num)
                         rdkit_idx = mol.AddAtom(atom)
                         mol_atom_indices[i] = rdkit_idx # Store mapping
                         actual_atoms.append({'model_idx': i, 'rdkit_idx': rdkit_idx, 'atomic_num': atomic_num})
                # else: Silently ignore nodes with predicted types not in map or if atomic_num is 0

            # Add bonds to RDKit Mol object
            bonds_added = [] # Store {'u': rdkit_idx1, 'v': rdkit_idx2, 'type': rdkit_bond_type, ...}
            adj_bond_idx_np = pred_bond_type_idx[0].cpu().numpy() # Shape (N, N)
            bond_exists_np = bond_exists[0].cpu().numpy() # Shape (N, N)

            for i in range(num_nodes):
                for j in range(i + 1, num_nodes): # Avoid self-loops and double counting
                    # Check if a bond is predicted *and* if both nodes correspond to actual atoms
                    if bond_exists_np[i, j] and i in mol_atom_indices and j in mol_atom_indices:
                        model_bond_idx = adj_bond_idx_np[i, j]
                        if model_bond_idx in model_bond_type_map:
                            rdkit_bond_type = model_bond_type_map[model_bond_idx]
                            idx1_rdkit = mol_atom_indices[i]
                            idx2_rdkit = mol_atom_indices[j]
                            mol.AddBond(idx1_rdkit, idx2_rdkit, rdkit_bond_type)
                            bonds_added.append({
                                'u': idx1_rdkit, 'v': idx2_rdkit, 'type': rdkit_bond_type,
                                'model_u': i, 'model_v': j, 'model_type_idx': model_bond_idx
                            })

            # --- Molecule Validation and Sanitization ---
            if mol.GetNumAtoms() == 0:
                print(f"Attempt {attempt + 1}: No valid atoms generated. Skipping.")
                continue # Try next attempt

            try:
                # Use a copy for sanitization to avoid modifying original if it fails partially
                mol_copy = RWMol(mol)
                Chem.SanitizeMol(mol_copy) # Check valency, aromaticity etc.

                # Check connectivity (only allow single fragment molecules)
                smiles = MolToSmiles(mol_copy)
                if '.' not in smiles:
                    print(f"Generated connected SMILES: {smiles} (Attempt {attempt + 1})", flush=True)
                    # Optional: Add Hydrogens if desired for final structure
                    # mol_final = Chem.AddHs(mol_copy)
                    # smiles = MolToSmiles(mol_final)
                    # return mol_final, smiles, z, actual_atoms, bonds_added, atom_map
                    return mol_copy, smiles, z, actual_atoms, bonds_added, atom_map # Return sanitized mol
                else:
                    print(f"Disconnected SMILES: {smiles} (Attempt {attempt + 1})", flush=True)
            except ValueError as ve: # Catch specific sanitization errors like kekulization
                 print(f"RDKit sanitization ValueError on attempt {attempt + 1}: {ve}", flush=True)
            except Exception as e: # Catch any other rdkit error
                print(f"RDKit processing error on attempt {attempt + 1}: {e}", flush=True)
                # Optionally draw the failed molecule for debugging:
                # try: Draw.MolToFile(mol, f'failed_mol_attempt_{attempt+1}.png') except: pass

    # --- Fallback if max_attempts reached ---
    print(f"Could not generate a connected, valid molecule after {max_attempts} attempts.", flush=True)
    # Try to return the last generated mol, even if disconnected/failed, for potential inspection
    try:
        # Basic check if mol has atoms before trying SMILES
        if mol.GetNumAtoms() > 0:
             smiles = MolToSmiles(mol, isomericSmiles=False, canonical=True) # Try basic SMILES
        else:
             smiles = "N/A (No atoms)"
    except:
        smiles = "Invalid Structure (SMILES failed)"
    print(f"Returning last attempt result (SMILES: {smiles})", flush=True)
    # Return the *unsanitized* last attempt - might be useful for debugging
    return mol, smiles, z, actual_atoms, bonds_added, atom_map


# --- Enhanced Simulation/Visualization ---
def simulate_molecule_creation(model, output_dir="simulations", temp=0.7, seed=123):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Generating molecule data...")
    mol, smiles, z, actual_atoms, bonds_added, atom_map = generate_molecule_data(model, temp=temp, seed=seed)

    if mol is None or mol.GetNumAtoms() == 0:
        print("Failed to generate a molecule structure with atoms. Cannot create animation.")
        return
    if "Invalid Structure" in smiles or "N/A" in smiles:
         print(f"Generated molecule seems invalid (SMILES: {smiles}). Animation might look incorrect.")
         # Decide if you want to proceed or stop
         # return # Uncomment to stop if generation failed badly

    print(f"Preparing animation for SMILES: {smiles} ({mol.GetNumAtoms()} atoms, {mol.GetNumBonds()} bonds)")

    # --- Prepare data for animation ---
    final_positions = None
    initial_positions = None
    try:
        # Generate 2D coordinates for the final structure
        mol_for_coords = RWMol(mol) # Work on a copy
        AllChem.Compute2DCoords(mol_for_coords)
        # Optional: Add Hs for better geometry, but remove later if not needed for viz
        # mol_for_coords = Chem.AddHs(mol_for_coords)
        # AllChem.Compute2DCoords(mol_for_coords)
        conf = mol_for_coords.GetConformer()
        final_positions = np.array([conf.GetAtomPosition(i) for i in range(mol_for_coords.GetNumAtoms())])[:, :2]

        # Create initial positions (e.g., slightly perturbed from final)
        perturbation_scale = max(final_positions.std(axis=0).mean() * 0.5, 0.5) # Scale perturbation based on molecule size
        initial_positions = final_positions + np.random.normal(0, perturbation_scale, size=final_positions.shape)

    except Exception as e:
        print(f"Warning: Compute2DCoords failed ({e}). Animation will use random layout.")
        num_atoms_gen = len(actual_atoms) # Use number of atoms we intended to add
        if num_atoms_gen > 0:
             final_positions = np.random.rand(num_atoms_gen, 2) * 5 # Scale random positions
             initial_positions = final_positions + np.random.normal(0, 0.5, size=final_positions.shape)
        else:
             print("Error: No atoms available to position.")
             return # Cannot proceed without atoms

    # Ensure positions array matches the number of atoms in 'actual_atoms' if RDKit reordered indices
    if len(actual_atoms) != final_positions.shape[0]:
         print(f"Warning: Mismatch between actual_atoms ({len(actual_atoms)}) and calculated positions ({final_positions.shape[0]}). This might indicate issues with atom mapping or coordinate generation.")
         # Fallback: Create positions based on actual_atoms count
         num_atoms_actual = len(actual_atoms)
         final_positions = np.random.rand(num_atoms_actual, 2) * 5
         initial_positions = final_positions + np.random.normal(0, 0.5, size=final_positions.shape)
         # Re-map rdkit_idx in actual_atoms to the index in the new position arrays
         for i, atom_info in enumerate(actual_atoms):
              atom_info['pos_idx'] = i # Add a direct index into our position arrays
    else:
         # Add pos_idx mapping assuming order matches RDKit's GetAtoms()
         rdkit_idx_map = {atom.GetIdx(): i for i, atom in enumerate(mol.GetAtoms())}
         for atom_info in actual_atoms:
              atom_info['pos_idx'] = rdkit_idx_map.get(atom_info['rdkit_idx'], -1)
              if atom_info['pos_idx'] == -1:
                   print(f"Error: Could not find position index for rdkit_idx {atom_info['rdkit_idx']}")
                   return


    # Atom properties mapping
    atom_symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
    atom_colors = {
        1: '#FFFFFF', 6: '#222222', 7: '#0000FF', 8: '#FF0000', 9: '#00FF00',
        15: '#FFA500', 16: '#FFFF00', 17: '#00FFFF', 35: '#A52A2A', 53: '#9400D3'
    }
    atom_radii = {1: 0.35, 6: 0.7, 7: 0.65, 8: 0.6, 9: 0.55, 15: 1.0, 16: 1.0, 17: 1.0, 35: 1.15, 53: 1.4}

    # Create a lookup dictionary for atoms based on their *position index*
    pos_idx_to_info = {}
    for atom_info in actual_atoms:
        pos_idx = atom_info.get('pos_idx', -1)
        if pos_idx != -1:
             num = atom_info['atomic_num']
             pos_idx_to_info[pos_idx] = {
                 'final_pos': final_positions[pos_idx],
                 'initial_pos': initial_positions[pos_idx],
                 'symbol': atom_symbols.get(num, '?'),
                 'color': atom_colors.get(num, '#808080'),
                 'radius': atom_radii.get(num, 0.6) * 300 # Scale radius for plotting size
             }

    # Bond properties mapping
    bond_widths = {BondType.SINGLE: 1.5, BondType.DOUBLE: 3.0, BondType.TRIPLE: 4.5, BondType.AROMATIC: 2.0}

    # Get atom position indices for each bond
    bonds_with_pos_indices = []
    # Need mapping from rdkit_idx back to position index
    rdkit_to_pos_idx = {info['rdkit_idx']: info['pos_idx'] for info in actual_atoms if 'pos_idx' in info}

    for bond_info in bonds_added:
         u_pos_idx = rdkit_to_pos_idx.get(bond_info['u'], -1)
         v_pos_idx = rdkit_to_pos_idx.get(bond_info['v'], -1)
         if u_pos_idx != -1 and v_pos_idx != -1:
              bonds_with_pos_indices.append({
                  'u_pos': u_pos_idx,
                  'v_pos': v_pos_idx,
                  'type': bond_info['type']
              })

    # --- Animation Setup ---
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.style.use('seaborn-v0_8-darkgrid')

    # Increase frames for new phases
    total_frames = 150
    num_atoms = len(pos_idx_to_info)
    num_bonds = len(bonds_with_pos_indices)

    # Define phase durations (adjust sum to total_frames)
    frames_latent = 15         # 15
    frames_atom_placement = 30 # 45
    frames_bond_connect = 35   # 80 (Grow bonds)
    frames_relaxation = 30     # 110 (Move to final shape)
    frames_atom_typing = 25    # 135
    frames_final_display = 15  # 150
    phase_ends = np.cumsum([frames_latent, frames_atom_placement, frames_bond_connect, frames_relaxation, frames_atom_typing, frames_final_display])

    # Determine atom and bond appearance order using BFS (using position indices now)
    # Start BFS from a valid position index (e.g., 0 if it exists)
    bfs_atom_pos_order = []
    bfs_bond_order = []
    if num_atoms > 0:
        adj = {i: [] for i in pos_idx_to_info.keys()}
        for bond in bonds_with_pos_indices:
            adj[bond['u_pos']].append(bond['v_pos'])
            adj[bond['v_pos']].append(bond['u_pos'])

        start_node = list(pos_idx_to_info.keys())[0] # Start BFS from the first available atom pos_idx
        q = deque([start_node])
        visited_atoms = {start_node}
        # Use a set to track visited bonds using tuple(sorted(u,v))
        visited_bonds = set()
        bfs_atom_pos_order.append(start_node)


        processed_nodes = 0
        while q:
            u_pos = q.popleft()
            processed_nodes += 1
            if processed_nodes > num_atoms * 2: break # Safety break for weird graphs

            # Find neighbors using the adjacency list 'adj'
            if u_pos in adj:
                 neighbors = adj[u_pos]
            else:
                 neighbors = [] # Node might have no bonds

            for v_pos in neighbors:
                 bond_key = tuple(sorted((u_pos, v_pos)))
                 # Find the corresponding bond dict
                 current_bond_info = next((b for b in bonds_with_pos_indices if tuple(sorted((b['u_pos'], b['v_pos']))) == bond_key), None)

                 if current_bond_info and bond_key not in visited_bonds:
                     bfs_bond_order.append(current_bond_info)
                     visited_bonds.add(bond_key)

                 if v_pos not in visited_atoms:
                     visited_atoms.add(v_pos)
                     bfs_atom_pos_order.append(v_pos)
                     q.append(v_pos)

    # Ensure all atoms/bonds are included if graph was disconnected
    remaining_atoms = [idx for idx in pos_idx_to_info.keys() if idx not in visited_atoms]
    bfs_atom_pos_order.extend(remaining_atoms)
    bond_keys_in_bfs = {tuple(sorted((b['u_pos'], b['v_pos']))) for b in bfs_bond_order}
    remaining_bonds = [b for b in bonds_with_pos_indices if tuple(sorted((b['u_pos'], b['v_pos']))) not in bond_keys_in_bfs]
    bfs_bond_order.extend(remaining_bonds)

    print(f"Animation order: {len(bfs_atom_pos_order)} atoms, {len(bfs_bond_order)} bonds.")

    # --- Utility for drawing bonds ---
    def draw_bond(ax, pos1, pos2, bond_type, width_map, alpha=1.0, color='black'):
        width = width_map.get(bond_type, 1.0)
        line_style = '-' # Default solid line

        # Draw multiple lines slightly offset for double/triple bonds
        if bond_type == BondType.DOUBLE or bond_type == BondType.TRIPLE:
            offset_factor = 0.05 # Adjust for visual spacing
            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]
            length = np.sqrt(dx**2 + dy**2)
            if length > 1e-6: # Avoid division by zero
                offset_x = -dy / length * offset_factor
                offset_y = dx / length * offset_factor
            else:
                offset_x, offset_y = 0, 0

            if bond_type == BondType.DOUBLE:
                ax.plot([pos1[0] + offset_x, pos2[0] + offset_x], [pos1[1] + offset_y, pos2[1] + offset_y],
                        color=color, linewidth=width/2, alpha=alpha, solid_capstyle='round', linestyle=line_style)
                ax.plot([pos1[0] - offset_x, pos2[0] - offset_x], [pos1[1] - offset_y, pos2[1] - offset_y],
                        color=color, linewidth=width/2, alpha=alpha, solid_capstyle='round', linestyle=line_style)
            else: # Triple
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                        color=color, linewidth=width/3, alpha=alpha, solid_capstyle='round', linestyle=line_style)
                ax.plot([pos1[0] + 1.5*offset_x, pos2[0] + 1.5*offset_x], [pos1[1] + 1.5*offset_y, pos2[1] + 1.5*offset_y],
                        color=color, linewidth=width/3, alpha=alpha, solid_capstyle='round', linestyle=line_style)
                ax.plot([pos1[0] - 1.5*offset_x, pos2[0] - 1.5*offset_x], [pos1[1] - 1.5*offset_y, pos2[1] - 1.5*offset_y],
                        color=color, linewidth=width/3, alpha=alpha, solid_capstyle='round', linestyle=line_style)
        else: # Single or Aromatic
             ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                     color=color, linewidth=width, alpha=alpha, solid_capstyle='round', linestyle=line_style)

    # --- Animation Update Function ---
    def update(frame):
        ax.clear()
        ax.set_facecolor('#e9e9e9') # Slightly lighter gray background
        ax.set_aspect('equal', adjustable='box')

        # Determine plot limits based on FINAL positions, add padding
        if final_positions is not None and final_positions.size > 0:
             min_coords = final_positions.min(axis=0)
             max_coords = final_positions.max(axis=0)
             center = (min_coords + max_coords) / 2
             span = (max_coords - min_coords)
             # Ensure span is not zero in either dimension
             span[span < 1e-6] = 1.0
             max_span = span.max()
             padding = max(max_span * 0.2, 1.5) # More padding
             ax.set_xlim(center[0] - max_span/2 - padding, center[0] + max_span/2 + padding)
             ax.set_ylim(center[1] - max_span/2 - padding, center[1] + max_span/2 + padding)
        else:
             ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.axis('off')

        # --- Phase Timing ---
        current_phase = np.searchsorted(phase_ends, frame + 1) # Find which phase the frame belongs to
        start_frame_phase = 0 if current_phase == 0 else phase_ends[current_phase - 1]
        phase_duration = phase_ends[current_phase] - start_frame_phase
        phase_progress = (frame - start_frame_phase) / phase_duration if phase_duration > 0 else 1.0

        # --- Draw based on phase ---
        if current_phase == 0: # Latent Space
            ax.set_title("Phase 1: Sampling Latent Vector", fontsize=14, pad=20)
            num_points = 50
            current_center = center if 'center' in locals() else np.array([0,0]) # Use calculated center or default
            noise_scale = (1 - phase_progress**2) * 2
            center_noise = np.random.normal(0, noise_scale, size=(num_points, 2))
            colors = plt.cm.viridis(np.linspace(0, 1, num_points))
            ax.scatter(current_center[0] + center_noise[:, 0], current_center[1] + center_noise[:, 1],
                       s=50 * (1 - phase_progress) + 10, alpha=0.6 * (1-phase_progress) + 0.1, c=colors)

        elif current_phase == 1: # Atom Placement (at initial positions)
            ax.set_title("Phase 2: Placing Atoms (Initial Positions)", fontsize=14, pad=20)
            num_atoms_to_show = int(np.ceil(phase_progress * len(bfs_atom_pos_order)))

            for i in range(num_atoms_to_show):
                pos_idx = bfs_atom_pos_order[i]
                if pos_idx in pos_idx_to_info:
                    info = pos_idx_to_info[pos_idx]
                    # Calculate alpha for smooth fade-in
                    appearance_point = (i + 1) / len(bfs_atom_pos_order)
                    alpha = np.clip((phase_progress - (appearance_point * 0.8)) / 0.2, 0, 1)
                    # Draw as gray placeholder at INITIAL position
                    ax.scatter(info['initial_pos'][0], info['initial_pos'][1],
                               s=info['radius'] * 0.8, c='gray', alpha=alpha * 0.8, edgecolors='dimgray', linewidth=0.5)

        elif current_phase == 2: # Bond Connection (at initial positions)
            ax.set_title("Phase 3: Connecting Bonds (Growing)", fontsize=14, pad=20)
            num_bonds_to_show = len(bfs_bond_order) # Determine how many *start* drawing
            atoms_already_drawn = len(bfs_atom_pos_order) # All atoms are placed now

            # Draw all placed atoms (at initial positions)
            for i in range(atoms_already_drawn):
                pos_idx = bfs_atom_pos_order[i]
                if pos_idx in pos_idx_to_info:
                    info = pos_idx_to_info[pos_idx]
                    ax.scatter(info['initial_pos'][0], info['initial_pos'][1], s=info['radius']*0.8, c='gray', alpha=0.8, edgecolors='dimgray', linewidth=0.5)

            # Draw bonds sequentially, making them "grow"
            for i in range(num_bonds_to_show):
                bond_info = bfs_bond_order[i]
                u_pos, v_pos = bond_info['u_pos'], bond_info['v_pos']

                if u_pos in pos_idx_to_info and v_pos in pos_idx_to_info:
                    info_u = pos_idx_to_info[u_pos]
                    info_v = pos_idx_to_info[v_pos]
                    pos1 = info_u['initial_pos']
                    pos2 = info_v['initial_pos']
                    bond_type = bond_info['type']

                    # Calculate appearance window for this bond
                    appearance_start_progress = i / num_bonds_to_show if num_bonds_to_show > 0 else 0
                    appearance_end_progress = (i + 1) / num_bonds_to_show if num_bonds_to_show > 0 else 1
                    duration_this_bond = appearance_end_progress - appearance_start_progress

                    # Interpolation factor for bond growth (0 to 1 within its window)
                    interp_factor = 0.0
                    if phase_progress >= appearance_start_progress:
                         interp_factor = np.clip((phase_progress - appearance_start_progress) / duration_this_bond, 0, 1) if duration_this_bond > 1e-6 else 1.0

                    alpha = interp_factor # Fade in as it grows

                    # Calculate the growing end point
                    pos_end_interp = pos1 + interp_factor * (pos2 - pos1)

                    # Draw the growing bond
                    draw_bond(ax, pos1, pos_end_interp, bond_type, bond_widths, alpha=alpha, color='dimgray')


        elif current_phase == 3: # Relaxation (Move from initial to final)
            ax.set_title("Phase 4: Relaxing to Final Shape", fontsize=14, pad=20)
            lerp_factor = phase_progress # Interpolation factor from 0 to 1

            # Draw atoms interpolating position
            current_atom_positions = {}
            for pos_idx, info in pos_idx_to_info.items():
                current_pos = info['initial_pos'] * (1 - lerp_factor) + info['final_pos'] * lerp_factor
                current_atom_positions[pos_idx] = current_pos
                ax.scatter(current_pos[0], current_pos[1], s=info['radius']*0.8, c='gray', alpha=0.9, edgecolors='dimgray', linewidth=0.5)

            # Draw bonds between interpolated atom positions
            for bond_info in bfs_bond_order:
                u_pos, v_pos = bond_info['u_pos'], bond_info['v_pos']
                if u_pos in current_atom_positions and v_pos in current_atom_positions:
                    pos1 = current_atom_positions[u_pos]
                    pos2 = current_atom_positions[v_pos]
                    bond_type = bond_info['type']
                    draw_bond(ax, pos1, pos2, bond_type, bond_widths, alpha=0.8, color='dimgray')


        elif current_phase == 4: # Atom Typing (at final positions)
            ax.set_title("Phase 5: Assigning Atom Types", fontsize=14, pad=20)
            num_atoms_typed = int(np.ceil(phase_progress * len(bfs_atom_pos_order)))

            # Draw all bonds at final positions first
            for bond_info in bfs_bond_order:
                 u_pos, v_pos = bond_info['u_pos'], bond_info['v_pos']
                 if u_pos in pos_idx_to_info and v_pos in pos_idx_to_info:
                     pos1 = pos_idx_to_info[u_pos]['final_pos']
                     pos2 = pos_idx_to_info[v_pos]['final_pos']
                     bond_type = bond_info['type']
                     draw_bond(ax, pos1, pos2, bond_type, bond_widths, alpha=0.9, color='black')

            # Draw atoms, coloring them sequentially
            for i, pos_idx in enumerate(bfs_atom_pos_order):
                if pos_idx in pos_idx_to_info:
                    info = pos_idx_to_info[pos_idx]
                    final_pos = info['final_pos']

                    # Determine if this atom should be typed yet
                    appearance_point = (i + 1) / len(bfs_atom_pos_order) if len(bfs_atom_pos_order)>0 else 0
                    is_typed = phase_progress >= appearance_point
                    type_progress = np.clip((phase_progress - appearance_point) / (1.0 / len(bfs_atom_pos_order) if len(bfs_atom_pos_order)>0 else 1.0) , 0, 1)

                    color = info['color'] if is_typed else 'gray'
                    base_alpha = 1.0 if is_typed else 0.9
                    # Add a 'pop' effect when typed
                    size_scale = 1.0 + 0.4 * np.sin(type_progress * np.pi) if is_typed else 0.8 # Pops then settles

                    # Draw atom circle at FINAL position
                    ax.scatter(final_pos[0], final_pos[1],
                               s=info['radius'] * size_scale,
                               c=color, alpha=base_alpha, edgecolors='black', linewidth=1.0, zorder=3)

                    # Add symbol if atom is not C/H (and is typed)
                    if is_typed and info['symbol'] not in ['C', 'H']:
                        text_color = 'white' if mcolors.rgb_to_hsv(mcolors.to_rgb(color))[-1] < 0.5 else 'black'
                        ax.text(final_pos[0], final_pos[1], info['symbol'],
                                ha='center', va='center', fontsize=10, color=text_color, weight='bold', zorder=4)


        elif current_phase == 5: # Final Display
            ax.set_title(f"Phase 6: Generated Molecule\nSMILES: {smiles}", fontsize=14, pad=20)
            final_alpha = np.clip(phase_progress * 1.5, 0, 1) # Fade in final structure

            # Draw final bonds
            for bond_info in bfs_bond_order:
                u_pos, v_pos = bond_info['u_pos'], bond_info['v_pos']
                if u_pos in pos_idx_to_info and v_pos in pos_idx_to_info:
                    pos1 = pos_idx_to_info[u_pos]['final_pos']
                    pos2 = pos_idx_to_info[v_pos]['final_pos']
                    bond_type = bond_info['type']
                    draw_bond(ax, pos1, pos2, bond_type, bond_widths, alpha=final_alpha, color='black')

            # Draw final atoms
            for pos_idx, info in pos_idx_to_info.items():
                final_pos = info['final_pos']
                color = info['color']
                ax.scatter(final_pos[0], final_pos[1],
                           s=info['radius'], # Final size
                           c=color, alpha=final_alpha, edgecolors='black', linewidth=1.0, zorder=3)
                # Add symbol if not C/H
                if info['symbol'] not in ['C', 'H']:
                     text_color = 'white' if mcolors.rgb_to_hsv(mcolors.to_rgb(color))[-1] < 0.5 else 'black'
                     ax.text(final_pos[0], final_pos[1], info['symbol'],
                             ha='center', va='center', fontsize=10, color=text_color, weight='bold', alpha=final_alpha, zorder=4)

        return [] # Return list of artists changed (can be empty for blit=False)

    # --- Create and Save Animation ---
    anim = FuncAnimation(fig, update, frames=total_frames, blit=False, interval=80) # Adjust interval (ms) for speed

    output_filename = os.path.join(output_dir, f"molecule_build_seed{seed}_temp{temp}.gif")
    print(f"Saving animation to {output_filename}...")
    try:
        anim.save(output_filename, writer='pillow', dpi=150) # Pillow writer is common for GIFs
        print("Animation saved successfully.", flush=True)
    except Exception as e:
        print(f"Error saving animation: {e}", flush=True)
        print("Make sure 'Pillow' is installed (`pip install Pillow`). You might also need ffmpeg for other writers.")

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and visualize molecule creation using MoFlow.")
    parser.add_argument("--save_dir", type=str, default="./results/qm9", help="Directory where model checkpoints and hyperparams are saved.")
    parser.add_argument("--snapshot-epoch", type=int, default=None, help="Epoch number of the snapshot to load. If None, tries to find the latest.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID (0, 1, ...). Set to -1 for CPU.")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature for noise sampling during generation (controls stochasticity).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, default="molecule_animations", help="Directory to save the output animation.")
    args = parser.parse_args()

    # --- Determine snapshot path ---
    if args.snapshot_epoch is None:
        try:
            snapshot_files = [f for f in os.listdir(args.save_dir) if f.startswith("model_snapshot_epoch_") and f.split('_')[-1].isdigit()]
            if not snapshot_files:
                 raise FileNotFoundError(f"No model snapshots found in save_dir: {args.save_dir}")
            epochs = [int(f.split('_')[-1]) for f in snapshot_files]
            latest_epoch = max(epochs)
            snapshot_path = os.path.join(args.save_dir, f"model_snapshot_epoch_{latest_epoch}")
            print(f"Using latest snapshot found: Epoch {latest_epoch} ({snapshot_path})")
        except Exception as e:
            print(f"Error finding latest snapshot in '{args.save_dir}': {e}")
            sys.exit(1)
    else:
        snapshot_path = os.path.join(args.save_dir, f"model_snapshot_epoch_{args.snapshot_epoch}")

    hyperparams_path = os.path.join(args.save_dir, "moflow-params.json") # Adjust filename if needed

    # --- Set device ---
    if args.gpu >= 0 and torch.cuda.is_available():
        device_str = f"cuda:{args.gpu}"
        multigpu_flag = False # Keep simple for now, load_model handles prefix removal
    else:
        device_str = "cpu"
        multigpu_flag = False

    # --- Load Model ---
    try:
        model = load_model(snapshot_path, hyperparams_path, device=device_str, multigpu=multigpu_flag)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    # --- Generate and Simulate ---
    simulate_molecule_creation(model, output_dir=args.output_dir, temp=args.temp, seed=args.seed)

    print("Script finished.")