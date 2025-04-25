import sys
import os

# Add project root (Molecule_Generation/moflow) to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import torch
from train_moflow import train  # assuming your original script is renamed to `train_script.py`
from types import SimpleNamespace

args = SimpleNamespace(
    data_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')),
    data_name='qm9',
    save_dir='results/qm9',
    save_interval=20,
    load_params=0,
    load_snapshot='',
    learning_rate=0.001,
    lr_decay=0.999995,
    max_epochs=5000,
    gpu=0,
    save_epochs=1,
    batch_size=12,
    shuffle=False,
    num_workers=0,
    b_n_flow=10,
    b_n_block=1,
    b_hidden_ch="128,128",
    b_conv_lu=1,
    a_n_flow=27,
    a_n_block=1,
    a_hidden_gnn="64,",
    a_hidden_lin="128,64",
    mask_row_size_list="1,",
    mask_row_stride_list="1,",
    seed=1,
    debug=True,
    learn_dist=True,
    noise_scale=0.6
)

train(args)
