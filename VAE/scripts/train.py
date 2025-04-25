import os
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import numpy as np

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import *
from models.encoder import build_encoder
from models.decoder import build_decoder
from models.vae import MolecularVAE
from utils.data_utils import preprocess_qm9_dataset, generate_tf_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train VAE for molecular generation')
    parser.add_argument('--dataset_path', type=str, default='VAE/data/qm9/processed_qm9.csv',
                        help='Path to QM9 dataset')
    parser.add_argument('--target_prop', type=str, default='qed',
                        help='Property to predict (qed, gap, homo, lumo)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--learn_rate', type=float, default=1e-5,
                        help='Optimizer learning rate')
    parser.add_argument('--latent_size', type=int, default=LATENT_DIM,
                        help='Size of latent space')
    parser.add_argument('--save_dir', type=str, default='models/saved',
                        help='Directory to save trained model')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Log device configuration
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.warning("GPU detected but disabled. Using CPU.")
    else:
        logger.info("Running on CPU.")
    
    cpus = tf.config.list_physical_devices('CPU')
    logger.info(f"CPU devices: {cpus}")
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset_path}...")
    try:
        train_data, test_data = preprocess_qm9_dataset(args.dataset_path, target_prop=args.target_prop)
        logger.info(f"Training set size: {len(train_data)} molecules")
        logger.info(f"Test set size: {len(test_data)} molecules")
        
        # Validate dataset
        logger.info("Validating dataset...")
        for data, label in [(train_data, 'train'), (test_data, 'test')]:
            if args.target_prop not in data.columns:
                logger.error(f"Missing '{args.target_prop}' in {label} set")
                raise ValueError(f"Missing '{args.target_prop}'")
            if data[args.target_prop].isna().any():
                logger.warning(f"NaN values in {args.target_prop} for {label} set")
            if data[args.target_prop].isin([np.inf, -np.inf]).any():
                logger.warning(f"Infinite values in {args.target_prop} for {label} set")
            logger.info(f"{label} set - {args.target_prop} stats: {data[args.target_prop].describe()}")
    except Exception as e:
        logger.error(f"Dataset loading failed: {e}")
        raise
    
    # Prepare datasets
    logger.info("Preparing data pipelines...")
    try:
        train_dataset = generate_tf_dataset(
            dataframe=train_data,
            prop_column=args.target_prop,
            max_atoms=NUM_ATOMS,
            batch_size=args.batch_size,
            shuffle_data=True
        )
        test_dataset = generate_tf_dataset(
            dataframe=test_data,
            prop_column=args.target_prop,
            max_atoms=NUM_ATOMS,
            batch_size=args.batch_size,
            shuffle_data=False
        )
        
        # Inspect dataset
        logger.info("Inspecting dataset format...")
        for batch in train_dataset.take(1):
            (edges, nodes), props = batch
            logger.info(f"Batch shapes: edges={edges.shape}, nodes={nodes.shape}, props={props.shape}")
            if tf.reduce_any(tf.math.is_nan(edges)) or tf.reduce_any(tf.math.is_nan(nodes)) or tf.reduce_any(tf.math.is_nan(props)):
                logger.warning("NaN values in training batch")
    except Exception as e:
        logger.error(f"Dataset creation failed: {e}")
        raise
    
    # Initialize optimizer
    logger.info("Setting up optimizer...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learn_rate, clipnorm=1.0)
    
    # Build encoder
    logger.info("Constructing encoder...")
    try:
        encoder = build_encoder(
            graph_units=[9],
            edge_shape=(BOND_DIM, NUM_ATOMS, NUM_ATOMS),
            node_shape=(NUM_ATOMS, ATOM_DIM),
            latent_space_dim=args.latent_size,
            fully_connected_units=[128],
            dropout_prob=0.2,
        )
    except Exception as e:
        logger.error(f"Encoder construction failed: {e}")
        raise
    
    # Build decoder
    logger.info("Constructing decoder...")
    try:
        decoder = build_decoder(
            fully_connected_units=[32, 64, 128],
            dropout_prob=0.4,
            latent_space_dim=args.latent_size,
            edge_shape=(BOND_DIM, NUM_ATOMS, NUM_ATOMS),
            node_shape=(NUM_ATOMS, ATOM_DIM)
        )
    except Exception as e:
        logger.error(f"Decoder construction failed: {e}")
        raise
    
    # Build VAE
    logger.info("Constructing VAE...")
    try:
        vae_model = MolecularVAE(
            encoder, decoder, max_atoms=NUM_ATOMS,
            kl_factor=1e-3,
            recon_factor=1.0,
            prop_factor=0.5,
            grad_factor=0.001
        )
        vae_model.compile(optimizer=optimizer)
    except Exception as e:
        logger.error(f"VAE construction failed: {e}")
        raise
    
    # Configure callbacks
    log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    )
    
    checkpoint_path = os.path.join(args.save_dir, f"qm9_vae_{args.target_prop}.weights.h5")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )
    
    class NaNMonitor(tf.keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            if logs is None:
                logs = {}
            if any(np.isnan(v) for v in logs.values()):
                logger.warning(f"NaN in batch {batch}: {logs}")
                self.model.stop_training = True
    
    # Train model
    logger.info(f"Training for {args.num_epochs} epochs...")
    try:
        history = vae_model.fit(
            train_dataset,
            epochs=args.num_epochs,
            validation_data=test_dataset,
            callbacks=[tensorboard, checkpoint, NaNMonitor()],
            verbose=1
        )
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise
    
    # Save final model
    final_path = os.path.join(args.save_dir, f"qm9_vae_{args.target_prop}_final.weights.h5")
    try:
        vae_model.save_weights(final_path)
        logger.info(f"Model saved to {final_path}")
    except Exception as e:
        logger.error(f"Model saving failed: {e}")
        raise
    
    # Visualize training
    try:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['total_loss'])
        if 'val_loss' in history.history:  # Check for val_loss
            plt.plot(history.history['val_loss'], label='val')
        plt.title('VAE Total Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        for loss_key in ['kl_loss', 'recon_loss', 'prop_loss', 'graph_loss']:
            if loss_key in history.history:
                plt.plot(history.history[loss_key], label=loss_key.capitalize())
        plt.title('Loss Breakdown')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        
        plt.tight_layout()
        plot_path = os.path.join(args.save_dir, f"training_plot_{args.target_prop}.png")
        plt.savefig(plot_path)
        logger.info(f"Training plot saved to {plot_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Training visualization failed: {e}")

if __name__ == "__main__":
    main()