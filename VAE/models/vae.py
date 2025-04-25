import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from VAE.models.layers import LatentSampler
from VAE.utils.data_utils import molecular_graph_to_molecule

class MolecularVAE(keras.Model):
    def __init__(self, encoder_model, decoder_model, max_atoms, kl_factor=1e-3, recon_factor=1.0, prop_factor=1.0, grad_factor=0.01, **kwargs):
        super().__init__(**kwargs)
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.prop_predictor = layers.Dense(1)  # Linear output
        self.max_atoms = max_atoms
        self.latent_sampler = LatentSampler()
        self.kl_factor = kl_factor
        self.recon_factor = recon_factor / max_atoms
        self.prop_factor = prop_factor
        self.grad_factor = grad_factor

        # Trackers for loss metrics
        self.train_loss_tracker = keras.metrics.Mean(name="train_total_loss")
        self.val_loss_tracker = keras.metrics.Mean(name="val_total_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.prop_loss_tracker = keras.metrics.Mean(name="prop_loss")
        self.graph_loss_tracker = keras.metrics.Mean(name="graph_loss")

    @property
    def metrics(self):
        return [
            self.train_loss_tracker,
            self.val_loss_tracker,
            self.kl_loss_tracker,
            self.recon_loss_tracker,
            self.prop_loss_tracker,
            self.graph_loss_tracker,
        ]

    def train_step(self, data):
        mol_data, mol_prop = data
        graph_input = mol_data
        self.batch_size = tf.shape(mol_prop)[0]
        
        with tf.GradientTape() as tape:
            latent_mean, latent_log_var, prop_pred, \
            edge_recon, node_recon = self(mol_data, training=True)
            graph_output = [edge_recon, node_recon]
            total_loss, kl_loss, prop_loss, edge_loss, node_loss, graph_loss = self.compute_losses(
                latent_log_var,
                latent_mean,
                mol_prop,
                prop_pred,
                graph_input,
                graph_output,
                training=True
            )

        # Update weights
        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        # Log metrics
        self.train_loss_tracker.update_state(total_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.prop_loss_tracker.update_state(prop_loss)
        self.recon_loss_tracker.update_state(edge_loss + node_loss)
        self.graph_loss_tracker.update_state(graph_loss)
        
        return {
            "total_loss": self.train_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "prop_loss": self.prop_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "graph_loss": self.graph_loss_tracker.result(),
        }

    def test_step(self, data):
        mol_data, mol_prop = data
        latent_mean, latent_log_var, prop_pred, \
        edge_recon, node_recon = self(mol_data, training=False)
                                                                
        total_loss, kl_loss, prop_loss, edge_loss, node_loss, graph_loss = self.compute_losses(
            latent_log_var,
            latent_mean,
            mol_prop,
            prop_pred,
            graph_input=mol_data,
            graph_output=[edge_recon, node_recon],
            training=False
        )

        # Log validation loss
        self.val_loss_tracker.update_state(total_loss)
        
        return {
            "total_loss": self.val_loss_tracker.result(),
        }

    def compute_losses(self, log_var, mean, true_prop, pred_prop, graph_input, graph_output, training):
        edge_true, node_true = graph_input
        edge_pred, node_pred = graph_output

        # Small constant to prevent log(0)
        small_eps = 1e-10
        
        # Stabilize outputs
        edge_pred = tf.clip_by_value(edge_pred, small_eps, 1.0 - small_eps)
        node_pred = tf.clip_by_value(node_pred, small_eps, 1.0 - small_eps)
        
        # Compute reconstruction losses
        edge_recon_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.categorical_crossentropy(edge_true, edge_pred),
                axis=(1, 2)
            )
        )
        node_recon_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.categorical_crossentropy(node_true, node_pred),
                axis=1
            )
        )
        
        # Compute KL divergence with stabilization
        kl_loss = -0.5 * tf.reduce_sum(
            1 + tf.clip_by_value(log_var, -20, 20) - 
            tf.square(tf.clip_by_value(mean, -20, 20)) - 
            tf.exp(tf.clip_by_value(log_var, -20, 20)), 
            axis=1
        )
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss = tf.clip_by_value(kl_loss, 0, 1000)

        # Compute property prediction loss
        prop_loss = tf.reduce_mean(
            tf.square(true_prop - pred_prop)
        )
        prop_loss = tf.clip_by_value(prop_loss, 0, 100)

        # Compute gradient penalty for training
        graph_loss = self._compute_stable_grad_penalty(graph_input, graph_output) if training else 0

        # Aggregate losses
        total_loss = (self.kl_factor * kl_loss +
                     self.prop_factor * prop_loss +
                     self.grad_factor * graph_loss +
                     self.recon_factor * (edge_recon_loss + node_recon_loss))
        
        # Handle NaN values
        total_loss = tf.where(tf.math.is_nan(total_loss), tf.constant(1000.0, dtype=tf.float32), total_loss)
        
        return total_loss, kl_loss, prop_loss, edge_recon_loss, node_recon_loss, graph_loss

    def _compute_stable_grad_penalty(self, graph_input, graph_output):
        edge_true, node_true = graph_input
        edge_pred, node_pred = graph_output

        # Sample interpolation factor
        alpha = tf.random.uniform([self.batch_size], 0.2, 0.8)
        
        # Reshape for interpolation
        alpha_edge = tf.reshape(alpha, (self.batch_size, 1, 1, 1))
        alpha_node = tf.reshape(alpha, (self.batch_size, 1, 1))
        
        # Interpolate graphs
        edge_interp = edge_true * alpha_edge + (1 - alpha_edge) * edge_pred
        node_interp = node_true * alpha_node + (1 - alpha_node) * node_pred

        with tf.GradientTape() as tape:
            tape.watch([edge_interp, node_interp])
            _, _, logits, _, _ = self([edge_interp, node_interp], training=True)

        gradients = tape.gradient(logits, [edge_interp, node_interp])
        
        # Stabilize gradients
        grad_edge_norm = tf.clip_by_value(tf.norm(gradients[0], axis=1), 0.1, 10.0)
        grad_node_norm = tf.clip_by_value(tf.norm(gradients[1], axis=2), 0.1, 10.0)
        
        # Compute penalties
        edge_penalty = tf.square(grad_edge_norm - 1.0)
        node_penalty = tf.square(grad_node_norm - 1.0)
        
        penalty = tf.reduce_mean(
            tf.reduce_mean(edge_penalty, axis=(-2, -1)) +
            tf.reduce_mean(node_penalty, axis=-1)
        )
        
        return tf.clip_by_value(penalty, 0, 100)

    def sample_molecules(self, batch_size, temp=1.0):
        latent_samples = tf.random.normal((batch_size, self.encoder_model.output[0].shape[1])) * temp
        edge_recon, node_recon = self.decoder_model(latent_samples)
        edge_indices = tf.argmax(edge_recon, axis=1)
        edge_one_hot = tf.one_hot(edge_indices, depth=edge_recon.shape[1], axis=1)
        edge_one_hot = tf.linalg.set_diag(edge_one_hot, tf.zeros(tf.shape(edge_one_hot)[:-1]))
        node_indices = tf.argmax(node_recon, axis=2)
        node_one_hot = tf.one_hot(node_indices, depth=node_recon.shape[2], axis=2)
        molecules = []
        for i in range(batch_size):
            mol = molecular_graph_to_molecule([edge_one_hot[i].numpy(), node_one_hot[i].numpy()])
            molecules.append(mol)
        return molecules
        
    def optimize_molecules(self, num_samples, target_prop, opt_steps=100, step_lr=0.1):
        latent_vars = tf.Variable(tf.random.normal((num_samples, self.encoder_model.output[0].shape[1])))
        for _ in range(opt_steps):
            with tf.GradientTape() as tape:
                edge_out, node_out = self.decoder_model(latent_vars)
                prop_out = self.prop_predictor(latent_vars)
                loss = tf.reduce_mean(tf.square(prop_out - target_prop))
            gradients = tape.gradient(loss, latent_vars)
            latent_vars.assign_sub(step_lr * gradients)
        edge_recon, node_recon = self.decoder_model(latent_vars)
        edge_indices = tf.argmax(edge_recon, axis=1)
        edge_one_hot = tf.one_hot(edge_indices, depth=edge_recon.shape[1], axis=1)
        edge_one_hot = tf.linalg.set_diag(edge_one_hot, tf.zeros(tf.shape(edge_one_hot)[:-1]))
        node_indices = tf.argmax(node_recon, axis=2)
        node_one_hot = tf.one_hot(node_indices, depth=node_recon.shape[2], axis=2)
        molecules = []
        prop_values = []
        for i in range(num_samples):
            mol = molecular_graph_to_molecule([edge_one_hot[i].numpy(), node_one_hot[i].numpy()])
            molecules.append(mol)
            prop_values.append(float(self.prop_predictor(latent_vars[i:i+1]).numpy()[0]))
        return molecules, prop_values

    def call(self, inputs):
        latent_mean, latent_log_var = self.encoder_model(inputs)
        latent_sample = self.latent_sampler([latent_mean, latent_log_var])
        edge_recon, node_recon = self.decoder_model(latent_sample)
        prop_pred = self.prop_predictor(latent_mean)
        return latent_mean, latent_log_var, prop_pred, edge_recon, node_recon