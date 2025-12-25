import torch
import torch.nn as nn
from .galaxy_gnn import GalaxyLSSBackbone, GalaxyLSSWrapper


class GalaxyReconstructor(nn.Module):
    """
    Pre-training model that forces the FIRST dimensions of the latent space
    to reconstruct the physical properties (Redshift and Shape).
    """

    def __init__(self, config):
        super().__init__()
        # Use exact same backbone configuration as main model
        self.backbone = GalaxyLSSBackbone(
            hidden_dim=config['gnn_hidden_dim'],
            num_layers=config.get('num_layers', 3),
            input_scalar_dim=1,
            input_vector_dim=2
        )
        self.wrapper = GalaxyLSSWrapper(self.backbone)

        # We need to know where the vector features start in the concatenated output
        self.scalar_dim = self.backbone.s_dim

    def forward(self, pos, z, shapes, edge_index):
        # Get full latent representation: [h_s, h_v]
        # shape: [N, scalar_dim + vector_dim]
        feats = self.wrapper(pos, z, shapes, edge_index)

        # --- TRUNCATION STRATEGY ---
        # 1. Predict Redshift using the FIRST Scalar dimension
        pred_z = feats[:, 0]  # Shape [N]

        # 2. Predict Shape using the FIRST Vector pair
        # The wrapper concatenates [scalars, vectors].
        # Vectors start at index `self.scalar_dim`.
        # We take the first pair (x, y) which corresponds to (e1, e2).
        pred_shape = feats[:, self.scalar_dim: self.scalar_dim + 2]  # Shape [N, 2]

        return pred_z, pred_shape

    def loss(self, pos, z, shapes, edge_index, mask):
        """
        Reconstruction Loss forcing latent structure to match physics.
        """
        pred_z, pred_shape = self.forward(pos, z, shapes, edge_index)

        target_z = z[mask].squeeze()
        target_shape = shapes[mask]

        # Slice predictions to masked nodes
        masked_pred_z = pred_z[mask]
        masked_pred_shape = pred_shape[mask]

        # MSE Loss
        loss_z = nn.functional.mse_loss(masked_pred_z, target_z)
        loss_shape = nn.functional.mse_loss(masked_pred_shape, target_shape)

        # Weighting: Shape is usually smaller (-1 to 1) than Z (0 to 1.5),
        # but both are roughly order 1.
        return loss_z + loss_shape