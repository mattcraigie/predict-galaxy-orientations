import torch
import torch.nn as nn
from .galaxy_gnn import GalaxyLSSBackbone, GalaxyLSSWrapper


class GalaxyReconstructor(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1. Use the SAME Backbone as your main model
        self.backbone = GalaxyLSSBackbone(
            hidden_dim=config['hidden_dim'],
            num_layers=config.get('num_layers', 3),
            input_scalar_dim=1,
            input_vector_dim=2
        )
        self.wrapper = GalaxyLSSWrapper(self.backbone)

        # 2. Reconstruction Heads
        # Predict Redshift (Scalar)
        self.z_head = nn.Sequential(
            nn.Linear(self.wrapper.out_size, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

        # Predict Input Shape (Vector components e1, e2)
        self.shape_head = nn.Sequential(
            nn.Linear(self.wrapper.out_size, 32),
            nn.SiLU(),
            nn.Linear(32, 2)
        )

    def forward(self, pos, z, shapes, edge_index):
        # Get latent representation
        feats = self.wrapper(pos, z, shapes, edge_index)

        # Reconstruct properties
        pred_z = self.z_head(feats)
        pred_shape = self.shape_head(feats)

        return pred_z, pred_shape

    def loss(self, pos, z, shapes, edge_index, mask):
        """
        Reconstruction Loss (MSE) on masked nodes only.
        """
        pred_z, pred_shape = self.forward(pos, z, shapes, edge_index)

        # Compute loss only on the MASKED nodes
        # We want to predict the values we hid!
        target_z = z[mask]
        target_shape = shapes[mask]

        loss_z = nn.functional.mse_loss(pred_z[mask], target_z)
        loss_shape = nn.functional.mse_loss(pred_shape[mask], target_shape)

        # You can weight these if one dominates
        return loss_z + loss_shape