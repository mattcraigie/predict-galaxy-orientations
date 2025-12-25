import torch
import torch.nn as nn
from torch.distributions import VonMises
import numpy as np

from .basic_models import MLP
from .galaxy_gnn import GalaxyLSSBackbone, GalaxyLSSWrapper


def init_vmdn(model_config):
    # 1. Extract Dimensions separately
    gnn_dim = model_config.get("gnn_hidden_dim", 64)  # Backbone capacity
    vmdn_dim = model_config.get("vmdn_hidden_dim", 32)  # Head capacity (MLP width)

    # 2. Extract Regularization Params
    lambda_kappa = model_config.get("vmdn_kappa_weight", 1e-3)
    iso_weight = model_config.get("vmdn_isotropy_weight", 1.0)
    num_layers = model_config.get("num_layers", 3)

    # 3. Initialize Backbone with GNN Dim
    backbone = GalaxyLSSBackbone(hidden_dim=gnn_dim, num_layers=num_layers)
    wrapper = GalaxyLSSWrapper(backbone)

    # 4. Initialize VMDN with Head Dim
    # We create a list [vmdn_dim, vmdn_dim] to define the hidden layers of the MLP
    return VMDN(
        wrapper,
        hidden_layers=[vmdn_dim, vmdn_dim],
        lambda_kappa=lambda_kappa,
        # iso_weight=iso_weight
    )


import torch
import torch.nn as nn
from torch.distributions import VonMises


class VMDN(nn.Module):
    """
    Von Mises Density Network.

    Predicts mean (mu) and concentration (kappa).
    """

    def __init__(
            self,
            compression_network,
            hidden_layers=(32, 32),
            lambda_kappa=0.0,
            dropout=0.0,
    ):
        super().__init__()

        self.compression_network = compression_network
        self.lambda_kappa = lambda_kappa
        self.dropout = dropout

        # Angle network outputs (x, y) to preserve angular continuity
        self.angle_network = MLP(
            input_dim=compression_network.out_size,
            output_dim=2,
            hidden_layers=hidden_layers,
        )

        self.kappa_network = MLP(
            input_dim=compression_network.out_size,
            output_dim=1,
            hidden_layers=hidden_layers,
        )

    def forward(self, pos, z, shapes, edge_index=None):
        compressed = self.compression_network(
            pos, z, shapes, edge_index
        )

        # Predict angle as unit vector, then convert to angle
        mean_xy = self.angle_network(compressed)
        mean_xy = mean_xy / (mean_xy.norm(dim=-1, keepdim=True) + 1e-8)

        mu = torch.atan2(mean_xy[..., 1], mean_xy[..., 0])

        log_kappa = self.kappa_network(compressed)
        log_kappa = torch.clamp(log_kappa, min=-10.0, max=-5.0)
        kappa = torch.exp(log_kappa)

        return mu, kappa

    def loss(
            self,
            pos,
            z,
            shapes,
            target,
            edge_index=None,
            mask=None,
    ):
        """
        Full-batch loss with stochastic masking and soft kappa regularization.

        target: expected to be 2 * phi (Spin-1 domain)
        """
        mu, kappa = self.forward(pos, z, shapes, edge_index)

        # Ensure shapes match
        if target.dim() > 1:
            target = target.view(-1)
        if mu.dim() > 1:
            mu = mu.view(-1)
        if kappa.dim() > 1:
            kappa = kappa.view(-1)

        # Von Mises log-probability
        dist = VonMises(mu, kappa)
        log_prob = dist.log_prob(target)

        # Apply mask if provided
        if mask is not None:
            log_prob = log_prob[mask]
            active_kappa = kappa[mask]
        else:
            active_kappa = kappa

        # 1. Negative log-likelihood
        nll = -log_prob.mean()

        # 2. Soft kappa regularization (entropy penalty)
        kappa_penalty = self.lambda_kappa * active_kappa.mean()

        # Total loss
        total_loss = nll + kappa_penalty

        return total_loss
