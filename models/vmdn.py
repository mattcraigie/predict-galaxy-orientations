import torch
import torch.nn as nn
from torch.distributions import VonMises
import numpy as np

from .basic_models import MLP
from .galaxy_gnn import GalaxyLSSBackbone, GalaxyLSSWrapper


def init_vmdn(model_config):
    hidden_dim = model_config.get("hidden_dim", 64)
    backbone = GalaxyLSSBackbone(hidden_dim=hidden_dim, num_layers=model_config.get("num_layers", 3))
    wrapper = GalaxyLSSWrapper(backbone)
    return VMDN(wrapper)


class VMDN(nn.Module):
    """
    Von Mises Density Network.
    Predicts mean (mu) and concentration (kappa).
    """

    def __init__(self, compression_network, hidden_layers=[32, 32], lambda_kappa=0.0, dropout=0.0):
        super().__init__()
        self.compression_network = compression_network
        self.lambda_kappa = lambda_kappa

        # Angle Network outputs (x, y) vector to maintain continuity
        self.angle_network = MLP(input_dim=compression_network.out_size, output_dim=2, hidden_layers=hidden_layers)
        self.kappa_network = MLP(input_dim=compression_network.out_size, output_dim=1, hidden_layers=hidden_layers)

        self.dropout = dropout

    def forward(self, pos, z, shapes, edge_index=None):
        compressed = self.compression_network(pos, z, shapes, edge_index)

        # Predict angle as vector (x, y) then normalize -> atan2
        mean_xy = self.angle_network(compressed)
        mean_xy = mean_xy / (mean_xy.norm(dim=-1, keepdim=True) + 1e-8)
        mu = torch.atan2(mean_xy[..., 1], mean_xy[..., 0])

        log_kappa = self.kappa_network(compressed)
        log_kappa = torch.clamp(log_kappa, min=-6.91, max=-4.61)
        kappa = torch.exp(log_kappa)
        return mu, kappa

    def loss(self, pos, z, shapes, target, edge_index=None, mask=None):
        """
        Full-batch loss with stochastic masking.
        target: Should be 2*phi (Spin-1 domain).
        """
        mu, kappa = self.forward(pos, z, shapes, edge_index)

        # Ensure shapes match
        if target.dim() > 1: target = target.view(-1)
        if mu.dim() > 1: mu = mu.view(-1)
        if kappa.dim() > 1: kappa = kappa.view(-1)

        # VMDN Log Prob
        dist_vonmises = VonMises(mu, kappa)
        log_prob = dist_vonmises.log_prob(target)

        # Apply Stochastic Mask (Subsampling)
        if mask is not None:
            log_prob = log_prob[mask]

        nll = -log_prob.mean()

        # Optional Regularization (disabled by default)
        if self.training and self.lambda_kappa > 0 and mask is not None:
            # Pairwise regularization logic here if needed
            pass

        return nll