import torch
import torch.nn as nn
from torch.distributions import VonMises
import numpy as np
from .basic_models import MLP
from .galaxy_gnn import GalaxyLSSBackbone, GalaxyLSSWrapper


def init_vmdn(config):
    # Config Unpacking
    hidden_dim = config.get("hidden_dim", 64)
    backbone = GalaxyLSSBackbone(hidden_dim=hidden_dim, num_layers=config.get("num_layers", 3))
    wrapper = GalaxyLSSWrapper(backbone)
    return VMDN(wrapper)


class VMDN(nn.Module):
    def __init__(self, compression_network, hidden_layers=[32, 32], dropout=0.0):
        super().__init__()
        self.compression_network = compression_network
        self.angle_net = MLP(compression_network.out_size, 1, hidden_layers)
        self.kappa_net = MLP(compression_network.out_size, 1, hidden_layers)
        self.dropout = dropout

    def forward(self, pos, z, shapes, edge_index=None):
        # Returns [N, Features]
        feats = self.compression_network(pos, z, shapes, edge_index)

        mu = self.angle_net(feats) % (2 * np.pi)
        log_kappa = torch.clamp(self.kappa_net(feats), -10, 5)
        kappa = torch.exp(log_kappa)
        return mu, kappa

    def loss(self, pos, z, shapes, target, edge_index=None, mask=None):
        """
        Computes loss with optional stochastic masking.
        mask: Bool Tensor [N], True = use this node for loss.
        """
        mu, kappa = self.forward(pos, z, shapes, edge_index)

        # Targets are [N]
        if target.dim() > 1: target = target.squeeze()
        if mu.dim() > 1: mu = mu.squeeze()
        if kappa.dim() > 1: kappa = kappa.squeeze()

        dist = VonMises(mu, kappa)
        log_prob = dist.log_prob(target)

        # --- Stochastic Subsampling ---
        if mask is not None:
            log_prob = log_prob[mask]
        # ------------------------------

        return -log_prob.mean()