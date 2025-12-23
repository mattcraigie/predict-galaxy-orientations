import torch
import torch.nn as nn
from torch.distributions import VonMises
import numpy as np

from .basic_models import MLP
from .galaxy_gnn import GalaxyLSSBackbone, GalaxyLSSWrapper


def init_vmdn(model_config):
    """
    Initialize the Galaxy GNN + VMDN model.
    """
    hidden_dim = model_config.get("hidden_dim", 64)
    num_layers = model_config.get("num_layers", 3)

    input_scalar_dim = model_config.get("input_scalar_dim", 1)
    input_vector_dim = model_config.get("input_vector_dim", 2)

    vmdn_hidden_layers = model_config.get("vmdn_hidden_layers", [32, 32])
    vmdn_regularization = model_config.get("vmdn_regularization", 0.0)
    vmdn_dropout = model_config.get("vmdn_dropout", 0.0)

    gnn_backbone = GalaxyLSSBackbone(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        input_scalar_dim=input_scalar_dim,
        input_vector_dim=input_vector_dim
    )

    compression_model = GalaxyLSSWrapper(gnn_backbone)

    model = VMDN(
        compression_network=compression_model,
        hidden_layers=vmdn_hidden_layers,
        lambda_kappa=vmdn_regularization,
        dropout=vmdn_dropout
    )

    return model


class VMDN(nn.Module):
    """
    Von Mises Density Network (VMDN).
    Predicts mean (mu) and concentration (kappa) of a Von Mises distribution.
    """

    def __init__(self, compression_network, hidden_layers=None, lambda_kappa=0.0, dropout=0.0):
        super().__init__()
        self.compression_network = compression_network
        self.lambda_kappa = lambda_kappa

        if hidden_layers is None:
            hidden_layers = [32, 32]

        self.angle_network = MLP(input_dim=compression_network.out_size, output_dim=2, hidden_layers=hidden_layers)
        self.kappa_network = MLP(input_dim=compression_network.out_size, output_dim=1, hidden_layers=hidden_layers)

        self.dropout = dropout
        self.device = None

    def forward(self, *args):
        compressed = self.compression_network(*args)
        mean_xy = self.angle_network(compressed)
        mean_xy = mean_xy / (mean_xy.norm(dim=-1, keepdim=True) + 1e-8)
        mu = torch.atan2(mean_xy[..., 1], mean_xy[..., 0])

        log_kappa = self.kappa_network(compressed)
        log_kappa = torch.clamp(log_kappa, min=-10, max=3)
        kappa = torch.exp(log_kappa)
        return mu, kappa

    def loss(self, *args, target=None):
        mu, kappa = self.forward(*args)

        # Flatten target to match the flattened mu/kappa
        target = target.view(-1)

        phi_target = 2 * target
        dist_vonmises = VonMises(mu.squeeze(), kappa.squeeze())
        log_prob = dist_vonmises.log_prob(phi_target)

        if self.training:
            node_mask = torch.rand(log_prob.size(0), device=log_prob.device) > self.dropout
        else:
            node_mask = torch.ones(log_prob.size(0), dtype=torch.bool, device=log_prob.device)

        masked_log_prob = log_prob[node_mask]
        nll = -masked_log_prob.mean()
        total_loss = nll

        if self.training and self.lambda_kappa > 0:
            N = mu.shape[0]
            if N > 1:
                num_pairs = 1000
                idx = torch.randint(0, N, (num_pairs, 2), device=mu.device)

                mask_idx = idx[:, 0] != idx[:, 1]
                idx = idx[mask_idx]

                a1 = mu[idx[:, 0]]
                a2 = mu[idx[:, 1]]

                diff = torch.remainder(a1 - a2, 2 * torch.pi)
                diff = torch.where(diff > torch.pi, 2 * torch.pi - diff, diff)

                penalty = torch.exp(- (diff ** 2) / (2 * 0.5 ** 2))
                pairwise_penalty = penalty.mean()
                total_loss += self.lambda_kappa * pairwise_penalty

        return total_loss

    def sample(self, *args, n_samples=1):
        mu, kappa = self.forward(*args)
        dist_vonmises = VonMises(mu, kappa)
        samples = dist_vonmises.sample((n_samples,))
        return samples[0]

    def to(self, device):
        super().to(device)
        self.device = device
        return self
