import torch
import torch.nn as nn


class GalaxyLSSBackbone(nn.Module):
    """
    Lightweight backbone that embeds per-galaxy features.
    """

    def __init__(self, hidden_dim=64, num_layers=3, input_scalar_dim=1, input_vector_dim=2):
        super().__init__()
        input_dim = 2 + input_scalar_dim + input_vector_dim
        layers = []
        in_dim = input_dim

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, hidden_dim))
        self.net = nn.Sequential(*layers)
        self.out_size = hidden_dim

    def forward(self, pos, redshift, input_shapes):
        features = torch.cat([pos, redshift, input_shapes], dim=-1)
        return self.net(features)


class GalaxyLSSWrapper(nn.Module):
    """
    Pool node embeddings into a single patch embedding.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.out_size = backbone.out_size

    def forward(self, pos, redshift, input_shapes):
        node_embeddings = self.backbone(pos, redshift, input_shapes)
        return node_embeddings.mean(dim=0, keepdim=True)
