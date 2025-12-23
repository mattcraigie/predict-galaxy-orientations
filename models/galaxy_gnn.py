import torch
import torch.nn as nn


class GalaxyLSSBackbone(nn.Module):
    """
    Lightweight backbone that embeds per-galaxy features.
    """

    def __init__(self, hidden_dim=64, num_layers=3, input_scalar_dim=1, input_vector_dim=2):
        super().__init__()
        if input_vector_dim % 2 != 0:
            raise ValueError("input_vector_dim must be even for spin-2 blocks.")

        input_dim = input_scalar_dim + 4
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
        eps = 1e-8
        num_nodes = pos.size(0)

        delta = pos.unsqueeze(0) - pos.unsqueeze(1)
        dist = torch.sqrt((delta ** 2).sum(dim=-1) + eps)
        weights = 1.0 / (dist + eps)

        self_mask = torch.eye(num_nodes, device=pos.device, dtype=torch.bool)
        weights = weights.masked_fill(self_mask, 0.0)

        cos2_theta = input_shapes[:, 0]
        sin2_theta = input_shapes[:, 1]

        sum_cos = (weights * cos2_theta[:, None]).sum(dim=0)
        sum_sin = (weights * sin2_theta[:, None]).sum(dim=0)

        alpha = 0.5 * torch.atan2(sum_sin, sum_cos)
        alpha = torch.where(
            (sum_cos**2 + sum_sin**2) < eps,
            torch.zeros_like(alpha),
            alpha,
        )

        phi_global = torch.atan2(delta[..., 1], delta[..., 0])
        delta_phi = phi_global - alpha[None, :]
        rel_cos = torch.cos(2 * delta_phi)
        rel_sin = torch.sin(2 * delta_phi)

        weight_sum = weights.sum(dim=0).clamp_min(eps)
        rel_cos = (weights * rel_cos).sum(dim=0) / weight_sum
        rel_sin = (weights * rel_sin).sum(dim=0) / weight_sum

        rot_angle = 2 * (alpha[:, None] - alpha[None, :])
        rot_cos = torch.cos(rot_angle)
        rot_sin = torch.sin(rot_angle)

        cos2_src = cos2_theta[:, None]
        sin2_src = sin2_theta[:, None]
        rot_cos2 = cos2_src * rot_cos - sin2_src * rot_sin
        rot_sin2 = cos2_src * rot_sin + sin2_src * rot_cos

        rot_cos2 = (weights * rot_cos2).sum(dim=0) / weight_sum
        rot_sin2 = (weights * rot_sin2).sum(dim=0) / weight_sum

        features = torch.cat(
            [
                redshift,
                rot_cos2[:, None],
                rot_sin2[:, None],
                rel_cos[:, None],
                rel_sin[:, None],
            ],
            dim=-1,
        )
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
