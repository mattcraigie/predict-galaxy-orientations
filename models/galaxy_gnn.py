from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree


class GraphBuilder:
    @staticmethod
    def build_edges(pos: torch.Tensor, mode: str = "knn", k: int = 20) -> torch.Tensor:
        device = pos.device
        pos_np = pos.detach().cpu().numpy()

        if pos.dim() == 3 and pos.shape[0] == 1:
            pos_np = pos_np[0]

        num_points = pos_np.shape[0]

        if mode == "knn":
            k_eff = min(k, num_points - 1)
            tree = cKDTree(pos_np)
            _, idx = tree.query(pos_np, k=k_eff + 1)

            # Remove self-loops (index 0)
            src = idx[:, 1:].flatten()
            dst = np.repeat(np.arange(num_points), k_eff)

            edge_index = torch.stack([
                torch.tensor(src, dtype=torch.long),
                torch.tensor(dst, dtype=torch.long)
            ], dim=0).to(device)

            return edge_index
        else:
            raise NotImplementedError("Only kNN supported currently")


class SparseLocalFrameLayer(nn.Module):
    def __init__(self, scalar_dim: int, vector_dim: int, spin_symmetry: int = 2):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.spin_symmetry = spin_symmetry

        input_dim = (2 * scalar_dim) + vector_dim + 3

        self.message_mlp = nn.Sequential(
            nn.Linear(input_dim, scalar_dim + vector_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim + vector_dim, scalar_dim + vector_dim),
        )

    def forward(self, h_s, h_v, edge_index, pos, orientation):
        src, dst = edge_index[0], edge_index[1]

        # Geometry
        delta_pos = pos[src] - pos[dst]
        dist = torch.norm(delta_pos, dim=1, keepdim=True) + 1e-6
        phi_global = torch.atan2(delta_pos[:, 1], delta_pos[:, 0])

        alpha_i = orientation[dst].squeeze(-1)
        delta_phi = phi_global - alpha_i

        geo_feat = torch.cat([
            dist,
            torch.cos(self.spin_symmetry * delta_phi).unsqueeze(-1),
            torch.sin(self.spin_symmetry * delta_phi).unsqueeze(-1)
        ], dim=-1)

        # Vector Rotation
        beta_j = orientation[src].squeeze(-1)
        rot_angle = (beta_j - alpha_i) * self.spin_symmetry

        c, s = torch.cos(rot_angle), torch.sin(rot_angle)

        v_j = h_v[src]
        n_vecs = self.vector_dim // 2
        v_j = v_j.view(-1, n_vecs, 2)

        v_j_rot_x = v_j[..., 0] * c.unsqueeze(-1) - v_j[..., 1] * s.unsqueeze(-1)
        v_j_rot_y = v_j[..., 0] * s.unsqueeze(-1) + v_j[..., 1] * c.unsqueeze(-1)

        v_j_rot = torch.stack([v_j_rot_x, v_j_rot_y], dim=-1).view(-1, self.vector_dim)

        # Message Passing
        msg_input = torch.cat([h_s[src], h_s[dst], v_j_rot, geo_feat], dim=-1)
        raw_msg = self.message_mlp(msg_input)

        msg_s = raw_msg[:, :self.scalar_dim]
        msg_v = raw_msg[:, self.scalar_dim:]

        out_s = torch.zeros_like(h_s)
        out_v = torch.zeros_like(h_v)

        out_s.index_add_(0, dst, msg_s)
        out_v.index_add_(0, dst, msg_v)

        return h_s + out_s, h_v + out_v


class GalaxyLSSBackbone(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3, input_scalar_dim=1, input_vector_dim=2):
        super().__init__()
        self.s_dim = hidden_dim // 2
        self.v_dim = hidden_dim // 2

        self.enc_s = nn.Sequential(nn.Linear(input_scalar_dim, self.s_dim), nn.ReLU())
        self.enc_v = nn.Linear(input_vector_dim, self.v_dim, bias=False)

        self.layers = nn.ModuleList([
            SparseLocalFrameLayer(self.s_dim, self.v_dim, spin_symmetry=2)
            for _ in range(num_layers)
        ])

    def forward(self, pos, redshift, input_shapes, edge_index=None, k=20):
        if pos.dim() == 3: pos = pos.squeeze(0)
        if redshift.dim() == 3: redshift = redshift.squeeze(0)
        if input_shapes.dim() == 3: input_shapes = input_shapes.squeeze(0)

        if edge_index is None:
            edge_index = GraphBuilder.build_edges(pos, k=k)

        h_s = self.enc_s(redshift)
        h_v = self.enc_v(input_shapes)

        orientation = torch.zeros((pos.shape[0], 1), device=pos.device)

        for layer in self.layers:
            h_s, h_v = layer(h_s, h_v, edge_index, pos, orientation)

        return h_s, h_v


class GalaxyLSSWrapper(nn.Module):
    """
    Compression network using arctan2 to convert equivariant vectors to angles.
    """

    def __init__(self, backbone: GalaxyLSSBackbone):
        super().__init__()
        self.backbone = backbone

        # Scalar channels + 1 Angle channel per vector pair
        num_vectors = backbone.v_dim // 2
        self.out_size = backbone.s_dim + num_vectors

    def forward(self, pos, redshift, input_shapes, edge_index=None):
        h_s, h_v = self.backbone(pos, redshift, input_shapes, edge_index)

        # Reshape to Pairs [N, Pairs, 2]
        num_vectors = self.backbone.v_dim // 2
        v_vectors = h_v.view(-1, num_vectors, 2)

        # Convert to Angles [N, Pairs]
        # This angle represents 2*theta because of the spin-2 symmetry in the layers
        v_angles = torch.atan2(v_vectors[:, :, 1], v_vectors[:, :, 0] + 1e-5)

        # Fuse [N, s_dim + num_vectors]
        features = torch.cat([h_s, v_angles], dim=-1)

        return features