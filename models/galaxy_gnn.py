from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:  # Optional dependency for graph construction
    from scipy.spatial import Delaunay, cKDTree
except Exception:  # pragma: no cover
    Delaunay = None
    cKDTree = None


class GraphBuilder:
    """Helper to construct edge indices from positions on the fly."""

    @staticmethod
    def build_edges(pos: torch.Tensor, mode: str = "knn", k: int = 5) -> torch.Tensor:
        """
        Build sparse edge indices for a batch of point clouds.
        """
        if Delaunay is None or cKDTree is None:
            raise ImportError("GraphBuilder requires scipy to be installed.")

        device = pos.device
        pos_np = pos.detach().cpu().numpy()

        if pos.dim() == 2:
            pos_np = [pos_np]

        all_edges_src = []
        all_edges_dst = []
        offset = 0

        for p in pos_np:
            num_points = p.shape[0]
            if mode == "knn":
                tree = cKDTree(p)
                _, idx = tree.query(p, k=k + 1)
                # idx[:, 0] is the point itself, so we take [:, 1:]
                src = idx[:, 1:].flatten()
                dst = np.repeat(np.arange(num_points), k)
            elif mode == "delaunay":
                tri = Delaunay(p)
                indices = tri.simplices
                edges = np.concatenate(
                    [indices[:, [0, 1]], indices[:, [1, 2]], indices[:, [2, 0]]], axis=0
                )
                edges_rev = edges[:, ::-1]
                full_edges = np.concatenate([edges, edges_rev], axis=0)
                full_edges = np.unique(full_edges, axis=0)
                src = full_edges[:, 1]
                dst = full_edges[:, 0]
            else:
                raise ValueError(f"Unknown graph mode: {mode}")

            all_edges_src.append(torch.tensor(src, dtype=torch.long) + offset)
            all_edges_dst.append(torch.tensor(dst, dtype=torch.long) + offset)
            offset += num_points

        edge_index = torch.stack([torch.cat(all_edges_src), torch.cat(all_edges_dst)], dim=0).to(device)
        return edge_index


class SparseLocalFrameLayer(nn.Module):
    """
    The core GNN layer that handles rotation equivariance.
    """

    def __init__(self, scalar_dim: int, vector_dim: int, spin_symmetry: int = 2):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        self.spin_symmetry = spin_symmetry

        geo_dim = 3
        input_dim = (2 * scalar_dim) + vector_dim + geo_dim

        self.message_mlp = nn.Sequential(
            nn.Linear(input_dim, scalar_dim + vector_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim + vector_dim, scalar_dim + vector_dim),
        )

    def forward(
            self,
            h_scalar: torch.Tensor,
            h_vector: torch.Tensor,
            edge_index: torch.Tensor,
            pos: torch.Tensor,
            orientation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index[0], edge_index[1]

        delta_pos = pos[src] - pos[dst]

        dist = torch.norm(delta_pos, dim=1, keepdim=True) + 1e-6
        phi_global = torch.atan2(delta_pos[:, 1], delta_pos[:, 0])

        alpha_i = orientation[dst].squeeze(-1)
        delta_phi = phi_global - alpha_i

        geo_feat = torch.cat(
            [
                dist,
                torch.cos(self.spin_symmetry * delta_phi).unsqueeze(-1),
                torch.sin(self.spin_symmetry * delta_phi).unsqueeze(-1),
            ],
            dim=-1,
        )

        beta_j = orientation[src].squeeze(-1)
        rot_angle = (beta_j - alpha_i) * self.spin_symmetry

        c = torch.cos(rot_angle)
        s = torch.sin(rot_angle)

        row1 = torch.stack([c, -s], dim=-1)
        row2 = torch.stack([s, c], dim=-1)
        R = torch.stack([row1, row2], dim=-2)

        v_j = h_vector[src]
        num_vecs = self.vector_dim // 2
        v_j_reshaped = v_j.view(-1, num_vecs, 2).unsqueeze(-1)
        v_j_rot = torch.matmul(R.unsqueeze(1), v_j_reshaped)
        v_j_rot = v_j_rot.view(-1, self.vector_dim)

        msg_input = torch.cat([h_scalar[src], h_scalar[dst], v_j_rot, geo_feat], dim=-1)
        raw_msg = self.message_mlp(msg_input)

        out_scalar = torch.zeros_like(h_scalar)
        out_vector = torch.zeros_like(h_vector)

        msg_scalar = raw_msg[:, : self.scalar_dim]
        msg_vector = raw_msg[:, self.scalar_dim:]

        out_scalar.index_add_(0, dst, msg_scalar)
        out_vector.index_add_(0, dst, msg_vector)

        return h_scalar + out_scalar, h_vector + out_vector


class GalaxyLSSBackbone(nn.Module):
    """
    Equivariant GNN Backbone for Galaxy Shape prediction.
    Processes a graph of galaxies and returns per-galaxy latent features.
    """

    def __init__(
            self,
            hidden_dim: int = 64,
            num_layers: int = 3,
            input_scalar_dim: int = 1,
            input_vector_dim: int = 2,
    ):
        super().__init__()

        if hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even to split between scalar and vector channels.")

        self.s_dim = hidden_dim // 2
        self.v_dim = hidden_dim // 2

        # Encoders
        # Redshift (scalar) -> scalar latent
        self.enc_s = nn.Sequential(
            nn.Linear(input_scalar_dim, self.s_dim),
            nn.ReLU(),
            nn.Linear(self.s_dim, self.s_dim)
        )

        # Shape/Shear (vector) -> vector latent
        # Bias=False preserves rotation equivariance of zero vector
        self.enc_v = nn.Linear(input_vector_dim, self.v_dim, bias=False)

        # Message Passing Layers
        self.layers = nn.ModuleList([
            SparseLocalFrameLayer(
                scalar_dim=self.s_dim,
                vector_dim=self.v_dim,
                spin_symmetry=2  # Spin-2 symmetry for galaxy shear
            )
            for _ in range(num_layers)
        ])

    def forward(
            self,
            pos: torch.Tensor,
            redshift: torch.Tensor,
            input_shapes: torch.Tensor,
            graph_mode: str = "knn",
            k: int = 20
    ) -> tuple[torch.Tensor, torch.Tensor]:

        B, N, _ = pos.shape
        device = pos.device

        # Flatten batch dims for graph processing: [B*N, Features]
        pos_flat = pos.view(-1, 2)
        z_flat = redshift.view(-1, redshift.shape[-1])
        vec_flat = input_shapes.view(-1, input_shapes.shape[-1])

        # Default orientation 0 (Survey Frame)
        orientation_flat = torch.zeros((B * N, 1), device=device)

        # Build Graph (connects neighbors across the batch)
        edge_index = GraphBuilder.build_edges(pos, mode=graph_mode, k=k)

        # Encode Inputs
        h_s = self.enc_s(z_flat)
        h_v = self.enc_v(vec_flat)

        # Run Message Passing
        for layer in self.layers:
            h_s, h_v = layer(
                h_scalar=h_s,
                h_vector=h_v,
                edge_index=edge_index,
                pos=pos_flat,
                orientation=orientation_flat
            )

        # We return the flattened features directly [B*N, Dim]
        # This matches what the VMDN expects (Total_Nodes)
        return h_s, h_v


class GalaxyLSSWrapper(nn.Module):
    """
    Wraps the Equivariant Backbone to work with VMDN.
    Flattens the Scalar and Vector outputs into a single feature vector per galaxy.
    """

    def __init__(self, backbone: GalaxyLSSBackbone):
        super().__init__()
        self.backbone = backbone
        # Output size is sum of scalar and vector latent dims
        self.out_size = backbone.s_dim + backbone.v_dim

    def forward(self, pos, redshift, input_shapes):
        # 1. Run GNN
        # Returns [Total_Nodes, s_dim] and [Total_Nodes, v_dim]
        h_s, h_v = self.backbone(pos, redshift, input_shapes)

        # 2. Fuse Features
        # Concatenate scalar and vector latents
        features = torch.cat([h_s, h_v], dim=-1)

        # 3. Flatten (Redundant if backbone returns flattened, but safe)
        return features.view(-1, self.out_size)