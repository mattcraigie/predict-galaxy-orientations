import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree


class GalaxyDataset(Dataset):
    def __init__(self, csv_file, num_neighbors=20, mode='train', split_ratios=(0.8, 0.1, 0.1), seed=42):
        """
        Args:
            csv_file (str): Path to the CSV catalog.
            num_neighbors (int): Number of neighbors to include in each patch (graph size).
            mode (str): 'train', 'val', or 'test'.
            split_ratios (tuple): Ratios for splitting the data.
            seed (int): Random seed for reproducibility.
        """
        super().__init__()
        self.num_neighbors = num_neighbors

        # 1. Load Data
        df = pd.read_csv(csv_file)

        # 2. Extract Columns
        # Converting RA/Dec to simple float32 arrays
        self.ra = df['ra'].values.astype(np.float32)
        self.dec = df['dec'].values.astype(np.float32)
        self.z = df['mean_z'].values.astype(np.float32)
        self.phi_deg = df['phi_deg'].values.astype(np.float32)

        # 3. Create Spatial Index (KDTree) for fast neighbor lookup
        # We stack RA/Dec into a [Total_Galaxies, 2] array
        self.coords = np.stack([self.ra, self.dec], axis=1)
        self.tree = cKDTree(self.coords)

        # 4. Handle Train/Val/Test Split
        total_len = len(df)
        indices = np.arange(total_len)

        np.random.seed(seed)
        np.random.shuffle(indices)

        n_train = int(total_len * split_ratios[0])
        n_val = int(total_len * split_ratios[1])

        if mode == 'train':
            self.indices = indices[:n_train]
        elif mode == 'val':
            self.indices = indices[n_train:n_train + n_val]
        else:
            self.indices = indices[n_train + n_val:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the global index from our split subset
        global_idx = self.indices[idx]

        # 1. Find neighbors for this galaxy
        # k = num_neighbors. We query k+1 because the point itself is included (dist=0)
        center_point = self.coords[global_idx]
        _, neighbor_indices = self.tree.query(center_point, k=self.num_neighbors)

        # 2. Gather Data for this patch
        # Shape: [N, 2]
        patch_ra = self.ra[neighbor_indices]
        patch_dec = self.dec[neighbor_indices]
        patch_coords = np.stack([patch_ra, patch_dec], axis=1)

        # Normalize coordinates relative to the central galaxy
        # (This helps the neural net significantly)
        patch_coords = patch_coords - center_point

        # Shape: [N, 1]
        patch_z = self.z[neighbor_indices][:, None]

        # Shape: [N]
        patch_phi_deg = self.phi_deg[neighbor_indices]

        # 3. Convert Phi to Spin-2 Vector and Radians
        # Input to GNN needs to be vector components (e1, e2)
        patch_phi_rad = np.deg2rad(patch_phi_deg)

        # Spin-2 projection: cos(2*phi), sin(2*phi)
        e1 = np.cos(2 * patch_phi_rad)
        e2 = np.sin(2 * patch_phi_rad)
        patch_input_shapes = np.stack([e1, e2], axis=1)  # [N, 2]

        # 4. Prepare Tensor Output
        # The central node is usually the first one returned by KDTree (index 0 in this patch)
        # We return the whole patch. The loss function will mask out training
        # or we rely on the GNN not self-looping.

        sample = {
            'pos': torch.tensor(patch_coords, dtype=torch.float32),
            'redshift': torch.tensor(patch_z, dtype=torch.float32),
            'input_shapes': torch.tensor(patch_input_shapes, dtype=torch.float32),
            'target_phi': torch.tensor(patch_phi_rad, dtype=torch.float32)  # Keep targets in radians
        }

        return sample