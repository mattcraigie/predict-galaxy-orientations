import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree


class GalaxyDataset(Dataset):
    def __init__(self, csv_file, num_neighbors=20, mode='train', split_ratios=(0.8, 0.1, 0.1), seed=42):
        super().__init__()
        self.num_neighbors = num_neighbors

        # 1. Load and Clean Data
        df = pd.read_csv(csv_file)
        df = df.dropna(subset=["ra", "dec", "mean_z", "phi_deg"]).reset_index(drop=True)

        self.ra = df['ra'].values.astype(np.float32)
        self.dec = df['dec'].values.astype(np.float32)
        self.z = df['mean_z'].values.astype(np.float32)
        self.phi_deg = df['phi_deg'].values.astype(np.float32)

        # 2. Build Tree
        self.coords = np.stack([self.ra, self.dec], axis=1)
        self.tree = cKDTree(self.coords)

        # 3. Splits
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
        global_idx = self.indices[idx]
        center_point = self.coords[global_idx]

        # Query Neighbors
        _, neighbor_indices = self.tree.query(center_point, k=self.num_neighbors)

        # Gather Data
        patch_coords = self.coords[neighbor_indices] - center_point  # Normalize positions
        patch_z = self.z[neighbor_indices][:, None]
        patch_phi_deg = self.phi_deg[neighbor_indices]

        # Convert targets
        patch_phi_rad = np.deg2rad(patch_phi_deg)

        # Create Input Shapes (Spin-2 Vector)
        e1 = np.cos(2 * patch_phi_rad)
        e2 = np.sin(2 * patch_phi_rad)
        patch_input_shapes = np.stack([e1, e2], axis=1)

        return {
            'pos': torch.tensor(patch_coords, dtype=torch.float32),
            'redshift': torch.tensor(patch_z, dtype=torch.float32),
            'input_shapes': torch.tensor(patch_input_shapes, dtype=torch.float32),
            'target_phi': torch.tensor(patch_phi_rad, dtype=torch.float32)  # Keep targets in radians
        }

        return sample
