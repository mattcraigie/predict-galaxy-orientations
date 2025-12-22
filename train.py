import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scipy.spatial import cKDTree  # Needed for the mock signal

from models.vmdn import init_vmdn
from models.galaxy_gnn import GraphBuilder


# --- NEW HELPER FUNCTION ---
def inject_mock_signal(pos, old_targets):
    """
    Overwrites targets so every galaxy points directly at its nearest neighbor.
    This creates a perfect geometric correlation that the GNN must learn from positions.
    """
    print("!!! INJECTING MOCK SIGNAL: RADIAL ALIGNMENT !!!")
    print("Targets will align to nearest neighbors. Inputs will be randomized.")

    # 1. Find nearest neighbor for every point
    # k=2 because the first result is always the point itself (dist=0)
    tree = cKDTree(pos)
    _, indices = tree.query(pos, k=2)

    # indices[:, 0] is self, indices[:, 1] is the neighbor
    neighbor_idx = indices[:, 1]

    # 2. Calculate Vector to Neighbor
    # Shape: [N, 2]
    vec_to_neighbor = pos[neighbor_idx] - pos

    # 3. Calculate Angle (radians)
    # This is the "perfect" physical alignment signal
    new_targets = np.arctan2(vec_to_neighbor[:, 1], vec_to_neighbor[:, 0])

    # 4. Randomize Inputs
    # We scramble the input shapes so the model CANNOT just pass the input
    # through to the output. It MUST look at the neighbor's position.
    random_angles = np.random.uniform(0, 2 * np.pi, size=len(pos))
    new_e1 = np.cos(2 * random_angles)
    new_e2 = np.sin(2 * random_angles)
    new_inputs = np.stack([new_e1, new_e2], axis=1)

    return new_inputs.astype(np.float32), new_targets.astype(np.float32)


# ---------------------------

def load_full_data(config):
    """Loads ALL data into GPU tensors and pre-computes the graph."""
    print(f"Loading full catalog from {config['csv_path']}...")
    df = pd.read_csv(config['csv_path'])

    # 1. Clean
    df = df.dropna(subset=['ra', 'dec', 'mean_z', 'phi_deg'])

    # 2. Convert to Numpy
    ra = df['ra'].values.astype(np.float32)
    dec = df['dec'].values.astype(np.float32)
    z = df['mean_z'].values.astype(np.float32)
    phi_rad = np.deg2rad(df['phi_deg'].values.astype(np.float32))

    # 3. Create Features
    # Position: Normalize roughly so KNN makes sense
    pos = np.stack([ra, dec], axis=1)
    pos -= pos.mean(axis=0)

    # Shape inputs (Spin-2)
    e1 = np.cos(2 * phi_rad)
    e2 = np.sin(2 * phi_rad)
    shapes = np.stack([e1, e2], axis=1)

    # --- SIGNAL INJECTION BLOCK ---
    if config.get('inject_signal', False):
        shapes, phi_rad = inject_mock_signal(pos, phi_rad)
    # ------------------------------

    # 4. To Tensor (Move to GPU immediately)
    device = config['device']
    t_pos = torch.tensor(pos, device=device)
    t_z = torch.tensor(z[:, None], device=device)
    t_shapes = torch.tensor(shapes, device=device)
    t_target = torch.tensor(phi_rad, device=device)

    # 5. Pre-compute Graph (Static)
    print(f"Building static graph for {len(df)} galaxies (k={config['num_neighbors']})...")
    edge_index = GraphBuilder.build_edges(t_pos, k=config['num_neighbors'])
    print(f"Graph built with {edge_index.shape[1]} edges.")

    return t_pos, t_z, t_shapes, t_target, edge_index


def train(config):
    # 1. Load Everything
    pos, z, shapes, target, edge_index = load_full_data(config)
    N = pos.shape[0]

    # 2. Split (Indices)
    indices = torch.randperm(N)
    n_train = int(N * 0.8)
    n_val = int(N * 0.1)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]

    print(f"Train nodes: {len(train_idx)}, Val nodes: {len(val_idx)}")

    # 3. Model
    model = init_vmdn(config).to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    writer = SummaryWriter(config['log_dir'])

    best_loss = float('inf')
    patience = config['patience']
    patience_counter = 0

    print("Starting Full-Batch Training...")

    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()

        # --- STOCHASTIC SUBSAMPLING ---
        batch_size = int(len(train_idx) * config.get('subsample_ratio', 0.25))
        step_indices = train_idx[torch.randperm(len(train_idx))[:batch_size]]

        step_mask = torch.zeros(N, dtype=torch.bool, device=config['device'])
        step_mask[step_indices] = True

        loss = model.loss(pos, z, shapes, target, edge_index=edge_index, mask=step_mask)

        loss.backward()
        optimizer.step()

        # --- VALIDATION ---
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_mask = torch.zeros(N, dtype=torch.bool, device=config['device'])
                val_mask[val_idx] = True

                val_loss = model.loss(pos, z, shapes, target, edge_index=edge_index, mask=val_mask)

                print(f"Epoch {epoch}: Train Loss {loss.item():.4f} | Val Loss {val_loss.item():.4f}")
                writer.add_scalar('Loss/Train', loss.item(), epoch)
                writer.add_scalar('Loss/Val', val_loss.item(), epoch)

                # Early Stop Check
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), f"{config['log_dir']}/best_model.pth")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping.")
                        break


if __name__ == "__main__":
    conf = {
        "csv_path": "data/des_metacal_angles_minimal.csv",
        "lr": 1e-3,
        "epochs": 1000,
        "patience": 20,
        "num_neighbors": 20,
        "hidden_dim": 64,
        "subsample_ratio": 0.25,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "log_dir": "runs/mock_signal_experiment",  # Changed log dir

        # --- ENABLE SIGNAL INJECTION HERE ---
        "inject_signal": True
    }
    train(conf)