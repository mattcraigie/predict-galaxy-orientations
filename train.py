import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.vmdn import init_vmdn
from models.galaxy_gnn import GraphBuilder


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
    # (Simple option: subtract mean. For spherical, RA/Dec directly is okay for local,
    # but projecting to cartesian is better. Keeping it simple for now as per previous code)
    pos = np.stack([ra, dec], axis=1)
    pos -= pos.mean(axis=0)

    # Shape inputs (Spin-2)
    e1 = np.cos(2 * phi_rad)
    e2 = np.sin(2 * phi_rad)
    shapes = np.stack([e1, e2], axis=1)

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
    # Rest is test, unused in loop

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
        # 1. Forward pass on EVERYTHING (Propagation sees full context)
        # Note: We pass the FULL edge_index.
        # 2. Masking: Select random subset of TRAIN nodes for loss

        # Generate mask for this step:
        # Take the fixed train_idx, shuffle it, take top 25% (or config ratio)
        batch_size = int(len(train_idx) * config.get('subsample_ratio', 0.25))
        step_indices = train_idx[torch.randperm(len(train_idx))[:batch_size]]

        # Create boolean mask for loss function (optional, but clean)
        # Or just slice the inputs? No, we must forward full inputs.
        # We slice the targets and the output.

        # VMDN.loss internally calls forward(all), then computes log_prob(all), then slices.
        # To save memory, we can slice `log_prob` inside loss.
        # Let's pass a boolean mask of size [N] to the loss function.
        step_mask = torch.zeros(N, dtype=torch.bool, device=config['device'])
        step_mask[step_indices] = True

        loss = model.loss(pos, z, shapes, target, edge_index=edge_index, mask=step_mask)

        loss.backward()
        optimizer.step()

        # --- VALIDATION ---
        # Evaluate on ALL validation nodes (no subsampling)
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
        "subsample_ratio": 0.25,  # <--- The stochasticity param
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "log_dir": "runs/full_batch_run"
    }
    train(conf)