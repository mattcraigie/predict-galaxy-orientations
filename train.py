import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scipy.spatial import cKDTree

from models.vmdn import init_vmdn
from models.galaxy_gnn import GraphBuilder


# --- PLOTTING HELPER ---
def create_density_figure(true, pred, title):
    """
    Creates a Matplotlib Figure for P(pred | true).
    Returns the figure object for TensorBoard logging.
    """
    # Normalize to [0, 2pi]
    true = np.remainder(true, 2 * np.pi)
    pred = np.remainder(pred, 2 * np.pi)

    bins = 64
    range_lim = [[0, 2 * np.pi], [0, 2 * np.pi]]

    # Compute 2D Histogram
    H, xedges, yedges = np.histogram2d(true, pred, bins=bins, range=range_lim)

    # Normalize rows to get P(pred | true)
    row_sums = H.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    H_norm = H / row_sums

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(H_norm.T, origin='lower', extent=[0, 2 * np.pi, 0, 2 * np.pi],
                   aspect='auto', cmap='viridis', vmin=0, vmax=np.max(H_norm) * 0.8)

    ax.set_title(title)
    ax.set_xlabel('True Angle')
    ax.set_ylabel('Predicted Angle')

    # Diagonal reference line
    ax.plot([0, 2 * np.pi], [0, 2 * np.pi], 'r--', alpha=0.5)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    return fig


# --- SIGNAL HELPER ---
def inject_identity_signal(pos, old_targets):
    print("!!! INJECTING SIGNAL: IDENTITY MAPPING !!!")
    new_targets = np.random.uniform(0, 2 * np.pi, size=len(pos)).astype(np.float32)
    e1 = np.cos(2 * new_targets)
    e2 = np.sin(2 * new_targets)
    new_inputs = np.stack([e1, e2], axis=1).astype(np.float32)
    return new_inputs, new_targets


def load_full_data(config):
    print(f"Loading full catalog from {config['csv_path']}...")
    df = pd.read_csv(config['csv_path'])
    df = df.dropna(subset=['ra', 'dec', 'mean_z', 'phi_deg'])

    ra = df['ra'].values.astype(np.float32)
    dec = df['dec'].values.astype(np.float32)
    z = df['mean_z'].values.astype(np.float32)
    phi_rad = np.deg2rad(df['phi_deg'].values.astype(np.float32))

    pos = np.stack([ra, dec], axis=1)
    pos -= pos.mean(axis=0)

    # Signal Injection
    if config.get('inject_signal', False):
        shapes, phi_rad = inject_identity_signal(pos, phi_rad)
    else:
        e1 = np.cos(2 * phi_rad)
        e2 = np.sin(2 * phi_rad)
        shapes = np.stack([e1, e2], axis=1)

    device = config['device']
    t_pos = torch.tensor(pos, device=device)
    t_z = torch.tensor(z[:, None], device=device)
    t_shapes = torch.tensor(shapes, device=device)
    t_target = torch.tensor(phi_rad, device=device)

    print(f"Building graph (k={config['num_neighbors']})...")
    edge_index = GraphBuilder.build_edges(t_pos, k=config['num_neighbors'])

    return t_pos, t_z, t_shapes, t_target, edge_index


def train(config):
    # 1. Load Data
    pos, z, shapes, target, edge_index = load_full_data(config)
    N = pos.shape[0]

    # 2. Split Indices
    indices = torch.randperm(N)
    n_train = int(N * 0.8)
    n_val = int(N * 0.1)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # 3. Setup
    model = init_vmdn(config).to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    writer = SummaryWriter(config['log_dir'])
    Path(config['log_dir']).mkdir(parents=True, exist_ok=True)

    best_loss = float('inf')
    patience_counter = 0

    print("Starting Training...")

    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()

        # Stochastic Train Step
        batch_size = int(len(train_idx) * config.get('subsample_ratio', 0.25))
        step_indices = train_idx[torch.randperm(len(train_idx))[:batch_size]]
        step_mask = torch.zeros(N, dtype=torch.bool, device=config['device'])
        step_mask[step_indices] = True

        loss = model.loss(pos, z, shapes, target, edge_index=edge_index, mask=step_mask)
        loss.backward()
        optimizer.step()

        # Validation Step
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_mask = torch.zeros(N, dtype=torch.bool, device=config['device'])
                val_mask[val_idx] = True
                val_loss = model.loss(pos, z, shapes, target, edge_index=edge_index, mask=val_mask)

                print(f"Epoch {epoch}: Train Loss {loss.item():.4f} | Val Loss {val_loss.item():.4f}")
                writer.add_scalar('Loss/Train', loss.item(), epoch)
                writer.add_scalar('Loss/Val', val_loss.item(), epoch)

                # Checkpoint
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), f"{config['log_dir']}/best_model.pth")
                else:
                    patience_counter += 1
                    if patience_counter >= config['patience']:
                        print("Early stopping triggered.")
                        break

        # --- TENSORBOARD PLOTTING (Every 20 epochs) ---
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                mu_all, _ = model(pos, z, shapes, edge_index)
                mu_np = mu_all.cpu().numpy().flatten()
                target_np = target.cpu().numpy().flatten()

                # Plot Train Subset
                train_sub = train_idx[:10000].cpu().numpy()  # Subsample for speed
                fig_train = create_density_figure(
                    target_np[train_sub], mu_np[train_sub],
                    f"Train Density (Epoch {epoch})"
                )
                writer.add_figure("Density/Train", fig_train, epoch)

                # Plot Val Subset
                val_sub = val_idx.cpu().numpy()
                fig_val = create_density_figure(
                    target_np[val_sub], mu_np[val_sub],
                    f"Val Density (Epoch {epoch})"
                )
                writer.add_figure("Density/Val", fig_val, epoch)

    print("Training complete. Check TensorBoard for plots.")
    writer.close()


if __name__ == "__main__":
    conf = {
        "csv_path": "data/des_metacal_angles_minimal.csv",
        "lr": 5e-3,
        "epochs": 1000,
        "patience": 50,
        "num_neighbors": 20,
        "hidden_dim": 64,
        "subsample_ratio": 0.25,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "log_dir": "runs/identity_check_tb",
        "inject_signal": True
    }
    train(conf)