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
from datetime import datetime

from models.vmdn import init_vmdn
from models.galaxy_gnn import GraphBuilder


# --- PLOTTING HELPER ---
def create_density_figure(true, pred, title):
    """Plots P(pred | true) on the 2*phi domain (0 to 2pi)."""
    true = np.remainder(true, 2 * np.pi)
    pred = np.remainder(pred, 2 * np.pi)

    bins = 64
    range_lim = [[0, 2 * np.pi], [0, 2 * np.pi]]

    H, xedges, yedges = np.histogram2d(true, pred, bins=bins, range=range_lim)

    row_sums = H.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    H_norm = H / row_sums

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(H_norm.T, origin='lower', extent=[0, 2 * np.pi, 0, 2 * np.pi],
                   aspect='auto', cmap='viridis', vmin=0, vmax=np.max(H_norm) * 0.8)

    ax.set_title(title)
    ax.set_xlabel('True Double Angle (2*phi)')
    ax.set_ylabel('Predicted Double Angle (2*phi)')

    ax.plot([0, 2 * np.pi], [0, 2 * np.pi], 'r--', alpha=0.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def inject_identity_signal(pos, N):
    print("!!! INJECTING SIGNAL: IDENTITY MAPPING !!!")
    # Generate random targets in 0-2pi (representing 2*phi)
    new_targets = np.random.uniform(0, 2 * np.pi, size=N).astype(np.float32)

    # Input is cos/sin of this target
    e1 = np.cos(new_targets)  # Note: new_targets is already 2*phi
    e2 = np.sin(new_targets)
    new_inputs = np.stack([e1, e2], axis=1).astype(np.float32)

    return new_inputs, new_targets


def load_full_data(config):
    print(f"Loading full catalog from {config['csv_path']}...")
    df = pd.read_csv(config['csv_path'])
    df = df.dropna(subset=['ra', 'dec', 'mean_z', 'phi_deg'])

    ra = df['ra'].values.astype(np.float32)
    dec = df['dec'].values.astype(np.float32)
    z = df['mean_z'].values.astype(np.float32)

    # --- CRITICAL FIX: USE DOUBLE ANGLES ---
    # Galaxy shapes are Spin-2 (180 deg symmetry).
    # VMDN models Spin-1 (360 deg symmetry).
    # We must train on 2*phi so the topology matches.
    phi_rad = np.deg2rad(df['phi_deg'].values.astype(np.float32))
    double_phi = 2.0 * phi_rad  # [0, 2pi]
    # ---------------------------------------

    pos = np.stack([ra, dec], axis=1)
    pos -= pos.mean(axis=0)

    if config.get('inject_signal', False):
        shapes, target_angles = inject_identity_signal(pos, len(pos))
    else:
        # Standard Spin-2 Input
        e1 = np.cos(double_phi)
        e2 = np.sin(double_phi)
        shapes = np.stack([e1, e2], axis=1)
        target_angles = double_phi

    device = config['device']
    t_pos = torch.tensor(pos, device=device)
    t_z = torch.tensor(z[:, None], device=device)
    t_shapes = torch.tensor(shapes, device=device)
    t_target = torch.tensor(target_angles, device=device)

    print(f"Building graph (k={config['num_neighbors']})...")
    edge_index = GraphBuilder.build_edges(t_pos, k=config['num_neighbors'])

    return t_pos, t_z, t_shapes, t_target, edge_index


def train(config):
    # 1. Load
    pos, z, shapes, target, edge_index = load_full_data(config)
    N = pos.shape[0]

    # 2. Split
    indices = torch.randperm(N)
    n_train = int(N * 0.8)
    n_val = int(N * 0.1)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # 3. Setup with Timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{config.get('run_name', 'run')}_{timestamp}"
    log_dir = Path(config['log_dir']) / run_name

    print(f"Logging to: {log_dir}")

    model = init_vmdn(config).to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    writer = SummaryWriter(log_dir)

    best_loss = float('inf')
    patience_counter = 0

    print("Starting Training...")

    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()

        # Subsample for Train Step
        batch_size = int(len(train_idx) * config.get('subsample_ratio', 0.25))
        step_indices = train_idx[torch.randperm(len(train_idx))[:batch_size]]
        step_mask = torch.zeros(N, dtype=torch.bool, device=config['device'])
        step_mask[step_indices] = True

        loss = model.loss(pos, z, shapes, target, edge_index=edge_index, mask=step_mask)
        loss.backward()
        optimizer.step()

        # Validation
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_mask = torch.zeros(N, dtype=torch.bool, device=config['device'])
                val_mask[val_idx] = True
                val_loss = model.loss(pos, z, shapes, target, edge_index=edge_index, mask=val_mask)

                print(f"Epoch {epoch}: Train Loss {loss.item():.4f} | Val Loss {val_loss.item():.4f}")
                writer.add_scalar('Loss/Train', loss.item(), epoch)
                writer.add_scalar('Loss/Val', val_loss.item(), epoch)

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), log_dir / "best_model.pth")
                else:
                    patience_counter += 1
                    if patience_counter >= config['patience']:
                        print("Early stopping.")
                        break

        # Plotting
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                mu_all, _ = model(pos, z, shapes, edge_index)
                mu_np = mu_all.cpu().numpy().flatten()
                target_np = target.cpu().numpy().flatten()

                # Plot Train Subset (First 5000)
                sub_idx = train_idx[:5000].cpu().numpy()
                fig = create_density_figure(target_np[sub_idx], mu_np[sub_idx], f"Train Density (Epoch {epoch})")
                writer.add_figure("Density/Train", fig, epoch)

    writer.close()


if __name__ == "__main__":
    conf = {
        "csv_path": "data/des_metacal_angles_minimal.csv",
        "lr": 5e-4,
        "epochs": 1000,
        "patience": 200,
        "num_neighbors": 20,
        "hidden_dim": 128,
        "subsample_ratio": 0.05,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "log_dir": "runs",
        "run_name": "identity_check",
        "inject_signal": True
    }
    train(conf)