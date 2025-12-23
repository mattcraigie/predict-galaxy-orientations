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
def plot_conditional_density(true, pred, title, filename):
    """
    Plots the conditional density P(pred | true).
    x-axis: True Angle
    y-axis: Predicted Angle
    Color: Probability density (columns sum to 1).
    """
    # Ensure inputs are numpy arrays in range [0, 2pi]
    true = np.remainder(true, 2 * np.pi)
    pred = np.remainder(pred, 2 * np.pi)

    # Binning configuration
    bins = 64
    range_lim = [[0, 2 * np.pi], [0, 2 * np.pi]]

    # 1. Compute 2D Histogram
    # H[i, j] is count in x_bin[i] and y_bin[j]
    # x (rows) = true, y (cols) = pred
    H, xedges, yedges = np.histogram2d(true, pred, bins=bins, range=range_lim)

    # 2. Normalize to get P(pred | true)
    # We divide each "True" row by the sum of that row
    # (handling division by zero)
    row_sums = H.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    H_norm = H / row_sums

    # 3. Plot
    fig, ax = plt.subplots(figsize=(8, 7))
    # We transpose H_norm because imshow expects (y, x) but histogram2d returns (x, y) logic relative to bins
    im = ax.imshow(H_norm.T, origin='lower', extent=[0, 2 * np.pi, 0, 2 * np.pi],
                   aspect='auto', cmap='viridis', vmin=0, vmax=np.max(H_norm) * 0.8)

    ax.set_title(title)
    ax.set_xlabel('True Angle (Radians)')
    ax.set_ylabel('Predicted Angle (Radians)')

    # Add diagonal reference line
    ax.plot([0, 2 * np.pi], [0, 2 * np.pi], 'r--', alpha=0.5, label='Perfect Prediction')
    ax.legend()

    plt.colorbar(im, ax=ax, label='P(Pred | True)')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")


# --- SIGNAL HELPERS ---
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

    # Signal Injection Logic
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

    # 2. Split Indices (Train / Val / Test)
    indices = torch.randperm(N)
    n_train = int(N * 0.8)
    n_val = int(N * 0.1)
    # The remainder is Test

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
        if epoch % 10 == 0:
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

    # --- FINAL EVALUATION ---
    print("\n--- Generating Evaluation Plots ---")

    # Load best model
    model.load_state_dict(torch.load(f"{config['log_dir']}/best_model.pth"))
    model.eval()

    with torch.no_grad():
        # Get predictions for all data (faster than batching since we loaded full batch anyway)
        mu_all, kappa_all = model(pos, z, shapes, edge_index)

        # Move to CPU numpy for plotting
        mu_np = mu_all.cpu().numpy().flatten()
        target_np = target.cpu().numpy().flatten()

        # Slice for Train
        train_mu = mu_np[train_idx.cpu().numpy()]
        train_true = target_np[train_idx.cpu().numpy()]

        # Slice for Test
        test_mu = mu_np[test_idx.cpu().numpy()]
        test_true = target_np[test_idx.cpu().numpy()]

        # Plot
        plot_conditional_density(
            train_true, train_mu,
            title=f"Train Set: P(Pred | True) - {config.get('log_dir').split('/')[-1]}",
            filename=f"{config['log_dir']}/density_train.png"
        )

        plot_conditional_density(
            test_true, test_mu,
            title=f"Test Set: P(Pred | True) - {config.get('log_dir').split('/')[-1]}",
            filename=f"{config['log_dir']}/density_test.png"
        )


if __name__ == "__main__":
    conf = {
        "csv_path": "data/des_metacal_angles_minimal.csv",
        "lr": 5e-3,
        "epochs": 1000,
        "patience": 20,
        "num_neighbors": 20,
        "hidden_dim": 64,
        "subsample_ratio": 0.25,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "log_dir": "runs/identity_check_v2",
        "inject_signal": True
    }
    train(conf)