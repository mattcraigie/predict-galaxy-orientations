import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from scipy.spatial import cKDTree

# Import our models
from models.vmdn import init_vmdn
from models.galaxy_gnn import GraphBuilder

# Default config fallback
DEFAULT_CONFIG = {
    "csv_path": "data/des_metacal_angles_minimal.csv",
    "lr": 2e-3,
    "epochs": 1000,
    "patience": 30,
    "num_neighbors": 20,
    "hidden_dim": 64,
    "subsample_ratio": 0.25,
    "device": "auto",
    "log_dir": "runs",
    "run_name": "galaxy_gnn",
    "inject_signal": False
}


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    merged = {**DEFAULT_CONFIG, **config}

    if merged["device"] == "auto":
        merged["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    repo_root = Path(__file__).resolve().parent
    for path_key in ("csv_path", "log_dir"):
        path_value = Path(merged[path_key])
        if not path_value.is_absolute():
            merged[path_key] = str((repo_root / path_value).resolve())

    return merged


# --- PLOTTING HELPERS ---
def create_density_figure(true, pred, title):
    """Plots P(pred | true) on the 2*phi domain (0 to 2pi)."""
    # Normalize to 0-2pi
    true = np.remainder(true, 2 * np.pi)
    pred = np.remainder(pred, 2 * np.pi)

    bins = 64
    range_lim = [[0, 2 * np.pi], [0, 2 * np.pi]]

    H, xedges, yedges = np.histogram2d(true, pred, bins=bins, range=range_lim)

    # Normalize rows (P(pred|true))
    row_sums = H.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    H_norm = H / row_sums

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(H_norm.T, origin='lower', extent=[0, 2 * np.pi, 0, 2 * np.pi],
                   aspect='auto', cmap='viridis', vmin=0, vmax=np.max(H_norm) * 0.8)

    ax.set_title(title)
    ax.set_xlabel('True Double Angle (2*phi)')
    ax.set_ylabel('Pred Double Angle (2*phi)')

    # Diagonal reference line
    ax.plot([0, 2 * np.pi], [0, 2 * np.pi], 'r--', alpha=0.5)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


# --- DATA LOADING & SIGNAL INJECTION ---
def inject_identity_signal(pos, N):
    print("!!! INJECTING SIGNAL: IDENTITY MAPPING !!!")
    # Generate random targets in 0-2pi (representing 2*phi)
    new_targets = np.random.uniform(0, 2 * np.pi, size=N).astype(np.float32)

    # Input is cos/sin of this target
    e1 = np.cos(new_targets)
    e2 = np.sin(new_targets)
    new_inputs = np.stack([e1, e2], axis=1).astype(np.float32)

    return new_inputs, new_targets


def load_full_data(config):
    print(f"Loading full catalog from {config['csv_path']}...")
    df = pd.read_csv(config['csv_path'])
    df = df.dropna(subset=['ra', 'dec', 'mean_z', 'phi_deg'])

    # Coordinates
    ra = df['ra'].values.astype(np.float32)
    dec = df['dec'].values.astype(np.float32)
    pos = np.stack([ra, dec], axis=1)
    # Center positions to avoid huge coords affecting precision
    pos -= pos.mean(axis=0)

    z = df['mean_z'].values.astype(np.float32)

    # --- CRITICAL: USE DOUBLE ANGLES ---
    # Galaxy shapes are Spin-2. VMDN predicts Spin-1.
    # We train on 2*phi to match topologies.
    phi_rad = np.deg2rad(df['phi_deg'].values.astype(np.float32))
    target_angles = 2.0 * phi_rad  # [0, 2pi]

    # --- SIGNAL INJECTION ---
    if config.get('inject_signal', False):
        shapes, target_angles = inject_identity_signal(pos, len(pos))
    else:
        # Standard Spin-2 Input
        e1 = np.cos(target_angles)
        e2 = np.sin(target_angles)
        shapes = np.stack([e1, e2], axis=1)

    device = config['device']
    t_pos = torch.tensor(pos, device=device)
    t_z = torch.tensor(z[:, None], device=device)
    t_shapes = torch.tensor(shapes, device=device)
    t_target = torch.tensor(target_angles, device=device)

    print(f"Building graph (k={config['num_neighbors']})...")
    edge_index = GraphBuilder.build_edges(t_pos, k=config['num_neighbors'])

    return t_pos, t_z, t_shapes, t_target, edge_index


# --- MAIN TRAIN LOOP ---
def train(config: dict) -> None:
    # 1. Load Data (Full Batch)
    pos, z, shapes, target, edge_index = load_full_data(config)
    N = pos.shape[0]

    # 2. Split Indices
    indices = torch.randperm(N)
    n_train = int(N * 0.8)
    n_val = int(N * 0.1)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    # Remaining are test, unused in loop

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # 3. Setup Logging
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{config.get('run_name', 'run')}_{timestamp}"
    log_dir = Path(config['log_dir']) / run_name
    print(f"Logging to: {log_dir}")

    writer = SummaryWriter(log_dir)

    # 4. Model Setup
    model = init_vmdn(config).to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    best_loss = float('inf')
    patience_counter = 0

    print("Starting Training...")

    for epoch in range(config["epochs"]):
        # --- TRAINING STEP ---
        model.train()
        optimizer.zero_grad()

        # Stochastic Subsampling: Pick random subset of TRAIN nodes
        batch_size = int(len(train_idx) * config.get('subsample_ratio', 0.25))
        mask_idx = train_idx[torch.randperm(len(train_idx))[:batch_size]]
        step_mask = torch.zeros(N, dtype=torch.bool, device=config['device'])
        step_mask[mask_idx] = True

        # Forward Pass (Full Graph) -> Masked Loss
        loss = model.loss(pos, z, shapes, target, edge_index=edge_index, mask=step_mask)
        loss.backward()
        optimizer.step()

        # --- VALIDATION STEP ---
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_mask = torch.zeros(N, dtype=torch.bool, device=config['device'])
                val_mask[val_idx] = True

                val_loss = model.loss(pos, z, shapes, target, edge_index=edge_index, mask=val_mask)

                print(f"Epoch {epoch}: Train {loss.item():.4f} | Val {val_loss.item():.4f}")
                writer.add_scalar('Loss/Train', loss.item(), epoch)
                writer.add_scalar('Loss/Val', val_loss.item(), epoch)

                # Checkpointing
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), log_dir / "best_model.pth")
                else:
                    patience_counter += 1
                    if patience_counter >= config['patience']:
                        print("Early stopping triggered.")
                        break

        # --- PLOTTING DIAGNOSTICS ---
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                # Get predictions for full graph
                mu_all, _ = model(pos, z, shapes, edge_index)

                # Convert to numpy
                t_np = target.cpu().numpy().flatten()
                m_np = mu_all.cpu().numpy().flatten()

                # Plot subset of train data for speed
                sub = train_idx[:5000].cpu().numpy()
                fig = create_density_figure(t_np[sub], m_np[sub], f"Train Density (Epoch {epoch})")
                writer.add_figure("Density/Train", fig, epoch)

    writer.close()
    print("Training Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the galaxy orientation model.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,  # Require config file
        help="Path to a YAML config file.",
    )
    args = parser.parse_args()

    train(load_config(args.config))