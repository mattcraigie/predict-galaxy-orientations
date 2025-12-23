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
    "subsample_ratio": 0.25,  # Fraction of nodes to train on per step
    "device": "auto",
    "log_dir": "runs",
    "run_name": "galaxy_gnn_real",
    "inject_signal": False  # Set to True to test with synthetic swirl
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


# --- METRICS & PLOTTING ---
def circular_mae(true, pred):
    """
    Computes Mean Absolute Error for angular data in [0, 2pi].
    Handles wrapping: distance between 0.1 and 6.2 is small (0.18), not large.
    """
    diff = torch.abs(true - pred)
    # The shortest distance on a circle is min(diff, 2pi - diff)
    dist = torch.min(diff, 2 * np.pi - diff)
    return dist.mean().item()


def create_density_figure(true, pred, title, normalizer=None):

    true = np.remainder(true, 2 * np.pi)
    pred = np.remainder(pred, 2 * np.pi)

    # ... (histogram logic same as before) ...
    bins = 64
    H, _, _ = np.histogram2d(true, pred, bins=bins, range=[[0, 2 * np.pi], [0, 2 * np.pi]])
    row_sums = H.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    H_norm = H / row_sums

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(H_norm.T, origin='lower', extent=[0, 2 * np.pi, 0, 2 * np.pi], aspect='auto', cmap='viridis', vmin=0,
                   vmax=np.max(H_norm) * 0.8)

    # Update Label based on whether we are normalized
    label_suffix = " (Warped)" if normalizer else ""
    ax.set_title(title + label_suffix)
    ax.set_xlabel(f'True Angle{label_suffix}')
    ax.set_ylabel(f'Pred Angle{label_suffix}')

    ax.plot([0, 2 * np.pi], [0, 2 * np.pi], 'r--', alpha=0.5)
    plt.colorbar(im, ax=ax)
    return fig


# --- NEW: GLOBAL DISTRIBUTION REMOVAL ---
class CDFNormalizer:
    """
    Learns the global distribution P(phi) and warps the space
    so that the global distribution becomes Uniform(0, 2pi).

    This forces the model to learn purely LOCAL deviations (intrinsic alignment)
    rather than global systematics (grid locking).
    """

    def __init__(self):
        self.cdf_func = None
        self.inv_cdf_func = None

    def fit(self, angles):
        """
        Compute empirical CDF from training data.
        angles: numpy array of angles in [0, 2pi]
        """
        # Sort data to get the empirical distribution
        sorted_angles = np.sort(angles)
        n = len(sorted_angles)

        # y-values for CDF (0 to 1)
        y_vals = np.arange(n) / (n - 1)

        # Create interpolation function F(x) -> u
        # We pad with 0 and 2pi to handle edge cases
        x_pad = np.concatenate([[0.0], sorted_angles, [2 * np.pi]])
        y_pad = np.concatenate([[0.0], y_vals, [1.0]])

        self.cdf_func = interp1d(x_pad, y_pad, kind='linear', bounds_error=False, fill_value=(0, 1))

        # Inverse for plotting later (optional)
        self.inv_cdf_func = interp1d(y_pad, x_pad, kind='linear', bounds_error=False, fill_value="extrapolate")
        print(" Global CDF fitted. Systematic biases capture initiated.")

    def transform(self, angles):
        """ Warps angles to be Uniform [0, 2pi] """
        if self.cdf_func is None:
            raise ValueError("Run fit() first!")

        # 1. Map to [0, 1] using CDF
        u = self.cdf_func(angles)

        # 2. Map to [0, 2pi]
        return (u * 2 * np.pi).astype(np.float32)


# --- SIGNAL INJECTION (FOR DEBUGGING) ---
def inject_smooth_signal(pos):
    """Creates a giant spatial swirl for testing."""
    print("!!! INJECTING SIGNAL: SMOOTH SPATIAL FIELD !!!")
    x, y = pos[:, 0], pos[:, 1]
    freq = 5.0
    target_angles = np.remainder(np.arctan2(y, x) * freq, 2 * np.pi).astype(np.float32)
    e1, e2 = np.cos(target_angles), np.sin(target_angles)
    return np.stack([e1, e2], axis=1).astype(np.float32), target_angles


# --- DATA LOADING ---
def load_full_data(config):
    print(f"Loading full catalog from {config['csv_path']}...")
    df = pd.read_csv(config['csv_path'])
    df = df.dropna(subset=['ra', 'dec', 'mean_z', 'phi_deg'])

    # 1. Remove Duplicates
    df['ra_round'] = df['ra'].round(5)
    df['dec_round'] = df['dec'].round(5)
    df = df.drop_duplicates(subset=['ra_round', 'dec_round'])

    ra = df['ra'].values.astype(np.float32)
    dec = df['dec'].values.astype(np.float32)
    pos = np.stack([ra, dec], axis=1)

    pos_min, pos_max = pos.min(axis=0), pos.max(axis=0)
    pos = (pos - pos_min) / (pos_max - pos_min) * 2 - 1

    z = df['mean_z'].values.astype(np.float32)

    # 2. Process Targets
    if config.get('inject_signal', False):
        shapes, target_angles = inject_smooth_signal(pos)
        # Note: We usually DON'T normalize injected signals because they are already smooth/known
        # But if you want to test the normalizer, you can fit it here too.
        normalizer = None
    else:
        # Real Data
        phi_rad = np.deg2rad(df['phi_deg'].values.astype(np.float32))
        raw_targets = 2.0 * phi_rad

        # --- APPLY CDF NORMALIZATION ---
        normalizer = CDFNormalizer()
        normalizer.fit(raw_targets)  # Learn the "Grid Locking" or other biases
        target_angles = normalizer.transform(raw_targets)  # Warp to Uniform
        e1 = np.cos(target_angles)
        e2 = np.sin(target_angles)
        shapes = np.stack([e1, e2], axis=1)

    device = config['device']
    return (
        torch.tensor(pos, device=device),
        torch.tensor(z[:, None], device=device),
        torch.tensor(shapes, device=device),
        torch.tensor(target_angles, device=device),
        GraphBuilder.build_edges(torch.tensor(pos, device=device), k=config['num_neighbors']),
        normalizer  # Return this so we can plot correctly later
    )


# --- DETERMINISTIC INFERENCE ---
def generate_full_catalog_predictions(model, pos, z, shapes, edge_index, num_passes=4):
    """
    Predicts every galaxy by masking it in one of K passes.
    Guarantees no leakage while using ~75% of neighbors for context.
    """
    print(f"\n--- Generating Full Catalog Predictions ({num_passes}-Pass Strategy) ---")
    model.eval()
    N = pos.shape[0]
    all_mu = torch.zeros(N, device=pos.device)
    all_kappa = torch.zeros(N, device=pos.device)

    with torch.no_grad():
        for i in range(num_passes):
            # 1. Mask Indices: i, i+4, i+8...
            mask_indices = torch.arange(i, N, num_passes, device=pos.device)

            # 2. Prepare Masked Input (Zero out the targets for this pass)
            masked_shapes = shapes.clone()
            masked_shapes[mask_indices] = 0.0

            # 3. Forward Pass (Predict everyone)
            mu, kappa = model(pos, z, masked_shapes, edge_index)

            # 4. Store ONLY the predictions for the masked nodes
            all_mu[mask_indices] = mu[mask_indices].squeeze()
            all_kappa[mask_indices] = kappa[mask_indices].squeeze()

    return all_mu, all_kappa


# --- MAIN LOOP ---
def train(config: dict) -> None:
    # Capture the normalizer in the return tuple
    pos, z, shapes, target, edge_index, normalizer = load_full_data(config)
    N = pos.shape[0]

    # Splits
    indices = torch.randperm(N)
    n_train = int(N * 0.8)
    n_val = int(N * 0.1)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{config.get('run_name', 'run')}_{timestamp}"
    log_dir = Path(config['log_dir']) / run_name
    print(f"Logging to: {log_dir}")
    writer = SummaryWriter(log_dir)

    model = init_vmdn(config).to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    best_loss = float('inf')
    patience_cnt = 0

    print("Starting Training...")
    for epoch in range(config["epochs"]):
        model.train()
        optimizer.zero_grad()

        # --- MASKED TRAINING STEP ---
        batch_size = int(len(train_idx) * config.get('subsample_ratio', 0.25))
        mask_idx = train_idx[torch.randperm(len(train_idx))[:batch_size]]

        masked_shapes = shapes.clone()
        masked_shapes[mask_idx] = 0.0

        step_mask = torch.zeros(N, dtype=torch.bool, device=config['device'])
        step_mask[mask_idx] = True

        loss = model.loss(pos, z, masked_shapes, target, edge_index=edge_index, mask=step_mask)
        loss.backward()
        optimizer.step()

        # --- VALIDATION ---
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_shapes = shapes.clone()
                val_shapes[val_idx] = 0.0
                val_mask = torch.zeros(N, dtype=torch.bool, device=config['device'])
                val_mask[val_idx] = True

                val_loss = model.loss(pos, z, val_shapes, target, edge_index=edge_index, mask=val_mask)

                print(f"Epoch {epoch}: Train {loss.item():.4f} | Val {val_loss.item():.4f}")
                writer.add_scalar('Loss/Train', loss.item(), epoch)
                writer.add_scalar('Loss/Val', val_loss.item(), epoch)

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_cnt = 0
                    torch.save(model.state_dict(), log_dir / "best_model.pth")
                else:
                    patience_cnt += 1
                    if patience_cnt >= config['patience']:
                        print("Early stopping.")
                        break

    # --- FINAL EVALUATION ---
    print("\nTraining Complete. Loading best model...")
    model.load_state_dict(torch.load(log_dir / "best_model.pth"))

    # 1. Generate clean predictions for EVERYONE
    pred_mu, pred_kappa = generate_full_catalog_predictions(model, pos, z, shapes, edge_index, num_passes=4)

    # 2. Compute Metrics
    # Baseline Random Guess MAE = pi/2 approx 1.5708
    baseline_mae = np.pi / 2

    def evaluate_subset(name, idx_tensor):
        subset_true = target[idx_tensor]
        subset_pred = pred_mu[idx_tensor]

        mae = circular_mae(subset_true, subset_pred)
        improvement = (baseline_mae - mae) / baseline_mae * 100.0

        print(f"--- {name} SET RESULTS ---")
        print(f"  MAE (Model):   {mae:.4f} rad")
        print(f"  MAE (Random):  {baseline_mae:.4f} rad")
        print(f"  Improvement:   {improvement:.2f}%")

        writer.add_scalar(f'MAE/{name}', mae, 0)
        writer.add_scalar(f'Improvement/{name}', improvement, 0)

        # Generate Plot
        fig = create_density_figure(subset_true.cpu().numpy(), subset_pred.cpu().numpy(), f"{name} Predictions",
                                    normalizer)
        writer.add_figure(f"Density/{name}_Final", fig, 0)

    # Evaluate all splits
    evaluate_subset("Train", train_idx)
    evaluate_subset("Validation", val_idx)
    evaluate_subset("Test", test_idx)

    writer.close()
    print(f"\nResults saved to {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    train(load_config(args.config))