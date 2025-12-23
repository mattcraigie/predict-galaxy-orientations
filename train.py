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
from scipy.interpolate import interp1d

from models.vmdn import init_vmdn
from models.galaxy_gnn import GraphBuilder

# Default config
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
    "run_name": "com_test",
    "inject_signal": True  # Enabled for this test
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
    # Normalize to [0, 2pi]
    true = torch.remainder(true, 2 * np.pi)
    pred = torch.remainder(pred, 2 * np.pi)
    diff = torch.abs(true - pred)
    dist = torch.min(diff, 2 * np.pi - diff)
    return dist.mean().item()


def create_density_figure(true, pred, title, normalizer=None):
    true = np.remainder(true, 2 * np.pi)
    pred = np.remainder(pred, 2 * np.pi)
    bins = 64
    H, _, _ = np.histogram2d(true, pred, bins=bins, range=[[0, 2 * np.pi], [0, 2 * np.pi]])
    row_sums = H.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    H_norm = H / row_sums
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(H_norm.T, origin='lower', extent=[0, 2 * np.pi, 0, 2 * np.pi], aspect='auto', cmap='viridis', vmin=0,
                   vmax=np.max(H_norm) * 0.8)
    label_suffix = " (Warped)" if normalizer else ""
    ax.set_title(title + label_suffix)
    ax.set_xlabel(f'True Angle{label_suffix}')
    ax.set_ylabel(f'Pred Angle{label_suffix}')
    ax.plot([0, 2 * np.pi], [0, 2 * np.pi], 'r--', alpha=0.5)
    plt.colorbar(im, ax=ax)
    return fig


# --- CDF NORMALIZER ---
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


# --- NEW SIGNAL INJECTION: CENTER OF MASS ---
def inject_com_signal(pos, edge_index, N, device):
    """
    1. Assigns RANDOM input shapes (model must ignore them).
    2. Calculates the geometric Center of Mass (CoM) of neighbors.
    3. Sets Target = Angle pointing towards that CoM.
    """
    print("!!! INJECTING SIGNAL: POINT TO NEIGHBOR CENTER OF MASS !!!")

    # 1. Random Inputs
    # The model must learn that 'input_shapes' are useless noise
    # and look at the graph structure/geometry instead.
    rand_angles = torch.rand(N, device=device) * 2 * np.pi
    input_e1 = torch.cos(rand_angles)
    input_e2 = torch.sin(rand_angles)
    inputs = torch.stack([input_e1, input_e2], dim=1).float()

    # 2. Calculate CoM of Neighbors
    src, dst = edge_index

    # Sum neighbor positions grouped by center node (dst)
    sum_pos = torch.zeros((N, 2), device=device)
    sum_pos.index_add_(0, dst, pos[src])

    # Count neighbors per node
    ones = torch.ones(src.shape[0], device=device)
    counts = torch.zeros(N, device=device)
    counts.index_add_(0, dst, ones)
    counts = counts.clamp(min=1.0).unsqueeze(-1)

    com = sum_pos / counts

    # 3. Vector from Self to CoM
    vec_to_com = com - pos

    # 4. Target Angle
    # Calculate physical angle theta of the vector
    theta = torch.atan2(vec_to_com[:, 1], vec_to_com[:, 0])

    # Convert to "Double Angle" (Spin-2 domain) target
    # If the galaxy "points" along the vector, its orientation phi = theta
    # Our model predicts 2*phi, so we target 2*theta.
    target_angles = torch.remainder(2 * theta, 2 * np.pi)

    return inputs, target_angles


def load_full_data(config):
    print(f"Loading full catalog from {config['csv_path']}...")
    df = pd.read_csv(config['csv_path'])
    df = df.dropna(subset=['ra', 'dec', 'mean_z', 'phi_deg'])

    # strict radius filter
    ra, dec = df['ra'].values, df['dec'].values
    tree = cKDTree(np.stack([ra, dec], axis=1))
    dist, _ = tree.query(np.stack([ra, dec], axis=1), k=2)
    valid_mask = dist[:, 1] > 0.0005
    df = df[valid_mask]

    ra = df['ra'].values.astype(np.float32)
    dec = df['dec'].values.astype(np.float32)
    pos = np.stack([ra, dec], axis=1)
    pos_min, pos_max = pos.min(axis=0), pos.max(axis=0)
    pos = (pos - pos_min) / (pos_max - pos_min) * 2 - 1

    z = df['mean_z'].values.astype(np.float32)
    device = config['device']

    # 1. Build Graph
    print(f"Building graph (k={config['num_neighbors']})...")
    t_pos = torch.tensor(pos, device=device)
    edge_index = GraphBuilder.build_edges(t_pos, k=config['num_neighbors'])

    # 2. Inject Signal OR Load Real Data
    if config.get('inject_signal', False):
        # We pass t_pos and edge_index because the target depends on geometry now
        t_shapes, t_target = inject_com_signal(t_pos, edge_index, len(pos), device)
        normalizer = None
    else:
        phi_rad = np.deg2rad(df['phi_deg'].values.astype(np.float32))
        raw_targets = 2.0 * phi_rad

        normalizer = CDFNormalizer()
        normalizer.fit(raw_targets)
        target_angles = normalizer.transform(raw_targets)

        e1 = np.cos(target_angles)
        e2 = np.sin(target_angles)
        t_shapes = torch.stack([torch.tensor(e1), torch.tensor(e2)], dim=1).to(device).float()
        t_target = torch.tensor(target_angles, device=device).float()

    t_z = torch.tensor(z[:, None], device=device)

    return t_pos, t_z, t_shapes, t_target, edge_index, normalizer


# --- DETERMINISTIC INFERENCE ---
def generate_full_catalog_predictions(model, pos, z, shapes, edge_index, num_passes=4):
    print(f"\n--- Generating Full Catalog Predictions ({num_passes}-Pass Strategy) ---")
    model.eval()
    N = pos.shape[0]
    all_mu = torch.zeros(N, device=pos.device)
    all_kappa = torch.zeros(N, device=pos.device)

    with torch.no_grad():
        for i in range(num_passes):
            mask_indices = torch.arange(i, N, num_passes, device=pos.device)
            masked_shapes = shapes.clone()
            masked_shapes[mask_indices] = 0.0  # Blind spot

            mu, kappa = model(pos, z, masked_shapes, edge_index)

            all_mu[mask_indices] = mu[mask_indices].squeeze()
            all_kappa[mask_indices] = kappa[mask_indices].squeeze()

    return all_mu, all_kappa


# --- TRAIN LOOP ---
def train(config: dict) -> None:
    pos, z, shapes, target, edge_index, normalizer = load_full_data(config)
    N = pos.shape[0]

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

        batch_size = int(len(train_idx) * config.get('subsample_ratio', 0.25))
        mask_idx = train_idx[torch.randperm(len(train_idx))[:batch_size]]

        masked_shapes = shapes.clone()
        masked_shapes[mask_idx] = 0.0

        step_mask = torch.zeros(N, dtype=torch.bool, device=config['device'])
        step_mask[mask_idx] = True

        loss = model.loss(pos, z, masked_shapes, target, edge_index=edge_index, mask=step_mask)
        loss.backward()
        optimizer.step()

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
                        print("Early stopping.");
                        break

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                # Plot with train set blinded
                plot_shapes = shapes.clone()
                plot_shapes[train_idx] = 0.0
                mu_all, _ = model(pos, z, plot_shapes, edge_index)
                t_np = target.cpu().numpy().flatten()
                m_np = mu_all.cpu().numpy().flatten()

                sub = train_idx[:5000].cpu().numpy()
                fig = create_density_figure(t_np[sub], m_np[sub], f"Train Density (Epoch {epoch})", normalizer)
                writer.add_figure("Density/Train", fig, epoch)

    print("\nTraining Complete. Loading best model...")
    model.load_state_dict(torch.load(log_dir / "best_model.pth"))

    pred_mu, pred_kappa = generate_full_catalog_predictions(model, pos, z, shapes, edge_index, num_passes=4)

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
        fig = create_density_figure(subset_true.cpu().numpy(), subset_pred.cpu().numpy(), f"{name} Predictions",
                                    normalizer)
        writer.add_figure(f"Density/{name}_Final", fig, 0)

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