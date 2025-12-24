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

# --- HYPEROPT IMPORTS ---
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

# Import your models (Assumed available in your path)
from models.vmdn import init_vmdn
from models.galaxy_gnn import GraphBuilder
from models.pretraining import GalaxyReconstructor


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


def run_pretraining(config, pos, z, shapes, edge_index):
    print("\n=== STARTING PRE-TRAINING (Latent Truncation Strategy) ===")

    pre_model = GalaxyReconstructor(config).to(config['device'])
    optimizer = torch.optim.Adam(pre_model.parameters(), lr=1e-3)  # Lower LR for stability

    # --- STABILITY FIX: Normalize Targets ---
    # We want z to be roughly mean 0, std 1 so MSE doesn't explode
    z_mean = z.mean()
    z_std = z.std() + 1e-6
    z_norm = (z - z_mean) / z_std
    # ----------------------------------------

    pre_model.train()
    epochs = config.get('pretrain_epochs', 50)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Mask 25% of nodes
        N = pos.shape[0]
        batch_size = int(N * 0.25)
        mask_idx = torch.randperm(N)[:batch_size]

        # Create Inputs (Masked)
        masked_shapes = shapes.clone()
        masked_z = z_norm.clone()

        # Zero out info for masked nodes
        masked_shapes[mask_idx] = 0.0
        masked_z[mask_idx] = 0.0

        # Boolean Mask
        mask = torch.zeros(N, dtype=torch.bool, device=config['device'])
        mask[mask_idx] = True

        # Pass NORMALIZED z as target, but masked z as input
        loss = pre_model.loss(pos, masked_z, masked_shapes, edge_index, mask)

        if torch.isnan(loss):
            print("!!! LOSS BECAME NAN !!! Stopping pre-training.")
            break

        loss.backward()

        # Gradient Clipping (Safety net for explosions)
        torch.nn.utils.clip_grad_norm_(pre_model.parameters(), 1.0)

        optimizer.step()

        if epoch % 10 == 0:
            print(f"  [Pretrain Epoch {epoch}] Loss: {loss.item():.6f}")

    print("=== PRE-TRAINING COMPLETE ===\n")
    torch.save(pre_model.backbone.state_dict(), "pretrained_backbone.pth")
    return "pretrained_backbone.pth"


# --- 1. SEPARATE DATA LOADING ---
# We load data ONCE globally to save massive amounts of time
def load_data_global(config):
    print("--- GLOBAL DATA LOAD START ---")
    # This uses your existing logic, but returns the tensors directly
    # Note: We hardcode device to CPU for storage, move to GPU inside the trial
    # to avoid OOM if running many trials.

    df = pd.read_csv(config['csv_path'])
    df = df.dropna(subset=['ra', 'dec', 'mean_z', 'phi_deg'])

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

    # Build Graph
    print(f"Building global graph (k={config['num_neighbors']})...")
    # Note: If 'num_neighbors' is a hyperparameter, you must move this INSIDE the trial.
    # Assuming k is fixed for now.
    t_pos_cpu = torch.tensor(pos)
    edge_index_cpu = GraphBuilder.build_edges(t_pos_cpu, k=config['num_neighbors'])

    if config.get('inject_signal', False):
        # Calculate targets (using dummy device for now, will move later)
        t_shapes_cpu, t_target_cpu = inject_com_signal(t_pos_cpu, edge_index_cpu, len(pos), 'cpu')
        normalizer = None
    else:
        phi_rad = np.deg2rad(df['phi_deg'].values.astype(np.float32))
        raw_targets = 2.0 * phi_rad
        normalizer = CDFNormalizer()
        normalizer.fit(raw_targets)
        target_angles = normalizer.transform(raw_targets)

        e1 = np.cos(target_angles)
        e2 = np.sin(target_angles)
        t_shapes_cpu = torch.stack([torch.tensor(e1), torch.tensor(e2)], dim=1).float()
        t_target_cpu = torch.tensor(target_angles).float()

    t_z_cpu = torch.tensor(z[:, None])

    print("--- GLOBAL DATA LOAD DONE ---")
    return t_pos_cpu, t_z_cpu, t_shapes_cpu, t_target_cpu, edge_index_cpu, normalizer


# --- 2. THE OBJECTIVE FUNCTION ---
def objective(params):
    """
    This function runs ONE training experiment with a specific set of params.
    """

    # 1. Merge params into config
    # We use a global variable 'GLOBAL_CONFIG' for base settings,
    # and 'GLOBAL_DATA' for the dataset.
    config = {**GLOBAL_CONFIG, **params}

    device = config['device']
    print(f"\n>>> Starting Trial. Params: {params}")

    # 2. Move Data to GPU (only for this trial)
    t_pos = GLOBAL_DATA[0].to(device)
    t_z = GLOBAL_DATA[1].to(device)
    t_shapes = GLOBAL_DATA[2].to(device)
    t_target = GLOBAL_DATA[3].to(device)
    edge_index = GLOBAL_DATA[4].to(device)
    normalizer = GLOBAL_DATA[5]

    N = t_pos.shape[0]

    # Standard Split
    indices = torch.randperm(N)
    n_train = int(N * 0.8)
    n_val = int(N * 0.1)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # 3. Model & Optimizer
    # Note: hidden_dim changes the architecture, so we init a fresh model
    model = init_vmdn(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # (Optional) Pre-training
    # If hidden_dim changes, we technically need to re-pretrain or skip it for speed.
    # For HPO speed, often we skip pretraining or do a very short version.
    # For this example, I'll assume we skip it to save time, or you can uncomment:
    # run_pretraining(config, t_pos, t_z, t_shapes, edge_index)

    # 4. Training Loop (Condensed)
    best_val_loss = float('inf')
    early_stop = 0

    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()

        # Subsample using the hyperparameter ratio
        batch_size = int(len(train_idx) * config['subsample_ratio'])
        mask_idx = train_idx[torch.randperm(len(train_idx))[:batch_size]]

        masked_shapes = t_shapes.clone()
        masked_shapes[mask_idx] = 0.0
        step_mask = torch.zeros(N, dtype=torch.bool, device=device)
        step_mask[mask_idx] = True

        loss = model.loss(t_pos, t_z, masked_shapes, t_target, edge_index=edge_index, mask=step_mask)
        loss.backward()
        optimizer.step()

        # Validation every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_shapes = t_shapes.clone()
                val_shapes[val_idx] = 0.0
                val_mask = torch.zeros(N, dtype=torch.bool, device=device)
                val_mask[val_idx] = True
                val_loss = model.loss(t_pos, t_z, val_shapes, t_target, edge_index=edge_index, mask=val_mask).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop = 0
                else:
                    early_stop += 1

            if early_stop >= 5:  # stricter patience for HPO
                break

    # 5. Final Evaluation (Test Set)
    model.eval()

    # We need to calculate the actual metric (MAE Improvement) for the report
    # Using your generation strategy
    pred_mu, _ = generate_full_catalog_predictions(model, t_pos, t_z, t_shapes, edge_index, num_passes=4)

    subset_true = t_target[test_idx]
    subset_pred = pred_mu[test_idx]
    mae = circular_mae(subset_true, subset_pred)
    baseline_mae = np.pi / 2
    improvement = (baseline_mae - mae) / baseline_mae * 100.0

    # 6. Return Dictionary for Hyperopt
    # We minimize 'loss'. Usually we minimize Val Loss, but log Test Improvement.
    return {
        'loss': best_val_loss,  # The metric Hyperopt minimizes
        'status': STATUS_OK,  # Required key
        'test_improvement': improvement,  # Custom metric to log
        'test_mae': mae,  # Custom metric to log
        'params': params  # Log the params used
    }


# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default="config.yaml")
    args = parser.parse_args()

    # Load Defaults
    GLOBAL_CONFIG = load_config(args.config)
    # Decrease epochs for HPO trials to speed things up?
    GLOBAL_CONFIG['epochs'] = 1000

    # Load Data Once
    GLOBAL_DATA = load_data_global(GLOBAL_CONFIG)

    # --- MODIFIED SEARCH SPACE ---
    space = {
        # 1. FIXED: We lock the model size to the reliable winner
        'hidden_dim': hp.choice('hidden_dim', [64]),

        # 2. SHIFTED RIGHT: exploring slightly higher LRs now that we have larger batches
        # Previous best was ~6e-5. We search from 3e-5 up to 3e-4.
        'lr': hp.loguniform('lr', np.log(5e-5), np.log(5e-4)),
        'subsample_ratio': hp.uniform('subsample_ratio', 0.1, 0.5)
    }

    # Initialize Trials object to store results
    trials = Trials()

    print(">>> Starting Hyperopt Optimization...")

    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,  # Tree-structured Parzen Estimator
        max_evals=20,  # Number of trials
        trials=trials
    )

    print("\n\n=== HPO COMPLETE ===")
    print("Best Params:", best)

    # --- ANALYSIS OF RESULTS ---
    print("\n--- Detailed Results ---")
    results = []
    for t in trials.trials:
        res = t['result']
        p = res['params']
        results.append({
            'lr': p['lr'],
            'hidden_dim': p['hidden_dim'],
            'subsample': p['subsample_ratio'],
            'val_loss': res['loss'],
            'test_imp': res['test_improvement']
        })

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by='test_imp', ascending=False)
    print(df_res)

    # Optional: Save results to CSV for plotting later
    df_res.to_csv("hpo_results_fixed_dim.csv", index=False)