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


# --- KEEP YOUR HELPER FUNCTIONS ---
# (Paste your existing circular_mae, create_density_figure,
# CDFNormalizer, inject_com_signal, load_config here...)
# ----------------------------------

# [Past helper functions omitted for brevity - assume they exist as in your script]

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
    GLOBAL_CONFIG['epochs'] = 200

    # Load Data Once
    GLOBAL_DATA = load_data_global(GLOBAL_CONFIG)

    # Define Search Space
    space = {
        'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-2)),
        'subsample_ratio': hp.loguniform('subsample_ratio', np.log(0.001), np.log(0.5)),
        'hidden_dim': hp.choice('hidden_dim', [32, 64, 128, 256, 512])
    }

    # Initialize Trials object to store results
    trials = Trials()

    print(">>> Starting Hyperopt Optimization...")

    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,  # Tree-structured Parzen Estimator
        max_evals=5,  # Number of trials
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
    df_res.to_csv("hpo_results.csv", index=False)