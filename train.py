import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scipy.spatial import cKDTree
from datetime import datetime

from dataset import GalaxyDataset
from models.vmdn import init_vmdn


DEFAULT_CONFIG = {
    "csv_path": "data/des_metacal_angles_minimal.csv",
    "batch_size": 256,
    "lr": 1e-3,
    "epochs": 100,
    "patience": 10,
    "num_workers": 4,
    "log_dir": "runs/galaxy_vmdn_experiment_1",
    "num_neighbors": 5,
    "hidden_dim": 64,
    "num_layers": 3,
    "vmdn_regularization": 0.1,
    "device": "auto",
}


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


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


def train(config: dict) -> None:
    # 1. Setup Data
    print(f"Loading dataset from {config['csv_path']}...")
    train_ds = GalaxyDataset(config["csv_path"], num_neighbors=config["num_neighbors"], mode="train")
    val_ds = GalaxyDataset(config["csv_path"], num_neighbors=config["num_neighbors"], mode="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

    # 2. Setup Model
    model = init_vmdn(config).to(config["device"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # 3. Logging & Checkpointing
    writer = SummaryWriter(config["log_dir"])
    stopper = EarlyStopper(patience=config["patience"])

    print("Starting training...")

    for epoch in range(config["epochs"]):
        # --- TRAINING LOOP ---
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}")
        for batch in pbar:
            pos = batch["pos"].to(config["device"])
            z = batch["redshift"].to(config["device"])
            shapes = batch["input_shapes"].to(config["device"])
            target = batch["target_phi"].to(config["device"])


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

        with torch.no_grad():
            for batch in val_loader:
                pos = batch["pos"].to(config["device"])
                z = batch["redshift"].to(config["device"])
                shapes = batch["input_shapes"].to(config["device"])
                target = batch["target_phi"].to(config["device"])

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


        # Save best model
        if avg_val_loss == stopper.min_validation_loss:
            torch.save(model.state_dict(), os.path.join(config["log_dir"], "best_model.pth"))

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the galaxy orientation model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML config file.")
    args = parser.parse_args()

    train(load_config(args.config))
