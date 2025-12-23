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

            optimizer.zero_grad()

            # Forward pass is handled inside loss() usually,
            # but we call loss() directly on the VMDN wrapper
            loss = model.loss(pos, z, shapes, target=target)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = np.mean(train_losses)

        # --- VALIDATION LOOP ---
        model.eval()
        val_losses = []
        all_mus = []
        all_kappas = []

        with torch.no_grad():
            for batch in val_loader:
                pos = batch["pos"].to(config["device"])
                z = batch["redshift"].to(config["device"])
                shapes = batch["input_shapes"].to(config["device"])
                target = batch["target_phi"].to(config["device"])

                loss = model.loss(pos, z, shapes, target=target)
                val_losses.append(loss.item())

                # Capture stats for Tensorboard
                mu, kappa = model(pos, z, shapes)
                all_mus.append(mu.cpu().numpy().flatten())
                all_kappas.append(kappa.cpu().numpy().flatten())

        avg_val_loss = np.mean(val_losses)

        # --- LOGGING ---
        print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)

        # Log distribution of predictions to ensure we aren't collapsing
        flat_mus = np.concatenate(all_mus)
        flat_kappas = np.concatenate(all_kappas)
        writer.add_histogram('Distribution/Predicted_Mu', flat_mus, epoch)
        writer.add_histogram('Distribution/Predicted_Kappa', flat_kappas, epoch)

        # --- EARLY STOPPING ---
        if stopper.early_stop(avg_val_loss):
            print("Early stopping triggered!")
            break

        # Save best model
        if avg_val_loss == stopper.min_validation_loss:
            torch.save(model.state_dict(), os.path.join(config["log_dir"], "best_model.pth"))

    print("Training complete.")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the galaxy orientation model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Path to a YAML config file.",
    )
    args = parser.parse_args()

    train(load_config(args.config))
