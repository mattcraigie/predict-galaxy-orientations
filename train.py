import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Import our custom modules
from dataset import GalaxyDataset
from vmdn import init_vmdn

# --- Configuration ---
CONFIG = {
    "csv_path": "des_metacal_angles_minimal.csv",
    "batch_size": 256,
    "lr": 1e-3,
    "epochs": 100,
    "patience": 10,  # Early stopping patience
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "log_dir": "runs/galaxy_vmdn_experiment_1",

    # Model Config
    "num_neighbors": 5,  # Size of each GNN graph
    "hidden_dim": 64,
    "num_layers": 3,
    "vmdn_regularization": 0.1,  # Lambda kappa
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


def train():
    # 1. Setup Data
    print(f"Loading dataset from {CONFIG['csv_path']}...")
    train_ds = GalaxyDataset(CONFIG['csv_path'], num_neighbors=CONFIG['num_neighbors'], mode='train')
    val_ds = GalaxyDataset(CONFIG['csv_path'], num_neighbors=CONFIG['num_neighbors'], mode='val')

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

    # 2. Setup Model
    model = init_vmdn(CONFIG).to(CONFIG['device'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])

    # 3. Logging & Checkpointing
    writer = SummaryWriter(CONFIG['log_dir'])
    stopper = EarlyStopper(patience=CONFIG['patience'])

    print("Starting training...")

    for epoch in range(CONFIG['epochs']):
        # --- TRAINING LOOP ---
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}")
        for batch in pbar:
            pos = batch['pos'].to(CONFIG['device'])
            z = batch['redshift'].to(CONFIG['device'])
            shapes = batch['input_shapes'].to(CONFIG['device'])
            target = batch['target_phi'].to(CONFIG['device'])

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
                pos = batch['pos'].to(CONFIG['device'])
                z = batch['redshift'].to(CONFIG['device'])
                shapes = batch['input_shapes'].to(CONFIG['device'])
                target = batch['target_phi'].to(CONFIG['device'])

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
            torch.save(model.state_dict(), os.path.join(CONFIG['log_dir'], "best_model.pth"))

    print("Training complete.")
    writer.close()


if __name__ == "__main__":
    train()