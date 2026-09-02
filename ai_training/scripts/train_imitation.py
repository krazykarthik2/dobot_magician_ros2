import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "demos")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

class DemonstrationDataset(Dataset):
    def __init__(self, data_dir):
        files = glob.glob(os.path.join(data_dir, "demo_*.npz"))
        if not files:
            raise ValueError(f"No demonstration files found in {data_dir}. Generate demos first!")
        
        all_obs = []
        all_acts = []
        for f in sorted(files):
            data = np.load(f)
            all_obs.append(data['observations'])
            all_acts.append(data['actions'])
            
        self.observations = np.concatenate(all_obs, axis=0)
        self.actions = np.concatenate(all_acts, axis=0)
        
        # Compute mean & std normalization statistics
        self.obs_mean = np.mean(self.observations, axis=0)
        self.obs_std = np.std(self.observations, axis=0) + 1e-6
        
        self.act_mean = np.mean(self.actions, axis=0)
        self.act_std = np.std(self.actions, axis=0) + 1e-6
        
        # Save normalization statistics
        stats_path = os.path.join(MODEL_DIR, "norm_stats.npz")
        np.savez(stats_path, obs_mean=self.obs_mean, obs_std=self.obs_std, act_mean=self.act_mean, act_std=self.act_std)
        print(f"Saved normalization statistics -> {stats_path}")

        # Normalized data
        self.norm_obs = (self.observations - self.obs_mean) / self.obs_std
        self.norm_acts = (self.actions - self.act_mean) / self.act_std
        
        print(f"Loaded {len(files)} demonstrations: total {len(self.observations)} transition frames.")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.norm_obs[idx], dtype=torch.float32),
            torch.tensor(self.norm_acts[idx], dtype=torch.float32)
        )

class DobotResidualPolicy(nn.Module):
    """Deep Residual MLP Policy with LayerNorm and GELU activations."""
    def __init__(self, obs_dim=18, act_dim=5, hidden_dim=256):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Residual Block 1
        self.res1_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.res1_ln1 = nn.LayerNorm(hidden_dim)
        self.res1_gelu = nn.GELU()
        self.res1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.res1_ln2 = nn.LayerNorm(hidden_dim)
        
        # Residual Block 2
        self.res2_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.res2_ln1 = nn.LayerNorm(hidden_dim)
        self.res2_gelu = nn.GELU()
        self.res2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.res2_ln2 = nn.LayerNorm(hidden_dim)

        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, act_dim)
        )

    def forward(self, x):
        h = self.in_proj(x)
        # ResBlock 1
        r1 = self.res1_gelu(self.res1_ln1(self.res1_fc1(h)))
        h = h + self.res1_ln2(self.res1_fc2(r1))
        # ResBlock 2
        r2 = self.res2_gelu(self.res2_ln1(self.res2_fc1(h)))
        h = h + self.res2_ln2(self.res2_fc2(r2))
        
        return self.out_head(h)

def train(epochs=150, batch_size=128, lr=3e-4):
    print("=" * 65)
    print("   Dobot Magician Deep Residual Imitation Learning")
    print("=" * 65)
    
    dataset = DemonstrationDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    obs_dim = dataset.observations.shape[1]
    act_dim = dataset.actions.shape[1]
    
    model = DobotResidualPolicy(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=256)
    
    # Check if existing checkpoint exists and load weights safely
    model_path = os.path.join(MODEL_DIR, "dobot_bc_policy.pth")
    resumed_training = False
    
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            model_dict = model.state_dict()
            
            # Verify layer shapes & parameter counts
            compatible_dict = {}
            for k, v in checkpoint.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    compatible_dict[k] = v
                else:
                    print(f" [!] Skipping incompatible parameter layer: {k}")

            if len(compatible_dict) == len(model_dict):
                model.load_state_dict(compatible_dict)
                print(f">> [CHECKPOINT RESUME] Successfully loaded 100% weights from: {os.path.basename(model_path)}")
                resumed_training = True
            elif len(compatible_dict) > 0:
                model_dict.update(compatible_dict)
                model.load_state_dict(model_dict)
                print(f">> [WARM START] Loaded {len(compatible_dict)}/{len(model_dict)} layers from existing checkpoint.")
                resumed_training = True
            else:
                print(">> [INFO] Existing checkpoint architecture mismatch. Training from fresh initialization.")
        except Exception as e:
            print(f">> [WARNING] Could not load existing checkpoint ({e}). Starting fresh.")
    else:
        print(">> [INFO] No existing checkpoint found. Training fresh model.")

    optimizer = optim.AdamW(model.parameters(), lr=lr if resumed_training else 5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.SmoothL1Loss()

    mode_str = "Fine-tuning from checkpoint" if resumed_training else "Training from scratch"
    print(f"\n>> {mode_str} on CPU across {len(dataset)} transitions ({epochs} epochs)...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for obs_batch, act_batch in dataloader:
            optimizer.zero_grad()
            pred_act = model(obs_batch)
            loss = criterion(pred_act, act_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(obs_batch)

        scheduler.step()
        avg_loss = total_loss / len(dataset)
        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:03d}/{epochs}] - Huber Loss: {avg_loss:.7f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save updated checkpoint
    torch.save(model.state_dict(), model_path)
    print(f"\n[SUCCESS] Updated checkpoint saved -> {model_path}")

if __name__ == "__main__":
    train()
