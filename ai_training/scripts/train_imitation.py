import os
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "demos")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

WINDOW_SIZE = 8   # Past observation history sequence (T_obs = 8)
CHUNK_SIZE = 8    # Future action prediction horizon (H_action = 8)

class ActionChunkingDataset(Dataset):
    """
    Action Chunking Dataset:
    Inputs: Sequence of past observations [T_obs=8, obs_dim=11]
    Targets: Future Action Trajectory Chunk [H_action=8, act_dim=6]
             where act = [dx, dy, dz, dyaw, gripper_cmd, success_signal]
    """
    def __init__(self, data_dir, window_size=WINDOW_SIZE, chunk_size=CHUNK_SIZE):
        self.window_size = window_size
        self.chunk_size = chunk_size
        files = sorted(glob.glob(os.path.join(data_dir, "demo_*.npz")))
        if not files:
            raise ValueError(f"No demonstration files found in {data_dir}. Generate demos first!")

        episodes_obs = []
        episodes_act = []

        all_obs_flat = []
        all_motion_flat = []

        for f in files:
            data = np.load(f)
            obs = data['observations']  # [N, 11]
            act = data['actions']       # [N, 6] (or [N, 5])
            if act.shape[1] == 5:
                # Add default zero success column if old format
                succ_col = np.zeros((len(act), 1), dtype=np.float32)
                act = np.concatenate([act, succ_col], axis=-1)

            episodes_obs.append(obs)
            episodes_act.append(act)

            all_obs_flat.append(obs)
            all_motion_flat.append(act[:, :4])

        all_obs_concat = np.concatenate(all_obs_flat, axis=0)
        all_motion_concat = np.concatenate(all_motion_flat, axis=0)

        self.obs_mean = np.mean(all_obs_concat, axis=0)
        self.obs_std = np.std(all_obs_concat, axis=0) + 1e-6

        self.motion_mean = np.mean(all_motion_concat, axis=0)
        self.motion_std = np.std(all_motion_concat, axis=0) + 1e-6

        stats_path = os.path.join(MODEL_DIR, "norm_stats.npz")
        np.savez(
            stats_path,
            obs_mean=self.obs_mean,
            obs_std=self.obs_std,
            motion_mean=self.motion_mean,
            motion_std=self.motion_std,
            window_size=self.window_size,
            chunk_size=self.chunk_size
        )
        print(f"Saved Action Chunking normalization statistics -> {stats_path}")

        self.samples = []
        for obs_ep, act_ep in zip(episodes_obs, episodes_act):
            norm_obs_ep = (obs_ep - self.obs_mean) / self.obs_std
            
            motion_ep = act_ep[:, :4]
            discrete_ep = act_ep[:, 4:6] # [N, 2: gripper_cmd, success_signal]
            norm_motion_ep = (motion_ep - self.motion_mean) / self.motion_std
            norm_act_ep = np.concatenate([norm_motion_ep, discrete_ep], axis=-1)

            ep_len = len(obs_ep)
            for t in range(ep_len):
                # 1. Past observation window [window_size, 11]
                start_idx = max(0, t - window_size + 1)
                window_obs = norm_obs_ep[start_idx : t + 1]
                if len(window_obs) < window_size:
                    pad = np.repeat(norm_obs_ep[0:1], window_size - len(window_obs), axis=0)
                    window_obs = np.concatenate([pad, window_obs], axis=0)

                # 2. Future action chunk [chunk_size, 6]
                end_idx = min(ep_len, t + chunk_size)
                chunk_act = norm_act_ep[t:end_idx]
                if len(chunk_act) < chunk_size:
                    pad_act = np.repeat(norm_act_ep[-1:], chunk_size - len(chunk_act), axis=0)
                    chunk_act = np.concatenate([chunk_act, pad_act], axis=0)

                self.samples.append((
                    window_obs.astype(np.float32),
                    chunk_act.astype(np.float32)
                ))

        print(f"Loaded {len(files)} episodes -> {len(self.samples)} Action-Chunk samples (Horizon={chunk_size}).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs_seq, chunk_act = self.samples[idx]
        return torch.tensor(obs_seq), torch.tensor(chunk_act)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class DobotActionChunkTransformer(nn.Module):
    """
    Action Chunking Transformer with 3 Multi-Head Outputs:
    - Continuous Velocity Motion Chunk [B, H, 4] -> dx, dy, dz, dyaw
    - Binary Gripper Logits Chunk [B, H, 1] -> Grasp command
    - Task Success Classification Logit [B, 1] -> Self-evaluated task completion
    """
    def __init__(self, obs_dim=11, chunk_size=CHUNK_SIZE, d_model=128, nhead=4, num_layers=3, dim_feedforward=256):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.05,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 1. Action Chunking Trajectory Head
        self.chunk_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, chunk_size * 5)  # H * (4 motion + 1 gripper)
        )

        # 2. Self-Evaluated Success Head (1 scalar logit for whole episode state)
        self.success_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Linear(dim_feedforward // 2, 1)
        )

    def forward(self, x_seq):
        tokens = self.input_proj(x_seq)
        tokens = self.pos_enc(tokens)

        trans_out = self.transformer(tokens)  # [B, T_obs, d_model]
        current_rep = trans_out[:, -1, :]     # [B, d_model]

        flat_chunk = self.chunk_head(current_rep) # [B, H * 5]
        chunk_out = flat_chunk.view(-1, self.chunk_size, 5) # [B, H, 5]

        motion_chunk = chunk_out[:, :, :4]    # [B, H, 4: dx, dy, dz, dyaw]
        grip_chunk_logits = chunk_out[:, :, 4:5] # [B, H, 1: gripper logit]
        
        success_logit = self.success_head(current_rep) # [B, 1: task completed logit]
        return motion_chunk, grip_chunk_logits, success_logit

def train(epochs=180, batch_size=128, lr=5e-4):
    print("=" * 65)
    print("   Dobot Action Chunking Transformer Policy Training")
    print("   (With Self-Evaluated Success Head & Randomized Platform)")
    print("=" * 65)

    dataset = ActionChunkingDataset(DATA_DIR, window_size=WINDOW_SIZE, chunk_size=CHUNK_SIZE)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    obs_dim = dataset.samples[0][0].shape[-1]
    model = DobotActionChunkTransformer(obs_dim=obs_dim, chunk_size=CHUNK_SIZE, d_model=128, nhead=4, num_layers=3)

    model_path = os.path.join(MODEL_DIR, "dobot_bc_policy.pth")
    if os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location="cpu")
            model_dict = model.state_dict()
            compat = {k: v for k, v in ckpt.items() if k in model_dict and model_dict[k].shape == v.shape}
            if len(compat) == len(model_dict):
                model.load_state_dict(compat)
                print(f">> [RESUME] Loaded 100% ACT weights from: {os.path.basename(model_path)}")
            elif len(compat) > 0:
                model_dict.update(compat)
                model.load_state_dict(model_dict)
                print(f">> [WARM START] Loaded {len(compat)}/{len(model_dict)} layers.")
        except Exception as e:
            print(f">> [INFO] Initializing fresh ACT Transformer.")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    huber_loss_fn = nn.SmoothL1Loss(reduction='none')
    bce_loss_fn = nn.BCEWithLogitsLoss()
    axis_weights = torch.tensor([1.0, 1.0, 4.0, 1.0], dtype=torch.float32)

    print(f"\n>> Training ACT Model on CPU across {len(dataset)} Action-Chunks ({epochs} epochs)...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_motion = 0.0
        total_grip = 0.0
        total_succ = 0.0

        for seq_batch, target_chunk in dataloader:
            optimizer.zero_grad()
            pred_motion_chunk, pred_grip_chunk, pred_succ = model(seq_batch)

            target_motion = target_chunk[:, :, :4]
            target_grip = target_chunk[:, :, 4:5]
            target_succ = target_chunk[:, -1, 5:6]  # Success flag at current frame

            # Weighted motion Huber loss across all chunk timesteps
            raw_motion_loss = huber_loss_fn(pred_motion_chunk, target_motion) # [B, H, 4]
            weighted_motion_loss = (raw_motion_loss * axis_weights).mean()

            # Binary Cross Entropy for gripper across chunk
            grip_loss = bce_loss_fn(pred_grip_chunk, target_grip)
            
            # Binary Cross Entropy for self-evaluated task success
            succ_loss = bce_loss_fn(pred_succ, target_succ)

            loss = weighted_motion_loss + 3.0 * grip_loss + 2.0 * succ_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(seq_batch)
            total_motion += weighted_motion_loss.item() * len(seq_batch)
            total_grip += grip_loss.item() * len(seq_batch)
            total_succ += succ_loss.item() * len(seq_batch)

        scheduler.step()
        avg_loss = total_loss / len(dataset)
        avg_motion = total_motion / len(dataset)
        avg_grip = total_grip / len(dataset)
        avg_succ = total_succ / len(dataset)

        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:03d}/{epochs}] - Total: {avg_loss:.5f} | Motion: {avg_motion:.5f} | Grip: {avg_grip:.5f} | Succ: {avg_succ:.5f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    torch.save(model.state_dict(), model_path)
    print(f"\n[SUCCESS] Action Chunking Policy saved -> {model_path}")

if __name__ == "__main__":
    train()
