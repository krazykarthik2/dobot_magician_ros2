import os
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "demos")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

WINDOW_SIZE = 8   # Sequence window (T_obs = 8 past states)
CHUNK_SIZE = 8    # Future trajectory chunk horizon (H_action = 8)

class SmolVLADataset(Dataset):
    """
    Dataset for SmolVLA / Pi0 Generalist Policy:
    Input:
      - Multi-Modal Scene Observation Sequence [T_obs=8, obs_dim=11]
    Target:
      - Future Action Trajectory Chunk [H_action=8, 6]:
        [dx, dy, dz, dyaw, gripper_cmd, success_signal]
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
        print(f"Saved SmolVLA / Pi0 normalization statistics -> {stats_path}")

        self.samples = []
        for obs_ep, act_ep in zip(episodes_obs, episodes_act):
            norm_obs_ep = (obs_ep - self.obs_mean) / self.obs_std
            
            motion_ep = act_ep[:, :4]
            discrete_ep = act_ep[:, 4:6]
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

        print(f"Loaded {len(files)} episodes -> {len(self.samples)} SmolVLA sequence samples (Horizon={chunk_size}).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs_seq, chunk_act = self.samples[idx]
        return torch.tensor(obs_seq), torch.tensor(chunk_act)


# -----------------------------------------------------------------------------
# SmolVLM-2 / Pi0 Architecture Modules: Multi-Layer Perception Backbone & Action Expert
# -----------------------------------------------------------------------------

class MultiLayerPerceptionBackbone(nn.Module):
    """
    Lightweight SmolVLM-2 style Multi-Layer Transformer Perception Backbone:
    Processes the raw sensorimotor scene tokens (EE, Cube, Platform) across T_obs timesteps.
    Extracts multi-layer intermediate hidden representations across all its layers.
    """
    def __init__(self, in_dim=11, d_model=128, nhead=4, num_layers=3, dim_feedforward=256):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # Transformer Layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=0.05,
                activation="gelu",
                batch_first=True
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # x: [B, T_obs, 11]
        h = self.in_proj(x)
        layer_outputs = []
        for layer in self.layers:
            h = layer(h)
            layer_outputs.append(h) # Collect representations from each layer
        return layer_outputs


class ActionExpertCrossAttentionBlock(nn.Module):
    """
    Action Expert Transformer Block with:
    - Multi-Head Causal Self-Attention (for intra-action trajectory consistency)
    - Multi-Head Cross-Attention (conditions on multi-layer VLM perception features)
    - Feed-Forward MLP with GELU
    """
    def __init__(self, d_model=128, nhead=4, dim_feedforward=256):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.05, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.05, batch_first=True)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, act_tokens, vlm_layer_feat):
        # 1. Self-Attention over action chunk tokens
        sa_out, _ = self.self_attn(act_tokens, act_tokens, act_tokens)
        act_tokens = self.norm1(act_tokens + sa_out)

        # 2. Cross-Attention over VLM layer features (Physical Intelligence Pi0 / SmolVLA mechanism)
        ca_out, _ = self.cross_attn(query=act_tokens, key=vlm_layer_feat, value=vlm_layer_feat)
        act_tokens = self.norm2(act_tokens + ca_out)

        # 3. Feed-forward
        ffn_out = self.ffn(act_tokens)
        act_tokens = self.norm3(act_tokens + ffn_out)
        return act_tokens


class SmolVLAPolicy(nn.Module):
    """
    SmolVLA / Pi0 Generalist Policy Architecture:
    1. VLM Perception Backbone (SmolVLM-2 style): Generates multi-layer scene representations.
    2. Multi-Layer Feature Aggregator: Fuses all intermediate VLM layers.
    3. Action Expert: Interleaved cross-attention blocks decoding the future trajectory chunk H=8.
    4. Multi-Head Output:
       - Continuous Velocity Motion Chunk [B, H, 4] (dx, dy, dz, dyaw)
       - Discrete Gripper Logits Chunk [B, H, 1]
       - Self-Evaluated Success Logit [B, 1]
    """
    def __init__(self, obs_dim=11, chunk_size=CHUNK_SIZE, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model

        # 1. Perception VLM Backbone
        self.vlm_backbone = MultiLayerPerceptionBackbone(
            in_dim=obs_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=256
        )

        # 2. Multi-layer fusion: Projects concatenated multi-layer outputs back to d_model
        self.layer_fusion = nn.Sequential(
            nn.Linear(d_model * num_layers, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # 3. Action Expert: Learned query tokens for H=8 future timesteps
        self.action_queries = nn.Parameter(torch.randn(1, chunk_size, d_model) * 0.02)
        
        self.action_expert_layers = nn.ModuleList([
            ActionExpertCrossAttentionBlock(d_model=d_model, nhead=nhead, dim_feedforward=256)
            for _ in range(num_layers)
        ])

        # 4. Output Heads
        self.motion_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 4)  # dx, dy, dz, dyaw
        )

        self.gripper_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1)  # Gripper logit
        )

        self.success_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1)  # Task success confidence logit
        )

    def forward(self, obs_seq):
        # obs_seq: [B, T_obs, 11]
        batch_size = obs_seq.size(0)

        # Step 1: Extract all layer outputs from VLM perception backbone
        vlm_all_layers = self.vlm_backbone(obs_seq) # List of [B, T_obs, d_model]

        # Step 2: Multi-layer fusion across all VLM layers
        fused_vlm_feats = torch.cat(vlm_all_layers, dim=-1) # [B, T_obs, d_model * num_layers]
        vlm_context = self.layer_fusion(fused_vlm_feats)     # [B, T_obs, d_model]

        # Step 3: Expand learned action queries across batch
        act_tokens = self.action_queries.expand(batch_size, -1, -1) # [B, H, d_model]

        # Step 4: Pass through Action Expert with Cross-Attention over each VLM layer
        for i, expert_block in enumerate(self.action_expert_layers):
            layer_vlm_feat = vlm_all_layers[i] # Layer-specific conditioning (Pi0 / SmolVLA)
            act_tokens = expert_block(act_tokens, layer_vlm_feat)

        # Step 5: Multi-Head Action & Success Predictions
        motion_chunk = self.motion_head(act_tokens)       # [B, H, 4]
        grip_chunk_logits = self.gripper_head(act_tokens) # [B, H, 1]

        # Success evaluated from final VLM context state
        success_logit = self.success_head(vlm_context[:, -1, :]) # [B, 1]

        return motion_chunk, grip_chunk_logits, success_logit


# Alias for backward compatibility
DobotActionChunkTransformer = SmolVLAPolicy


def train(epochs=180, batch_size=128, lr=5e-4):
    print("=" * 68)
    print("   SmolVLA / Pi0 Generalist Policy Training (Multi-Layer VLM Fusion)")
    print("=" * 68)

    dataset = SmolVLADataset(DATA_DIR, window_size=WINDOW_SIZE, chunk_size=CHUNK_SIZE)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    obs_dim = dataset.samples[0][0].shape[-1]
    model = SmolVLAPolicy(obs_dim=obs_dim, chunk_size=CHUNK_SIZE, d_model=128, nhead=4, num_layers=3)

    model_path = os.path.join(MODEL_DIR, "dobot_bc_policy.pth")
    if os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location="cpu")
            model_dict = model.state_dict()
            compat = {k: v for k, v in ckpt.items() if k in model_dict and model_dict[k].shape == v.shape}
            if len(compat) == len(model_dict):
                model.load_state_dict(compat)
                print(f">> [RESUME] Loaded 100% SmolVLA weights from: {os.path.basename(model_path)}")
            elif len(compat) > 0:
                model_dict.update(compat)
                model.load_state_dict(model_dict)
                print(f">> [WARM START] Loaded {len(compat)}/{len(model_dict)} layers.")
        except Exception as e:
            print(f">> [INFO] Initializing fresh SmolVLA Transformer.")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    huber_loss_fn = nn.SmoothL1Loss(reduction='none')
    bce_loss_fn = nn.BCEWithLogitsLoss()
    axis_weights = torch.tensor([1.0, 1.0, 4.0, 1.0], dtype=torch.float32)

    print(f"\n>> Training SmolVLA Model on CPU across {len(dataset)} Action-Chunks ({epochs} epochs)...")

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
            target_succ = target_chunk[:, -1, 5:6]

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
    print(f"\n[SUCCESS] SmolVLA / Pi0 Generalist Policy saved -> {model_path}")

if __name__ == "__main__":
    train()
