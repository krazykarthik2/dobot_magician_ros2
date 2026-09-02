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

# -----------------------------------------------------------------------------
# 1. Simple Robust Character/Word Language Tokenizer
# -----------------------------------------------------------------------------
VOCAB = [
    "<pad>", "<unk>", "pick", "up", "the", "red", "cube", "block", "object",
    "and", "place", "it", "on", "green", "platform", "box", "target",
    "grasp", "move", "to", "transfer", "onto"
]
WORD_TO_IDX = {w: i for i, w in enumerate(VOCAB)}
MAX_PROMPT_LEN = 12

def tokenize_prompt(prompt_text, max_len=MAX_PROMPT_LEN):
    tokens = prompt_text.lower().replace(".", "").replace(",", "").split()
    indices = [WORD_TO_IDX.get(t, WORD_TO_IDX["<unk>"]) for t in tokens][:max_len]
    while len(indices) < max_len:
        indices.append(WORD_TO_IDX["<pad>"])
    return np.array(indices, dtype=np.int64)

# -----------------------------------------------------------------------------
# 2. Multimodal VLA Dataset (Vision + Language + Proprioception)
# -----------------------------------------------------------------------------
class SmolVLAMultimodalDataset(Dataset):
    """
    Multimodal Dataset for SmolVLA / Pi0:
    Inputs:
      - RGB Camera Frame Image: [3, 64, 64]
      - Past Proprioception Window: [T_obs=8, 5] (EE pos + gripper)
      - Tokenized Language Instruction: [max_len=12]
    Targets:
      - Future Action Trajectory Chunk: [H_action=8, 6]
        [dx, dy, dz, dyaw, gripper_cmd, success_signal]
    """
    def __init__(self, data_dir, window_size=WINDOW_SIZE, chunk_size=CHUNK_SIZE):
        self.window_size = window_size
        self.chunk_size = chunk_size
        files = sorted(glob.glob(os.path.join(data_dir, "demo_*.npz")))
        if not files:
            raise ValueError(f"No demonstration files found in {data_dir}. Generate demos first!")

        episodes_img = []
        episodes_proprio = []
        episodes_act = []
        episodes_prompt = []

        all_proprio_flat = []
        all_motion_flat = []

        for f in files:
            data = np.load(f, allow_pickle=True)
            
            # Check if multimodal keys exist
            if 'images' in data and 'proprioception' in data:
                imgs = data['images']               # [N, 3, 64, 64]
                proprio = data['proprioception']    # [N, 5]
            else:
                # Fallback synthesized image from 11-dim observation
                obs = data['observations']
                N = len(obs)
                imgs = np.zeros((N, 3, 64, 64), dtype=np.float32)
                proprio = obs[:, :5]

            act = data['actions']                   # [N, 6] (or [N, 5])
            if act.shape[1] == 5:
                succ_col = np.zeros((len(act), 1), dtype=np.float32)
                act = np.concatenate([act, succ_col], axis=-1)

            prompt_str = str(data['prompt'][0]) if 'prompt' in data else "pick up the red cube and place it on the green platform"

            episodes_img.append(imgs)
            episodes_proprio.append(proprio)
            episodes_act.append(act)
            episodes_prompt.append(tokenize_prompt(prompt_str))

            all_proprio_flat.append(proprio)
            all_motion_flat.append(act[:, :4])

        all_proprio_concat = np.concatenate(all_proprio_flat, axis=0)
        all_motion_concat = np.concatenate(all_motion_flat, axis=0)

        self.proprio_mean = np.mean(all_proprio_concat, axis=0)
        self.proprio_std = np.std(all_proprio_concat, axis=0) + 1e-6

        self.motion_mean = np.mean(all_motion_concat, axis=0)
        self.motion_std = np.std(all_motion_concat, axis=0) + 1e-6

        stats_path = os.path.join(MODEL_DIR, "norm_stats.npz")
        np.savez(
            stats_path,
            proprio_mean=self.proprio_mean,
            proprio_std=self.proprio_std,
            motion_mean=self.motion_mean,
            motion_std=self.motion_std,
            window_size=self.window_size,
            chunk_size=self.chunk_size
        )
        print(f"Saved SmolVLA normalization statistics -> {stats_path}")

        self.samples = []
        for imgs_ep, proprio_ep, act_ep, prompt_tok in zip(episodes_img, episodes_proprio, episodes_act, episodes_prompt):
            norm_proprio_ep = (proprio_ep - self.proprio_mean) / self.proprio_std
            
            motion_ep = act_ep[:, :4]
            discrete_ep = act_ep[:, 4:6]
            norm_motion_ep = (motion_ep - self.motion_mean) / self.motion_std
            norm_act_ep = np.concatenate([norm_motion_ep, discrete_ep], axis=-1)

            ep_len = len(proprio_ep)
            for t in range(ep_len):
                # 1. Current RGB camera frame [3, 64, 64]
                img_t = imgs_ep[t]

                # 2. Past proprioception sequence [T_obs=8, 5]
                start_idx = max(0, t - window_size + 1)
                window_proprio = norm_proprio_ep[start_idx : t + 1]
                if len(window_proprio) < window_size:
                    pad = np.repeat(norm_proprio_ep[0:1], window_size - len(window_proprio), axis=0)
                    window_proprio = np.concatenate([pad, window_proprio], axis=0)

                # 3. Future action trajectory chunk [H=8, 6]
                end_idx = min(ep_len, t + chunk_size)
                chunk_act = norm_act_ep[t:end_idx]
                if len(chunk_act) < chunk_size:
                    pad_act = np.repeat(norm_act_ep[-1:], chunk_size - len(chunk_act), axis=0)
                    chunk_act = np.concatenate([chunk_act, pad_act], axis=0)

                self.samples.append((
                    img_t.astype(np.float32),
                    window_proprio.astype(np.float32),
                    prompt_tok.astype(np.int64),
                    chunk_act.astype(np.float32)
                ))

        print(f"Loaded {len(files)} episodes -> {len(self.samples)} Multimodal SmolVLA samples (Horizon={chunk_size}).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, proprio_seq, prompt, chunk_act = self.samples[idx]
        return torch.tensor(img), torch.tensor(proprio_seq), torch.tensor(prompt), torch.tensor(chunk_act)


# -----------------------------------------------------------------------------
# 3. SmolVLM-2 Perception Backbone with Multi-Layer Extraction & Action Expert
# -----------------------------------------------------------------------------

class VisionPatchEncoder(nn.Module):
    """
    Lightweight Vision Patch Tokenizer for SmolVLM-2:
    Takes [B, 3, 64, 64] RGB Image -> Conv/Patch Projection -> [B, N_patches=16, d_model=128]
    """
    def __init__(self, in_channels=3, d_model=128, patch_size=16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, img):
        # img: [B, 3, 64, 64]
        x = self.conv(img) # [B, d_model, 4, 4]
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # [B, 16, d_model]
        return self.norm(x)


class ActionExpertCrossAttentionBlock(nn.Module):
    """
    Action Expert Transformer Block (SmolVLA / Pi0):
    - Causal Self-Attention over future action tokens (ensures trajectory smoothness)
    - Cross-Attention over VLM layer tokens (conditions actions on vision + language)
    - Feed-Forward MLP
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
        # 1. Self-attention over action chunk tokens
        sa_out, _ = self.self_attn(act_tokens, act_tokens, act_tokens)
        act_tokens = self.norm1(act_tokens + sa_out)

        # 2. Cross-attention over specific VLM layer features (SmolVLA Multi-Layer Fusion)
        ca_out, _ = self.cross_attn(query=act_tokens, key=vlm_layer_feat, value=vlm_layer_feat)
        act_tokens = self.norm2(act_tokens + ca_out)

        # 3. Feed-forward
        ffn_out = self.ffn(act_tokens)
        act_tokens = self.norm3(act_tokens + ffn_out)
        return act_tokens


class SmolVLAPolicy(nn.Module):
    """
    Complete SmolVLA / Pi0 Vision-Language-Action Policy:
    1. Vision Encoder: Raw RGB Camera Image [3, 64, 64] -> Visual Tokens [B, 16, D]
    2. Language Embedder: Instruction Prompt -> Text Tokens [B, 12, D]
    3. Proprioception Projector: Robot State History [T_obs=8, 5] -> State Tokens [B, 8, D]
    4. SmolVLM-2 Perception Backbone: Multi-Modal Transformer extracting representations across all layers.
    5. Action Expert: Cross-Attention Decoder over all intermediate VLM layers.
    6. Multi-Head Action & Success Output:
       - Continuous Motion Chunk [B, H=8, 4] (dx, dy, dz, dyaw)
       - Discrete Gripper Chunk [B, H=8, 1]
       - Self-Evaluated Task Success [B, 1]
    """
    def __init__(self, vocab_size=len(VOCAB), chunk_size=CHUNK_SIZE, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model

        # 1. Modality Encoders
        self.vision_encoder = VisionPatchEncoder(in_channels=3, d_model=d_model, patch_size=16)
        self.lang_embedding = nn.Embedding(vocab_size, d_model)
        self.proprio_proj = nn.Linear(5, d_model)

        # 2. Multi-Modal Perception Transformer (SmolVLM-2 style)
        self.vlm_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=256,
                dropout=0.05,
                activation="gelu",
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # 3. Action Expert Transformer (SmolVLA Action Expert)
        self.action_queries = nn.Parameter(torch.randn(1, chunk_size, d_model) * 0.02)
        self.action_expert_layers = nn.ModuleList([
            ActionExpertCrossAttentionBlock(d_model=d_model, nhead=nhead, dim_feedforward=256)
            for _ in range(num_layers)
        ])

        # 4. Multi-Layer Feature Fusion
        self.fusion_proj = nn.Linear(d_model * num_layers, d_model)

        # 5. Output Heads
        self.motion_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 4) # dx, dy, dz, dyaw
        )

        self.gripper_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1) # Gripper logit
        )

        self.success_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1) # Self-evaluated task completion
        )

    def forward(self, img, proprio_seq, prompt_tokens):
        # img: [B, 3, 64, 64]
        # proprio_seq: [B, T_obs=8, 5]
        # prompt_tokens: [B, max_len=12]
        batch_size = img.size(0)

        # Modality Token Projections
        vis_tokens = self.vision_encoder(img)              # [B, 16, D]
        lang_tokens = self.lang_embedding(prompt_tokens)   # [B, 12, D]
        proprio_tokens = self.proprio_proj(proprio_seq)    # [B, 8, D]

        # Multi-Modal Prefix Sequence: [Language + Vision + Proprioception]
        multimodal_seq = torch.cat([lang_tokens, vis_tokens, proprio_tokens], dim=1) # [B, 36, D]

        # Step 1: Extract all layer outputs from SmolVLM-2 Perception Backbone
        vlm_all_layers = []
        h = multimodal_seq
        for layer in self.vlm_layers:
            h = layer(h)
            vlm_all_layers.append(h)

        # Step 2: Action Expert queries cross-attend to each VLM layer
        act_tokens = self.action_queries.expand(batch_size, -1, -1) # [B, H, D]
        for i, expert_block in enumerate(self.action_expert_layers):
            layer_feat = vlm_all_layers[i]
            act_tokens = expert_block(act_tokens, layer_feat)

        # Step 3: Multi-Head Action Predictions
        motion_chunk = self.motion_head(act_tokens)       # [B, H, 4]
        grip_chunk_logits = self.gripper_head(act_tokens) # [B, H, 1]

        # Success evaluated from fused VLM multimodal context
        fused_vlm = torch.cat(vlm_all_layers, dim=-1) # [B, 36, D * num_layers]
        global_context = self.fusion_proj(fused_vlm).mean(dim=1) # [B, D]
        success_logit = self.success_head(global_context) # [B, 1]

        return motion_chunk, grip_chunk_logits, success_logit


# Alias for backward compatibility
DobotActionChunkTransformer = SmolVLAPolicy


def train(epochs=180, batch_size=128, lr=5e-4):
    print("=" * 68)
    print("   SmolVLA / Pi0 Multimodal Generalist Policy Training")
    print("   (Vision 64x64 + Language Instruction + 5D Proprioception)")
    print("=" * 68)

    dataset = SmolVLAMultimodalDataset(DATA_DIR, window_size=WINDOW_SIZE, chunk_size=CHUNK_SIZE)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SmolVLAPolicy(vocab_size=len(VOCAB), chunk_size=CHUNK_SIZE, d_model=128, nhead=4, num_layers=3)

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

        for img_batch, proprio_batch, prompt_batch, target_chunk in dataloader:
            optimizer.zero_grad()
            pred_motion_chunk, pred_grip_chunk, pred_succ = model(img_batch, proprio_batch, prompt_batch)

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

            total_loss += loss.item() * len(img_batch)
            total_motion += weighted_motion_loss.item() * len(img_batch)
            total_grip += grip_loss.item() * len(img_batch)
            total_succ += succ_loss.item() * len(img_batch)

        scheduler.step()
        avg_loss = total_loss / len(dataset)
        avg_motion = total_motion / len(dataset)
        avg_grip = total_grip / len(dataset)
        avg_succ = total_succ / len(dataset)

        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:03d}/{epochs}] - Total: {avg_loss:.5f} | Motion: {avg_motion:.5f} | Grip: {avg_grip:.5f} | Succ: {avg_succ:.5f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    torch.save(model.state_dict(), model_path)
    print(f"\n[SUCCESS] Multimodal SmolVLA Policy saved -> {model_path}")

if __name__ == "__main__":
    train()
