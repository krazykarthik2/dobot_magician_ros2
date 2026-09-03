import os
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Maximize CPU parallelism across all available cores
NUM_THREADS = min(8, os.cpu_count() or 4)
torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(NUM_THREADS)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "demos")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

WINDOW_SIZE = 8   # Sequence window (T_obs = 8 past states)
CHUNK_SIZE = 8    # Future trajectory chunk horizon (H_action = 8)

# -----------------------------------------------------------------------------
# 1. Expanded Language Vocabulary for Multi-Object & Color Grounding
# -----------------------------------------------------------------------------
VOCAB = [
    "<pad>", "<unk>", "pick", "up", "the", "cube", "block", "object",
    "and", "place", "it", "on", "platform", "box", "target", "grasp", "move", "to", "transfer", "onto",
    # Color tokens for visual grounding
    "red", "blue", "yellow", "green", "purple", "orange", "cyan"
]
WORD_TO_IDX = {w: i for i, w in enumerate(VOCAB)}
MAX_PROMPT_LEN = 14

def tokenize_prompt(prompt_text, max_len=MAX_PROMPT_LEN):
    tokens = prompt_text.lower().replace(".", "").replace(",", "").split()
    indices = [WORD_TO_IDX.get(t, WORD_TO_IDX["<unk>"]) for t in tokens][:max_len]
    while len(indices) < max_len:
        indices.append(WORD_TO_IDX["<pad>"])
    return np.array(indices, dtype=np.int64)

# -----------------------------------------------------------------------------
# 2. High-Speed Pre-Tensorized Multimodal Dataset
# -----------------------------------------------------------------------------
class SmolVLAMultimodalDataset(Dataset):
    """
    High-Speed In-Memory Pre-Tensorized Dataset:
    Zero runtime per-item tensor conversion overhead.
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
            
            if 'images' in data and 'proprioception' in data:
                imgs = data['images']               # [N, 3, 64, 64]
                proprio = data['proprioception']    # [N, 5]
            else:
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

        img_list = []
        proprio_list = []
        prompt_list = []
        act_list = []

        for imgs_ep, proprio_ep, act_ep, prompt_tok in zip(episodes_img, episodes_proprio, episodes_act, episodes_prompt):
            norm_proprio_ep = (proprio_ep - self.proprio_mean) / self.proprio_std
            
            motion_ep = act_ep[:, :4]
            discrete_ep = act_ep[:, 4:6]
            norm_motion_ep = (motion_ep - self.motion_mean) / self.motion_std
            norm_act_ep = np.concatenate([norm_motion_ep, discrete_ep], axis=-1)

            ep_len = len(proprio_ep)
            for t in range(ep_len):
                img_t = imgs_ep[t]

                start_idx = max(0, t - window_size + 1)
                window_proprio = norm_proprio_ep[start_idx : t + 1]
                if len(window_proprio) < window_size:
                    pad = np.repeat(norm_proprio_ep[0:1], window_size - len(window_proprio), axis=0)
                    window_proprio = np.concatenate([pad, window_proprio], axis=0)

                end_idx = min(ep_len, t + chunk_size)
                chunk_act = norm_act_ep[t:end_idx]
                if len(chunk_act) < chunk_size:
                    pad_act = np.repeat(norm_act_ep[-1:], chunk_size - len(chunk_act), axis=0)
                    chunk_act = np.concatenate([chunk_act, pad_act], axis=0)

                img_list.append(img_t)
                proprio_list.append(window_proprio)
                prompt_list.append(prompt_tok)
                act_list.append(chunk_act)

        # Pre-convert everything into single contiguous PyTorch tensors in memory
        self.imgs_tensor = torch.tensor(np.array(img_list, dtype=np.float32), dtype=torch.float32)
        self.proprio_tensor = torch.tensor(np.array(proprio_list, dtype=np.float32), dtype=torch.float32)
        self.prompt_tensor = torch.tensor(np.array(prompt_list, dtype=np.int64), dtype=torch.int64)
        self.acts_tensor = torch.tensor(np.array(act_list, dtype=np.float32), dtype=torch.float32)

        print(f"Loaded {len(files)} episodes -> {len(self.imgs_tensor)} Multimodal SmolVLA samples into RAM.")

    def __len__(self):
        return len(self.imgs_tensor)

    def __getitem__(self, idx):
        return self.imgs_tensor[idx], self.proprio_tensor[idx], self.prompt_tensor[idx], self.acts_tensor[idx]


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
        x = self.conv(img)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


class ActionExpertCrossAttentionBlock(nn.Module):
    """
    Action Expert Transformer Block (SmolVLA / Pi0):
    - Causal Self-Attention over future action tokens
    - Cross-Attention over VLM layer tokens
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
        sa_out, _ = self.self_attn(act_tokens, act_tokens, act_tokens)
        act_tokens = self.norm1(act_tokens + sa_out)

        ca_out, _ = self.cross_attn(query=act_tokens, key=vlm_layer_feat, value=vlm_layer_feat)
        act_tokens = self.norm2(act_tokens + ca_out)

        ffn_out = self.ffn(act_tokens)
        act_tokens = self.norm3(act_tokens + ffn_out)
        return act_tokens


class SmolVLAPolicy(nn.Module):
    """
    Complete SmolVLA / Pi0 Vision-Language-Action Policy
    """
    def __init__(self, vocab_size=len(VOCAB), chunk_size=CHUNK_SIZE, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model

        self.vision_encoder = VisionPatchEncoder(in_channels=3, d_model=d_model, patch_size=16)
        self.lang_embedding = nn.Embedding(vocab_size, d_model)
        self.proprio_proj = nn.Linear(5, d_model)

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

        self.action_queries = nn.Parameter(torch.randn(1, chunk_size, d_model) * 0.02)
        self.action_expert_layers = nn.ModuleList([
            ActionExpertCrossAttentionBlock(d_model=d_model, nhead=nhead, dim_feedforward=256)
            for _ in range(num_layers)
        ])

        self.fusion_proj = nn.Linear(d_model * num_layers, d_model)

        self.motion_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 4)
        )

        self.gripper_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1)
        )

        self.success_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1)
        )

    def forward(self, img, proprio_seq, prompt_tokens):
        batch_size = img.size(0)

        vis_tokens = self.vision_encoder(img)
        lang_tokens = self.lang_embedding(prompt_tokens)
        proprio_tokens = self.proprio_proj(proprio_seq)

        multimodal_seq = torch.cat([lang_tokens, vis_tokens, proprio_tokens], dim=1)

        vlm_all_layers = []
        h = multimodal_seq
        for layer in self.vlm_layers:
            h = layer(h)
            vlm_all_layers.append(h)

        act_tokens = self.action_queries.expand(batch_size, -1, -1)
        for i, expert_block in enumerate(self.action_expert_layers):
            layer_feat = vlm_all_layers[i]
            act_tokens = expert_block(act_tokens, layer_feat)

        motion_chunk = self.motion_head(act_tokens)
        grip_chunk_logits = self.gripper_head(act_tokens)

        fused_vlm = torch.cat(vlm_all_layers, dim=-1)
        global_context = self.fusion_proj(fused_vlm).mean(dim=1)
        success_logit = self.success_head(global_context)

        return motion_chunk, grip_chunk_logits, success_logit


# Alias for backward compatibility
DobotActionChunkTransformer = SmolVLAPolicy


def train(epochs=120, batch_size=256, lr=7e-4):
    print("=" * 68)
    print("   SmolVLA / Pi0 High-Speed Multimodal Policy Training")
    print(f"   (Optimized: Batch Size={batch_size}, CPU Threads={NUM_THREADS}, Pre-Tensorized RAM)")
    print("=" * 68)

    dataset = SmolVLAMultimodalDataset(DATA_DIR, window_size=WINDOW_SIZE, chunk_size=CHUNK_SIZE)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

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

    print(f"\n>> Accelerating Training across {len(dataset)} samples ({epochs} epochs)...")

    best_loss = float('inf')

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            total_motion = 0.0
            total_grip = 0.0
            total_succ = 0.0

            for img_batch, proprio_batch, prompt_batch, target_chunk in dataloader:
                optimizer.zero_grad(set_to_none=True)
                pred_motion_chunk, pred_grip_chunk, pred_succ = model(img_batch, proprio_batch, prompt_batch)

                target_motion = target_chunk[:, :, :4]
                target_grip = target_chunk[:, :, 4:5]
                target_succ = target_chunk[:, -1, 5:6]

                raw_motion_loss = huber_loss_fn(pred_motion_chunk, target_motion)
                weighted_motion_loss = (raw_motion_loss * axis_weights).mean()

                grip_loss = bce_loss_fn(pred_grip_chunk, target_grip)
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

            if avg_loss < best_loss or epoch % 10 == 0:
                best_loss = min(best_loss, avg_loss)
                torch.save(model.state_dict(), model_path)

            if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
                print(f"Epoch [{epoch:03d}/{epochs}] - Total: {avg_loss:.5f} | Motion: {avg_motion:.5f} | Grip: {avg_grip:.5f} | Succ: {avg_succ:.5f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    except KeyboardInterrupt:
        print("\n\n[INFO] Training interrupted by user! Saving current checkpoint...")
        torch.save(model.state_dict(), model_path)
        print(f"[SAVED] Checkpoint saved successfully before exiting -> {model_path}")
        return

    torch.save(model.state_dict(), model_path)
    print(f"\n[SUCCESS] Multimodal SmolVLA Policy saved -> {model_path}")

if __name__ == "__main__":
    train()
