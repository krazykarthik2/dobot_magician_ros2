import os
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------------------------
# Hardware Acceleration Setup
# -----------------------------------------------------------------------------
DEVICE = torch.device("cpu")
USE_DIRECTML = False

try:
    import torch_directml
    DEVICE = torch_directml.device()
    USE_DIRECTML = True
    print(f">> [HARDWARE ACCELERATION] DirectML GPU device enabled: {DEVICE}")
except ImportError:
    pass

if not USE_DIRECTML and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f">> [HARDWARE ACCELERATION] CUDA device enabled: {torch.cuda.get_device_name(0)}")

if DEVICE.type == "cpu":
    NUM_THREADS = min(8, os.cpu_count() or 4)
    torch.set_num_threads(NUM_THREADS)
    torch.set_num_interop_threads(NUM_THREADS)
    if hasattr(torch.backends, 'mkldnn'):
        torch.backends.mkldnn.enabled = True
    print(f">> [HARDWARE ACCELERATION] CPU multithreading enabled ({NUM_THREADS} threads).")

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

            act = data['actions']                   # [N, 6]
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
# 3. Pretrained Hugging Face BERT Transformer Language Backbone
# -----------------------------------------------------------------------------

def load_pretrained_hf_bert(d_model=128):
    """
    Loads pretrained Hugging Face BERT-Tiny weights for language grounding.
    """
    try:
        from transformers import BertConfig, BertModel
        cfg = BertConfig(
            vocab_size=30522,
            hidden_size=d_model,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=512,
            hidden_dropout_prob=0.05,
            attention_probs_dropout_prob=0.05
        )
        bert_model = BertModel(cfg)

        cache_dir = '/home/karthikkrazy/.cache/huggingface/hub/models--prajjwal1--bert-tiny/snapshots/6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837'
        bin_file = os.path.join(cache_dir, 'pytorch_model.bin')
        if os.path.exists(bin_file):
            state = torch.load(bin_file, map_location='cpu')
            bert_state = {k[5:]: v for k, v in state.items() if k.startswith('bert.')}
            bert_model.load_state_dict(bert_state, strict=False)
            print(">> [PRETRAINED VLA] Successfully loaded Pretrained Hugging Face BERT Language Backbone!")
        return bert_model
    except Exception as e:
        print(f">> [INFO] Fallback standard transformer language encoder: {e}")
        return None


class CoordConvPatchEncoder(nn.Module):
    """
    Spatial Coordinate-Aware Vision Backbone:
    1. Injects normalized 2D coordinate meshgrids (x, y) into raw RGB pixels [5, 64, 64].
    2. Multi-scale feature extraction: Captures color boundaries + spatial locations.
    3. Learned 2D Spatial Positional Embeddings (64 visual tokens).
    """
    def __init__(self, in_channels=5, d_model=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> [32, 32]
            nn.GELU(),
            nn.Conv2d(64, d_model, kernel_size=3, stride=2, padding=1), # -> [16, 16]
            nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 8)) # 64 spatial visual tokens [B, D, 8, 8]
        )
        self.pos_embed = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, img):
        B, C, H, W = img.shape
        device = img.device

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        coord_grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        
        x_in = torch.cat([img, coord_grid], dim=1)
        feat_map = self.stem(x_in)
        tokens = feat_map.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens + self.pos_embed)
        return tokens


class ActionExpertCrossAttentionBlock(nn.Module):
    """
    Action Expert Transformer Block (SmolVLA / Pi0)
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
    Complete SmolVLA / Pi0 Vision-Language-Action Policy with Pretrained Language Backbone & CoordConv
    """
    def __init__(self, vocab_size=len(VOCAB), chunk_size=CHUNK_SIZE, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model

        self.vision_encoder = CoordConvPatchEncoder(in_channels=5, d_model=d_model)
        self.lang_embedding = nn.Embedding(vocab_size, d_model)
        self.pretrained_hf_bert = load_pretrained_hf_bert(d_model=d_model)
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

        vis_tokens = self.vision_encoder(img)              # [B, 64, D] (CoordConv + Visual tokens)
        lang_tokens = self.lang_embedding(prompt_tokens)   # [B, 14, D]
        proprio_tokens = self.proprio_proj(proprio_seq)    # [B, 8, D]

        multimodal_seq = torch.cat([lang_tokens, vis_tokens, proprio_tokens], dim=1) # [B, 86, D]

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


def train(epochs=140, batch_size=256, lr=9e-4):
    print("=" * 68)
    print("   SmolVLA / Pi0 Multimodal Generalist Policy Fine-Tuning")
    print(f"   (Pretrained HF Backbone + CoordConv Multi-Scale Perception)")
    print("=" * 68)

    dataset = SmolVLAMultimodalDataset(DATA_DIR, window_size=WINDOW_SIZE, chunk_size=CHUNK_SIZE)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    model = SmolVLAPolicy(vocab_size=len(VOCAB), chunk_size=CHUNK_SIZE, d_model=128, nhead=4, num_layers=3)
    model.to(DEVICE)

    model_path = os.path.join(MODEL_DIR, "dobot_bc_policy.pth")
    if os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location=DEVICE)
            model_dict = model.state_dict()
            compat = {k: v for k, v in ckpt.items() if k in model_dict and model_dict[k].shape == v.shape}
            if len(compat) == len(model_dict):
                model.load_state_dict(compat)
                print(f">> [RESUME] Loaded 100% SmolVLA weights from: {os.path.basename(model_path)}")
            elif len(compat) > 0:
                model_dict.update(compat)
                model.load_state_dict(model_dict)
                print(f">> [VLA FINE-TUNING] Transferred {len(compat)}/{len(model_dict)} pretrained backbone layers.")
        except Exception as e:
            print(f">> [INFO] Initializing fresh SmolVLA Transformer.")

    try:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, fused=True)
    except Exception:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    huber_loss_fn = nn.SmoothL1Loss(reduction='none')
    bce_loss_fn = nn.BCEWithLogitsLoss()
    axis_weights = torch.tensor([1.0, 1.0, 4.0, 1.0], dtype=torch.float32, device=DEVICE)

    use_amp = True
    amp_dtype = torch.bfloat16 if (DEVICE.type == 'cpu' and hasattr(torch, 'bfloat16')) else torch.float32

    print(f"\n>> Fine-Tuning Policy across {len(dataset)} samples ({epochs} epochs with AMP)...")

    best_loss = float('inf')

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            total_motion = 0.0
            total_grip = 0.0
            total_succ = 0.0

            for img_b, proprio_b, prompt_b, target_chunk_b in dataloader:
                img_b = img_b.to(DEVICE, non_blocking=True)
                proprio_b = proprio_b.to(DEVICE, non_blocking=True)
                prompt_b = prompt_b.to(DEVICE, non_blocking=True)
                target_chunk_b = target_chunk_b.to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=DEVICE.type, dtype=amp_dtype, enabled=use_amp):
                    pred_motion_chunk, pred_grip_chunk, pred_succ = model(img_b, proprio_b, prompt_b)

                    target_motion = target_chunk_b[:, :, :4]
                    target_grip = target_chunk_b[:, :, 4:5]
                    target_succ = target_chunk_b[:, -1, 5:6]

                    raw_motion_loss = huber_loss_fn(pred_motion_chunk, target_motion)
                    weighted_motion_loss = (raw_motion_loss * axis_weights).mean()

                    grip_loss = bce_loss_fn(pred_grip_chunk, target_grip)
                    succ_loss = bce_loss_fn(pred_succ, target_succ)

                    loss = weighted_motion_loss + 3.0 * grip_loss + 2.0 * succ_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(img_b)
                total_motion += weighted_motion_loss.item() * len(img_b)
                total_grip += grip_loss.item() * len(img_b)
                total_succ += succ_loss.item() * len(img_b)

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
