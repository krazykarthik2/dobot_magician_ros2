import os
import sys
import time
import torch
import numpy as np
import pygame

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "env"))
from dobot_env import DobotPickPlaceSim
from train_imitation import DobotActionChunkTransformer, MODEL_DIR, WINDOW_SIZE, CHUNK_SIZE

def world_to_screen(x, y):
    sx = int(200 + (y / 0.30) * 160)
    sy = int(350 - (x / 0.35) * 300)
    return sx, sy

def world_to_side_screen(x, z, offset_x=400):
    sx = int(offset_x + 50 + (x / 0.35) * 200)
    sy = int(350 - (z / 0.25) * 280)
    return sx, sy

def render_gui(screen, font, font_bold, sim, ep, total_eps, step, max_steps, model_succ_prob, is_success):
    screen.fill((25, 27, 34))

    # Top-Down Panel
    pygame.draw.rect(screen, (35, 38, 48), (20, 20, 360, 360), border_radius=8)
    pygame.draw.circle(screen, (70, 75, 95), (200, 350), 30, 2)
    
    # Platform (Top)
    px, py = world_to_screen(sim.platform_pos[0], sim.platform_pos[1])
    pygame.draw.rect(screen, (40, 200, 70), (px - 20, py - 20, 40, 40), border_radius=4)
    
    # Cube (Top)
    cx, cy = world_to_screen(sim.cube_pos[0], sim.cube_pos[1])
    pygame.draw.rect(screen, (230, 45, 45), (cx - 10, cy - 10, 20, 20), border_radius=2)
    
    # End-Effector (Top)
    ex, ey = world_to_screen(sim.ee_pos[0], sim.ee_pos[1])
    grip_color = (255, 80, 80) if sim.gripper_closed else (100, 200, 255)
    pygame.draw.circle(screen, grip_color, (ex, ey), 8)
    pygame.draw.line(screen, (160, 170, 190), (200, 350), (ex, ey), 3)

    top_label = font.render("TOP-DOWN VIEW (X-Y)", True, (170, 180, 200))
    screen.blit(top_label, (30, 30))

    # Side Elevation Panel
    pygame.draw.rect(screen, (35, 38, 48), (400, 20, 360, 360), border_radius=8)
    pygame.draw.line(screen, (60, 65, 80), (410, 350), (750, 350), 2)
    
    # Platform (Side)
    psx, psy = world_to_side_screen(sim.platform_pos[0], sim.platform_pos[2])
    pygame.draw.rect(screen, (40, 200, 70), (psx - 20, psy - 4, 40, 8), border_radius=2)

    # Cube (Side)
    csx, csy = world_to_side_screen(sim.cube_pos[0], sim.cube_pos[2])
    pygame.draw.rect(screen, (230, 45, 45), (csx - 8, csy - 8, 16, 16), border_radius=2)

    # End-Effector (Side)
    esx, esy = world_to_side_screen(sim.ee_pos[0], sim.ee_pos[2])
    pygame.draw.circle(screen, grip_color, (esx, esy), 8)

    side_label = font.render("SIDE ELEVATION VIEW (X-Z)", True, (170, 180, 200))
    screen.blit(side_label, (410, 30))

    # Bottom Status HUD
    pygame.draw.rect(screen, (30, 33, 42), (20, 395, 740, 105), border_radius=8)
    
    title_str = f"ACT AI (Ensembled): Episode {ep} / {total_eps}"
    screen.blit(font_bold.render(title_str, True, (100, 210, 255)), (35, 405))

    steps_str = f"Step: {step} / {max_steps}"
    screen.blit(font.render(steps_str, True, (200, 200, 210)), (450, 408))

    # Display Model's Self-Evaluated Success Confidence
    succ_color = (80, 255, 120) if model_succ_prob > 0.60 else (200, 200, 210)
    conf_str = f"Model Self-Evaluated Success: {model_succ_prob*100:.1f}%"
    screen.blit(font_bold.render(conf_str, True, succ_color), (35, 435))

    if is_success:
        succ_label = font_bold.render("[TASK COMPLETED: CUBE PLACED!]", True, (80, 255, 120))
        screen.blit(succ_label, (420, 435))

    info_str = font.render(f"EE: [{sim.ee_pos[0]:.2f}, {sim.ee_pos[1]:.2f}, {sim.ee_pos[2]:.2f}] | Goal: [{sim.platform_pos[0]:.2f}, {sim.platform_pos[1]:.2f}] | [ESC] Exit", True, (140, 145, 160))
    screen.blit(info_str, (35, 468))

    pygame.display.flip()

def evaluate(episodes=10):
    model_path = os.path.join(MODEL_DIR, "dobot_bc_policy.pth")
    stats_path = os.path.join(MODEL_DIR, "norm_stats.npz")
    
    if not os.path.exists(model_path) or not os.path.exists(stats_path):
        print("\n [ERROR] Model or stats not found! Train model first.")
        return

    stats = np.load(stats_path)
    obs_mean = stats['obs_mean']
    obs_std = stats['obs_std']
    motion_mean = stats['motion_mean']
    motion_std = stats['motion_std']
    window_size = int(stats['window_size']) if 'window_size' in stats else WINDOW_SIZE
    chunk_size = int(stats['chunk_size']) if 'chunk_size' in stats else CHUNK_SIZE

    model = DobotActionChunkTransformer(obs_dim=len(obs_mean), chunk_size=chunk_size, d_model=128, nhead=4, num_layers=3)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    sim = DobotPickPlaceSim()

    pygame.init()
    screen = pygame.display.set_mode((780, 520))
    pygame.display.set_caption("Dobot Temporal Ensembling ACT Evaluation")
    font = pygame.font.SysFont("Arial", 15)
    font_bold = pygame.font.SysFont("Arial", 18, bold=True)
    clock = pygame.time.Clock()

    successes = 0
    max_steps = 180

    # Temporal Ensembling Exponential Weights: w_i = exp(-k * i)
    exp_weights = np.exp(-0.4 * np.arange(chunk_size))
    exp_weights = exp_weights / exp_weights.sum()

    print("=" * 65)
    print("   Testing ACT Policy with Exponential Temporal Ensembling")
    print("   (Random Red Cube & Random Green Platform Scenes)")
    print("=" * 65)

    for ep in range(1, episodes + 1):
        obs = sim.reset(random_scene=True)
        print(f"\nEpisode {ep}/{episodes} - Cube: [{obs[5]:.3f}, {obs[6]:.3f}] | Platform: [{obs[8]:.3f}, {obs[9]:.3f}]")
        
        norm_obs_init = (obs - obs_mean) / obs_std
        obs_history = [norm_obs_init.copy() for _ in range(window_size)]
        
        # Buffer of overlapping predicted action chunks for temporal ensembling
        # List of chunks currently active at time t
        active_chunks = []

        ep_success = False
        aborted = False
        model_succ_prob = 0.0

        for step in range(1, max_steps + 1):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    aborted = True

            if aborted:
                break

            # 1. Query policy at every frame
            seq_t = torch.tensor(np.array([obs_history]), dtype=torch.float32)
            with torch.no_grad():
                pred_motion_chunk, pred_grip_chunk, pred_succ_logit = model(seq_t)
                
                # Denormalize motion
                pred_motion = (pred_motion_chunk.squeeze(0).numpy() * motion_std) + motion_mean
                # Sigmoid grip probabilities
                pred_grip = torch.sigmoid(pred_grip_chunk).squeeze(0).numpy()
                # Self-evaluated task success probability
                model_succ_prob = torch.sigmoid(pred_succ_logit).item()

            # Store new predicted chunk: [H, 5]
            new_chunk = np.concatenate([pred_motion, pred_grip], axis=-1)
            active_chunks.append((new_chunk, 0)) # (chunk_array, age_index)

            # 2. Temporal Ensembling: Blend overlapping actions across time
            ensembled_action = np.zeros(5, dtype=np.float32)
            total_weight = 0.0

            updated_active = []
            for chunk_arr, age in active_chunks:
                if age < chunk_size:
                    w = exp_weights[age]
                    ensembled_action += w * chunk_arr[age]
                    total_weight += w
                    updated_active.append((chunk_arr, age + 1))
            active_chunks = updated_active

            if total_weight > 0:
                ensembled_action /= total_weight

            # Execute ensembled smooth action
            grip_cmd = 1.0 if ensembled_action[4] > 0.50 else 0.0
            full_delta = np.array([
                ensembled_action[0], ensembled_action[1], ensembled_action[2], ensembled_action[3], grip_cmd
            ], dtype=np.float32)

            obs, is_succ = sim.step_delta(full_delta, max_step=0.007)
            if is_succ or model_succ_prob > 0.85:
                ep_success = True

            # Update observation sequence history
            norm_obs = (obs - obs_mean) / obs_std
            obs_history.pop(0)
            obs_history.append(norm_obs)

            render_gui(screen, font, font_bold, sim, ep, episodes, step, max_steps, model_succ_prob, is_succ)
            time.sleep(0.015)

            # Terminate episode if model reports high self-evaluated completion after release
            if is_succ and step > 120:
                break

        if aborted:
            break

        if ep_success:
            successes += 1
            print(f"Episode {ep}: SUCCESS! Task completed.")
        else:
            print(f"Episode {ep}: Missed platform.")

        time.sleep(0.3)

    print(f"\n==================================================")
    print(f" Final Score: {successes} / {episodes} Successes ({(successes/episodes)*100:.1f}%)")
    print(f"==================================================")

    pygame.quit()

if __name__ == "__main__":
    evaluate()
