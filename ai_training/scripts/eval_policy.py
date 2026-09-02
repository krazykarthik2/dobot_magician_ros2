import os
import sys
import time
import torch
import numpy as np
import pygame

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "env"))
from dobot_env import DobotPickPlaceSim
from train_imitation import DobotResidualPolicy, MODEL_DIR

def world_to_screen(x, y):
    sx = int(200 + (y / 0.30) * 160)
    sy = int(350 - (x / 0.35) * 300)
    return sx, sy

def world_to_side_screen(x, z, offset_x=400):
    sx = int(offset_x + 50 + (x / 0.35) * 200)
    sy = int(350 - (z / 0.25) * 280)
    return sx, sy

def render_gui(screen, font, font_bold, sim, ep, total_eps, step, max_steps, is_success):
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
    
    title_str = f"AI Autopilot Evaluation: Episode {ep} / {total_eps}"
    screen.blit(font_bold.render(title_str, True, (100, 210, 255)), (35, 405))

    steps_str = f"Step: {step} / {max_steps}"
    screen.blit(font.render(steps_str, True, (200, 200, 210)), (450, 408))

    if is_success:
        succ_label = font_bold.render("[SUCCESS: CUBE PLACED ON PLATFORM!]", True, (80, 255, 120))
        screen.blit(succ_label, (35, 440))
    else:
        status_mode = "Grasping / Transporting Cube..." if sim.grasped else "Navigating toward Cube..."
        screen.blit(font.render(f"Policy Action: {status_mode}", True, (255, 220, 100)), (35, 440))

    info_str = font.render(f"EE: [{sim.ee_pos[0]:.3f}, {sim.ee_pos[1]:.3f}, {sim.ee_pos[2]:.3f}] | Cube: [{sim.cube_pos[0]:.3f}, {sim.cube_pos[1]:.3f}] | [ESC] Exit", True, (140, 145, 160))
    screen.blit(info_str, (35, 468))

    pygame.display.flip()

def evaluate(episodes=10):
    model_path = os.path.join(MODEL_DIR, "dobot_bc_policy.pth")
    stats_path = os.path.join(MODEL_DIR, "norm_stats.npz")
    
    if not os.path.exists(model_path) or not os.path.exists(stats_path):
        print("\n" + "=" * 60)
        print(" [ERROR] Trained model or normalization stats not found!")
        print(f" Expected at: {model_path} and {stats_path}")
        print(" Please run Step [1] (Generate Demos) and Step [3] (Train Model) first.")
        print("=" * 60 + "\n")
        return

    # Load stats
    stats = np.load(stats_path)
    obs_mean = stats['obs_mean']
    obs_std = stats['obs_std']
    act_mean = stats['act_mean']
    act_std = stats['act_std']

    # Load policy
    model = DobotResidualPolicy(obs_dim=len(obs_mean), act_dim=len(act_mean), hidden_dim=256)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    sim = DobotPickPlaceSim()

    pygame.init()
    screen = pygame.display.set_mode((780, 520))
    pygame.display.set_caption("Dobot AI Autopilot Evaluation (Top & Side View)")
    font = pygame.font.SysFont("Arial", 15)
    font_bold = pygame.font.SysFont("Arial", 18, bold=True)
    clock = pygame.time.Clock()

    successes = 0
    max_steps = 145

    print("=" * 60)
    print("      Testing Trained AI Policy on Random Cube Positions")
    print("=" * 60)

    for ep in range(1, episodes + 1):
        obs = sim.reset(random_cube=True)
        print(f"\nEpisode {ep}/{episodes} - Initial Cube Pos: [{obs[5]:.3f}, {obs[6]:.3f}]")
        
        ep_success = False
        aborted = False

        for step in range(1, max_steps + 1):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    aborted = True

            if aborted:
                break

            # Normalize observation before feeding to neural network
            norm_obs = (obs - obs_mean) / obs_std

            with torch.no_grad():
                obs_t = torch.tensor(norm_obs, dtype=torch.float32).unsqueeze(0)
                norm_act = model(obs_t).squeeze(0).numpy()

            # Denormalize action to physical world coordinates
            pred_act = norm_act * act_std + act_mean

            obs, is_succ = sim.step(pred_act)
            if is_succ:
                ep_success = True

            render_gui(screen, font, font_bold, sim, ep, episodes, step, max_steps, ep_success)
            time.sleep(0.015)

            if ep_success and step > 130:
                break

        if aborted:
            break

        if ep_success:
            successes += 1
            print(f"Episode {ep}: SUCCESS! Cube placed on platform.")
        else:
            print(f"Episode {ep}: Missed platform.")

        time.sleep(0.5)

    print(f"\n==================================================")
    print(f" Final Score: {successes} / {episodes} Successes ({(successes/episodes)*100:.1f}%)")
    print(f"==================================================")

    pygame.quit()

if __name__ == "__main__":
    evaluate()
