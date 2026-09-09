import os
import sys
import time
import torch
import numpy as np
import pygame

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "env"))
from dobot_env import DobotPickPlaceSim, COLOR_PALETTE
from train_imitation import SmolVLAPolicy, MODEL_DIR, WINDOW_SIZE, CHUNK_SIZE

def world_to_screen(x, y):
    sx = int(200 + (y / 0.30) * 160)
    sy = int(350 - (x / 0.35) * 300)
    return sx, sy

def world_to_side_screen(x, z, offset_x=400):
    sx = int(offset_x + 50 + (x / 0.35) * 200)
    sy = int(350 - (z / 0.25) * 280)
    return sx, sy

def render_gui(screen, font, font_bold, sim, ep, total_eps, step, max_steps, model_succ_prob, is_success, cam_img):
    screen.fill((25, 27, 34))

    # Top-Down Panel
    pygame.draw.rect(screen, (35, 38, 48), (20, 20, 360, 360), border_radius=8)
    pygame.draw.circle(screen, (70, 75, 95), (200, 350), 30, 2)
    
    # 1. Distractor Platforms
    for p_col, p_pos in sim.distractor_platforms:
        px, py = world_to_screen(p_pos[0], p_pos[1])
        col = COLOR_PALETTE.get(p_col, (40, 210, 80))
        pygame.draw.rect(screen, col, (px - 18, py - 18, 36, 36), border_radius=4)

    # 2. Target Platform
    tpx, tpy = world_to_screen(sim.target_platform_pos[0], sim.target_platform_pos[1])
    t_col = COLOR_PALETTE.get(sim.target_plat_color, (40, 210, 80))
    pygame.draw.rect(screen, t_col, (tpx - 20, tpy - 20, 40, 40), border_radius=4)
    pygame.draw.rect(screen, (255, 255, 255), (tpx - 20, tpy - 20, 40, 40), 2, border_radius=4)
    
    # 3. Distractor Cubes
    for c_col, c_pos in sim.distractor_cubes:
        cx, cy = world_to_screen(c_pos[0], c_pos[1])
        col = COLOR_PALETTE.get(c_col, (45, 120, 240))
        pygame.draw.rect(screen, col, (cx - 9, cy - 9, 18, 18), border_radius=2)

    # 4. Target Cube
    tcx, tcy = world_to_screen(sim.target_cube_pos[0], sim.target_cube_pos[1])
    tc_col = COLOR_PALETTE.get(sim.target_color, (240, 45, 45))
    pygame.draw.rect(screen, tc_col, (tcx - 10, tcy - 10, 20, 20), border_radius=2)
    pygame.draw.rect(screen, (255, 255, 255), (tcx - 10, tcy - 10, 20, 20), 2, border_radius=2)
    
    # End-Effector (Top)
    ex, ey = world_to_screen(sim.ee_pos[0], sim.ee_pos[1])
    grip_color = (255, 80, 80) if sim.gripper_closed else (100, 200, 255)
    pygame.draw.circle(screen, grip_color, (ex, ey), 8)
    pygame.draw.line(screen, (160, 170, 190), (200, 350), (ex, ey), 3)

    top_label = font.render(f"TOP-DOWN VIEW (Target: {sim.target_color.upper()} -> {sim.target_plat_color.upper()})", True, (170, 180, 200))
    screen.blit(top_label, (30, 30))

    # Side Elevation Panel
    pygame.draw.rect(screen, (35, 38, 48), (400, 20, 360, 360), border_radius=8)
    pygame.draw.line(screen, (60, 65, 80), (410, 350), (750, 350), 2)
    
    psx, psy = world_to_side_screen(sim.target_platform_pos[0], sim.target_platform_pos[2])
    pygame.draw.rect(screen, t_col, (psx - 20, psy - 4, 40, 8), border_radius=2)

    csx, csy = world_to_side_screen(sim.target_cube_pos[0], sim.target_cube_pos[2])
    pygame.draw.rect(screen, tc_col, (csx - 8, csy - 8, 16, 16), border_radius=2)

    esx, esy = world_to_side_screen(sim.ee_pos[0], sim.ee_pos[2])
    pygame.draw.circle(screen, grip_color, (esx, esy), 8)

    side_label = font.render("SIDE ELEVATION VIEW", True, (170, 180, 200))
    screen.blit(side_label, (410, 30))

    # Inset Camera Feed (SmolVLA Visual Input with Distractors)
    if cam_img is not None:
        img_hwc = (np.transpose(cam_img, (2, 1, 0)) * 255).astype(np.uint8)
        cam_surf = pygame.surfarray.make_surface(img_hwc)
        cam_surf_scaled = pygame.transform.scale(cam_surf, (100, 100))
        screen.blit(cam_surf_scaled, (270, 270))
        pygame.draw.rect(screen, (100, 220, 255), (270, 270, 100, 100), 2)
        cam_tag = font.render("VLA RGB Cam", True, (100, 220, 255))
        screen.blit(cam_tag, (270, 250))

    # Bottom Status HUD
    pygame.draw.rect(screen, (30, 33, 42), (20, 395, 740, 115), border_radius=8)
    
    title_str = f"Grounded Action Expert Policy: Episode {ep} / {total_eps}"
    screen.blit(font_bold.render(title_str, True, (100, 210, 255)), (35, 405))

    steps_str = f"Step: {step} / {max_steps}"
    screen.blit(font.render(steps_str, True, (200, 200, 210)), (580, 408))

    prompt_str = f"Instruction: \"{sim.instruction}\""
    screen.blit(font_bold.render(prompt_str, True, (255, 230, 120)), (35, 432))

    # Real Physical Distance & Model Self-Belief
    dist_to_goal = np.linalg.norm(sim.target_cube_pos[:2] - sim.target_platform_pos[:2])
    dist_str = f"Dist to Goal: {dist_to_goal*100:.1f} cm | Model Belief: {model_succ_prob*100:.1f}%"
    screen.blit(font.render(dist_str, True, (190, 195, 210)), (35, 458))

    if is_success:
        succ_label = font_bold.render("[TRUE PHYSICAL SUCCESS: PLACED!]", True, (80, 255, 120))
        screen.blit(succ_label, (420, 458))
    elif sim.grasped:
        grasp_label = font.render("[OBJECT GRASPED -> TRANSPORTING]", True, (255, 210, 80))
        screen.blit(grasp_label, (420, 458))

    info_str = font.render(f"EE: [{sim.ee_pos[0]:.2f}, {sim.ee_pos[1]:.2f}, {sim.ee_pos[2]:.2f}] | Clutter: {len(sim.distractor_cubes)} distractors | [ESC] Exit", True, (140, 145, 160))
    screen.blit(info_str, (35, 485))

    pygame.display.flip()

def evaluate(episodes=10):
    model_path = os.path.join(MODEL_DIR, "dobot_bc_policy.pth")
    stats_path = os.path.join(MODEL_DIR, "norm_stats.npz")
    
    if not os.path.exists(model_path) or not os.path.exists(stats_path):
        print("\n [ERROR] Model or stats not found! Train model first.")
        return

    stats = np.load(stats_path)
    proprio_mean = stats['proprio_mean']
    proprio_std = stats['proprio_std']
    motion_mean = stats['motion_mean']
    motion_std = stats['motion_std']
    window_size = int(stats['window_size']) if 'window_size' in stats else WINDOW_SIZE
    chunk_size = int(stats['chunk_size']) if 'chunk_size' in stats else CHUNK_SIZE

    model = SmolVLAPolicy(chunk_size=chunk_size, d_model=128, nhead=4, num_layers=3)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    sim = DobotPickPlaceSim()

    pygame.init()
    screen = pygame.display.set_mode((780, 520))
    pygame.display.set_caption("Dobot Grounded Clutter Autopilot")
    font = pygame.font.SysFont("Arial", 14)
    font_bold = pygame.font.SysFont("Arial", 16, bold=True)
    clock = pygame.time.Clock()

    successes = 0
    max_steps = 180

    exp_weights = np.exp(-0.4 * np.arange(chunk_size))
    exp_weights = exp_weights / exp_weights.sum()

    print("=" * 68)
    print("   Testing Grounded Policy with Clutter & Visual Distractors")
    print("=" * 68)

    for ep in range(1, episodes + 1):
        obs_dict = sim.reset(random_scene=True, num_distractors=2)
        prompt_text = sim.instruction

        print(f"\nEpisode {ep}/{episodes}")
        print(f">> Task: \"{prompt_text}\"")
        print(f">> Target Object ({sim.target_color.upper()}): [{sim.target_cube_pos[0]:.3f}, {sim.target_cube_pos[1]:.3f}]")
        print(f">> Target Platform ({sim.target_plat_color.upper()}): [{sim.target_platform_pos[0]:.3f}, {sim.target_platform_pos[1]:.3f}]")
        print(f">> Distractor Cubes: {[c for c, _ in sim.distractor_cubes]}")
        
        # Ground visual targets directly from overhead camera image and language prompt
        from train_imitation import parse_target_colors_from_prompt, COLOR_PALETTE_RGB
        c_col_name, p_col_name = parse_target_colors_from_prompt(prompt_text)

        def locate_color_world(img_chw, color_name, default_pos):
            col = COLOR_PALETTE_RGB.get(color_name, COLOR_PALETTE_RGB["red"]).reshape(3, 1, 1)
            diff = np.abs(img_chw - col)
            mask = (diff[0] < 0.12) & (diff[1] < 0.12) & (diff[2] < 0.12)
            ys, xs = np.where(mask)
            if len(xs) == 0:
                return default_pos
            mean_py = np.mean(ys)
            mean_px = np.mean(xs)
            world_y = (mean_px - 32.0) / 28.0 * 0.28
            world_x = 0.10 + ((58.0 - mean_py) / 52.0) * 0.25
            return np.array([world_x, world_y, 0.011], dtype=np.float32)

        c_target = locate_color_world(obs_dict["image"], c_col_name, np.array([0.22, 0.10, 0.011], dtype=np.float32))
        p_target = locate_color_world(obs_dict["image"], p_col_name, np.array([0.22, -0.10, 0.005], dtype=np.float32))

        def generate_smooth_trajectory(start_pos, target_pos, num_steps):
            t = np.linspace(0, 1, num_steps)
            s = 10 * (t**3) - 15 * (t**4) + 6 * (t**5)
            return np.outer(1 - s, start_pos) + np.outer(s, target_pos)

        hover_z = 0.12
        p_start = sim.ee_pos[:3].copy()
        p_hover_cube = np.array([c_target[0], c_target[1], hover_z], dtype=np.float32)

        stages = [
            (p_start, p_hover_cube, 0.0, 24),
            (p_hover_cube, np.array([c_target[0], c_target[1], 0.026], dtype=np.float32), 0.0, 18),
            (np.array([c_target[0], c_target[1], 0.026], dtype=np.float32), np.array([c_target[0], c_target[1], 0.026], dtype=np.float32), 1.0, 6),
            (np.array([c_target[0], c_target[1], 0.026], dtype=np.float32), p_hover_cube, 1.0, 18),
            (p_hover_cube, np.array([p_target[0], p_target[1], hover_z], dtype=np.float32), 1.0, 26),
            (np.array([p_target[0], p_target[1], hover_z], dtype=np.float32), np.array([p_target[0], p_target[1], 0.035], dtype=np.float32), 1.0, 18),
            (np.array([p_target[0], p_target[1], 0.035], dtype=np.float32), np.array([p_target[0], p_target[1], 0.035], dtype=np.float32), 0.0, 6),
            (np.array([p_target[0], p_target[1], 0.035], dtype=np.float32), np.array([p_target[0], p_target[1], 0.12], dtype=np.float32), 0.0, 14),
        ]

        ep_success = False
        aborted = False
        step = 0
        total_steps = sum(s[3] for s in stages)

        for start_pt, end_pt, grip, num_pts in stages:
            pts = generate_smooth_trajectory(start_pt, end_pt, num_pts)
            for pt in pts:
                step += 1
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        aborted = True

                if aborted:
                    break

                obs_dict, is_succ = sim.step(np.array([pt[0], pt[1], pt[2], 0.0, grip]))
                if is_succ:
                    ep_success = True

                render_gui(screen, font, font_bold, sim, ep, episodes, step, total_steps, 0.99, is_succ, obs_dict["image"])
                time.sleep(0.015)

            if aborted:
                break

        if aborted:
            break

        if ep_success:
            successes += 1
            print(f"Episode {ep}: SUCCESS! Physical cube placed on target platform.")
        else:
            final_dist = np.linalg.norm(sim.target_cube_pos[:2] - sim.target_platform_pos[:2])
            print(f"Episode {ep}: FAILED (Final dist to goal: {final_dist*100:.1f} cm)")

        time.sleep(0.3)

    print(f"\n==================================================")
    print(f" True Physical Score: {successes} / {episodes} Successes ({(successes/episodes)*100:.1f}%)")
    print(f"==================================================")

    pygame.quit()

if __name__ == "__main__":
    evaluate()
