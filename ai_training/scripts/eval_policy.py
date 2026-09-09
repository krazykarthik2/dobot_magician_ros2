import os
import sys
import time
import torch
import numpy as np
import pygame

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "env"))
from dobot_env import DobotPickPlaceSim, COLOR_PALETTE
from train_imitation import SmolVLA2Policy, SmolVLAPolicy, MODEL_DIR, WINDOW_SIZE, CHUNK_SIZE

def world_to_screen(x, y):
    sx = int(200 + (y / 0.30) * 160)
    sy = int(350 - (x / 0.35) * 300)
    return sx, sy

def world_to_side_screen(x, z, offset_x=400):
    sx = int(offset_x + 50 + (x / 0.35) * 200)
    sy = int(350 - (z / 0.25) * 280)
    return sx, sy

def render_gui(screen, font, font_bold, sim, ep, total_eps, step, max_steps, model_succ_prob, is_success, cam_img, latent_tokens=None, attn_maps=None):
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

    # Inset Camera Feed (Raw RGB Input)
    if cam_img is not None:
        img_hwc = (np.transpose(cam_img, (2, 1, 0)) * 255).astype(np.uint8)
        cam_surf = pygame.surfarray.make_surface(img_hwc)
        cam_surf_scaled = pygame.transform.scale(cam_surf, (80, 80))
        screen.blit(cam_surf_scaled, (290, 290))
        pygame.draw.rect(screen, (100, 220, 255), (290, 290, 80, 80), 2)
        cam_tag = font.render("RGB Cam", True, (100, 220, 255))
        screen.blit(cam_tag, (290, 272))

    # Bottom Status HUD (Top half of bottom area)
    pygame.draw.rect(screen, (30, 33, 42), (20, 395, 740, 100), border_radius=8)
    
    title_str = f"SmolVLA-2 Neural Policy Evaluator: Episode {ep} / {total_eps}"
    screen.blit(font_bold.render(title_str, True, (100, 210, 255)), (35, 405))

    steps_str = f"Step: {step} / {max_steps}"
    screen.blit(font.render(steps_str, True, (200, 200, 210)), (580, 408))

    prompt_str = f"Instruction: \"{sim.instruction}\""
    screen.blit(font_bold.render(prompt_str, True, (255, 230, 120)), (35, 430))

    dist_to_goal = np.linalg.norm(sim.target_cube_pos[:2] - sim.target_platform_pos[:2])
    dist_str = f"Dist to Goal: {dist_to_goal*100:.1f} cm | Neural Trajectory Horizon: 128 steps"
    screen.blit(font.render(dist_str, True, (190, 195, 210)), (35, 455))

    if is_success:
        succ_label = font_bold.render("[TRUE PHYSICAL SUCCESS: PLACED!]", True, (80, 255, 120))
        screen.blit(succ_label, (430, 455))
    elif sim.grasped:
        grasp_label = font.render("[OBJECT GRASPED -> TRANSPORTING]", True, (255, 210, 80))
        screen.blit(grasp_label, (430, 455))

    info_str = font.render(f"EE: [{sim.ee_pos[0]:.2f}, {sim.ee_pos[1]:.2f}, {sim.ee_pos[2]:.2f}] | Clutter: {len(sim.distractor_cubes)} distractors | [ESC] Exit", True, (140, 145, 160))
    screen.blit(info_str, (35, 475))

    # Neural Spatial Softmax Latent Token Space Visualizer (Dedicated Bottom Panel)
    if latent_tokens is not None:
        pygame.draw.rect(screen, (20, 22, 28), (20, 505, 740, 105), border_radius=8)
        pygame.draw.rect(screen, (130, 90, 240), (20, 505, 740, 105), 2, border_radius=8)
        
        token_title = font_bold.render("SmolVLA-2 Internal Latent Token Space (Z_attn)", True, (200, 160, 255))
        screen.blit(token_title, (35, 515))
        
        k = latent_tokens
        cube_token_str = f"Target Obj Token  (z1, z2): [{k[0]:+.4f}, {k[1]:+.4f}]"
        plat_token_str = f"Target Plat Token (z3, z4): [{k[2]:+.4f}, {k[3]:+.4f}]"
        screen.blit(font.render(cube_token_str, True, (255, 200, 120)), (35, 540))
        screen.blit(font.render(plat_token_str, True, (120, 255, 200)), (35, 562))

        status_note = font.render("Neural Spatial Attn Activations (Norm Coord: [-1, +1])", True, (160, 165, 180))
        screen.blit(status_note, (35, 584))

        # Mini Latent Attention Grid Map (Right side of bottom panel)
        grid_x, grid_y = 660, 515
        pygame.draw.rect(screen, (35, 38, 50), (grid_x, grid_y, 80, 80), border_radius=6)
        pygame.draw.line(screen, (65, 70, 85), (grid_x + 40, grid_y), (grid_x + 40, grid_y + 80), 1)
        pygame.draw.line(screen, (65, 70, 85), (grid_x, grid_y + 40), (grid_x + 80, grid_y + 40), 1)
        
        # Plot attention centroids
        cx_dot = int(grid_x + 40 + k[0] * 34)
        cy_dot = int(grid_y + 40 + k[1] * 34)
        px_dot = int(grid_x + 40 + k[2] * 34)
        py_dot = int(grid_y + 40 + k[3] * 34)
        pygame.draw.circle(screen, (255, 80, 80), (cx_dot, cy_dot), 5) # Cube focus
        pygame.draw.circle(screen, (80, 255, 120), (px_dot, py_dot), 5) # Plat focus
        
        # Legend
        l_obj = font.render("Obj", True, (255, 80, 80))
        l_plt = font.render("Plat", True, (80, 255, 120))
        screen.blit(l_obj, (595, 535))
        screen.blit(l_plt, (595, 560))

    pygame.display.flip()

def evaluate(episodes=10):
    model_path = os.path.join(MODEL_DIR, "dobot_bc_policy.pth")
    if not os.path.exists(model_path):
        print("\n [ERROR] Model not found! Train model first.")
        return

    model = SmolVLA2Policy()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    sim = DobotPickPlaceSim()

    pygame.init()
    screen = pygame.display.set_mode((780, 625))
    pygame.display.set_caption("SmolVLA-2 Neural Policy & Latent Space Visualizer")
    font = pygame.font.SysFont("Arial", 14)
    font_bold = pygame.font.SysFont("Arial", 16, bold=True)
    clock = pygame.time.Clock()

    successes = 0
    total_steps = 128

    print("=" * 68)
    print("   Testing SmolVLA-2 Neural Policy (With Latent Token Visualizer)")
    print("=" * 68)

    for ep in range(1, episodes + 1):
        obs_dict = sim.reset(random_scene=True, num_distractors=2)
        prompt_text = sim.instruction

        print(f"\nEpisode {ep}/{episodes}")
        print(f">> Task: \"{prompt_text}\"")
        print(f">> Target Object ({sim.target_color.upper()}): [{sim.target_cube_pos[0]:.3f}, {sim.target_cube_pos[1]:.3f}]")
        print(f">> Target Platform ({sim.target_plat_color.upper()}): [{sim.target_platform_pos[0]:.3f}, {sim.target_platform_pos[1]:.3f}]")
        print(f">> Distractor Cubes: {[c for c, _ in sim.distractor_cubes]}")
        
        # Authentic SmolVLA-2 Forward pass directly from raw RGB image + text prompt string
        img_t = torch.tensor(obs_dict["image"], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred_traj_t, latent_kps_t, attn_maps_t = model(img_t, prompt_str=[prompt_text])
            pred_traj = pred_traj_t.squeeze(0).numpy()       # [128, 4]
            latent_kps = latent_kps_t.squeeze(0).numpy()     # [4]

        ep_success = False
        aborted = False

        for step in range(total_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    aborted = True

            if aborted:
                break

            pt = pred_traj[step]
            target_xyz = pt[:3]
            grip_cmd = 1.0 if pt[3] > 0.5 else 0.0

            obs_dict, is_succ = sim.step(np.array([target_xyz[0], target_xyz[1], target_xyz[2], 0.0, grip_cmd]))
            if is_succ:
                ep_success = True

            render_gui(screen, font, font_bold, sim, ep, episodes, step + 1, total_steps, 0.99, is_succ, obs_dict["image"], latent_tokens=latent_kps)
            time.sleep(0.015)

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
