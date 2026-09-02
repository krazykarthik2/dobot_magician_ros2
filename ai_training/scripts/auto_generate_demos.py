import os
import sys
import time
import glob
import re
import numpy as np
import pygame

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "env"))
from dobot_env import DobotPickPlaceSim

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "demos")
os.makedirs(DATA_DIR, exist_ok=True)

def get_next_demo_index():
    files = glob.glob(os.path.join(DATA_DIR, "demo_*.npz"))
    if not files:
        return 1
    indices = []
    for f in files:
        m = re.search(r"demo_(\d+)\.npz", os.path.basename(f))
        if m:
            indices.append(int(m.group(1)))
    return max(indices) + 1 if indices else 1

def count_saved_demos():
    return len(glob.glob(os.path.join(DATA_DIR, "demo_*.npz")))

def world_to_screen(x, y):
    sx = int(200 + (y / 0.30) * 160)
    sy = int(350 - (x / 0.35) * 300)
    return sx, sy

def world_to_side_screen(x, z, offset_x=400):
    sx = int(offset_x + 50 + (x / 0.35) * 200)
    sy = int(350 - (z / 0.25) * 280)
    return sx, sy

def generate_smooth_trajectory(start_pos, target_pos, num_steps):
    t = np.linspace(0, 1, num_steps)
    s = 10 * (t**3) - 15 * (t**4) + 6 * (t**5)
    traj = np.outer(1 - s, start_pos) + np.outer(s, target_pos)
    return traj

def render_gui(screen, font, font_bold, sim, current_demo_num, total_in_batch, total_saved, stage_name, trajectory_len, fps_mode):
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
    
    status_str = f"Generating Demo #{current_demo_num} (Batch: {total_in_batch} | Total Dataset: {total_saved})"
    screen.blit(font_bold.render(status_str, True, (100, 210, 255)), (35, 405))

    stage_str = f"Phase: {stage_name}"
    screen.blit(font_bold.render(stage_str, True, (255, 220, 90)), (35, 440))

    frames_str = f"Frames: {trajectory_len} timesteps"
    screen.blit(font.render(frames_str, True, (200, 200, 210)), (580, 440))

    speed_info = font.render(f"Speed: {fps_mode} | [F] Toggle Fast/Turbo | [Q] Stop", True, (150, 155, 170))
    screen.blit(speed_info, (35, 468))

    pygame.display.flip()

def run_auto_demonstrator(num_demos=50, base_delay=0.005):
    print("=" * 65)
    print("   Dobot Magician Automated Delta-Action Demonstration Generator")
    print("=" * 65)
    print(f">> Existing Demos in Dataset: {count_saved_demos()}")
    print(f">> Generating Batch of {num_demos} smooth time-series demonstrations...")
    print(">> Action format: [dx, dy, dz, dyaw, gripper_target]")
    print("=" * 65)

    sim = DobotPickPlaceSim()

    pygame.init()
    screen = pygame.display.set_mode((780, 520))
    pygame.display.set_caption("Dobot Delta-Action Expert Demonstration Generator")
    font = pygame.font.SysFont("Arial", 15)
    font_bold = pygame.font.SysFont("Arial", 18, bold=True)
    clock = pygame.time.Clock()

    delay = base_delay
    fps_label = "Real-time High-Speed (120 FPS)"
    demos_completed = 0

    while demos_completed < num_demos:
        demo_idx = get_next_demo_index()
        obs = sim.reset(random_cube=True)
        cube_start = sim.cube_pos.copy()
        platform_target = sim.platform_pos.copy()

        trajectory = []
        aborted = False

        hover_cube_z = 0.12
        p_start = sim.ee_pos[:3].copy()
        p_hover_cube = np.array([cube_start[0], cube_start[1], hover_cube_z], dtype=np.float32)
        
        stages = [
            ("1. Move Over Red Cube (Top View)", p_start, p_hover_cube, 0.0, 24),
            ("2. Descend Toward Cube (Side View)", p_hover_cube, np.array([cube_start[0], cube_start[1], 0.026], dtype=np.float32), 0.0, 18),
            ("3. Close Gripper & Grasp Cube", np.array([cube_start[0], cube_start[1], 0.026], dtype=np.float32), np.array([cube_start[0], cube_start[1], 0.026], dtype=np.float32), 1.0, 10),
            ("4. Lift Cube Upward (Side View)", np.array([cube_start[0], cube_start[1], 0.026], dtype=np.float32), p_hover_cube, 1.0, 18),
            ("5. Carry Cube Over Green Box (Top View)", p_hover_cube, np.array([platform_target[0], platform_target[1], hover_cube_z], dtype=np.float32), 1.0, 26),
            ("6. Lower Onto Platform (Side View)", np.array([platform_target[0], platform_target[1], hover_cube_z], dtype=np.float32), np.array([platform_target[0], platform_target[1], 0.035], dtype=np.float32), 1.0, 18),
            ("7. Open Gripper & Release Cube", np.array([platform_target[0], platform_target[1], 0.035], dtype=np.float32), np.array([platform_target[0], platform_target[1], 0.035], dtype=np.float32), 0.0, 10),
            ("8. Retract Arm to Hover Position", np.array([platform_target[0], platform_target[1], 0.035], dtype=np.float32), np.array([platform_target[0], platform_target[1], 0.12], dtype=np.float32), 0.0, 16),
        ]

        for stage_name, start_pt, end_pt, grip_state, steps in stages:
            pts = generate_smooth_trajectory(start_pt, end_pt, steps)
            for pt in pts:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            aborted = True
                        elif event.key == pygame.K_f:
                            if delay > 0.001:
                                delay = 0.0001
                                fps_label = "Turbo Speed (Max FPS)"
                            else:
                                delay = base_delay
                                fps_label = "Real-time High-Speed (120 FPS)"

                if aborted:
                    break

                obs_before = sim.get_observation()
                
                current_ee = sim.ee_pos.copy()
                dx = pt[0] - current_ee[0]
                dy = pt[1] - current_ee[1]
                dz = pt[2] - current_ee[2]
                dyaw = 0.0
                
                delta_action = np.array([dx, dy, dz, dyaw, grip_state], dtype=np.float32)
                sim.step_delta(delta_action)

                trajectory.append((obs_before, delta_action))

                total_saved_now = count_saved_demos()
                render_gui(screen, font, font_bold, sim, demo_idx, num_demos, total_saved_now, stage_name, len(trajectory), fps_label)
                if delay > 0:
                    time.sleep(delay)

            if aborted:
                break

        if aborted:
            print("\n[INFO] Demonstration generation stopped by user.")
            break

        filename = os.path.join(DATA_DIR, f"demo_{demo_idx:03d}.npz")
        obs_arr = np.array([step[0] for step in trajectory], dtype=np.float32)
        act_arr = np.array([step[1] for step in trajectory], dtype=np.float32)
        np.savez_compressed(filename, observations=obs_arr, actions=act_arr)
        demos_completed += 1
        total_now = count_saved_demos()
        print(f"[SUCCESS] Saved Demo #{demo_idx:03d} (Progress: {demos_completed}/{num_demos} | Total Dataset: {total_now}) -> {os.path.basename(filename)}")

    pygame.quit()
    print(f"\n[DONE] Finished batch! Total {count_saved_demos()} demonstration datasets in: {DATA_DIR}")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_auto_demonstrator(num_demos=n)
