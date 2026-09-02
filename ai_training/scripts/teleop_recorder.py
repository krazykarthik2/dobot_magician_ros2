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

def main():
    print("=" * 60)
    print("      Dobot Magician AI Demonstration Recorder (Append Mode)")
    print("=" * 60)
    print("Controls:")
    print("  [W / S]       : Move End-Effector Forward / Backward (X axis)")
    print("  [A / D]       : Move End-Effector Left / Right (Y axis)")
    print("  [Q / E]       : Move End-Effector Up / Down (Z axis)")
    print("  [J / L]       : Rotate Tool Yaw")
    print("  [SPACE]       : Toggle Gripper (Open / Close)")
    print("  [R]           : Toggle Recording ON / OFF")
    print("  [ENTER]       : Save Demonstration & Next Cube Position")
    print("  [N]           : Skip / Randomize Cube (Discard current demo)")
    print("  [ESC]         : Exit")
    print("=" * 60)

    sim = DobotPickPlaceSim()

    pygame.init()
    screen = pygame.display.set_mode((780, 520))
    pygame.display.set_caption("Dobot AI Demonstration Recorder (Accumulating)")
    font = pygame.font.SysFont("Arial", 15)
    font_bold = pygame.font.SysFont("Arial", 18, bold=True)
    clock = pygame.time.Clock()

    step_size = 0.0035
    yaw_step = 0.05
    is_recording = False
    current_trajectory = []
    saved_count = count_saved_demos()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    sim.gripper_closed = not sim.gripper_closed
                elif event.key == pygame.K_r:
                    is_recording = not is_recording
                    if is_recording:
                        current_trajectory = []
                        print(">> Recording STARTED...")
                    else:
                        print(f">> Recording PAUSED ({len(current_trajectory)} steps).")
                elif event.key == pygame.K_RETURN:
                    if len(current_trajectory) > 5:
                        demo_idx = get_next_demo_index()
                        filename = os.path.join(DATA_DIR, f"demo_{demo_idx:03d}.npz")
                        obs_arr = np.array([step[0] for step in current_trajectory], dtype=np.float32)
                        act_arr = np.array([step[1] for step in current_trajectory], dtype=np.float32)
                        np.savez_compressed(filename, observations=obs_arr, actions=act_arr)
                        saved_count = count_saved_demos()
                        print(f"[SUCCESS] Appended Demo #{demo_idx:03d} (Total Dataset: {saved_count} demos) -> {filename}")
                        current_trajectory = []
                        is_recording = False
                        sim.reset(random_cube=True)
                elif event.key == pygame.K_n:
                    current_trajectory = []
                    is_recording = False
                    sim.reset(random_cube=True)
                    print(">> Random cube spawned.")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            sim.ee_pos[0] += step_size
        if keys[pygame.K_s]:
            sim.ee_pos[0] -= step_size
        if keys[pygame.K_a]:
            sim.ee_pos[1] += step_size
        if keys[pygame.K_d]:
            sim.ee_pos[1] -= step_size
        if keys[pygame.K_q]:
            sim.ee_pos[2] += step_size
        if keys[pygame.K_e]:
            sim.ee_pos[2] -= step_size
        if keys[pygame.K_j]:
            sim.ee_pos[3] += yaw_step
        if keys[pygame.K_l]:
            sim.ee_pos[3] -= yaw_step

        sim.ee_pos[0] = np.clip(sim.ee_pos[0], 0.12, 0.32)
        sim.ee_pos[1] = np.clip(sim.ee_pos[1], -0.25, 0.25)
        sim.ee_pos[2] = np.clip(sim.ee_pos[2], 0.015, 0.22)

        obs_before = sim.get_observation()
        target_action = np.array([
            sim.ee_pos[0], sim.ee_pos[1], sim.ee_pos[2], sim.ee_pos[3],
            1.0 if sim.gripper_closed else 0.0
        ], dtype=np.float32)

        obs_after, is_success = sim.step(target_action)

        if is_recording:
            current_trajectory.append((obs_before, target_action))

        # Render 2D Multi-view Graphics
        screen.fill((25, 27, 34))

        # Top-down View panel
        pygame.draw.rect(screen, (35, 38, 48), (20, 20, 360, 360), border_radius=8)
        pygame.draw.circle(screen, (70, 75, 95), (200, 350), 30, 2)
        
        # Platform (Top-down)
        px, py = world_to_screen(sim.platform_pos[0], sim.platform_pos[1])
        pygame.draw.rect(screen, (40, 200, 70), (px - 20, py - 20, 40, 40), border_radius=4)
        
        # Cube (Top-down)
        cx, cy = world_to_screen(sim.cube_pos[0], sim.cube_pos[1])
        pygame.draw.rect(screen, (230, 45, 45), (cx - 10, cy - 10, 20, 20), border_radius=2)
        
        # Arm End-Effector (Top-down)
        ex, ey = world_to_screen(sim.ee_pos[0], sim.ee_pos[1])
        grip_color = (255, 80, 80) if sim.gripper_closed else (100, 200, 255)
        pygame.draw.circle(screen, grip_color, (ex, ey), 8)
        pygame.draw.line(screen, (160, 170, 190), (200, 350), (ex, ey), 3)

        top_label = font.render("TOP-DOWN VIEW (X-Y)", True, (170, 180, 200))
        screen.blit(top_label, (30, 30))

        # Side Elevation View panel
        pygame.draw.rect(screen, (35, 38, 48), (400, 20, 360, 360), border_radius=8)
        pygame.draw.line(screen, (60, 65, 80), (410, 350), (750, 350), 2)
        
        # Platform (Side)
        psx, psy = world_to_side_screen(sim.platform_pos[0], sim.platform_pos[2])
        pygame.draw.rect(screen, (40, 200, 70), (psx - 20, psy - 4, 40, 8), border_radius=2)

        # Cube (Side)
        csx, csy = world_to_side_screen(sim.cube_pos[0], sim.cube_pos[2])
        pygame.draw.rect(screen, (230, 45, 45), (csx - 8, csy - 8, 16, 16), border_radius=2)

        # Arm End-Effector (Side)
        esx, esy = world_to_side_screen(sim.ee_pos[0], sim.ee_pos[2])
        pygame.draw.circle(screen, grip_color, (esx, esy), 8)

        side_label = font.render("SIDE ELEVATION VIEW (X-Z)", True, (170, 180, 200))
        screen.blit(side_label, (410, 30))

        # Bottom Status & HUD
        pygame.draw.rect(screen, (30, 33, 42), (20, 395, 740, 105), border_radius=8)
        
        status_color = (80, 240, 110) if is_recording else (240, 170, 60)
        status_str = f"Status: {'RECORDING...' if is_recording else 'IDLE / TELEOP'}"
        screen.blit(font_bold.render(status_str, True, status_color), (35, 405))

        demo_str = f"Total Dataset Demos: {saved_count}"
        screen.blit(font_bold.render(demo_str, True, (100, 200, 255)), (250, 405))

        steps_str = f"Trajectory: {len(current_trajectory)} frames"
        screen.blit(font.render(steps_str, True, (200, 200, 210)), (520, 408))

        if is_success:
            succ_label = font_bold.render("[SUCCESS: CUBE ON PLATFORM!]", True, (80, 255, 120))
            screen.blit(succ_label, (35, 435))
        else:
            inst1 = font.render("Move: W/S (X), A/D (Y), Q/E (Z) | SPACE: Gripper | R: Record | ENTER: Save Demo", True, (200, 205, 220))
            screen.blit(inst1, (35, 440))

        inst2 = font.render("N: Randomize Cube (Discard) | ESC: Exit", True, (150, 155, 170))
        screen.blit(inst2, (35, 468))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    print(f"Demonstration session ended. Total demos in dataset: {count_saved_demos()}")

if __name__ == "__main__":
    main()
