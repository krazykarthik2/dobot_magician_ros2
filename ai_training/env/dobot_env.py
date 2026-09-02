import os
import numpy as np
import pygame

class DobotKinematics:
    """Exact Analytical Forward and Inverse Kinematics for Dobot Magician 4-DOF."""
    def __init__(self):
        self.L1 = 0.138    # Base to Rear Arm joint height
        self.L2 = 0.135    # Rear arm length
        self.L3 = 0.147    # Forearm length
        self.L4 = 0.060    # Tool end-effector offset

    def forward(self, j1, j2, j3, j4):
        r = self.L2 * np.cos(j2) + self.L3 * np.cos(j2 + j3) + self.L4
        x = r * np.cos(j1)
        y = r * np.sin(j1)
        z = self.L1 + self.L2 * np.sin(j2) + self.L3 * np.sin(j2 + j3)
        yaw = j1 + j4
        return np.array([x, y, z, yaw], dtype=np.float32)

    def inverse(self, x, y, z, yaw=0.0):
        j1 = np.arctan2(y, x)
        r_total = np.sqrt(x**2 + y**2)
        r = r_total - self.L4
        dz = z - self.L1

        D_sq = r**2 + dz**2
        D = np.sqrt(D_sq)

        cos_j3 = (D_sq - self.L2**2 - self.L3**2) / (2 * self.L2 * self.L3)
        cos_j3 = np.clip(cos_j3, -1.0, 1.0)
        j3 = -np.arccos(cos_j3)

        alpha = np.arctan2(dz, r)
        beta = np.arctan2(self.L3 * np.sin(-j3), self.L2 + self.L3 * np.cos(-j3))
        j2 = alpha + beta

        j4 = yaw - j1
        return np.array([j1, j2, j3, j4], dtype=np.float32)

class DobotPickPlaceSim:
    """
    Simulation Environment for Vision-Language-Action (VLA) Learning:
    - Robot Base Origin: (0, 0, 0)
    - Multimodal Observations:
        * RGB Camera Image: [3, 64, 64] raw pixel visual feed (top-down / side scene)
        * Proprioception Robot State: [ee_x, ee_y, ee_z, ee_yaw, gripper_status] (5 dims)
        * Language Instruction: string prompt
    """
    def __init__(self):
        self.kin = DobotKinematics()
        self.cube_size = 0.022
        self.gripper_closed = False
        self.grasped = False
        
        # State
        self.ee_pos = np.array([0.20, 0.0, 0.12, 0.0], dtype=np.float32)
        self.cube_pos = np.array([0.22, 0.12, 0.011], dtype=np.float32)
        self.platform_pos = np.array([0.22, -0.15, 0.005], dtype=np.float32)
        self.instruction = "pick up the red cube and place it on the green platform"
        
        # Offscreen camera surface for rendering raw RGB vision
        pygame.init()
        self.cam_surface = pygame.Surface((64, 64))
        self.reset()

    def reset(self, random_scene=True, prompt=None):
        self.ee_pos = np.array([0.20, 0.0, 0.12, 0.0], dtype=np.float32)
        self.gripper_closed = False
        self.grasped = False

        if prompt is not None:
            self.instruction = prompt
        else:
            prompts = [
                "pick up the red cube and place it on the green platform",
                "grasp the red cube and move to green box",
                "transfer red block onto green platform",
                "pick red object and place on green target"
            ]
            self.instruction = np.random.choice(prompts)

        if random_scene:
            # 1. Random Red Cube Position
            angle_c = np.random.uniform(-0.65, 0.65)
            dist_c = np.random.uniform(0.18, 0.28)
            cx = dist_c * np.cos(angle_c)
            cy = dist_c * np.sin(angle_c)
            self.cube_pos = np.array([cx, cy, 0.011], dtype=np.float32)

            # 2. Random Green Platform Position
            while True:
                angle_p = np.random.uniform(-0.75, 0.75)
                dist_p = np.random.uniform(0.18, 0.28)
                px = dist_p * np.cos(angle_p)
                py = dist_p * np.sin(angle_p)
                if np.hypot(px - cx, py - cy) > 0.10:
                    self.platform_pos = np.array([px, py, 0.005], dtype=np.float32)
                    break
        else:
            self.cube_pos = np.array([0.22, 0.12, 0.011], dtype=np.float32)
            self.platform_pos = np.array([0.22, -0.15, 0.005], dtype=np.float32)

        return self.get_vla_observation()

    def render_camera_rgb(self):
        """
        Renders an overhead / perspective camera view as a [3, 64, 64] float32 RGB tensor normalized to [0, 1].
        """
        surf = self.cam_surface
        surf.fill((30, 32, 40)) # Dark workspace mat background

        # World coordinates (x: 0.10..0.35, y: -0.25..0.25) to 64x64 pixel frame
        def to_cam_px(x, y):
            px = int(32 + (y / 0.28) * 28)
            py = int(58 - ((x - 0.10) / 0.25) * 52)
            return px, py

        # 1. Draw Green Platform
        gx, gy = to_cam_px(self.platform_pos[0], self.platform_pos[1])
        pygame.draw.rect(surf, (40, 210, 80), (gx - 5, gy - 5, 10, 10), border_radius=1)

        # 2. Draw Red Cube
        cx, cy = to_cam_px(self.cube_pos[0], self.cube_pos[1])
        pygame.draw.rect(surf, (240, 45, 45), (cx - 3, cy - 3, 6, 6))

        # 3. Draw Robot Base Origin
        bx, by = to_cam_px(0.08, 0.0)
        pygame.draw.circle(surf, (90, 95, 115), (bx, by), 5)

        # 4. Draw Robot Arm Linkage
        ex, ey = to_cam_px(self.ee_pos[0], self.ee_pos[1])
        pygame.draw.line(surf, (170, 175, 195), (bx, by), (ex, ey), 2)

        # 5. Draw Robot End-Effector Gripper
        grip_color = (255, 90, 90) if self.gripper_closed else (90, 200, 255)
        pygame.draw.circle(surf, grip_color, (ex, ey), 3)

        # Extract RGB array: [64, 64, 3] -> [3, 64, 64] float32 in [0, 1]
        rgb_hwc = pygame.surfarray.array3d(surf) # [64, 64, 3]
        rgb_chw = np.transpose(rgb_hwc, (2, 1, 0)).astype(np.float32) / 255.0
        return rgb_chw

    def get_proprioception(self):
        """Returns 5-dim robot state: [x, y, z, yaw, grip_state]"""
        grip_val = 1.0 if self.gripper_closed else 0.0
        return np.array([
            self.ee_pos[0], self.ee_pos[1], self.ee_pos[2], self.ee_pos[3], grip_val
        ], dtype=np.float32)

    def get_vla_observation(self):
        """Returns complete Vision-Language-Action observation dictionary."""
        return {
            "image": self.render_camera_rgb(),        # [3, 64, 64] RGB pixel image
            "proprio": self.get_proprioception(),     # [5] EE pose + gripper
            "prompt": self.instruction                # Natural language prompt string
        }

    def get_observation(self):
        """Legacy 11-dim state observation for compatibility."""
        grip_val = 1.0 if self.gripper_closed else 0.0
        return np.array([
            self.ee_pos[0], self.ee_pos[1], self.ee_pos[2], self.ee_pos[3],
            grip_val,
            self.cube_pos[0], self.cube_pos[1], self.cube_pos[2],
            self.platform_pos[0], self.platform_pos[1], self.platform_pos[2]
        ], dtype=np.float32)

    def step_delta(self, delta_action, max_step=0.006):
        """Applies bounded velocity delta displacement."""
        dx = np.clip(delta_action[0], -max_step, max_step)
        dy = np.clip(delta_action[1], -max_step, max_step)
        dz = np.clip(delta_action[2], -max_step, max_step)
        dyaw = np.clip(delta_action[3], -0.1, 0.1)

        new_x = np.clip(self.ee_pos[0] + dx, 0.12, 0.32)
        new_y = np.clip(self.ee_pos[1] + dy, -0.25, 0.25)
        new_z = np.clip(self.ee_pos[2] + dz, 0.015, 0.22)
        new_yaw = self.ee_pos[3] + dyaw

        target_ee = [new_x, new_y, new_z, new_yaw, delta_action[4]]
        return self.step(target_ee)

    def step(self, target_ee):
        self.ee_pos = np.array(target_ee[:4], dtype=np.float32)
        self.gripper_closed = bool(target_ee[4] > 0.5)

        ee_xyz = self.ee_pos[:3]
        dist_to_cube = np.linalg.norm(ee_xyz - self.cube_pos)

        # Grasping physics
        if self.gripper_closed:
            if dist_to_cube < 0.032:
                self.grasped = True
        else:
            self.grasped = False

        if self.grasped:
            self.cube_pos = ee_xyz.copy()
            self.cube_pos[2] = max(self.cube_pos[2] - 0.015, 0.011)
        else:
            if self.cube_pos[2] > 0.011:
                self.cube_pos[2] = max(self.cube_pos[2] - 0.01, 0.011)

        obs = self.get_vla_observation()
        
        dist_to_goal = np.linalg.norm(self.cube_pos[:2] - self.platform_pos[:2])
        is_success = bool(dist_to_goal < 0.04 and self.cube_pos[2] <= 0.025 and not self.gripper_closed)

        return obs, is_success
