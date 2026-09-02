import os
import numpy as np

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
    Simulation Environment:
    - Robot Base Origin: (0, 0, 0)
    - State:
        * EE Pose: [x, y, z, yaw] (4)
        * Gripper Status: [grip_val] (1)
        * Red Cube Position: [x, y, z] (3) -> Randomly sampled each episode
        * Green Platform Position: [x, y, z] (3) -> Randomly sampled each episode
    Total State Observation = 11 dimensions.
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
        self.reset()

    def reset(self, random_scene=True):
        self.ee_pos = np.array([0.20, 0.0, 0.12, 0.0], dtype=np.float32)
        self.gripper_closed = False
        self.grasped = False

        if random_scene:
            # 1. Random Red Cube Position
            angle_c = np.random.uniform(-0.65, 0.65)
            dist_c = np.random.uniform(0.18, 0.28)
            cx = dist_c * np.cos(angle_c)
            cy = dist_c * np.sin(angle_c)
            self.cube_pos = np.array([cx, cy, 0.011], dtype=np.float32)

            # 2. Random Green Platform Position (ensuring separation from cube)
            while True:
                angle_p = np.random.uniform(-0.75, 0.75)
                dist_p = np.random.uniform(0.18, 0.28)
                px = dist_p * np.cos(angle_p)
                py = dist_p * np.sin(angle_p)
                # Keep platform and cube at least 10 cm apart
                if np.hypot(px - cx, py - cy) > 0.10:
                    self.platform_pos = np.array([px, py, 0.005], dtype=np.float32)
                    break
        else:
            self.cube_pos = np.array([0.22, 0.12, 0.011], dtype=np.float32)
            self.platform_pos = np.array([0.22, -0.15, 0.005], dtype=np.float32)

        return self.get_observation()

    def get_observation(self):
        grip_val = 1.0 if self.gripper_closed else 0.0
        return np.array([
            self.ee_pos[0], self.ee_pos[1], self.ee_pos[2], self.ee_pos[3], # 0..3: Robot EE Pose
            grip_val,                                                       # 4: Gripper Status
            self.cube_pos[0], self.cube_pos[1], self.cube_pos[2],           # 5..7: Cube Position
            self.platform_pos[0], self.platform_pos[1], self.platform_pos[2]# 8..10: Goal Platform
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

        obs = self.get_observation()
        
        dist_to_goal = np.linalg.norm(self.cube_pos[:2] - self.platform_pos[:2])
        is_success = bool(dist_to_goal < 0.04 and self.cube_pos[2] <= 0.025 and not self.gripper_closed)

        return obs, is_success
