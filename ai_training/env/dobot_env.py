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
        """Compute end-effector (X, Y, Z, Yaw) from joint angles (radians)."""
        r = self.L2 * np.cos(j2) + self.L3 * np.cos(j2 + j3) + self.L4
        x = r * np.cos(j1)
        y = r * np.sin(j1)
        z = self.L1 + self.L2 * np.sin(j2) + self.L3 * np.sin(j2 + j3)
        yaw = j1 + j4
        return np.array([x, y, z, yaw], dtype=np.float32)

    def inverse(self, x, y, z, yaw=0.0):
        """Compute joint angles [j1, j2, j3, j4] from target (X, Y, Z, Yaw)."""
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
    """Fast, deterministic physics simulation for Dobot Pick & Place with normalized state spaces."""
    def __init__(self):
        self.kin = DobotKinematics()
        self.cube_size = 0.022
        self.platform_pos = np.array([0.22, -0.15, 0.005], dtype=np.float32)
        self.gripper_closed = False
        self.grasped = False
        
        # State
        self.ee_pos = np.array([0.20, 0.0, 0.12, 0.0], dtype=np.float32)
        self.cube_pos = np.array([0.22, 0.12, 0.011], dtype=np.float32)
        self.reset()

    def reset(self, random_cube=True):
        self.ee_pos = np.array([0.20, 0.0, 0.12, 0.0], dtype=np.float32)
        self.gripper_closed = False
        self.grasped = False

        if random_cube:
            angle = np.random.uniform(-np.pi/4, np.pi/4)
            dist = np.random.uniform(0.18, 0.28)
            cx = dist * np.cos(angle)
            cy = dist * np.sin(angle)
            if cy < -0.06 and cx < 0.26:
                cy += 0.12
            self.cube_pos = np.array([cx, cy, 0.011], dtype=np.float32)
        else:
            self.cube_pos = np.array([0.22, 0.12, 0.011], dtype=np.float32)

        return self.get_observation()

    def get_observation(self):
        """
        Rich 16-dimensional observation vector with relative vectors:
        - EE Pos (X, Y, Z, Yaw) [4]
        - Gripper open/closed [1]
        - Cube Pos (X, Y, Z) [3]
        - Goal Platform Pos (X, Y, Z) [3]
        - Vector from EE to Cube (dx, dy, dz) [3]
        - Vector from Cube to Goal Platform (dx, dy, dz) [3]
        - Grasped status flag [1]
        Total: 18 dimensions
        """
        ee_xyz = self.ee_pos[:3]
        vec_to_cube = self.cube_pos - ee_xyz
        vec_cube_to_goal = self.platform_pos - self.cube_pos
        grip_val = 1.0 if self.gripper_closed else 0.0
        grasp_val = 1.0 if self.grasped else 0.0

        return np.array([
            self.ee_pos[0], self.ee_pos[1], self.ee_pos[2], self.ee_pos[3], # 0..3: EE Pose
            grip_val,                                                       # 4: Gripper status
            self.cube_pos[0], self.cube_pos[1], self.cube_pos[2],           # 5..7: Cube pos
            self.platform_pos[0], self.platform_pos[1], self.platform_pos[2],# 8..10: Goal pos
            vec_to_cube[0], vec_to_cube[1], vec_to_cube[2],                 # 11..13: EE -> Cube
            vec_cube_to_goal[0], vec_cube_to_goal[1], vec_cube_to_goal[2], # 14..16: Cube -> Goal
            grasp_val                                                       # 17: Grasp flag
        ], dtype=np.float32)

    def step(self, target_ee):
        """
        target_ee: [X, Y, Z, Yaw, Gripper (0: open, 1: closed)]
        """
        self.ee_pos = np.array(target_ee[:4], dtype=np.float32)
        self.gripper_closed = bool(target_ee[4] > 0.5)

        ee_xyz = self.ee_pos[:3]
        dist_to_cube = np.linalg.norm(ee_xyz - self.cube_pos)

        # Grasping logic
        if self.gripper_closed:
            if dist_to_cube < 0.03:
                self.grasped = True
        else:
            self.grasped = False

        # If grasped, cube follows end-effector
        if self.grasped:
            self.cube_pos = ee_xyz.copy()
            self.cube_pos[2] = max(self.cube_pos[2] - 0.015, 0.011)
        else:
            # Gravity
            if self.cube_pos[2] > 0.011:
                self.cube_pos[2] = max(self.cube_pos[2] - 0.01, 0.011)

        obs = self.get_observation()
        
        # Check success
        dist_to_goal = np.linalg.norm(self.cube_pos[:2] - self.platform_pos[:2])
        is_success = bool(dist_to_goal < 0.04 and self.cube_pos[2] <= 0.025 and not self.gripper_closed)

        return obs, is_success
