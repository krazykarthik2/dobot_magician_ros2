# Dobot Magician ROS 2 Visualization Package

Official URDF, high-fidelity COLLADA (`.dae`) 3D meshes, and ROS 2 launch configurations for the **Dobot Magician 4-DOF** robotic arm.

---

## Features
- **4-DOF Robotic Arm Model**: Full kinematic chain including base rotation, rear arm, forearm, and end-effector rotation.
- **Multiple End-Effectors**:
  - Pneumatic Parallel Gripper (Standard & Extended)
  - Suction Cup
  - Writing / Drawing Pen
  - Bare Tool Flange
- **Optional Sensor Attachment**: Intel RealSense D435i depth camera.
- **Custom Visual Tuning**:
  - White plastic protective enclosures rendered with `0.4` alpha (semi-transparent).
  - Main structural aluminum beams, motor housings, and metal hardware rendered with `1.0` alpha (solid).
  - TF coordinate frames rendered with `0.2` alpha.
- **Interactive Launcher**: Simple terminal menu to choose and launch any end-effector setup without memorizing arguments.

---

## Prerequisites
- ROS 2 (e.g., Jazzy, Humble, Rolling)
- `robot_state_publisher`, `joint_state_publisher_gui`, `rviz2`, `xacro`

---

## Build & Setup

In your workspace:
```bash
colcon build
source install/setup.bash
```

---

## How to Run

### Interactive Mode
Run the launcher script and choose your configuration via numbers `1`–`7`:
```bash
./launch_visualizer.sh
```

### Direct ROS 2 Launch
```bash
ros2 launch dobot_description display.launch.py gui:=true DOF:=4 tool:=gripper
```
*(Options for `tool`: `gripper`, `extended_gripper`, `suction_cup`, `pen`, `none`. Set `use_camera:=true` to mount the RealSense camera.)*
