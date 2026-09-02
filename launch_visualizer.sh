#!/bin/bash
set -e

# Source ROS2 and local workspace
source /opt/ros/jazzy/setup.bash
source /home/karthikkrazy/dobot/install/setup.bash

echo "======================================================"
echo "          Dobot Magician 4-DOF Visualizer             "
echo "======================================================"
echo " Select an End Effector configuration:"
echo ""
echo "  [1] Standard Gripper               (4-DOF)"
echo "  [2] Extended Gripper               (4-DOF)"
echo "  [3] Suction Cup                    (4-DOF)"
echo "  [4] Writing / Drawing Pen          (3-DOF)"
echo "  [5] Bare Tool Flange (No Tool)     (3-DOF)"
echo "  [6] Gripper + RealSense D435i Cam  (4-DOF)"
echo "  [7] Suction Cup + RealSense Cam    (4-DOF)"
echo "======================================================"

# If argument passed directly, use it; otherwise ask interactively
if [ -n "$1" ]; then
    CHOICE="$1"
elif [ -t 0 ]; then
    read -p "Enter choice [1-7] (default: 1): " -n 1 -r CHOICE
    echo ""
else
    CHOICE="1"
fi

case "$CHOICE" in
    2)
        TOOL="extended_gripper"
        DOF="4"
        CAMERA="false"
        TOOL_NAME="Extended Gripper (4-DOF)"
        ;;
    3)
        TOOL="suction_cup"
        DOF="4"
        CAMERA="false"
        TOOL_NAME="Suction Cup (4-DOF)"
        ;;
    4)
        TOOL="pen"
        DOF="3"
        CAMERA="false"
        TOOL_NAME="Writing / Drawing Pen (3-DOF)"
        ;;
    5)
        TOOL="none"
        DOF="3"
        CAMERA="false"
        TOOL_NAME="Bare Tool Flange (3-DOF)"
        ;;
    6)
        TOOL="gripper"
        DOF="4"
        CAMERA="true"
        TOOL_NAME="Standard Gripper + RealSense D435i Camera (4-DOF)"
        ;;
    7)
        TOOL="suction_cup"
        DOF="4"
        CAMERA="true"
        TOOL_NAME="Suction Cup + RealSense D435i Camera (4-DOF)"
        ;;
    1|*)
        TOOL="gripper"
        DOF="4"
        CAMERA="false"
        TOOL_NAME="Standard Gripper (4-DOF)"
        ;;
esac

echo ""
echo ">> Launching: $TOOL_NAME"
echo ">> Press Ctrl+C in this terminal to stop the visualizer."
echo ""

ros2 launch dobot_description display.launch.py gui:=true DOF:="$DOF" tool:="$TOOL" use_camera:="$CAMERA"
