#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "======================================================"
echo "          Dobot Magician AI Training Suite            "
echo "======================================================"
echo " Choose an action:"
echo ""
echo "  [1] Auto-Generate 50 Expert Demonstrations (Visual & Fast)"
echo "  [2] Manual Teleoperation Recorder          (Keyboard GUI)"
echo "  [3] Train Imitation Model                  (Fast CPU Behavior Cloning)"
echo "  [4] Test / Autopilot Replay                (Evaluate on novel scenes)"
echo "  [5] Clear / Delete All Recorded Demos"
echo "  [6] Delete Trained Policy Model Checkpoints (.pth / .npz)"
echo "======================================================"

if [ -n "$1" ]; then
    CHOICE="$1"
elif [ -t 0 ]; then
    read -p "Enter choice [1-6] (default: 1): " -n 1 -r CHOICE
    echo ""
else
    CHOICE="1"
fi

case "$CHOICE" in
    2)
        echo ">> Launching Manual Teleop Demonstration Recorder..."
        python3 "$DIR/scripts/teleop_recorder.py"
        ;;
    3)
        echo ">> Starting CPU Imitation Learning Training..."
        python3 "$DIR/scripts/train_imitation.py"
        ;;
    4)
        echo ">> Running AI Autopilot Evaluation on Random Cubes..."
        python3 "$DIR/scripts/eval_policy.py"
        ;;
    5)
        echo ">> Removing all recorded demonstrations..."
        rm -f "$DIR/data/demos"/*.npz
        echo ">> Demos folder cleared!"
        ;;
    6)
        echo ">> Deleting trained policy model checkpoints and normalization stats..."
        rm -f "$DIR/models"/*.pth "$DIR/models"/*.npz
        echo ">> Trained model checkpoints successfully deleted!"
        ;;
    1|*)
        echo ">> Launching Auto-Demonstration Generator..."
        python3 "$DIR/scripts/auto_generate_demos.py" 50
        ;;
esac
