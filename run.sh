#!/bin/zsh
cd "$(dirname "$0")" || exit 1
eval "$(/opt/homebrew/bin/conda shell.zsh hook)"
conda activate ros_env
source install/setup.zsh

ros2 run crearobot_brain orchestrator &
ORCH_PID=$!
trap 'kill $ORCH_PID 2>/dev/null' EXIT

ros2 launch crearobot_brain launch.py
