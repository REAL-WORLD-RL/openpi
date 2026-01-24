#!/bin/bash

# Set the repository root directory directly
REPO_ROOT="/home/magictavern/projects/openpi"

# Use the Python interpreter directly from the maniskill virtual environment
MANISKILL_PYTHON="/home/magictavern/miniconda3/envs/serl2/bin/python"

if [ ! -f "$MANISKILL_PYTHON" ]; then
    echo "Error: Python interpreter not found at $MANISKILL_PYTHON"
    echo "Please check if maniskill virtual environment exists at /share_data/caiyishuai/REAL-WORLD-RL/ManiSkill/.venv"
    exit 1
fi

# Change to the working directory
cd $REPO_ROOT/examples/maniskill/eval

# Add ManiSkill to PYTHONPATH (ManiSkill is installed here)
export PYTHONPATH=$PYTHONPATH:/share_data/wangpeishuo/maniskill

# Add openpi_client to PYTHONPATH (only need the client library, not the full openpi codebase)
export PYTHONPATH=$PYTHONPATH:$REPO_ROOT/packages/openpi-client/src

# Create output directory if it doesn't exist
mkdir -p $REPO_ROOT/data/maniskill/videos

# Run the evaluation script using the maniskill conda environment's Python interpreter
# Use GPU rendering now that Vulkan is properly configured
# $MANISKILL_PYTHON maniskill_eval_main.py \
$MANISKILL_PYTHON maniskill_eval_main_obs8.py \
    --args.host 0.0.0.0 \
    --args.port 8000 \
    --args.sim-backend cpu \
    --args.render-backend "gpu" \
    --args.num-trials-per-task 100 \
    --args.max-episode-steps 300 \
    --args.video-out-path $REPO_ROOT/data/maniskill/videos \
    --args.env-id PushCube-v1 \


    # --args.env-id StackCube-v1 \
    # --args.env-id PushCube-v1

    # --args.env-id PullCubeTool-v1
    # --args.env-id PushCube-v1
    # --args.env-id StackCube-v1 \