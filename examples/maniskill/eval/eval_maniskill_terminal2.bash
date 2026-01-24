#!/bin/bash

# Set the repository root directory directly
REPO_ROOT="/home/magictavern/projects/openpi"

# Change to the repository root
cd "$REPO_ROOT" || exit 1

# Disable jaxtyping type checking to avoid inference issues
export JAXTYPING_DISABLE=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Launch the policy server with uv
# Using your trained model checkpoint (32-dim action output)
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_base_maniskill5_lora_8obs \
    --policy.dir=checkpoints/pi0_base_lora_maniskill_stackcube_4000 \


    # --policy.dir=checkpoints/pi05_base_maniskill5_lora_8obs/pi05_base_maniskill5_lora_8obs_stackcube/4000 \

# stackcube 20000: 77.% 10000 85.9%
# 4000 81%


# lsof -i :8000
    # --policy.dir=checkpoints/pi05_base_maniskill5_lora_8obs/pi05_base_maniskill5_lora_8obs_plugcharger/40000

    # --policy.dir=checkpoints/pi05_base_maniskill5_lora_8obs/pi05_base_maniskill5_lora_8obs_stackcube/29999


