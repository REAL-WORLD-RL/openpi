

# 先运行统计脚本
cd /share_data/caiyishuai/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_mujoco_franka_lora

# 然后运行训练脚本
cd /share_data/caiyishuai/openpi/examples/mujoco_franka
bash 1_train_full_mujoco_franka_tmux.sh
bash 1_train_lora_mujoco_franka_tmux.sh

# 新建一个终端查看
tmux attach -t franka_train_mujoco

# 杀死 tmux 会话
# tmux kill-session -t franka_train_mujoco






##############################
# 先运行统计脚本
cd /share_data/caiyishuai/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_mujoco_franka

# 然后运行训练脚本
cd /share_data/caiyishuai/openpi/examples/mujoco_franka
bash 1_train_full_mujoco_franka_tmux.sh

# 新建一个终端查看
tmux attach -t franka_train_mujoco

# 杀死 tmux 会话
# tmux kill-session -t franka_train_mujoco