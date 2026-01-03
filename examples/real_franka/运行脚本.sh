
# 数据存储位置
# rm -rf /home/liyu/.cache/huggingface/lerobot/plug_insert_100_demos_lerobot /share_data/caiyishuai/openpi/data/real_franka/plug_insert_100_demos_lerobot




# 先运行统计脚本
cd /share_data/caiyishuai/openpi
uv run scripts/compute_norm_stats.py --config-name real_plug_insert_pi05_libero_lora


# tmux send-keys -t franka_train "cd /share_data/caiyishuai/openpi && export HF_LEROBOT_HOME=/share_data/caiyishuai/openpi/data/real_franka && export CUDA_VISIBLE_DEVICES=4,5,6,7 && uv run scripts/train.py real_plug_insert_pi05_libero_lora --exp-name lora_libero_real_franka_plug_insert_100demos --overwrite" C-m



# 然后运行训练脚本
cd /share_data/caiyishuai/openpi/examples/real_franka/plug_insert
bash 1_train_full_real_franka_tmux.sh
bash 1_train_lora_real_franka_tmux.sh

# 新建一个终端查看
tmux attach -t franka_train

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