# kill the tmux session by running the following command
# tmux kill-session -t serl_session

# 训练好的模型保存在
# ls -lh /share_data/caiyishuai/openpi/checkpoints/pi05_real_franka_lora/real_franka_pickup_100demos_lora/




# 模型下载
# jax
# modelscope download --model Gnepua/pi05_libero  --local_dir ./pi05_libero
modelscope download --model hairuoliu/pi05_base
# 模型地址
~/.cache/openpi/openpi-assets/checkpoints/pi05_libero

# 全量微调
uv run scripts/compute_norm_stats.py --config-name pi05_libero
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite

# LoRA 微调
export CUDA_VISIBLE_DEVICES=4,5,6,7 && \
uv run scripts/train.py --config-name pi05_fast_libero_low_mem_finetune --exp-name franka_sim_100_lora --overwrite

uv run scripts/compute_norm_stats.py --config-name pi05_fast_libero_low_mem_finetune
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_fast_libero_low_mem_finetune --exp-name=my_experiment --overwrite



export CUDA_VISIBLE_DEVICES=6,7
export CUDA_VISIBLE_DEVICES=4,5


cd /share_data/tianyang/real_world_rl/openpi
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m openpi.training.train --config pi05_real_franka



# 模型上传
cd /share_data/caiyishuai/openpi/examples/real_franka
python upload_with_progress.py