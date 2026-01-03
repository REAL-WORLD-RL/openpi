#!/bin/bash

# Real Franka 训练脚本 - Tmux 版本
# 使用 GPU 4,5,6,7

# Tmux 会话名称
SESSION_NAME="franka_train"

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 创建 logs 目录
LOGS_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOGS_DIR"

# 生成带时间戳的日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOGS_DIR}/train_${TIMESTAMP}.log"

# 检查 tmux 会话是否已存在
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Tmux 会话 '$SESSION_NAME' 已存在"
    echo "选项:"
    echo "  1. 附加到现有会话: tmux attach -t $SESSION_NAME"
    echo "  2. 杀死现有会话: tmux kill-session -t $SESSION_NAME"
    echo "  3. 然后重新运行此脚本"
    exit 1
fi

echo "=========================================="
echo "创建 tmux 会话: $SESSION_NAME"
echo "=========================================="
echo ""
echo "提示:"
echo "  - 附加到会话: tmux attach -t $SESSION_NAME"
echo "  - 分离会话: Ctrl+B 然后按 D"
echo "  - 杀死会话: tmux kill-session -t $SESSION_NAME"
echo "  - 日志文件: $LOG_FILE"
echo ""

# 创建 tmux 会话并运行训练
tmux new-session -d -s $SESSION_NAME

# 开启日志记录 - 将所有输出保存到日志文件
tmux pipe-pane -t $SESSION_NAME -o "cat >> '$LOG_FILE'"

# 发送环境变量设置命令
tmux send-keys -t $SESSION_NAME "export UV_LINK_MODE=copy" C-m
tmux send-keys -t $SESSION_NAME "export CUDA_VISIBLE_DEVICES=4,5,6,7" C-m
tmux send-keys -t $SESSION_NAME "export XLA_PYTHON_CLIENT_PREALLOCATE=false" C-m
tmux send-keys -t $SESSION_NAME "export XLA_PYTHON_CLIENT_ALLOCATOR=platform" C-m
tmux send-keys -t $SESSION_NAME "export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9" C-m
tmux send-keys -t $SESSION_NAME "export WANDB_API_KEY=5f07bbe343d183f389c30a3a6245463dca80ae0e" C-m
tmux send-keys -t $SESSION_NAME "export HF_HOME=/share_data/caiyishuai/.cache/huggingface" C-m
tmux send-keys -t $SESSION_NAME "export HF_DATASETS_CACHE=/share_data/caiyishuai/.cache/huggingface/datasets" C-m 
tmux send-keys -t $SESSION_NAME "export HF_LEROBOT_HOME=/share_data/caiyishuai/openpi/data/real_franka" C-m  # 注意：这里需要修改为实际的数据集路径
tmux send-keys -t $SESSION_NAME "export HF_ENDPOINT=https://hf-mirror.com" C-m

# 进入项目目录
tmux send-keys -t $SESSION_NAME "cd /share_data/caiyishuai/openpi" C-m

# 显示配置信息
tmux send-keys -t $SESSION_NAME "echo '==========================================' " C-m
tmux send-keys -t $SESSION_NAME "echo '配置信息:' " C-m
tmux send-keys -t $SESSION_NAME "echo '  GPU: 4,5,6,7' " C-m
tmux send-keys -t $SESSION_NAME "echo '  Config: pi05_libero_real_franka' " C-m
tmux send-keys -t $SESSION_NAME "echo '  Dataset: pickup_100_demos_lerobot' " C-m
tmux send-keys -t $SESSION_NAME "echo '  Dataset Path: \$HF_LEROBOT_HOME/pickup_100_demos_lerobot' " C-m
tmux send-keys -t $SESSION_NAME "echo '  HF Cache: \$HF_HOME' " C-m
tmux send-keys -t $SESSION_NAME "echo '  模型: ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero/params' " C-m
tmux send-keys -t $SESSION_NAME "echo '  资产: ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets' " C-m
tmux send-keys -t $SESSION_NAME "echo '==========================================' " C-m

# 步骤1: 计算归一化统计数据
tmux send-keys -t $SESSION_NAME "echo '' " C-m
tmux send-keys -t $SESSION_NAME "echo '步骤1: 计算归一化统计数据...' " C-m
tmux send-keys -t $SESSION_NAME "uv run scripts/compute_norm_stats.py --config-name pi05_libero_real_franka && echo '归一化统计完成!' || echo '归一化统计失败!'" C-m

# 等待第一步完成，然后开始训练
tmux send-keys -t $SESSION_NAME "echo '' " C-m
tmux send-keys -t $SESSION_NAME "echo '步骤2: 开始训练...' " C-m
tmux send-keys -t $SESSION_NAME "uv run scripts/train.py pi05_libero_real_franka --exp-name full_libero_real_franka_pickup_100demos --overwrite" C-m

# 训练完成提示
tmux send-keys -t $SESSION_NAME "echo '' " C-m
tmux send-keys -t $SESSION_NAME "echo '训练完成！' " C-m

echo "=========================================="
echo "Tmux 会话 '$SESSION_NAME' 已启动！"
echo ""
echo "使用以下命令附加到会话查看进度:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "在会话内按 Ctrl+B 然后按 D 可以分离会话（训练继续运行）"
echo ""
echo "日志文件保存在:"
echo "  $LOG_FILE"
echo ""
echo "查看实时日志: tail -f $LOG_FILE"
echo "=========================================="



# tmux attach -t franka_train
# 查看下载的文件大小
# cd /share_data/caiyishuai/openpi
# export HF_LEROBOT_HOME=/share_data/caiyishuai/openpi/data/real_franka
# uv run scripts/compute_norm_stats.py --config-name pi05_libero_real_franka