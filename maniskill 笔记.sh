# ln -s /share_data/wangpeishuo/huggingface_cache/lerobot/wps852/pushcube /home/caiyishuai/.cache/huggingface/lerobot/wps852/pushcube
# ln -s /share_data/wangpeishuo/huggingface_cache/lerobot/wps852/pushcube /home/liyu/.cache/huggingface/lerobot/wps852/pushcube


uv run scripts/compute_norm_stats.py --config-name pi05_maniskill5_lora

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_maniskill5_lora --exp-name=pushcube --overwrite