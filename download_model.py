import os
import logging
from openpi.shared import download
from openpi.policies import policy_config
from openpi.training import config as _config
import numpy as np

# --- 设置下载目录 ---
# # 这里修改为你想要存放模型的路径
# CUSTOM_DATA_HOME = os.path.abspath("./download_models")
# os.environ["OPENPI_DATA_HOME"] = CUSTOM_DATA_HOME
# ------------------

config = _config.get_config("pi05_libero")

# 配置日志以便看到下载进度
logging.basicConfig(level=logging.INFO)

# print(f"开始下载 pi05_libero 模型到: {CUSTOM_DATA_HOME} ...")

# 下载模型
# 最终路径会是: {CUSTOM_DATA_HOME}/openpi-assets/checkpoints/pi05_libero
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
print(f"实际下载路径: {checkpoint_dir}")
exit()
# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)
print('created policy')
# Run inference on a dummy example.
example = {
    "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
    "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
    "observation/state": np.random.rand(8),
    # "observation/wrist_image_left": ...,
    # ...
    "prompt": "pick up the fork",
}
action_chunk = policy.infer(example)["actions"]
print(f"模型下载完成！保存路径: {checkpoint_dir}")
