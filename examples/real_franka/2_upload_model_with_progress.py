#!/usr/bin/env python3
"""
Upload model checkpoint to ModelScope with progress tracking
"""

import os
from modelscope.hub.api import HubApi
from pathlib import Path
from tqdm import tqdm
import shutil

MODEL_SCOPE_TOKEN = "ms-f8cd0f84-70dc-42be-910b-1d801832b214"
# Configuration
# LOCAL_CHECKPOINT_PATH = "checkpoints/pi05_libero_real_franka_lora/lora_libero_real_franka_pickup_100demos/29999"
LOCAL_CHECKPOINT_PATH = "checkpoints/pi05_real_franka_lora/full_libero_real_franka_pickup_100demos/29999"
MODEL_ID = "YishuaiCai/pi05_libero_full_real_franka_pickup"
# MODEL_ID = "YishuaiCai/pi05_base_lora_real_franka_pickup"


# LOCAL_CHECKPOINT_PATH = "checkpoints/pi05_mujoco_franka_lora/pi05_libero_mujoco_franka_lora/29999"
# MODEL_ID = "YishuaiCai/pi05_libero_lora_mujoco_franka_pickup"

def upload_model():
    """Upload model checkpoint to ModelScope"""
    
    # Get absolute path
    workspace_path = "/share_data/caiyishuai/openpi"
    checkpoint_path = os.path.join(workspace_path, LOCAL_CHECKPOINT_PATH)
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Model ID: {MODEL_ID}")
    
    # Initialize ModelScope API with token
    api = HubApi()
    api.login(MODEL_SCOPE_TOKEN)
    print("✓ Logged in to ModelScope")
    
    # Create temporary directory for organizing files
    temp_dir = os.path.join(workspace_path, "temp_model_upload")
    
    # 清理旧的临时目录（如果存在）
    if os.path.exists(temp_dir):
        print("清理旧的临时文件...")
        shutil.rmtree(temp_dir)
    
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        print("\nPreparing files for upload...")
        
        # Copy checkpoint files to temp directory
        checkpoint_files = os.listdir(checkpoint_path)
        for file in tqdm(checkpoint_files, desc="Copying files"):
            src = os.path.join(checkpoint_path, file)
            dst = os.path.join(temp_dir, file)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            elif os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
        
        # Upload files using HubApi
        print("\n开始上传到 ModelScope...")
        print("注意：这可能需要几分钟到几十分钟，取决于模型大小和网络速度")
        
        # 计算总文件大小
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
                file_count += 1
        
        print(f"总大小: {total_size / (1024**3):.2f} GB ({file_count} 个文件)")
        
        # 使用新的 upload_folder API（有进度显示）
        result = api.upload_folder(
            folder_path=temp_dir,
            repo_id=MODEL_ID,
            commit_message=f"Update checkpoint from {LOCAL_CHECKPOINT_PATH}"
        )
        
        print(f"\n✓ Successfully uploaded model to ModelScope!")
        print(f"Model URL: https://www.modelscope.cn/models/{MODEL_ID}")
        
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            print("\nCleaning up temporary files...")
            shutil.rmtree(temp_dir)
            print("✓ Cleanup complete")

if __name__ == "__main__":
    upload_model()

