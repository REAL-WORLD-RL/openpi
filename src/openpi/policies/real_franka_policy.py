# Real Franka robot policy for training and inference
# Based on the Franka data format from convert_franka_data_to_lerobot.py
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H,W,C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RealFrankaInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. 
    It is used for both training and inference for Real Franka robot.

    Data format based on convert_franka_data_to_lerobot.py:
    - State: observation.state (19-dimensional)
    - Images: observation.images.<camera_name> (multiple cameras possible)
    - Action: action (variable dimension, extracted from data)
    - Task: task (natural language instruction)
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse images to uint8 (H,W,C) since LeRobot stores as float32 (C,H,W)
        # Real Franka data has images stored in "observation.images.<camera_name>"
        # The camera names are dynamically determined from the data
        
        # Get all available cameras from flattened keys
        # LeRobot stores images as "observation.images.<camera_name>", not nested dicts
        available_cameras = []
        image_prefix = "observation.images."
        for key in data.keys():
            if key.startswith(image_prefix):
                cam_name = key[len(image_prefix):]
                available_cameras.append(cam_name)
        
        # Pi0 models support three image inputs: one third-person view,
        # and two wrist views (left and right).
        # We'll map the available cameras to these three slots.
        
        # Initialize with None
        base_image = None
        left_wrist_image = None
        right_wrist_image = None
        
        # Map cameras based on naming conventions
        for cam_name in available_cameras:
            cam_lower = cam_name.lower()
            img = _parse_image(data[f"observation.images.{cam_name}"])
            
            # Assign to appropriate slot based on camera name
            # Base/exterior/third-person cameras (including "side_policy")
            if any(keyword in cam_lower for keyword in ["base", "exterior", "third", "side", "policy"]):
                if base_image is None:
                    base_image = img
            # Wrist cameras
            elif "wrist" in cam_lower or "hand" in cam_lower:
                # Extract number from camera name (e.g., wrist_1, wrist_2)
                if "1" in cam_name or "left" in cam_lower:
                    left_wrist_image = img
                elif "2" in cam_name or "right" in cam_lower:
                    right_wrist_image = img
                else:
                    # If wrist camera without left/right specification, use as left
                    if left_wrist_image is None:
                        left_wrist_image = img
                    elif right_wrist_image is None:
                        right_wrist_image = img
            else:
                # Unknown camera type, use as base if not set
                if base_image is None:
                    base_image = img
        
        # If we still don't have a base image, use the first available camera
        if base_image is None and available_cameras:
            base_image = _parse_image(data[f"observation.images.{available_cameras[0]}"])
        
        # If no base image at all, create a dummy one
        if base_image is None:
            base_image = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # Fill in missing wrist images with zeros
        if left_wrist_image is None:
            left_wrist_image = np.zeros_like(base_image)
        if right_wrist_image is None:
            right_wrist_image = np.zeros_like(base_image)
        
        # Determine which images are actually available (not all zeros)
        has_left_wrist = not np.all(left_wrist_image == 0)
        has_right_wrist = not np.all(right_wrist_image == 0)

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation.state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST.
                "left_wrist_0_rgb": np.True_ if has_left_wrist else (np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_),
                "right_wrist_0_rgb": np.True_ if has_right_wrist else (np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_),
            },
        }

        # Actions are only available during training.
        if "action" in data:
            inputs["actions"] = data["action"]

        # Pass the prompt (aka language instruction) to the model.
        if "task" in data:
            inputs["prompt"] = data["task"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RealFrankaOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back to the dataset specific format. 
    It is used for inference only.

    For Real Franka robot, the action dimension varies based on your robot configuration.
    Common values:
    - 7: Standard Franka Panda arm (7 DOF, joint positions only)
    - 8: Arm + gripper (7 DOF arm + 1 gripper)
    - 9: Arm + gripper width (7 DOF arm + 2 gripper values)
    
    Set this value to match the action dimension in your training dataset.
    """
    
    # Action dimension for Real Franka robot
    # This should match the action dimension in your dataset
    # Check convert_franka_data_to_lerobot.py output or your pkl file to confirm
    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        return {"actions": np.asarray(data["actions"][:, :self.action_dim])}