# Mujoco Franka robot policy for training and inference
# Based on the sim_franka data format
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
class MujocoFrankaInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. 
    It is used for both training and inference for Mujoco Franka robot.

    Data format based on sim_franka:
    - State: observation.state (7-dimensional joint positions)
    - Images: observation.images.front, observation.images.wrist
    - Action: action (4-dimensional joint velocities)
    - Task: task (natural language instruction)
    """

    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse images to uint8 (H,W,C)
        base_image = _parse_image(data["observation.images.front"])
        wrist_image = _parse_image(data["observation.images.wrist"])
        
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                # Pi0 models support three image inputs: one third-person view,
                # and two wrist views (left and right).
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                # Pi0-FAST models support three image inputs: two base views and one wrist view.
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation.state"],
            "image": dict(zip(names, images)),
            "image_mask": dict(zip(names, image_masks)),
        }

        # Actions are only available during training.
        if "action" in data:
            inputs["actions"] = data["action"]

        # Pass the prompt (aka language instruction) to the model.
        if "task" in data:
            inputs["prompt"] = data["task"]

        return inputs


@dataclasses.dataclass(frozen=True)
class MujocoFrankaOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back to the dataset specific format. 
    It is used for inference only.
    """
    
    # Action dimension for Mujoco Franka robot (joint velocities)
    action_dim: int = 4

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions
        actions = np.asarray(data["actions"])
        # Handle both single sample (action_horizon, action_dim) and batch (batch_size, action_horizon, action_dim)
        if actions.ndim == 2:
            return {"actions": actions[:, :self.action_dim]}
        elif actions.ndim == 3:
            return {"actions": actions[:, :, :self.action_dim]}
        else:
            raise ValueError(f"Unexpected actions shape: {actions.shape}")
