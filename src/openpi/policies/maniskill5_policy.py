# wps: 我们自己的基于maniskill数据集使用的policy，训练用
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class Maniskill5Inputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # For maniskill dataset, images are stored in "observation.images.base_camera" and optionally "observation.images.hand_camera"
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros.
        base_image = _parse_image(data["observation.images.base_camera"])
        
        # hand_camera 是可选的（某些任务没有）
        if "observation.images.hand_camera" in data:
            hand_image = _parse_image(data["observation.images.hand_camera"])
            has_hand_camera = True
        else:
            hand_image = np.zeros_like(base_image)
            has_hand_camera = False

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": np.asarray(data["observation.state"]),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": hand_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": True,
                "left_wrist_0_rgb": True if has_hand_camera else (True if self.model_type == _model.ModelType.PI0_FAST else False),
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": True if self.model_type == _model.ModelType.PI0_FAST else False,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        # For maniskill dataset, action key is "action" not "actions"
        if "action" in data:
            inputs["actions"] = data["action"]

        # Pass the prompt (aka language instruction) to the model.
        # For maniskill dataset, the instruction is stored in "task" key.
        if "task" in data:
            inputs["prompt"] = data["task"]

        return inputs


@dataclasses.dataclass(frozen=True)
class Maniskill5Outputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For maniskill5 dataset with pd_ee_delta_pose control mode:
        # - Action dimension is 7: 6 EEF delta pose + 1 gripper
        # - Actions are delta (incremental) values
        
        actions = np.asarray(data["actions"])
        # Handle both single sample (action_horizon, action_dim) and batch (batch_size, action_horizon, action_dim)
        if actions.ndim == 2:
            # Single sample: (action_horizon, action_dim)
            return {"actions": actions[:, :7]}
        elif actions.ndim == 3:
            # Batch: (batch_size, action_horizon, action_dim)
            return {"actions": actions[:, :, :7]}
        else:
            raise ValueError(f"Unexpected actions shape: {actions.shape}")
