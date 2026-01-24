from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        
        # Detect if input already has batch dimension
        # Check state shape: if ndim==1, it's single sample; if ndim==2, it's batched
        state_array = np.asarray(inputs["state"])
        has_batch_dim = state_array.ndim >= 2
        
        logging.info(f"[Policy.infer] After transform, state.shape = {state_array.shape}, has_batch_dim = {has_batch_dim}")
        
        # Log all input keys and shapes after transform
        for key, value in inputs.items():
            if isinstance(value, (np.ndarray, jnp.ndarray)):
                logging.info(f"[Policy.infer] inputs['{key}'].shape = {np.asarray(value).shape}")
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (np.ndarray, jnp.ndarray)):
                        logging.info(f"[Policy.infer] inputs['{key}']['{sub_key}'].shape = {np.asarray(sub_value).shape}")
        
        if not self._is_pytorch_model:
            # JAX model
            if not has_batch_dim:
                # Single input: selectively add batch dimension
                # For images: check ndim, if 3 (H,W,C) add batch, if 4 (B,H,W,C) keep as is
                if "image" in inputs:
                    inputs["image"] = {
                        k: jnp.asarray(v)[np.newaxis, ...] if np.asarray(v).ndim == 3 else jnp.asarray(v)
                        for k, v in inputs["image"].items()
                    }
                # For image_mask: convert scalars to 1D arrays with batch dimension
                if "image_mask" in inputs:
                    inputs["image_mask"] = {
                        k: jnp.asarray([v]) if np.asarray(v).ndim == 0 else jnp.asarray(v)
                        for k, v in inputs["image_mask"].items()
                    }
                # For state and other arrays: add batch dim
                inputs["state"] = jnp.asarray(inputs["state"])[np.newaxis, ...]
                # For other optional fields
                for key in ["actions", "tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask"]:
                    if key in inputs and isinstance(inputs[key], (np.ndarray, jnp.ndarray)):
                        arr = np.asarray(inputs[key])
                        # Only add batch dim if it doesn't have one
                        if arr.ndim >= 1:
                            inputs[key] = jnp.asarray(arr)[np.newaxis, ...]
            else:
                # Already batched: just convert to jax arrays
                if "image" in inputs:
                    inputs["image"] = {k: jnp.asarray(v) for k, v in inputs["image"].items()}
                if "image_mask" in inputs:
                    inputs["image_mask"] = {
                        k: jnp.asarray(v) if np.asarray(v).ndim >= 1 else jnp.asarray([v] * state_array.shape[0])
                        for k, v in inputs["image_mask"].items()
                    }
                inputs["state"] = jnp.asarray(inputs["state"])
                # For tokenized prompts: if they exist but don't have batch dim, repeat them
                for key in ["tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask"]:
                    if key in inputs:
                        arr = np.asarray(inputs[key])
                        if isinstance(inputs[key], (np.ndarray, jnp.ndarray)):
                            # If array doesn't start with batch dimension, repeat it
                            if arr.ndim >= 1:
                                # Check if first dim matches batch size
                                if arr.shape[0] != state_array.shape[0]:
                                    # Need to repeat: tile the array to match batch size
                                    inputs[key] = jnp.tile(jnp.asarray(arr)[None, ...], (state_array.shape[0], *([1] * arr.ndim)))
                                else:
                                    inputs[key] = jnp.asarray(arr)
                for key in ["actions"]:
                    if key in inputs and isinstance(inputs[key], (np.ndarray, jnp.ndarray)):
                        inputs[key] = jnp.asarray(inputs[key])
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # PyTorch model
            if not has_batch_dim:
                # Single input: selectively add batch dimension
                if "image" in inputs:
                    inputs["image"] = {
                        k: torch.from_numpy(np.array(v)).to(self._pytorch_device)[None, ...] if np.asarray(v).ndim == 3 
                        else torch.from_numpy(np.array(v)).to(self._pytorch_device)
                        for k, v in inputs["image"].items()
                    }
                if "image_mask" in inputs:
                    inputs["image_mask"] = {
                        k: torch.from_numpy(np.array([v])).to(self._pytorch_device) if np.asarray(v).ndim == 0 
                        else torch.from_numpy(np.array(v)).to(self._pytorch_device)
                        for k, v in inputs["image_mask"].items()
                    }
                inputs["state"] = torch.from_numpy(np.array(inputs["state"])).to(self._pytorch_device)[None, ...]
                for key in ["actions", "tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask"]:
                    if key in inputs and isinstance(inputs[key], (np.ndarray, list)):
                        arr = np.array(inputs[key])
                        if arr.ndim >= 1:
                            inputs[key] = torch.from_numpy(arr).to(self._pytorch_device)[None, ...]
            else:
                # Already batched: just convert to torch tensors
                if "image" in inputs:
                    inputs["image"] = {k: torch.from_numpy(np.array(v)).to(self._pytorch_device) for k, v in inputs["image"].items()}
                if "image_mask" in inputs:
                    inputs["image_mask"] = {
                        k: torch.from_numpy(np.array(v) if np.asarray(v).ndim >= 1 else np.array([v] * state_array.shape[0])).to(self._pytorch_device)
                        for k, v in inputs["image_mask"].items()
                    }
                inputs["state"] = torch.from_numpy(np.array(inputs["state"])).to(self._pytorch_device)
                # For tokenized prompts: if they exist but don't have batch dim, repeat them
                for key in ["tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask"]:
                    if key in inputs:
                        arr = np.array(inputs[key])
                        if isinstance(inputs[key], (np.ndarray, list)):
                            # If array doesn't start with batch dimension, repeat it
                            if arr.ndim >= 1:
                                # Check if first dim matches batch size
                                if arr.shape[0] != state_array.shape[0]:
                                    # Need to repeat: tile the array to match batch size
                                    inputs[key] = torch.from_numpy(np.tile(arr[None, ...], (state_array.shape[0], *([1] * arr.ndim)))).to(self._pytorch_device)
                                else:
                                    inputs[key] = torch.from_numpy(arr).to(self._pytorch_device)
                for key in ["actions"]:
                    if key in inputs and isinstance(inputs[key], (np.ndarray, list)):
                        inputs[key] = torch.from_numpy(np.array(inputs[key])).to(self._pytorch_device)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        
        # Log inputs before model inference
        logging.info(f"[Policy.infer] Before model inference:")
        logging.info(f"  observation.state.shape = {observation.state.shape}")
        if observation.images:
            for img_key, img_val in observation.images.items():
                logging.info(f"  observation.images['{img_key}'].shape = {np.asarray(img_val).shape}")
        if observation.tokenized_prompt is not None:
            logging.info(f"  observation.tokenized_prompt.shape = {np.asarray(observation.tokenized_prompt).shape}")
        
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        
        logging.info(f"[Policy.infer] Model inference took {model_time * 1000:.2f} ms")
        logging.info(f"[Policy.infer] Raw model output actions.shape = {np.asarray(outputs['actions']).shape}")
        
        # Remove batch dimension if we added it (for single input)
        if self._is_pytorch_model:
            if not has_batch_dim:
                outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
            else:
                outputs = jax.tree.map(lambda x: np.asarray(x.detach().cpu()), outputs)
        else:
            if not has_batch_dim:
                outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
            else:
                outputs = jax.tree.map(lambda x: np.asarray(x), outputs)

        logging.info(f"[Policy.infer] After unbatching (if needed), actions.shape = {np.asarray(outputs['actions']).shape}")
        
        outputs = self._output_transform(outputs)
        
        logging.info(f"[Policy.infer] After output_transform, actions.shape = {np.asarray(outputs['actions']).shape}")
        logging.info(f"[Policy.infer] Final output keys: {list(outputs.keys())}")
        
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
