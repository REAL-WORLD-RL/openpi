import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class AdaptiveActionDimWeightLoader(WeightLoader):
    """Loads weights from a checkpoint and adapts action dimension mismatches.

    This loader wraps a CheckpointWeightLoader and adapts the action projection layers
    (action_in_proj and action_out_proj) when the checkpoint has a different action_dim
    than the target model. It slices the weights to match the target action_dim.

    Args:
        params_path: Path to the checkpoint to load.
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        """Load weights and adapt action dimension mismatches."""
        # Load checkpoint directly (bypassing shape checks)
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)

        # Flatten both parameter dictionaries for easier manipulation
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

        # Adapt action projection layers if dimensions don't match
        result = {}
        for k, v_ref in flat_ref.items():
            if k in flat_loaded:
                v_loaded = flat_loaded[k]
                # Check if this is an action projection layer that needs adaptation
                if "action_in_proj/kernel" in k:
                    # action_in_proj.kernel: (action_dim, hidden_dim)
                    # Need to slice first dimension (rows)
                    if v_loaded.shape[0] != v_ref.shape[0]:
                        logger.info(
                            f"Adapting {k}: slicing from {v_loaded.shape} to {v_ref.shape} "
                            f"(taking first {v_ref.shape[0]} rows)"
                        )
                        result[k] = v_loaded[: v_ref.shape[0], :].astype(v_ref.dtype)
                    elif v_loaded.shape == v_ref.shape:
                        result[k] = v_loaded.astype(v_ref.dtype) if v_loaded.dtype != v_ref.dtype else v_loaded
                    else:
                        logger.warning(
                            f"Shape mismatch for {k}: loaded {v_loaded.shape} != expected {v_ref.shape}. "
                            "Using reference (initialized) weights."
                        )
                        result[k] = v_ref
                elif "action_out_proj/kernel" in k:
                    # action_out_proj.kernel: (hidden_dim, action_dim)
                    # Need to slice second dimension (columns)
                    if v_loaded.shape[1] != v_ref.shape[1]:
                        logger.info(
                            f"Adapting {k}: slicing from {v_loaded.shape} to {v_ref.shape} "
                            f"(taking first {v_ref.shape[1]} columns)"
                        )
                        result[k] = v_loaded[:, : v_ref.shape[1]].astype(v_ref.dtype)
                    elif v_loaded.shape == v_ref.shape:
                        result[k] = v_loaded.astype(v_ref.dtype) if v_loaded.dtype != v_ref.dtype else v_loaded
                    else:
                        logger.warning(
                            f"Shape mismatch for {k}: loaded {v_loaded.shape} != expected {v_ref.shape}. "
                            "Using reference (initialized) weights."
                        )
                        result[k] = v_ref
                elif "action_out_proj/bias" in k:
                    # action_out_proj.bias: (action_dim,)
                    # Need to slice first dimension
                    if v_loaded.shape[0] != v_ref.shape[0]:
                        logger.info(
                            f"Adapting {k}: slicing from {v_loaded.shape} to {v_ref.shape} "
                            f"(taking first {v_ref.shape[0]} elements)"
                        )
                        result[k] = v_loaded[: v_ref.shape[0]].astype(v_ref.dtype)
                    elif v_loaded.shape == v_ref.shape:
                        result[k] = v_loaded.astype(v_ref.dtype) if v_loaded.dtype != v_ref.dtype else v_loaded
                    else:
                        logger.warning(
                            f"Shape mismatch for {k}: loaded {v_loaded.shape} != expected {v_ref.shape}. "
                            "Using reference (initialized) weights."
                        )
                        result[k] = v_ref
                else:
                    # For other parameters, use normal merging (shape must match)
                    if v_loaded.shape == v_ref.shape:
                        result[k] = v_loaded.astype(v_ref.dtype) if v_loaded.dtype != v_ref.dtype else v_loaded
                    else:
                        # Shape mismatch for non-action layers - use reference (will be initialized)
                        logger.warning(
                            f"Shape mismatch for {k}: loaded {v_loaded.shape} != expected {v_ref.shape}. "
                            "Using reference (initialized) weights."
                        )
                        result[k] = v_ref
            else:
                # Key not in loaded params, use reference (will be initialized)
                result[k] = v_ref

        # Merge LoRA weights that might be missing
        return _merge_params(
            flax.traverse_util.unflatten_dict(result, sep="/"), params, missing_regex=".*lora.*"
        )


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")