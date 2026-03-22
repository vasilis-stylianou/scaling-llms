from __future__ import annotations
import importlib
import json
from typing import Any, Tuple
import torch
from dataclasses import dataclass

from scaling_llms.checkpointing.manager import CHECKPOINT_KEYS
from scaling_llms.constants import METADATA_FILES
from scaling_llms.tracking import Run


@dataclass(frozen=True)
class ModelClassInfoKeys:
    model_module: str = "model_module"
    model_class_name: str = "model_class_name"
    config_module: str = "config_module"
    config_class_name: str = "config_class_name"

MODEL_CLASS_INFO_KEYS = ModelClassInfoKeys()


def get_model_class_info(model: torch.nn.Module) -> dict[str, str]:
    """Extract model class and config class info for serialization."""
    model_class = type(model)
    config_class = type(model.cfg)
    return {
        MODEL_CLASS_INFO_KEYS.model_module: model_class.__module__,
        MODEL_CLASS_INFO_KEYS.model_class_name: model_class.__name__,
        MODEL_CLASS_INFO_KEYS.config_module: config_class.__module__,
        MODEL_CLASS_INFO_KEYS.config_class_name: config_class.__name__,
    }


def load_model_class(class_info: dict[str, str]) -> type:
    """Dynamically import and return the model class."""
    module = importlib.import_module(class_info[MODEL_CLASS_INFO_KEYS.model_module])
    return getattr(module, class_info[MODEL_CLASS_INFO_KEYS.model_class_name])

def load_config_class(class_info: dict[str, str]) -> type:
    """Dynamically import and return the config class."""
    module = importlib.import_module(class_info[MODEL_CLASS_INFO_KEYS.config_module])
    return getattr(module, class_info[MODEL_CLASS_INFO_KEYS.config_class_name])


def instantiate_model_from_run(run: Run) -> torch.nn.Module:
    """Instantiate the model (and its config) described in the run metadata.

    Returns a freshly constructed model (weights not loaded).
    """

    # Load model and config class info
    model_class_path = run.artifacts_dir.metadata_path(METADATA_FILES.model_class)
    class_info = json.loads(model_class_path.read_text())

    # Dynamically load the model and config class
    ModelClass = load_model_class(class_info)
    ConfigClass = load_config_class(class_info)
    
    # Load model config
    model_cfg_path = run.artifacts_dir.metadata_path(METADATA_FILES.model_config)
    model_cfg = ConfigClass.from_json(model_cfg_path)
    
    # Instantiate model
    model = ModelClass(model_cfg)

    return model


def load_model_from_checkpoint(
    run: Run,
    ckpt_name: str,
    map_location: str | None = None,
    strict: bool = True,
) -> Tuple[torch.nn.Module, dict[str, Any]]:
    """Convenience: instantiate model from run metadata, load checkpoint, move to device.

    Returns (model, trainer_state).
    """
    # Instantiate model from run metadata
    model = instantiate_model_from_run(run)

    # Load checkpoint state into model
    ckpt_path = run.artifacts_dir.checkpoint_path(ckpt_name)
    ckpt = torch.load(ckpt_path, map_location=map_location)

    if ckpt.get(CHECKPOINT_KEYS.model) is None:
        raise ValueError(f"Checkpoint at {ckpt_path} does not contain model state.")
   
    model.load_state_dict(ckpt[CHECKPOINT_KEYS.model], strict=strict)

    return model