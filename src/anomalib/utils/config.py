"""Get configurable parameters."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Iterable, Sequence, ValuesView
from pathlib import Path
from typing import *

from jsonargparse import Namespace
from jsonargparse import Path as JSONArgparsePath
from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)


def _convert_nested_path_to_str(config: Any) -> Any:  # noqa: ANN401
    """Goes over the dictionary and converts all path values to str."""
    if isinstance(config, dict):
        for key, value in config.items():
            config[key] = _convert_nested_path_to_str(value)
    elif isinstance(config, list):
        for i, item in enumerate(config):
            config[i] = _convert_nested_path_to_str(item)
    elif isinstance(config, Path | JSONArgparsePath):
        config = str(config)
    return config


def to_nested_dict(config: dict) -> dict:
    """Convert the flattened dictionary to nested dictionary.

    Examples:
        >>> config = {
                "dataset.category": "bottle",
                "dataset.image_size": 224,
                "model_name": "padim",
            }
        >>> to_nested_dict(config)
        {
            "dataset": {
                "category": "bottle",
                "image_size": 224,
            },
            "model_name": "padim",
        }
    """
    out: dict[str, Any] = {}
    for key, value in config.items():
        keys = key.split(".")
        _dict = out
        for k in keys[:-1]:
            _dict = _dict.setdefault(k, {})
        _dict[keys[-1]] = value
    return out


def to_yaml(config: Namespace | ListConfig | DictConfig) -> str:
    """Convert the config to a yaml string.

    Args:
        config (Namespace | ListConfig | DictConfig): Config

    Returns:
        str: YAML string
    """
    _config = config.clone() if isinstance(config, Namespace) else config.copy()
    if isinstance(_config, Namespace):
        _config = _config.as_dict()
        _config = _convert_nested_path_to_str(_config)
    return OmegaConf.to_yaml(_config)


def to_tuple(input_size: int | ListConfig) -> tuple[int, int]:
    """Convert int or list to a tuple.

    Args:
        input_size (int | ListConfig): input_size

    Example:
        >>> to_tuple(256)
        (256, 256)
        >>> to_tuple([256, 256])
        (256, 256)

    Raises:
        ValueError: Unsupported value type.

    Returns:
        tuple[int, int]: Tuple of input_size
    """
    ret_val: tuple[int, int]
    if isinstance(input_size, int):
        ret_val = cast(tuple[int, int], (input_size,) * 2)
    elif isinstance(input_size, ListConfig | Sequence):
        if len(input_size) != 2:
            msg = "Expected a single integer or tuple of length 2 for width and height."
            raise ValueError(msg)

        ret_val = cast(tuple[int, int], tuple(input_size))
    else:
        msg = f"Expected either int or ListConfig, got {type(input_size)}"
        raise TypeError(msg)
    return ret_val


def convert_valuesview_to_tuple(values: ValuesView) -> list[tuple]:
    """Convert a ValuesView object to a list of tuples.

    This is useful to get list of possible values for each parameter in the config and a tuple for values that are
    are to be patched. Ideally this is useful when used with product.

    Example:
        >>> params = DictConfig({
                "dataset.category": [
                    "bottle",
                    "cable",
                ],
                "dataset.image_size": 224,
                "model_name": ["padim"],
            })
        >>> convert_to_tuple(params.values())
        [('bottle', 'cable'), (224,), ('padim',)]
        >>> list(itertools.product(*convert_to_tuple(params.values())))
        [('bottle', 224, 'padim'), ('cable', 224, 'padim')]

    Args:
        values: ValuesView: ValuesView object to be converted to a list of tuples.

    Returns:
        list[Tuple]: List of tuples.
    """
    return_list = []
    for value in values:
        if isinstance(value, Iterable) and not isinstance(value, str):
            return_list.append(tuple(value))
        else:
            return_list.append((value,))
    return return_list


def flatten_dict(config: dict, prefix: str = "") -> dict:
    """Flatten the dictionary.

    Examples:
        >>> config = {
                "dataset": {
                    "category": "bottle",
                    "image_size": 224,
                },
                "model_name": "padim",
            }
        >>> flatten_dict(config)
        {
            "dataset.category": "bottle",
            "dataset.image_size": 224,
            "model_name": "padim",
        }
    """
    out = {}
    for key, value in config.items():
        if isinstance(value, dict):
            out.update(flatten_dict(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def namespace_from_dict(container: dict) -> Namespace:
    """Convert dictionary to Namespace recursively.

    Examples:
        >>> container = {
                "dataset": {
                    "category": "bottle",
                    "image_size": 224,
                },
                "model_name": "padim",
            }
        >>> namespace_from_dict(container)
        Namespace(dataset=Namespace(category='bottle', image_size=224), model_name='padim')
    """
    output = Namespace()
    for k, v in container.items():
        if isinstance(v, dict):
            setattr(output, k, namespace_from_dict(v))
        else:
            setattr(output, k, v)
    return output


def dict_from_namespace(container: Namespace) -> dict:
    """Convert Namespace to dictionary recursively.

    Examples:
        >>> from jsonargparse import Namespace
        >>> ns = Namespace()
        >>> ns.a = 1
        >>> ns.b = Namespace()
        >>> ns.b.c = 2
        >>> dict_from_namespace(ns)
        {'a': 1, 'b': {'c': 2}}
    """
    output = {}
    for k, v in container.__dict__.items():
        if isinstance(v, Namespace):
            output[k] = dict_from_namespace(v)
        else:
            output[k] = v
    return output


def update_config(config: DictConfig | ListConfig | Namespace) -> DictConfig | ListConfig | Namespace:
    """Update config.

    Args:
        config: Configurable parameters.

    Returns:
        DictConfig | ListConfig | Namespace: Updated config.
    """
    _show_warnings(config)

    return _update_nncf_config(config)


def _update_nncf_config(config: DictConfig | ListConfig) -> DictConfig | ListConfig:
    """Set the NNCF input size based on the value of the crop_size parameter in the configurable parameters object.

    Args:
        config (DictConfig | ListConfig): Configurable parameters of the current run.

    Returns:
        DictConfig | ListConfig: Updated configurable parameters in DictConfig object.
    """
    if "optimization" in config and "nncf" in config.optimization:
        if "input_info" not in config.optimization.nncf:
            config.optimization.nncf["input_info"] = {"sample_size": None}
        config.optimization.nncf.input_info.sample_size = [1, 3, 10, 10]
        if config.optimization.nncf.apply and "update_config" in config.optimization.nncf:
            return OmegaConf.merge(config, config.optimization.nncf.update_config)
    return config


def _show_warnings(config: DictConfig | ListConfig | Namespace) -> None:
    """Show warnings if any based on the configuration settings.

    Args:
        config (DictConfig | ListConfig | Namespace): Configurable parameters for the current run.
    """
    if "clip_length_in_frames" in config.data and config.data.init_args.clip_length_in_frames > 1:
        logger.warning(
            "Anomalib's models and visualizer are currently not compatible with video datasets with a clip length > 1. "
            "Custom changes to these modules will be needed to prevent errors and/or unpredictable behaviour.",
        )
    if (
        "devices" in config.trainer
        and (config.trainer.devices is None or config.trainer.devices != 1)
        and config.trainer.accelerator != "cpu"
    ):
        logger.warning("Anomalib currently does not support multi-gpu training. Setting devices to 1.")
        config.trainer.devices = 1

def update_input_size_config(config: Union[DictConfig, ListConfig]) -> Union[DictConfig, ListConfig]:
    """Update config with image size as tuple, effective input size and tiling stride.

    Convert integer image size parameters into tuples, calculate the effective input size based on image size
    and crop size, and set tiling stride if undefined.

    Args:
        config (Union[DictConfig, ListConfig]): Configurable parameters object

    Returns:
        Union[DictConfig, ListConfig]: Configurable parameters with updated values
    """
    # handle image size
    if isinstance(config.dataset.image_size, int):
        config.dataset.image_size = (config.dataset.image_size,) * 2

    config.model.input_size = config.dataset.image_size

    if "tiling" in config.dataset.keys() and config.dataset.tiling.apply:
        if isinstance(config.dataset.tiling.tile_size, int):
            config.dataset.tiling.tile_size = (config.dataset.tiling.tile_size,) * 2
        if config.dataset.tiling.stride is None:
            config.dataset.tiling.stride = config.dataset.tiling.tile_size

    return config
def get_configurable_parameters(
    model_name: Optional[str] = None,
    config_path: Optional[Union[Path, str]] = None,
    weight_file: Optional[str] = None,
    config_filename: Optional[str] = "config",
    config_file_extension: Optional[str] = "yaml",
) -> Union[DictConfig, ListConfig]:
    """Get configurable parameters.

    Args:
        model_name: Optional[str]:  (Default value = None)
        config_path: Optional[Union[Path, str]]:  (Default value = None)
        weight_file: Path to the weight file
        config_filename: Optional[str]:  (Default value = "config")
        config_file_extension: Optional[str]:  (Default value = "yaml")

    Returns:
        Union[DictConfig, ListConfig]: Configurable parameters in DictConfig object.
    """
    if model_name is None and config_path is None:
        raise ValueError(
            "Both model_name and model config path cannot be None! "
            "Please provide a model name or path to a config file!"
        )

    if config_path is None:
        config_path = Path(f"anomalib/models/{model_name}/{config_filename}.{config_file_extension}")

    config = OmegaConf.load(config_path)

    # Dataset Configs
    if "format" not in config.dataset.keys():
        config.dataset.format = "mvtec"

    config = update_input_size_config(config)

    # Project Configs
    project_path = Path(config.project.path) / config.model.name / config.dataset.name
    if config.dataset.format.lower() in ("btech", "mvtec"):
        project_path = project_path / config.dataset.category

    (project_path / "weights").mkdir(parents=True, exist_ok=True)
    (project_path / "images").mkdir(parents=True, exist_ok=True)
    config.project.path = str(project_path)
    # loggers should write to results/model/dataset/category/ folder
    config.trainer.default_root_dir = str(project_path)

    if weight_file:
        config.trainer.resume_from_checkpoint = weight_file

    config = _update_nncf_config(config)

    # thresholding
    if "metrics" in config.keys():
        if "pixel_default" not in config.metrics.threshold.keys():
            config.metrics.threshold.pixel_default = config.metrics.threshold.image_default

    return config
