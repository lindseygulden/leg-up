"""Vendored subset of `utils.io` — YAML reader with env-var placeholder expansion."""
import logging
import os
from pathlib import PosixPath
from typing import Union

import yaml

logger = logging.getLogger(__name__)


def _expand_env_placeholders(value):
    """Recursively expand ${VAR} and $VAR environment variables in every
    string nested inside `value`. Non-strings are returned unchanged. Used by
    `yaml_to_dict` so configs can reference per-user paths (e.g.
    ``${CCS_GHG_DEV_DIR}``) that are set by each project's `paths.py`."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env_placeholders(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env_placeholders(v) for k, v in value.items()}
    return value


def yaml_to_dict(yaml_filepath: Union[str, PosixPath]) -> dict:
    """One-line function to read a yml config at a path into a dictionary.
    Any ``${VAR}`` or ``$VAR`` placeholders in string values are expanded
    from the current environment, so configs can be portable across users.

    Args:
        yaml_filepath: path to yaml file
    Returns:
        dictionary: contents of yaml file, with env-var placeholders expanded
    """

    if (str(yaml_filepath).split(".", maxsplit=1)[-1] != "yml") and (
        str(yaml_filepath).split(".", maxsplit=1)[-1] != "yaml"
    ):
        raise TypeError(
            "yaml_to_dict requires a yaml file as input (& we're assuming you named it with a .yml or .yaml extension)"
        )

    with open(yaml_filepath, "r", encoding="utf-8") as file:
        try:
            dictionary = yaml.safe_load(file)
        except yaml.YAMLError as excp:
            logging.exception(excp)

    return _expand_env_placeholders(dictionary)
