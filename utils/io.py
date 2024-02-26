""" Utility functions for input/output of data"""

import logging
from pathlib import PosixPath
from typing import Union

import yaml

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def yaml_to_dict(yaml_filepath: Union[str, PosixPath]) -> dict:
    """One-line function to read a yml config at a path into a dictionary
    Args:
        yaml_filepath: path to yaml file
    Returns:
        dictionary: contents of yaml file
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

    return dictionary
