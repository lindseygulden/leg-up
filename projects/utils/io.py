import yaml
from typing import Union
from pathlib import PosixPath
import logging

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

    if (str(yaml_filepath).split(".")[-1] != "yml") and (
        str(yaml_filepath).split(".")[-1] != "yaml"
    ):
        raise TypeError(
            "yaml_to_dict requires a yaml file as input (and we're assuming you named it with a .yml or .yaml extension)"
        )

    with open(yaml_filepath, "r") as file:
        try:
            dictionary = yaml.safe_load(file)
        except yaml.YAMLError as e:
            logging.exception(e)

    return dictionary
