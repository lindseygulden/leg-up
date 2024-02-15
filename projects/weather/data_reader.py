""" Class for reading data from a remote source, processing it, and writing it locally"""

from projects.utils.io import yaml_to_dict
from pathlib import Path, PosixPath
from abc import ABC, abstractmethod
from typing import Union
import logging


# Create and configure logger
logging.basicConfig()

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)


class DataReader(ABC):

    subclasses = {}

    @classmethod
    def register_subclass(cls, data_source):
        def decorator(subclass):
            cls.subclasses[data_source] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, data_source, params):
        if data_source not in cls.subclasses:
            raise ValueError("Bad data source {}".format(data_source))

        return cls.subclasses[data_source](params)

    def __init__(self, config_path: Union[str, PosixPath]):
        self.config = yaml_to_dict(config_path)
        self.entry_point = self.config["entry_point"]
        self.api_key = self.config["api_key"]
        self.output_directory = self.config["output_directory"]
        self.output_file_suffix = self.config["output_file_suffix"]

    @abstractmethod
    def prep_query(self):
        pass

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def postprocess_data(self):
        pass

    def write_data(self):
        # TODO add checks for presence of directory
        for key, df in self.data_dict.items():
            df.to_csv(
                Path(self.output_directory)
                / Path(f"{str(key)}_{self.output_file_suffix}.csv")
            )
