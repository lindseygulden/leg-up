from projects.utils.io import yaml_to_dict

from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.info)


class DataReader(ABC):
    def __init__(self, config_path):
        self.config = yaml_to_dict(config_path)
        self.entry_point = self.config["entry_point"]
        self.api_key = self.config["api_key"]
        self.output_location = self.config["output_file"]

    @abstractmethod
    def prep_data(self):
        pass

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def postprocess_data(self):
        pass

    def write_data(self):
        pass
