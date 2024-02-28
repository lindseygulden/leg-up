""" Class for reading data from a remote source, processing it, and writing it locally"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path, PosixPath
from typing import Union

import pandas as pd

from utils.io import yaml_to_dict

logging.basicConfig(level=logging.INFO)


class DataReader(ABC):
    """Parent class for various remote-source data readers"""

    subclasses = {}

    @classmethod
    def register_subclass(cls, data_source: str):
        """creates decorator that automatically registers subclasses
        To use, decorate the child class definition with @DataReader.register_subclass("[name-of-child-class]")
        """

        def decorator(subclass):
            cls.subclasses[data_source] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, data_source: str, params: dict):
        """Creates a new child class using the abbreviation for the data source (e.g., "wwo")"""
        if data_source not in cls.subclasses:
            raise ValueError(f"Bad data source {data_source}")

        return cls.subclasses[data_source](params)

    def __init__(self, config_path: Union[str, PosixPath]):
        self.config = yaml_to_dict(config_path)
        self.entry_point = self.config["entry_point"]
        self.api_key = self.config["api_key"]
        self.output_file_suffix = self.config["output_file_suffix"]
        self.data_renaming_dict = (
            {}
        )  # dictionary that holds optional variable-renaming values
        self.data_dict = {}  # dictionary for holding postprocessed dataframes

    @abstractmethod
    def prep_query(self):
        """Abstract method for preparing an API query, as needed based on data source"""

    @abstractmethod
    def get_data(self):
        """Abstract method for querying API"""

    @abstractmethod
    def postprocess_data(self):
        """Abstract method for postprocessing of data obtained by API query (e.g., formatting into dataframe(s))"""

    def write_data(self, output_directory: Union[str, PosixPath]):
        """Writes dataframe(s) to specified path(s)"""
        # TODO add checks for presence of directory
        logging.info("Writing data to %s", output_directory)
        for key, df in self.data_dict.items():
            fixed_key = key.replace(",", "_").replace(".", "pt")
            df.to_csv(
                Path(output_directory)
                / Path(f"{str(fixed_key)}_{self.output_file_suffix}.csv")
            )

    def _rename_columns(self, df: pd.DataFrame):
        """Uses renaming dictionary to rename dataframe columns
        Args:
            df: dataframe whose columns will be renamed using self.data_renaming_dict
        Returns:
            None
        """
        vartypes = list(self.data_renaming_dict.keys())

        for vartype in vartypes:
            df.rename(columns=self.data_renaming_dict[vartype], inplace=True)
        return df
