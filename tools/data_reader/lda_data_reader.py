
"""Builds subclass for DataReader that queries lobbying disclosure act API"""

import datetime as dt
import logging
from itertools import chain
from pathlib import PosixPath
from typing import Dict, Literal, Union

import pandas as pd
import requests

from tools.data_reader.data_reader import DataReader
from utils.data import zero_pad
from utils.io import yaml_to_dict
from utils.time import convert_to_datetime, first_day_of_next_month

logging.basicConfig(level=logging.INFO)


@DataReader.register_subclass("lda")
class LDADataReader(DataReader):
    """Reads lobbying disclosure data from LDA API"""

    def __init__(self, config_path: Union[str, PosixPath]):
        # initialize parent class __init__
        super().__init__(config_path)
        self.source = "lda"



    def prep_query(self):
        """Abstract method for preparing an API query, as needed based on data source"""

 
    def get_data(self):
        """Abstract method for querying API"""

    
    def postprocess_data(self):
        """Abstract method for postprocessing of data obtained by API query (e.g., formatting into dataframe(s))"""

    def write_data(self, output_directory: Union[str, PosixPath]):