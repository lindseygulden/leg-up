"""Builds command-line application pulls World Weather Organization's Local Weather History and puts it in csv"""

from pathlib import Path, PosixPath
from typing import Dict, List, Optional, Union, Literal


import pandas as pd
import requests
import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from abc import ABC, abstractmethod

from dateutil.relativedelta import relativedelta

from projects.utils.io import yaml_to_dict
from projects.weather.data_reader import DataReader
from projects.weather.wwo_utils import split_date_range, expand_weather_data
from projects.utils.data import convert_to_datetime


@DataReader.register_subclass("wwo")
class WWODataReader(DataReader):
    """Reads data from World Weather Online API"""

    def __init__(self, config_path: Union[str, PosixPath]):
        # initialize parent class __init__
        super().__init__(config_path)
        self.source = "wwo"

        # assign start date, end date, locations, and the renaming-to-make-it-pythonic dictionary
        self.start_date = convert_to_datetime(self.config["start_date"])
        self.end_date = convert_to_datetime(self.config["end_date"])
        self.locations = self.config["locations"]
        self.data_renaming_dict = yaml_to_dict(self.config["data_config_file"])

        # assign optional configuration variables
        for variable, value in {"frequency": 24, "timeout_seconds": 30}.items():
            setattr(self, variable, value)
            if variable in self.config:
                setattr(self, variable, self.config[variable])

        # import lists of variable names to keep
        for v in ["astronomy_variables", "hourly_variables", "daily_variables"]:
            setattr(self, v, [])
            if v in self.config:
                setattr(self, v, self.config[v])

        # once we've assigned all variables in the config file, delete the attribute
        delattr(self, "config")

    def prep_query(self):
        self.data_chunks = split_date_range(self.start_date, self.end_date)

    def get_data(
        self,
        include_loc: Literal["yes", "no"] = "yes",
        fmt="json",
    ) -> Dict[str, requests.Response]:
        """Executes an API query to the WWO API according to config parameters for each location and each 'date chunk'
        Args:
        Returns:
        """
        wwo_response_dict = {}
        for loc in self.locations:
            wwo_response_dict[loc] = {}
            for start_date, end_date in self.data_chunks.items():
                try:
                    # pylint: disable=line-too-long
                    wwo_response_dict[loc][start_date] = requests.get(
                        f"{self.entry_point}?key={self.api_key}&date={start_date}&enddate={end_date}&q={loc}&tp={self.frequency}&format={fmt}&includelocation={include_loc}",
                        timeout=self.timeout_seconds,
                    )
                except requests.exceptions.Timeout:
                    logging.info(
                        f"Timed out for {loc} between {start_date} and {end_date}"
                    )

        self.raw_data_dict = wwo_response_dict

    def postprocess_data(self):
        self.data_dict = {}
        for loc in self.locations:
            granular_data_list = []
            for _, response in self.raw_data_dict[loc].items():
                j = response.json()

                df = pd.DataFrame.from_dict(j)

                granular_df = expand_weather_data(df, self.data_renaming_dict)

                granular_data_list.append(
                    granular_df[
                        self.daily_variables
                        + self.hourly_variables
                        + self.astronomy_variables
                    ]
                )

        self.data_dict[loc] = pd.concat(granular_data_list)
