"""Builds command-line application pulls World Weather Organization's Local Weather History and puts it in csv"""

from pathlib import Path, PosixPath
from typing import Dict, List, Optional, Union, Literal
import datetime as dt

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
from projects.utils.data import convert_to_datetime, zero_pad


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
        self.columns_to_keep = self.config["columns_to_keep"]

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

                daily_variables = list(
                    self.data_renaming_dict["daily_variables"].keys()
                )
                hourly_variables = list(
                    self.data_renaming_dict["hourly_variables"].keys()
                )
                astronomy_variables = list(
                    self.data_renaming_dict["astronomy_variables"].keys()
                )

                granular_data_list.append(
                    granular_df[
                        daily_variables + astronomy_variables + hourly_variables
                    ]
                )

        combined_df = pd.concat(granular_data_list)
        for vartype in ["daily_variables", "astronomy_variables", "hourly_variables"]:
            combined_df.rename(columns=self.data_renaming_dict[vartype], inplace=True)

        if "datetime" in self.columns_to_keep:
            combined_df["datetime"] = [
                dt.datetime.combine(
                    dt.datetime.strptime(d, "%Y-%m-%d").date(),
                    dt.datetime.strptime(
                        zero_pad(t, max_string_length=4, front_or_back="front"), "%H%M"
                    ).time(),
                )
                for d, t in zip(combined_df.date, combined_df.time)
            ]
        combined_df["location"] = loc
        self.data_dict[loc] = combined_df[self.columns_to_keep]
