"""Builds command-line application pulls World Weather Organization's Local Weather History and puts it in csv"""

import datetime as dt
import json
import logging
from pathlib import Path, PosixPath
from typing import Dict, List, Optional, Union

import click
import pandas as pd
import requests


logging.basicConfig(level=logging.info)
from abc import ABC, abstractmethod

from dateutil.relativedelta import relativedelta

from projects.utils.io import yaml_to_dict
from projects.weather.data_reader import DataReader
from projects.weather.wwo_utils import split_date_range


class WWODataReader(DataReader):
    """Reads data from World Weather Online API"""

    def __init__(self, config_path):
        # initialize parent class __init__
        super().__init__(config_path)

        # assign start date, end date, locations, and the renaming-to-make-it-pythonic dictionary
        self.start_date = self.config["start_date"]
        self.end_date = self.config["end_date"]
        self.locations = self.config["locations"]
        self.data_renaming_dict = yaml_to_dict(self.config["data_config_file"])

        # assign optional configuration variables
        for variable, value in {"frequency": 24, "timeout_seconds": 30}:
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

    # pylint: disable=line-too-long
    def get_data(
        self,
        include_loc="yes",
        fmt="json",
    ) -> Dict[str, requests.Response]:
        """Executes an API query to the WWO API according to config parameters
        Args:
        Returns:
        """
        wwo_response_dict = {}
        for loc in self.locations:
            try:
                wwo_response_dict[loc] = requests.get(
                    f"{self.entry_point}?key={self.api_key}&date={self.start_date}&enddate={self.end_date}&q={loc}&tp={self.frequency}&format={fmt}&includelocation={include_loc}",
                    timeout=self.timeout_seconds,
                )
            except requests.exceptions.Timeout:
                logging.info(
                    f"Timed out for {loc} between {self.start_date} and {self.end_date}"
                )

        self.raw_data_dict = wwo_response_dict


def write_json_to_file(
    api_response: requests.Response, filepath: Union[str, PosixPath]
) -> None:
    """Write the response from the api to a json file"""
    with open(filepath, "wb") as file:
        for chunk in api_response.iter_content(chunk_size=128):
            file.write(chunk)

    return


def extract_data(
    df: pd.DataFrame,
    daily_weather_variables: list,
    astronomy_variables: list,
    hourly_variables: list,
    extract_weather_desc: bool,
) -> pd.DataFrame:
    """Extracts desired data from returned response object and assembles it into a dataframe
    Args:
        response: a requests Response object containing formatted weather data
        daily_weather_variables: a list of strings of weather variables desired in output
        astronomy_variables: a list of strings of astronomy variables desired in output
        hourly_variables: a list of strings of hourly weather variables desired in output
        extract_weather_desc: boolean indicating if user wants to extract text weather description
    Return:
        dataframe containing extracted weather from response object
    """

    weather = json.loads(response.content.decode("utf-8"))["data"]["weather"]

    weather_list = []

    for i in range(len(weather)):
        weather_dict = {k: weather[i][k] for k in tuple(daily_weather_variables)}
        weather_dict.update(
            {k: weather[i]["astronomy"][0][k] for k in tuple(astronomy_variables)}
        )
        for h in range(len(weather[i]["hourly"])):
            weather_dict.update(
                {k: weather[i]["hourly"][h][k] for k in tuple(hourly_variables)}
            )
            if extract_weather_desc:
                weather_dict.update(
                    {"weatherDesc": weather[i]["hourly"][h]["weatherDesc"][0]["value"]}
                )
            weather_list.append(weather_dict.copy())

    return pd.DataFrame(weather_list)


@click.command()
@click.argument("config_yml", type=click.Path(), required=True)
@click.argument("output_path", type=click.Path(exists=True), required=True)
def get_weather(config_yml: str, output_path: str):
    """Uses command-line arguments to set up a call to the WWO API: assembles data in single csv"""
    config = yaml_to_dict(config_yml)
    print(config)
    # Get list of locations
    locations = config["locations"]

    # Convert dates to list that is usable by the API
    query_start_date = dt.datetime.strptime(config["start_date"].title(), "%d-%b-%Y")
    query_end_date = dt.datetime.strptime(config["end_date"].title(), "%d-%b-%Y")
    dt_list = split_date_range(query_start_date, query_end_date)

    date_chunks = []

    for loc in locations:
        # for every subset of dates
        for i, dt_start in enumerate(dt_list[:-2]):
            dt_end = dt_list[i + 1]
            request_response = query_wwo_api(
                config["entry_point"],
                config["api_key"],
                loc,
                config["frequency"],
                dt_start,
                dt_end,
                config["timeout_seconds"],
            )

            df = pd.DataFrame.from_dict(request_response.json())


if __name__ == "__main__":
    get_weather()
