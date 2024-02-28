"""Builds subclass for DataReader that queries World Weather Organization's Local Weather History API"""

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


@DataReader.register_subclass("wwo")
class WWODataReader(DataReader):
    """Reads data for specified dates, from specified locations, from World Weather Online API"""

    def __init__(self, config_path: Union[str, PosixPath]):
        # initialize parent class __init__
        super().__init__(config_path)
        self.source = "wwo"

        # initialize data dictionaries for holding raw data
        self.raw_data_dict = {}

        # assign start date, end date, locations, and the renaming-to-make-it-pythonic dictionary
        end_date = self.config["start_date"]
        if "end_date" in self.config:
            end_date = self.config["end_date"]
        self.date_range = [
            convert_to_datetime(self.config["start_date"]),
            convert_to_datetime(end_date),
        ]
        self.data_chunks = (
            {}
        )  # initialize value: used to break up date ranges that are longer than 1 mo
        self.locations = self.config["locations"]
        self.data_renaming_dict = yaml_to_dict(self.config["data_config_file"])
        self.columns_to_keep = self.config["columns_to_keep"]

        # Assign optional configuration variables if present in config dictionary
        self.frequency = 24  # one observation each day as the default
        if "frequency" in self.config:
            self.frequency = self.config["frequency"]
        self.timeout_seconds = (
            30  # default amount of time to time out API query if no response
        )
        if "timeout_seconds" in self.config:
            self.timeout_seconds = self.config["timeout_seconds"]

        # import lists of variable names to keep
        for v in [
            "astronomy_variables",
            "hourly_variables",
            "daily_variables",
            "nearest_area_variables",
        ]:
            setattr(self, v, [])
            if v in self.config:
                setattr(self, v, self.config[v])

        # once we've assigned all variables in the config file, delete the attribute
        delattr(self, "config")

        logging.info(" Initialized a WWO weather-data reader.")

    def prep_query(self):
        """WWO only returns data packets of limited length. This method splits the date range into one-month chunks"""

        self._split_date_range()

    def get_data(
        self,
        include_loc: Literal["yes", "no"] = "yes",
        fmt: Literal["json", "xml"] = "json",
    ) -> Dict[str, requests.Response]:
        """Executes an API query to the WWO API according to config parameters for each location and each 'date chunk'
        Args:
            include_loc: str: 'yes' returns nearest weather point for which data are returned; 'no' otherwise
            fmt: string format in which you would like the response object; can either be 'json' or 'xml'
        Returns:
        """
        wwo_response_dict = {}
        # locations can be US/Canada ZIP codes, lat/lon strings, city names, etc.
        # For more info see https://www.worldweatheronline.com/weather-api/api/docs/historical-weather-api.aspx
        for loc in self.locations:
            logging.info(" Querying WWO API for location %s", loc)
            wwo_response_dict[loc] = {}

            # for each month-long subset of the full date range, query the api
            for start_date, end_date in self.data_chunks.items():
                try:
                    # pylint: disable=line-too-long
                    wwo_response_dict[loc][start_date] = requests.get(
                        f"{self.entry_point}?key={self.api_key}&date={start_date}&enddate={end_date}&q={loc}&tp={self.frequency}&format={fmt}&includelocation={include_loc}",
                        timeout=self.timeout_seconds,
                    )
                except requests.exceptions.Timeout:
                    logging.info(
                        " Timed out for %s between %s and %s", loc, start_date, end_date
                    )

        self.raw_data_dict = wwo_response_dict
        logging.info(" API querying completed.")

    def postprocess_data(self):
        """Extracts info from json response: each resulting df row contains data for one observation hour"""

        # flatten lists of renamed columns for extracting from raw data
        column_names = list(
            chain.from_iterable(
                [
                    list(self.data_renaming_dict[k].keys())
                    for k in self.data_renaming_dict.keys()
                ]
            )
        )

        # Extract weather and location data from raw response objects
        for loc in self.locations:
            logging.info(" Postprocessing obtained data for location %s.", loc)
            granular_data_list = []
            for _, response in self.raw_data_dict[loc].items():
                j = response.json()

                df = pd.DataFrame.from_dict(j)

                granular_df = self._expand_wwo_weather_data(df)

                granular_data_list.append(granular_df[column_names])

            combined_df = pd.concat(granular_data_list)

            combined_df = self._rename_columns(combined_df)

            combined_df["date"] = [
                convert_to_datetime(x).date() for x in combined_df.date
            ]
            if "datetime" in self.columns_to_keep:
                combined_df["datetime"] = [
                    dt.datetime.combine(
                        d,
                        dt.datetime.strptime(
                            zero_pad(t, max_string_length=4, front_or_back="front"),
                            "%H%M",
                        ).time(),
                    )
                    for d, t in zip(combined_df.date, combined_df.time)
                ]
            combined_df["location"] = loc
            self.data_dict[loc] = combined_df[self.columns_to_keep]

    def _extract_nearest_area_info(self, df: pd.DataFrame):
        """Pulls information about the weather station from which data are reported
        Args:
            df: a dataframe containing raw data extracted from the WWO API Response object
        Returns:
            flattened_dict: a single-level dictionary containing nearest-area information
        """
        flattened_dict = {}
        for key, value in df.loc["nearest_area",]["data"][0].items():
            if isinstance(value, list):
                flattened_dict[key] = value[0][list(value[0].keys())[0]]
            else:
                flattened_dict[key] = value

        return flattened_dict

    def _expand_wwo_weather_data(self, df: pd.DataFrame):
        """Pulls data from within different levels of existing dataframe to a single, most-granular level
        Args:
            df: a dataframe containing raw data extracted from the WWO API Response object
        Returns:
            a dataframe containing a row for each sub-daily observation
        """

        # For each entry, extract sub-daily weather information from weather-data list
        n_entries = len(df.loc["weather",]["data"])
        w_list = []
        for d in range(n_entries):
            w_list.append(df.loc["weather",]["data"][d])

        all_weather_df = pd.DataFrame(w_list)

        # Append nearest-area information to dataframe
        nearest_area_dict = self._extract_nearest_area_info(df)

        for key, value in nearest_area_dict.items():
            all_weather_df[key] = value

        # Append sub-daily weather information, each sub-daily interval it its own row
        h_list = (
            []
        )  # bucket for expanded dfs containing data extracted from each member of the 'hourly_data_list'

        for d in range(n_entries):
            # iterate through data for each subdaily interval ('hour') in the day's 'hourly' list of dictionary
            for hour in all_weather_df.loc[d, "hourly"]:
                hour["date"] = all_weather_df.loc[d, "date"]  # today's date
                # extract astronomy variables from list
                for v in list(self.data_renaming_dict["astronomy_variables"].keys()):
                    hour[v] = all_weather_df.loc[d, "astronomy"][0][v]
                # write the daily average variables to this hour's row
                for v in list(self.data_renaming_dict["daily_variables"].keys()):
                    hour[v] = all_weather_df.loc[d, v]
                # write the nearest-area information to this hour's row
                for v in list(self.data_renaming_dict["nearest_area_variables"].keys()):
                    hour[v] = all_weather_df.loc[d, v]

                h_list.append(pd.DataFrame(hour))

        return pd.concat(h_list)

    def _wwo_format(self, d: dt.datetime) -> str:
        """Takes a date-time object and returns a--rather unique--format appropriate for the WWO API
        Args:
            d: datetime object
        Returns:
            string representation of date-time in WWO format"""
        if not isinstance(d, dt.datetime):
            raise TypeError("Input argument d must be a python datetime object.")
        return dt.datetime.strftime(d, "%d-%b-%Y").upper()

    def _split_date_range(self) -> Dict[dt.datetime, dt.datetime]:
        """Builds list of WWO-formatted dates spanning dates b/w self.date_range[0] & self.date_range[1]
        N.B.: from WWO API documentation: the enddate parameter must have the same month and
        year as the start date parameter.
        """

        dt_list = []
        dt_now = self.date_range[0]
        while dt_now <= self.date_range[1]:
            next_mo = first_day_of_next_month(dt_now)
            dt_list.append(
                [
                    self._wwo_format(dt_now),
                    self._wwo_format(next_mo - dt.timedelta(days=1)),
                ]
            )
            dt_now = next_mo
        # replace end date for final date chunk with the correct end date
        dt_list[-1][1] = self._wwo_format(self.date_range[1])
        self.data_chunks = dict(dt_list)
