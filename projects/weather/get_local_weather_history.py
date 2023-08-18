"""Builds command-line application pulls World Weather Organization's Local Weather History and puts it in csv"""
import datetime as dt
import io
import json
from pathlib import Path, PosixPath
from typing import List, Optional, Union

import click
import pandas as pd
import requests
import validators
import yaml
from dateutil.relativedelta import relativedelta
from schema import And, Optional, Schema, SchemaError, Use

from projects.utils.io import yaml_to_dict


def get_and_validate_config(config_file_path: Union[str, PosixPath]) -> dict:
    """Reads and validates API query configuration file; adds a boolean
    Args:
        config_file_path: string or path to configuration yaml file
    Returns:
        config: dict containing processed, validated information from yaml input file
    """
    # Read in configuration from YAML file
    config = yaml_to_dict(config_file_path)
    schema = Schema(
        {
            "locations": And(list, lambda x: all([isinstance(xx, str) for x in xx])),
            "api_key": And(str),
            Optional(
                "entry_point",
                default="https://api.worldweatheronline.com/premium/v1/past-weather.ashx",
            ): And(str, lambda x: validators.url(x)),
            "end_date": And(str, lambda x: len(x) == 11),
            "start_date": And(str, lambda x: len(x) == 11),
            Optional("timeout_seconds", default=30): And(int, lambda t: t > 0),
            Optional("frequency", default=1): And(
                lambda f: f in [1, 3, 6, 12, 24], int
            ),
            "daily_weather_variables": And(
                list,
                lambda x: all(
                    [
                        xx
                        in [
                            "date",
                            "maxtempC",
                            "maxtempF",
                            "mintempC",
                            "mintempF",
                            "avgtempC",
                            "avgtempF",
                            "totalSnow_cm",
                            "sunHour",
                            "uvIndex",
                        ]
                        for xx in x
                    ]
                ),
            ),
            "astronomy_variables": And(
                list,
                lambda x: all(
                    [
                        xx
                        in [
                            "sunrise",
                            "sunset",
                            "moonrise",
                            "moonset",
                            "moon_phase",
                            "moon_illumination",
                        ]
                        for xx in x
                    ]
                ),
            ),
            "hourly_variables": And(
                list,
                lambda x: all(
                    [
                        xx
                        in [
                            "time",
                            "tempC",
                            "tempF",
                            "windspeedMiles",
                            "windspeedKmph",
                            "winddirDegree",
                            "winddir16Point",
                            "weatherCode",
                            "weatherDesc",
                            "precipMM",
                            "precipInches",
                            "humidity",
                            "visibility",
                            "visibilityMiles",
                            "pressure",
                            "pressureInches",
                            "cloudcover",
                            "HeatIndexC",
                            "HeatIndexF",
                            "DewPointC",
                            "DewPointF",
                            "WindChillC",
                            "WindChillF",
                            "WindGustMiles",
                            "WindGustKmph",
                            "FeelsLikeC",
                            "FeelsLikeF",
                            "uvIndex",
                        ]
                        for xx in x
                    ]
                ),
            ),
        }
    )
    schema.validate(config)
    config["extract_weather_desc"] = False
    if "weatherDesc" in config["hourly_variables"]:
        config["hourly_variables"] = [
            x for x in config["hourly_variables"] if x != "weatherDesc"
        ]
        config["extract_weather_desc"] = True

    return config


def wwo_format(d: dt.datetime) -> str:
    """Takes a date-time object and returns a--rather unique--format appropriate for the WWO API"""
    if not isinstance(d, dt.datetime):
        raise TypeError("Input argument d must be a python datetime object.")
    return dt.datetime.strftime(d, "%d-%b-%Y").upper()


def first_day_of_next_month(some_date: dt.datetime) -> dt.datetime:
    """Returns a datetime specifying the first day of the next month relative to some_date
    Args:
        some_date: a datetime object for any date in a month
    Returns:
        start_of_next_month: a datetime object for the first day of the month immediately following that of some_date
    """
    if not isinstance(some_date, dt.datetime):
        raise TypeError("Input argument some_date must be a python datetime object.")

    one_month_later = some_date + relativedelta(months=1)
    start_of_next_month = one_month_later - relativedelta(days=one_month_later.day - 1)
    return start_of_next_month


def split_date_range(
    dt_start: dt.datetime, dt_end: Optional[dt.datetime]
) -> List[dt.datetime]:
    """Builds list of WWO-formatted dates spanning the dates between dt_start and dt_end
    N.B.: from WWO API documentation: the enddate parameter must have the same month and
    year as the start date parameter.
    """
    if dt_end is not None:
        dt_list = []
        dt_now = dt_start
        while dt_now < dt_end:
            dt_list.append(wwo_format(dt_now))
            dt_now = first_day_of_next_month(dt_now)
        dt_list.append(wwo_format(dt_end))
        return dt_list
    return [wwo_format(dt_start)] * 2


# pylint: disable=line-too-long
def query_wwo_api(
    api_entry_point: str,
    api_key: str,
    loc: str,
    hrs: int,
    start_date: str,
    end_date: str,
    timeout_seconds: int,
    include_loc="yes",
    fmt="json",
) -> requests.Response:
    """Executes an API query to the WWO API according to config parameters
    Args:
    Returns:
    """
    try:
        wwo_response = requests.get(
            f"{api_entry_point}?key={api_key}&date={start_date}&enddate={end_date}&q={loc}&tp={hrs}&format={fmt}&includelocation={include_loc}",
            timeout=timeout_seconds,
        )
    except requests.exceptions.Timeout:
        print(f"Timed out for {loc} between {start_date} and {end_date}")

    return wwo_response


def write_json_to_file(
    api_response: requests.Response, filepath: Union[str, PosixPath]
) -> None:
    """Write the response from the api to a json file"""
    with open(filepath, "wb") as file:
        for chunk in api_response.iter_content(chunk_size=128):
            file.write(chunk)

    return


def extract_data(
    response: requests.Response,
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
@click.argument("config_yml", type=click.File("rb"), required=True)
@click.argument("output_path", type=click.Path(exists=True), required=True)
def get_weather(config_yml: str, output_path: str):
    """Uses command-line arguments to set up a call to the WWO API: assembles data in single csv"""
    config = get_config(config_yml)

    # Get list of locations
    locations = config["locations"]

    # Convert dates to list that is usable by the API
    query_start_date = dt.datetime.strptime(config["start_date"].title(), "%d-%b-%Y")
    query_end_date = dt.datetime.strptime(config["end_date"].title(), "%d-%b-%Y")
    dt_list = split_date_range(query_start_date, query_end_date)

    for loc in locations:
        # for every subset of dates
        for i, dt_start in enumerate(dt_list[:-2]):
            dt_end = dt_list[i + 1]
            request_response = query_wwo_api(
                config["api_entry_point"],
                config["api_key"],
                loc,
                config["hours"],
                config["timeout_seconds"],
                dt_start,
                dt_end,
            )

            date_chunks.append(
                pd.read_json(io.StringIO(request_response.content.decode("utf-8")))
            )
            write_json_to_file(
                request_response,
                Path(f"{output_path}/raw_json/{loc}_{dt_start}_to_{dt_end}.json"),
            )


if __name__ == "__main__":
    get_weather()
