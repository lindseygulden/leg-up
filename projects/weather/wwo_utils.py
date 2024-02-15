import datetime as dt
import pandas as pd
from typing import List, Optional, Dict

from dateutil.relativedelta import relativedelta
from projects.utils.io import yaml_to_dict


def expand_weather_data(df: pd.DataFrame, data_dict):
    """Pulls data from within different levels of existing dataframe to a single, most-granular level"""
    n_entries = len(df.loc["weather",]["data"])
    w_list = []
    for d in range(n_entries):
        w_list.append(df.loc["weather",]["data"][d])

    all_weather_df = pd.DataFrame(w_list)

    h_list = (
        []
    )  # bucket for expanded dfs containing data extracted from each member of the 'hourly_data_list'
    for d in range(n_entries):
        hourly_data_list = all_weather_df.loc[d, "hourly"]
        astronomy_data_list = all_weather_df.loc[d, "astronomy"][0]
        today_date = all_weather_df.loc[d, "date"]
        for hour in hourly_data_list:
            hour["date"] = today_date
            # add daily variables to hourly variables
            for v in list(data_dict["astronomy_variables"].keys()):
                hour[v] = astronomy_data_list[v]
            for v in list(data_dict["daily_variables"].keys()):
                hour[v] = all_weather_df.loc[d, v]
            h_list.append(hour)
    return pd.DataFrame(h_list)


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
) -> Dict[dt.datetime, dt.datetime]:
    """Builds list of WWO-formatted dates spanning the dates between dt_start and dt_end
    N.B.: from WWO API documentation: the enddate parameter must have the same month and
    year as the start date parameter.
    """
    print(dt_start)
    print(isinstance(dt_start, dt.datetime))
    if dt_end is not None:
        dt_list = []
        dt_now = dt_start
        while dt_now < dt_end:
            next_mo = first_day_of_next_month(dt_now)
            dt_list.append(
                [wwo_format(dt_now), wwo_format(next_mo - dt.timedelta(days=1))]
            )
            dt_now = next_mo
        # replace end date for final date chunk with the correct end date
        dt_list[-1][1] = wwo_format(dt_end)
        return dict(dt_list)
    return dict([wwo_format(dt_start)] * 2)
