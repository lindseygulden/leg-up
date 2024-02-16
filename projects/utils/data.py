import datetime as dt
import pandas as pd
import numpy as np
from typing import Literal


def convert_to_datetime(x, fmt: str = "%Y-%m-%d"):
    """Takes common date/time representations and returns them as a datetime object"""
    if isinstance(x, str):
        return dt.datetime.strptime(x, fmt)
    if isinstance(x, pd.Timestamp):
        return x.to_pydatetime()
    if isinstance(x, np.datetime64):
        return pd.Timestamp(x).to_pydatetime()
    if isinstance(x, dt.date):
        return dt.datetime.combine(x, dt.time(0, 0))
    if isinstance(x, dt.datetime):
        return x
    return None  # TODO raise error


def zero_pad(
    x, front_or_back: Literal["front", "back"] = "front", max_string_length: int = 2
):
    """convert a number or its string representation to a zero-padded string of length max_string_length"""
    if len(str(x)) > max_string_length:
        raise ValueError(f"{x} has more than {max_string_length} digits")
    if len(str(x)) == max_string_length:
        return str(x)
    if front_or_back == "front":
        return "0" * (max_string_length - len(str(x))) + str(x)
    return str(x) + "0" * (max_string_length - len(str(x)))
