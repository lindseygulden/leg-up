import datetime as dt
import pandas as pd
import numpy as np


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
