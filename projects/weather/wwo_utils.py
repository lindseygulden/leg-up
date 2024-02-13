import datetime as dt
from typing import List, Optional

from dateutil.relativedelta import relativedelta


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
