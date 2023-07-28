"""Test get_local_weather_history module"""
import datetime as dt

import pytest

from projects.weather.get_local_weather_history import (
    first_day_of_next_month,
    split_date_range,
    wwo_format,
)


def test_split_date_range():
    dt_start = dt.datetime(1215, 6, 15)
    dt_end = dt.datetime(1216, 1, 22)
    test_output = split_date_range(dt_start, dt_end)
    correct_output = [
        "15-JUN-1215",
        "01-JUL-1215",
        "01-AUG-1215",
        "01-SEP-1215",
        "01-OCT-1215",
        "01-NOV-1215",
        "01-DEC-1215",
        "01-JAN-1216",
        "22-JAN-1216",
    ]
    assert test_output == correct_output


def test_first_day_of_next_month():
    # test that exception is properly raised
    with pytest.raises(
        TypeError, match="Input argument some_date must be a python datetime object."
    ):
        first_day_of_next_month("2016-11-8")
    # test that function works correctly
    assert dt.datetime(1920, 9, 1) == first_day_of_next_month(dt.datetime(1920, 8, 18))


def test_wwo_format():
    # test that exception is properly raised
    with pytest.raises(
        TypeError, match="Input argument d must be a python datetime object."
    ):
        wwo_format("2022-6-24")
    # test that function works as expected
    assert wwo_format(dt.datetime(1973, 1, 22)) == "22-JAN-1973"
