import datetime as dt

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from utils.time import (
    convert_to_datetime,
    convert_to_decimal_year,
    first_day_of_next_month,
)


def test_first_day_of_next_month():
    # test that exception is properly raised
    with pytest.raises(
        TypeError, match="Input argument some_date must be a python datetime object."
    ):
        first_day_of_next_month("2016-11-8")
    # test that function works correctly
    assert dt.datetime(1920, 9, 1) == first_day_of_next_month(dt.datetime(1920, 8, 18))


def test_convert_to_datetime():
    test_cases = [
        ("1919-06-04", dt.datetime(1919, 6, 4)),
        (pd.Timestamp("1848-07-19"), pd.Timestamp("1848-07-19").to_pydatetime()),
        (np.datetime64("1973-01-22"), pd.Timestamp("1973-01-22").to_pydatetime()),
        (dt.date(1972, 6, 23), dt.datetime(1972, 6, 23)),
        (dt.datetime(1963, 6, 10), dt.datetime(1963, 6, 10)),
    ]

    for x, expected in test_cases:
        result = convert_to_datetime(x)
        assert (
            result == expected
        ), f"convert_to_datetime failed for input: {x}, expected: {expected}, got: {result}"

    # make sure problematic input raises a ValueError
    try:
        convert_to_datetime("RIP RBG")
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for crappy input to convert_to_datetime, but no exception was raised"
        )


def test_convert_to_decimal_year():
    assert 2020 == convert_to_decimal_year(dt.datetime(2020, 1, 1))
    assert_almost_equal(
        1947.0630136986301, convert_to_decimal_year(dt.date(1947, 1, 24)), decimal=7
    )
    assert_almost_equal(
        2020.0874316939892, convert_to_decimal_year(dt.date(2020, 2, 2)), decimal=7
    )
    try:
        convert_to_decimal_year("1944-01-24")
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for crappy input to convert_to_decimal_year, but no exception was raised"
        )
