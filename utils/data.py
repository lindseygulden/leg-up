"""Assorted functions for manipuling data, strings, etc."""

from itertools import compress
from typing import List, Literal, Union

import numpy as np
import pandas as pd


def rgb_tuple(red: int, green: int, blue: int, denominator=255):
    """given an integer RGB representation, returns a floating point representation (where R, G, & B fall on [0,1])
    Args:
        red: integer specifying red saturation
        green: integer specifying green saturation
        blue: integer specifying blue saturation
        denominator: scale on which the above r/g/b values are specified
    Returns:
        tuple with RGB values on a [0,1] range
    """
    if red < 0 or blue < 0 or green < 0:
        raise ValueError(
            f"input arguments r ({red}), g ({green}), and b ({blue}) must be >= 0"
        )
    if red > denominator or blue > denominator or green > denominator:
        raise ValueError(
            f"input arguments r, g, and b must be <= denominator (currently {denominator})"
        )
    if denominator <= 0:
        raise ValueError(f"denominator ({denominator}) must be >0")
    return (red / denominator, green / denominator, blue / denominator)


def get_smarties(
    row: Union[pd.Series, List[Union[bool, int]]], names: List[str]
) -> List[str]:
    """Inverse (sort of) of pd.get_dummies. Returns list column names corresponding to True values"""
    if isinstance(row, pd.Series):
        return list(compress(names, row[names].values.tolist()))
    if isinstance(row, list):
        return list(compress(names, row))
    raise TypeError("get_smarties argument 'row' must be a Pandas Series or a list")


def zero_pad(
    x, front_or_back: Literal["front", "back"] = "front", max_string_length: int = 5
):
    """convert a number or its string representation to a zero-padded string of length max_string_length
    Args:
        x: integer or string to be zero-padded
        front_or_back: append zeros before x ('front') or after x ('back')
        max_string_length: desired length of output string
    Returns:
        string of length max_string_length, zero-padded if original length of x < max_string_length
    """
    # TODO do error checking for data type of x
    if len(str(x)) > max_string_length:
        raise ValueError(f"{x} has more than {max_string_length} digits")
    if len(str(x)) == max_string_length:
        return str(x)
    if front_or_back == "front":
        return "0" * (max_string_length - len(str(x))) + str(x)
    return str(x) + "0" * (max_string_length - len(str(x)))


def contains_substring(s: str, text_list: List[str], match_case: bool = False) -> bool:
    """Returns true if text string contains any of the substrings in a list
    Args:
        s: string within which to search for substrings
        text_list: list of substrings
        match_case: set to True if you want it to match case when searching
    Returns:
        True if one of the text_list substrings is in string s, False otherwise
    """
    if not isinstance(s, str):
        raise TypeError(f"First argument must be string. {s} is not a string.")
    if not isinstance(text_list, list):
        raise TypeError("Second argument must be a list of strings (it is not a list)")
    for t in text_list:
        if not isinstance(t, str):
            raise TypeError(f"All list members must be strings. {t} is not a string.")
        if match_case:
            if t in s:
                return True
        else:
            if t.lower() in s.lower():
                return True
    return False


def replace_lots(x: str, replace_dict: dict = None):
    """Replaces a bunch of stuff in input string x; returns modified string
    Args:
        x: string on which to do the replacement operations
        replace_dict: optional input dictionary that contains the replacement instructions
    Returns:
        x: string modified with replacements
    """

    if replace_dict is None:
        replace_dict = {
            "/": "per",
            "&": "and",
            "(": "",
            ")": "",
            "-": "",
            " ": "_",
            "__": "_",
            ":": "to",
        }

    for k, v in replace_dict.items():
        x = x.replace(k, v)
    return x


def interpolate_xy_point(x, y, new_x):
    """
    Interpolate a single y value at new_x from x, y points.
    Args:
        x : list or array-like known x values
        y : list or array-like of known y values
        new_x : float or int with X location to interpolate to
    Returns:
        float of interpolated y value (NaN if outside bounds)
    """

    df = pd.DataFrame({"x": x, "y": y}).sort_values("x").dropna()
    if isinstance(new_x, list) | isinstance(new_x, pd.Series):
        new_x = np.mean(new_x)
    return float(
        np.interp(new_x, df["x"].values, df["y"].values, left=np.nan, right=np.nan)
    )
