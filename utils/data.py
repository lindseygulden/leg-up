""" Assorted functions for manipuling data, strings, etc."""

from typing import Literal


def zero_pad(
    x, front_or_back: Literal["front", "back"] = "front", max_string_length: int = 2
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
