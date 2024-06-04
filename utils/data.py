""" Assorted functions for manipuling data, strings, etc."""

from typing import Literal


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
