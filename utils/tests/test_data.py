import pytest

from utils.data import zero_pad


def test_zero_pad():
    test_cases = [
        ((7, "front", 3), "007"),
        ((7, "back", 6), "700000"),
        ((9944, "front", 4), "9944"),
        ((9944, "back", 4), "9944"),
    ]

    # test good input
    for args, expected in test_cases:
        result = zero_pad(*args)
        assert (
            result == expected
        ), f"zero_pad fails with arguments: {args}, expected: {expected}, got: {result}"

    # test bad input
    try:
        zero_pad(9944, "front", 3)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for crappy input to zero_pad, but no exception was raised"
        )
