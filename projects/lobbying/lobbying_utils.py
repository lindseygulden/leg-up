"""Utilities for parsing lobbying disclosure documents"""

from math import floor


def which_congress(which_year, based_on_year=True):
    """assigns a given year to the congress that was in session during that year"""
    if not based_on_year:
        raise NotImplementedError()

    return floor((which_year - 1787) / 2)


def terms_present(phrase, term_list):
    """utility function to see if terms in terms_list are present in a given phrase
    Args:
        phrase: phrase to be searched
        term_list: list of strings to be searched for within phrase
    Returns:
        int: 1 if one or more of the terms are present in phrase, 0 otherwise
    """
    if phrase is None:
        return 0
    for term in term_list:
        if str(term).lower() in str(phrase).lower():
            return 1
    return 0
