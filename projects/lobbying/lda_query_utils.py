"""Utilities for parsing lobbying disclosure documents"""

from math import floor
from typing import List

import pandas as pd


def assemble_issue_search_string(
    term_list_path: Union[str, PosixPath], law_list_path: Union[str, PosixPath]
):
    """joins terms in term lists with an OR and returns as a single string for use in get query"""
    term_list_dict = yaml_to_dict(term_list_path)
    law_list_dict = yaml_to_dict(law_list_path)
    search_string = "OR".join(
        term_list_dict["search_term_list"]  # +
        # law_list_dict["mostly_ccs_provisions"]
    )
    # law_list_dict["contains_ccs_provisions"]
    return search_string


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


def get_list_govt_entities(entity_endpoint: str, session: object):
    """Queries constants endpoint to get a standardized list of government entities"""
    govt_entities = session.get(entity_endpoint, timeout=60)
    entity_df = pd.DataFrame(govt_entities.json())
    entities = sorted([x.lower() for x in list(entity_df["name"])])
    return entities


def remove_unwanted_filing_types(discard_filing_types: List[str], df: pd.DataFrame):
    """removes filing types listed in discard_filing_types from dataframe df"""
    return df.loc[[not x in discard_filing_types for x in df.filing_type]].copy()
