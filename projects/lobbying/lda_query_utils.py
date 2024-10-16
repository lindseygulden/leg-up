"""Utilities for querying lobbying disclosure documents (called by lda_query.py)"""

from math import floor
from pathlib import PosixPath
from typing import List, Union

import pandas as pd

from utils.io import yaml_to_dict


def assemble_organization_search_string(term_list_path: Union[str, PosixPath]):
    """joins terms in term lists with an OR and returns as a single string for use in get query"""
    term_list_dict = yaml_to_dict(term_list_path)
    return term_list_dict["organization_list"]


def assemble_issue_search_string(
    term_list_path: Union[str, PosixPath],
    search_string_chunk_size: int = 15,
    max_search_string_length: int = 3000,
):
    """joins terms in term lists with an OR and returns as a single string for use in get query"""

    term_list_dict = yaml_to_dict(term_list_path)
    term_list = term_list_dict["search_term_list"]
    search_string = "OR".join(term_list)

    if len(search_string) < max_search_string_length:
        return [search_string]

    search_string_list = []
    for i in range(0, len(term_list), search_string_chunk_size):
        search_string_list.append(
            "OR".join(term_list[i : i + search_string_chunk_size])
        )
    return search_string_list


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
