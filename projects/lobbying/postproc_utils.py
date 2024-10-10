"""utility functions to compile and postprocess files read in from LDA API"""

import datetime as dt
import re
from itertools import compress
from typing import Dict, List, Tuple, Union

import pandas as pd
from cleanco import basename

from utils.io import yaml_to_dict


def terms_present(phrase, term_list, find_any=True):
    """utility function to see if terms in terms_list are present in a given phrase
    Args:
        phrase: phrase to be searched
        term_list: list of strings to be searched for within phrase
        find_any: boolean -- true if function should return true if any of the terms are in the phrase;
            false if the function should only return true if all terms are present
    Returns:
        int: 1 if one or more of the terms are present in phrase, 0 otherwise
    """

    if not isinstance(phrase, str):
        raise TypeError("phrase must be a string")
    if not isinstance(term_list, list):
        raise TypeError(
            "term_list must be a list of strings (even if it is a list of length 1)"
        )
    if phrase is None:
        return 0
    n_present = 0

    for term in term_list:
        if isinstance(term, list):
            if terms_present(phrase, term, find_any=False):
                n_present += 1
        elif term.lower() in phrase.lower():
            n_present += 1
    if find_any is False:  # find all
        if len(term_list) == n_present:
            return 1
        return 0
    return n_present


def get_list_govt_entities(entity_endpoint: str, session: object):
    """Queries constants endpoint to get a standardized list of government entities"""
    govt_entities = session.get(entity_endpoint, timeout=60)
    entity_df = pd.DataFrame(govt_entities.json())
    entities = sorted([x.lower() for x in list(entity_df["name"])])
    return entities


def substitute(
    x: Union[str, List[str]],
    use_basename: bool = False,
    re_types: str = r"[^\w\s]",
    replace_str: str = "",
):
    """wrapper function for regular expression substitute funciton, linked with basename lib"""
    # use basename for company names

    if (not isinstance(x, str)) and (not isinstance(x, list)):
        return replace_str
    if isinstance(x, list):
        return [
            substitute(
                xx,
                use_basename=use_basename,
                re_types=re_types,
                replace_str=replace_str,
            )
            for xx in x
        ]

    if use_basename:
        return basename(re.sub(re_types, replace_str, x))
    # don't use basename for general strings
    return re.sub(re_types, replace_str, x)  # .rstrip().lstrip()


def parse_client_names(
    input_df, config, client_rename_col="client_rename"
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """bespoke function for parsing organization names from LDA queries
    Args:
        input_df: input pandas dataframe (read raw from files saved after api access)
        config: configuration dictionary containing details on how to handle names
        client_rename_column: name for new column in dataframe that is returned
    Returns:
        output_df: processed column with renamed clients and removed organziations
        client_name_rename_dict: dictionary used for renaming (contains original and new names)
    """
    output_df = input_df.copy(deep=True)

    starting_client_names = sorted(list(output_df.client_name.unique()))
    client_names = sorted(list(output_df.client_name.unique()))
    client_names = [
        substitute(company_name, use_basename=True) for company_name in client_names
    ]

    # take rightmost component of client name
    for term in config["take_terms_to_the_right_of_these_words"]:
        client_names = [x.split(term)[-1] for x in client_names]
    # take leftmost component of client name
    for term in config["take_terms_to_the_left_of_these_words"]:
        client_names = [x.split(term)[0] for x in client_names]
    # remove words
    for n in config["remove_these_phrases"]:
        client_names = [x.replace(n, "") for x in client_names]
    # get rid of double spaces
    client_names = [x.replace("  ", " ") for x in client_names]
    # trim spaces on ends of names
    client_names = [x.rstrip().lstrip() for x in client_names]

    # extract shorter, well-known names from longer names
    for co in config["use_these_name_subsets_for_organiztions"]:
        client_names = [co if co in x else x for x in client_names]

    # bespoke replacements and handling of mergers
    for key, value in config["replace_names_on_left_with_names_on_right"].items():
        client_names = [x.replace(key, value) for x in client_names]

    # make a renaming dictionary
    client_name_rename_dict = dict(zip(starting_client_names, client_names))

    # add the 'remove' companies to the rename dictionary
    remove_companies = config["remove_companies_containing_these_terms"]
    for original_name, new_name in client_name_rename_dict.items():
        if terms_present(
            new_name,
            remove_companies,
            find_any=True,
        ):
            client_name_rename_dict[original_name] = "remove"

    # make new column with renames
    output_df[client_rename_col] = [
        client_name_rename_dict[x] for x in output_df.client_name
    ]
    output_df = output_df.loc[output_df[client_rename_col] != "remove"]

    return output_df, client_name_rename_dict


def get_smarties(
    row: Union[pd.Series, List[Union[bool, int]]], names: List[str]
) -> List[str]:
    """opposite of 'get dummies': returns list of all bool columns for which value is True"""
    if isinstance(row, pd.Series):
        if len(row) == 0:
            return []

        return list(compress(names, row[names].values.tolist()))
    if isinstance(row, list):
        return list(compress(names, row))
    raise TypeError("get_smarties argument 'row' must be a Pandas Series or a list")


def get_latest_filings(
    df: pd.DataFrame, groupby_cols: List[str], date_col="filing_dt_posted"
):
    """get only the latest filing for a given lobbying firm, client, and quarter"""
    if df[date_col].dtype == str:
        df[date_col] = [dt.datetime.fromisoformat(d) for d in df[date_col]]

    df.sort_values(by=date_col, ascending=False, inplace=True)

    df = df.groupby(groupby_cols).first().reset_index()
    return df


def invert_sector_dict(sectors_path) -> Dict[str, str]:
    """reads in the sector assignment yaml to dict; inverts dict s.t. each company is a key"""
    sector_assignments = yaml_to_dict(sectors_path)

    all_companies = []
    for _, value in sector_assignments.items():
        all_companies = all_companies + value
    # print(all_companies)

    company_sector_dict = {}
    for k, vv in sector_assignments.items():
        for v in vv:
            company_sector_dict = company_sector_dict | {v: k}

    return company_sector_dict
