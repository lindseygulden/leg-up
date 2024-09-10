"""functions to postprocess files read in from LDA API"""

import requests
import pandas as pd
from flatten_json import flatten
from utils.io import yaml_to_dict
import numpy as np
from typing import List, Tuple, Dict, Union
from itertools import compress
import pandas as pd
import re
import datetime as dt
from cleanco import basename
from json import dumps
from typing import List, Tuple, Dict, Union
from pathlib import PosixPath
import pandas as pd
import re
from cleanco import basename
import click


def invert_sector_dict(sectors_path: Union[str, PosixPath]) -> Dict[str, str]:
    """reads in the sector assignment yaml to dict; inverts dict s.t. each company is a key"""
    sector_assignments = yaml_to_dict(sectors_path)

    all_companies = []
    for _, value in sector_assignments.items():
        all_companies = all_companies + value
    print(all_companies)

    company_sector_dict = {}
    for k, vv in sector_assignments.items():
        for v in vv:
            company_sector_dict = company_sector_dict | {v: k}

    return company_sector_dict


def apportion_filing_dollars_to_specific_activities(df: pd.DataFrame):
    """bespoke function: apportions total lobbying USD (on filing) to specific activities, using 2 methods"""
    df["activity_apportioned_usd"] = [
        usd / number_lobbying
        for usd, number_lobbying in zip(
            df.dollars_spent_lobbying, df.total_number_lobbying_activities
        )
    ]
    df["lobbyist_apportioned_usd"] = [
        (
            usd * (n_activity_lobbyists / total_lobbyists)
            if total_lobbyists > 0
            else activity_apportioned
        )
        for usd, n_activity_lobbyists, total_lobbyists, activity_apportioned in zip(
            df.dollars_spent_lobbying,
            df.n_lobbyists_for_activity,
            df.total_number_of_lobbyists_on_filing,
            df.activity_apportioned_usd,
        )
    ]
    return df


def get_list_govt_entities(entity_endpoint: str, session: object):
    """Queries constants endpoint to get a standardized list of government entities"""
    govt_entities = session.get(entity_endpoint, timeout=60)
    entity_df = pd.DataFrame(govt_entities.json())
    entities = sorted([x.lower() for x in list(entity_df["name"])])
    return entities


def api_authenticate(
    authentication_endpoint,
    username,
    apikey,
    username_key="username",
    password_key="password",
):
    """logs in to the api using provided authentication endpoint and credentials"""
    authenticated_session = requests.Session()

    authenticated_session.post(
        authentication_endpoint,
        json={
            username_key: username,
            password_key: apikey,
        },
        timeout=60,
    )

    return authenticated_session


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

    if phrase is None:
        return 0
    n_present = 0
    for term in term_list:
        if not isinstance(term, str):
            raise TypeError(" all terms in term_list must be strings")
        if term.lower() in phrase.lower():
            if find_any:
                return 1
            n_present += 1
    if n_present == len(term_list):
        return 1
    return 0


def substitute(
    x: str,
    use_basename: bool = False,
    re_types: str = r"[^\w\s]",
    replace_str: str = "",
):
    """wrapper function for regular expression substitute funciton, linked with basename lib"""
    # use basename for company names
    if not isinstance(x, str):
        x = ""
    if use_basename:
        return basename(re.sub(re_types, replace_str, x))
    # don't use basename for general strings
    return re.sub(re_types, replace_str, x).rstrip().lstrip()


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
    for x in client_name_rename_dict.keys():
        if terms_present(client_name_rename_dict[x], remove_companies, find_any=True):
            client_name_rename_dict[x] = "remove"

    # make new column with renames
    output_df[client_rename_col] = [
        client_name_rename_dict[x] for x in output_df.client_name
    ]
    output_df = output_df.loc[output_df[client_rename_col] != "remove"]

    return output_df, client_name_rename_dict


def get_smarties(
    row: Union[pd.Series, List[Union[bool, int]]], names: List[str]
) -> List[str]:
    if isinstance(row, pd.Series):
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


def get_search_terms(search_term_list: List[str]):
    """extracts search terms from list used for LDA API query and puts them into lists for use here"""

    terms = []
    for t in search_term_list:
        if "," in t:
            terms.append(t.replace('"', "").split(","))
        else:
            terms.append([t.replace('"', "")])

    terms = [[substitute(t) for t in tt] for tt in terms]

    return terms


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--output_filepath", type=click.Path(file_okay=True, dir_okay=False), required=True
)
def postproc(config: Union[str, PosixPath], output_filepath: Union[str, PosixPath]):
    config_info = yaml_to_dict(config)
    groupby_cols = [
        "filing_year",
        "filing_period",
        "client_id",
        "registrant_id",
        "activity_id",
    ]
    entities = get_list_govt_entities(
        config_info["entity_endpoint"],
        session=api_authenticate(
            config_info["authentication_endpoint"],
            config_info["lda_username"],
            config_info["lda_apikey"],
        ),
    )
    df_list = []
    rename_dict = {}
    for i in range(1, 50):
        api_results_df = pd.read_csv(
            f"/Volumes/Samsung_T5/data/lobbying/ccslaws/ccs_lda_filings_{i}.csv",
            index_col=[0],
            # parse_dates=["filing_dt_posted"],
            dtype={"filing_year": int},
            low_memory=False,
        )
        # remove unwanted filing types
        api_results_df = api_results_df.loc[
            [x[0] != "R" for x in api_results_df.filing_type]
        ]

        # parse company names, remove unwanted names, add to list
        df, this_rename_dict = parse_client_names(
            api_results_df, yaml_to_dict(config_info["organization_name_handling_path"])
        )

        # append the rename dictionary to the whole thing
        rename_dict = rename_dict | this_rename_dict
        # compress entities into a single string column and get rid of entity columns
        df["entities"] = df[entities].T.apply(
            lambda x: dumps(get_smarties(x, entities))
        )
        df.drop(entities, axis=1, inplace=True)
        df["clean_description"] = [
            substitute(d, use_basename=True) for d in df["description"]
        ]
        df["clean_client_general_description"] = [
            substitute(d, use_basename=False) for d in df["client_general_description"]
        ]
        df = get_latest_filings(df, groupby_cols)
        df_list.append(df)

    ccs_df = pd.concat(df_list)
    ccs_df = get_latest_filings(ccs_df, groupby_cols)

    ccs_df.to_csv(output_filepath)
    return ccs_df


if __name__ == "__main__":
    postproc()
