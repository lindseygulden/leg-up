# pylint: disable=too-many-locals
"""Command-line application to query Lobbying Disclosure Act API
formats results into multiple CSVs for filings and lobbyists"""
import datetime as dt
import logging
from math import ceil
from pathlib import Path, PosixPath
from time import sleep
from typing import Dict, List, Union

import click
import pandas as pd

from projects.lobbying.lda_query_utils import (
    assemble_issue_search_string,
    assemble_organization_search_string,
    get_list_govt_entities,
    which_congress,
)
from utils.api import api_authenticate
from utils.io import yaml_to_dict

logging.basicConfig(level=logging.INFO)


def parse_dollars_spent(income, expense):
    """Combine money paid to external lobbyists & expenses for firm lobbying on behalf of itself"""
    if (income is None) & (expense is None):
        return "income and expenses are zero", 0.0
    if income is None:
        return "corporation lobbying for itself", float(expense)
    if expense is None:
        return "hired lobbying firm", float(income)
    return "both income and expense > $0", float(income) + float(expense)


def initialize_row(govt_entities, result, filing_id):
    """initializes the dictionary for a given filing document"""
    # set up row dictionary using entity booleans
    initialize_row_dict = dict(
        zip(
            govt_entities,
            [0] * len(govt_entities),
        )
    )
    (
        initialize_row_dict["who_is_lobbying"],
        initialize_row_dict["dollars_spent_lobbying"],
    ) = parse_dollars_spent(result["income"], result["expenses"])

    # assign this filing document a unique identifier
    initialize_row_dict["filing_id"] = filing_id

    # affiliated organizations present
    initialize_row_dict["affiliated_organizations_present"] = False
    if len(result["affiliated_organizations"]) > 0:
        initialize_row_dict["affiliated_organizations_present"] = True
    # specify whether this document lists any conviction disclosures
    initialize_row_dict["convictions_present"] = False
    if len(result["conviction_disclosures"]) > 0:
        initialize_row_dict["convictions_present"] = True

    # result keys that require data transformation
    initialize_row_dict["filing_year"] = int(result["filing_year"])
    initialize_row_dict["filing_dt_posted"] = dt.datetime.fromisoformat(
        result["dt_posted"]
    )  # timestamps are in iso format
    # rest of result keys
    for result_key in [
        "filing_uuid",
        "url",
        "filing_period",
        "filing_type",
        "posted_by_name",
    ]:
        initialize_row_dict[result_key] = result[result_key]
    # registrant/lobbyist keys
    for registrant_key in ["id", "name", "contact_name"]:
        initialize_row_dict["registrant_" + registrant_key] = result["registrant"][
            registrant_key
        ]

    # compute total number of lobbying activities reported on this filing
    initialize_row_dict["total_number_lobbying_activities"] = len(
        result["lobbying_activities"]
    )

    # compute total number of lobbyists reported on this filing (note this is not total number of UNIQUE lobbyists)
    nlobbyists = 0
    for activity in result["lobbying_activities"]:
        nlobbyists += len(activity["lobbyists"])
    initialize_row_dict["total_number_of_lobbyists_on_filing"] = nlobbyists

    # get client info with client keys
    for client_key in [
        "id",
        "name",
        "general_description",
        "state",
        "country",
        "ppb_state",
        "ppb_country",
    ]:
        initialize_row_dict["client_" + client_key] = result["client"][client_key]

    return initialize_row_dict


def parse_lobbyists(lobbyists: dict, details: dict) -> List[dict]:
    """parses lobbyist information for a given lobbying activity"""
    lobbyist_list = []

    lobby_dict = {}
    # take information from the filing document details dictionary
    for details_key in [
        "registrant_name",
        "client_name",
        "general_issue_code",
        "description",
        "filing_period",
        "filing_year",
        "filing_id",
        "url",
        "filing_uuid",
        "activity_id",
    ]:
        lobby_dict[details_key] = details[details_key]

    # unpack lobbyists list
    for lobbyist in lobbyists:
        if not isinstance(lobbyist["lobbyist"]["last_name"], str):
            lobbyist["lobbyist"]["last_name"] = ""
        if not isinstance(lobbyist["lobbyist"]["first_name"], str):
            lobbyist["lobbyist"]["first_name"] = ""
        lobby_dict["name"] = (
            lobbyist["lobbyist"]["last_name"]
            + ", "
            + lobbyist["lobbyist"]["first_name"]
        )

        lobby_dict["covered_position"] = "None"
        if "covered_position" in lobbyist:
            lobby_dict["covered_position"] = lobbyist["covered_position"]
        lobby_dict["id"] = lobbyist["lobbyist"]["id"]
        lobbyist_list.append(lobby_dict.copy())

    return lobbyist_list


def streamline_description(replace_dict: Dict[str, str], x_df: pd.DataFrame):
    """goes through descriptions in dataframe and standardizes description components"""
    for key, value in replace_dict.items():
        x_df["description"] = [
            x.replace(key, value) if x is not None else "" for x in x_df.description
        ]
    return x_df


def consolidate_rows(
    replace_dict: Dict[str, str],
    row_list: List[dict],
):
    """Consolidates rows corresponding to filing activities into a single dataframe"""
    ccs_df = pd.DataFrame(row_list)

    # ccs_df = ccs_df.drop_duplicates(subset=ccs_df.columns.difference(["filing_id"]))

    ccs_unique_filing_ids = list(
        ccs_df.filing_id.unique()
    )  # keep a list of the non-duplicate filing ids

    # assign number of congress to dataframe
    ccs_df["which_congress"] = [which_congress(y) for y in ccs_df["filing_year"]]

    # somewhat streamline references to bills in lobbying descriptions

    ccs_df = streamline_description(replace_dict, ccs_df)

    return ccs_df, ccs_unique_filing_ids


def lda_get_query(
    session: object,
    endpoint: str,
    params: dict,
    return_value: str = "result",
    timeout=1000,
):
    """queries filing enpoint for a given page with a given set of get-request parameters"""
    # TODO make params an argument
    while True:
        f = session.get(
            endpoint,
            params=params,
            timeout=timeout,
        )
        try:
            if "results" in f.json():
                if return_value == "results":
                    query_result = f.json()["results"]
                elif return_value == "page_count":
                    query_result = ceil(f.json()["count"] / 25)
                break
            if "detail" in f.json():
                n_wait_seconds = int(f.json()["detail"].split(" ")[-2])
                logging.info(" Throttled: waiting for %s seconds.", str(n_wait_seconds))
                sleep(n_wait_seconds)
        except ValueError:
            logging.info(" Error: %s .", str(f.msg))
    return query_result


def write_out_subset(
    output_dir: Union[str, PosixPath],
    which_chunk: int,
    lobby_list: List[dict],
    row_list: List[dict],
    config_info: dict,
    output_filename_prefix: str,
    lobbyist_filename_prefix: str,
):
    """writies this subset's lobbying info and lobbying activity info to csvs"""
    ccs_df, ccs_unique_filing_ids = consolidate_rows(
        yaml_to_dict(config_info["bill_name_prefixes_replace_dict_path"]),
        row_list,
    )
    # write out CCS lobbying info for this subset ('chunk')
    ccs_df.to_csv(
        Path(output_dir) / Path(f"{output_filename_prefix}_{which_chunk}.csv")
    )
    # write out lobbyist data  for this subset ('chunk')
    lobbyists_df = pd.DataFrame(lobby_list)
    lobbyists_df.loc[lobbyists_df.filing_id.isin(ccs_unique_filing_ids)].to_csv(
        Path(output_dir) / Path(f"{lobbyist_filename_prefix}_{which_chunk}.csv")
    )


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--output_dir",
    type=click.Path(file_okay=False, dir_okay=True),
    required=False,
    default=".",
)
def query_lda(config: Union[str, PosixPath], output_dir: Union[str, PosixPath]):
    """Queries the Lobbying Disclosure Act lda API given terms"""
    config_info = yaml_to_dict(config)

    # get size of subset (number of pages before writing to a file...memory management)

    # login/authenticate
    authenticated_session = api_authenticate(
        config_info["authentication_endpoint"],
        config_info["lda_username"],
        config_info["lda_apikey"],
    )

    # get govt entity names
    govt_entities = get_list_govt_entities(
        config_info["entity_endpoint"], session=authenticated_session
    )
    search_string_list = []
    if config_info["query_param"] == "filing_specific_lobbying_issues":
        # set up get parameters dictionary
        search_string_list = assemble_issue_search_string(
            config_info["search_term_list_path"]
        )
    elif config_info["query_param"] == "client_name":
        search_string_list = assemble_organization_search_string(
            config_info["search_term_list_path"]
        )
    n_search_strings = len(search_string_list)
    logging.info(
        " --- Identified %s search string(s) for lobbying activities ---",
        str(n_search_strings),
    )

    # Set up years for query (only use a range of years if query_start_year and query_end_year are specified)
    query_years = [
        "all years"
    ]  # default if query start year and end year are not specified in the config file
    search_term_dict = yaml_to_dict(config_info["search_term_list_path"])
    if ("query_start_year" in search_term_dict) & (
        "query_end_year" in search_term_dict
    ):
        query_years = list(
            range(
                search_term_dict["query_start_year"],
                search_term_dict["query_end_year"] + 1,
            )
        )

    # initialize counting variables for subsets of queried pages ('chunks')
    which_chunk = 1
    if "chunk_start" in config_info:
        which_chunk = config_info["chunk_start"]
    # initialize unique id for filing documents (note that we record the filing_uuid from the API, too)
    filing_id = 0
    # loop through search strings
    for which_search_string, search_string in enumerate(search_string_list):

        for y in query_years:
            # all available years are searched if query_start_year and query_end_year are not specified in the config file
            if y == "all years":
                params = {
                    config_info["query_param"]: f"{search_string}",
                }
            else:
                logging.info("Querying for filing year %s", y)
                params = {
                    config_info["query_param"]: f"{search_string}",
                    "filing_year": str(y),
                }
            if config_info["query_param"] == "client_name":
                logging.info(" >>> ... Searching for %s", search_string)

            # each page contains 25 filings: use total number of filings to compute total number of pages
            n_pages = lda_get_query(
                authenticated_session,
                config_info["filings_endpoint"],
                params,
                return_value="page_count",
            )

            # compute number of file subsets ('chunks') for writing out and not overloading memory
            chunk_size = config_info["chunk_size"]
            n_chunks = ceil(n_pages / chunk_size)

            logging.info(
                " --- Preparing %s files for search string %s of % s ---",
                str(n_chunks),
                str(which_search_string + 1),
                str(n_search_strings),
            )

            row_list = []  # each row holds info for one lobbying activity
            lobby_list = []  # initialize holder for lobbyist info
            for page in range(1, n_pages + 1):
                # initialize holders for upcoming subset's information ('chunk')

                logging.info(" Querying page %s of %s pages", str(page), str(n_pages))

                # query api for this page of results
                results = lda_get_query(
                    authenticated_session,
                    config_info["filings_endpoint"],
                    params | {"page": page},
                    return_value="results",
                )

                # extract data from each filing form returned from query
                for result in results:
                    # TODO functionalize this
                    row_dict_base = initialize_row(govt_entities, result, filing_id)
                    activities = result["lobbying_activities"]

                    for activity_id, activity in enumerate(activities):
                        row_dict = row_dict_base.copy()

                        # which activity is this
                        row_dict["activity_id"] = activity_id
                        row_dict["general_issue_code"] = activity["general_issue_code"]
                        row_dict["description"] = activity["description"]
                        lobbyists_for_this_activity = parse_lobbyists(
                            activity["lobbyists"], row_dict
                        )
                        row_dict["n_lobbyists_for_activity"] = len(
                            lobbyists_for_this_activity
                        )
                        lobby_list = lobby_list + lobbyists_for_this_activity

                        # parse all government entitites lobbied, using boolean columns
                        for entity in activity["government_entities"]:
                            row_dict[entity["name"].lower()] = 1

                        row_list.append(row_dict.copy())

                    filing_id += 1  # each result
                # if we have the number of pages in a subset or we're at the end of the pages...
                if ((page % chunk_size) == 0) | (page == n_pages):
                    write_out_subset(
                        output_dir,
                        which_chunk,
                        lobby_list,
                        row_list,
                        config_info,
                        search_term_dict["output_filename_prefix"],
                        search_term_dict["lobbyist_filename_prefix"],
                    )
                    logging.info(" Writing chunk %s to CSV", str(which_chunk))

                    # increase chunk counter for next subset
                    which_chunk += 1
                    # re-initialize row and lobby lists for next subset/chunk:
                    row_list = []  # each row holds info for one lobbying activity
                    lobby_list = []  # initialize holder for lobbyist info

    logging.info(
        " ----- Finished writing all subsets ('chunks') to CSVs ----- ",
    )


if __name__ == "__main__":
    query_lda()
