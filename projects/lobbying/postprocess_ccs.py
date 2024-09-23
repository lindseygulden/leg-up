"""script to read in files read out by ccs and compile them into a single csv"""

import pandas as pd
from utils.io import yaml_to_dict

from typing import Union, Tuple, List
import pandas as pd

from json import dumps

from os import listdir
from os.path import isfile, join

from pathlib import PosixPath
import pandas as pd
import logging
import click
from utils.api import api_authenticate
from projects.lobbying.postproc_utils import (
    parse_client_names,
    get_smarties,
    get_list_govt_entities,
    substitute,
    get_latest_filings,
    invert_sector_dict,
    terms_present,
)

logging.basicConfig(level=logging.INFO)


def adjust_company_names(ccs_df: pd.DataFrame, config_info: dict):
    """Fills nans, replaces company names according to replacmeents specified in yaml"""
    ccs_df[config_info["clean_client_description_col"]] = ccs_df[
        config_info["clean_client_description_col"]
    ].fillna("")
    ccs_df[config_info["company_rename_col"]] = ccs_df[
        config_info["company_rename_col"]
    ].fillna("")

    replace_dict = yaml_to_dict(config_info["company_name_replacements"])

    ccs_df[config_info["company_rename_col"]] = [
        replace_dict[x] if x in list(replace_dict.keys()) else r
        for x, r in zip(
            ccs_df["client_name"], ccs_df[config_info["company_rename_col"]]
        )
    ]
    return ccs_df


def get_term_lists(config_info: dict) -> Tuple[List[str], List[List[str]]]:
    """Gets and processes term lists describing CCS"""

    search_terms = yaml_to_dict(config_info["search_term_list_path"])[
        "search_term_list"
    ]

    terms = []
    for t in search_terms:
        if "," in t:
            terms.append(t.replace('"', "").split(","))
        else:
            terms.append([t.replace('"', "")])

    # terms = [[substitute(t) for t in tt] for tt in terms]

    single_terms = []
    multiple_terms = []
    for x in terms:
        if len(x) == 1:
            single_terms.append(substitute(x[0], use_basename=False))
        else:
            multiple_terms.append([substitute(xx, use_basename=False) for xx in x])

    return (single_terms, multiple_terms)


def get_ccs_bills(config_info: dict):
    # get names of CCS bills
    ccs_bills = yaml_to_dict(config_info["law_list_path"])["mostly_ccs_provisions"]
    ccs_bills = [substitute(x, use_basename=False) for x in ccs_bills]
    return ccs_bills


def assign_sectors(df: pd.DataFrame, config_info: dict):
    """Given info in yamls, assigns companies to sectors and lumped sectors; adds new columns to df"""
    inverted_sector_dict = invert_sector_dict(config_info["company_sector_assignments"])
    lumped_sector_dict = yaml_to_dict(config_info["path_to_lumped_sector_info"])
    df["sector"] = [
        inverted_sector_dict[x] for x in df[config_info["company_rename_col"]]
    ]
    df["lumped_sector"] = [lumped_sector_dict["lightly_lumped"][s] for s in df.sector]
    df["very_lumped_sector"] = [lumped_sector_dict["very_lumped"][s] for s in df.sector]
    return df


def identify_ccs(df: pd.DataFrame, config_info: dict):
    """Use terms, law names, & sectors to identify very-likely, likely, and potentially ccs activities"""
    single_terms, multiple_terms = get_term_lists(config_info)
    ccs_bills = get_ccs_bills(config_info)

    # for simple dictionaries defined in the search term lists (not ccs, probably ccs, and maybe ccs)
    search_term_dict = yaml_to_dict(config_info["search_term_list_path"])

    # get rid of nans in lobbying activity description
    df.clean_description = df.clean_description.fillna(" ")

    # is ccs described in the lobbying description? (intermediate variables)
    df["ccs_single"] = [terms_present(x, single_terms) for x in df.clean_description]
    df["ccs_multiple"] = [
        any([terms_present(x, y, find_any=False) for y in multiple_terms])
        for x in df.clean_description
    ]
    # if description of lobbying activity contains either terms from the single-term list
    # or from the multi-term list, indicate that the activity contains a ccs description
    df["contains_ccs_description"] = [
        max(sgl, mlt) for sgl, mlt in zip(df["ccs_single"], df["ccs_multiple"])
    ]
    # is this a company dedicated to CCS tech and operations?
    df["ccs_company"] = [1 if x == "ccs" else 0 for x in df.sector]

    # is a ccs bill or a ccs-heavy bill with keyword terms (e.g. 'capture') directly mentioned?
    df["ccs_bills"] = [terms_present(x, ccs_bills) for x in df.clean_description]

    # are some of the terms that negate it being likely ccs (e.g., 'healthcare') present?
    df["not_ccs"] = [
        terms_present(x, search_term_dict["not_ccs"]) for x in df.clean_description
    ]

    # are terms that indicate this is probably--but not definitely ccs present?
    df["terms_probably_ccs"] = [
        terms_present(x, search_term_dict["probably_ccs"]) for x in df.clean_description
    ]

    # are terms that indicate this is probably--but not definitely ccs present?
    df["terms_maybe_ccs"] = [
        terms_present(x, search_term_dict["maybe_ccs"]) for x in df.clean_description
    ]

    # classify something as very likely CCS if it has a ccs description, is a ccs company,
    # mentions ccs bills, and does not contain a 'not ccs' term
    df["very_likely_ccs"] = [
        1 if ((d + b + c) > 0) & (n == 0) else 0
        for d, b, c, n in zip(
            df.contains_ccs_description, df.ccs_bills, df.ccs_company, df.not_ccs
        )
    ]

    # find those that, b/c of industry, 'probably ccs' is likely ccs.

    # omit findings of the 'low carbon economy' act, which has the term 'low carbon', but didn't deal with ccs
    df["low_carbon_economy_act"] = [
        1 if terms_present(x, ["low carbon economy", "lowcarbon economy"]) else 0
        for x in df.clean_description
    ]
    df["leaning_ccs"] = [
        (
            1
            if (probably and (not lca) and (sector in config_info["core_ff_sectors"]))
            else 0
        )
        for probably, sector, lca in zip(
            df.terms_probably_ccs, df.sector, df.low_carbon_economy_act
        )
    ]
    # likely CCS are all the 'very likely ccs' plus the 'core FF sector' organizations paired with 'probably' ccs
    df["likely_ccs"] = [
        1 if (((d + b + c + lean) > 0) & (n == 0)) else 0
        for d, b, c, n, lean in zip(
            df.contains_ccs_description,
            df.ccs_bills,
            df.ccs_company,
            df.not_ccs,
            df.leaning_ccs,  # maybes that are likely ccs b/c of industry
        )
    ]
    df["could_be_ccs"] = [
        (
            1
            if (probably and (not lca) and (sector in config_info["core_ff_sectors"]))
            else 0
        )
        for maybe, sector, lca in zip(
            df.terms_maybe_ccs, df.sector, df.low_carbon_economy_act
        )
    ]
    df["potentially_ccs"] = [
        1 if ((d + m + b + p + c) > 0) & (n == 0) else 0
        for d, b, m, p, c, n in zip(
            df.contains_ccs_description,
            df.ccs_bills,
            df.could_be_ccs,
            df.likely_ccs,
            df.ccs_company,
            df.not_ccs,
        )
    ]
    print("")
    print(
        df[
            [
                "not_ccs",
                "terms_probably_ccs",
                "terms_maybe_ccs",
                "contains_ccs_description",
                "leaning_ccs",
                "very_likely_ccs",
                "likely_ccs",
                "potentially_ccs",
                "ccs_bills",
            ]
        ].sum()
    )
    print("")

    return df


def apportion_filing_dollars_to_activities(df: pd.DataFrame, config_info: dict):
    """apportion total lobbying dollars spent (on filing) to individual lobbying activities"""

    # basic method: aportion using total number of activities
    df["activity_apportioned_usd"] = [
        usd / number_lobbying
        for usd, number_lobbying in zip(
            df.dollars_spent_lobbying, df.total_number_lobbying_activities
        )
    ]
    # apportion total filing dollars to activities using the fraction of the total number
    # lobbyists on the filing assigned to this activity
    df["lobbying_activity_usd"] = [
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


def subset_to_ccs_only(df: pd.DataFrame, config_info: dict):
    cleaned_df = df.loc[
        (df.lumped_sector != "REMOVE")
        & ((df.very_likely_ccs == 1) | (df.likely_ccs == 1) | (df.potentially_ccs == 1))
    ].copy(deep=True)

    return cleaned_df


def expand_and_tabulate_entities(
    df, config_info: dict
) -> Tuple[pd.DataFrame, List[str]]:
    """Use entity string to expand and tabulate counts of lobbying activities"""
    entities = set()
    for i in df.entities:
        entities = entities.union(set(eval(i)))
    entities = list(entities)

    # expand the json string of the entity list
    df["entity_expanded"] = [eval(x) for x in df.entities]
    df["n_entities_lobbied"] = [len(x) for x in df.entity_expanded]
    df["legistlative_entities_lobbied"] = [
        np.sum([("congress" in x) | ("house" in x) | ("senate" in x) for x in xx])
        for xx in df.entity_expanded
    ]
    df["executive_entities_lobbied"] = [
        t - l
        for t, l in zip(
            cleaned_df.n_entities_lobbied, cleaned_df.legistlative_entities_lobbied
        )
    ]
    # make binary variables for each govt. entity
    for e in entities:
        df[e] = [1 if e in entity_list else 0 for entity_list in df.entity_expanded]
    return df, entities


def add_political_party(df: pd.DataFrame, config_info: dict):
    """identify party in control of house, senate, and white house at time of each activity"""
    # get mapping for each congress and each branch of govt
    party_dict = yaml_to_dict(config_info["political_party_congress_path"])

    for b in ["senate", "house", "white_house"]:
        df["party_controlling_" + b] = [party_dict[b][c] for c in df.which_congress]

    return df


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--input_file", type=click.Path(file_okay=True, dir_okay=False), required=True
)
@click.option(
    "--output_file", type=click.Path(file_okay=True, dir_okay=False), required=True
)
def postprocess_ccs(
    config: Union[str, PosixPath],
    input_file: Union[str, PosixPath],
    output_file: Union[str, PosixPath],
):
    """Reads compiled ccs LDA API query outputs and identifies CCS lobbying activities"""
    config_info = yaml_to_dict(config)

    all_df = pd.read_csv(input_file)

    all_df = adjust_company_names(all_df, config_info)

    all_df = assign_sectors(all_df, config_info)

    all_df = identify_ccs(all_df, config_info)

    ccs_df = subset_to_ccs_only(all_df, config_info)

    ccs_df = apportion_filing_dollars_to_activities(ccs_df, config_info)

    ccs_df, entities = expand_and_tabulate_entities(ccs_df, config_info)

    ccs_df = add_political_party(ccs_df, config_info)

    ccs_df[config_info["subset_and_order_of_writeout_columns"] + entities].to_csv(
        output_file, index=False
    )

    logging.info(
        " ----- Postprocessed file written to %s ",
        str(output_file),
    )


if __name__ == "__main__":
    postprocess_ccs()
