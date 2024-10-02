# pylint: disable=use-a-generator
"""script to read in files read out by ccs and compile them into a single csv"""

import logging
from pathlib import PosixPath
from typing import List, Tuple, Union

import click
import numpy as np
import pandas as pd
from asteval import Interpreter

from projects.lobbying.postproc_utils import (
    invert_sector_dict,
    substitute,
    terms_present,
)
from utils.io import yaml_to_dict

logging.basicConfig(level=logging.INFO)


def quarter_to_decimal(quart: str) -> float:
    """converts string quarter to decimal fraction of year"""
    if quart == "second_quarter":
        return 135 / 365.25
    if quart == "third_quarter":
        return 227 / 365.25
    if quart == "first_quarter":
        return 46 / 365.25
    if quart == "fourth_quarter":
        return 319 / 365.25
    if quart == "year_end":
        return 274 / 265.25
    if quart == "mid_year":
        return 91 / 365.25
    raise ValueError(f"{quart} quarter string is not one of accepted values.")


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


def get_term_lists(list_path: str, list_key: str) -> Tuple[List[str], List[List[str]]]:
    """Gets and processes term lists describing CCS"""

    search_terms = yaml_to_dict(list_path)[list_key]

    terms = []
    for term in search_terms:
        if "," in term:
            terms.append(term.replace('"', "").split(","))
        else:
            terms.append([term.replace('"', "")])

    single_terms = []
    multiple_terms = []
    for x in terms:
        if len(x) == 1:
            single_terms.append(x[0])
        else:
            multiple_terms.append(x)

    return (single_terms, multiple_terms)


def get_ccs_bills(config_info: dict, which_laws: str):
    """get names of CCS bills from the appropriate yaml file"""
    ccs_bills = yaml_to_dict(config_info["law_list_path"])[which_laws]

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
    single_terms, multiple_terms = get_term_lists(
        config_info["postproc_term_list_path"], "search_term_list"
    )
    _, bill_with_ccs_term = get_term_lists(
        config_info["law_list_path"], "ccs_law_with_ccs_term"
    )
    bill_with_ccs_term = [
        [substitute(x, use_basename=False) for x in xx] for xx in bill_with_ccs_term
    ]
    ccs_bills, _ = get_term_lists(config_info["law_list_path"], "mostly_ccs_provisions")
    ccs_bills = [substitute(x, use_basename=False) for x in ccs_bills]

    law_with_ccs_provisions, _ = get_term_lists(
        config_info["law_list_path"], "contains_ccs_provisions"
    )
    law_with_ccs_provisions = [
        substitute(x, use_basename=False) for x in law_with_ccs_provisions
    ]

    # get dictionary with congress number/bill number for CCS bills
    ccs_bill_numbers = yaml_to_dict(config_info["law_list_path"])["congress_bill_nos"]

    # for simple dictionaries defined in the search term lists (not ccs, probably ccs, and maybe ccs)
    search_term_dict = yaml_to_dict(config_info["postproc_term_list_path"])

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
        1 if (sgl + mlt) > 0 else 0
        for sgl, mlt in zip(df["ccs_single"], df["ccs_multiple"])
    ]
    # identify descriptions in which there is a specific larger law (with ccs provisions) paired
    # with a specific phrase
    df["bill_with_ccs_term"] = [
        any([terms_present(x, y, find_any=False) for y in bill_with_ccs_term])
        for x in df.clean_description
    ]
    # is this a company dedicated to CCS tech and operations?
    df["ccs_company"] = [1 if x == "ccs" else 0 for x in df.sector]

    # is a ccs bill or a ccs-heavy bill with keyword terms (e.g. 'capture') directly mentioned?
    df["ccs_bills"] = [terms_present(x, ccs_bills) for x in df.clean_description]

    df["ccs_bills_number_only"] = [
        (
            1
            if (
                terms_present(d, ccs_bill_numbers[which_congress])
                | terms_present(
                    d, [x.replace(" ", "") for x in ccs_bill_numbers[which_congress]]
                )
            )
            else 0
        )
        for d, which_congress in zip(df.clean_description, df.which_congress)
    ]

    df["bill_contains_some_ccs"] = [
        terms_present(x, law_with_ccs_provisions) for x in df.clean_description
    ]
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
        1 if ((d + b + bn + c) > 0) & (n == 0) else 0
        for d, b, bn, c, n in zip(
            df.contains_ccs_description,
            df.ccs_bills,
            df.ccs_bills_number_only,
            df.ccs_company,
            df.not_ccs,
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
        1 if (((vl + lean + law) > 0) & (n == 0)) else 0
        for vl, n, lean, law in zip(
            df.very_likely_ccs,
            df.not_ccs,
            df.leaning_ccs,  # maybes that are likely ccs b/c of industry
            df.bill_with_ccs_term,
        )
    ]
    df["could_be_ccs"] = [
        (
            1
            if (maybe and (not lca) and (sector in config_info["core_ff_sectors"]))
            else 0
        )
        for maybe, sector, lca in zip(
            df.terms_maybe_ccs, df.sector, df.low_carbon_economy_act
        )
    ]
    df["potentially_ccs"] = [
        1 if ((cb + p + l) > 0) & (n == 0) else 0
        for cb, p, l, n in zip(
            df.could_be_ccs,
            df.terms_probably_ccs,
            df.likely_ccs,
            df.not_ccs,
        )
    ]

    return df


def apportion_filing_dollars_to_activities(df: pd.DataFrame):
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
    """Get rid of the rows that likely don't have much to do with CCS"""
    cleaned_df = df.loc[
        (df.lumped_sector != config_info["remove_sector_name"])
        & ((df.very_likely_ccs == 1) | (df.likely_ccs == 1) | (df.potentially_ccs == 1))
    ].copy(deep=True)

    return cleaned_df


def expand_and_tabulate_entities(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Use entity string to expand and tabulate counts of lobbying activities"""
    aeval = Interpreter()
    entities = set()
    for i in df.entities:
        entities = entities.union(set(aeval(i)))
    entities = list(entities)

    # expand the json string of the entity list
    df["entity_expanded"] = [aeval(x) for x in df.entities]
    df["n_entities_lobbied"] = [len(x) for x in df.entity_expanded]
    df["legistlative_entities_lobbied"] = [
        np.sum([("congress" in x) | ("house" in x) | ("senate" in x) for x in xx])
        for xx in df.entity_expanded
    ]
    df["executive_entities_lobbied"] = [
        t - l for t, l in zip(df.n_entities_lobbied, df.legistlative_entities_lobbied)
    ]
    # make binary variables for each govt. entity
    for entity in entities:
        df[entity] = [
            1 if entity in entity_list else 0 for entity_list in df.entity_expanded
        ]
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
    logging.info(" >>> Replacing company names")
    all_df = adjust_company_names(all_df, config_info)
    logging.info(" >>> Assigning companies to sectors")
    all_df = assign_sectors(all_df, config_info)
    logging.info(" >>> Identifying CCS lobbying activities")
    all_df = identify_ccs(all_df, config_info)
    logging.info(" >>> Subsetting CCS lobbying activities")
    ccs_df = subset_to_ccs_only(all_df, config_info)
    logging.info(" >>> Apportioning dollars to individual lobbying activities")
    ccs_df = apportion_filing_dollars_to_activities(ccs_df)

    ccs_df, entities = expand_and_tabulate_entities(ccs_df)

    ccs_df = add_political_party(ccs_df, config_info)
    logging.info(" >>> Writing out data")
    ccs_df[config_info["subset_and_order_of_writeout_columns"] + entities].to_csv(
        output_file, index=False
    )

    logging.info(
        " ----- Postprocessed file written to %s ",
        str(output_file),
    )


if __name__ == "__main__":
    postprocess_ccs()
