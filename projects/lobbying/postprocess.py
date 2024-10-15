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


def find_description(
    df: pd.DataFrame,
    descriptor: str,
    filepath: str,
    return_count: bool = True,
):
    """For each lobbying activity (row), determine whether any of a set of terms are in description"""
    # are terms consistent with descriptor in the lobbying description? (intermediate variables)

    terms = yaml_to_dict(filepath)[descriptor]
    terms = substitute(terms, use_basename=False)

    df[f"count_{descriptor}"] = [terms_present(x, terms) for x in df.clean_description]

    # if description of lobbying activity contains either terms from the single-term list
    # or from the multi-term list, indicate that the activity contains description
    df[f"{descriptor}"] = df[f"count_{descriptor}"].astype(bool).astype(int)
    if return_count:
        return df
    df.drop(f"count_{descriptor}", axis=1, inplace=True)
    return df


def find_bill_numbers_for_congress(df: pd.DataFrame, postproc_specs: str, id_str: str):
    """identify lobbying activities (rows) that mention CCS-focused bills for a given congress"""
    # get dictionary with congress number/bill number for CCS bills

    ccs_bill_numbers = yaml_to_dict(postproc_specs)[id_str]
    # identify activities that reference specific bill numbers
    return [
        (
            1
            if (
                terms_present(d, ccs_bill_numbers[which_congress])
                | terms_present(
                    d, [x.replace(" ", "") for x in ccs_bill_numbers[which_congress]]
                )
                | terms_present(
                    d, [x.replace("SB ", "S") for x in ccs_bill_numbers[which_congress]]
                )
                | terms_present(
                    d,
                    [x.replace("SB ", "S ") for x in ccs_bill_numbers[which_congress]],
                )
            )
            else 0
        )
        for d, which_congress in zip(df.clean_description, df.which_congress)
    ]


def identify_ccs(df: pd.DataFrame, config_info: dict):
    """Use terms, law names, & sectors to identify very-likely, likely, & potentially ccs activities
    Args:
        df: pandas dataframe containing lobbying activity information
        config_info: dictionary read in from configuration yaml with paths to data
    Returns:
        df: updated pandas dataframe with binary columns and term-count columns indicating
        ccs-related lobbying activity
    """

    # find descriptions of subsets of terms for which also want a term count
    # identify activities that are explicitly CCS
    for t in ["contains_description", "clean_h2_description", "bills_with_some_ccs"]:
        df = find_description(
            df, t, config_info["postproc_specs_path"], return_count=True
        )

    # find descriptions of subsets of terms for which we don't want a term count
    for t in [
        "ccs_bills",
        "bills_with_ccs_terms",
        "not_ccs",
        "h2_mention",
        "terms_consistent_with_ccs",
        "terms_could_be_ccs",
    ]:
        df = find_description(
            df, t, config_info["postproc_specs_path"], return_count=False
        )

    # identify activities that reference bills/laws with CCS provisions; return count
    df["ccs_bills_number_only"] = find_bill_numbers_for_congress(
        df, config_info["postproc_specs_path"], "congress_bill_nos"
    )

    postproc_dict = yaml_to_dict(config_info["postproc_specs_path"])
    # handle 'nebulous' hydrogen terms (e.g., "low-carbon hydrogen", NOT terms such as "clean hydrogen",
    # which is definitely CCS, and which is handled as a CCS term, above)

    df["h2_mention_core_ff"] = [
        1 if ((h == 1) and (s in postproc_dict["core_ff_sectors"])) else 0
        for h, s in zip(df.h2_mention, df.sector)
    ]
    df["h2_mention_ff_adjacent"] = [
        1 if ((h == 1) and (s in postproc_dict["ff_adjacent_sectors"])) else 0
        for h, s in zip(df.h2_mention, df.sector)
    ]
    # identify a few 'always CCS' and 'always not CCS' sectors
    for sector in ["ccs", "clean hydrogen", "green hydrogen"]:
        # is this a company dedicated to CCS tech and operations?
        df[f"{sector.replace(' ','_')}_company"] = [
            1 if x == sector else 0 for x in df.sector
        ]

    # find those that, b/c of industry, 'probably ccs' is almost certainly ccs.
    # omit findings of the 'low carbon economy' act, which has the term 'low carbon', but didn't deal with ccs
    df["low_carbon_economy_act"] = [
        1 if terms_present(x, ["low carbon economy", "lowcarbon economy"]) else 0
        for x in df.clean_description
    ]
    df["ccs_because_of_who_says_it"] = [
        (
            1
            if (
                (
                    probably
                    and (not lca)
                    and (sector in postproc_dict["ff_adjacent_sectors"])
                )
                | (bill_with_ccs and (sector in postproc_dict["ff_adjacent_sectors"]))
            )
            else 0
        )
        for probably, bill_with_ccs, sector, lca in zip(
            df.terms_consistent_with_ccs,
            df.bills_with_ccs_terms,
            df.sector,
            df.low_carbon_economy_act,
        )
    ]

    # classify something as very likely CCS if it has a ccs description, is a ccs company/clean hydrogen company,
    # mentions ccs bills, and does not contain a 'not ccs' term
    df["definitely_ccs"] = [
        1 if ((d + +b + bn + c + h + hff) > 0) & (n == 0) else 0
        for d, b, bn, c, h, hff, n in zip(
            df.contains_description,
            df.ccs_bills,
            df.ccs_bills_number_only,
            df.ccs_company,
            df.clean_hydrogen_company,
            df.h2_mention_core_ff,
            df.not_ccs,
        )
    ]

    #
    df["very_likely_ccs"] = [
        1 if ((d + h + w) > 0) & (n == 0) else 0
        for d, h, w, n in zip(
            df.definitely_ccs,
            df.h2_mention_ff_adjacent,
            df.ccs_because_of_who_says_it,
            df.not_ccs,
        )
    ]

    # likely CCS are all the 'very likely ccs' plus the 'core FF sector' organizations paired with 'probably' ccs
    df["likely_ccs"] = [
        1 if (((vl + lean + law) > 0) & (n == 0)) else 0
        for vl, n, lean, law in zip(
            df.very_likely_ccs,
            df.not_ccs,
            df.terms_consistent_with_ccs,  # probably ccs, but not definitely
            df.bills_with_ccs_terms,
        )
    ]
    # identify the activities that could be ccs because of ccs-relevant terms and industry
    df["could_be_ccs"] = [
        (
            1
            if (
                maybe and (not lca) and (sector in postproc_dict["ff_adjacent_sectors"])
            )
            else 0
        )
        for maybe, sector, lca in zip(
            df.terms_could_be_ccs, df.sector, df.low_carbon_economy_act
        )
    ]
    df["potentially_ccs"] = [
        1 if ((cb + l) > 0) & (n == 0) else 0
        for cb, l, n in zip(
            df.could_be_ccs,
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
def postprocess(
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

    logging.info(" >>> Classifying lobbying activities")
    # get rid of nans in lobbying activity description
    all_df.clean_description = all_df.clean_description.fillna(" ")
    all_df = identify_ccs(all_df, config_info)

    if config_info["subset_to_ccs_only"]:
        logging.info(" >>> Subsetting lobbying activities")
        ccs_df = subset_to_ccs_only(all_df, config_info)
    else:
        ccs_df = all_df

    logging.info(" >>> Apportioning dollars to individual lobbying activities")
    ccs_df = apportion_filing_dollars_to_activities(ccs_df)

    ccs_df, entities = expand_and_tabulate_entities(ccs_df)

    ccs_df = add_political_party(ccs_df, config_info)
    logging.info(" >>> Writing out data")
    postproc_dict = yaml_to_dict(config_info["postproc_specs_path"])
    if "rename_columns" in postproc_dict:
        ccs_df.rename(columns=postproc_dict["rename_columns"], inplace=True)

    ccs_df = ccs_df[postproc_dict["subset_and_order_of_writeout_columns"] + entities]

    # order values such that someone who is reading the output first sees all the 'definitely ccs' ones
    ccs_df.sort_values(
        by=[
            "definitely_ccs",
            "very_likely_ccs",
            "likely_ccs",
            "potentially_ccs",
            "sector",
            "organization",
            "filing_year",
        ],
        ascending=False,
        inplace=True,
    )
    ccs_df.to_csv(output_file, index=False)

    logging.info(
        " ----- Postprocessed file written to %s ",
        str(output_file),
    )


if __name__ == "__main__":
    postprocess()
