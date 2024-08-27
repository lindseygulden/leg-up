import requests
import json
import pandas as pd
from flatten_json import flatten
from utils.io import dict_to_yaml, yaml_to_dict
from lobbying_utils import which_congress, terms_present
from math import ceil
from typing import List, Union
from pathlib import PosixPath
import click


CHUNK_SIZE_DEFAULT = 30
lda_username = "lgulden"
lda_apikey = "4d1a4bc3be920e859b3862a25d3725d741028d42"

issues_string = "OR".join(
    [
        '"carbon capture"',
        '"co2 capture"',
        '"carbon dioxide capture"',
        '"capture and store"',
        '"capture and storage"',
        '"capture transportation and storage"',
        '"capture transport and storage"',
        '"capture transport utilization and storage"',
        '"capture transport use and storage"',
        '"capture transport use and sequestration"',
        '"capture transport utilization and sequestration"',
        '"capture transport and store"',
        '"capture of carbon dioxide"',
        '"capture of co2"',
        '"hydrogen hub"',
        '"hydrogen hubs"',
        '"clean hydrogen"',
        '" hydrogen "," greenhouse "',
        '"blue hydrogen"',
        '"capture and sequestration"',
        '"capture utilization and sequestration"',
        '"capture use and storage"',
        '"capture use and sequestration"',
        '" CCS "',
        '" CC&S "',
        '" CCUS "',
        '" 45Q "',
        '" 45V "',
        '"enhanced oil recovery"',
        '" EOR "',
        "carbon management",
        '"low carbon solutions"',
        '"carbon dioxide pipelines"',
        '"carbon dioxide pipeline"',
        "co2 pipeline",
        '"co2 pipelines"',
    ]
)

# for testing: issues_string = '"co2 pipelines"'


def get_list_govt_entities():
    govt_entities = requests.get(
        "https://lda.senate.gov/api/v1/constants/filing/governmententities/"
    )
    entity_df = pd.DataFrame(govt_entities.json())
    entities = sorted([x.lower() for x in list(entity_df["name"])])
    return entities


def parse_dollars_spent(income, expense):
    if (income is None) & (expense is None):
        return "income and expenses are zero", 0.0
    if income is None:
        return "corporation lobbying for itself", float(expense)
    if expense is None:
        return "hired lobbying firm", float(income)
    else:
        return "both income and expense > $0", float(income) + float(expense)


def initialize_row(govt_entities, result, filing_id):
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

    initialize_row_dict["filing_id"] = filing_id
    initialize_row_dict["url"] = result["url"]
    initialize_row_dict["filing_year"] = int(result["filing_year"])
    initialize_row_dict["filing_period"] = result["filing_period"]
    initialize_row_dict["filing_type"] = result["filing_type"]
    initialize_row_dict["lobbyist_posted_by_name"] = result["posted_by_name"]
    initialize_row_dict["lobbyist_registrant_id"] = result["registrant"]["id"]
    initialize_row_dict["lobbyist_registrant_name"] = result["registrant"]["name"]
    initialize_row_dict["lobbyist_registrant_contact"] = result["registrant"][
        "contact_name"
    ]
    initialize_row_dict["client_id"] = result["client"]["id"]
    initialize_row_dict["client_client_id"] = result["client"]["client_id"]
    initialize_row_dict["client_name"] = result["client"]["name"]
    initialize_row_dict["client_general_description"] = result["client"][
        "general_description"
    ]
    initialize_row_dict["client_state"] = result["client"]["state"]
    initialize_row_dict["client_country"] = result["client"]["country"]

    initialize_row_dict["affiliated_organizations_present"] = False
    if len(result["affiliated_organizations"]) > 0:
        initialize_row_dict["affiliated_organizations_present"] = True

    initialize_row_dict["convictions_present"] = False
    if len(result["conviction_disclosures"]) > 0:
        initialize_row_dict["convictions_present"] = True

    return initialize_row_dict


def parse_lobbyists(lobbyists: dict, details: dict) -> List[dict]:

    lobbyist_list = []

    lobby_dict = {}
    lobby_dict["firm_name"] = details["lobbyist_registrant_name"]
    lobby_dict["client_name"] = details["client_name"]
    lobby_dict["general_issue_code"] = details["general_issue_code"]
    lobby_dict["description"] = details["description"]
    lobby_dict["filing_period"] = details["filing_period"]
    lobby_dict["filing_year"] = details["filing_year"]
    lobby_dict["url"] = details["url"]
    lobby_dict["filing_id"] = details["filing_id"]

    # unpack lobbyists list
    for lobbyist in lobbyists:
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


def streamline_bill_references(x_df: pd.DataFrame):

    replace_dict = {
        "H.Con.Res": "HCR",
        "H. Con. Res.": "HCR",
        "S.Con.Res.": "SCR",
        "S. Con. Res.": "SCR",
        "H. J. Res.": "HJR",
        "H. J. Res": "HJR",
        "H.J. Res.": "HJR",
        "H.J. Res": "HJR",
        "S. J. Res.": "SJR",
        "H.J.Res.": "HJR",
        "Public Law No.:": "PL",
        "Public Law No:": "PL",
        "Public Law": "PL",
        "P. L.": "PL",
        "P.L.": "PL",
        "PL": " PL",
        " H ": " HR",
        "H.R.": " HR",
        "H.R": " HR",
        "H. R.": " HR",
        "S.": " SB",
        "S ": " SB",
        " S ": " SB",
        "/S": "/ SB",
        "U. SB": "US",  # fix the unavoidable U.S. -> U.SB error
        "U SB": "US",  # fixing errors
        "CC SB": "CCS",  # fixing errors
        "CCU SB": "CCUS",  # fixing errors
        "HR.": " HR",
        "SB ": "SB",
        "HR ": "HR",
        "HCR ": "HCR",
        "HJR ": "HJR",
        "SCR ": "SCR",
        "SJR ": "SJR",
        "PL ": "PL",
    }
    for key, value in replace_dict.items():
        x_df["description"] = [
            x.replace(key, value) if x is not None else "" for x in x_df.description
        ]
    return x_df


def consolidate_rows(
    row_list: List[dict], govt_entities: List[str], discard_filing_types: List[str]
):
    tmp_df = pd.DataFrame(row_list)
    entities_influenced = tmp_df[govt_entities].sum()
    zeroed = list(entities_influenced[entities_influenced == 0].index)
    ccs_df = tmp_df[[x for x in tmp_df.columns.values if x not in zeroed]].copy()
    ccs_df = ccs_df.drop_duplicates(subset=ccs_df.columns.difference(["filing_id"]))
    ccs_unique_filing_ids = list(
        ccs_df.filing_id.unique()
    )  # keep a list of the non-duplicate filing ids

    # remove unwanted filing types
    ccs_df = ccs_df.loc[ccs_df.filing_type.isin(discard_filing_types)]
    # assign number of congress to dataframe
    ccs_df["which_congress"] = [which_congress(y) for y in ccs_df["filing_year"]]

    # somewhat streamline references to bills in lobbying descriptions
    ccs_df = streamline_bill_references(ccs_df)

    return ccs_df, ccs_unique_filing_ids


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
def query_ccs(config: Union[str, PosixPath], output_dir: Union[str, PosixPath]):

    config_info = yaml_to_dict(config)

    chunk_size = CHUNK_SIZE_DEFAULT
    if "chunk_size" in config_info:
        chunk_size = config_info["chunk_size"]
    discard_filing_types = ["RR", "RA"]
    if "discard_filing_types" in config_info:
        discard_filing_types = config_info["discard_filing_types"]

    # login/authenticate
    requests.post(
        "https://lda.senate.gov/api/auth/login/",
        data={
            "username": config_info["lda_username"],
            "password": config_info["lda_apikey"],
        },
    )

    # get govt entity names
    govt_entities = get_list_govt_entities()

    row_list = (
        []
    )  # initialize holder for each row (which corresponds to a single lobbying activity)
    lobby_list = []  # initialize holder for lobbyist info
    filing_id = 0  # initialize unique id for filing documents
    ccs_unique_filing_ids = []

    lobbyists_df = None  # initialize data frame for storing lobbyist data

    query_all_filings = f"https://lda.senate.gov/api/v1/filings/?filing_specific_lobbying_issues={issues_string}"
    f = requests.get(query_all_filings)
    n_pages = int(ceil(f.json()["count"] + 1) / 25)
    which_chunk = 1
    n_chunks = int(ceil(n_pages / chunk_size))
    print("")
    print(f"Preparing {n_chunks} files for lobbying activities and lobbyists")
    print("")

    for page in range(1, n_pages):
        print(f"Querying page {page} of {n_pages-1}")
        query = query_all_filings + f"&page={page}"

        f = requests.get(query)

        results = f.json()["results"]

        # extract data from each filing form returned from query
        for result in results:

            row_dict_base = initialize_row(govt_entities, result, filing_id)
            activities = result["lobbying_activities"]

            for activity_count, activity in enumerate(activities):
                row_dict = row_dict_base.copy()
                # set up row dictionary using entity booleans
                row_dict["activity_count"] = (
                    activity_count  # how many activiites are on one filing document?
                )
                row_dict["general_issue_code"] = activity["general_issue_code"]
                row_dict["description"] = activity["description"]
                lobbyists_for_this_activity = parse_lobbyists(
                    activity["lobbyists"], row_dict
                )
                lobby_list = lobby_list + lobbyists_for_this_activity
                # parse lobbyists
                lobbyist_id_list = []
                for lobbyist in activity["lobbyists"]:
                    lobbyist_id_list.append(lobbyist["lobbyist"]["id"])

                row_dict["lobbyist_ids"] = "; ".join(
                    ["None" if x is None else str(x) for x in lobbyist_id_list]
                )
                # parse all government entitites lobbied
                for entity in activity["government_entities"]:
                    row_dict[entity["name"].lower()] = 1

                row_list.append(row_dict.copy())

                row_dict.clear()
            filing_id += 1
        if (page % chunk_size) == 0:
            ccs_df, ccs_unique_filing_ids = consolidate_rows(
                row_list, govt_entities, discard_filing_types
            )
            # write out lobbyist data
            lobbyists_df = pd.DataFrame(lobby_list)
            lobbyists_df.loc[lobbyists_df.filing_id.isin(additional_unique_ids)].to_csv(
                Path(output_dir)
                / Path(config_info["lobbyist_file_name"])
                / Path(f"{which_chunk}_of_{n_chunks}.csv")
            )
            # write out CCS lobbying info
            ccs_df.to_csv(
                Path(output_dir)
                / Path(config_info["output_file_name"])
                / Path(f"{which_chunk}_of_{n_chunks}.csv")
            )
            print(f"------------ writing {which_chunk} of {n_chunks} chunks to CSV")

            # reinitialize holders for next chunk
            row_list = []
            lobby_list = []
            which_chunk += 1

    ccs_df, ccs_unique_filing_ids = consolidate_rows(
        row_list, govt_entities, discard_filing_types
    )
    ccs_df.to_csv(
        Path(output_dir)
        / Path(config_info["output_file_name"])
        / Path(f"{which_chunk}_of_{n_chunks}.csv")
    )

    # write out lobbyist data
    lobbyists_df = pd.DataFrame(lobby_list)
    lobbyists_df.loc[lobbyists_df.filing_id.isin(additional_unique_ids)].to_csv(
        Path(output_dir)
        / Path(config_info["lobbyist_file_name"])
        / Path(f"{which_chunk}_of_{n_chunks}.csv")
    )


if __name__ == "__main__":
    query_ccs()
