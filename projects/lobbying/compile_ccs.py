"""script to read in files read out by ccs and compile them into a single csv"""

import pandas as pd
from utils.io import yaml_to_dict

from typing import Union
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
    terms_present,
)

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--input_dir", type=click.Path(file_okay=False, dir_okay=True), required=True
)
@click.option(
    "--output_file", type=click.Path(file_okay=True, dir_okay=False), required=True
)
def compile_ccs_files(
    config: Union[str, PosixPath],
    input_dir: Union[str, PosixPath],
    output_file: Union[str, PosixPath] = "compiled.csv",
):
    """Reads CSVs in input_dir (i.e, the LDA API outputs), processes them, & writes compilation to output_file"""
    config_info = yaml_to_dict(config)

    groupby_cols = config_info["groupby_columns"]

    entities = get_list_govt_entities(
        config_info["entity_endpoint"],
        session=api_authenticate(
            config_info["authentication_endpoint"],
            config_info["lda_username"],
            config_info["lda_apikey"],
        ),
    )
    # get all csv files in input directory
    input_files = [
        join(input_dir, f)
        for f in listdir(input_dir)
        if isfile(join(input_dir, f)) and (f.split(".")[-1] == "csv")
    ]
    logging.info(
        " ----- Reading, postprocessing, and compiling the %s files in %s ",
        str(len(input_files)),
        str(input_dir),
    )
    df_list = []
    rename_dict = {}
    for file in input_files:
        logging.info("  %s", str(file))
        api_results_df = pd.read_csv(
            file,
            index_col=[0],
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

        remove_sector_descriptions = yaml_to_dict(
            config_info["sector_description_file"]
        )[config_info["remove_org_key"][0]][config_info["remove_org_key"][1]]

        df["client_rename"] = [
            "remove" if terms_present(x, remove_sector_descriptions) else n
            for x, n in zip(df.clean_client_general_description, df.client_rename)
        ]

        df = get_latest_filings(df, groupby_cols)
        df = df.loc[df.client_rename != "remove"]
        df_list.append(df)

    ccs_df = pd.concat(df_list)
    ccs_df = get_latest_filings(ccs_df, groupby_cols)
    # fill in nans/nones with empty string for description and rename of client
    ccs_df.clean_client_general_description = (
        ccs_df.clean_client_general_description.fillna("")
    )
    ccs_df.client_rename = ccs_df.client_rename.fillna("")

    ccs_df["batch"] = ""
    if "batch_name" in config_info:
        ccs_df["batch"] = config_info["batch_name"]
    ccs_df.to_csv(output_file, index=False)

    logging.info(
        " ----- Compiled file written to %s ",
        str(output_file),
    )


if __name__ == "__main__":
    compile_ccs_files()
