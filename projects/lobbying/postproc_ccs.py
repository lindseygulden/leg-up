'''script to read in files read out by ccs'''
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
from projects.lobbying.postproc import invert_sector_dict


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
    
    config_info = yaml_to_dict(
    "/Users/lindseygulden/dev/leg-up-private/projects/lobbying/config_ccs_lda.yml"
)
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
# remove_sector_descriptions = yaml_to_dict(
#    "/Users/lindseygulden/dev/leg-up-private/projects/lobbying/sector_company_description_terms.yml"
# )["remove"]
remove_sector_descriptions = yaml_to_dict(
    "/Users/lindseygulden/dev/leg-up-private/projects/lobbying/sector_descriptions.yml"
)["remove these organizations"]["keep"]
df_list = []
rename_dict = {}
for i in range(1, 5):  # 19):  # 159):
    api_results_df = pd.read_csv(
        # f"/Volumes/Samsung_T5/data/lobbying/ccslaws/ccs_lda_filings_{i}.csv",
        # f"/Volumes/Samsung_T5/data/lobbying/ccs/ccs_lda_filings_{i}.csv",
        f"/Volumes/Samsung_T5/data/lobbying/ccs_additional/ccs_lda_filings_{i}.csv",
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
    df["entities"] = df[entities].T.apply(lambda x: dumps(get_smarties(x, entities)))
    df.drop(entities, axis=1, inplace=True)
    df["clean_description"] = [
        substitute(d, use_basename=True) for d in df["description"]
    ]
    df["clean_client_general_description"] = [
        substitute(d, use_basename=False) for d in df["client_general_description"]
    ]
    df["client_rename"] = [
        "remove" if terms_present(x, remove_sector_descriptions) else n
        for x, n in zip(df.clean_client_general_description, df.client_rename)
    ]
    df["client_rename"] = [
        "remove" if terms_present(x, remove_sector_descriptions) else x
        for x in df.client_rename
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

ccs_df["batch"] = "additional"
ccs_df.to_csv("/Volumes/Samsung_T5/data/lobbying/ccs/ccs_additional_compiled.csv")
# ccs_df["batch"] = "ccs description and/or ccs specific laws and bills"
# ccs_df.to_csv("/Volumes/Samsung_T5/data/lobbying/ccs/ccs_compiled.csv")
# ccs_df["batch"] = "relevant laws"
# ccs_df.to_csv("/Volumes/Samsung_T5/data/lobbying/ccs/ccslaws_compiled.csv")