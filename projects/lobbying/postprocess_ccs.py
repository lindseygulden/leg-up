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
    "--input_file", type=click.Path(file_okay=True, dir_okay=False), required=True
)
@click.option(
    "--output_file", type=click.Path(file_okay=True, dir_okay=False), required=True
)
def get_term_lists():
    search_term_dict = yaml_to_dict(
        "/Users/lindseygulden/dev/leg-up-private/projects/lobbying/search_term_list.yml"
    )

    search_terms = search_term_dict["search_term_list"]
    probably_ccs = search_term_dict["probably_ccs"]
    maybe_ccs = search_term_dict["maybe_ccs"]
    not_ccs = search_term_dict["not_ccs"]

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
            single_terms.append(x[0])
        else:
            multiple_terms.append(x)

    # get names of CCS bills
    ccs_bills = yaml_to_dict(
        "/Users/lindseygulden/dev/leg-up-private/projects/lobbying/ccs_laws.yml"
    )["mostly_ccs_provisions"]
    ccs_bills = [re.sub(r"[^\w\s]", "", x) for x in ccs_bills]


def postprocess_ccs(
    config: Union[str, PosixPath],
    input_file: Union[str, PosixPath],
    output_file: Union[str, PosixPath],
):
    """Reads compiled ccs LDA API query outputs and identifies CCS lobbying activities"""
    config_info = yaml_to_dict(config)

    ccs_df = pd.read_csv(input_file)
    ccs_df.clean_client_general_description = (
        ccs_df.clean_client_general_description.fillna("")
    )
    ccs_df.client_rename = ccs_df.client_rename.fillna("")

    portland_replace_dict = yaml_to_dict(
        "/Users/lindseygulden/dev/leg-up-private/projects/lobbying/company_name_replacements.yml"
    )

    ccs_df["client_rename"] = [
        portland_replace_dict[x] if x in list(portland_replace_dict.keys()) else r
        for x, r in zip(ccs_df.client_name, ccs_df.client_rename)
    ]
    logging.info(
        " ----- Compiled file written to %s ",
        str(output_file),
    )


if __name__ == "__main__":
    compile_ccs_files()
