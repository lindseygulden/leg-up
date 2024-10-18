"""Use when new companies are in LDA query results: assembles yaml file with client names that are missing
from sector assignments"""

import logging
from pathlib import Path, PosixPath
from typing import Union

import click
import pandas as pd

from projects.lobbying.postproc_utils import invert_sector_dict
from utils.io import dict_to_yaml, yaml_to_dict

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--input_file",
    type=click.Path(file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--output_dir",
    type=click.Path(file_okay=False, dir_okay=True),
    required=False,
    default=".",
)
def list_new(
    config: Union[str, PosixPath],
    input_file: Union[str, PosixPath],
    output_dir: Union[str, PosixPath],
):
    """Queries the Lobbying Disclosure Act lda API given terms"""
    config_info = yaml_to_dict(config)

    replace_dict = yaml_to_dict(config_info["company_name_replacements"])
    logging.info(" >>> Reading compiled query results from %s", input_file)
    df = pd.read_csv(input_file)
    df["client_rename"] = df.client_rename.fillna("")
    df["client_rename"] = [
        replace_dict[x] if x in list(replace_dict.keys()) else r
        for x, r in zip(df.client_name, df.client_rename)
    ]

    inverted_sector_dict = invert_sector_dict(config_info["company_sector_assignments"])

    companies = [c.rstrip().lstrip() for c in list(inverted_sector_dict.keys())]

    not_assigned = sorted(list(set(df.client_rename.unique()) - set(companies)))
    missing_dict = {}
    missing_list = []
    for c in not_assigned:
        names = list(df.loc[df.client_rename == c].client_name.unique())
        for n in names:
            missing_dict = missing_dict | {n: c}
            missing_list.append(c)
    dict_to_yaml(
        missing_dict, str(Path(output_dir) / Path("missing_company_mapping.yml"))
    )
    dict_to_yaml(
        {"for sector mapping": sorted(list(set(missing_list)))},
        str(Path(output_dir) / Path("missing_company_sector_list.yml")),
    )
    logging.info(" >>> Companies not yet assigned to sectors written to %s", output_dir)


if __name__ == "__main__":
    list_new()
