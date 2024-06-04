"""Script to extract US cement production facilities from global database """

# Requires input data extracted from the following location, in CSV format:
# https://datadryad.org/stash/dataset/doi:10.5061/dryad.6t1g1jx4f
# used the file entitled SFI-Global-Cement-Database-assets.csv
from pathlib import PosixPath
from typing import Union

import click
import pandas as pd


@click.command()
@click.option(
    "--input_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--output_file",
    type=click.Path(file_okay=True, dir_okay=False),
    required=True,
)
def cement(input_file: Union[str, PosixPath], output_file: Union[str, PosixPath]):
    """Function for ingesting and lightly processing cement-facility location data"""
    cement_df = pd.read_csv(input_file)
    us_cement_df = (
        cement_df[
            [
                "latitude",
                "longitude",
                "city",
                "state",
                "status",
                "plant_type",
                "capacity",
                "production_type",
            ]
        ]
        .loc[cement_df.country == "United States of America"]
        .copy(deep=True)
    )

    us_cement_df.rename(columns={"capacity": "capacity_million_tons"}, inplace=True)
    us_cement_df.to_csv(output_file)
    return us_cement_df


if __name__ == "__main__":
    cement()
