"""Script to assign apprixmate lat/lon locations to ammonia plants
(whose location is identified with city and state) compute drinking water source percentages for each US county"""

# Requires input data extracted from the following location, in CSV format with a 'city' and 'state' column
# https://www.statista.com/statistics/1266392/ammonia-plant-capacities-united-states/ pointed to:
# https://nutrien-prod-asset.s3.us-east-2.amazonaws.com/s3fs-public/uploads/2023-11/Nutrien_2023Fact%20Book_Update_112723.pdf
# see (p. 13)
# corrected 'Fort Dodge OA to Fort Dodge IA'

from pathlib import PosixPath
from typing import Union

import click
import pandas as pd

from utils.location import city_lat_lon


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
def ammonia(input_file: Union[str, PosixPath], output_file: Union[str, PosixPath]):
    """Script to assign approximate lat/lon location to ammonia plants, using string city/state as inputs"""
    ammonia_df = pd.read_csv(input_file)
    ammonia_df["location"] = [
        city_lat_lon(city, state)
        for city, state in zip(ammonia_df.city, ammonia_df.state)
    ]
    ammonia_df[["latitude", "longitude"]] = pd.DataFrame(
        ammonia_df["location"].to_list(), index=ammonia_df.index
    )
    ammonia_df.to_csv(output_file)
    return ammonia_df


if __name__ == "__main__":
    ammonia()
