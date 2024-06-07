"""Script to convert price data to specific year's dollar values, using CPI"""

# Requires input data path to daily prices with columns as hard coded below ['Price','Date'].
# Built for converting Brent oil prices to 2023 USD. Data obtained at
# https://www.kaggle.com/datasets/mabusalah/brent-oil-prices
from pathlib import PosixPath
from typing import Union

import click
import cpi
import pandas as pd

from utils.time import convert_multiple_formats_to_datetime


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
@click.option("--price_year", type=click.STRING, required=True)
def adjust_prices(
    input_file: Union[str, PosixPath],
    output_file: Union[str, PosixPath],
    price_year=2023,
):
    """Converts daily prices to USD values in units of USD values for price_year. Computes rolling annual avg
    Args:
        input_file: location of unconverted pricing data
        output_file: location to write converted data
        price_year: price_year to which pricing data is converted (e.g., to convert to 2023 USD, use price_year=2023)
    Returns:
        price_df: converted price dataframe including rolling annual average price column
    """
    price_df = pd.read_csv(input_file)

    price_df["date"] = [
        convert_multiple_formats_to_datetime(x, formats=["%d-%b-%y", "%b %d, %Y"])
        for x in price_df["Date"]
    ]
    price_df["year"] = [int(d.year) for d in price_df["date"]]
    price_df["usd_cpi_" + str(price_year)] = [
        cpi.inflate(d, y, to=int(price_year))
        for d, y in zip(price_df["Price"], price_df["year"])
    ]
    # compute rolling annual mean and shift data so that annual rolling mean is centered on proper date
    price_df["rolling_annual_average_usd_per_unit"] = (
        price_df["usd_cpi_" + str(price_year)].rolling(365).mean().shift(-182)
    )
    price_df.to_csv(output_file)


if __name__ == "__main__":
    adjust_prices()
