""" script to process query output for RRC and OCD data"""

import pandas as pd
from calendar import isleap
import click
from typing import Union
from pathlib import PosixPath
import logging


logging.basicConfig(level=logging.INFO)


def days_in_month(month: int, year: int):
    if not isinstance(month, int):
        raise ValueError(" Input arguments for month and year must be integers.")
    if month == 2:
        if isleap(year):
            return 29
        return 28
    if month in [4, 6, 9, 11]:
        return 30
    return 31


@click.command()
@click.option(
    "--tx",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--nm",
    type=click.Path(file_okay=True, dir_okay=False),
    required=True,
)
def clean(tx: Union[str, PosixPath], nm: Union[str, PosixPath]):
    """cleans and processes query results from RRC and OCD"""
    tx_df = pd.read_csv(tx)
    nm_df = pd.read_csv(nm)

    # get rid of extra spaces
    tx_df.columns = [x.strip().lower() for x in tx_df.columns.values]
    nm_df.columns = [x.strip().lower() for x in nm_df.columns.values]

    for df in [tx_df, nm_df]:
        # make 'should be numeric' values numeric
        cols = [
            x for x in df.columns.values if x not in ["operator", "district", "state"]
        ]
        for c in cols:
            df[c] = [
                x if isinstance(x, (int, float)) else float(x.replace(",", ""))
                for x in df[c]
            ]
        # get number of days in month (for rates per day calculations)
        df["n_days"] = [days_in_month(m, y) for m, y in zip(df["month"], df["year"])]

    tx_df["state"] = "TX"
    nm_df["state"] = "NM"
    # compute total boe for both states
    tx_df["boe"] = tx_df[
        [
            "oil_bbl",
            "condensate_bbl",
            "casinghead_gas_boe",
            "gw_gas_boe",
        ]
    ].sum(axis=1)
    nm_df["boe"] = nm_df[
        [
            "oil_produced_bbl",
            "gas_produced_boe",
        ]
    ].sum(axis=1)

    # write out cleaned data
    tx_df.to_csv(
        "texas_rrc_district_level_oil_and_gas_production_by_month_and_operator.csv",
        index=None,
    )
    nm_df.to_csv(
        "nm_ocd_statewide_oil_and_gas_production_by_month_and_operator.csv", index=None
    )

    logging.info(
        " ----- Processed TX and NM data and wrote to csv ----- ",
    )


if __name__ == "__main__":
    clean()
