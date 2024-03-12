"""Santa says this is a script for summing coal data from the EIA for years 1983-2022, by county, into big lumps"""
# Requires data to be in format available for historical data at  https://www.eia.gov/coal/data.php.
# User should save individual xls files to a local directory, whose path specified by 'data_dir' in the function below
from pathlib import Path, PosixPath
from typing import Union

import click
import pandas as pd

from utils.datasets import get_county_df, get_us_state_to_abbr_dict


@click.command()
@click.option(
    "--data_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
)
@click.option(
    "--output_file", type=click.Path(file_okay=True, dir_okay=False), required=True
)
@click.option("--start_yr")
@click.option("--end_yr")
def lump_coal(
    data_dir: Union[str, PosixPath],
    output_file: str,
    start_yr: int = 1983,
    end_yr: int = 2022,
):
    """imports, processes, and sums coal production data by county; writes/returns usable dataframe"""
    # get county/state name, fips codes in dataframe, state-to-abbreviation conversion dictionary
    county_df = get_county_df()
    state_abbr_dict = get_us_state_to_abbr_dict()

    df_list = []  # list for concatenation
    for y in range(int(start_yr), int(end_yr) + 1):
        print(f"Reading coal data for the year {y}")

        coal_df = pd.read_excel(Path(data_dir) / Path(f"coalpublic{y}.xls"))

        # column names are in row 2 of this dataset; name columns, get rid of empty rows
        coal_df.columns = coal_df.loc[2, :]
        coal_df = coal_df.loc[3:, :]
        coal_df.columns = [x.replace(" ", "_").lower() for x in coal_df.columns.values]

        # keep only the columns we'll use for summing (lumping!) the coal
        coal_df = coal_df[
            [
                "year",
                "mine_state",
                "mine_county",
                "mine_type",
                "mine_status",
                "production_(short_tons)",
            ]
        ]
        coal_df["production_short_tons"] = [
            int(x.replace(",", "")) if isinstance(x, str) else x
            for x in coal_df["production_(short_tons)"]
        ]
        coal_df["mine_state"] = [x.split(" (")[0] for x in coal_df["mine_state"]]

        # convert full state names to abbreviations for easier joining
        # pylin
        coal_df["state_abbr"] = [
            state_abbr_dict[x] if x in state_abbr_dict else ""
            for x in coal_df.mine_state
        ]
        coal_df = coal_df.merge(
            county_df[["fips", "state_abbr", "county_name"]],
            left_on=["state_abbr", "mine_county"],
            right_on=["state_abbr", "county_name"],
        )
        # put this year's data into a list for future concatenation
        df_list.append(
            coal_df[
                [
                    "year",
                    "fips",
                    "mine_state",
                    "state_abbr",
                    "mine_county",
                    "mine_type",
                    "production_short_tons",
                ]
            ]
            .groupby(
                [
                    "year",
                    "mine_state",
                    "state_abbr",
                    "mine_county",
                    "fips",
                    "mine_type",
                ]
            )
            .sum()
            .reset_index()
        )

    coal_df = pd.concat(df_list)

    # sum total coal production by fips and mine type (surface or underground)
    coal_df = (
        coal_df[
            ["fips", "state_abbr", "mine_county", "mine_type", "production_short_tons"]
        ]
        .groupby(["fips", "state_abbr", "mine_county", "mine_type"])
        .sum()
        .reset_index()
    )

    # make separate produciton columns for surface and underground production, then sum for total
    coal_df = coal_df.pivot(
        columns="mine_type", index=["fips", "state_abbr", "mine_county"]
    ).reset_index()
    coal_df.columns = [
        "fips",
        "state_abbr",
        "county",
        "surface_prod_short_tons",
        "underground_prod_short_tons",
    ]
    for v in ["surface_prod_short_tons", "underground_prod_short_tons"]:
        coal_df.fillna({v: 0}, inplace=True)
    coal_df["total_production_short_tons"] = [
        s + u
        for s, u in zip(
            coal_df.surface_prod_short_tons, coal_df.underground_prod_short_tons
        )
    ]

    # Write data to specified location
    coal_df.to_csv(Path(data_dir) / Path(output_file))

    return coal_df


if __name__ == "__main__":
    lump_coal()
