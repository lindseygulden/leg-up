"""Script to  compute drinking water source percentages for each US county"""
# Requires input data in the format exported from this EPA website:
# https://ordspub.epa.gov/ords/sfdw_rest/r/sfdw/sdwis_fed_reports_public/103
# User should save individual xls file to a local directory, whose path specified by 'input_file' in the
# command-line function below
from pathlib import Path, PosixPath
from typing import Union

import click
import pandas as pd

from utils.datasets import get_county_df


@click.command()
@click.option(
    "--input_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
def water_source(
    input_file: Union[str, PosixPath],
):
    """imports, processes, and sums coal production data by county; writes/returns usable dataframe"""
    # get county/state name, fips codes in dataframe, state-to-abbreviation conversion dictionary
    county_info_df = get_county_df()

    # File should be save locally to the path specified by 'input_file'
    water_df = pd.read_excel(input_file)

    # get rid of first few columns (without data), and assign column names
    water_df.columns = water_df.loc[3, :]
    water_df = water_df.loc[4:, :]
    water_df.columns = [x.replace(" ", "_").lower() for x in water_df.columns.values]

    # Original county field has two counties. For these purposes, we're keeping the first and
    # ignoring the second member of the list, which is usually the same as the first
    water_df["county"] = [
        x.split(",")[0] if isinstance(x, str) else "" for x in water_df.counties_served
    ]
    water_df["county2"] = [
        x.split(",")[1] if (isinstance(x, str)) and ("," in x) else ""
        for x in water_df.counties_served
    ]

    # Use state name and first county to link water sources to FIPS numbers
    water_df = water_df.merge(
        county_info_df[["fips", "state", "county_name"]],
        left_on=["primacy_agency", "county"],
        right_on=["state", "county_name"],
        how="left",
    )
    # Housekeeping
    water_df = water_df[
        [
            "pws_id",
            "state",
            "fips",
            "county",
            "primary_source",
            "is_wholesaler",
            "population<br>_served_count",
        ]
    ]
    water_df.rename(
        columns={"population<br>_served_count": "population_served"}, inplace=True
    )
    water_df.dropna(inplace=True)

    # sum total number of people served, grouping by FIPS and by primary source
    water_df = (
        water_df[["state", "county", "fips", "primary_source", "population_served"]]
        .groupby(["state", "county", "fips", "primary_source"])
        .sum()
        .reset_index()
    )

    # separate number of poeple served by primary source type (groundwater, surface water, combination, etc.)
    water_df = (
        water_df.pivot(index=["state", "county", "fips"], columns="primary_source")
        .reset_index()
        .fillna(0)
    )

    # more housekeeping/column-name streamlining
    water_df.columns = [x[0] if x[1] == "" else x[1] for x in water_df.columns.values]
    water_df.columns = [x.replace(" ", "_").lower() for x in water_df.columns.values]

    # find total number of people served by each county
    water_df["population_served"] = water_df[
        [
            "ground_water",
            "ground_water_purchased",
            "groundwater_under_influence_of_surface_water",
            "purchased_ground_water_under_influence_of_surface_water_source",
            "surface_water",
            "surface_water_purchased",
            "unknown_primary_source",
        ]
    ].sum(axis=1)

    # slice and dice/group the data in a few more ways
    water_df["any_ground_water"] = water_df[
        ["ground_water", "ground_water_purchased"]
    ].sum(axis=1)
    water_df["any_surface_water"] = water_df[
        ["surface_water", "surface_water_purchased"]
    ].sum(axis=1)
    water_df["any_groundwater_under_surface_water_influence"] = water_df[
        [
            "groundwater_under_influence_of_surface_water",
            "purchased_ground_water_under_influence_of_surface_water_source",
        ]
    ].sum(axis=1)
    water_df["local_water"] = water_df[
        [
            "ground_water",
            "groundwater_under_influence_of_surface_water",
            "surface_water",
        ]
    ].sum(axis=1)
    water_df["purchased_water"] = water_df[
        [
            "ground_water_purchased",
            "purchased_ground_water_under_influence_of_surface_water_source",
            "surface_water_purchased",
        ]
    ].sum(axis=1)

    # compute percentages of county population for each type of drinking water/designation
    for v in [
        "ground_water",
        "ground_water_purchased",
        "groundwater_under_influence_of_surface_water",
        "purchased_ground_water_under_influence_of_surface_water_source",
        "surface_water",
        "unknown_primary_source",
        "surface_water_purchased",
        "any_ground_water",
        "any_surface_water",
        "any_groundwater_under_surface_water_influence",
        "local_water",
        "purchased_water",
    ]:
        water_df["pct_population_" + v] = [
            100 * x / t for x, t in zip(water_df[v], water_df["population_served"])
        ]

    # get rid of duplicative columns
    water_df = water_df[
        [
            "state",
            "county",
            "fips",
            "population_served",
            "pct_population_ground_water",
            "pct_population_ground_water_purchased",
            "pct_population_groundwater_under_influence_of_surface_water",
            "pct_population_purchased_ground_water_under_influence_of_surface_water_source",
            "pct_population_surface_water",
            "pct_population_surface_water_purchased",
            "pct_population_unknown_primary_source",
            "pct_population_any_ground_water",
            "pct_population_any_surface_water",
            "pct_population_any_groundwater_under_surface_water_influence",
            "pct_population_local_water",
            "pct_population_purchased_water",
        ]
    ]

    # write the processed file to the same directory
    water_df.to_csv(
        Path(input_file.rsplit("/", 1)[0])
        / Path("us_drinking_water_source_by_county.csv")
    )
    return water_df


if __name__ == "__main__":
    water_source()
