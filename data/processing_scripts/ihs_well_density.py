""" Pulls in USGS's distillation of US historical oil & gas wells; processes by county"""

# import fiona # fiona is under geopandas: use it to see layers in a gdb (search 'fiona' below)
import logging
from pathlib import Path, PosixPath
from typing import Union

import click
import geopandas as gpd
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
from utils.pandas import lowercase_columns


@click.command()
@click.option(
    "--data_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
)
@click.option(
    "--output_file", type=click.Path(file_okay=True, dir_okay=False), required=True
)
def usgs_ihs_well_data(
    data_dir: Union[str, PosixPath], output_file: Union[str, PosixPath]
):
    """Reads raw .gdb file from USGS, lumps well data by year and county, computes intensity, writes out"""

    logging.info(" --- Processing USGS well-density data ---")
    # Open data soft's US county boundaries data; https://public.opendatasoft.com.geojson'))
    counties_gdf = gpd.read_file(
        Path(data_dir) / Path("us_geo/us-county-boundaries.geojson")
    )
    counties_gdf = lowercase_columns(counties_gdf)
    counties_gdf["area_m2"] = [
        a + w for a, w in zip(counties_gdf.aland, counties_gdf.awater)
    ]
    counties_gdf["area_km2"] = counties_gdf.area_m2 / 1000000.0
    logging.info(" County geojson read and processed")
    # import Aggregated Oil and Natural Gas Drilling and Production History of the United States (ver. 1.1, April 2023)
    # https://doi.org/10.5066/P9UIR5HE. From USGS scientists, source data is IHS Markit

    # fiona.listlayers('/Volumes/Samsung_T5/data/oilgas/USOilGasAggregation.gdb')
    # other layers in the geodatabase are:
    #        ['USWells_10Mile',
    #        'USWells_1Mile',
    #        'USWells_10Mile_1Year',
    #        'USWells_1Mile_1Year',
    #        'USProduction_2Mile_1Year',
    #        'USProduction_10Mile_1Year',
    #        'USProduction_2Mile',
    #        'USProduction_10Mile']
    ihs_markit_wells_gdf = gpd.read_file(
        Path(data_dir) / Path("oilgas/USOilGasAggregation.gdb"),
        layer="USWells_1Mile_1Year",
    )
    ihs_markit_wells_gdf.columns = [
        x.lower().replace(" ", "_") for x in ihs_markit_wells_gdf.columns.values
    ]

    # convert ihs projection to counties projection
    ihs_markit_wells_gdf.to_crs(counties_gdf.crs, inplace=True)

    # Assign a unique ID to each grid/year
    ihs_markit_wells_gdf.reset_index(inplace=True)
    ihs_markit_wells_gdf.rename(columns={"index": "grid_id"}, inplace=True)
    logging.info(
        "USGS IHS Markit summary information read, processed, and re-projected"
    )

    # spatial join between US counties and the 1x1 mi
    ihs_join_counties_gdf = ihs_markit_wells_gdf.sjoin(
        counties_gdf, how="left", predicate="intersects"
    )

    ihs_join_counties_gdf["intersection_area"] = ihs_join_counties_gdf.geometry.area
    # order polygons by grid-year id ('grid_id') and area, in descending size order
    ihs_join_counties_gdf.sort_values(
        by=["grid_id", "intersection_area"], ascending=False, inplace=True
    )
    # Assign each 1x1-mi grid cell to one and only one county (to prevent double counting)
    ihs_join_counties_gdf.drop_duplicates(
        keep="first", subset=["grid_id", "year"], inplace=True
    )
    logging.info(" USGS data binned into US counties, by year")

    # I get a bizarre error when I try to write this to geojson with Geopandas 'to_file': this is a workaround
    ihs_join_counties_gdf["wkt"] = ihs_join_counties_gdf.geometry.to_wkt()
    ihs_join_counties_df = pd.DataFrame(
        ihs_join_counties_gdf[
            [x for x in ihs_join_counties_gdf.columns.values if x != "geometry"]
        ]
    )
    ihs_join_counties_df.to_csv(
        data_dir
        / Path("oilgas/processed_data/us_counties_joined_with_ihs_markit_1mi.csv")
    )
    logging.info(
        "USGS county-level data written to data_dir + oilgas/processed_data/us_counties_joined_with_ihs_markit_1mi.csv"
    )

    # names of the columns in the IHS data that count different types of wells
    well_count_cols = [
        "count_well",
        "count_oil_well",
        "count_gas_well",
        "count_dry_well",
        "count_injection_well",
        "count_horizontal_well",
        "count_fractured_well",
    ]

    # use well-count & year columns to compute the mean & median year of O&G development for each county
    yr_times_well_count_cols = []
    for c in well_count_cols:
        ihs_join_counties_df[f"yr_times_{c}"] = [
            w * y for w, y in zip(ihs_join_counties_df[c], ihs_join_counties_df["year"])
        ]
        ihs_join_counties_df[f"yr_times_{c}"].replace(
            0, np.nan, inplace=True
        )  # to avoid screwing up the median and mean computations
        yr_times_well_count_cols.append(f"yr_times_{c}")

    # compute median year of oil and gas drilling by county
    ihs_median_year_df = (
        ihs_join_counties_df[yr_times_well_count_cols + ["geoid"]]
        .groupby("geoid")
        .median()
        .reset_index()
    )
    ihs_median_year_df.rename(
        columns={
            c: "median_" + c.replace("times_count_", "")
            for c in yr_times_well_count_cols
        },
        inplace=True,
    )
    # in two steps, compute mean year of oil and gas drilling
    ihs_mean_year_df = (
        ihs_join_counties_df[yr_times_well_count_cols + well_count_cols + ["geoid"]]
        .groupby("geoid")
        .sum()
        .reset_index()
    )
    for c in well_count_cols:
        name_shortened = c.replace("count_", "")
        ihs_mean_year_df[f"mean_year_{name_shortened}"] = [
            n / d if d > 0 else 0
            for n, d in zip(ihs_mean_year_df[f"yr_times_{c}"], ihs_mean_year_df[f"{c}"])
        ]
        ihs_mean_year_df[f"mean_year_{name_shortened}"].replace(0, np.nan, inplace=True)

    # omit columns needed only for calculation
    ihs_mean_year_df.drop(yr_times_well_count_cols, axis=1, inplace=True)

    # merge mean and median year calculations into a single dataframe
    ihs_df = ihs_mean_year_df.merge(ihs_median_year_df, on="geoid")
    logging.info(
        "Finished computing mean and median year of drilling activity in each county"
    )
    # link county-specific oil and gas drilling well timing & count data to county geographic info
    ihs_fips_stats_gdf = (
        counties_gdf[["geoid", "name", "state_name", "area_km2", "geometry"]]
        .merge(ihs_df, how="left", on="geoid")
        .fillna(0)
    )

    # compute well inten
    for c in well_count_cols:
        name_shortened = c.replace("count_", "")
        ihs_fips_stats_gdf[f"{c}_per_km2"] = [
            c / a for c, a in zip(ihs_fips_stats_gdf[c], ihs_fips_stats_gdf["area_km2"])
        ]
        ihs_fips_stats_gdf[f"mean_exposure_years_{name_shortened}"] = (
            2024 - ihs_fips_stats_gdf[f"mean_year_{name_shortened}"]
        )
        ihs_fips_stats_gdf[f"median_exposure_years_{name_shortened}"] = (
            2024 - ihs_fips_stats_gdf[f"median_yr_{name_shortened}"]
        )
        ihs_fips_stats_gdf[f"intensity_{name_shortened}_yr_per_km2"] = [
            w * y
            for w, y in zip(
                ihs_fips_stats_gdf[f"{c}_per_km2"],
                ihs_fips_stats_gdf[f"mean_exposure_years_{name_shortened}"],
            )
        ]
    logging.info(" Finished computing drilling intensity for each county")
    pd.DataFrame(ihs_fips_stats_gdf).to_csv(
        Path(data_dir) / Path(f"oilgas/processed_data/{output_file}")
    )
    logging.info(
        "Output written to %s/oilgas/processed_data/%s", str(data_dir), output_file
    )
    return ihs_fips_stats_gdf


if __name__ == "__main__":
    usgs_ihs_well_data()
