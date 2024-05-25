"""WIP script to use fractracker wells data: replaced by ihs_well_denisty.py"""

from pathlib import PosixPath
from typing import Union

import click
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from utils.pandas import lowercase_columns


@click.command()
@click.option(
    "--texas_wells",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--no_tx_wells",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--counties",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
def og_wells(
    texas_wells: Union[str, PosixPath],
    no_tx_wells: Union[str, PosixPath],
    counties: Union[str, PosixPath],
):
    """imports and processes well location information; sums number of wells by US county"""
    # get wells data for Texas and no texas.
    tx_df = pd.read_csv(texas_wells)
    tx_df.rename(
        columns={"LATITUDE": "Latitude", "LONGITUDE": "Longitude"}, inplace=True
    )  # housekeeping
    no_tx_df = pd.read_csv(no_tx_wells, encoding="latin-1")
    wells_df = pd.concat([tx_df, no_tx_df])
    wells_df.columns = [x.lower() for x in wells_df.columns.values]

    # Turn well location data into a geodataframe
    geometry = [Point(xy) for xy in zip(wells_df.longitude, wells_df.latitude)]
    # df = df.drop(['Lon', 'Lat'], axis=1)
    wells_gdf = gpd.GeoDataFrame(wells_df, crs="EPSG:4326", geometry=geometry)

    # Open data soft's US county boundaries data; obtained at:
    # https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/us-county-boundaries/records?limit=-1'))
    counties_gdf = gpd.read_file(counties)
    counties_gdf = lowercase_columns(counties_gdf)
    counties_gdf["area_m2"] = [
        a + w for a, w in zip(counties_gdf.aland, counties_gdf.awater)
    ]
    counties_gdf["area_km2"] = counties_gdf.area_m2 / 1000000.0

    county_wells_gdf = counties_gdf.join(
        gpd.sjoin(wells_gdf, counties_gdf)
        .groupby("index_right")
        .size()
        .rename("n_wells"),
        how="left",
    )
    county_wells_gdf["n_wells_per_unit_area_km2"] = [
        n / a for n, a in zip(county_wells_gdf.n_wells, county_wells_gdf.area_km2)
    ]
    county_wells_gdf["n_wells_per_unit_land_area_km2"] = [
        n / (a / 1000000.0)
        for n, a in zip(county_wells_gdf.n_wells, county_wells_gdf.aland)
    ]
    county_wells_gdf["n_wells_per_unit_water_area_km2"] = [
        n / (a / 1000000.0) if a > 0 else 0
        for n, a in zip(county_wells_gdf.n_wells, county_wells_gdf.awater)
    ]


if __name__ == "__main__":
    og_wells()
