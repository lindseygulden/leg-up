"""Functions to extract, process, and update costs for carbon capture, transport, and storage"""

from pathlib import Path, PosixPath
from typing import List, Tuple, Union

import geopandas as gpd
import pandas as pd
from geopy.distance import geodesic

from utils.io import yaml_to_dict


def get_facility_locations(
    configuration: Union[dict, Union[str, PosixPath]]
) -> gpd.GeoDataFrame:
    """Gets lat/lon locations for potential candidate-location facilities for CCS, by industry"""
    if isinstance(configuration, dict):
        config = configuration
    else:
        config = yaml_to_dict(configuration)
    # get locations w/ iron/steel & hydrogen operations from EPA's GHG reporting/Flight database summary
    expand_dict = config["epa_flight_abbr"]
    flight_df = pd.read_csv(config["epa_flight_summary_data"])
    flight_df.columns = [x.lower() for x in flight_df.columns.values]

    # convert letter jumbles to interpretable/more easily search
    flight_df["subparts_explained"] = (
        flight_df["subparts"].copy(deep=True).replace(",", " ")
    )
    for k, v in expand_dict.items():
        flight_df["subparts_explained"] = [
            x.replace(k, v) for x in flight_df["subparts_explained"]
        ]

    # subset/extract locations with iron/steel and hydrogen operations
    steel_df = flight_df.loc[["iron_steel" in x for x in flight_df.subparts_explained]]
    hydrogen_df = flight_df.loc[["hydrogen" in x for x in flight_df.subparts_explained]]

    # additional industries taken from EIA information and industry-specific info (for cement and ammonia)
    refineries_df = gpd.read_file(config["eia_refineries"])
    refineries_df.columns = [x.lower() for x in refineries_df.columns.values]

    ethylene_df = gpd.read_file(config["ethylene"])
    ethylene_df.columns = [x.lower() for x in ethylene_df.columns.values]

    ethanol_df = gpd.read_file(config["ethanol"])
    ethanol_df.columns = [x.lower() for x in ethanol_df.columns.values]

    ngp_df = gpd.read_file(config["ngp"])
    ngp_df.columns = [x.lower() for x in ngp_df.columns.values]

    ammonia_df = pd.read_csv(config["ammonia"], index_col=[0])
    cement_df = pd.read_csv(config["cement"])

    # Assemble all industries into a single df and corresponding geodataframe
    keep_columns = ["industry", "latitude", "longitude"]
    all_data = []
    for name, data_df in zip(
        [
            "Refinery",
            "Ethylene",
            "Ethanol",
            "NG Processing",
            "Ammonia",
            "Cement",
            "Hydrogen",
            "Iron/Steel",
        ],
        [
            refineries_df,
            ethylene_df,
            ethanol_df,
            ngp_df,
            ammonia_df,
            cement_df,
            hydrogen_df,
            steel_df,
        ],
    ):
        data_df["industry"] = name
        all_data.append(data_df[keep_columns].copy(deep=True))

    all_industries_df = pd.DataFrame(pd.concat(all_data))
    all_industries_df.dropna(inplace=True)  # get rid of the nans
    all_industries_gdf = gpd.GeoDataFrame(
        all_industries_df,
        geometry=gpd.points_from_xy(
            all_industries_df.longitude, all_industries_df.latitude
        ),
        crs="EPSG:4326",
    )

    return all_industries_gdf


def compute_distances(
    loc_gdf: gpd.GeoDataFrame, storage_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Computes distances between each facility location and all storage regions; finds closest region"""

    # compute distance between facilities and storage regions, in km
    regions = list(storage_gdf.index)
    for region in regions:
        regional_point = (
            storage_gdf.loc[region, "lat"],
            storage_gdf.loc[region, "lon"],
        )
        loc_gdf["km_to_" + region] = [
            geodesic(regional_point, (lat, lon)).km
            for lat, lon in zip(loc_gdf.latitude, loc_gdf.longitude)
        ]
    distance_cols = ["km_to_" + region for region in regions]

    # identify the closest storage location and the distance to that location
    loc_gdf["min_dist"] = loc_gdf[distance_cols].min(axis=1)
    loc_gdf["storage_region"] = loc_gdf[distance_cols].idxmin(axis=1)
    # fix name of storage region by removing 'km_to_' prefix
    loc_gdf["storage_region"] = [
        x.replace("km_to_", "") for x in loc_gdf["storage_region"]
    ]
    loc_gdf = loc_gdf[
        ["latitude", "longitude", "storage_region", "min_dist", "industry", "geometry"]
    ]
    loc_gdf.rename(columns={"min_dist": "dist_to_storage_km"}, inplace=True)
    return loc_gdf


def get_capture_data(
    excel_path: Union[str, PosixPath],
    excel_sheet: str,
    row_name: str,
    industries: List[str],
) -> pd.DataFrame:
    """Extracts capture data, by industry, from outputs of GaffneyCline capture cost cash flow model"""
    capture_df = pd.read_excel(
        excel_path,
        excel_sheet,
        index_col=[0],
    )

    capture = []
    for industry in industries:
        row_dict = {}
        row_dict["industry"] = industry
        row_dict["capture_low_usd_per_tco2"] = capture_df.loc[
            row_name, industry + "_low"
        ]
        row_dict["capture_high_usd_per_tco2"] = capture_df.loc[
            row_name, industry + "_high"
        ]
        # compute data for triangular distribution sampling
        row_dict["capture_center_usd_per_tco2"] = (
            row_dict["capture_low_usd_per_tco2"] + row_dict["capture_high_usd_per_tco2"]
        ) / 2
        row_dict["capture_center_shifted_high_usd_per_tco2"] = 0.75 * (
            row_dict["capture_low_usd_per_tco2"] + row_dict["capture_high_usd_per_tco2"]
        )
        row_dict["capture_center_shifted_low_usd_per_tco2"] = 0.25 * (
            row_dict["capture_low_usd_per_tco2"] + row_dict["capture_high_usd_per_tco2"]
        )
        capture.append(row_dict)

    capture_df = pd.DataFrame(capture)

    # fix column names
    capture_df["industry"] = [x.title() for x in capture_df["industry"]]

    for k, v in {
        "Steel": "Iron/Steel",
        "Ngp": "NG Processing",
        "Industrial": "Ethylene",
    }.items():
        capture_df["industry"] = [x.replace(k, v) for x in capture_df["industry"]]

    capture_df.set_index("industry", inplace=True)

    return capture_df


def get_storage_info(
    filepath: Union[str, PosixPath], cost_scalar: float = None
) -> gpd.GeoDataFrame:
    """Imports storage region location/avg. cost data and converts to geodataframe"""
    storage_df = pd.read_csv(filepath)
    storage_df.dropna(inplace=True)  # get rid of extra lines
    storage_df.columns = [x.lower() for x in storage_df.columns.values]
    if cost_scalar is not None:
        storage_df["storage_cost_usd_per_tco2"] = (
            storage_df["storage_cost_usd_per_tco2"] * cost_scalar
        )
    storage_gdf = gpd.GeoDataFrame(
        storage_df,
        geometry=gpd.points_from_xy(storage_df.lon, storage_df.lat),
        crs="EPSG:4326",
    )
    storage_gdf.set_index("region", inplace=True)
    return storage_gdf


def compute_industry_transport_costs(
    all_industries_gdf: gpd.GeoDataFrame,
    transport_cost_range: List[float],
    upper_bound_percentile=0.975,
) -> pd.DataFrame:
    """Computes average carbon transport cost, by industry"""
    # for each industry, find the mean distance to the closest storage region
    transport_df = (
        all_industries_gdf[["industry", "dist_to_storage_km"]]
        .groupby("industry")
        .mean()
        .sort_values(by="dist_to_storage_km")
    )
    transport_df.columns = ["mean_distance_to_storage_region"]

    # treat lower bound of transport distance as 0 km. Unless specified, use 97.5 pctile as upper bound for scaling
    upper_bound = transport_df["mean_distance_to_storage_region"].quantile(
        upper_bound_percentile
    )

    # map the distance to storage region to
    transport_df["transport_usd_per_tco2"] = [
        transport_cost_range[0] + (x / upper_bound) * transport_cost_range[1]
        for x in transport_df.mean_distance_to_storage_region
    ]
    return transport_df


def compute_industry_storage_costs(
    all_industries_gdf: gpd.GeoDataFrame, storage_gdf: gpd.GeoDataFrame
):
    """finds facility-count-weighted mean storage cost for each industry"""

    # Find the count of facilities routing to each storage region, grouped by industry
    cost_df = (
        all_industries_gdf[["industry", "storage_region", "latitude"]]
        .groupby(["industry", "storage_region"])
        .count()
        .reset_index()
    )
    cost_df.columns = ["industry", "storage_region", "facility_count"]

    # find the intermediate product for computing the facility-count-weighted mean
    cost_df = cost_df.merge(storage_gdf, left_on="storage_region", right_index=True)
    cost_df["facility_count_times_usd_per_tco2"] = (
        cost_df["facility_count"] * cost_df["storage_cost_usd_per_tco2"]
    )
    # sum both the facility count and the product of facility count and regional storage cost
    cost_df = (
        cost_df[["facility_count", "facility_count_times_usd_per_tco2", "industry"]]
        .groupby("industry")
        .sum()
    )
    # use two sums to compute mean storage cost for each industry
    cost_df["storage_usd_per_tco2"] = (
        cost_df["facility_count_times_usd_per_tco2"] / cost_df["facility_count"]
    )
    return pd.DataFrame(cost_df["storage_usd_per_tco2"])


def costs(
    configuration: Union[str, PosixPath]
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Converts daily prices to USD values in units of USD values for price_year. Computes rolling annual avg
    Args:
        configuration: path to Yaml configuration file, which contains file locations, industry names, etc.
        output_dir: Optional: if defined, function will write out final dataframes to directory
    Returns:
        all_locations_gdf: geodataframe with all locations of facilities considered for transport
            and storage cost estimates
        costs_by_industry_df: pandas dataframe containing cost information, updated for 2023 dollars
            with RHG assumptions, for capture, transport, and storage by industry
    """

    # read configuration file
    config = yaml_to_dict(configuration)

    # get locations
    all_locations_gdf = get_facility_locations(config)

    # get storage region centroids and costs; adjust costs from 2021 dollars to 2023 dollars
    storage_gdf = get_storage_info(config["npc_storage_region_info"], cost_scalar=1.12)

    # compute distances between all facilities and storage centers: find closest storage region for each facility
    all_locations_gdf = compute_distances(all_locations_gdf, storage_gdf)

    # compute mean transport costs by industry
    transport_df = compute_industry_transport_costs(all_locations_gdf, [2.2, 42.5])

    # compute mean storage costs by industry
    storage_df = compute_industry_storage_costs(all_locations_gdf, storage_gdf)

    # extract capture-cost data from gaffney-cline model outputs
    capture_df = get_capture_data(
        config["capture_excel_file"],
        config["capture_assumptions_sheet_name"],
        config["capture_unit_cost_row_name"],
        config["industries"],
    )

    # combine costs for capture, transport, and storage (by industry) into a single dataframe
    costs_by_industry_df = capture_df.merge(
        transport_df, right_index=True, left_index=True
    ).merge(storage_df, right_index=True, left_index=True)

    all_locations_gdf.to_file(
        Path(config["output_dir"]) / "all_industry_facility_locations.geojson",
        driver="GeoJSON",
    )
    costs_by_industry_df.to_csv(
        Path(config["output_dir"]) / "ccs_costs_by_industry.csv"
    )
    return all_locations_gdf, costs_by_industry_df


if __name__ == "__main__":
    costs()
