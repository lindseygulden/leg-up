"""bespoke script for compiling solar production, electricity rate, carbon intensity and PV array installation cost"""

from pathlib import Path

import pandas as pd

DATA_DIR = "/Volumes/Samsung_t5/data/"


def all_solar_data():
    """reads in data and assembles a processed version of it for use in economic modeling"""
    # build bespoke dataframe linking states to NPC storage regions from fig 2-9
    state_to_region_dict = (
        dict(zip(["MT", "ND", "SD", "NE", "UT", "CO", "WY"], ["north_central"] * 7))
        | dict(zip(["KS", "OK", "NM", "TXW"], ["south_central"] * 5))
        | dict(
            zip(["IL", "IN", "MI", "OH", "WV", "KY", "TN", "PA", "NY"], ["midwest"] * 9)
        )
        | dict(zip(["TXE", "LA", "MS", "AL", "FL", "AR"], ["gulf_coast"] * 7))
        | dict(zip(["CALA", "CASF", "CA"], ["california"] * 3))
    )
    state_region_df = pd.DataFrame(
        state_to_region_dict.items(), columns=["state", "region"]
    )

    # read in carbon-intensity and cents-per-kwh data for each state. Manually adjust for subset of states in CA and TX

    # get data obtained from the PVWatts (NREL) api for 6kw arrays
    solar_df = pd.read_csv(
        DATA_DIR / Path("ccs/regsions_solar_production_6kw_api_v8.csv"),
        index_col=[0],
    )

    # get electricity rate data
    rates_df = pd.read_csv(
        DATA_DIR / Path("ccs/residential_cents_per_kwh_by_state_march_2024_eia.csv")
    )
    rates_df.columns = ["name", "cents_per_kwh"]

    # get carbon-intensity data
    carbon_df = pd.read_csv(
        DATA_DIR
        / Path("ccs/us_state_power_sector_carbon_intensity_2023_carnegie_mellon.csv")
    )

    # get lat/lon data for states
    state_df = pd.read_csv(
        DATA_DIR / Path("us_geo/state_lat_lon_with_CA_TX_expansions.csv")
    )

    # get installation costs
    install_df = pd.read_csv(DATA_DIR / Path("ccs/6kw_solar_array_install_cost.csv"))
    # subset to key data columns for installation costs
    install_df = install_df[
        ["name", "avg_install_cost_2024_usd", "min_cost_2024_usd", "max_cost_2024_usd"]
    ]

    if "state" not in rates_df.columns:
        #  add electricity rate data for sub-state areas for TX and CA
        # per wattbuy.org, the cents/kwh across the state of texas is uniform
        rates_df = rates_df.merge(state_df[["state", "name"]], on="name")
        rates_df = pd.concat(
            [
                rates_df,
                pd.DataFrame(
                    {
                        "name": [
                            "California Los Angeles",
                            "California San Francisco",
                            "Texas West",
                            "Texas East",
                        ],
                        "state": ["CALA", "CASF", "TXW", "TXE"],
                        "cents_per_kwh": [29.1, 41.3, 14.9, 14.9],
                    }
                ),
            ]
        )

        # add carbon intensity data for sub-state areas for TX and CA
        # per wattbuy.org, TX has consistent carbon intensity data for all electricity sources
        carbon_df = carbon_df[["state", "tco2/kwh"]]
        carbon_df = pd.concat(
            [
                carbon_df,
                pd.DataFrame(
                    {
                        "state": ["CALA", "CASF", "TXW", "TXE"],
                        "tco2/kwh": [0.00025515, 0.0001899, 0.0003443180, 0.0003443180],
                    }
                ),
            ]
        )

        # Assume that installation cost ranges are consistent across CA and TX. Add data for sub-state areas
        tmp_records = (
            install_df.loc[install_df.name.isin(["California", "Texas"])].to_dict(
                "records"
            )
            * 2
        )
        tmp_records[0]["name"] = "California Los Angeles"
        tmp_records[1]["name"] = "Texas West"
        tmp_records[2]["name"] = "California San Francisco"
        tmp_records[3]["name"] = "Texas East"
        install_df = pd.concat([install_df, pd.DataFrame(tmp_records)])

    # Merge solar production, region identification, carbon-intensity, electricity-rate, and solar installation costs
    all_df = solar_df.merge(
        state_region_df.merge(carbon_df, on="state").merge(
            rates_df.merge(install_df, on="name"), on="state"
        ),
        on="state",
    )

    # extract only needed columns from array
    all_df = all_df[
        [
            "ac_annual",
            "lat",
            "lon",
            "state",
            "tilt",
            "azimuth",
            "pv_array_size_kw",
            "region",
            "tco2/kwh",
            "cents_per_kwh",
            "min_cost_2024_usd",
            "max_cost_2024_usd",
        ]
    ]

    # rename to be more descriptive
    all_df.columns = [
        "kwh_per_yr",
        "lat",
        "lon",
        "state",
        "tilt",
        "azimuth",
        "kw",
        "region",
        "tco2_per_kwh",
        "cents_per_kwh",
        "low_install_usd",
        "high_install_usd",
    ]

    # melt to make each row have data for exactly one RooftopSolarProject instance
    all_df = all_df.melt(
        id_vars=[
            "kwh_per_yr",
            "lat",
            "lon",
            "state",
            "tilt",
            "azimuth",
            "kw",
            "region",
            "tco2_per_kwh",
            "cents_per_kwh",
        ],
        value_vars=["low_install_usd", "high_install_usd"],
        value_name="installation_cost_usd",
        var_name="installation_cost_scenario",
    )
    # convert from cents to dollars
    all_df["usd_per_kwh"] = all_df["cents_per_kwh"] / 100
    all_df.drop("cents_per_kwh", inplace=True, axis=1)

    # write out
    all_df.to_csv(DATA_DIR / Path("ccs/6kw_solar_npc_states_3tilts_9azimuths.csv"))


if __name__ == "__main__":
    all_solar_data()
