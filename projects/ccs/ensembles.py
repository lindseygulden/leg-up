"""Script that generates ensemble simluations of the Unit-Economics Simulator using
randomly sampled oil prices, capture/transport/storage costs, and oil breakeven price
"""

import datetime as dt

import numpy as np
import pandas as pd

from projects.ccs.ccs_project import CCSProject


def ues_ensemble(
    config: dict,
    costs_df: pd.DataFrame,
    brent_df: pd.DataFrame,
    breakeven_df: pd.DataFrame,
) -> pd.DataFrame:
    """Runs an ensemble of simulations of the UES, randomly sampling input parameters for each
    Args:
        config: parameter dictionary for simulation
        costs_df: costs for capture, storage, and transport of co2 -- must have industries as index
        brent_df: oil-price data from which to sample
        breakeven_df: data for breakeven price distributions
    Returns:
        scenarios_df: pd.DataFrame containing outputs of simulations
    """

    low_end_scalar = config["low_end_scalar"]
    high_end_scalar = config["high_end_scalar"]
    scenario_list = []

    for _ in range(config["nsamples"]):
        for brent_since_yr in config["brent_since_yrs"]:
            for well_type in config["well_types"]:
                for industry in config["industries"]:
                    params = {
                        "project_length_yrs": config["project_length_yrs"],
                        "inflation_rate": config["inflation_rate"],
                        "discount_rate": config["discount_rate"],
                        "industry": industry,
                        "tco2_sequestered_per_yr": config["tco2_sequestered_per_yr"],
                        "oil_prices": list(
                            brent_df.loc[brent_df["year"] >= brent_since_yr]
                            .dropna()
                            .rolling_annual_average_usd_per_unit.sample(
                                config["project_length_yrs"], replace=True
                            )
                        ),
                        "oil_breakeven_price": np.random.triangular(
                            left=breakeven_df.loc[well_type, "low"],
                            mode=breakeven_df.loc[well_type, "mid"],
                            right=breakeven_df.loc[well_type, "high"],
                        ),
                        "capture_cost_usd_per_tco2": np.random.triangular(
                            left=costs_df.loc[industry, "capture_low_usd_per_tco2"],
                            mode=costs_df.loc[industry, "capture_center_usd_per_tco2"],
                            right=costs_df.loc[industry, "capture_high_usd_per_tco2"],
                        ),
                        "transport_cost_usd_per_tco2": np.random.triangular(
                            left=costs_df.loc[industry, "transport_usd_per_tco2"]
                            * low_end_scalar,
                            mode=costs_df.loc[industry, "transport_usd_per_tco2"],
                            right=costs_df.loc[industry, "transport_usd_per_tco2"]
                            * high_end_scalar,
                        ),
                        "storage_cost_usd_per_tco2": np.random.triangular(
                            left=costs_df.loc[industry, "storage_usd_per_tco2"]
                            * low_end_scalar,
                            mode=costs_df.loc[industry, "storage_usd_per_tco2"],
                            right=costs_df.loc[industry, "storage_usd_per_tco2"]
                            * high_end_scalar,
                        ),
                        "scenario": config["scenario_name"],
                        "cost_method": "defined",
                        "revenue_method": "computed",
                    }
                    row_dict = {}
                    row_dict["industry"] = industry
                    row_dict["well_type"] = well_type
                    row_dict["brent_since_yr"] = brent_since_yr
                    row_dict["simulation_date"] = dt.date.today()
                    row_dict["mean_oil_price"] = np.mean(params["oil_prices"])
                    row_dict["oil_breakeven_price"] = params["oil_breakeven_price"]
                    row_dict["scenario"] = params["scenario"]

                    project = CCSProject(params)
                    row_dict["total_eor_usd_per_tco2"] = project.total_eor_usd_per_tco2
                    row_dict["total_gs_usd_per_tco2"] = project.total_gs_usd_per_tco2
                    scenario_list.append(row_dict)
    scenarios_df = pd.DataFrame(scenario_list)

    scenarios_df.to_csv(config["output_path"])

    return scenarios_df


if __name__ == "__main__":
    ues_ensemble()
