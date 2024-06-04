"""Script to assemble and run the Unit-Economics Simulator with three Rhodium Group (RHG)
Emissions Pathways/Scenarios, specified in a configuration yaml"""

from pathlib import Path, PosixPath
from typing import Union

import pandas as pd

from projects.ccs.ccs_project import CCSProject
from utils.io import yaml_to_dict


def rhg(config_path: Union[str, PosixPath]):
    """Converts daily prices to USD values in units of USD values for price_year. Computes rolling annual avg
    Args:
        input_file: location of unconverted pricing data
        output_file: location to write converted data
        price_year: price_year to which pricing data is converted (e.g., to convert to 2023 USD, use price_year=2023)
    Returns:
        price_df: converted price dataframe including rolling annual average price column
    """
    # Project parameters for RHG cases:

    config = yaml_to_dict(
        config_path,
    )
    # import costs data
    costs_df = pd.read_csv(config["costs_file"], index_col="industry")
    industries = list(costs_df.index)

    # import digitized RHG oil cases
    rhg_oil_cases_df = pd.read_csv(config["rhg_oil_cases_file"])

    # for each specified industry, run unit-economics simulator for each of the three
    # Rhodium scenarios (low, mid, and high Emissions pathways, as described in May 2024 report)
    scenario_list = []
    parameter_list = []
    for industry in industries:
        rhg_low_emissions_pathway_params = {
            "project_length_yrs": config["project_length_yrs"],
            "inflation_rate": config["low_inflation_rate"],
            "discount_rate": config["discount_rate"],
            "industry": industry,
            "tco2_sequestered_per_yr": config["tco2_sequestered_per_yr"],
            "oil_prices": list(rhg_oil_cases_df["high"]),
            "oil_breakeven_price": config["oil_breakeven_price"],
            "capture_cost_usd_per_tco2": costs_df.loc[
                industry, "capture_low_usd_per_tco2"
            ],
            "transport_cost_usd_per_tco2": costs_df.loc[
                industry, "transport_usd_per_tco2"
            ]
            * config["low_price_scalar"],  # transport_low,
            "storage_cost_usd_per_tco2": costs_df.loc[industry, "storage_usd_per_tco2"]
            * config["low_price_scalar"],
            "rhg_emissions_pathway": "low",
            "cost_method": "defined",
            "revenue_method": "computed",
        }

        rhg_mid_emissions_pathway_params = {
            "project_length_yrs": config["project_length_yrs"],
            "inflation_rate": config["low_inflation_rate"],
            "discount_rate": config["discount_rate"],
            "industry": industry,
            "tco2_sequestered_per_yr": config["tco2_sequestered_per_yr"],
            "oil_prices": list(rhg_oil_cases_df["mid"]),
            "oil_breakeven_price": config["oil_breakeven_price"],
            "capture_cost_usd_per_tco2": costs_df.loc[
                industry, "capture_center_usd_per_tco2"
            ],
            "transport_cost_usd_per_tco2": costs_df.loc[
                industry, "transport_usd_per_tco2"
            ],
            "storage_cost_usd_per_tco2": costs_df.loc[industry, "storage_usd_per_tco2"],
            "rhg_emissions_pathway": "mid",
            "cost_method": "defined",
            "revenue_method": "computed",
        }

        rhg_high_emissions_pathway_params = {
            "project_length_yrs": config["project_length_yrs"],
            "inflation_rate": config["high_inflation_rate"],
            "discount_rate": config["discount_rate"],
            "industry": industry,
            "tco2_sequestered_per_yr": config["tco2_sequestered_per_yr"],
            "oil_prices": list(rhg_oil_cases_df["low"]),
            "oil_breakeven_price": config["oil_breakeven_price"],
            "capture_cost_usd_per_tco2": costs_df.loc[
                industry, "capture_high_usd_per_tco2"
            ],
            "transport_cost_usd_per_tco2": costs_df.loc[
                industry, "transport_usd_per_tco2"
            ]
            * config["high_price_scalar"],  # transport_low,
            "storage_cost_usd_per_tco2": costs_df.loc[industry, "storage_usd_per_tco2"]
            * config["high_price_scalar"],
            "rhg_emissions_pathway": "high",
            "cost_method": "defined",
            "revenue_method": "computed",
        }
        parameter_list = parameter_list + [
            rhg_low_emissions_pathway_params,
            rhg_mid_emissions_pathway_params,
            rhg_high_emissions_pathway_params,
        ]

        for params in [
            rhg_low_emissions_pathway_params,
            rhg_mid_emissions_pathway_params,
            rhg_high_emissions_pathway_params,
        ]:
            row_dict = {}
            row_dict["industry"] = industry
            row_dict["rhg_emissions_pathway"] = params["rhg_emissions_pathway"]

            project = CCSProject(params)
            row_dict["total_eor_usd_per_tco2"] = project.total_eor_usd_per_tco2
            row_dict["total_gs_usd_per_tco2"] = project.total_gs_usd_per_tco2
            scenario_list.append(row_dict)

    scenarios_df = pd.DataFrame(scenario_list)
    input_parameters_df = pd.DataFrame(parameter_list)

    # Assemble published results from Fig 3 of May 2024 Rhodium report:
    # Record 2040 total MMtCO2 capacity for all industries
    rhodium_2040_df = (
        pd.DataFrame(
            {
                "low": [52, 13, 14, 26, 16, 11, 8, 3],
                "mid": [52, 13, 13, 8, 0, 0, 0, 0],
                "high": [50, 12, 10, 0, 0, 0, 0, 0],
            },
            index=[
                "Ethanol",
                "Ammonia",
                "NG Processing",
                "Refinery",
                "Hydrogen",
                "Ethylene",
                "Iron/Steel",
                "Cement",
            ],
        )
        .reset_index()
        .melt(
            id_vars="index",
            value_vars=["low", "mid", "high"],
            var_name="rhg_emissions_pathway",
            value_name="MMtCO2_per_yr",
        )
    )
    rhodium_2040_df.rename(columns={"index": "industry"}, inplace=True)

    # write data to output directory
    rhodium_2040_df.to_csv(
        Path(config["output_dir"])
        / "rhodium_2024_projections_total_ccs_capacity_by_industry_2040.csv"
    )
    scenarios_df.to_csv(
        Path(config["output_dir"])
        / "unit_economics_simulator_ccs_present_unit_value_by_industry_rhg_low_mid_high_scenarios.csv"
    )
    input_parameters_df.to_csv(
        Path(config["output_dir"])
        / "unit_economics_simulator_inputs_rhg_low_mid_high_scenarios.csv"
    )

    return rhodium_2040_df, scenarios_df, input_parameters_df


if __name__ == "__main__":
    rhg()
