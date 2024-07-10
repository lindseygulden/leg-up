""" Command-line script to run cash flow analyses of rooftop solar using subclass defined in rooftop_solar_project.py"""

import logging

import click
import pandas as pd

from projects.ccs.rooftop_solar_project import RooftopSolarProject
from utils.io import yaml_to_dict

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
def rooftop_solar_ensemble(config):
    """Sets up and runs an ensemble of simulations of the carbon-emissions and cash flow of rooftop solar projects"""
    config = yaml_to_dict(config)
    all_df = pd.read_csv(config["solar_kwh_data_file"], index_col=[0])

    # subset to include specified cost-scenario for installation
    all_df = all_df.loc[
        all_df.installation_cost_scenario == config["which_install_scenario"]
    ]

    # import the time-varying carbon intensity
    time_varying_carbon_intensity_df = pd.read_csv(
        config["time_varying_carbon_intensity_file"], index_col=[0]
    )

    # specify shared parameters for all solar arrays
    shared_params = {}
    shared_params["inflation_rate"] = config["inflation_rate"]
    shared_params["discount_rate"] = config["discount_rate"]
    shared_params["discount_rate_real"] = (
        (1 + shared_params["discount_rate"]) / (1 + shared_params["inflation_rate"])
    ) - 1
    shared_params["tco2_per_kwh"] = list(
        time_varying_carbon_intensity_df["carbon intensity"]
    )
    shared_params["project_length_yrs"] = config["project_length_yrs"]
    shared_params["bond_interest_rate"] = config["bond_interest_rate"]

    for i, row in all_df.iterrows():
        solar_array = RooftopSolarProject(
            row[
                [
                    "kwh_per_yr",
                    "lat",
                    "lon",
                    "state",
                    "tilt",
                    "azimuth",
                    "kw",
                    "region",
                    "installation_cost_usd",
                    "usd_per_kwh",
                ]
            ].to_dict()
            | shared_params
        )
        all_df.at[i, "npv"] = solar_array.npv
        all_df.at[i, "total_co2"] = solar_array.total_tco2
        all_df.at[i, "pv_homeowner_expense_usd"] = solar_array.pv_homeowner_expense_usd
        all_df.at[
            i, "pv_homeowner_expense_usd_per_tco2"
        ] = solar_array.pv_homeowner_expense_usd_per_tco2
        all_df.at[i, "npv_homeowner_usd"] = solar_array.npv_homeowner_usd
        all_df.at[i, "pv_govt_expense_usd"] = solar_array.pv_govt_expense_usd
        all_df.at[
            i, "pv_govt_expense_usd_per_tco2"
        ] = solar_array.pv_govt_expense_usd_per_tco2
        all_df.at[i, "pv_solar_revenue_usd"] = solar_array.pv_solar_revenue_usd
        all_df.at[i, "pv_solar_usd_per_tco2"] = solar_array.pv_solar_usd_per_tco2
        all_df.at[
            i, "npv_homeowner_usd_per_tco2"
        ] = solar_array.npv_homeowner_usd_per_tco2
    # compute the fraction of scenarios in which the homeowner has > 0 npv
    all_df["npv_to_homeowner_gt_0"] = [x > 0 for x in all_df.npv_homeowner_usd]

    # write output
    all_df.to_csv(config["output_file"])


if __name__ == "__main__":
    rooftop_solar_ensemble()
