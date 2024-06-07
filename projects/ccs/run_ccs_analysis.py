"""Command-line script to run main components of analysis of RHG's 2024 CCS economics evaluation"""

import logging
from pathlib import PosixPath
from typing import Union

import click
import pandas as pd

from projects.ccs.ccs_costs import costs
from projects.ccs.ensembles import ues_ensemble
from projects.ccs.rhg_scenarios import rhg
from utils.io import yaml_to_dict

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option(
    "--rhg_config",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--costs_config",
    type=click.Path(file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--real_world_config",
    type=click.Path(file_okay=True, dir_okay=False),
    required=True,
)
def run_ccs_analysis(
    rhg_config: Union[str, PosixPath],
    costs_config: Union[str, PosixPath],
    real_world_config=Union[str, PosixPath],
):
    """Computes CCS costs by industry, runs RHG scenarios, builds probabilistic ensemble
    Args:
        rhg_config: path to yaml file containing config information for Rhodium
            Group (RHG) emissions scenarios ('Pathways')
        costs_config: path to yaml file containing config information for generating and
            updating lower and upper bounds for industry capture, transport, and storage
            costs
        real_world_config: path to yaml file containing config information for the running
            of the ensemble of UES simulations in which each simluation is given input
            parameters drawn randomly from realistic distributions
    Returns:
        None (although it does write out the results of the ensemble simulation!)
    """
    # compute costs for all industries
    logging.info("Constructing CCS cost data for all industries.")
    _, costs_df = costs(
        costs_config,
    )

    # Use UES to run the three rhodium scenarios (script writes out files to
    # locations in config file)
    logging.info(
        "Running Rhodium Group's Emissions scenarios through the Unit Economics Simulator"
    )
    _, _, _ = rhg(rhg_config)

    # Use UES to build an ensemble of CCS project value simulations using driving
    # assumptions that are randomly sampled from realistic, historically informed
    # probabiliity distributions
    realworld = yaml_to_dict(real_world_config)
    brent_df = pd.read_csv(realworld["path_to_brent_data"], index_col=[0])
    costs_df = pd.read_csv(realworld["path_to_cost_data"], index_col=[0])
    breakeven_df = pd.read_csv(realworld["path_to_breakeven_data"], index_col=[0])
    # run ensemble (no need to keep output: it's written to the location specified
    # by the real_world_config's 'output_path' key/value pair)
    logging.info(
        "Building an ensemble of UES simulations with randomly sampled real-world data."
    )
    _ = ues_ensemble(realworld, costs_df, brent_df, breakeven_df)
    logging.info("CCS analysis complete.")


if __name__ == "__main__":
    run_ccs_analysis()
