"""This command-line script runs a sensitivity analysis with mini_iam

To use:

> python3 [path/to/this/file] --config [path/to/sensitivity_config.yml]

"""

import logging
import os
from pathlib import Path
from typing import List

import click
import numpy as np
import pandas as pd
from SALib.analyze import sobol
from SALib.sample import saltelli
from scipy.stats import qmc

from projects.iam.eslim import NestedLogitIAM
from utils.io import yaml_to_dict

logging.basicConfig(level=logging.INFO)


def set_nested_value(d: dict, keys: List[str], value):
    """Sets a value in a nested dictionary using a list of keys
    Args:
        d: dictionary to be modified
        keys: list of string-valued dictionary keys to follow
        value: value to set at path
    Returns:
        d: updated dictionary
    """
    current = d
    for key in keys[:-1]:  # go down to the second-to-last key
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def update_parameters(
    config: dict, list_of_pars_to_vary: List[dict], new_pars: pd.Series
):
    """updates values in nested dictionary according to the specified key path
    Args:
        config: dictionary whose values will be updated
        list_of_pars_to_vary: List of dictionaries that specify par name & key path for
            config dict
        new_pars: parameter sample for this instance; indices are parameter names
            matching 'name' field in each pars_to_vary
    Returns:
        None: updates config in memory
    """
    # new_pars = [float(x) for x in list(new_pars)]
    if len(new_pars) != len(list_of_pars_to_vary):
        raise ValueError(
            f""" In update_parameters, length key_paths {len(list_of_pars_to_vary)}
            must be same length as X ({len(new_pars)})"""
        )
    for pars_to_vary in list_of_pars_to_vary:
        if len(pars_to_vary["key_path"]) == 1:
            config[pars_to_vary["key_path"][0]] = new_pars.loc[pars_to_vary["name"]]
        else:
            set_nested_value(
                config, pars_to_vary["key_path"], new_pars.loc[pars_to_vary["name"]]
            )


def lhs(num_samples: int, param_bounds: List[List[float]]):
    """gets latin hypercube samples for parameter space
    Args:
        num_samples: number of samples to generate
        param_bounds: a list of ranges from which to sample (currently only does uniform dist)
    Returns:
        pandas dataframe containing scaled results for use in model"""
    # generate latin hypercube samples in [0, 1]
    sampler = qmc.LatinHypercube(d=len(param_bounds))
    lhs_unit = sampler.random(n=num_samples)

    # scale to parameter bounds; return to user
    return pd.DataFrame(
        qmc.scale(
            lhs_unit,
            [b[0] for b in param_bounds],
            [b[1] for b in param_bounds],
        )
    )


def saltelli_sample(
    num_samples: int, problem_definition: dict, calc_second_order: bool
) -> pd.DataFrame:
    """Uses sobol sampling to get a set of parameters for model sensitivity analysis"""

    # generate samples, return
    try:
        pars = saltelli.sample(
            problem_definition, N=num_samples, calc_second_order=calc_second_order
        )
        return pd.DataFrame(pars)
    except Exception as e:
        raise RuntimeError(" Sobol sampling failed ") from e


def si_to_dataframes(
    si_matrix: np.array, param_names: List[str], calc_second_order: bool = False
):
    """Convert SALib sobol Si matrix and parameter names into dfs."""
    n = len(param_names)

    # first-order saltelli indices and confidence outputs
    s1_df = pd.DataFrame(
        {"S1": si_matrix["S1"], "S1_conf": si_matrix.get("S1_conf", [np.nan] * n)},
        index=param_names,
    )
    s1_df.index.name = "parameter"
    s1_df.reset_index(inplace=True)

    # total-order indices and confidence outputs
    s_total_df = pd.DataFrame(
        {"ST": si_matrix["ST"], "ST_conf": si_matrix.get("ST_conf", [np.nan] * n)},
        index=param_names,
    )
    s_total_df.index.name = "parameter"
    s_total_df.reset_index(inplace=True)

    if calc_second_order:
        # Second-order indices
        s2_flat = si_matrix.get("S2", None)

        if s2_flat is None:
            s2_df = pd.DataFrame(np.nan, index=param_names, columns=param_names)

        else:
            s2_df = pd.DataFrame(s2_flat, index=param_names, columns=param_names)
            s2_df.fillna(0, inplace=True)

        s2_df.index.name = "parameter"
        s2_df.reset_index(inplace=True)
        return s1_df, s_total_df, s2_df
    return s1_df, s_total_df, pd.DataFrame()


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
def sensitivity(config: str):
    """Implements model sensitivity analysis for the mini IAM"""

    # get configuration for sensitivity analysis
    config_info = yaml_to_dict(config)

    # prep output directory
    output_dir = Path(".")
    if "output_dir" in config_info:
        output_dir = Path(config_info["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    if config_info["sampler"].lower() == "saltelli":
        logging.info(" Generating Saltelli Sobol parameter samples")

        # define problem
        problem_definition = {
            "num_vars": len(config_info["pars_to_vary"]),
            "names": [p["name"] for p in config_info["pars_to_vary"]],
            "bounds": [p["bounds"] for p in config_info["pars_to_vary"]],
        }
        parameter_samples_df = saltelli_sample(
            config_info["num_samples"],
            problem_definition,
            config_info["calc_second_order"],
        )
    else:  # elif config_info["sampler"].lower() == "lhs":
        logging.info(" Generating Latin hypercube parameter samples")
        parameter_samples_df = lhs(
            config_info["num_samples"],
            [p["bounds"] for p in config_info["pars_to_vary"]],
        )

    # label parameter columns in  dataframe
    parameter_names = [p["name"] for p in config_info["pars_to_vary"]]
    parameter_samples_df.columns = parameter_names

    # read baseline configuration for running the model
    baseline_config = yaml_to_dict(config_info["baseline_model_config"])

    # initialize list for storing outputs
    dflist = []
    logging.info("There are a total of %s instances to run.", len(parameter_samples_df))

    # iterate through instances
    for which_instance, instance_parameter_values in parameter_samples_df.iterrows():
        if which_instance % 1000 == 0:
            logging.info(
                "completed %s instances of %s",
                which_instance,
                len(parameter_samples_df),
            )
        instance_config = baseline_config.copy()
        update_parameters(
            instance_config,
            config_info["pars_to_vary"],
            instance_parameter_values,
        )

        # instantiate IAM with this config set
        iam = NestedLogitIAM(instance_config)

        # simulate
        instance_df = iam.simulate(True)
        instance_df["iteration"] = which_instance

        dflist.append(instance_df.copy())

    # assemble simulation outputs/results
    df = pd.concat(dflist)
    df.index.name = "energy_source"
    df.reset_index(inplace=True)

    # write out paramters and un-processed results file
    parameter_samples_df.to_csv(
        output_dir / Path("simulation_parameters.csv"), index=False
    )
    df.to_csv(output_dir / Path("simulation_outputs.csv"), index=False)

    # additional analysis
    if config_info["sampler"] == "saltelli":
        metric = list(
            df.loc[
                df["energy_source"] == config_info["metric"], baseline_config["n_steps"]
            ]
        )
        # run sobol sensitivity analysis
        si = sobol.analyze(
            problem_definition,
            np.array(metric),
            calc_second_order=config_info["calc_second_order"],
        )

        s_first_order_df, s_total_df, s_second_order_df = si_to_dataframes(
            si,
            [p["name"] for p in config_info["pars_to_vary"]],
            config_info["calc_second_order"],
        )
        s_first_order_df.to_csv(output_dir / Path("s1_results.csv"), index=False)
        s_total_df.to_csv(output_dir / Path("s_total_results.csv"), index=False)
        if s_second_order_df is not None:
            s_second_order_df.to_csv(output_dir / Path("s2_results.csv"), index=False)


if __name__ == "__main__":
    sensitivity()
