"""Command-line entry point for computing CCS scenarios."""

import logging
import re

import click
import pandas as pd

from projects.ccs_ghg.emissions_scenarios.config import ScenarioConfig
from projects.ccs_ghg.emissions_scenarios.scenarios import (
    compute_scenario_emissions_factors,
    compute_scenarios,
    run_full_pipeline,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_INFO_COLS = [
    "uid",
    "project_name",
    "project_state",
    "project_owner_operator",
    "project_status",
    "lat_degrees_n",
    "lon_degrees_e",
    "lat_lon_confidence",
    "carbon_fate",
    "project_announced",
    "project_start",
    "product_unit",
    "product_amount",
    "co2_source",
    "is_methane_based",
]


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


@click.command("scenarios")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="Path to scenario config YAML (e.g. config/config_scenarios.yml).",
)
@click.option(
    "--output-projects",
    type=click.Path(file_okay=True, dir_okay=False),
    required=False,
    default=None,
    help="Path to write the project info CSV.",
)
@click.option(
    "--output-scenarios",
    type=click.Path(file_okay=True, dir_okay=False),
    required=False,
    default=None,
    help="Path to write the scenario results CSV.",
)
@click.option(
    "--weighted-factors-output",
    type=click.Path(file_okay=True, dir_okay=False),
    required=False,
    default=None,
    help="Path to write the weighted scenario emissions factors CSV.",
)
def scenarios_cli(
    config_path: str,
    output_projects: str,
    output_scenarios: str,
    weighted_factors_output: str,
):
    """Compute CCS emissions scenarios from a YAML config file."""
    scenario_config = ScenarioConfig.from_yaml(config_path)

    if scenario_config.use_precomputed:
        projects_df = pd.read_csv(scenario_config.projects_info_path, encoding="latin1")
        if "product_type_simple" in projects_df.columns:
            projects_df = projects_df.rename(columns={"product_type_simple": "product"})
        if "include" in projects_df.columns:
            excluded = projects_df[projects_df["include"] != 1]
            if len(excluded):
                logger.info("Excluding %d project(s) with include != 1", len(excluded))
            projects_df = projects_df[projects_df["include"] == 1].copy()
        emissions_factors_df = pd.read_csv(scenario_config.ccs_emissions_factors_path)

        if weighted_factors_output is not None:
            weighted_factors_df = compute_scenario_emissions_factors(
                emissions_factors_df, scenario_config
            )
            weighted_factors_df.to_csv(weighted_factors_output, index=False)
            logger.info("Weighted factors written to %s", weighted_factors_output)

        scenarios_df = compute_scenarios(
            projects_df=projects_df,
            emissions_factors_df=emissions_factors_df,
            scenario_config=scenario_config,
        )
    else:
        pipeline_result = run_full_pipeline(scenario_config)
        scenarios_df = pipeline_result["scenarios"]

    if "uid" not in scenarios_df.columns:
        scenarios_df["uid"] = scenarios_df["project_name"].map(_slugify)
    if "scenario_id" not in scenarios_df.columns:
        methane_part = scenarios_df.get(
            "methane_scenario", pd.Series("", index=scenarios_df.index)
        )
        scenarios_df["scenario_id"] = scenarios_df["scenario"].where(
            methane_part == "", scenarios_df["scenario"] + "-" + methane_part
        )

    if output_projects is not None or output_scenarios is not None:
        info_cols = [c for c in PROJECT_INFO_COLS if c in scenarios_df.columns]
        projects_info = scenarios_df[info_cols].drop_duplicates().reset_index(drop=True)

        if "mtco2_per_yr_captured_with_ccs" in scenarios_df.columns:
            # Use any non-`nameplate_implied` scenario row for the per-project
            # captured value (all weighted / baseline scenarios share the same
            # captured for a given project). Skip `nameplate_implied` because
            # its captured uses the operator/implied nameplate, which differs.
            captured = scenarios_df[
                scenarios_df["scenario"] != "nameplate_implied"
            ].drop_duplicates(subset="uid")[["uid", "mtco2_per_yr_captured_with_ccs"]]
            projects_info = projects_info.merge(captured, on="uid", how="left")

        # project_state is metadata (goes to the projects file) but is also
        # needed on every scenario row for state-faceted plots — duplicate it.
        _KEEP_IN_SCENARIOS = {"project_state", "is_methane_based"}
        other_cols = [
            c
            for c in scenarios_df.columns
            if (c not in PROJECT_INFO_COLS or c in _KEEP_IN_SCENARIOS)
            and c not in ("uid", "project_name")
        ]
        scenario_results = scenarios_df[["uid"] + other_cols]
        scenario_results = scenario_results.rename(
            columns={
                "nameplate_captured_mtco2_per_yr": "operator_reported_nameplate_captured_mtco2_per_yr",
            }
        )
        if "mtco2e_per_yr_captured_with_ccs" in scenario_results.columns:
            scenario_results = scenario_results.drop(
                columns=["mtco2e_per_yr_captured_with_ccs"]
            )

        if output_projects is not None:
            projects_info.to_csv(output_projects, index=False)
            logger.info("Project info written to %s", output_projects)
        if output_scenarios is not None:
            scenario_results.to_csv(output_scenarios, index=False)
            logger.info("Scenario results written to %s", output_scenarios)
    else:
        click.echo(scenarios_df.to_string())


if __name__ == "__main__":
    scenarios_cli()
