"""Command-line entry point for computing industrial emissions factors."""

import logging
from dataclasses import replace

import click
import pandas as pd

from projects.ccs_ghg.config import CCSProjectConfig
from projects.ccs_ghg.emissions_scenarios.config import MethaneScenario
from projects.ccs_ghg.industrial_emissions.compute import compute_industrial_emissions
from projects.ccs_ghg.industrial_emissions.config import IndustrialEmissionsConfig
from utils.io import yaml_to_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command("industrial-emissions")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="Path to project config YAML (e.g. config/config_scenarios.yml).",
)
@click.option(
    "--output",
    type=click.Path(file_okay=True, dir_okay=False),
    required=False,
    default=None,
    help="Path to write the resulting CSV.",
)
def industrial_emissions_cli(config_path: str, output: str):
    """Compute per-industry CO2/CO2e emissions factors from a project config YAML."""
    raw = yaml_to_dict(config_path)
    project_config = CCSProjectConfig.from_yaml(
        raw["infrastructure_config_path"],
        raw["industries_config_path"],
    )

    methane_scenarios = [
        MethaneScenario.from_dict(s) for s in raw.get("methane_scenarios", [])
    ]

    if methane_scenarios:
        frames = []
        for ch4 in methane_scenarios:
            modified = replace(
                project_config,
                constants=replace(
                    project_config.constants,
                    gwp_ch4=ch4.gwp_ch4,
                    methane_leak_rate=ch4.methane_leak_rate,
                ),
            )
            ie_config = IndustrialEmissionsConfig.from_project_config(modified)
            df = compute_industrial_emissions(ie_config)
            df["methane_scenario"] = ch4.name
            df["gwp_ch4"] = ch4.gwp_ch4
            df["methane_leak_rate"] = ch4.methane_leak_rate
            frames.append(df)
        result_df = pd.concat(frames, ignore_index=True)
    else:
        ie_config = IndustrialEmissionsConfig.from_project_config(project_config)
        result_df = compute_industrial_emissions(ie_config)

    if output is not None:
        result_df.to_csv(output, index=False)
        logger.info("Industrial emissions written to %s", output)
    else:
        click.echo(result_df.to_string())


if __name__ == "__main__":
    industrial_emissions_cli()
