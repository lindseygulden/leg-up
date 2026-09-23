"""Command-line entry point for computing CCS emissions factors."""

import logging
from dataclasses import replace

import click
import pandas as pd

from projects.ccs_ghg.ccs_emissions.compute import compute_ccs_emissions_factors
from projects.ccs_ghg.config import CCSProjectConfig
from projects.ccs_ghg.emissions_scenarios.config import MethaneScenario
from utils.io import yaml_to_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command("factors")
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
def factors_cli(config_path: str, output: str):
    """Compute CCS emissions factors from a project config YAML."""
    raw = yaml_to_dict(config_path)
    config = CCSProjectConfig.from_yaml(
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
                config,
                constants=replace(
                    config.constants,
                    gwp_ch4=ch4.gwp_ch4,
                    methane_leak_rate=ch4.methane_leak_rate,
                ),
            )
            df = compute_ccs_emissions_factors(config=modified)
            df["methane_scenario"] = ch4.name
            df["gwp_ch4"] = ch4.gwp_ch4
            df["methane_leak_rate"] = ch4.methane_leak_rate
            frames.append(df)
        emissions_factors_df = pd.concat(frames, ignore_index=True)
    else:
        emissions_factors_df = compute_ccs_emissions_factors(config=config)

    if output is not None:
        emissions_factors_df.to_csv(output, index=False)
        logger.info("Emissions factors written to %s", output)
    else:
        click.echo(emissions_factors_df.to_string())


if __name__ == "__main__":
    factors_cli()
