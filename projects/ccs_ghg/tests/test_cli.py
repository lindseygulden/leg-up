"""Smoke tests for the three Click CLIs.

Verifies that `factors_cli`, `industrial_emissions_cli`, and `scenarios_cli`
run to completion on the test-case configs and produce non-empty output files.
Guards against CLI-layer breakage that unit tests on the underlying compute
functions would miss (arg parsing, YAML loading, output paths, use_precomputed
branching, product_type_simple rename, etc.).
"""

from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from projects.ccs_ghg.ccs_emissions.cli import factors_cli
from projects.ccs_ghg.emissions_scenarios.cli import scenarios_cli
from projects.ccs_ghg.industrial_emissions.cli import industrial_emissions_cli

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_INFRASTRUCTURE_CONFIG = _CONFIG_DIR / "config_infrastructure_test_case.yml"
_INDUSTRIES_CONFIG = _CONFIG_DIR / "config_industries_test_case.yml"
_PROJECTS_CSV = _CONFIG_DIR / "ccs_projects_test_case.csv"


@pytest.fixture
def scenarios_config_path(tmp_path):
    """Write a temporary scenarios YAML pointing at local test fixtures.

    Overrides projects_info_path so the tests don't depend on T5_external paths.
    """
    yaml_text = f"""
projects_info_path: {_PROJECTS_CSV}
infrastructure_config_path: {_INFRASTRUCTURE_CONFIG}
industries_config_path: {_INDUSTRIES_CONFIG}
ccs_emissions_factors_path: {tmp_path / "emissions_factors.csv"}
emissions_for_industries_path: {tmp_path / "industrial_emissions.csv"}
use_precomputed: false

industrial_product_with_detail_col_name: industrial_product
product_col_name: simple_product_name
adjusted_capt_frac_col_name: adjusted_capture_fraction_tco2e
industry_emissions_cols:
  - industrial_product
  - simple_product_name
  - adjusted_capture_fraction_tco2e
  - adjusted_capture_fraction_tco2
industry_emissions_which_flavor:
  - hydrogen
  - ammonia
  - power_ng
  - 'carbon credits'

facility_keep_cols:
  - project_name
  - project_state
  - project_owner_operator
  - project_status
  - lat_degrees_n
  - lon_degrees_e
  - lat_lon_confidence
  - carbon_fate
  - project_announced
  - project_start
  - product
  - product_amount
  - product_unit
  - nameplate_captured_mtco2_per_yr
  - unabated_mtco2_per_yr
  - unabated_mtco2e_per_yr

projects_merge_on:
  - product
  - carbon_fate
ccs_emissions_merge_on:
  - product
  - stated_co2_destination

weighted_scenarios:
  - name: current_mix
    destination_pct:
      eor: 80
      gs: 20

include_nameplate_case: true
nameplate_ccs_col: nameplate_captured_mtco2_per_yr
project_start_col: project_start
project_unabated_mtco2e_col: unabated_mtco2e_per_yr
which_industries:
  - dac
  - hydrogen
  - ammonia
  - power_ng
usd_per_tco2_45q_eor: 85
usd_per_tco2_45q_gs: 85
recovery_factor_bbl_per_tco2: 3
"""
    config_path = tmp_path / "config_scenarios.yml"
    config_path.write_text(yaml_text.strip() + "\n")
    return config_path


class TestFactorsCLI:
    def test_runs_and_writes_non_empty_csv(self, scenarios_config_path, tmp_path):
        output = tmp_path / "factors.csv"
        result = CliRunner().invoke(
            factors_cli,
            ["--config", str(scenarios_config_path), "--output", str(output)],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}\n{result.exception}"
        assert output.exists()
        df = pd.read_csv(output)
        assert len(df) > 0
        assert "co2_source" in df.columns
        assert "co2_destination" in df.columns


class TestIndustrialEmissionsCLI:
    def test_runs_and_writes_non_empty_csv(self, scenarios_config_path, tmp_path):
        output = tmp_path / "industrial.csv"
        result = CliRunner().invoke(
            industrial_emissions_cli,
            ["--config", str(scenarios_config_path), "--output", str(output)],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}\n{result.exception}"
        assert output.exists()
        df = pd.read_csv(output)
        assert len(df) > 0
        assert "industrial_product" in df.columns
        assert "total_tco2_per_unit_product" in df.columns


class TestScenariosCLI:
    def test_full_pipeline_produces_projects_and_scenarios(
        self, scenarios_config_path, tmp_path
    ):
        projects_out = tmp_path / "projects.csv"
        scenarios_out = tmp_path / "scenarios.csv"
        result = CliRunner().invoke(
            scenarios_cli,
            [
                "--config",
                str(scenarios_config_path),
                "--output-projects",
                str(projects_out),
                "--output-scenarios",
                str(scenarios_out),
            ],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}\n{result.exception}"
        assert projects_out.exists()
        assert scenarios_out.exists()

        projects_df = pd.read_csv(projects_out)
        scenarios_df = pd.read_csv(scenarios_out)
        assert len(projects_df) > 0
        assert len(scenarios_df) > 0
        assert "uid" in projects_df.columns
        assert "uid" in scenarios_df.columns
        assert "scenario" in scenarios_df.columns
        # State tag column ends up in the projects CSV (PROJECT_INFO_COLS split).
        assert "project_state" in projects_df.columns
        # Both TX and gulf coast projects should appear in the projects output.
        assert {"TX", "gulf coast"}.issubset(set(projects_df["project_state"].unique()))
