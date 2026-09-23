"""End-to-end pipeline integration test using the _test_case config files.

Exercises both precomputed and full-pipeline (leak-rate sweep) modes.
Includes explicit hydrogen math checks so intermediate values can be audited.

Run with: pytest projects/ccs_ghg/tests/test_pipeline.py -v
"""

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from projects.ccs_ghg.ccs_emissions.compute import compute_ccs_emissions_factors
from projects.ccs_ghg.ccs_emissions.energy_emissions import (
    compute_gas_turbine_emissions_per_mwh,
    compute_grid_emissions_per_mwh,
)
from projects.ccs_ghg.config import CCSProjectConfig
from projects.ccs_ghg.emissions_scenarios.config import MethaneScenario, ScenarioConfig
from projects.ccs_ghg.emissions_scenarios.scenarios import (
    compute_scenarios,
    compute_scenarios_full_pipeline,
)
from projects.ccs_ghg.industrial_emissions.compute import (
    compute_industrial_emissions,
    compute_industry_row,
    resolve_combustion_co2_tco2,
)
from projects.ccs_ghg.industrial_emissions.config import IndustrialEmissionsConfig
from utils.io import yaml_to_dict

# ---------------------------------------------------------------------------
# Paths to test-case configs
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_INFRASTRUCTURE_CONFIG = str(_CONFIG_DIR / "config_infrastructure_test_case.yml")
_INDUSTRIES_CONFIG = str(_CONFIG_DIR / "config_industries_test_case.yml")
_PROJECTS_CSV = str(_CONFIG_DIR / "ccs_projects_test_case.csv")
_SCENARIOS_CONFIG = str(_CONFIG_DIR / "config_scenarios_test_case.yml")

# CH4 assumptions (duplicated here for readable math assertions)
GWP = 86
LEAK = 0.035

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def project_config():
    return CCSProjectConfig.from_yaml(
        _INFRASTRUCTURE_CONFIG,
        _INDUSTRIES_CONFIG,
    )


@pytest.fixture
def ie_config(project_config):
    return IndustrialEmissionsConfig.from_project_config(project_config)


@pytest.fixture
def ccs_factors_df(project_config):
    return compute_ccs_emissions_factors(project_config)


@pytest.fixture
def industry_df(ie_config):
    return compute_industrial_emissions(ie_config)


@pytest.fixture
def projects_df():
    return pd.read_csv(_PROJECTS_CSV)


@pytest.fixture
def scenario_config():
    raw = yaml_to_dict(_SCENARIOS_CONFIG)
    raw["projects_info_path"] = _PROJECTS_CSV
    return ScenarioConfig.from_dict(raw)


# ---------------------------------------------------------------------------
# Precomputed-mode tests
# ---------------------------------------------------------------------------


class TestPrecomputedMode:
    def test_produces_output_with_expected_scenarios(
        self, projects_df, ccs_factors_df, industry_df, scenario_config
    ):
        cfg = replace(scenario_config, use_precomputed=True)
        result = compute_scenarios(
            projects_df=projects_df,
            emissions_factors_df=ccs_factors_df,
            scenario_config=cfg,
            industry_df=industry_df,
        )
        assert len(result) > 0
        assert set(result["scenario"].unique()) == {
            "current_mix",
            "baseline",
            "nameplate_implied",
            "nameplate",
        }

    def test_all_test_projects_present(
        self, projects_df, ccs_factors_df, industry_df, scenario_config
    ):
        cfg = replace(scenario_config, use_precomputed=True)
        result = compute_scenarios(
            projects_df=projects_df,
            emissions_factors_df=ccs_factors_df,
            scenario_config=cfg,
            industry_df=industry_df,
        )
        assert set(projects_df["project_name"]) == set(result["project_name"].unique())

    def test_has_required_output_columns(
        self, projects_df, ccs_factors_df, industry_df, scenario_config
    ):
        cfg = replace(scenario_config, use_precomputed=True)
        result = compute_scenarios(
            projects_df=projects_df,
            emissions_factors_df=ccs_factors_df,
            scenario_config=cfg,
            industry_df=industry_df,
        )
        for col in [
            "scenario",
            "stated_co2_destination",
            "assumed_co2_destination",
            "mtco2e_per_yr_emitted_because_of_ccs",
            "mtco2e_per_yr_emissions_including_ccs",
            "pv_million_usd_credits_45q",
            "eor_fraction",
            "million_barrels_oil_per_yr",
        ]:
            assert col in result.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Full-pipeline-mode tests
# ---------------------------------------------------------------------------


class TestFullPipelineMode:
    def test_produces_output_with_methane_scenario_columns(self, scenario_config):
        cfg = replace(
            scenario_config,
            use_precomputed=False,
            infrastructure_config_path=_INFRASTRUCTURE_CONFIG,
            industries_config_path=_INDUSTRIES_CONFIG,
            methane_scenarios=[
                MethaneScenario("gwp20_base", 86, 0.035),
                MethaneScenario("gwp100_low", 29.8, 0.02),
            ],
        )
        result = compute_scenarios_full_pipeline(cfg)
        assert len(result) > 0
        assert "methane_scenario" in result.columns
        assert "gwp_ch4" in result.columns
        assert "methane_leak_rate" in result.columns
        assert set(result["methane_scenario"].unique()) == {
            "gwp20_base",
            "gwp100_low",
        }

    def test_lower_leak_rate_reduces_ccs_lifecycle_factor(self, scenario_config):
        cfg = replace(
            scenario_config,
            use_precomputed=False,
            infrastructure_config_path=_INFRASTRUCTURE_CONFIG,
            industries_config_path=_INDUSTRIES_CONFIG,
            methane_scenarios=[
                MethaneScenario("gwp20_base", 86, 0.035),
                MethaneScenario("gwp20_low", 86, 0.02),
            ],
        )
        result = compute_scenarios_full_pipeline(cfg)
        h2_cm = result[
            (result["product"] == "hydrogen") & (result["scenario"] == "current_mix")
        ]
        proj = h2_cm["project_name"].iloc[0]
        dest = h2_cm["stated_co2_destination"].iloc[0]
        mask = (h2_cm["project_name"] == proj) & (
            h2_cm["stated_co2_destination"] == dest
        )
        base_factor = h2_cm.loc[
            mask & (h2_cm["methane_scenario"] == "gwp20_base"),
            "tco2e_emitted_per_tco2_ccs",
        ].iloc[0]
        low_factor = h2_cm.loc[
            mask & (h2_cm["methane_scenario"] == "gwp20_low"),
            "tco2e_emitted_per_tco2_ccs",
        ].iloc[0]
        assert low_factor < base_factor

    def test_matches_precomputed_at_same_leak_rate(
        self, projects_df, ccs_factors_df, industry_df, scenario_config
    ):
        """Precomputed vs. full-pipeline paths agree, restricted to projects
        in the state whose constants match the precomputed factor table.

        The `ccs_factors_df` fixture is built from `project_config.constants`
        (top-level, which mirror TX in the test-case YAML), so only TX projects
        should agree bit-for-bit between the two paths. Gulf-coast projects
        deliberately use divergent state constants and are expected to differ."""
        cfg_pre = replace(scenario_config, use_precomputed=True)
        result_pre = compute_scenarios(
            projects_df=projects_df,
            emissions_factors_df=ccs_factors_df,
            scenario_config=cfg_pre,
            industry_df=industry_df,
        )
        cfg_full = replace(
            scenario_config,
            use_precomputed=False,
            infrastructure_config_path=_INFRASTRUCTURE_CONFIG,
            industries_config_path=_INDUSTRIES_CONFIG,
            methane_scenarios=[MethaneScenario("base", 86, 0.035)],
        )
        result_full = compute_scenarios_full_pipeline(cfg_full)

        col = "mtco2e_per_yr_emissions_including_ccs"
        for scenario in ["current_mix", "baseline"]:
            pre = (
                result_pre[
                    (result_pre["scenario"] == scenario)
                    & (result_pre["project_state"] == "TX")
                ]
                .set_index(["project_name", "stated_co2_destination"])[col]
                .sort_index()
            )
            full = (
                result_full[
                    (result_full["scenario"] == scenario)
                    & (result_full["project_state"] == "TX")
                ]
                .set_index(["project_name", "stated_co2_destination"])[col]
                .sort_index()
            )
            # rtol=1e-6 tolerates floating-point drift from the two paths doing
            # the same math via different sequences of operations (cross-joins,
            # groupbys, etc.). Well below any modeling uncertainty.
            pd.testing.assert_series_equal(pre, full, rtol=1e-6)

    def test_dac_unabated_is_zero(self, projects_df, scenario_config):
        """DAC has no industry-side emission factors (it neither emits nor
        removes CO2 unless CCS is operating). Convention: without CCS, DAC does
        nothing, so unabated_mtco2_per_yr = 0. With CCS, computed_captured =
        nameplate_captured, and net = 0 - nameplate + CCS_operational_emissions.

        This guards against regression where DAC unabated is set to -nameplate
        (which double-counts the removal since net = unabated - captured)."""
        cfg_full = replace(
            scenario_config,
            use_precomputed=False,
            infrastructure_config_path=_INFRASTRUCTURE_CONFIG,
            industries_config_path=_INDUSTRIES_CONFIG,
            methane_scenarios=[MethaneScenario("base", 86, 0.035)],
        )
        result = compute_scenarios_full_pipeline(cfg_full)

        dac_projects = projects_df[projects_df["product"] == "dac"]
        assert (
            len(dac_projects) > 0
        ), "test fixture must include at least one DAC project"

        for _, csv_row in dac_projects.iterrows():
            pipeline_rows = result[result["project_name"] == csv_row["project_name"]]
            assert (
                len(pipeline_rows) > 0
            ), f"pipeline output missing DAC project {csv_row['project_name']}"
            # Every scenario row for this DAC project should carry unabated=0.
            actual_unabated = pipeline_rows["unabated_mtco2_per_yr"].unique()
            assert len(actual_unabated) == 1, (
                f"unabated_mtco2 differs across scenarios for "
                f"{csv_row['project_name']}: {actual_unabated}"
            )
            assert actual_unabated[0] == pytest.approx(0.0, abs=1e-12), (
                f"DAC unabated_mtco2 should be 0 (without CCS, DAC does nothing) "
                f"but got {actual_unabated[0]} for {csv_row['project_name']}"
            )
            # For non-nameplate_operator scenarios, mtco2_per_yr_captured_with_ccs
            # = adjusted_capture_fraction * unabated; but adjusted_capture_fraction
            # comes from industry emissions (missing for DAC). Focus the assertion
            # on the input nameplate being carried through.
            assert pipeline_rows["nameplate_captured_mtco2_per_yr"].iloc[
                0
            ] == pytest.approx(
                csv_row["nameplate_captured_mtco2_per_yr"], rel=1e-9
            ), f"DAC nameplate lost in pipeline for {csv_row['project_name']}"

    def test_gulf_coast_diverges_from_tx(self, scenario_config):
        """Verify per-state constants actually propagate: gulf-coast projects
        should produce different fugitive-CH4-loaded factors than TX projects
        because the test-case YAML gives gulf coast a higher grid_tch4_per_mwh.

        This guards against a bug where the state loop silently used a single
        state's constants for all projects."""
        cfg_full = replace(
            scenario_config,
            use_precomputed=False,
            infrastructure_config_path=_INFRASTRUCTURE_CONFIG,
            industries_config_path=_INDUSTRIES_CONFIG,
            methane_scenarios=[MethaneScenario("base", 86, 0.035)],
        )
        result = compute_scenarios_full_pipeline(cfg_full)

        # Both TX and gulf coast projects should be present.
        states = set(result["project_state"].unique())
        assert {"TX", "gulf coast"}.issubset(states)

        # For every industry present in both states, gulf-coast rows should
        # have a strictly higher tco2e_emitted_per_tco2_ccs (from the higher
        # state grid_tch4 + higher grid_tco2 injected into transport/storage/
        # capture components).
        by_state = (
            result.groupby(["project_state", "co2_source"])[
                "mtco2e_per_yr_emissions_including_ccs"
            ]
            .first()
            .unstack("project_state")
        )
        # Gulf coast values not-NaN and TX values not-NaN:
        both_present = by_state.dropna(subset=["TX", "gulf coast"])
        assert len(both_present) > 0, "expected some industries in both states"
        # At least one industry should differ meaningfully between the two states.
        rel_diff = (
            both_present["gulf coast"] - both_present["TX"]
        ).abs() / both_present["TX"].abs().clip(lower=1e-9)
        assert (rel_diff > 1e-3).any(), (
            "Gulf-coast values are indistinguishable from TX; state constants "
            "may not be propagating through the pipeline"
        )


# ---------------------------------------------------------------------------
# Hydrogen math: industrial emissions
# ---------------------------------------------------------------------------


class TestHydrogenIndustrialEmissions:
    """Trace the hydrogen industrial emissions math step by step."""

    @pytest.fixture
    def h2_input(self, ie_config):
        return next(
            i for i in ie_config.industries if i.simple_product_name == "hydrogen"
        )

    @pytest.fixture
    def cc(self, ie_config):
        return ie_config.constants

    @pytest.fixture
    def h2_row(self, h2_input, cc):
        return compute_industry_row(h2_input, cc)

    def test_combustion_co2_is_direct(self, h2_input, cc):
        assert resolve_combustion_co2_tco2(h2_input, cc) == 3.816

    def test_process_co2e(self, h2_row):
        expected = 4.62 + LEAK * GWP * 1.68  # 4.62 + 5.0568 = 9.6768
        assert h2_row["process_tco2e_per_unit_product"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_combustion_co2e(self, h2_row):
        expected = 3.816 + LEAK * GWP * 1.387636
        assert h2_row["combustion_tco2e_per_unit_product"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_total_co2(self, h2_row):
        assert h2_row["total_tco2_per_unit_product"] == pytest.approx(
            4.62 + 3.816, rel=1e-9
        )

    def test_total_co2e(self, h2_row):
        expected = (4.62 + LEAK * GWP * 1.68) + (3.816 + LEAK * GWP * 1.387636)
        assert h2_row["total_tco2e_per_unit_product"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_co2_captured(self, h2_row):
        co2_captured = 0.90 * 4.62  # process only (fuel_cf = 0)
        assert co2_captured == pytest.approx(4.158, rel=1e-9)

    def test_adj_cf_tco2(self, h2_row):
        expected = (0.90 * 4.62) / (4.62 + 3.816)
        assert h2_row["adjusted_capture_fraction_tco2"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_adj_cf_tco2e(self, h2_row):
        total_co2e = (4.62 + LEAK * GWP * 1.68) + (3.816 + LEAK * GWP * 1.387636)
        expected = (0.90 * 4.62) / total_co2e
        assert h2_row["adjusted_capture_fraction_tco2e"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_pcf_tco2e(self, h2_row):
        process_co2e = 4.62 + LEAK * GWP * 1.68
        expected = (0.90 * 4.62) / process_co2e
        assert h2_row["process_capture_fraction_tco2e"] == pytest.approx(
            expected, rel=1e-9
        )


# ---------------------------------------------------------------------------
# Hydrogen math: CCS capture, transport, storage factors
# ---------------------------------------------------------------------------


class TestHydrogenCCSFactors:
    """Trace the CCS emissions factors math for hydrogen step by step."""

    @pytest.fixture
    def constants(self, project_config):
        return project_config.constants

    @pytest.fixture
    def emission_factors(self, project_config):
        return project_config.constants

    @pytest.fixture
    def grid_emissions(self, constants):
        return compute_grid_emissions_per_mwh(constants)

    @pytest.fixture
    def tco2e_per_mwh(self, grid_emissions):
        return grid_emissions[1]

    def test_grid_fugitive_ch4(self, constants, grid_emissions):
        expected_fugitive = constants.grid_tch4_per_mwh * GWP * LEAK
        grid_tco2, grid_tco2e = grid_emissions
        assert grid_tco2 == pytest.approx(constants.grid_tco2_per_mwh, rel=1e-9)
        assert grid_tco2e == pytest.approx(
            constants.grid_tco2_per_mwh + expected_fugitive, rel=1e-9
        )

    def test_tco2e_per_mwh(self, constants, grid_emissions):
        grid_tco2, grid_tco2e = grid_emissions
        assert grid_tco2e > grid_tco2

    def test_hydrogen_capture_factors(
        self, ccs_factors_df, constants, emission_factors, tco2e_per_mwh
    ):
        h2_eor = ccs_factors_df[
            (ccs_factors_df["co2_source"] == "hydrogen")
            & (ccs_factors_df["co2_destination"] == "eor")
        ].iloc[0]

        # --- capture ---
        captured = 404700 * 0.9  # 364230
        mwh_capture = 4.04 * 365 * 24  # 35390.4
        mwh_per_tco2 = mwh_capture / captured
        mmbtu_capture = 2330.0 * 365  # 850450
        mmbtu_per_tco2 = mmbtu_capture / captured

        tco2_elec = mwh_per_tco2 * constants.grid_tco2_per_mwh
        tco2e_elec = mwh_per_tco2 * tco2e_per_mwh
        tco2_combust = mmbtu_per_tco2 * emission_factors.natural_gas_tco2_per_mmbtu
        tco2e_combust = tco2_combust + (
            GWP * LEAK * mmbtu_per_tco2 * constants.tch4_per_mmbtu_pipeline_gas
        )
        escaped = (404700 - captured) / captured  # 1/9

        cap_tco2 = tco2_combust + escaped + tco2_elec
        cap_tco2e = tco2e_combust + escaped + tco2e_elec

        # --- transport (near) ---
        trans_tco2 = 0.09 * constants.grid_tco2_per_mwh
        trans_tco2e = 0.09 * tco2e_per_mwh

        # --- storage (eor) ---
        eor_tco2, eor_tco2e = compute_gas_turbine_emissions_per_mwh(
            constants, constants.natural_gas_tco2_per_mmbtu
        )
        stor_tco2 = 0.25 * eor_tco2
        stor_tco2e = 0.25 * eor_tco2e
        product_tco2 = (
            constants.oil_tco2_per_bbl * constants.recovery_factor_bbl_per_tco2
        )

        # --- totals ---
        expected_tco2 = cap_tco2 + trans_tco2 + stor_tco2 + product_tco2
        expected_tco2e = cap_tco2e + trans_tco2e + stor_tco2e + product_tco2

        assert h2_eor["tco2_emitted_per_tco2_ccs"] == pytest.approx(
            expected_tco2, rel=1e-9
        )
        assert h2_eor["tco2e_emitted_per_tco2_ccs"] == pytest.approx(
            expected_tco2e, rel=1e-9
        )
        assert h2_eor["net_tco2_emitted_per_tco2_ccs"] == pytest.approx(
            expected_tco2 - 1, rel=1e-9
        )
        assert h2_eor["net_tco2e_emitted_per_tco2_ccs"] == pytest.approx(
            expected_tco2e - 1, rel=1e-9
        )

    def test_hydrogen_gs_has_no_product_emissions(self, ccs_factors_df):
        h2_gs = ccs_factors_df[
            (ccs_factors_df["co2_source"] == "hydrogen")
            & (ccs_factors_df["co2_destination"] == "gs")
        ].iloc[0]
        h2_eor = ccs_factors_df[
            (ccs_factors_df["co2_source"] == "hydrogen")
            & (ccs_factors_df["co2_destination"] == "eor")
        ].iloc[0]
        assert h2_gs["tco2_emitted_per_tco2_ccs"] < h2_eor["tco2_emitted_per_tco2_ccs"]


# ---------------------------------------------------------------------------
# Hydrogen math: scenario-level output
# ---------------------------------------------------------------------------


class TestHydrogenScenarioOutput:
    """Trace the scenario math for a hydrogen project."""

    @pytest.fixture
    def h2_results(self, projects_df, ccs_factors_df, industry_df, scenario_config):
        cfg = replace(scenario_config, use_precomputed=True)
        result = compute_scenarios(
            projects_df=projects_df,
            emissions_factors_df=ccs_factors_df,
            scenario_config=cfg,
            industry_df=industry_df,
        )
        return result[result["product"] == "hydrogen"]

    def test_current_mix_weighted_factor(self, ccs_factors_df, h2_results):
        h2_eor_factor = ccs_factors_df.loc[
            (ccs_factors_df["co2_source"] == "hydrogen")
            & (ccs_factors_df["co2_destination"] == "eor"),
            "tco2e_emitted_per_tco2_ccs",
        ].iloc[0]
        h2_gs_factor = ccs_factors_df.loc[
            (ccs_factors_df["co2_source"] == "hydrogen")
            & (ccs_factors_df["co2_destination"] == "gs"),
            "tco2e_emitted_per_tco2_ccs",
        ].iloc[0]
        expected_weighted = (80 * h2_eor_factor + 20 * h2_gs_factor) / 100

        cm = h2_results[h2_results["scenario"] == "current_mix"]
        assert cm["tco2e_emitted_per_tco2_ccs"].iloc[0] == pytest.approx(
            expected_weighted, rel=1e-9
        )

    def test_nameplate_implied_uses_computed_capture(self, h2_results):
        """`nameplate_implied` scenario always uses the computed capture
        (unabated × capture_fraction), never the operator-supplied nameplate."""
        np_rows = h2_results[h2_results["scenario"] == "nameplate_implied"]
        for _, row in np_rows.iterrows():
            expected = (
                row["unabated_mtco2_per_yr"] * row["adjusted_capture_fraction_tco2"]
            )
            assert row["mtco2_per_yr_captured_with_ccs"] == pytest.approx(
                expected, rel=1e-9
            )

    def test_nameplate_scenario_uses_operator_when_supplied(self, h2_results):
        """`nameplate` scenario = operator-supplied nameplate when present,
        else falls back to the implied value."""
        np_rows = h2_results[h2_results["scenario"] == "nameplate"]
        for _, row in np_rows.iterrows():
            implied = (
                row["unabated_mtco2_per_yr"] * row["adjusted_capture_fraction_tco2"]
            )
            nameplate = row.get("nameplate_captured_mtco2_per_yr")
            expected = nameplate if pd.notna(nameplate) else implied
            assert row["mtco2_per_yr_captured_with_ccs"] == pytest.approx(
                expected, rel=1e-9
            )

    def test_captured_equals_unabated_times_adj_cf(self, h2_results):
        cm = h2_results[h2_results["scenario"] == "current_mix"]
        for _, row in cm.iterrows():
            expected = (
                row["unabated_mtco2_per_yr"] * row["adjusted_capture_fraction_tco2"]
            )
            assert row["mtco2_per_yr_captured_with_ccs"] == pytest.approx(
                expected, rel=1e-9
            )

    def test_emissions_after_ccs_formula(self, h2_results):
        cm = h2_results[h2_results["scenario"] == "current_mix"]
        for _, row in cm.iterrows():
            expected = (
                row["unabated_mtco2e_per_yr"]
                - row["mtco2_per_yr_captured_with_ccs"]
                + row["mtco2e_per_yr_emitted_because_of_ccs"]
            )
            assert row["mtco2e_per_yr_emissions_including_ccs"] == pytest.approx(
                expected, rel=1e-9
            )


# ---------------------------------------------------------------------------
# Ammonia: single-industry full-pipeline test
# ---------------------------------------------------------------------------

NH3_PROCESS_TCO2 = 1.082052661856717
NH3_COMBUSTION_TCO2 = 0.5976816039600324
NH3_PROCESS_TCH4 = 0.393474
NH3_COMBUSTION_TCH4 = 0.217339
NH3_CAPTURE_RATE = 0.95
NH3_UNABATED_MTCO2E = 3.870110
NH3_NAMEPLATE = 1.6


class TestAmmoniaFullPipeline:
    """End-to-end ammonia pipeline: industrial emissions through scenario output."""

    @pytest.fixture
    def nh3_industry_row(self, ie_config):
        nh3_input = next(
            i for i in ie_config.industries if i.simple_product_name == "ammonia"
        )
        cc = ie_config.constants
        return compute_industry_row(nh3_input, cc)

    @pytest.fixture
    def nh3_scenario_results(
        self, projects_df, ccs_factors_df, industry_df, scenario_config
    ):
        cfg = replace(scenario_config, use_precomputed=True)
        result = compute_scenarios(
            projects_df=projects_df,
            emissions_factors_df=ccs_factors_df,
            scenario_config=cfg,
            industry_df=industry_df,
        )
        return result[result["product"] == "ammonia"]

    def test_industrial_process_co2e(self, nh3_industry_row):
        expected = NH3_PROCESS_TCO2 + LEAK * GWP * NH3_PROCESS_TCH4
        assert nh3_industry_row["process_tco2e_per_unit_product"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_industrial_total_co2(self, nh3_industry_row):
        assert nh3_industry_row["total_tco2_per_unit_product"] == pytest.approx(
            NH3_PROCESS_TCO2 + NH3_COMBUSTION_TCO2, rel=1e-9
        )

    def test_adj_capture_fraction_tco2(self, nh3_industry_row):
        expected = (NH3_CAPTURE_RATE * NH3_PROCESS_TCO2) / (
            NH3_PROCESS_TCO2 + NH3_COMBUSTION_TCO2
        )
        assert nh3_industry_row["adjusted_capture_fraction_tco2"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_adj_capture_fraction_tco2e(self, nh3_industry_row):
        total_co2e = (NH3_PROCESS_TCO2 + LEAK * GWP * NH3_PROCESS_TCH4) + (
            NH3_COMBUSTION_TCO2 + LEAK * GWP * NH3_COMBUSTION_TCH4
        )
        expected = (NH3_CAPTURE_RATE * NH3_PROCESS_TCO2) / total_co2e
        assert nh3_industry_row["adjusted_capture_fraction_tco2e"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_scenario_captured_equals_unabated_times_adj_cf(self, nh3_scenario_results):
        cm = nh3_scenario_results[nh3_scenario_results["scenario"] == "current_mix"]
        for _, row in cm.iterrows():
            expected = (
                row["unabated_mtco2_per_yr"] * row["adjusted_capture_fraction_tco2"]
            )
            assert row["mtco2_per_yr_captured_with_ccs"] == pytest.approx(
                expected, rel=1e-9
            )

    def test_nameplate_scenario_uses_operator_when_supplied(self, nh3_scenario_results):
        """`nameplate` scenario uses the operator-supplied nameplate value."""
        np_rows = nh3_scenario_results[nh3_scenario_results["scenario"] == "nameplate"]
        for _, row in np_rows.iterrows():
            assert row["mtco2_per_yr_captured_with_ccs"] == pytest.approx(
                NH3_NAMEPLATE, rel=1e-9
            )

    def test_nameplate_implied_uses_computed_capture(self, nh3_scenario_results):
        np_rows = nh3_scenario_results[
            nh3_scenario_results["scenario"] == "nameplate_implied"
        ]
        for _, row in np_rows.iterrows():
            expected = (
                row["unabated_mtco2_per_yr"] * row["adjusted_capture_fraction_tco2"]
            )
            assert row["mtco2_per_yr_captured_with_ccs"] == pytest.approx(
                expected, rel=1e-9
            )

    def test_emissions_after_ccs_formula(self, nh3_scenario_results):
        for _, row in nh3_scenario_results.iterrows():
            expected = (
                row["unabated_mtco2e_per_yr"]
                - row["mtco2_per_yr_captured_with_ccs"]
                + row["mtco2e_per_yr_emitted_because_of_ccs"]
            )
            assert row["mtco2e_per_yr_emissions_including_ccs"] == pytest.approx(
                expected, rel=1e-9
            )

    def test_full_pipeline_with_methane_scenarios(self, scenario_config):
        cfg = replace(
            scenario_config,
            use_precomputed=False,
            infrastructure_config_path=_INFRASTRUCTURE_CONFIG,
            industries_config_path=_INDUSTRIES_CONFIG,
            methane_scenarios=[
                MethaneScenario("gwp20_base", 86, 0.035),
                MethaneScenario("gwp100_low", 29.8, 0.02),
            ],
        )
        result = compute_scenarios_full_pipeline(cfg)
        nh3 = result[result["product"] == "ammonia"]
        assert len(nh3) > 0
        assert set(nh3["methane_scenario"].unique()) == {"gwp20_base", "gwp100_low"}
        # Lower GWP should reduce tco2e lifecycle factor
        cm = nh3[nh3["scenario"] == "current_mix"]
        proj = cm["project_name"].iloc[0]
        dest = cm["stated_co2_destination"].iloc[0]
        mask = (cm["project_name"] == proj) & (cm["stated_co2_destination"] == dest)
        base = cm.loc[
            mask & (cm["methane_scenario"] == "gwp20_base"),
            "tco2e_emitted_per_tco2_ccs",
        ].iloc[0]
        low = cm.loc[
            mask & (cm["methane_scenario"] == "gwp100_low"),
            "tco2e_emitted_per_tco2_ccs",
        ].iloc[0]
        assert low < base

    def test_full_pipeline_no_methane_scenarios_uses_baseline(self, scenario_config):
        cfg = replace(
            scenario_config,
            use_precomputed=False,
            infrastructure_config_path=_INFRASTRUCTURE_CONFIG,
            industries_config_path=_INDUSTRIES_CONFIG,
            methane_scenarios=[],
        )
        result = compute_scenarios_full_pipeline(cfg)
        assert "methane_scenario" in result.columns
        assert set(result["methane_scenario"].unique()) == {"baseline"}

    def test_full_pipeline_product_type_simple_rename(self, scenario_config):
        import tempfile

        df = pd.read_csv(_PROJECTS_CSV)
        df = df.rename(columns={"product": "product_type_simple"})
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            df.to_csv(f, index=False)
            tmp_path = f.name
        cfg = replace(
            scenario_config,
            use_precomputed=False,
            projects_info_path=tmp_path,
            infrastructure_config_path=_INFRASTRUCTURE_CONFIG,
            industries_config_path=_INDUSTRIES_CONFIG,
            methane_scenarios=[MethaneScenario("base", 86, 0.035)],
        )
        result = compute_scenarios_full_pipeline(cfg)
        assert "product" in result.columns
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Config from_yaml / from_dict coverage
# ---------------------------------------------------------------------------


class TestConfigFromYaml:
    def test_scenario_config_from_yaml(self):
        cfg = ScenarioConfig.from_yaml(_SCENARIOS_CONFIG)
        assert isinstance(cfg, ScenarioConfig)
        assert cfg.use_precomputed is False

    def test_industrial_emissions_config_from_dict(self, project_config):
        from dataclasses import asdict

        cc = asdict(project_config.constants)
        h2_industry = {
            "simple_product_name": "hydrogen",
            "industrial_product": "hydrogen",
            "unit": "ton H2",
            "process_tco2_per_unit_product": 4.62,
            "combustion_tco2_per_unit_product": 3.816,
            "process_tch4_throughput_per_unit_product": 1.68,
            "combustion_tch4_throughput_per_unit_product": 1.387636,
            "ccs_applied_to": "process",
            "co2_stream_capture_rate": 0.90,
        }
        d = {"constants": cc, "industries": [h2_industry]}
        ie_cfg = IndustrialEmissionsConfig.from_dict(d)
        assert len(ie_cfg.industries) == 1
        assert ie_cfg.industries[0].simple_product_name == "hydrogen"

    def test_industrial_emissions_config_from_yaml(self, project_config):
        import tempfile
        from dataclasses import asdict

        import yaml

        cc = asdict(project_config.constants)
        h2_industry = {
            "simple_product_name": "hydrogen",
            "industrial_product": "hydrogen",
            "unit": "ton H2",
            "process_tco2_per_unit_product": 4.62,
            "combustion_tco2_per_unit_product": 3.816,
            "process_tch4_throughput_per_unit_product": 1.68,
            "combustion_tch4_throughput_per_unit_product": 1.387636,
            "ccs_applied_to": "process",
            "co2_stream_capture_rate": 0.90,
        }
        d = {"constants": cc, "industries": [h2_industry]}
        with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
            yaml.dump(d, f)
            tmp_path = f.name
        ie_cfg = IndustrialEmissionsConfig.from_yaml(tmp_path)
        assert isinstance(ie_cfg, IndustrialEmissionsConfig)
        assert len(ie_cfg.industries) == 1
