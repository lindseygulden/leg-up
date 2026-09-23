"""Full-pipeline regression test using the hand_test config for hydrogen.

Verifies every stage from industrial emissions through CCS factors to net
project emissions against hand-computed expected values.

Plant: 1000 tH2/yr hydrogen SMR facility.
Tests both 100% geologic storage (GS) and 100% enhanced oil recovery (EOR).
CH4 assumptions: GWP20 = 82.5, leak rate = 2.3% (from config_infrastructure.yml).

Note: real hydrogen SMR plants do not tend to have direct CH4 emissions, but
the hand_test config assigns nonzero direct_process_tch4 and
direct_combustion_tch4 values so that we can test the full pipeline's
handling of the direct-CH4 code path.

Run with: pytest projects/ccs_ghg/tests/test_hand_regression.py -v
"""

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from projects.ccs_ghg.ccs_emissions.compute import compute_ccs_emissions_factors
from projects.ccs_ghg.ccs_emissions.energy_emissions import (
    compute_gas_turbine_emissions_per_mwh,
    compute_grid_emissions_per_mwh,
    compute_onsite_gas_emissions_per_mmbtu,
)
from projects.ccs_ghg.config import CCSConstants, CCSProjectConfig, IndustryConfig
from projects.ccs_ghg.constants import (
    DAYS_PER_YEAR,
    HOURS_PER_DAY,
    MJ_PER_MMBTU,
    MJ_PER_MWH,
)
from projects.ccs_ghg.emissions_scenarios.config import ScenarioConfig
from projects.ccs_ghg.emissions_scenarios.scenarios import compute_scenarios
from projects.ccs_ghg.industrial_emissions.compute import (
    compute_grid_tch4_per_mwh,
    compute_industrial_emissions,
    compute_industry_row,
)
from projects.ccs_ghg.industrial_emissions.config import IndustrialEmissionsConfig
from utils.io import yaml_to_dict

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_INFRASTRUCTURE_CONFIG = str(_CONFIG_DIR / "config_infrastructure.yml")
_HAND_TEST_CONFIG = str(_CONFIG_DIR / "config_industries_hand_test.yml")

# Constants from config_infrastructure.yml
GWP = 82.5
LEAK = 0.023

# Hand-test emission profile inputs
PROCESS_TCO2 = 5.0
COMBUSTION_TCO2 = 4.0
PROCESS_TCH4 = 1.0
COMBUSTION_TCH4 = 0.9
DIRECT_PROCESS_TCH4 = 0.001
DIRECT_COMBUSTION_TCH4 = 0.1
ELECTRICITY_MWH = 0.5
PROCESS_CAPTURE_FRACTION = 0.90

# Hand-test capture case inputs
CO2_STREAM_CAPTURE_RATE = 0.90
NATURAL_GAS_MMBTU_PER_DAY = 2330
POWER_FOR_CAPTURE_MW = 4.04
TCO2_AVAILABLE_IN_CAPTURE_STREAM = 404700

# Plant size
PRODUCTION_T_PER_YR = 1000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def constants():
    # Hand-regression fixtures use project_state=TX; load TX-specific overrides.
    infra = yaml_to_dict(_INFRASTRUCTURE_CONFIG)
    merged = {
        **infra["constants"],
        **infra.get("emission_factors", {}),
        **infra["states"]["TX"],
    }
    return CCSConstants.from_dict(merged)


@pytest.fixture
def h2_config():
    ind_raw = yaml_to_dict(_HAND_TEST_CONFIG)
    return IndustryConfig.from_dict("hydrogen", ind_raw["hydrogen"])


@pytest.fixture
def project_config(constants, h2_config):
    infra = yaml_to_dict(_INFRASTRUCTURE_CONFIG)
    return CCSProjectConfig(
        constants=constants,
        industries={"hydrogen": h2_config},
        transport=infra["transport"],
        storage=infra["storage"],
    )


@pytest.fixture
def industry_row(h2_config, constants):
    return compute_industry_row(h2_config.emission_profiles[0], constants)


@pytest.fixture
def ccs_factors_df(project_config):
    return compute_ccs_emissions_factors(project_config)


@pytest.fixture
def h2_gs_factors(ccs_factors_df):
    mask = (ccs_factors_df["co2_source"] == "hydrogen") & (
        ccs_factors_df["co2_destination"] == "gs"
    )
    return ccs_factors_df.loc[mask].iloc[0]


# ---------------------------------------------------------------------------
# Part 1: Unabated industrial emissions
# ---------------------------------------------------------------------------


class TestUnabatedEmissions:
    """Verify per-unit and total unabated tCO2e for the hand-test hydrogen plant."""

    def test_process_co2e(self, industry_row):
        # process_co2e = process_tco2 + leak*gwp*throughput_ch4 + gwp*direct_ch4
        expected = PROCESS_TCO2 + LEAK * GWP * PROCESS_TCH4 + GWP * DIRECT_PROCESS_TCH4
        assert industry_row["process_tco2e_per_unit_product"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_combustion_co2e(self, industry_row):
        expected = (
            COMBUSTION_TCO2
            + LEAK * GWP * COMBUSTION_TCH4
            + GWP * DIRECT_COMBUSTION_TCH4
        )
        assert industry_row["combustion_tco2e_per_unit_product"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_grid_co2e(self, industry_row, constants):
        grid_tch4 = compute_grid_tch4_per_mwh(constants) * ELECTRICITY_MWH
        expected = (
            ELECTRICITY_MWH * constants.grid_tco2_per_mwh + LEAK * GWP * grid_tch4
        )
        assert industry_row["grid_tco2e_per_unit_product"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_total_tco2e_per_unit(self, industry_row):
        expected = (
            industry_row["process_tco2e_per_unit_product"]
            + industry_row["combustion_tco2e_per_unit_product"]
            + industry_row["grid_tco2e_per_unit_product"]
        )
        assert industry_row["total_tco2e_per_unit_product"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_total_unabated_tco2e(self, industry_row, constants):
        # Derived: unabated per-unit tCO2e times annual production.
        # Per-unit tCO2e = sum of process/combustion/grid components computed
        # in the tests above, so re-express here from the underlying industry_row.
        expected_per_unit = (
            industry_row["process_tco2e_per_unit_product"]
            + industry_row["combustion_tco2e_per_unit_product"]
            + industry_row["grid_tco2e_per_unit_product"]
        )
        total = industry_row["total_tco2e_per_unit_product"] * PRODUCTION_T_PER_YR
        assert total == pytest.approx(expected_per_unit * PRODUCTION_T_PER_YR, rel=1e-9)

    def test_adjusted_capture_fraction(self, industry_row):
        # CCS applied to process stream only
        co2_captured = PROCESS_CAPTURE_FRACTION * PROCESS_TCO2
        total_co2 = industry_row["total_tco2_per_unit_product"]
        expected = co2_captured / total_co2
        assert industry_row["adjusted_capture_fraction_tco2"] == pytest.approx(
            expected, rel=1e-9
        )


# ---------------------------------------------------------------------------
# Part 2: CCS emissions factors (hydrogen, geologic storage)
# ---------------------------------------------------------------------------


class TestCCSEmissionsFactors:
    """Verify capture, transport, and storage tCO2e factors for hydrogen/GS."""

    def test_capture_tco2e(self, h2_gs_factors, constants):
        # Electricity for capture
        captured_per_yr = TCO2_AVAILABLE_IN_CAPTURE_STREAM * CO2_STREAM_CAPTURE_RATE
        mwh_per_tco2 = (
            POWER_FOR_CAPTURE_MW * DAYS_PER_YEAR * HOURS_PER_DAY / captured_per_yr
        )
        _, grid_tco2e_per_mwh = compute_grid_emissions_per_mwh(constants)
        elec_tco2e = mwh_per_tco2 * grid_tco2e_per_mwh

        # On-site gas combustion for capture
        mmbtu_per_tco2 = NATURAL_GAS_MMBTU_PER_DAY * DAYS_PER_YEAR / captured_per_yr
        _, onsite_tco2e_per_mmbtu = compute_onsite_gas_emissions_per_mmbtu(
            constants, constants.natural_gas_tco2_per_mmbtu
        )
        gas_tco2e = mmbtu_per_tco2 * onsite_tco2e_per_mmbtu

        # Escaped CO2
        escaped = (TCO2_AVAILABLE_IN_CAPTURE_STREAM - captured_per_yr) / captured_per_yr

        expected = elec_tco2e + gas_tco2e + escaped
        assert h2_gs_factors["tco2e_emitted_per_tco2_captured"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_transport_tco2e(self, h2_gs_factors, constants):
        # near distance: 0.09 MWh/tCO2, grid-powered
        _, grid_tco2e_per_mwh = compute_grid_emissions_per_mwh(constants)
        expected = 0.09 * grid_tco2e_per_mwh
        assert h2_gs_factors["tco2e_emitted_per_tco2_transported"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_storage_tco2e(self, h2_gs_factors, constants):
        # GS: 0.2847 MWh/tCO2, grid-powered
        _, grid_tco2e_per_mwh = compute_grid_emissions_per_mwh(constants)
        expected = 0.2847 * grid_tco2e_per_mwh
        assert h2_gs_factors["tco2e_emitted_per_tco2_stored"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_total_lifecycle_tco2e(self, h2_gs_factors):
        expected = (
            h2_gs_factors["tco2e_emitted_per_tco2_captured"]
            + h2_gs_factors["tco2e_emitted_per_tco2_transported"]
            + h2_gs_factors["tco2e_emitted_per_tco2_stored"]
        )
        assert h2_gs_factors["tco2e_emitted_per_tco2_ccs"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_net_factor_is_lifecycle_minus_one(self, h2_gs_factors):
        expected = h2_gs_factors["tco2e_emitted_per_tco2_ccs"] - 1.0
        assert h2_gs_factors["net_tco2e_emitted_per_tco2_ccs"] == pytest.approx(
            expected, rel=1e-9
        )


# ---------------------------------------------------------------------------
# Part 3: Project-level captured carbon and net emissions
# ---------------------------------------------------------------------------


class TestProjectNetEmissions:
    """End-to-end: 1000 tH2/yr plant with 100% geologic storage."""

    @pytest.fixture
    def scenario_result(self, project_config, ccs_factors_df, industry_row):
        ie_config = IndustrialEmissionsConfig.from_project_config(project_config)
        industry_df = compute_industrial_emissions(ie_config)

        total_co2 = industry_row["total_tco2_per_unit_product"]
        unabated_co2 = PRODUCTION_T_PER_YR * total_co2
        total_co2e = industry_row["total_tco2e_per_unit_product"]

        projects_df = pd.DataFrame(
            [
                {
                    "project_name": "hand_test_h2",
                    "project_state": "TX",
                    "project_owner_operator": "test",
                    "project_status": "operating",
                    "lat_degrees_n": 29.0,
                    "lon_degrees_e": -95.0,
                    "lat_lon_confidence": "high",
                    "carbon_fate": "gs",
                    "project_announced": 2024,
                    "project_start": 2026,
                    "product": "hydrogen",
                    "product_amount": PRODUCTION_T_PER_YR,
                    "product_unit": "ton H2/yr",
                    "nameplate_captured_mtco2_per_yr": PRODUCTION_T_PER_YR
                    * PROCESS_CAPTURE_FRACTION
                    * PROCESS_TCO2
                    / 1e6,
                    "unabated_mtco2_per_yr": unabated_co2 / 1e6,
                    "unabated_mtco2e_per_yr": total_co2e * PRODUCTION_T_PER_YR / 1e6,
                }
            ]
        )

        scenario_config = ScenarioConfig.from_dict(
            {
                "emissions_for_industries_path": None,
                "ccs_emissions_factors_path": None,
                "projects_info_path": None,
                "use_precomputed": True,
                "industrial_product_with_detail_col_name": "industrial_product",
                "product_col_name": "simple_product_name",
                "adjusted_capt_frac_col_name": "adjusted_capture_fraction_tco2e",
                "industry_emissions_cols": [
                    "industrial_product",
                    "simple_product_name",
                    "adjusted_capture_fraction_tco2e",
                    "adjusted_capture_fraction_tco2",
                ],
                "industry_emissions_which_flavor": ["hydrogen"],
                "facility_keep_cols": [
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
                    "product",
                    "product_amount",
                    "product_unit",
                    "nameplate_captured_mtco2_per_yr",
                    "unabated_mtco2_per_yr",
                    "unabated_mtco2e_per_yr",
                ],
                "nameplate_ccs_col": "nameplate_captured_mtco2_per_yr",
                "project_start_col": "project_start",
                "project_unabated_mtco2e_col": "unabated_mtco2e_per_yr",
                "projects_merge_on": ["product", "carbon_fate"],
                "ccs_emissions_merge_on": ["product", "stated_co2_destination"],
                "weighted_scenarios": [
                    {
                        "name": "all_gs",
                        "plotting_name": "100% GS",
                        "destination_pct": {"eor": 0, "gs": 100},
                    }
                ],
                "include_nameplate_case": False,
                "methane_scenarios": [],
                "usd_per_tco2_45q_eor": 85,
                "usd_per_tco2_45q_gs": 85,
                "recovery_factor_bbl_per_tco2": 3,
                "discount_rate": 0.051,
                "escalation_year": 2027,
                "inflation_rate": 0.025,
            }
        )

        return compute_scenarios(
            projects_df=projects_df,
            emissions_factors_df=ccs_factors_df,
            scenario_config=scenario_config,
            industry_df=industry_df,
        )

    @staticmethod
    def _gs_row(scenario_result):
        return scenario_result[
            (scenario_result["scenario"] == "all_gs")
            & (scenario_result["stated_co2_destination"] == "gs")
        ].iloc[0]

    def test_single_scenario_row(self, scenario_result):
        gs_rows = scenario_result[
            (scenario_result["scenario"] == "all_gs")
            & (scenario_result["stated_co2_destination"] == "gs")
        ]
        assert len(gs_rows) == 1

    def test_captured_co2(self, scenario_result):
        row = self._gs_row(scenario_result)
        expected = PRODUCTION_T_PER_YR * PROCESS_CAPTURE_FRACTION * PROCESS_TCO2 / 1e6
        assert row["mtco2_per_yr_captured_with_ccs"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_ccs_caused_emissions(self, scenario_result, h2_gs_factors):
        row = self._gs_row(scenario_result)
        captured = row["mtco2_per_yr_captured_with_ccs"]
        expected = captured * h2_gs_factors["tco2e_emitted_per_tco2_ccs"]
        assert row["mtco2e_per_yr_emitted_because_of_ccs"] == pytest.approx(
            expected, rel=1e-6
        )

    def test_net_emissions(self, scenario_result, h2_gs_factors, industry_row):
        row = self._gs_row(scenario_result)
        unabated_co2e = (
            industry_row["total_tco2e_per_unit_product"] * PRODUCTION_T_PER_YR / 1e6
        )
        captured = row["mtco2_per_yr_captured_with_ccs"]
        lifecycle = h2_gs_factors["tco2e_emitted_per_tco2_ccs"]
        expected = unabated_co2e - captured + captured * lifecycle
        assert row["mtco2e_per_yr_emissions_including_ccs"] == pytest.approx(
            expected, rel=1e-6
        )

    def test_net_less_than_unabated(self, scenario_result):
        row = self._gs_row(scenario_result)
        assert (
            row["mtco2e_per_yr_emissions_including_ccs"] < row["unabated_mtco2e_per_yr"]
        )


# ---------------------------------------------------------------------------
# Part 4: CCS emissions factors (hydrogen, EOR)
# ---------------------------------------------------------------------------

# EOR storage config inputs (from config_infrastructure.yml)
EOR_MWH_PER_TCO2_STORED = 0.25
EOR_GAS_TURBINE_EFFICIENCY = 0.33
OIL_TCO2_PER_BBL = 0.43
RECOVERY_FACTOR_BBL_PER_TCO2 = 3


@pytest.fixture
def h2_eor_factors(ccs_factors_df):
    mask = (ccs_factors_df["co2_source"] == "hydrogen") & (
        ccs_factors_df["co2_destination"] == "eor"
    )
    return ccs_factors_df.loc[mask].iloc[0]


class TestEORCCSEmissionsFactors:
    """Verify capture, transport, and storage tCO2e factors for hydrogen/EOR."""

    def test_capture_tco2e_same_as_gs(self, h2_eor_factors, h2_gs_factors):
        assert h2_eor_factors["tco2e_emitted_per_tco2_captured"] == pytest.approx(
            h2_gs_factors["tco2e_emitted_per_tco2_captured"], rel=1e-9
        )

    def test_transport_tco2e_same_as_gs(self, h2_eor_factors, h2_gs_factors):
        assert h2_eor_factors["tco2e_emitted_per_tco2_transported"] == pytest.approx(
            h2_gs_factors["tco2e_emitted_per_tco2_transported"], rel=1e-9
        )

    def test_gas_turbine_tco2_per_mwh(self, constants):
        mmbtu_per_mwh = MJ_PER_MWH / EOR_GAS_TURBINE_EFFICIENCY / MJ_PER_MMBTU
        expected_tco2 = mmbtu_per_mwh * constants.natural_gas_tco2_per_mmbtu
        tco2, _ = compute_gas_turbine_emissions_per_mwh(
            constants, constants.natural_gas_tco2_per_mmbtu
        )
        assert tco2 == pytest.approx(expected_tco2, rel=1e-9)

    def test_gas_turbine_tco2e_per_mwh(self, constants):
        mmbtu_per_mwh = MJ_PER_MWH / EOR_GAS_TURBINE_EFFICIENCY / MJ_PER_MMBTU
        tch4_per_mwh = mmbtu_per_mwh * constants.tch4_per_mmbtu_pipeline_gas
        combustion_tco2 = mmbtu_per_mwh * constants.natural_gas_tco2_per_mmbtu
        fugitive_tco2e = tch4_per_mwh * constants.methane_leak_rate * constants.gwp_ch4
        expected_tco2e = combustion_tco2 + fugitive_tco2e
        _, tco2e = compute_gas_turbine_emissions_per_mwh(
            constants, constants.natural_gas_tco2_per_mmbtu
        )
        assert tco2e == pytest.approx(expected_tco2e, rel=1e-9)

    def test_storage_tco2e(self, h2_eor_factors, constants):
        _, turbine_tco2e = compute_gas_turbine_emissions_per_mwh(
            constants, constants.natural_gas_tco2_per_mmbtu
        )
        expected = EOR_MWH_PER_TCO2_STORED * turbine_tco2e
        assert h2_eor_factors["tco2e_emitted_per_tco2_stored"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_storage_tco2e_greater_than_gs(self, h2_eor_factors, h2_gs_factors):
        assert (
            h2_eor_factors["tco2e_emitted_per_tco2_stored"]
            > h2_gs_factors["tco2e_emitted_per_tco2_stored"]
        )

    def test_total_lifecycle_tco2e_includes_oil(self, h2_eor_factors):
        oil_product = OIL_TCO2_PER_BBL * RECOVERY_FACTOR_BBL_PER_TCO2
        expected = (
            h2_eor_factors["tco2e_emitted_per_tco2_captured"]
            + h2_eor_factors["tco2e_emitted_per_tco2_transported"]
            + h2_eor_factors["tco2e_emitted_per_tco2_stored"]
            + oil_product
        )
        assert h2_eor_factors["tco2e_emitted_per_tco2_ccs"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_net_factor_is_lifecycle_minus_one(self, h2_eor_factors):
        expected = h2_eor_factors["tco2e_emitted_per_tco2_ccs"] - 1.0
        assert h2_eor_factors["net_tco2e_emitted_per_tco2_ccs"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_eor_lifecycle_exceeds_one(self, h2_eor_factors):
        assert h2_eor_factors["tco2e_emitted_per_tco2_ccs"] > 1.0

    def test_eor_lifecycle_greater_than_gs(self, h2_eor_factors, h2_gs_factors):
        assert (
            h2_eor_factors["tco2e_emitted_per_tco2_ccs"]
            > h2_gs_factors["tco2e_emitted_per_tco2_ccs"]
        )


# ---------------------------------------------------------------------------
# Part 5: Project-level net emissions (hydrogen, 100% EOR)
# ---------------------------------------------------------------------------


class TestProjectNetEmissionsEOR:
    """End-to-end: 1000 tH2/yr plant with 100% EOR."""

    @pytest.fixture
    def scenario_result(self, project_config, ccs_factors_df, industry_row):
        ie_config = IndustrialEmissionsConfig.from_project_config(project_config)
        industry_df = compute_industrial_emissions(ie_config)

        total_co2 = industry_row["total_tco2_per_unit_product"]
        unabated_co2 = PRODUCTION_T_PER_YR * total_co2
        total_co2e = industry_row["total_tco2e_per_unit_product"]

        projects_df = pd.DataFrame(
            [
                {
                    "project_name": "hand_test_h2_eor",
                    "project_state": "TX",
                    "project_owner_operator": "test",
                    "project_status": "operating",
                    "lat_degrees_n": 29.0,
                    "lon_degrees_e": -95.0,
                    "lat_lon_confidence": "high",
                    "carbon_fate": "eor",
                    "project_announced": 2024,
                    "project_start": 2026,
                    "product": "hydrogen",
                    "product_amount": PRODUCTION_T_PER_YR,
                    "product_unit": "ton H2/yr",
                    "nameplate_captured_mtco2_per_yr": PRODUCTION_T_PER_YR
                    * PROCESS_CAPTURE_FRACTION
                    * PROCESS_TCO2
                    / 1e6,
                    "unabated_mtco2_per_yr": unabated_co2 / 1e6,
                    "unabated_mtco2e_per_yr": total_co2e * PRODUCTION_T_PER_YR / 1e6,
                }
            ]
        )

        scenario_config = ScenarioConfig.from_dict(
            {
                "emissions_for_industries_path": None,
                "ccs_emissions_factors_path": None,
                "projects_info_path": None,
                "use_precomputed": True,
                "industrial_product_with_detail_col_name": "industrial_product",
                "product_col_name": "simple_product_name",
                "adjusted_capt_frac_col_name": "adjusted_capture_fraction_tco2e",
                "industry_emissions_cols": [
                    "industrial_product",
                    "simple_product_name",
                    "adjusted_capture_fraction_tco2e",
                    "adjusted_capture_fraction_tco2",
                ],
                "industry_emissions_which_flavor": ["hydrogen"],
                "facility_keep_cols": [
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
                    "product",
                    "product_amount",
                    "product_unit",
                    "nameplate_captured_mtco2_per_yr",
                    "unabated_mtco2_per_yr",
                    "unabated_mtco2e_per_yr",
                ],
                "nameplate_ccs_col": "nameplate_captured_mtco2_per_yr",
                "project_start_col": "project_start",
                "project_unabated_mtco2e_col": "unabated_mtco2e_per_yr",
                "projects_merge_on": ["product", "carbon_fate"],
                "ccs_emissions_merge_on": ["product", "stated_co2_destination"],
                "weighted_scenarios": [
                    {
                        "name": "all_eor",
                        "plotting_name": "100% EOR",
                        "destination_pct": {"eor": 100, "gs": 0},
                    }
                ],
                "include_nameplate_case": False,
                "methane_scenarios": [],
                "usd_per_tco2_45q_eor": 85,
                "usd_per_tco2_45q_gs": 85,
                "recovery_factor_bbl_per_tco2": 3,
                "discount_rate": 0.051,
                "escalation_year": 2027,
                "inflation_rate": 0.025,
            }
        )

        return compute_scenarios(
            projects_df=projects_df,
            emissions_factors_df=ccs_factors_df,
            scenario_config=scenario_config,
            industry_df=industry_df,
        )

    @staticmethod
    def _eor_row(scenario_result):
        return scenario_result[
            (scenario_result["scenario"] == "all_eor")
            & (scenario_result["stated_co2_destination"] == "eor")
        ].iloc[0]

    def test_single_scenario_row(self, scenario_result):
        eor_rows = scenario_result[
            (scenario_result["scenario"] == "all_eor")
            & (scenario_result["stated_co2_destination"] == "eor")
        ]
        assert len(eor_rows) == 1

    def test_captured_co2(self, scenario_result):
        row = self._eor_row(scenario_result)
        expected = PRODUCTION_T_PER_YR * PROCESS_CAPTURE_FRACTION * PROCESS_TCO2 / 1e6
        assert row["mtco2_per_yr_captured_with_ccs"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_ccs_caused_emissions(self, scenario_result, h2_eor_factors):
        row = self._eor_row(scenario_result)
        captured = row["mtco2_per_yr_captured_with_ccs"]
        expected = captured * h2_eor_factors["tco2e_emitted_per_tco2_ccs"]
        assert row["mtco2e_per_yr_emitted_because_of_ccs"] == pytest.approx(
            expected, rel=1e-6
        )

    def test_net_emissions(self, scenario_result, h2_eor_factors, industry_row):
        row = self._eor_row(scenario_result)
        unabated_co2e = (
            industry_row["total_tco2e_per_unit_product"] * PRODUCTION_T_PER_YR / 1e6
        )
        captured = row["mtco2_per_yr_captured_with_ccs"]
        lifecycle = h2_eor_factors["tco2e_emitted_per_tco2_ccs"]
        expected = unabated_co2e - captured + captured * lifecycle
        assert row["mtco2e_per_yr_emissions_including_ccs"] == pytest.approx(
            expected, rel=1e-6
        )

    def test_net_exceeds_unabated(self, scenario_result):
        row = self._eor_row(scenario_result)
        assert (
            row["mtco2e_per_yr_emissions_including_ccs"] > row["unabated_mtco2e_per_yr"]
        )


# ---------------------------------------------------------------------------
# Part 6: Two-plant multi-scenario test
#
# Plant A: 1000 tH2/yr, carbon_fate = gs
# Plant B: 2000 tH2/yr, carbon_fate = eor
#
# Scenario 1 (baseline): each plant uses its stated destination
# Scenario 2 (all_gs):   both plants → 100% GS
# Scenario 3 (all_eor):  both plants → 100% EOR
# Scenario 4 (mix_8_92): both plants → 8% GS + 92% EOR
# ---------------------------------------------------------------------------

PLANT_A_PRODUCTION = 1000
PLANT_B_PRODUCTION = 2000


def _make_two_plant_scenario_config(weighted_scenarios):
    return ScenarioConfig.from_dict(
        {
            "emissions_for_industries_path": None,
            "ccs_emissions_factors_path": None,
            "projects_info_path": None,
            "use_precomputed": True,
            "industrial_product_with_detail_col_name": "industrial_product",
            "product_col_name": "simple_product_name",
            "adjusted_capt_frac_col_name": "adjusted_capture_fraction_tco2e",
            "industry_emissions_cols": [
                "industrial_product",
                "simple_product_name",
                "adjusted_capture_fraction_tco2e",
                "adjusted_capture_fraction_tco2",
            ],
            "industry_emissions_which_flavor": ["hydrogen"],
            "facility_keep_cols": [
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
                "product",
                "product_amount",
                "product_unit",
                "nameplate_captured_mtco2_per_yr",
                "unabated_mtco2_per_yr",
                "unabated_mtco2e_per_yr",
            ],
            "nameplate_ccs_col": "nameplate_captured_mtco2_per_yr",
            "project_start_col": "project_start",
            "project_unabated_mtco2e_col": "unabated_mtco2e_per_yr",
            "projects_merge_on": ["product", "carbon_fate"],
            "ccs_emissions_merge_on": ["product", "stated_co2_destination"],
            "weighted_scenarios": weighted_scenarios,
            "include_nameplate_case": False,
            "methane_scenarios": [],
            "usd_per_tco2_45q_eor": 85,
            "usd_per_tco2_45q_gs": 85,
            "recovery_factor_bbl_per_tco2": 3,
            "discount_rate": 0.051,
            "escalation_year": 2027,
            "inflation_rate": 0.025,
        }
    )


def _make_plant_row(name, carbon_fate, production, industry_row):
    total_co2 = industry_row["total_tco2_per_unit_product"]
    total_co2e = industry_row["total_tco2e_per_unit_product"]
    return {
        "project_name": name,
        "project_state": "TX",
        "project_owner_operator": "test",
        "project_status": "operating",
        "lat_degrees_n": 29.0,
        "lon_degrees_e": -95.0,
        "lat_lon_confidence": "high",
        "carbon_fate": carbon_fate,
        "project_announced": 2024,
        "project_start": 2026,
        "product": "hydrogen",
        "product_amount": production,
        "product_unit": "ton H2/yr",
        "nameplate_captured_mtco2_per_yr": production
        * PROCESS_CAPTURE_FRACTION
        * PROCESS_TCO2
        / 1e6,
        "unabated_mtco2_per_yr": production * total_co2 / 1e6,
        "unabated_mtco2e_per_yr": production * total_co2e / 1e6,
    }


class TestTwoPlantMultiScenario:
    """Two hydrogen plants under four destination-mix scenarios."""

    @pytest.fixture
    def two_plant_projects_df(self, industry_row):
        return pd.DataFrame(
            [
                _make_plant_row("plant_a_gs", "gs", PLANT_A_PRODUCTION, industry_row),
                _make_plant_row("plant_b_eor", "eor", PLANT_B_PRODUCTION, industry_row),
            ]
        )

    @pytest.fixture
    def industry_df(self, project_config):
        ie_config = IndustrialEmissionsConfig.from_project_config(project_config)
        return compute_industrial_emissions(ie_config)

    @pytest.fixture
    def scenario_result(self, two_plant_projects_df, ccs_factors_df, industry_df):
        cfg = _make_two_plant_scenario_config(
            [
                {
                    "name": "all_gs",
                    "plotting_name": "100% GS",
                    "destination_pct": {"eor": 0, "gs": 100},
                },
                {
                    "name": "all_eor",
                    "plotting_name": "100% EOR",
                    "destination_pct": {"eor": 100, "gs": 0},
                },
                {
                    "name": "mix_8_92",
                    "plotting_name": "8% GS / 92% EOR",
                    "destination_pct": {"eor": 92, "gs": 8},
                },
            ]
        )
        return compute_scenarios(
            projects_df=two_plant_projects_df,
            emissions_factors_df=ccs_factors_df,
            scenario_config=cfg,
            industry_df=industry_df,
        )

    def _get(self, result, scenario, project, destination):
        mask = (
            (result["scenario"] == scenario)
            & (result["project_name"] == project)
            & (result["stated_co2_destination"] == destination)
        )
        rows = result.loc[mask]
        assert (
            len(rows) == 1
        ), f"Expected 1 row for {scenario}/{project}/{destination}, got {len(rows)}"
        return rows.iloc[0]

    # ------------------------------------------------------------------
    # Baseline: each plant uses its stated destination factor
    # ------------------------------------------------------------------

    def test_baseline_plant_a_uses_gs_factor(self, scenario_result, h2_gs_factors):
        row = self._get(scenario_result, "baseline", "plant_a_gs", "gs")
        assert row["tco2e_emitted_per_tco2_ccs"] == pytest.approx(
            h2_gs_factors["tco2e_emitted_per_tco2_ccs"], rel=1e-9
        )

    def test_baseline_plant_b_uses_eor_factor(self, scenario_result, h2_eor_factors):
        row = self._get(scenario_result, "baseline", "plant_b_eor", "eor")
        assert row["tco2e_emitted_per_tco2_ccs"] == pytest.approx(
            h2_eor_factors["tco2e_emitted_per_tco2_ccs"], rel=1e-9
        )

    def test_baseline_plant_a_captured(self, scenario_result):
        row = self._get(scenario_result, "baseline", "plant_a_gs", "gs")
        expected = PLANT_A_PRODUCTION * PROCESS_CAPTURE_FRACTION * PROCESS_TCO2 / 1e6
        assert row["mtco2_per_yr_captured_with_ccs"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_baseline_plant_b_captured(self, scenario_result):
        row = self._get(scenario_result, "baseline", "plant_b_eor", "eor")
        expected = PLANT_B_PRODUCTION * PROCESS_CAPTURE_FRACTION * PROCESS_TCO2 / 1e6
        assert row["mtco2_per_yr_captured_with_ccs"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_baseline_plant_a_net(self, scenario_result, h2_gs_factors, industry_row):
        row = self._get(scenario_result, "baseline", "plant_a_gs", "gs")
        unabated = (
            industry_row["total_tco2e_per_unit_product"] * PLANT_A_PRODUCTION / 1e6
        )
        captured = row["mtco2_per_yr_captured_with_ccs"]
        expected = (
            unabated - captured + captured * h2_gs_factors["tco2e_emitted_per_tco2_ccs"]
        )
        assert row["mtco2e_per_yr_emissions_including_ccs"] == pytest.approx(
            expected, rel=1e-6
        )

    def test_baseline_plant_b_net(self, scenario_result, h2_eor_factors, industry_row):
        row = self._get(scenario_result, "baseline", "plant_b_eor", "eor")
        unabated = (
            industry_row["total_tco2e_per_unit_product"] * PLANT_B_PRODUCTION / 1e6
        )
        captured = row["mtco2_per_yr_captured_with_ccs"]
        expected = (
            unabated
            - captured
            + captured * h2_eor_factors["tco2e_emitted_per_tco2_ccs"]
        )
        assert row["mtco2e_per_yr_emissions_including_ccs"] == pytest.approx(
            expected, rel=1e-6
        )

    def test_baseline_plant_a_net_below_unabated(self, scenario_result):
        row = self._get(scenario_result, "baseline", "plant_a_gs", "gs")
        assert (
            row["mtco2e_per_yr_emissions_including_ccs"] < row["unabated_mtco2e_per_yr"]
        )

    def test_baseline_plant_b_net_above_unabated(self, scenario_result):
        row = self._get(scenario_result, "baseline", "plant_b_eor", "eor")
        assert (
            row["mtco2e_per_yr_emissions_including_ccs"] > row["unabated_mtco2e_per_yr"]
        )

    # ------------------------------------------------------------------
    # All-GS: both plants use the GS lifecycle factor
    # ------------------------------------------------------------------

    def test_all_gs_both_use_gs_factor(self, scenario_result, h2_gs_factors):
        for proj, dest in [("plant_a_gs", "gs"), ("plant_b_eor", "eor")]:
            row = self._get(scenario_result, "all_gs", proj, dest)
            assert row["tco2e_emitted_per_tco2_ccs"] == pytest.approx(
                h2_gs_factors["tco2e_emitted_per_tco2_ccs"], rel=1e-9
            )

    def test_all_gs_plant_b_net(self, scenario_result, h2_gs_factors, industry_row):
        row = self._get(scenario_result, "all_gs", "plant_b_eor", "eor")
        unabated = (
            industry_row["total_tco2e_per_unit_product"] * PLANT_B_PRODUCTION / 1e6
        )
        captured = row["mtco2_per_yr_captured_with_ccs"]
        expected = (
            unabated - captured + captured * h2_gs_factors["tco2e_emitted_per_tco2_ccs"]
        )
        assert row["mtco2e_per_yr_emissions_including_ccs"] == pytest.approx(
            expected, rel=1e-6
        )

    def test_all_gs_both_below_unabated(self, scenario_result):
        for proj, dest in [("plant_a_gs", "gs"), ("plant_b_eor", "eor")]:
            row = self._get(scenario_result, "all_gs", proj, dest)
            assert (
                row["mtco2e_per_yr_emissions_including_ccs"]
                < row["unabated_mtco2e_per_yr"]
            )

    # ------------------------------------------------------------------
    # All-EOR: both plants use the EOR lifecycle factor
    # ------------------------------------------------------------------

    def test_all_eor_both_use_eor_factor(self, scenario_result, h2_eor_factors):
        for proj, dest in [("plant_a_gs", "gs"), ("plant_b_eor", "eor")]:
            row = self._get(scenario_result, "all_eor", proj, dest)
            assert row["tco2e_emitted_per_tco2_ccs"] == pytest.approx(
                h2_eor_factors["tco2e_emitted_per_tco2_ccs"], rel=1e-9
            )

    def test_all_eor_plant_a_net(self, scenario_result, h2_eor_factors, industry_row):
        row = self._get(scenario_result, "all_eor", "plant_a_gs", "gs")
        unabated = (
            industry_row["total_tco2e_per_unit_product"] * PLANT_A_PRODUCTION / 1e6
        )
        captured = row["mtco2_per_yr_captured_with_ccs"]
        expected = (
            unabated
            - captured
            + captured * h2_eor_factors["tco2e_emitted_per_tco2_ccs"]
        )
        assert row["mtco2e_per_yr_emissions_including_ccs"] == pytest.approx(
            expected, rel=1e-6
        )

    def test_all_eor_both_above_unabated(self, scenario_result):
        for proj, dest in [("plant_a_gs", "gs"), ("plant_b_eor", "eor")]:
            row = self._get(scenario_result, "all_eor", proj, dest)
            assert (
                row["mtco2e_per_yr_emissions_including_ccs"]
                > row["unabated_mtco2e_per_yr"]
            )

    # ------------------------------------------------------------------
    # Mix 8% GS / 92% EOR: weighted lifecycle factor
    # ------------------------------------------------------------------

    def test_mix_weighted_factor(self, scenario_result, h2_gs_factors, h2_eor_factors):
        expected_lf = (
            0.92 * h2_eor_factors["tco2e_emitted_per_tco2_ccs"]
            + 0.08 * h2_gs_factors["tco2e_emitted_per_tco2_ccs"]
        )
        for proj, dest in [("plant_a_gs", "gs"), ("plant_b_eor", "eor")]:
            row = self._get(scenario_result, "mix_8_92", proj, dest)
            assert row["tco2e_emitted_per_tco2_ccs"] == pytest.approx(
                expected_lf, rel=1e-9
            )

    def test_mix_plant_a_net(
        self, scenario_result, h2_gs_factors, h2_eor_factors, industry_row
    ):
        row = self._get(scenario_result, "mix_8_92", "plant_a_gs", "gs")
        weighted_lf = (
            0.92 * h2_eor_factors["tco2e_emitted_per_tco2_ccs"]
            + 0.08 * h2_gs_factors["tco2e_emitted_per_tco2_ccs"]
        )
        unabated = (
            industry_row["total_tco2e_per_unit_product"] * PLANT_A_PRODUCTION / 1e6
        )
        captured = row["mtco2_per_yr_captured_with_ccs"]
        expected = unabated - captured + captured * weighted_lf
        assert row["mtco2e_per_yr_emissions_including_ccs"] == pytest.approx(
            expected, rel=1e-6
        )

    def test_mix_plant_b_net(
        self, scenario_result, h2_gs_factors, h2_eor_factors, industry_row
    ):
        row = self._get(scenario_result, "mix_8_92", "plant_b_eor", "eor")
        weighted_lf = (
            0.92 * h2_eor_factors["tco2e_emitted_per_tco2_ccs"]
            + 0.08 * h2_gs_factors["tco2e_emitted_per_tco2_ccs"]
        )
        unabated = (
            industry_row["total_tco2e_per_unit_product"] * PLANT_B_PRODUCTION / 1e6
        )
        captured = row["mtco2_per_yr_captured_with_ccs"]
        expected = unabated - captured + captured * weighted_lf
        assert row["mtco2e_per_yr_emissions_including_ccs"] == pytest.approx(
            expected, rel=1e-6
        )

    def test_mix_factor_between_gs_and_eor(
        self, scenario_result, h2_gs_factors, h2_eor_factors
    ):
        row = self._get(scenario_result, "mix_8_92", "plant_a_gs", "gs")
        assert (
            row["tco2e_emitted_per_tco2_ccs"]
            > h2_gs_factors["tco2e_emitted_per_tco2_ccs"]
        )
        assert (
            row["tco2e_emitted_per_tco2_ccs"]
            < h2_eor_factors["tco2e_emitted_per_tco2_ccs"]
        )

    # ------------------------------------------------------------------
    # Cross-scenario ordering: net emissions should increase with EOR %
    # ------------------------------------------------------------------

    def test_scenario_ordering_plant_a(self, scenario_result):
        gs = self._get(scenario_result, "all_gs", "plant_a_gs", "gs")
        mix = self._get(scenario_result, "mix_8_92", "plant_a_gs", "gs")
        eor = self._get(scenario_result, "all_eor", "plant_a_gs", "gs")
        net_gs = gs["mtco2e_per_yr_emissions_including_ccs"]
        net_mix = mix["mtco2e_per_yr_emissions_including_ccs"]
        net_eor = eor["mtco2e_per_yr_emissions_including_ccs"]
        assert net_gs < net_mix < net_eor

    def test_scenario_ordering_plant_b(self, scenario_result):
        gs = self._get(scenario_result, "all_gs", "plant_b_eor", "eor")
        mix = self._get(scenario_result, "mix_8_92", "plant_b_eor", "eor")
        eor = self._get(scenario_result, "all_eor", "plant_b_eor", "eor")
        net_gs = gs["mtco2e_per_yr_emissions_including_ccs"]
        net_mix = mix["mtco2e_per_yr_emissions_including_ccs"]
        net_eor = eor["mtco2e_per_yr_emissions_including_ccs"]
        assert net_gs < net_mix < net_eor

    def test_plant_b_double_plant_a_in_each_scenario(self, scenario_result):
        for scenario in ["all_gs", "all_eor", "mix_8_92"]:
            a = self._get(scenario_result, scenario, "plant_a_gs", "gs")
            b = self._get(scenario_result, scenario, "plant_b_eor", "eor")
            assert b["mtco2e_per_yr_emissions_including_ccs"] == pytest.approx(
                2.0 * a["mtco2e_per_yr_emissions_including_ccs"], rel=1e-6
            )


# ---------------------------------------------------------------------------
# Part 7: Combined two-plant scenario totals (tCO2e/yr)
#
# Verifies that summing both plants' emissions across each scenario
# reproduces the hand-computed combined values from the summary table.
# ---------------------------------------------------------------------------


class TestTwoPlantCombinedTotals:
    """Sum both plants' unabated and net tCO2e/yr across four scenarios."""

    @pytest.fixture
    def combined_unabated_tco2e_per_yr(self, industry_row):
        """Sum of both plants' unabated tCO2e/yr derived from the industry row."""
        return (PLANT_A_PRODUCTION + PLANT_B_PRODUCTION) * industry_row[
            "total_tco2e_per_unit_product"
        ]

    @pytest.fixture
    def two_plant_projects_df(self, industry_row):
        return pd.DataFrame(
            [
                _make_plant_row("plant_a_gs", "gs", PLANT_A_PRODUCTION, industry_row),
                _make_plant_row("plant_b_eor", "eor", PLANT_B_PRODUCTION, industry_row),
            ]
        )

    @pytest.fixture
    def industry_df(self, project_config):
        ie_config = IndustrialEmissionsConfig.from_project_config(project_config)
        return compute_industrial_emissions(ie_config)

    @pytest.fixture
    def scenario_result(self, two_plant_projects_df, ccs_factors_df, industry_df):
        cfg = _make_two_plant_scenario_config(
            [
                {
                    "name": "all_gs",
                    "plotting_name": "100% GS",
                    "destination_pct": {"eor": 0, "gs": 100},
                },
                {
                    "name": "all_eor",
                    "plotting_name": "100% EOR",
                    "destination_pct": {"eor": 100, "gs": 0},
                },
                {
                    "name": "mix_8_92",
                    "plotting_name": "8% GS / 92% EOR",
                    "destination_pct": {"eor": 92, "gs": 8},
                },
            ]
        )
        return compute_scenarios(
            projects_df=two_plant_projects_df,
            emissions_factors_df=ccs_factors_df,
            scenario_config=cfg,
            industry_df=industry_df,
        )

    def _combined(self, result, scenario):
        """Sum net and unabated across both plants for a given scenario."""
        rows = result[result["scenario"] == scenario]
        a = rows[
            (rows["project_name"] == "plant_a_gs")
            & (rows["stated_co2_destination"] == "gs")
        ]
        b = rows[
            (rows["project_name"] == "plant_b_eor")
            & (rows["stated_co2_destination"] == "eor")
        ]
        assert len(a) == 1 and len(b) == 1
        net = (
            a.iloc[0]["mtco2e_per_yr_emissions_including_ccs"]
            + b.iloc[0]["mtco2e_per_yr_emissions_including_ccs"]
        )
        unabated = (
            a.iloc[0]["unabated_mtco2e_per_yr"] + b.iloc[0]["unabated_mtco2e_per_yr"]
        )
        return net, unabated

    def test_combined_unabated(self, scenario_result, combined_unabated_tco2e_per_yr):
        _, unabated = self._combined(scenario_result, "baseline")
        assert unabated * 1e6 == pytest.approx(combined_unabated_tco2e_per_yr, rel=1e-6)

    def test_combined_unabated_value(self, scenario_result):
        """Sanity: combined unabated tCO2e is a positive, realistic magnitude."""
        _, unabated = self._combined(scenario_result, "baseline")
        assert 50_000 < unabated * 1e6 < 100_000

    def test_baseline_combined_net(
        self,
        scenario_result,
        h2_gs_factors,
        h2_eor_factors,
        combined_unabated_tco2e_per_yr,
    ):
        net, _ = self._combined(scenario_result, "baseline")
        captured_a = PLANT_A_PRODUCTION * PROCESS_CAPTURE_FRACTION * PROCESS_TCO2
        captured_b = PLANT_B_PRODUCTION * PROCESS_CAPTURE_FRACTION * PROCESS_TCO2
        expected = (
            combined_unabated_tco2e_per_yr
            - (captured_a + captured_b)
            + captured_a * h2_gs_factors["tco2e_emitted_per_tco2_ccs"]
            + captured_b * h2_eor_factors["tco2e_emitted_per_tco2_ccs"]
        )
        assert net * 1e6 == pytest.approx(expected, rel=1e-4)

    def test_baseline_combined_net_value(self, scenario_result):
        """Sanity: baseline net is positive and roughly the same magnitude as unabated."""
        net, unabated = self._combined(scenario_result, "baseline")
        assert 0.9 * unabated < net < 1.5 * unabated

    def test_all_gs_combined_net(
        self, scenario_result, h2_gs_factors, combined_unabated_tco2e_per_yr
    ):
        net, _ = self._combined(scenario_result, "all_gs")
        total_captured = (
            (PLANT_A_PRODUCTION + PLANT_B_PRODUCTION)
            * PROCESS_CAPTURE_FRACTION
            * PROCESS_TCO2
        )
        expected = (
            combined_unabated_tco2e_per_yr
            - total_captured
            + total_captured * h2_gs_factors["tco2e_emitted_per_tco2_ccs"]
        )
        assert net * 1e6 == pytest.approx(expected, rel=1e-4)

    def test_all_gs_combined_net_value(self, scenario_result):
        """Sanity: all-GS net is less than unabated (CCS reduces emissions)."""
        net, unabated = self._combined(scenario_result, "all_gs")
        assert net < unabated

    def test_all_eor_combined_net(
        self, scenario_result, h2_eor_factors, combined_unabated_tco2e_per_yr
    ):
        net, _ = self._combined(scenario_result, "all_eor")
        total_captured = (
            (PLANT_A_PRODUCTION + PLANT_B_PRODUCTION)
            * PROCESS_CAPTURE_FRACTION
            * PROCESS_TCO2
        )
        expected = (
            combined_unabated_tco2e_per_yr
            - total_captured
            + total_captured * h2_eor_factors["tco2e_emitted_per_tco2_ccs"]
        )
        assert net * 1e6 == pytest.approx(expected, rel=1e-4)

    def test_all_eor_combined_net_value(self, scenario_result):
        """Sanity: all-EOR net exceeds unabated (EOR-derived oil dominates)."""
        net, unabated = self._combined(scenario_result, "all_eor")
        assert net > unabated

    def test_mix_8_92_combined_net(
        self,
        scenario_result,
        h2_gs_factors,
        h2_eor_factors,
        combined_unabated_tco2e_per_yr,
    ):
        net, _ = self._combined(scenario_result, "mix_8_92")
        weighted_lf = (
            0.92 * h2_eor_factors["tco2e_emitted_per_tco2_ccs"]
            + 0.08 * h2_gs_factors["tco2e_emitted_per_tco2_ccs"]
        )
        total_captured = (
            (PLANT_A_PRODUCTION + PLANT_B_PRODUCTION)
            * PROCESS_CAPTURE_FRACTION
            * PROCESS_TCO2
        )
        expected = (
            combined_unabated_tco2e_per_yr
            - total_captured
            + total_captured * weighted_lf
        )
        assert net * 1e6 == pytest.approx(expected, rel=1e-4)

    def test_mix_8_92_combined_net_value(self, scenario_result):
        """Sanity: mix (mostly EOR) net exceeds unabated but less than pure EOR."""
        net, unabated = self._combined(scenario_result, "mix_8_92")
        net_eor, _ = self._combined(scenario_result, "all_eor")
        assert unabated < net < net_eor

    def test_only_all_gs_reduces_combined_emissions(self, scenario_result):
        for scenario, should_reduce in [
            ("all_gs", True),
            ("baseline", False),
            ("all_eor", False),
            ("mix_8_92", False),
        ]:
            net, unabated = self._combined(scenario_result, scenario)
            if should_reduce:
                assert net < unabated, f"{scenario}: expected net < unabated"
            else:
                assert net > unabated, f"{scenario}: expected net > unabated"

    def test_combined_net_ordering(self, scenario_result):
        net_gs, _ = self._combined(scenario_result, "all_gs")
        net_mix, _ = self._combined(scenario_result, "mix_8_92")
        net_eor, _ = self._combined(scenario_result, "all_eor")
        assert net_gs < net_mix < net_eor


# ============================================================================
# Part 8: LNG — hybrid capture energy path (B1) + direct_process_tch4 (B2)
# ============================================================================
#
# Tests the unique LNG code path where:
#   - mmbtu_methane_needed_per_tco2_captured is pre-specified (1.7408424)
#   - mwh_needed_per_tco2_captured is computed from power_for_capture_mw
#   - natural_gas_mmbtu_per_day is absent from the capture config
#   - direct_process_tch4 (0.0035) is nonzero while process_tch4_throughput is 0
#
# Uses real config_industries.yml values, 1000 tLNG/yr facility.

_INDUSTRIES_CONFIG = str(_CONFIG_DIR / "config_industries.yml")

# LNG emission profile inputs (from config_industries.yml)
LNG_PROCESS_TCO2 = 0.057
LNG_COMBUSTION_TCO2 = 0.288
LNG_PROCESS_TCH4_THROUGHPUT = 0.0
LNG_COMBUSTION_TCH4_THROUGHPUT = (
    0.099166  # matches config_industries.yml (was 0.1031029407 under old buggy HHV)
)
LNG_DIRECT_PROCESS_TCH4 = 0.0035
LNG_ELECTRICITY_MWH = 0.0

# LNG capture case inputs
LNG_CAPTURE_RATE = 0.99
LNG_MMBTU_PER_TCO2_CAPTURED = 1.7408424
LNG_POWER_FOR_CAPTURE_MW = 6.12
LNG_TCO2_AVAILABLE = 649255

# Facility
LNG_PRODUCTION = 1000


@pytest.fixture
def lng_config():
    ind_raw = yaml_to_dict(_INDUSTRIES_CONFIG)
    return IndustryConfig.from_dict("lng", ind_raw["industries"]["lng"])


@pytest.fixture
def lng_project_config(constants, lng_config):
    infra = yaml_to_dict(_INFRASTRUCTURE_CONFIG)
    return CCSProjectConfig(
        constants=constants,
        industries={"lng": lng_config},
        transport=infra["transport"],
        storage=infra["storage"],
    )


@pytest.fixture
def lng_industry_row(lng_config, constants):
    return compute_industry_row(lng_config.emission_profiles[0], constants)


@pytest.fixture
def lng_ccs_factors_df(lng_project_config):
    return compute_ccs_emissions_factors(lng_project_config)


@pytest.fixture
def lng_gs_factors(lng_ccs_factors_df):
    mask = (lng_ccs_factors_df["co2_source"] == "lng") & (
        lng_ccs_factors_df["co2_destination"] == "gs"
    )
    return lng_ccs_factors_df.loc[mask].iloc[0]


@pytest.fixture
def lng_eor_factors(lng_ccs_factors_df):
    mask = (lng_ccs_factors_df["co2_source"] == "lng") & (
        lng_ccs_factors_df["co2_destination"] == "eor"
    )
    return lng_ccs_factors_df.loc[mask].iloc[0]


class TestLNGIndustrialEmissions:
    """Verify LNG industrial emissions — direct_process_tch4 with zero throughput."""

    def test_process_co2e_uses_direct_ch4_not_throughput(self, lng_industry_row):
        # process_tch4_throughput is 0, so only direct_process_tch4 contributes
        expected = LNG_PROCESS_TCO2 + GWP * LNG_DIRECT_PROCESS_TCH4
        assert lng_industry_row["process_tco2e_per_unit_product"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_combustion_co2e(self, lng_industry_row):
        expected = LNG_COMBUSTION_TCO2 + (LEAK * GWP * LNG_COMBUSTION_TCH4_THROUGHPUT)
        assert lng_industry_row["combustion_tco2e_per_unit_product"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_total_co2(self, lng_industry_row):
        assert lng_industry_row["total_tco2_per_unit_product"] == pytest.approx(
            LNG_PROCESS_TCO2 + LNG_COMBUSTION_TCO2, rel=1e-9
        )

    def test_total_co2e(self, lng_industry_row):
        process_co2e = LNG_PROCESS_TCO2 + GWP * LNG_DIRECT_PROCESS_TCH4
        combustion_co2e = LNG_COMBUSTION_TCO2 + (
            LEAK * GWP * LNG_COMBUSTION_TCH4_THROUGHPUT
        )
        assert lng_industry_row["total_tco2e_per_unit_product"] == pytest.approx(
            process_co2e + combustion_co2e, rel=1e-9
        )

    def test_capture_fraction_tco2(self, lng_industry_row):
        total_co2 = LNG_PROCESS_TCO2 + LNG_COMBUSTION_TCO2
        expected = LNG_CAPTURE_RATE * LNG_PROCESS_TCO2 / total_co2
        assert lng_industry_row["adjusted_capture_fraction_tco2"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_capture_fraction_tco2e(self, lng_industry_row):
        process_co2e = LNG_PROCESS_TCO2 + GWP * LNG_DIRECT_PROCESS_TCH4
        combustion_co2e = LNG_COMBUSTION_TCO2 + (
            LEAK * GWP * LNG_COMBUSTION_TCH4_THROUGHPUT
        )
        total_co2e = process_co2e + combustion_co2e
        co2_captured = LNG_CAPTURE_RATE * LNG_PROCESS_TCO2
        assert lng_industry_row["adjusted_capture_fraction_tco2e"] == pytest.approx(
            co2_captured / total_co2e, rel=1e-9
        )


class TestLNGCaptureHybridPath:
    """Verify the LNG hybrid capture energy path.

    mmbtu_methane_needed_per_tco2_captured is pre-specified (not derived from
    natural_gas_mmbtu_per_day).  mwh_needed_per_tco2_captured is computed from
    power_for_capture_mw (not pre-specified).
    """

    def test_mwh_needed_computed_from_power(self, lng_gs_factors, constants):
        captured_per_yr = LNG_TCO2_AVAILABLE * LNG_CAPTURE_RATE
        mwh_per_yr = LNG_POWER_FOR_CAPTURE_MW * DAYS_PER_YEAR * HOURS_PER_DAY
        expected_mwh = mwh_per_yr / captured_per_yr
        grid_tco2, _ = compute_grid_emissions_per_mwh(constants)
        actual_elec_tco2 = lng_gs_factors["tco2_emitted_per_tco2_captured"] - (
            LNG_MMBTU_PER_TCO2_CAPTURED * constants.natural_gas_tco2_per_mmbtu
            + (LNG_TCO2_AVAILABLE - captured_per_yr) / captured_per_yr
        )
        assert actual_elec_tco2 == pytest.approx(expected_mwh * grid_tco2, rel=1e-9)

    def test_capture_tco2(self, lng_gs_factors, constants):
        captured_per_yr = LNG_TCO2_AVAILABLE * LNG_CAPTURE_RATE
        mwh_per_yr = LNG_POWER_FOR_CAPTURE_MW * DAYS_PER_YEAR * HOURS_PER_DAY
        mwh_needed = mwh_per_yr / captured_per_yr
        grid_tco2, _ = compute_grid_emissions_per_mwh(constants)
        onsite_tco2, _ = compute_onsite_gas_emissions_per_mmbtu(
            constants, constants.natural_gas_tco2_per_mmbtu
        )
        escaped = (LNG_TCO2_AVAILABLE - captured_per_yr) / captured_per_yr
        expected = (
            mwh_needed * grid_tco2 + LNG_MMBTU_PER_TCO2_CAPTURED * onsite_tco2 + escaped
        )
        assert lng_gs_factors["tco2_emitted_per_tco2_captured"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_capture_tco2e(self, lng_gs_factors, constants):
        captured_per_yr = LNG_TCO2_AVAILABLE * LNG_CAPTURE_RATE
        mwh_per_yr = LNG_POWER_FOR_CAPTURE_MW * DAYS_PER_YEAR * HOURS_PER_DAY
        mwh_needed = mwh_per_yr / captured_per_yr
        _, grid_tco2e = compute_grid_emissions_per_mwh(constants)
        _, onsite_tco2e = compute_onsite_gas_emissions_per_mmbtu(
            constants, constants.natural_gas_tco2_per_mmbtu
        )
        escaped = (LNG_TCO2_AVAILABLE - captured_per_yr) / captured_per_yr
        expected = (
            mwh_needed * grid_tco2e
            + LNG_MMBTU_PER_TCO2_CAPTURED * onsite_tco2e
            + escaped
        )
        assert lng_gs_factors["tco2e_emitted_per_tco2_captured"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_escaped_co2_at_99pct_capture(self, lng_gs_factors):
        captured_per_yr = LNG_TCO2_AVAILABLE * LNG_CAPTURE_RATE
        expected = (LNG_TCO2_AVAILABLE - captured_per_yr) / captured_per_yr
        assert expected == pytest.approx(0.010101010101, rel=1e-9)


class TestLNGLifecycleFactors:
    """Verify total lifecycle factors for LNG with GS and EOR."""

    def test_gs_tco2e_lifecycle(self, lng_gs_factors):
        # Sum of the three component factors already verified individually above.
        expected = (
            lng_gs_factors["tco2e_emitted_per_tco2_captured"]
            + lng_gs_factors["tco2e_emitted_per_tco2_transported"]
            + lng_gs_factors["tco2e_emitted_per_tco2_stored"]
        )
        assert lng_gs_factors["tco2e_emitted_per_tco2_ccs"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_gs_net_tco2e(self, lng_gs_factors):
        # Net = lifecycle - 1 (one tCO2 stored per tCO2 captured).
        expected = lng_gs_factors["tco2e_emitted_per_tco2_ccs"] - 1.0
        assert lng_gs_factors["net_tco2e_emitted_per_tco2_ccs"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_gs_is_net_negative(self, lng_gs_factors):
        assert lng_gs_factors["net_tco2e_emitted_per_tco2_ccs"] < 0

    def test_eor_tco2e_lifecycle(self, lng_eor_factors, constants):
        # EOR adds oil-product credit on top of capture+transport+storage.
        oil_product = (
            constants.oil_tco2_per_bbl * constants.recovery_factor_bbl_per_tco2
        )
        expected = (
            lng_eor_factors["tco2e_emitted_per_tco2_captured"]
            + lng_eor_factors["tco2e_emitted_per_tco2_transported"]
            + lng_eor_factors["tco2e_emitted_per_tco2_stored"]
            + oil_product
        )
        assert lng_eor_factors["tco2e_emitted_per_tco2_ccs"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_eor_net_tco2e(self, lng_eor_factors):
        expected = lng_eor_factors["tco2e_emitted_per_tco2_ccs"] - 1.0
        assert lng_eor_factors["net_tco2e_emitted_per_tco2_ccs"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_eor_increases_net_emissions(self, lng_eor_factors):
        assert lng_eor_factors["net_tco2e_emitted_per_tco2_ccs"] > 0

    def test_capture_factors_same_for_gs_and_eor(self, lng_gs_factors, lng_eor_factors):
        assert lng_gs_factors["tco2e_emitted_per_tco2_captured"] == pytest.approx(
            lng_eor_factors["tco2e_emitted_per_tco2_captured"], rel=1e-9
        )


# ============================================================================
# Part 9: Power Coal — lignite fuel type through full pipeline (B5)
# ============================================================================
#
# Tests the lignite code path where combustion_tco2 is derived from heat rate
# using coal_tco2_per_mmbtu (0.09772; lignite EF from 40 CFR 98) instead of natural_gas_tco2_per_mmbtu.
# All CH4 values are zero, so CO2 == CO2e.
# Capture uses pre-specified mwh_needed + mmbtu_methane_needed (like power_ng).

from projects.ccs_ghg.constants import BTU_PER_MMBTU, KWH_PER_MWH

COAL_HEAT_RATE = 10777
COAL_CAPTURE_RATE = 0.9
COAL_MWH_NEEDED = 0.16
COAL_MMBTU_NEEDED = 2.6


@pytest.fixture
def coal_config():
    ind_raw = yaml_to_dict(_INDUSTRIES_CONFIG)
    return IndustryConfig.from_dict("power_coal", ind_raw["industries"]["power_coal"])


@pytest.fixture
def coal_project_config(constants, coal_config):
    infra = yaml_to_dict(_INFRASTRUCTURE_CONFIG)
    return CCSProjectConfig(
        constants=constants,
        industries={"power_coal": coal_config},
        transport=infra["transport"],
        storage=infra["storage"],
    )


@pytest.fixture
def coal_industry_row(coal_config, constants):
    return compute_industry_row(coal_config.emission_profiles[0], constants)


@pytest.fixture
def coal_ccs_factors_df(coal_project_config):
    return compute_ccs_emissions_factors(coal_project_config)


@pytest.fixture
def coal_gs_factors(coal_ccs_factors_df):
    mask = (coal_ccs_factors_df["co2_source"] == "power_coal") & (
        coal_ccs_factors_df["co2_destination"] == "gs"
    )
    return coal_ccs_factors_df.loc[mask].iloc[0]


@pytest.fixture
def coal_eor_factors(coal_ccs_factors_df):
    mask = (coal_ccs_factors_df["co2_source"] == "power_coal") & (
        coal_ccs_factors_df["co2_destination"] == "eor"
    )
    return coal_ccs_factors_df.loc[mask].iloc[0]


class TestPowerCoalIndustrial:
    """Verify lignite heat-rate-derived combustion CO2."""

    def test_combustion_tco2_from_lignite_heat_rate(self, coal_industry_row, constants):
        expected = (
            COAL_HEAT_RATE * constants.coal_tco2_per_mmbtu * KWH_PER_MWH / BTU_PER_MMBTU
        )
        assert coal_industry_row["combustion_tco2_per_unit_product"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_co2_equals_co2e(self, coal_industry_row):
        assert coal_industry_row["total_tco2_per_unit_product"] == pytest.approx(
            coal_industry_row["total_tco2e_per_unit_product"], rel=1e-9
        )

    def test_capture_fraction_equals_stream_rate(self, coal_industry_row):
        assert coal_industry_row["adjusted_capture_fraction_tco2"] == pytest.approx(
            COAL_CAPTURE_RATE, rel=1e-9
        )

    def test_capture_fraction_tco2_equals_tco2e(self, coal_industry_row):
        assert coal_industry_row["adjusted_capture_fraction_tco2"] == pytest.approx(
            coal_industry_row["adjusted_capture_fraction_tco2e"], rel=1e-9
        )


class TestPowerCoalCCSFactors:
    """Verify full CCS pipeline for lignite power through GS and EOR."""

    def test_capture_tco2e(self, coal_gs_factors, constants):
        _, grid_tco2e = compute_grid_emissions_per_mwh(constants)
        _, onsite_tco2e = compute_onsite_gas_emissions_per_mmbtu(
            constants, constants.natural_gas_tco2_per_mmbtu
        )
        tco2_available = 1.0 / COAL_CAPTURE_RATE
        escaped = (tco2_available - 1.0) / 1.0
        expected = (
            COAL_MWH_NEEDED * grid_tco2e + COAL_MMBTU_NEEDED * onsite_tco2e + escaped
        )
        assert coal_gs_factors["tco2e_emitted_per_tco2_captured"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_gs_lifecycle(self, coal_gs_factors):
        # Lifecycle = capture + transport + storage (no oil for GS)
        expected = (
            coal_gs_factors["tco2e_emitted_per_tco2_captured"]
            + coal_gs_factors["tco2e_emitted_per_tco2_transported"]
            + coal_gs_factors["tco2e_emitted_per_tco2_stored"]
        )
        assert coal_gs_factors["tco2e_emitted_per_tco2_ccs"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_gs_net_negative(self, coal_gs_factors):
        assert coal_gs_factors["net_tco2e_emitted_per_tco2_ccs"] < 0

    def test_eor_lifecycle(self, coal_eor_factors, constants):
        # EOR adds oil-product credit on top of capture+transport+storage
        oil_product = (
            constants.oil_tco2_per_bbl * constants.recovery_factor_bbl_per_tco2
        )
        expected = (
            coal_eor_factors["tco2e_emitted_per_tco2_captured"]
            + coal_eor_factors["tco2e_emitted_per_tco2_transported"]
            + coal_eor_factors["tco2e_emitted_per_tco2_stored"]
            + oil_product
        )
        assert coal_eor_factors["tco2e_emitted_per_tco2_ccs"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_eor_net_positive(self, coal_eor_factors):
        assert coal_eor_factors["net_tco2e_emitted_per_tco2_ccs"] > 0


# ============================================================================
# Part 10: Refined Fuels — low capture rate + both direct CH4 types (B3/B4)
# ============================================================================
#
# Tests: process CCS with capture rate 0.53, direct_process_tch4 and
# direct_combustion_tch4 both nonzero, nonzero grid electricity, nonzero
# combustion_tch4_throughput. The 53% capture rate produces a large escaped
# CO2 ratio (~0.887), exercising a distinct regime from hydrogen (0.111).

RF_PROCESS_TCO2 = 0.01141040779
RF_COMBUSTION_TCO2 = 0.01872789534
RF_PROCESS_TCH4_THROUGHPUT = 0.0
RF_COMBUSTION_TCH4_THROUGHPUT = 0.003567314491
RF_DIRECT_PROCESS_TCH4 = 0.0000101992
RF_DIRECT_COMBUSTION_TCH4 = 0.000001028881825
RF_ELECTRICITY_MWH = 0.00641
RF_CAPTURE_RATE = 0.53
RF_MWH_NEEDED = 0.14
RF_MMBTU_NEEDED = 2.6
RF_TCO2_AVAILABLE = 723226.6


@pytest.fixture
def rf_config():
    ind_raw = yaml_to_dict(_INDUSTRIES_CONFIG)
    return IndustryConfig.from_dict(
        "refined_fuels", ind_raw["industries"]["refined_fuels"]
    )


@pytest.fixture
def rf_project_config(constants, rf_config):
    infra = yaml_to_dict(_INFRASTRUCTURE_CONFIG)
    return CCSProjectConfig(
        constants=constants,
        industries={"refined_fuels": rf_config},
        transport=infra["transport"],
        storage=infra["storage"],
    )


@pytest.fixture
def rf_industry_row(rf_config, constants):
    return compute_industry_row(rf_config.emission_profiles[0], constants)


@pytest.fixture
def rf_ccs_factors_df(rf_project_config):
    return compute_ccs_emissions_factors(rf_project_config)


@pytest.fixture
def rf_gs_factors(rf_ccs_factors_df):
    mask = (rf_ccs_factors_df["co2_source"] == "refined_fuels") & (
        rf_ccs_factors_df["co2_destination"] == "gs"
    )
    return rf_ccs_factors_df.loc[mask].iloc[0]


@pytest.fixture
def rf_eor_factors(rf_ccs_factors_df):
    mask = (rf_ccs_factors_df["co2_source"] == "refined_fuels") & (
        rf_ccs_factors_df["co2_destination"] == "eor"
    )
    return rf_ccs_factors_df.loc[mask].iloc[0]


class TestRefinedFuelsIndustrial:
    """Verify refined fuels: both direct CH4 types, grid electricity, process CCS."""

    def test_process_co2e_with_direct_ch4(self, rf_industry_row):
        expected = RF_PROCESS_TCO2 + GWP * RF_DIRECT_PROCESS_TCH4
        assert rf_industry_row["process_tco2e_per_unit_product"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_combustion_co2e_with_throughput_and_direct(self, rf_industry_row):
        expected = RF_COMBUSTION_TCO2 + (
            LEAK * GWP * RF_COMBUSTION_TCH4_THROUGHPUT + GWP * RF_DIRECT_COMBUSTION_TCH4
        )
        assert rf_industry_row["combustion_tco2e_per_unit_product"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_grid_co2e(self, rf_industry_row, constants):
        grid_tch4 = compute_grid_tch4_per_mwh(constants) * RF_ELECTRICITY_MWH
        expected = (
            RF_ELECTRICITY_MWH * constants.grid_tco2_per_mwh + LEAK * GWP * grid_tch4
        )
        assert rf_industry_row["grid_tco2e_per_unit_product"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_adj_cf_tco2(self, rf_industry_row, constants):
        total_co2 = (
            RF_PROCESS_TCO2
            + RF_COMBUSTION_TCO2
            + (RF_ELECTRICITY_MWH * constants.grid_tco2_per_mwh)
        )
        expected = RF_CAPTURE_RATE * RF_PROCESS_TCO2 / total_co2
        assert rf_industry_row["adjusted_capture_fraction_tco2"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_adj_cf_tco2e_less_than_tco2(self, rf_industry_row):
        assert (
            rf_industry_row["adjusted_capture_fraction_tco2e"]
            < rf_industry_row["adjusted_capture_fraction_tco2"]
        )


class TestRefinedFuelsCCSFactors:
    """Verify CCS pipeline for refined fuels — low capture rate, large escaped CO2."""

    def test_escaped_co2_ratio(self, rf_gs_factors, constants):
        captured_per_yr = RF_TCO2_AVAILABLE * RF_CAPTURE_RATE
        escaped = (RF_TCO2_AVAILABLE - captured_per_yr) / captured_per_yr
        assert escaped == pytest.approx(0.886792452830, rel=1e-6)

    def test_capture_tco2e(self, rf_gs_factors, constants):
        _, grid_tco2e = compute_grid_emissions_per_mwh(constants)
        _, onsite_tco2e = compute_onsite_gas_emissions_per_mmbtu(
            constants, constants.natural_gas_tco2_per_mmbtu
        )
        captured_per_yr = RF_TCO2_AVAILABLE * RF_CAPTURE_RATE
        escaped = (RF_TCO2_AVAILABLE - captured_per_yr) / captured_per_yr
        expected = RF_MWH_NEEDED * grid_tco2e + RF_MMBTU_NEEDED * onsite_tco2e + escaped
        assert rf_gs_factors["tco2e_emitted_per_tco2_captured"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_gs_lifecycle(self, rf_gs_factors):
        expected = (
            rf_gs_factors["tco2e_emitted_per_tco2_captured"]
            + rf_gs_factors["tco2e_emitted_per_tco2_transported"]
            + rf_gs_factors["tco2e_emitted_per_tco2_stored"]
        )
        assert rf_gs_factors["tco2e_emitted_per_tco2_ccs"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_gs_net_positive(self, rf_gs_factors):
        assert rf_gs_factors["net_tco2e_emitted_per_tco2_ccs"] > 0

    def test_eor_lifecycle(self, rf_eor_factors, constants):
        oil_product = (
            constants.oil_tco2_per_bbl * constants.recovery_factor_bbl_per_tco2
        )
        expected = (
            rf_eor_factors["tco2e_emitted_per_tco2_captured"]
            + rf_eor_factors["tco2e_emitted_per_tco2_transported"]
            + rf_eor_factors["tco2e_emitted_per_tco2_stored"]
            + oil_product
        )
        assert rf_eor_factors["tco2e_emitted_per_tco2_ccs"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_gs_net_less_than_eor_net(self, rf_gs_factors, rf_eor_factors):
        assert (
            rf_gs_factors["net_tco2e_emitted_per_tco2_ccs"]
            < rf_eor_factors["net_tco2e_emitted_per_tco2_ccs"]
        )


# ============================================================================
# Part 11: NGP — multiple emission profiles per industry (B6)
# ============================================================================
#
# NGP has two emission profiles under the same simple_product_name: "ngp"
# (standard feed gas) and "natural gas processing -- high CO2 gas".
# The industry_emissions_which_flavor config selects which profile is used
# in scenario merges. This test verifies:
#   1. Both profiles produce distinct emissions rows
#   2. The flavor filter selects the correct one
#   3. The high-CO2 profile has dramatically higher emissions


class TestNGPMultipleProfiles:
    """Verify NGP with two emission profiles under the same product name."""

    @pytest.fixture
    def ngp_config(self):
        ind_raw = yaml_to_dict(_INDUSTRIES_CONFIG)
        return IndustryConfig.from_dict("ngp", ind_raw["industries"]["ngp"])

    @pytest.fixture
    def ngp_industry_df(self, ngp_config, constants):
        from projects.ccs_ghg.industrial_emissions.config import (
            IndustrialEmissionsConfig,
        )

        infra = yaml_to_dict(_INFRASTRUCTURE_CONFIG)
        cfg = CCSProjectConfig(
            constants=constants,
            industries={"ngp": ngp_config},
            transport=infra["transport"],
            storage=infra["storage"],
        )
        ie_config = IndustrialEmissionsConfig.from_project_config(cfg)
        return compute_industrial_emissions(ie_config)

    def test_two_rows_produced(self, ngp_industry_df):
        assert len(ngp_industry_df) == 2

    def test_both_share_simple_product_name(self, ngp_industry_df):
        assert (ngp_industry_df["simple_product_name"] == "ngp").all()

    def test_different_industrial_product_names(self, ngp_industry_df):
        names = ngp_industry_df["industrial_product"].tolist()
        assert "ngp" in names
        assert "natural gas processing -- high CO2 gas" in names

    def test_high_co2_has_higher_emissions(self, ngp_industry_df):
        standard = ngp_industry_df.loc[
            ngp_industry_df["industrial_product"] == "ngp",
            "total_tco2e_per_unit_product",
        ].iloc[0]
        high_co2 = ngp_industry_df.loc[
            ngp_industry_df["industrial_product"]
            == "natural gas processing -- high CO2 gas",
            "total_tco2e_per_unit_product",
        ].iloc[0]
        assert high_co2 > 10 * standard

    def test_flavor_filter_selects_standard(self, ngp_industry_df):
        from projects.ccs_ghg.emissions_scenarios.scenarios import (
            load_industry_emissions,
        )

        cfg = ScenarioConfig.from_dict(
            {
                "emissions_for_industries_path": "",
                "projects_info_path": "",
                "ccs_emissions_factors_path": "",
                "industry_emissions_which_flavor": ["ngp"],
            }
        )
        filtered = load_industry_emissions(cfg, industry_df=ngp_industry_df)
        assert len(filtered) == 1

    def test_flavor_filter_selects_high_co2(self, ngp_industry_df):
        from projects.ccs_ghg.emissions_scenarios.scenarios import (
            load_industry_emissions,
        )

        cfg = ScenarioConfig.from_dict(
            {
                "emissions_for_industries_path": "",
                "projects_info_path": "",
                "ccs_emissions_factors_path": "",
                "industry_emissions_which_flavor": [
                    "natural gas processing -- high CO2 gas"
                ],
            }
        )
        filtered = load_industry_emissions(cfg, industry_df=ngp_industry_df)
        assert len(filtered) == 1


# ============================================================================
# Part 12: Ethanol — far transport distance + 0.99 capture rate (B7/B13)
# ============================================================================


class TestEthanolFarTransportAndHighCapture:
    """Verify far transport distance and near-perfect (0.99) capture rate."""

    @pytest.fixture
    def ethanol_config(self):
        ind_raw = yaml_to_dict(_INDUSTRIES_CONFIG)
        return IndustryConfig.from_dict("ethanol", ind_raw["industries"]["ethanol"])

    @pytest.fixture
    def ethanol_ccs_df(self, ethanol_config, constants):
        infra = yaml_to_dict(_INFRASTRUCTURE_CONFIG)
        cfg = CCSProjectConfig(
            constants=constants,
            industries={"ethanol": ethanol_config},
            transport=infra["transport"],
            storage=infra["storage"],
        )
        return compute_ccs_emissions_factors(cfg)

    @pytest.fixture
    def ethanol_gs(self, ethanol_ccs_df):
        mask = (ethanol_ccs_df["co2_source"] == "ethanol") & (
            ethanol_ccs_df["co2_destination"] == "gs"
        )
        return ethanol_ccs_df.loc[mask].iloc[0]

    def test_uses_far_transport(self, ethanol_gs):
        assert ethanol_gs["distance"] == "far"

    def test_far_transport_tco2e(self, ethanol_gs, constants):
        _, grid_tco2e = compute_grid_emissions_per_mwh(constants)
        expected = 0.11 * grid_tco2e
        assert ethanol_gs["tco2e_emitted_per_tco2_transported"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_escaped_co2_at_99pct(self, ethanol_gs):
        assert (
            ethanol_gs["tco2e_emitted_per_tco2_captured"]
            < ethanol_gs["tco2e_emitted_per_tco2_transported"]
            + ethanol_gs["tco2e_emitted_per_tco2_stored"]
            + 0.010102
        )

    def test_gs_lifecycle(self, ethanol_gs):
        expected = (
            ethanol_gs["tco2e_emitted_per_tco2_captured"]
            + ethanol_gs["tco2e_emitted_per_tco2_transported"]
            + ethanol_gs["tco2e_emitted_per_tco2_stored"]
        )
        assert ethanol_gs["tco2e_emitted_per_tco2_ccs"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_gs_net_negative(self, ethanol_gs):
        assert ethanol_gs["net_tco2e_emitted_per_tco2_ccs"] < 0
