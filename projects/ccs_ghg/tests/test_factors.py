"""Unit tests for the CCS emissions factors computation functions.

Run with: pytest projects/ccs_ghg/tests/test_factors.py
"""

import numpy as np
import pytest

from projects.ccs_ghg.ccs_emissions.capture import compute_capture_emissions
from projects.ccs_ghg.ccs_emissions.energy_emissions import (
    compute_gas_turbine_emissions_per_mwh,
    compute_grid_emissions_per_mwh,
    compute_onsite_gas_emissions_per_mmbtu,
)
from projects.ccs_ghg.config import CCSConstants
from projects.ccs_ghg.constants import MJ_PER_MMBTU, MJ_PER_MWH

# ---------------------------------------------------------------------------
# Fixture: a plausible set of constants for hand-calculations
# ---------------------------------------------------------------------------


@pytest.fixture
def constants():
    return CCSConstants(
        tch4_per_mmbtu_pipeline_gas=0.021101,  # 1055.056 / 50000 (equivalent to old formula)
        gwp_ch4=84.0,
        methane_leak_rate=0.023,
        grid_tco2_per_mwh=0.4,
        grid_tch4_per_mwh=0.064147,  # 0.40 * 7.6 * 1055.056 / 50000 (equivalent to old formula)
        eor_gas_turbine_efficiency=0.30,
        oil_tco2_per_bbl=0.43,
        recovery_factor_bbl_per_tco2=2.5,
        natural_gas_tco2_per_mmbtu=0.05306,
        coal_tco2_per_mmbtu=0.09772,
        case_name="test_case",
    )


NG_TCO2_PER_MMBTU = 0.05306


# ---------------------------------------------------------------------------
# Grid emissions
# ---------------------------------------------------------------------------


class TestGridEmissions:
    def test_returns_positive_floats(self, constants):
        tco2, tco2e = compute_grid_emissions_per_mwh(constants)
        assert isinstance(tco2, float)
        assert isinstance(tco2e, float)
        assert tco2 > 0
        assert tco2e > 0

    def test_hand_calculation(self, constants):
        """Verify against a manual step-by-step calculation."""
        expected_fugitive = (
            constants.grid_tch4_per_mwh
            * constants.gwp_ch4
            * constants.methane_leak_rate
        )
        tco2, tco2e = compute_grid_emissions_per_mwh(constants)
        assert tco2 == pytest.approx(constants.grid_tco2_per_mwh, rel=1e-9)
        assert tco2e == pytest.approx(
            constants.grid_tco2_per_mwh + expected_fugitive, rel=1e-9
        )

    def test_zero_leak_rate_tco2e_equals_tco2(self, constants):
        constants.methane_leak_rate = 0.0
        tco2, tco2e = compute_grid_emissions_per_mwh(constants)
        assert tco2e == tco2

    def test_tco2e_greater_than_tco2(self, constants):
        """tCO2e should always be >= tCO2 because it adds fugitive CH4."""
        tco2, tco2e = compute_grid_emissions_per_mwh(constants)
        assert tco2e >= tco2

    def test_texas_grid_emissions(self):
        """Pin the grid emissions calculation to realistic TX (EPA eGRID) values."""
        tx = CCSConstants(
            gwp_ch4=82.5,
            methane_leak_rate=0.023,
            grid_tco2_per_mwh=0.34849,
            grid_tch4_per_mwh=0.084390,
            tch4_per_mmbtu_pipeline_gas=0.01827,
            eor_gas_turbine_efficiency=0.33,
            oil_tco2_per_bbl=0.43,
            recovery_factor_bbl_per_tco2=3,
            natural_gas_tco2_per_mmbtu=0.05306,
            coal_tco2_per_mmbtu=0.09772,
        )

        tco2, tco2e = compute_grid_emissions_per_mwh(tx)
        assert tco2 == pytest.approx(0.34849, rel=1e-9)

        fugitive = tx.grid_tch4_per_mwh * tx.methane_leak_rate * tx.gwp_ch4
        assert tco2e == pytest.approx(tx.grid_tco2_per_mwh + fugitive, rel=1e-9)


# ---------------------------------------------------------------------------
# Gas turbine emissions
# ---------------------------------------------------------------------------


class TestGasTurbineEmissions:
    def test_returns_two_positive_floats(self, constants):
        tco2, tco2e = compute_gas_turbine_emissions_per_mwh(
            constants, NG_TCO2_PER_MMBTU
        )
        assert tco2 > 0
        assert tco2e > 0

    def test_tco2e_greater_than_tco2(self, constants):
        tco2, tco2e = compute_gas_turbine_emissions_per_mwh(
            constants, NG_TCO2_PER_MMBTU
        )
        assert tco2e > tco2

    def test_higher_efficiency_lowers_emissions(self, constants):
        tco2_1, tco2e_1 = compute_gas_turbine_emissions_per_mwh(
            constants, NG_TCO2_PER_MMBTU
        )
        constants.eor_gas_turbine_efficiency = 0.50
        tco2_2, tco2e_2 = compute_gas_turbine_emissions_per_mwh(
            constants, NG_TCO2_PER_MMBTU
        )
        assert tco2_2 < tco2_1
        assert tco2e_2 < tco2e_1

    def test_hand_calculation(self, constants):
        mmbtu_per_mwh = MJ_PER_MWH / constants.eor_gas_turbine_efficiency / MJ_PER_MMBTU
        tch4 = mmbtu_per_mwh * constants.tch4_per_mmbtu_pipeline_gas
        expected_tco2 = mmbtu_per_mwh * NG_TCO2_PER_MMBTU
        expected_fugitive = tch4 * constants.methane_leak_rate * constants.gwp_ch4

        tco2, tco2e = compute_gas_turbine_emissions_per_mwh(
            constants, NG_TCO2_PER_MMBTU
        )
        assert tco2 == pytest.approx(expected_tco2, rel=1e-9)
        assert tco2e == pytest.approx(expected_tco2 + expected_fugitive, rel=1e-9)


# ---------------------------------------------------------------------------
# On-site gas emissions
# ---------------------------------------------------------------------------


class TestOnsiteGasEmissions:
    def test_returns_two_positive_floats(self, constants):
        tco2, tco2e = compute_onsite_gas_emissions_per_mmbtu(
            constants, NG_TCO2_PER_MMBTU
        )
        assert tco2 > 0
        assert tco2e > 0

    def test_tco2e_greater_than_tco2(self, constants):
        tco2, tco2e = compute_onsite_gas_emissions_per_mmbtu(
            constants, NG_TCO2_PER_MMBTU
        )
        assert tco2e > tco2

    def test_hand_calculation(self, constants):
        expected_tco2 = NG_TCO2_PER_MMBTU
        expected_fugitive = (
            constants.tch4_per_mmbtu_pipeline_gas
            * constants.methane_leak_rate
            * constants.gwp_ch4
        )

        tco2, tco2e = compute_onsite_gas_emissions_per_mmbtu(
            constants, NG_TCO2_PER_MMBTU
        )
        assert tco2 == pytest.approx(expected_tco2, rel=1e-9)
        assert tco2e == pytest.approx(expected_tco2 + expected_fugitive, rel=1e-9)

    def test_zero_leak_rate_tco2e_equals_tco2(self, constants):
        constants.methane_leak_rate = 0.0
        tco2, tco2e = compute_onsite_gas_emissions_per_mmbtu(
            constants, NG_TCO2_PER_MMBTU
        )
        assert tco2e == tco2


# ---------------------------------------------------------------------------
# Capture emissions — fillna branch for pre-existing tco2_emitted column
# ---------------------------------------------------------------------------


class TestCaptureEmissionsFillna:
    def test_preexisting_tco2_column_fillna(self, constants):
        """When capture data includes tco2_emitted_per_tco2_captured with NaN,
        the NaN rows should be filled with the computed value."""
        grid_tco2, grid_tco2e = compute_grid_emissions_per_mwh(constants)
        capture_data = [
            {
                "industry": "hydrogen",
                "distance": "near",
                "primary_case": 1,
                "tco2_available_in_capture_stream": 404700.0,
                "co2_stream_capture_rate": 0.90,
                "captured_tco2_per_yr": np.nan,
                "power_for_capture_mw": 4.04,
                "natural_gas_mmbtu_per_day": 2330.0,
                "mwh_needed_per_tco2_captured": np.nan,
                "mmbtu_methane_needed_per_tco2_captured": np.nan,
                "tco2_emitted_per_tco2_captured": np.nan,
            }
        ]
        result = compute_capture_emissions(
            capture_data,
            constants,
            grid_tco2_per_mwh=grid_tco2,
            grid_tco2e_per_mwh=grid_tco2e,
            natural_gas_tco2_per_mmbtu=NG_TCO2_PER_MMBTU,
        )
        assert len(result) == 1
        assert result.iloc[0]["tco2_emitted_per_tco2_captured"] > 0
        assert not np.isnan(result.iloc[0]["tco2_emitted_per_tco2_captured"])
