"""Unit tests for the industrial emissions computation.

Run with: pytest projects/ccs_ghg/tests/test_industrial_emissions.py
"""

import pytest

from projects.ccs_ghg.config import CCSConstants, IndustryInput
from projects.ccs_ghg.constants import BTU_PER_MMBTU, KWH_PER_MWH
from projects.ccs_ghg.industrial_emissions.compute import (
    compute_grid_tch4_per_mwh,
    compute_industrial_emissions,
    compute_industry_row,
    resolve_combustion_co2_tco2,
)
from projects.ccs_ghg.industrial_emissions.config import IndustrialEmissionsConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GWP20 = 86
LEAK_RATE = 0.035

_CONSTANTS_DEFAULTS = dict(
    grid_tco2_per_mwh=0.37,
    grid_tch4_per_mwh=0.070847,  # 0.433 * 7.754 * 1055.056 / 50000 (numerically equivalent to old formula so hardcoded expected values still pass)
    tch4_per_mmbtu_pipeline_gas=0.021101,  # 1055.056 / 50000 (numerically equivalent to old formula)
    eor_gas_turbine_efficiency=0.33,
    oil_tco2_per_bbl=0.43,
    recovery_factor_bbl_per_tco2=3,
    natural_gas_tco2_per_mmbtu=0.05306,
    coal_tco2_per_mmbtu=0.09772,
)


def _make_constants(**overrides) -> CCSConstants:
    return CCSConstants(**{**_CONSTANTS_DEFAULTS, **overrides})


@pytest.fixture
def cc():
    """Standard conversion constants matching the workbook defaults."""
    return _make_constants(gwp_ch4=GWP20, methane_leak_rate=LEAK_RATE)


@pytest.fixture
def methanol(cc):
    return IndustryInput(
        industrial_product="methanol (ATR)",
        simple_product_name="methanol",
        unit="ton MeOH",
        process_tco2_per_unit_product=0.35,
        combustion_tco2_per_unit_product=0.05,
        process_tch4_throughput_per_unit_product=0.127273,
        combustion_tch4_throughput_per_unit_product=0.018182,
        ccs_applied_to="process",
        process_capture_fraction=0.95,
        combustion_capture_fraction=0.0,
    )


@pytest.fixture
def power_ccgt(cc):
    return IndustryInput(
        industrial_product="power_ng",
        simple_product_name="power_ng",
        unit="MWh",
        process_tco2_per_unit_product=0.0,
        combustion_tco2_per_unit_product=None,
        heat_rate_btu_per_kwh=7548,
        fuel_type="natural_gas",
        process_tch4_throughput_per_unit_product=0.0,
        combustion_tch4_throughput_per_unit_product=0.145636,
        ccs_applied_to="combustion",
        process_capture_fraction=0.0,
        combustion_capture_fraction=0.87,
    )


@pytest.fixture
def dac(cc):
    return IndustryInput(
        industrial_product="carbon credits",
        simple_product_name="dac",
        unit="ton CO2",
        process_tco2_per_unit_product=0.0,
        combustion_tco2_per_unit_product=0.0,
        process_tch4_throughput_per_unit_product=0.0,
        combustion_tch4_throughput_per_unit_product=0.0,
        ccs_applied_to="process",
        process_capture_fraction=0.0,
        combustion_capture_fraction=0.0,
    )


@pytest.fixture
def hydrogen(cc):
    return IndustryInput(
        industrial_product="hydrogen",
        simple_product_name="hydrogen",
        unit="ton H2",
        process_tco2_per_unit_product=4.62,
        combustion_tco2_per_unit_product=3.816,
        process_tch4_throughput_per_unit_product=1.68,
        combustion_tch4_throughput_per_unit_product=1.387636,
        ccs_applied_to="process",
        process_capture_fraction=0.95,
        combustion_capture_fraction=0.0,
    )


@pytest.fixture
def petrochemicals(cc):
    return IndustryInput(
        industrial_product="petrochemical generation",
        simple_product_name="petrochemicals",
        unit="ton ethylene",
        process_tco2_per_unit_product=0.026344,
        combustion_tco2_per_unit_product=0.255532,
        process_tch4_throughput_per_unit_product=0.0,
        combustion_tch4_throughput_per_unit_product=0.0,
        direct_process_tch4_per_unit_product=0.0,
        direct_combustion_tch4_per_unit_product=0.092921,
        ccs_applied_to="combustion",
        process_capture_fraction=0.0,
        combustion_capture_fraction=0.81,
    )


@pytest.fixture
def refined_fuels(cc):
    return IndustryInput(
        industrial_product="refined fuels",
        simple_product_name="refined_fuels",
        unit="bbl/yr",
        process_tco2_per_unit_product=0.01141,
        combustion_tco2_per_unit_product=0.018728,
        process_tch4_throughput_per_unit_product=0.0,
        combustion_tch4_throughput_per_unit_product=0.0,
        direct_process_tch4_per_unit_product=0.0,
        direct_combustion_tch4_per_unit_product=0.0068,
        ccs_applied_to="process",
        process_capture_fraction=0.53,
        combustion_capture_fraction=0.0,
    )


# ---------------------------------------------------------------------------
# Derived combustion tCO2
# ---------------------------------------------------------------------------


class TestResolveCombustionCO2:
    def test_direct_value(self, methanol, cc):
        assert resolve_combustion_co2_tco2(methanol, cc) == 0.05

    def test_derived_from_heat_rate_natural_gas(self, power_ccgt, cc):
        result = resolve_combustion_co2_tco2(power_ccgt, cc)
        expected = 7548 * 0.05306 * KWH_PER_MWH / BTU_PER_MMBTU
        assert result == pytest.approx(expected, rel=1e-9)

    def test_derived_from_heat_rate_lignite(self, cc):
        coal = IndustryInput(
            industrial_product="power (coal-fired)",
            simple_product_name="power_coal",
            unit="MWh",
            process_tco2_per_unit_product=0.0,
            combustion_tco2_per_unit_product=None,
            heat_rate_btu_per_kwh=10777,
            fuel_type="lignite",
            process_tch4_throughput_per_unit_product=0.0,
            combustion_tch4_throughput_per_unit_product=0.0,
            ccs_applied_to="combustion",
            process_capture_fraction=0.0,
            combustion_capture_fraction=0.9,
        )
        result = resolve_combustion_co2_tco2(coal, cc)
        expected = 10777 * 0.09772 * KWH_PER_MWH / BTU_PER_MMBTU
        assert result == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# tCO2e computation
# ---------------------------------------------------------------------------


class TestCO2eComputation:
    def test_methanol_process_tco2e(self, methanol, cc):
        row = compute_industry_row(methanol, cc)
        expected = 0.35 + LEAK_RATE * GWP20 * 0.127273
        assert row["process_tco2e_per_unit_product"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_methanol_combustion_tco2e(self, methanol, cc):
        row = compute_industry_row(methanol, cc)
        expected = 0.05 + LEAK_RATE * GWP20 * 0.018182
        assert row["combustion_tco2e_per_unit_product"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_methanol_total_tco2e(self, methanol, cc):
        row = compute_industry_row(methanol, cc)
        assert row["total_tco2e_per_unit_product"] == pytest.approx(
            row["process_tco2e_per_unit_product"]
            + row["combustion_tco2e_per_unit_product"]
            + row["grid_tco2e_per_unit_product"],
            rel=1e-9,
        )

    def test_zero_electricity_gives_zero_grid(self, methanol, cc):
        row = compute_industry_row(methanol, cc)
        assert row["grid_tco2_per_unit_product"] == 0.0
        assert row["grid_tch4_per_unit_product"] == 0.0
        assert row["grid_tco2e_per_unit_product"] == 0.0

    def test_hydrogen_process_tco2e(self, hydrogen, cc):
        """Hydrogen process matches workbook (9.6768) exactly."""
        row = compute_industry_row(hydrogen, cc)
        assert row["process_tco2e_per_unit_product"] == pytest.approx(9.6768, rel=1e-4)

    def test_dac_all_zeros(self, dac, cc):
        row = compute_industry_row(dac, cc)
        assert row["total_tco2_per_unit_product"] == 0.0
        assert row["total_tco2e_per_unit_product"] == 0.0
        assert row["adjusted_capture_fraction_tco2"] == 0.0
        assert row["adjusted_capture_fraction_tco2e"] == 0.0


# ---------------------------------------------------------------------------
# Capture fractions
# ---------------------------------------------------------------------------


class TestGridEmissions:
    """Tests for the three-way split with non-zero grid electricity."""

    @pytest.fixture
    def ammonia_with_grid(self, cc):
        return IndustryInput(
            industrial_product="ammonia",
            simple_product_name="ammonia",
            unit="ton NH3",
            process_tco2_per_unit_product=1.082,
            combustion_tco2_per_unit_product=0.5525,
            process_tch4_throughput_per_unit_product=0.393,
            combustion_tch4_throughput_per_unit_product=0.217,
            electricity_mwh_per_unit_product=0.12222,
            ccs_applied_to="process",
            process_capture_fraction=0.95,
            combustion_capture_fraction=0.0,
        )

    def test_grid_tco2(self, ammonia_with_grid, cc):
        row = compute_industry_row(ammonia_with_grid, cc)
        expected = 0.12222 * 0.37
        assert row["grid_tco2_per_unit_product"] == pytest.approx(expected, rel=1e-9)

    def test_grid_tch4(self, ammonia_with_grid, cc):
        row = compute_industry_row(ammonia_with_grid, cc)
        expected = 0.12222 * compute_grid_tch4_per_mwh(cc)
        assert row["grid_tch4_per_unit_product"] == pytest.approx(expected, rel=1e-9)

    def test_grid_co2e(self, ammonia_with_grid, cc):
        row = compute_industry_row(ammonia_with_grid, cc)
        grid_tch4 = 0.12222 * compute_grid_tch4_per_mwh(cc)
        expected = 0.12222 * 0.37 + LEAK_RATE * GWP20 * grid_tch4
        assert row["grid_tco2e_per_unit_product"] == pytest.approx(expected, rel=1e-9)

    def test_total_includes_grid(self, ammonia_with_grid, cc):
        row = compute_industry_row(ammonia_with_grid, cc)
        expected_total = 1.082 + 0.5525 + 0.12222 * 0.37
        assert row["total_tco2_per_unit_product"] == pytest.approx(
            expected_total, rel=1e-6
        )

    def test_grid_dilutes_capture_fraction(self, ammonia_with_grid, cc):
        """Grid CO2 is uncapturable, so it dilutes the adjusted capture fraction."""
        row = compute_industry_row(ammonia_with_grid, cc)
        # Without grid: adj_cf = (0.95 * 1.082) / (1.082 + 0.5525)
        cf_without_grid = (0.95 * 1.082) / (1.082 + 0.5525)
        assert row["adjusted_capture_fraction_tco2"] < cf_without_grid


class TestCaptureFractions:
    def test_methanol_adj_cf_tco2(self, methanol, cc):
        """adj_cf_tco2 = (pcf * process) / total for process-only CCS."""
        row = compute_industry_row(methanol, cc)
        expected = (0.95 * 0.35) / (0.35 + 0.05)
        assert row["adjusted_capture_fraction_tco2"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_methanol_adj_cf_tco2e(self, methanol, cc):
        """adj_cf_tco2e has same numerator as tco2 but denominator is total_tco2e."""
        row = compute_industry_row(methanol, cc)
        co2_captured = 0.95 * 0.35
        assert row["adjusted_capture_fraction_tco2e"] == pytest.approx(
            co2_captured / row["total_tco2e_per_unit_product"], rel=1e-9
        )

    def test_power_ccgt_adj_cf_tco2(self, power_ccgt, cc):
        """Fuel-only CCS: adj_cf = fuel_cf (since process = 0)."""
        row = compute_industry_row(power_ccgt, cc)
        assert row["adjusted_capture_fraction_tco2"] == pytest.approx(0.87, rel=1e-9)

    def test_power_ccgt_adj_cf_tco2e_less_than_tco2(self, power_ccgt, cc):
        """tCO2e capture fraction is lower because CH4 emissions dilute the denominator."""
        row = compute_industry_row(power_ccgt, cc)
        assert (
            row["adjusted_capture_fraction_tco2e"]
            < row["adjusted_capture_fraction_tco2"]
        )

    def test_process_cf_tco2e_for_process_ccs(self, methanol, cc):
        row = compute_industry_row(methanol, cc)
        expected = (0.95 * 0.35) / row["process_tco2e_per_unit_product"]
        assert row["process_capture_fraction_tco2e"] == pytest.approx(
            expected, rel=1e-9
        )

    def test_fuel_cf_tco2e_for_fuel_ccs(self, power_ccgt, cc):
        row = compute_industry_row(power_ccgt, cc)
        combustion_co2 = resolve_combustion_co2_tco2(power_ccgt, cc)
        expected = (0.87 * combustion_co2) / row["combustion_tco2e_per_unit_product"]
        assert row["combustion_capture_fraction_tco2e"] == pytest.approx(
            expected, rel=1e-9
        )


# ---------------------------------------------------------------------------
# Direct CH4 emissions (not supply-chain throughput)
# ---------------------------------------------------------------------------


class TestDirectCH4:
    """Direct facility CH4 uses GWP × amount (no leak_rate multiplier)."""

    def test_petrochemicals_combustion_co2e(self, petrochemicals, cc):
        row = compute_industry_row(petrochemicals, cc)
        expected = 0.255532 + GWP20 * 0.092921
        assert row["combustion_tco2e_per_unit_product"] == pytest.approx(
            expected, rel=1e-4
        )

    def test_petrochemicals_no_supply_chain_leakage(self, petrochemicals, cc):
        """With zero throughput CH4, only direct CH4 contributes to CO2e."""
        row = compute_industry_row(petrochemicals, cc)
        direct_co2e = GWP20 * 0.092921
        assert row["combustion_tco2e_per_unit_product"] == pytest.approx(
            0.255532 + direct_co2e, rel=1e-4
        )

    def test_direct_ch4_not_multiplied_by_leak_rate(self, petrochemicals, cc):
        """Changing leak rate should NOT affect direct CH4 contribution."""
        row_high = compute_industry_row(
            petrochemicals, _make_constants(gwp_ch4=GWP20, methane_leak_rate=0.05)
        )
        row_low = compute_industry_row(
            petrochemicals, _make_constants(gwp_ch4=GWP20, methane_leak_rate=0.01)
        )
        assert row_high["combustion_tco2e_per_unit_product"] == pytest.approx(
            row_low["combustion_tco2e_per_unit_product"], rel=1e-9
        )

    def test_refined_fuels_direct_ch4(self, refined_fuels, cc):
        row = compute_industry_row(refined_fuels, cc)
        expected = 0.018728 + GWP20 * 0.0068
        assert row["combustion_tco2e_per_unit_product"] == pytest.approx(
            expected, rel=1e-4
        )

    def test_direct_vs_throughput_magnitude(self, cc):
        """Direct CH4 (GWP × amount) is much larger than throughput (leak_rate × GWP × amount)."""
        ind_direct = IndustryInput(
            industrial_product="test_direct",
            simple_product_name="test",
            unit="unit",
            process_tco2_per_unit_product=1.0,
            combustion_tco2_per_unit_product=1.0,
            process_tch4_throughput_per_unit_product=0.0,
            combustion_tch4_throughput_per_unit_product=0.0,
            direct_combustion_tch4_per_unit_product=0.1,
            ccs_applied_to="process",
        )
        ind_throughput = IndustryInput(
            industrial_product="test_throughput",
            simple_product_name="test",
            unit="unit",
            process_tco2_per_unit_product=1.0,
            combustion_tco2_per_unit_product=1.0,
            process_tch4_throughput_per_unit_product=0.0,
            combustion_tch4_throughput_per_unit_product=0.1,
            ccs_applied_to="process",
        )
        row_direct = compute_industry_row(ind_direct, cc)
        row_throughput = compute_industry_row(ind_throughput, cc)
        assert (
            row_direct["combustion_tco2e_per_unit_product"]
            > row_throughput["combustion_tco2e_per_unit_product"]
        )

    def test_output_includes_direct_ch4_columns(self, petrochemicals, cc):
        row = compute_industry_row(petrochemicals, cc)
        assert row["direct_process_tch4_per_unit_product"] == 0.0
        assert row["direct_combustion_tch4_per_unit_product"] == 0.092921


# ---------------------------------------------------------------------------
# Sensitivity to GWP and leakage rate
# ---------------------------------------------------------------------------


class TestSensitivity:
    def test_lower_gwp_reduces_tco2e(self, methanol):
        cc_high = _make_constants(gwp_ch4=86, methane_leak_rate=0.035)
        cc_low = _make_constants(gwp_ch4=29.8, methane_leak_rate=0.035)
        high = compute_industry_row(methanol, cc_high)
        low = compute_industry_row(methanol, cc_low)
        assert (
            low["total_tco2e_per_unit_product"] < high["total_tco2e_per_unit_product"]
        )
        assert low["total_tco2_per_unit_product"] == high["total_tco2_per_unit_product"]

    def test_lower_leakage_reduces_tco2e(self, methanol):
        cc_high = _make_constants(gwp_ch4=86, methane_leak_rate=0.05)
        cc_low = _make_constants(gwp_ch4=86, methane_leak_rate=0.02)
        high = compute_industry_row(methanol, cc_high)
        low = compute_industry_row(methanol, cc_low)
        assert (
            low["total_tco2e_per_unit_product"] < high["total_tco2e_per_unit_product"]
        )

    def test_lower_gwp_increases_adj_cf_tco2e(self, methanol):
        """Lower GWP → less CH4-driven denominator → higher tCO2e capture fraction."""
        cc_high = _make_constants(gwp_ch4=86, methane_leak_rate=0.035)
        cc_low = _make_constants(gwp_ch4=29.8, methane_leak_rate=0.035)
        high = compute_industry_row(methanol, cc_high)
        low = compute_industry_row(methanol, cc_low)
        assert (
            low["adjusted_capture_fraction_tco2e"]
            > high["adjusted_capture_fraction_tco2e"]
        )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestComputeIndustrialEmissions:
    def test_returns_dataframe_with_all_industries(self, cc, methanol, dac):
        config = IndustrialEmissionsConfig(constants=cc, industries=[methanol, dac])
        df = compute_industrial_emissions(config)
        assert len(df) == 2
        assert set(df["simple_product_name"]) == {"methanol", "dac"}

    def test_output_columns(self, cc, methanol):
        config = IndustrialEmissionsConfig(constants=cc, industries=[methanol])
        df = compute_industrial_emissions(config)
        expected_cols = {
            "industrial_product",
            "simple_product_name",
            "unit_of_product",
            "process_tco2_per_unit_product",
            "combustion_tco2_per_unit_product",
            "grid_tco2_per_unit_product",
            "total_tco2_per_unit_product",
            "process_tco2e_per_unit_product",
            "combustion_tco2e_per_unit_product",
            "grid_tco2e_per_unit_product",
            "total_tco2e_per_unit_product",
            "process_tch4_throughput_per_unit_product",
            "combustion_tch4_throughput_per_unit_product",
            "direct_process_tch4_per_unit_product",
            "direct_combustion_tch4_per_unit_product",
            "grid_tch4_per_unit_product",
            "electricity_mwh_per_unit_product",
            "ccs_applied_to",
            "is_methane_based",
            "process_capture_fraction_tco2",
            "combustion_capture_fraction_tco2",
            "adjusted_capture_fraction_tco2",
            "process_capture_fraction_tco2e",
            "combustion_capture_fraction_tco2e",
            "adjusted_capture_fraction_tco2e",
        }
        assert set(df.columns) == expected_cols
