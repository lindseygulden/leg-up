"""Smoke tests for publication-ready table builders in export_tables.py.

Verifies that each of the three build_* functions produces a non-empty DataFrame
with the expected columns when given a real test-case CCSProjectConfig. These
are the tables that get published in the paper, so a broken column or missing
row here is a bug that would ship.
"""

from pathlib import Path

import pytest

from projects.ccs_ghg.config import CCSProjectConfig
from projects.ccs_ghg.export_tables import (
    build_ccs_gs_table,
    build_eor_table,
    build_industrial_emissions_table,
)

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_INFRASTRUCTURE_CONFIG = str(_CONFIG_DIR / "config_infrastructure_test_case.yml")
_INDUSTRIES_CONFIG = str(_CONFIG_DIR / "config_industries_test_case.yml")


@pytest.fixture
def project_config():
    return CCSProjectConfig.from_yaml(_INFRASTRUCTURE_CONFIG, _INDUSTRIES_CONFIG)


class TestIndustrialEmissionsTable:
    """Table A: per-unit-product process/combustion/grid emissions by industry."""

    @pytest.fixture
    def table(self, project_config):
        return build_industrial_emissions_table(project_config)

    def test_non_empty(self, table):
        assert len(table) > 0

    def test_expected_columns(self, table):
        expected = {
            "industry",
            "industrial_product",
            "unit",
            "process_tco2_per_unit",
            "combustion_tco2_per_unit",
            "grid_tco2_per_unit",
            "total_tco2_per_unit",
            "process_tch4_per_unit",
            "combustion_tch4_per_unit",
            "grid_tch4_per_unit",
            "electricity_mwh_per_unit",
            "ccs_applied_to",
            "co2_stream_capture_rate",
            "heat_rate_btu_per_kwh",
            "fuel_type",
        }
        assert expected.issubset(set(table.columns))

    def test_total_equals_sum_of_components(self, table):
        computed = (
            table["process_tco2_per_unit"]
            + table["combustion_tco2_per_unit"]
            + table["grid_tco2_per_unit"]
        )
        assert (table["total_tco2_per_unit"] - computed).abs().max() < 1e-9

    def test_capture_rates_are_fractions(self, table):
        assert table["co2_stream_capture_rate"].between(0, 1).all()


class TestCCSGSTable:
    """Table B: per-industry CCS emissions for geologic storage pathway."""

    @pytest.fixture
    def table(self, project_config):
        return build_ccs_gs_table(project_config)

    def test_non_empty(self, table):
        assert len(table) > 0

    def test_expected_columns(self, table):
        expected = {
            "industry",
            "case_name",
            "distance",
            "co2_stream_capture_rate",
            "mwh_per_tco2_captured",
            "mmbtu_gas_per_tco2_captured",
            "capture_electricity_tco2",
            "capture_gas_combustion_tco2",
            "capture_escaped_tco2",
            "capture_total_tco2",
            "tco2_emitted_per_tco2_transported",
            "tco2_emitted_per_tco2_stored",
            "total_tco2_per_tco2_sequestered",
            "net_tco2_per_tco2_sequestered",
            "capture_electricity_tco2e",
            "capture_gas_combustion_tco2e",
            "capture_total_tco2e",
            "tco2e_emitted_per_tco2_transported",
            "tco2e_emitted_per_tco2_stored",
            "total_tco2e_per_tco2_sequestered",
            "net_tco2e_per_tco2_sequestered",
        }
        assert expected.issubset(set(table.columns))

    def test_gs_table_has_no_oil_column(self, table):
        # Oil-product emissions only apply to EOR.
        assert "tco2_emitted_from_ccs_product_per_tco2_ccs" not in table.columns

    def test_tco2e_at_least_tco2(self, table):
        # CO2e includes fugitive CH4, so should be >= CO2 for every row.
        assert (
            table["total_tco2e_per_tco2_sequestered"]
            >= table["total_tco2_per_tco2_sequestered"] - 1e-9
        ).all()

    def test_net_equals_total_minus_one(self, table):
        assert (
            table["net_tco2_per_tco2_sequestered"]
            - (table["total_tco2_per_tco2_sequestered"] - 1)
        ).abs().max() < 1e-9


class TestEORTable:
    """Table C: per-industry CCS emissions for EOR pathway (includes oil combustion)."""

    @pytest.fixture
    def table(self, project_config):
        return build_eor_table(project_config)

    def test_non_empty(self, table):
        assert len(table) > 0

    def test_has_oil_column(self, table):
        assert "tco2_emitted_from_ccs_product_per_tco2_ccs" in table.columns

    def test_oil_column_positive(self, table):
        assert (table["tco2_emitted_from_ccs_product_per_tco2_ccs"] > 0).all()

    def test_eor_lifecycle_exceeds_gs(self, project_config, table):
        """For a given industry, EOR lifecycle factor should exceed GS lifecycle
        factor (EOR adds oil-combustion emissions on top of everything else)."""
        gs = build_ccs_gs_table(project_config).set_index("industry")[
            "total_tco2e_per_tco2_sequestered"
        ]
        eor = table.set_index("industry")["total_tco2e_per_tco2_sequestered"]
        common = gs.index.intersection(eor.index)
        assert len(common) > 0
        assert (eor.loc[common] > gs.loc[common] - 1e-9).all()
