"""Unit tests for physical and unit-conversion constants.

Run with: pytest projects/ccs_ghg/tests/test_constants.py
"""

import pytest

from projects.ccs_ghg.constants import (
    BBL_IN_KBOE,
    BTU_PER_MMBTU,
    DAYS_PER_YEAR,
    GJ_PER_MMBTU,
    HOURS_PER_DAY,
    KWH_PER_MWH,
    MJ_PER_GJ,
    MJ_PER_MMBTU,
    MJ_PER_MWH,
    MOLECULAR_WEIGHT_CH4_G_PER_MOL,
    MOLECULAR_WEIGHT_CO2_G_PER_MOL,
    MWH_PER_TWH,
    TCO2_PER_MTCO2,
)


class TestEnergyConversions:
    def test_mj_per_gj(self):
        assert MJ_PER_GJ == 1000

    def test_gj_per_mmbtu(self):
        assert GJ_PER_MMBTU == pytest.approx(1.055056, rel=1e-6)

    def test_mj_per_mmbtu(self):
        assert MJ_PER_MMBTU == pytest.approx(1055.056, rel=1e-6)

    def test_mj_per_mmbtu_is_product(self):
        assert MJ_PER_MMBTU == pytest.approx(GJ_PER_MMBTU * MJ_PER_GJ, rel=1e-12)

    def test_mj_per_mwh(self):
        assert MJ_PER_MWH == pytest.approx(3600.0, rel=1e-9)

    def test_btu_per_mmbtu(self):
        assert BTU_PER_MMBTU == 1_000_000

    def test_kwh_per_mwh(self):
        assert KWH_PER_MWH == 1_000


class TestMolecularWeights:
    def test_co2(self):
        assert MOLECULAR_WEIGHT_CO2_G_PER_MOL == pytest.approx(44.0095, rel=1e-4)

    def test_ch4(self):
        assert MOLECULAR_WEIGHT_CH4_G_PER_MOL == pytest.approx(16.04246, rel=1e-4)

    def test_co2_heavier_than_ch4(self):
        assert MOLECULAR_WEIGHT_CO2_G_PER_MOL > MOLECULAR_WEIGHT_CH4_G_PER_MOL


class TestTimeConstants:
    def test_days_per_year(self):
        assert DAYS_PER_YEAR == 365

    def test_hours_per_day(self):
        assert HOURS_PER_DAY == 24


class TestScaleFactors:
    def test_tco2_per_mtco2(self):
        assert TCO2_PER_MTCO2 == 1_000_000

    def test_mwh_per_twh(self):
        assert MWH_PER_TWH == 1_000_000

    def test_bbl_in_kboe(self):
        assert BBL_IN_KBOE == 1_000
