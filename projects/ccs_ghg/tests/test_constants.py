"""Unit tests for physical and unit-conversion constants.

Run with: pytest projects/ccs_ghg/tests/test_constants.py
"""

import pytest

from ccs_ghg.constants import (
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


class TestKappaDerivation:
    """Pin each intermediate quantity in the κ derivation to the value stated
    in paper §1.1.2, so future edits to `constants.py` can't silently drift
    the pipeline-gas methane coefficient away from the published derivation.
    Tolerances are set to the paper's stated precision (4 significant figures)."""

    def test_pure_ch4_hhv(self):
        from ccs_ghg.constants import HHV_CH4_BTU_PER_SCF

        # Armstrong & Jobe (1982), NBSIR 82-2401, pure methane at 60°F, 14.73 psia.
        assert abs(HHV_CH4_BTU_PER_SCF - 1014.0) < 0.05

    def test_pure_c2h6_hhv(self):
        from ccs_ghg.constants import HHV_C2H6_BTU_PER_SCF

        # Armstrong & Jobe (1982), NBSIR 82-2401, pure ethane at 60°F, 14.73 psia.
        assert abs(HHV_C2H6_BTU_PER_SCF - 1787.92) < 0.05

    def test_ch4_mole_fraction(self):
        from ccs_ghg.constants import MOLE_FRACTION_CH4_PIPELINE_GAS

        assert abs(MOLE_FRACTION_CH4_PIPELINE_GAS - 0.9126) < 5e-4

    def test_c2h6_mole_fraction(self):
        from ccs_ghg.constants import MOLE_FRACTION_C2H6_PIPELINE_GAS

        assert abs(MOLE_FRACTION_C2H6_PIPELINE_GAS - 0.0574) < 5e-4

    def test_inert_mole_fractions(self):
        from ccs_ghg.constants import (
            MOLE_FRACTION_CO2_PIPELINE_GAS,
            MOLE_FRACTION_N2_PIPELINE_GAS,
        )

        assert MOLE_FRACTION_CO2_PIPELINE_GAS == 0.02
        assert MOLE_FRACTION_N2_PIPELINE_GAS == 0.01

    def test_mixture_molar_mass(self):
        from ccs_ghg.constants import MOLAR_MASS_PIPELINE_GAS_G_PER_MOL

        assert abs(MOLAR_MASS_PIPELINE_GAS_G_PER_MOL - 17.527) < 0.005

    def test_ch4_mass_fraction(self):
        from ccs_ghg.constants import MASS_FRACTION_CH4_PIPELINE_GAS

        assert abs(MASS_FRACTION_CH4_PIPELINE_GAS - 0.8353) < 5e-4

    def test_pipeline_gas_density(self):
        from ccs_ghg.constants import PIPELINE_GAS_DENSITY_KG_PER_SCF

        assert abs(PIPELINE_GAS_DENSITY_KG_PER_SCF - 0.020960) < 5e-6

    def test_mass_hhv(self):
        from ccs_ghg.constants import MASS_HHV_PIPELINE_GAS_MMBTU_PER_T

        assert abs(MASS_HHV_PIPELINE_GAS_MMBTU_PER_T - 49.046) < 0.02

    def test_kappa_matches_paper(self):
        """Final derived κ matches paper Eq. (eq:kappa) to 4 sig figs."""
        from ccs_ghg.constants import KAPPA_TCH4_PER_MMBTU

        assert abs(KAPPA_TCH4_PER_MMBTU - 0.01703) < 5e-6
