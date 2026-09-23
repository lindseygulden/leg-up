"""Emissions from energy sources used in CCS: grid electricity, gas turbines, on-site gas."""

import logging
from typing import Tuple

from projects.ccs_ghg.config import CCSConstants
from projects.ccs_ghg.constants import MJ_PER_MMBTU, MJ_PER_MWH

logger = logging.getLogger(__name__)


def compute_grid_emissions_per_mwh(constants: CCSConstants) -> Tuple[float, float]:
    """Emissions per MWh of grid electricity: combustion CO2 + fugitive CH4.

    Both are direct per-state inputs: grid_tco2_per_mwh covers combustion CO2
    across all fuel types, and grid_tch4_per_mwh is the tCH4 associated with the
    gas burned to produce one MWh of grid electricity.

    Returns (tco2_per_mwh, tco2e_per_mwh).
    """
    tco2_per_mwh = constants.grid_tco2_per_mwh
    tch4_per_mwh = constants.grid_tch4_per_mwh

    fugitive_tco2e = tch4_per_mwh * constants.methane_leak_rate * constants.gwp_ch4
    tco2e_per_mwh = tco2_per_mwh + fugitive_tco2e

    logger.info(
        "Grid emissions: %.6f tCO2/MWh, %.6f tCO2e/MWh", tco2_per_mwh, tco2e_per_mwh
    )
    return tco2_per_mwh, tco2e_per_mwh


def compute_gas_turbine_emissions_per_mwh(
    constants: CCSConstants,
    natural_gas_tco2_per_mmbtu: float,
) -> Tuple[float, float]:
    """Emissions per MWh from an open-cycle gas turbine (used for EOR injection).

    Combustion CO2 from EPA emission factor (40 CFR 98 Table C-1), consistent
    with on-site gas and power-plant calculations. Turbine efficiency
    determines gas consumed per MWh.

    Returns (tco2_per_mwh, tco2e_per_mwh).
    """
    mmbtu_per_mwh = MJ_PER_MWH / constants.eor_gas_turbine_efficiency / MJ_PER_MMBTU
    tch4_per_mwh = mmbtu_per_mwh * constants.tch4_per_mmbtu_pipeline_gas

    tco2_per_mwh = mmbtu_per_mwh * natural_gas_tco2_per_mmbtu

    fugitive_tco2e = tch4_per_mwh * constants.methane_leak_rate * constants.gwp_ch4
    tco2e_per_mwh = tco2_per_mwh + fugitive_tco2e

    logger.info(
        "Gas turbine emissions: %.6f tCO2/MWh, %.6f tCO2e/MWh",
        tco2_per_mwh,
        tco2e_per_mwh,
    )
    return tco2_per_mwh, tco2e_per_mwh


def compute_onsite_gas_emissions_per_mmbtu(
    constants: CCSConstants,
    natural_gas_tco2_per_mmbtu: float,
) -> Tuple[float, float]:
    """Emissions per MMBTU of natural gas burned on-site for CO2 capture.

    Combustion CO2 from EPA emission factor (40 CFR 98 Table C-1).
    Fugitive CH4 from the gas supply chain for each MMBTU consumed.

    Returns (tco2_per_mmbtu, tco2e_per_mmbtu).
    """
    tco2_per_mmbtu = natural_gas_tco2_per_mmbtu

    tch4_per_mmbtu = constants.tch4_per_mmbtu_pipeline_gas

    fugitive_tco2e = tch4_per_mmbtu * constants.methane_leak_rate * constants.gwp_ch4
    tco2e_per_mmbtu = tco2_per_mmbtu + fugitive_tco2e

    return tco2_per_mmbtu, tco2e_per_mmbtu
