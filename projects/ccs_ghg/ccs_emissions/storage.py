"""Compute emissions from the co2 storage for CCS"""

import logging

import numpy as np
import pandas as pd

from projects.ccs_ghg.ccs_emissions.energy_emissions import (
    compute_gas_turbine_emissions_per_mwh,
)
from projects.ccs_ghg.config import CCSConstants

logger = logging.getLogger(__name__)


def compute_storage_emissions(
    storage_data: list,
    constants: CCSConstants,
    grid_tco2_per_mwh: float,
    grid_tco2e_per_mwh: float,
    natural_gas_tco2_per_mmbtu: float,
) -> pd.DataFrame:
    """Compute emissions incurred during co2 storage (injection underground)
    Handles both EOR (enhanced oil recovery) and GS (geologic storage) destinations
    Args:
        storage_data: List of dicts from the yml 'storage' block
        constants: typed constants
        grid_tco2_per_mwh: grid-average tco2 per MWh (combustion CO2 only).
        grid_tco2e_per_mwh: grid-average tco2e per MWh (CO2 + fugitive CH4).
    Returns:
        pd.DataFrame, columns include:
            co2_destination,
            tco2_emitted_per_tco2_stored,
            tco2e_emitted_per_tco2_stored,
            tco2_emitted_from_ccs_product_per_tco2_ccs
    """
    df = pd.DataFrame(storage_data)
    is_eor = df["co2_destination"] == "eor"

    eor_tco2, eor_tco2e = compute_gas_turbine_emissions_per_mwh(
        constants, natural_gas_tco2_per_mmbtu
    )

    df["tco2_emitted_per_tco2_stored"] = np.where(
        is_eor,
        df["mwh_per_tco2_stored"] * eor_tco2,
        df["mwh_per_tco2_stored"] * grid_tco2_per_mwh,
    )

    df["tco2e_emitted_per_tco2_stored"] = np.where(
        is_eor,
        df["mwh_per_tco2_stored"] * eor_tco2e,
        df["mwh_per_tco2_stored"] * grid_tco2e_per_mwh,
    )

    # compute the tco2 emitted when the direct products of CCS (i.e., EOR-produced oil) are combusted
    df["tco2_emitted_from_ccs_product_per_tco2_ccs"] = np.where(
        is_eor,
        constants.oil_tco2_per_bbl * constants.recovery_factor_bbl_per_tco2,
        0.0,  # if not eor, is gs: no emissions from oil produced for gs
    )
    logger.info(" --- Computed emissions factors for storage component of CCS")
    return df[
        [
            "co2_destination",
            "tco2_emitted_per_tco2_stored",
            "tco2e_emitted_per_tco2_stored",
            "tco2_emitted_from_ccs_product_per_tco2_ccs",
        ]
    ].copy()
