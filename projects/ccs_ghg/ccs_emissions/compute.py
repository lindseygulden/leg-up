"""Assemble CCS emissions factors across capture, transport, and storage."""

import logging

import pandas as pd

from projects.ccs_ghg.ccs_emissions.capture import compute_capture_emissions
from projects.ccs_ghg.ccs_emissions.energy_emissions import (
    compute_grid_emissions_per_mwh,
)
from projects.ccs_ghg.ccs_emissions.storage import compute_storage_emissions
from projects.ccs_ghg.ccs_emissions.transport import compute_transport_emissions
from projects.ccs_ghg.config import CCSProjectConfig

logger = logging.getLogger(__name__)


def compute_ccs_emissions_factors(
    config: CCSProjectConfig,
) -> pd.DataFrame:
    """Compute per-(source, destination) emissions factors for CCS.

    Returns raw eor and gs factors for each primary-case industry.
    No weighted scenarios or industry filtering — those belong in
    the scenarios module.

    Args:
        config: Parsed and validated CCS project configuration.
    Returns:
        DataFrame with per-(co2_source, co2_destination) emissions factors.
    """
    constants = config.constants
    grid_tco2_per_mwh, grid_tco2e_per_mwh = compute_grid_emissions_per_mwh(constants)

    capture_df = compute_capture_emissions(
        config.get_all_capture_cases(),
        constants,
        grid_tco2_per_mwh,
        grid_tco2e_per_mwh,
        constants.natural_gas_tco2_per_mmbtu,
    )
    transport_df = compute_transport_emissions(
        config.transport, grid_tco2_per_mwh, grid_tco2e_per_mwh
    )
    storage_df = compute_storage_emissions(
        config.storage,
        constants,
        grid_tco2_per_mwh,
        grid_tco2e_per_mwh,
        constants.natural_gas_tco2_per_mmbtu,
    )

    logger.info(
        "Computed emissions for %d capture, %d transport, %d storage rows",
        len(capture_df),
        len(transport_df),
        len(storage_df),
    )

    # Assemble capture, transport, and storage join on distance, then cross-join with storage
    emissions_df = transport_df.merge(capture_df, on="distance").merge(
        storage_df, how="cross"
    )

    # Sum component factors for both tCO2 and tCO2e
    for ghg in ["tco2", "tco2e"]:
        component_cols = [
            f"{ghg}_emitted_per_tco2_captured",
            f"{ghg}_emitted_per_tco2_transported",
            f"{ghg}_emitted_per_tco2_stored",
            "tco2_emitted_from_ccs_product_per_tco2_ccs",
        ]
        emissions_df[f"{ghg}_emitted_per_tco2_ccs"] = emissions_df[component_cols].sum(
            axis=1
        )
        emissions_df[f"net_{ghg}_emitted_per_tco2_ccs"] = (
            emissions_df[f"{ghg}_emitted_per_tco2_ccs"] - 1  # 1 tCO2 sequestered
        )

    output_cols = [
        "co2_source",
        "co2_destination",
        "distance",
        "tco2_emitted_per_tco2_ccs",
        "net_tco2_emitted_per_tco2_ccs",
        "tco2e_emitted_per_tco2_ccs",
        "net_tco2e_emitted_per_tco2_ccs",
        "tco2_emitted_per_tco2_captured",
        "tco2e_emitted_per_tco2_captured",
        "tco2_emitted_per_tco2_transported",
        "tco2e_emitted_per_tco2_transported",
        "tco2_emitted_per_tco2_stored",
        "tco2e_emitted_per_tco2_stored",
    ]

    return emissions_df[output_cols]
