"""compute emissions from the transportation of CCS co2"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_transport_emissions(
    transport_data: list,
    grid_tco2_per_mwh: float,
    grid_tco2e_per_mwh: float,
) -> pd.DataFrame:
    """calculates emissions incurred by transport of co2
    Args:
        transport_data: List of dicts from the yml 'transport' block.
        grid_tco2_per_mwh: grid-average tco2 per MWh (combustion CO2 only).
        grid_tco2e_per_mwh: grid-average tco2e per MWh (CO2 + fugitive CH4).
    Returns:
        pd.DataFrame: including columns for distance,
            tco2_emitted_per_tco2_transported,
            tco2e_emitted_per_tco2_transported
    """
    df = pd.DataFrame(transport_data)

    df["tco2_emitted_per_tco2_transported"] = (
        df["mwh_per_tco2_transported"] * grid_tco2_per_mwh
    )
    df["tco2e_emitted_per_tco2_transported"] = (
        df["mwh_per_tco2_transported"] * grid_tco2e_per_mwh
    )
    logger.info(" --- Computed emissions factors for transport of CO2")

    return df[
        [
            "distance",
            "tco2_emitted_per_tco2_transported",
            "tco2e_emitted_per_tco2_transported",
        ]
    ].copy()
