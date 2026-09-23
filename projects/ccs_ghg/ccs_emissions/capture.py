"""compute emissions from the capture stage of CCS."""

import logging

import pandas as pd

from projects.ccs_ghg.ccs_emissions.energy_emissions import (
    compute_onsite_gas_emissions_per_mmbtu,
)
from projects.ccs_ghg.config import CCSConstants
from projects.ccs_ghg.constants import DAYS_PER_YEAR, HOURS_PER_DAY

logger = logging.getLogger(__name__)


def compute_capture_emissions(
    capture_data: list,
    constants: CCSConstants,
    grid_tco2_per_mwh: float,
    grid_tco2e_per_mwh: float,
    natural_gas_tco2_per_mmbtu: float,
) -> pd.DataFrame:
    """Compute emissions incurred during CO2 capture for each industry.

    Args:
        capture_data: List of dicts from the configuration yml 'capture' block.
        constants: typed constants.
        grid_tco2_per_mwh: grid-average tco2/MWh (combustion CO2 only).
        grid_tco2e_per_mwh: grid-average tco2e/MWh (CO2 + fugitive CH4).
        natural_gas_tco2_per_mmbtu: EPA emission factor for natural gas (40 CFR 98).
    Returns:
        DataFrame with columns:
            co2_source, distance,
            tco2_emitted_per_tco2_captured, tco2e_emitted_per_tco2_captured
    """
    df = pd.DataFrame(capture_data)

    for col in (
        "captured_tco2_per_yr",
        "mwh_needed_per_tco2_captured",
        "mmbtu_methane_needed_per_tco2_captured",
        "natural_gas_mmbtu_per_day",
        "tco2_available_in_capture_stream",
        "power_for_capture_mw",
    ):
        if col not in df.columns:
            df[col] = float("nan")

    # a facility needs to either specify the total captured tco2 per year or the total unabated emitted
    # for the canonical facilities for which the annual tCO2 capture is not specified, compute it
    df["captured_tco2_per_yr"] = df["captured_tco2_per_yr"].fillna(
        df["tco2_available_in_capture_stream"] * df["co2_stream_capture_rate"]
    )
    # for the canonical facilities for which the annual unabated tCO2 emitted is not specified, compute it
    df["tco2_available_in_capture_stream"] = df[
        "tco2_available_in_capture_stream"
    ].fillna(df["captured_tco2_per_yr"] / df["co2_stream_capture_rate"])

    # electricity required for capture per year (MWh)
    df["mwh_for_capture_per_yr"] = (
        df["power_for_capture_mw"] * DAYS_PER_YEAR * HOURS_PER_DAY
    )
    # if the energy needed per tco2 captured isn't specified, compute it
    df["mwh_needed_per_tco2_captured"] = df["mwh_needed_per_tco2_captured"].fillna(
        df["mwh_for_capture_per_yr"] / df["captured_tco2_per_yr"]
    )

    # Methane required for combustion to power capture each year converted to mmbtu
    df["mmbtu_methane_for_capture_per_yr"] = (
        df["natural_gas_mmbtu_per_day"] * DAYS_PER_YEAR
    )
    # Methane needed per tco2 captured (if it isn't specified, compute it)
    df["mmbtu_methane_needed_per_tco2_captured"] = df[
        "mmbtu_methane_needed_per_tco2_captured"
    ].fillna(df["mmbtu_methane_for_capture_per_yr"] / df["captured_tco2_per_yr"])

    # emissions from electricity per tco2
    df["tco2_from_electricity_per_tco2_captured"] = (
        df["mwh_needed_per_tco2_captured"] * grid_tco2_per_mwh
    )
    # emissions from electricity per tco2e (i.e., co2 + methane)
    df["tco2e_from_electricity_per_tco2_captured"] = (
        df["mwh_needed_per_tco2_captured"] * grid_tco2e_per_mwh
    )

    # emissions from on-site natural gas combustion for capture
    (
        onsite_tco2_per_mmbtu,
        onsite_tco2e_per_mmbtu,
    ) = compute_onsite_gas_emissions_per_mmbtu(constants, natural_gas_tco2_per_mmbtu)
    df["tco2_emitted_from_combustion_per_tco2_captured"] = (
        df["mmbtu_methane_needed_per_tco2_captured"] * onsite_tco2_per_mmbtu
    )
    df["tco2e_emitted_from_combustion_per_tco2_captured"] = (
        df["mmbtu_methane_needed_per_tco2_captured"] * onsite_tco2e_per_mmbtu
    )

    # escaped co2 (not captured)
    df["escaped_tco2_per_tco2_captured"] = (
        df["tco2_available_in_capture_stream"] - df["captured_tco2_per_yr"]
    ) / df["captured_tco2_per_yr"]

    # total co2 emissions per tco2 captured
    computed_tco2 = (
        df["tco2_emitted_from_combustion_per_tco2_captured"]
        + df["escaped_tco2_per_tco2_captured"]
        + df["tco2_from_electricity_per_tco2_captured"]
    )
    if "tco2_emitted_per_tco2_captured" in df.columns:
        df["tco2_emitted_per_tco2_captured"] = df[
            "tco2_emitted_per_tco2_captured"
        ].fillna(computed_tco2)
    else:
        df["tco2_emitted_per_tco2_captured"] = computed_tco2
    # total co2e emissions per tco2 captured (methane + co2)
    df["tco2e_emitted_per_tco2_captured"] = (
        df["tco2e_emitted_from_combustion_per_tco2_captured"]
        + df["escaped_tco2_per_tco2_captured"]
        + df["tco2e_from_electricity_per_tco2_captured"]
    )

    # Keep only primary cases, rename for downstream joins
    capture_emissions_df = (
        df.loc[
            df["primary_case"] == 1,
            [
                "industry",
                "distance",
                "tco2_emitted_per_tco2_captured",
                "tco2e_emitted_per_tco2_captured",
            ],
        ]
        .rename(columns={"industry": "co2_source"})
        .copy()
    )
    logger.info(" --- Computed emissions factors for capture component of CCS")

    return capture_emissions_df
