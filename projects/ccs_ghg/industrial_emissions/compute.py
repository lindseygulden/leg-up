"""Compute per-industry CO2 and CO2e emissions factors with adjusted capture fractions."""

import pandas as pd

from projects.ccs_ghg.config import CCSConstants, IndustryInput
from projects.ccs_ghg.constants import BTU_PER_MMBTU, KWH_PER_MWH
from projects.ccs_ghg.industrial_emissions.config import IndustrialEmissionsConfig

_EF_BY_FUEL = {
    "natural_gas": "natural_gas_tco2_per_mmbtu",
    "lignite": "coal_tco2_per_mmbtu",
}


def _resolve_heat_rate_btu_per_kwh(ind: IndustryInput, cc: CCSConstants) -> float:
    """Return the plant heat rate in Btu/kWh.

    If the industry YAML specifies `heat_rate_btu_per_kwh`, use it.
    If it's None and the fuel is natural gas, fall back to the state-average
    NG fleet heat rate (`cc.grid_ng_heat_rate_mmbtu_per_mwh`, converted from
    MMBtu/MWh to Btu/kWh). This lets `power_ng_retrofit` inherit the state's
    fleet-avg efficiency without pinning a hard number in the YAML.
    """
    if ind.heat_rate_btu_per_kwh is not None:
        return ind.heat_rate_btu_per_kwh
    if (
        ind.fuel_type == "natural_gas"
        and cc.grid_ng_heat_rate_mmbtu_per_mwh is not None
    ):
        # MMBtu/MWh -> Btu/kWh: multiply by 1000 (MMBtu/MWh × 1e6 Btu/MMBtu / 1e3 kWh/MWh = 1000 × Btu/kWh)
        return cc.grid_ng_heat_rate_mmbtu_per_mwh * 1000.0
    raise ValueError(
        f"Industry {ind.industrial_product!r} has heat_rate_btu_per_kwh=None but "
        f"no state fallback is available (fuel_type={ind.fuel_type!r}, "
        f"grid_ng_heat_rate_mmbtu_per_mwh={cc.grid_ng_heat_rate_mmbtu_per_mwh!r})."
    )


def resolve_combustion_co2_tco2(ind: IndustryInput, cc: CCSConstants) -> float:
    """Return combustion tCO2; derive from heat rate for power plants."""
    if ind.combustion_tco2_per_unit_product is not None:
        return ind.combustion_tco2_per_unit_product
    ef = getattr(cc, _EF_BY_FUEL[ind.fuel_type])
    heat_rate = _resolve_heat_rate_btu_per_kwh(ind, cc)
    return heat_rate * ef * KWH_PER_MWH / BTU_PER_MMBTU


def resolve_combustion_tch4_throughput(ind: IndustryInput, cc: CCSConstants) -> float:
    """Return combustion tCH4 throughput; derive from heat rate for gas plants.

    Symmetric with resolve_combustion_co2_tco2: for NG-fueled power plants where
    the industry YAML leaves combustion_tch4_throughput_per_unit_product = None,
    derive it as `heat_rate (MMBtu/MWh) × tch4_per_mmbtu_pipeline_gas`.
    """
    if ind.combustion_tch4_throughput_per_unit_product is not None:
        return ind.combustion_tch4_throughput_per_unit_product
    if ind.fuel_type == "natural_gas":
        heat_rate_btu = _resolve_heat_rate_btu_per_kwh(ind, cc)
        heat_rate_mmbtu_per_mwh = heat_rate_btu / 1000.0
        return heat_rate_mmbtu_per_mwh * cc.tch4_per_mmbtu_pipeline_gas
    raise ValueError(
        f"Industry {ind.industrial_product!r} has combustion_tch4_throughput=None "
        f"and no derivation path for fuel_type={ind.fuel_type!r}."
    )


def compute_grid_tch4_per_mwh(cc: CCSConstants) -> float:
    """CH4 throughput per MWh of grid electricity (state-specific direct input)."""
    return cc.grid_tch4_per_mwh


def compute_industry_row(ind: IndustryInput, cc: CCSConstants) -> dict:
    """Compute all emissions columns for a single industry."""

    # --- Process emissions (industrial process itself) ---
    process_co2 = ind.process_tco2_per_unit_product

    # --- On-site combustion emissions (fossil fuels burned at the facility) ---
    combustion_co2 = resolve_combustion_co2_tco2(ind, cc)
    combustion_tch4 = resolve_combustion_tch4_throughput(ind, cc)

    # --- Grid electricity emissions (distributed, from power consumed) ---
    grid_tch4_per_mwh = compute_grid_tch4_per_mwh(cc)
    grid_co2 = ind.electricity_mwh_per_unit_product * cc.grid_tco2_per_mwh
    grid_tch4 = ind.electricity_mwh_per_unit_product * grid_tch4_per_mwh

    # --- Total CO2 ---
    total_co2 = process_co2 + combustion_co2 + grid_co2

    # --- CO2-equivalent ---
    # Supply-chain fugitive CH4: throughput × leak_rate × GWP
    # Direct facility CH4: emitted at 100%, so just × GWP (no leak_rate)
    process_co2e = process_co2 + (
        cc.methane_leak_rate * cc.gwp_ch4 * ind.process_tch4_throughput_per_unit_product
        + cc.gwp_ch4 * ind.direct_process_tch4_per_unit_product
    )
    combustion_co2e = combustion_co2 + (
        cc.methane_leak_rate * cc.gwp_ch4 * combustion_tch4
        + cc.gwp_ch4 * ind.direct_combustion_tch4_per_unit_product
    )
    grid_co2e = grid_co2 + (cc.methane_leak_rate * cc.gwp_ch4 * grid_tch4)

    total_co2e = process_co2e + combustion_co2e + grid_co2e

    # --- Capture fractions ---
    # CCS captures from process or combustion streams only; grid is uncapturable
    co2_captured = (
        ind.process_capture_fraction * process_co2
        + ind.combustion_capture_fraction * combustion_co2
    )

    adj_cf_tco2 = co2_captured / total_co2 if total_co2 > 0 else 0.0
    pcf_tco2e = (
        (ind.process_capture_fraction * process_co2) / process_co2e
        if process_co2e > 0
        else 0.0
    )
    fcf_tco2e = (
        (ind.combustion_capture_fraction * combustion_co2) / combustion_co2e
        if combustion_co2e > 0
        else 0.0
    )
    adj_cf_tco2e = co2_captured / total_co2e if total_co2e > 0 else 0.0

    return {
        "industrial_product": ind.industrial_product,
        "simple_product_name": ind.simple_product_name,
        "unit_of_product": ind.unit,
        "process_tco2_per_unit_product": process_co2,
        "combustion_tco2_per_unit_product": combustion_co2,
        "grid_tco2_per_unit_product": grid_co2,
        "total_tco2_per_unit_product": total_co2,
        "process_tco2e_per_unit_product": process_co2e,
        "combustion_tco2e_per_unit_product": combustion_co2e,
        "grid_tco2e_per_unit_product": grid_co2e,
        "total_tco2e_per_unit_product": total_co2e,
        "process_tch4_throughput_per_unit_product": ind.process_tch4_throughput_per_unit_product,
        "combustion_tch4_throughput_per_unit_product": combustion_tch4,
        "direct_process_tch4_per_unit_product": ind.direct_process_tch4_per_unit_product,
        "direct_combustion_tch4_per_unit_product": ind.direct_combustion_tch4_per_unit_product,
        "grid_tch4_per_unit_product": grid_tch4,
        "electricity_mwh_per_unit_product": ind.electricity_mwh_per_unit_product,
        "ccs_applied_to": ind.ccs_applied_to,
        "is_methane_based": ind.is_methane_based,
        "process_capture_fraction_tco2": ind.process_capture_fraction,
        "combustion_capture_fraction_tco2": ind.combustion_capture_fraction,
        "adjusted_capture_fraction_tco2": adj_cf_tco2,
        "process_capture_fraction_tco2e": pcf_tco2e,
        "combustion_capture_fraction_tco2e": fcf_tco2e,
        "adjusted_capture_fraction_tco2e": adj_cf_tco2e,
    }


def compute_industrial_emissions(
    config: IndustrialEmissionsConfig,
) -> pd.DataFrame:
    """Compute per-industry emissions factors and adjusted capture fractions.

    Args:
        config: Configuration with conversion constants and per-industry inputs.

    Returns:
        DataFrame with one row per industry and columns for tCO2, tCO2e,
        CH4 throughput, and capture fractions.
    """
    rows = [compute_industry_row(ind, config.constants) for ind in config.industries]
    return pd.DataFrame(rows)
