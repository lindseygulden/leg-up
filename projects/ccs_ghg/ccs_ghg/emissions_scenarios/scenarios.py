"""Compute total GHG emissions for various industrial CCS scenarios."""

# pylint: disable=cell-var-from-loop
import logging
import re
from dataclasses import replace
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ccs_ghg.ccs_emissions.compute import compute_ccs_emissions_factors
from ccs_ghg.config import CCSProjectConfig
from ccs_ghg.emissions_scenarios.config import MethaneScenario, ScenarioConfig
from ccs_ghg.industrial_emissions.compute import compute_industrial_emissions
from ccs_ghg.industrial_emissions.config import IndustrialEmissionsConfig

logger = logging.getLogger(__name__)


def _escalated_amount(
    base_amount: float,
    year: int,
    escalation_year: int,
    inflation_rate: float,
) -> float:
    """Return *base_amount* escalated for inflation starting in *escalation_year*.

    Years before escalation_year use the base amount unchanged.
    In escalation_year the first inflation adjustment applies:
        amount(y) = base_amount * (1 + r)^(y - escalation_year + 1)  for y >= escalation_year
    """
    if inflation_rate == 0.0 or year < escalation_year:
        return base_amount
    return base_amount * (1 + inflation_rate) ** (year - escalation_year + 1)


def present_value(
    annual_cost: float,
    project_start_year: int,
    project_duration: int,
    discount_rate: float = 0.0448,
    base_year: float = None,
    escalation_year: int = None,
    inflation_rate: float = 0.0,
) -> float:
    """Present value, as of `base_year`, of a nominal annual cash flow.

    Each year's payment is escalated exactly as in `nominal_total`, treated as
    received at year end (y + 1), and discounted at the nominal `discount_rate`:
        PV = sum_y amount(y) / (1 + d)^(max(0, (y + 1) - base_year))
    `base_year` may be fractional (e.g. 2026.5 = July 1, 2026). Payments after
    `base_year` are discounted; payments at or before `base_year` are taken at
    their nominal (escalated) face value --- they are NOT compounded forward.
    This treats pre-valuation cash flows as already-expressed in `base_year`
    purchasing power, which is appropriate for retrospective 45Q payments and
    oil revenue on facilities whose CCS operation began before the valuation
    date.
    """
    if base_year is None:
        base_year = project_start_year

    pv = 0.0
    for year in range(project_start_year, project_start_year + project_duration):
        t = max(0.0, (year + 1) - base_year)
        amount = _escalated_amount(annual_cost, year, escalation_year, inflation_rate)
        pv += amount / (1 + discount_rate) ** t

    return pv


def nominal_total(
    annual_cost: float,
    project_start_year: int,
    project_duration: int,
    escalation_year: int = None,
    inflation_rate: float = 0.0,
) -> float:
    """Undiscounted sum of (possibly escalated) annual payments over the project life."""
    total = 0.0
    for year in range(project_start_year, project_start_year + project_duration):
        total += _escalated_amount(annual_cost, year, escalation_year, inflation_rate)
    return total


def solar_mwh_from_45q(
    base_annual_45q_m_usd: float,
    project_start_year: int,
    project_duration: int,
    escalation_year: int,
    inflation_rate: float,
    solar_lcoe_base_usd_per_mwh: float,
    solar_lcoe_escalation_rate: float,
    base_year: int,
) -> float:
    """Total MWh of utility-scale solar purchasable with a facility's 45Q stream.

    For each year y ∈ [y_j, y_j + D), computes:
        nominal_45Q(y) [M$] = base_annual_45q_m_usd × (1 + υ)^(y - υ_esc + 1) for y >= υ_esc
        nominal_LCOE(y) [$/MWh] = base × (1 + r_solar)^(y - base_year)
        MWh(y) = nominal_45Q(y) × 1e6 / nominal_LCOE(y)
    and returns the sum across all years in the window.

    Args:
        base_annual_45q_m_usd: Base 45Q payment per year in million USD (C_j × p_d).
        project_start_year: y_j, the first year of CCS operation.
        project_duration: D (typically 12).
        escalation_year: Year the 45Q inflation adjustment starts (y_esc = 2027).
        inflation_rate: 45Q nominal escalation rate (υ, e.g., 0.025).
        solar_lcoe_base_usd_per_mwh: Base LCOE in USD/MWh (default $60).
        solar_lcoe_escalation_rate: Nominal LCOE escalation rate; 0 means a
            flat LCOE in every year of the window.
        base_year: Year the base LCOE applies to; LCOE escalates relative to
            this year. Typically cfg.base_year (2025).
    """
    total_mwh = 0.0
    for year in range(project_start_year, project_start_year + project_duration):
        nominal_45q_m_usd = _escalated_amount(
            base_annual_45q_m_usd, year, escalation_year, inflation_rate
        )
        if solar_lcoe_escalation_rate == 0.0:
            nominal_lcoe = solar_lcoe_base_usd_per_mwh
        else:
            nominal_lcoe = solar_lcoe_base_usd_per_mwh * (
                1.0 + solar_lcoe_escalation_rate
            ) ** (year - base_year)
        total_mwh += nominal_45q_m_usd * 1e6 / nominal_lcoe
    return total_mwh


def _destination_mix_label(destination_pct: dict) -> str:
    """Build a human-readable label like '80% eor/20% gs' from a destination_pct dict."""
    parts = [f"{int(pct)}% {dest}" for dest, pct in destination_pct.items() if pct > 0]
    return "/".join(parts)


def compute_scenario_emissions_factors(
    emissions_factors_df: pd.DataFrame,
    scenario_config: ScenarioConfig,
) -> pd.DataFrame:
    """Compute long-form scenario emissions factors from per-(source, destination) factors.

    Builds weighted scenarios from ``scenario_config.weighted_scenarios``,
    a ``baseline`` scenario (raw per-source/destination values), and optionally
    a ``nameplate_implied`` scenario (zero lifecycle factors), returning a
    long-form DataFrame with a ``scenario`` column.

    Args:
        emissions_factors_df: Per-(co2_source, co2_destination) factors from
            :func:`ccs_emissions.compute.compute_ccs_emissions_factors`.
        scenario_config: Typed scenario configuration with weighted_scenarios.

    Returns:
        DataFrame with columns: co2_source, stated_co2_destination, scenario,
        assumed_co2_destination, plus the four factor columns.
    """
    factor_cols = [
        "tco2e_emitted_per_tco2_ccs",
        "tco2_emitted_per_tco2_ccs",
        "net_tco2e_emitted_per_tco2_ccs",
        "net_tco2_emitted_per_tco2_ccs",
    ]

    all_destinations = emissions_factors_df["co2_destination"].unique()
    dest_df = pd.DataFrame({"stated_co2_destination": all_destinations})
    scenario_frames = []

    # --- Weighted scenarios from config (vectorized) ---
    for scenario in scenario_config.weighted_scenarios:
        weights = scenario.destination_pct
        total_weight = sum(weights.values())
        if total_weight == 0:
            continue

        ef = emissions_factors_df.copy()
        ef["_weight"] = ef["co2_destination"].map(weights)

        weighted = (
            ef.groupby("co2_source", sort=False)
            .apply(
                lambda g: pd.Series(
                    {
                        col: (g[col] * g["_weight"]).sum() / total_weight
                        for col in factor_cols
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )
        weighted["scenario"] = scenario.name
        weighted["assumed_co2_destination"] = _destination_mix_label(weights)

        # Cross-join with all destinations to preserve current schema
        weighted = weighted.merge(dest_df, how="cross")
        scenario_frames.append(weighted)

    # --- Baseline scenario: raw per-(source, destination) factors ---
    baseline_df = emissions_factors_df[
        ["co2_source", "co2_destination"] + factor_cols
    ].copy()
    baseline_df = baseline_df.rename(
        columns={"co2_destination": "stated_co2_destination"}
    )
    baseline_df["scenario"] = "baseline"
    baseline_df["assumed_co2_destination"] = baseline_df["stated_co2_destination"]
    scenario_frames.append(baseline_df)

    # --- Nameplate scenarios: zero lifecycle factors ---
    if scenario_config.include_nameplate_case:
        sources = emissions_factors_df["co2_source"].unique()
        # Two nameplate scenarios: `nameplate_implied` (always computed from
        # unabated × capture_fraction) and `nameplate` (union — uses operator-
        # supplied nameplate when present, else falls back to the implied
        # value). Both use zero lifecycle emissions.
        for np_name in ("nameplate_implied", "nameplate"):
            nameplate_df = pd.DataFrame({"co2_source": sources})
            for col in factor_cols:
                nameplate_df[col] = 0.0
            nameplate_df["scenario"] = np_name
            nameplate_df["assumed_co2_destination"] = "as stated"
            nameplate_df = nameplate_df.merge(dest_df, how="cross")
            scenario_frames.append(nameplate_df)

    return pd.concat(scenario_frames, ignore_index=True)


def load_industry_emissions(
    cfg: ScenarioConfig,
    industry_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Load and filter industry emissions data.

    Args:
        cfg: Scenario configuration with path and column info.
        industry_df: Optional pre-computed DataFrame (e.g. from
            :func:`industrial_emissions.compute_industrial_emissions`).
            When provided, CSV read is skipped.
    Returns:
        DataFrame with columns: product, adjusted_capture_fraction_tco2,
        adjusted_capture_fraction_tco2e
    """
    if industry_df is None:
        industry_df = pd.read_csv(
            cfg.emissions_for_industries_path,
            usecols=cfg.industry_emissions_cols,
        )
    keep_cols = [cfg.product_col_name, cfg.adjusted_capt_frac_col_name]
    rename_map = {
        cfg.product_col_name: "product",
        cfg.adjusted_capt_frac_col_name: "adjusted_capture_fraction_tco2e",
    }
    if "adjusted_capture_fraction_tco2" in industry_df.columns:
        keep_cols.append("adjusted_capture_fraction_tco2")
    industry_df = industry_df.loc[
        industry_df[cfg.industrial_product_with_detail_col_name].isin(
            cfg.industry_emissions_which_flavor
        )
    ][keep_cols].rename(columns=rename_map)
    return industry_df


def merge_project_factors(
    projects_df: pd.DataFrame,
    scenario_factors_df: pd.DataFrame,
    industry_df: pd.DataFrame,
    cfg: ScenarioConfig,
) -> pd.DataFrame:
    """Join scenario factors with industry capture fractions and project data.

    Args:
        projects_df: Per-project data (capacities, start years, unabated emissions).
        scenario_factors_df: Long-form scenario emissions factors.
        industry_df: Industry emissions with capture fractions.
        cfg: Scenario configuration with merge keys and column names.
    Returns:
        Merged DataFrame ready for emissions computation.
    """
    # Join scenario factors with industry capture fractions
    ccs_info_df = scenario_factors_df.merge(
        industry_df,
        right_on="product",
        left_on="co2_source",
    )
    ccs_info_df["product"] = ccs_info_df["product"].fillna("dac")

    # Join with project data
    keep = [c for c in cfg.facility_keep_cols if c in projects_df.columns]
    ccs_df = projects_df[keep].merge(
        ccs_info_df,
        left_on=cfg.projects_merge_on,
        right_on=cfg.ccs_emissions_merge_on,
    )
    return ccs_df


def compute_project_emissions(
    ccs_df: pd.DataFrame,
    cfg: ScenarioConfig,
) -> pd.DataFrame:
    """Compute 45Q credits, captured CO2, and net emissions.

    Pure arithmetic on columns — no I/O or merging.

    Args:
        ccs_df: Merged project + factors DataFrame.
        cfg: Scenario configuration with financial parameters.
    Returns:
        DataFrame with computed emissions columns added.
    """
    df = ccs_df.copy()

    # EOR fraction — needed for blended 45Q rate and oil production
    eor_lookup = {}
    for ws in cfg.weighted_scenarios:
        total = sum(ws.destination_pct.values())
        if total > 0:
            eor_lookup[ws.name] = ws.destination_pct.get("eor", 0) / total

    df["eor_fraction"] = df["scenario"].map(eor_lookup)
    not_weighted = df["eor_fraction"].isna()
    df.loc[not_weighted, "eor_fraction"] = (
        df.loc[not_weighted, "stated_co2_destination"] == "eor"
    ).astype(float)

    # Two candidate capture bases:
    #   implied_capture     — unabated × industry-standard capture fraction
    #   effective_nameplate — operator-supplied nameplate if present, else implied
    # `cfg.use_operator_nameplate_when_available` picks the "source of truth" that
    # feeds all baseline/weighted scenarios and all downstream calcs (45Q + oil).
    # The dedicated `nameplate_implied` and `nameplate` scenarios still show
    # their own semantics regardless of the flag (see block below).
    implied_capture = df["unabated_mtco2_per_yr"] * df["adjusted_capture_fraction_tco2"]
    effective_nameplate = df[cfg.nameplate_ccs_col].fillna(implied_capture)
    source_capture = (
        effective_nameplate
        if cfg.use_operator_nameplate_when_available
        else implied_capture
    )

    # Base captured = the config-chosen source of truth (implied by default;
    # effective_nameplate if `use_operator_nameplate_when_available=True`).
    # This applies to baseline, current_mix, all_gs, all_eor.
    captured = source_capture.copy()
    # The two dedicated nameplate scenarios always show their own semantics,
    # regardless of the config flag, so a user can compare against the other
    # option without re-running the pipeline:
    #   `nameplate_implied` → captured = implied_capture (always computed)
    #   `nameplate`         → captured = effective_nameplate (operator-when-present)
    is_np_implied = df["scenario"] == "nameplate_implied"
    captured[is_np_implied] = implied_capture[is_np_implied]
    is_np = df["scenario"] == "nameplate"
    captured[is_np] = effective_nameplate[is_np]
    # DAC has unabated=0 by convention, so implied_capture is 0 for it. Use the
    # effective nameplate for DAC across all scenarios — DAC's whole purpose is
    # the atmospheric removal defined by its nameplate capacity.
    is_dac = df["product"] == "dac"
    captured[is_dac] = effective_nameplate[is_dac]
    df["mtco2_per_yr_captured_with_ccs"] = captured

    # 45Q tax-credit calculation (blended EOR/GS rate, with inflation escalation).
    # Direct-air-capture facilities earn the higher IRC §45Q DAC rate for both
    # EOR and GS destinations; all other industries use the standard rates.
    # `captured` (not `source_capture`) is the correct basis so DAC — which has
    # unabated_mtco2=0 and therefore source_capture=0 — still earns 45Q on its
    # effective-nameplate tonnes.
    subsidy_basis = captured
    eligible = df[cfg.project_start_col] >= cfg.forty_five_q_earliest_start

    eor_rate = np.where(is_dac, cfg.usd_per_tco2_45q_dac, cfg.usd_per_tco2_45q_eor)
    gs_rate = np.where(is_dac, cfg.usd_per_tco2_45q_dac, cfg.usd_per_tco2_45q_gs)
    blended_45q_rate = (
        df["eor_fraction"] * eor_rate + (1 - df["eor_fraction"]) * gs_rate
    )
    base_annual_45q = subsidy_basis * blended_45q_rate

    df["avg_million_usd_per_yr_45q"] = 0.0
    df.loc[eligible, "avg_million_usd_per_yr_45q"] = df.loc[eligible].apply(
        lambda row: nominal_total(
            annual_cost=base_annual_45q[row.name],
            project_start_year=int(row[cfg.project_start_col]),
            project_duration=cfg.project_duration,
            escalation_year=cfg.escalation_year,
            inflation_rate=cfg.inflation_rate,
        )
        / cfg.project_duration,
        axis=1,
    )

    df["pv_million_usd_credits_45q"] = 0.0
    df.loc[eligible, "pv_million_usd_credits_45q"] = df.loc[eligible].apply(
        lambda row: present_value(
            annual_cost=base_annual_45q[row.name],
            project_start_year=int(row[cfg.project_start_col]),
            project_duration=cfg.project_duration,
            discount_rate=cfg.discount_rate,
            base_year=cfg.pv_valuation_year,
            escalation_year=cfg.escalation_year,
            inflation_rate=cfg.inflation_rate,
        ),
        axis=1,
    )

    # Equivalent solar energy a facility's 45Q stream could purchase, year-by-
    # year at that year's nominal LCOE (so the 45Q escalation and LCOE
    # escalation are both tracked in nominal dollars within each year).
    # Avoided-GHG column `mtco2e_avoided_from_solar` is populated downstream
    # in run_full_pipeline, where state grid intensities are in scope.
    df["mwh_solar_from_45q"] = 0.0
    df.loc[eligible, "mwh_solar_from_45q"] = df.loc[eligible].apply(
        lambda row: solar_mwh_from_45q(
            base_annual_45q_m_usd=base_annual_45q[row.name],
            project_start_year=int(row[cfg.project_start_col]),
            project_duration=cfg.project_duration,
            escalation_year=cfg.escalation_year,
            inflation_rate=cfg.inflation_rate,
            solar_lcoe_base_usd_per_mwh=cfg.solar_lcoe_usd_per_mwh,
            solar_lcoe_escalation_rate=cfg.solar_lcoe_escalation_rate,
            base_year=int(cfg.base_year),
        ),
        axis=1,
    )

    # Per-scenario CO2e emissions
    df["mtco2e_per_yr_emitted_because_of_ccs"] = (
        captured * df["tco2e_emitted_per_tco2_ccs"]
    )
    df["mtco2e_per_yr_emissions_including_ccs"] = (
        df[cfg.project_unabated_mtco2e_col]
        - captured
        + df["mtco2e_per_yr_emitted_because_of_ccs"]
    )

    # Per-scenario CO2 emissions (pure CO2, no methane GWP weighting)
    df["mtco2_per_yr_emitted_because_of_ccs"] = (
        captured * df["tco2_emitted_per_tco2_ccs"]
    )
    df["mtco2_per_yr_emissions_including_ccs"] = (
        df["unabated_mtco2_per_yr"]
        - captured
        + df["mtco2_per_yr_emitted_because_of_ccs"]
    )

    # Oil accounting (barrels + revenue) uses `captured` (not `source_capture`)
    # so DAC rows contribute oil barrels proportional to their effective
    # nameplate when EOR is their destination. For non-DAC industries,
    # `captured == source_capture` for the main scenarios (baseline, all_gs,
    # current_mix, all_eor); the nameplate scenarios use their own semantics.
    df["million_barrels_oil_per_yr"] = (
        captured * df["eor_fraction"] * cfg.recovery_factor_bbl_per_tco2
    )

    df["million_usd_per_yr_oil_revenue"] = (
        captured
        * df["eor_fraction"]
        * cfg.recovery_factor_bbl_per_tco2
        * cfg.profit_per_bbl_usd
    )
    df["nominal_million_usd_oil_revenue"] = df.apply(
        lambda row: nominal_total(
            annual_cost=row["million_usd_per_yr_oil_revenue"],
            project_start_year=int(row[cfg.project_start_col]),
            project_duration=cfg.project_duration,
            escalation_year=int(row[cfg.project_start_col]) + 1,
            inflation_rate=cfg.inflation_rate,
        ),
        axis=1,
    )
    df["pv_million_usd_oil_revenue"] = df.apply(
        lambda row: present_value(
            annual_cost=row["million_usd_per_yr_oil_revenue"],
            project_start_year=int(row[cfg.project_start_col]),
            project_duration=cfg.project_duration,
            discount_rate=cfg.discount_rate,
            base_year=cfg.pv_valuation_year,
            escalation_year=int(row[cfg.project_start_col]) + 1,
            inflation_rate=cfg.inflation_rate,
        ),
        axis=1,
    )

    return df


def compute_scenarios(
    projects_df: pd.DataFrame,
    emissions_factors_df: pd.DataFrame,
    scenario_config: ScenarioConfig,
    industry_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute total GHG emissions for each CCS project under multiple scenarios.

    Orchestrates: CO2e adjustment → weighted factor computation → data loading
    → merging → emissions math.

    ``unabated_mtco2e_per_yr`` is always recomputed from ``unabated_mtco2_per_yr``
    using the CO2e/CO2 ratio from the industry emissions data, so the result
    reflects the current methane scenario regardless of what the projects CSV
    contains.

    Args:
        projects_df: Per-project data (capacities, start years, unabated emissions).
        emissions_factors_df: Output of :func:`ccs_emissions.compute.compute_ccs_emissions_factors`.
        scenario_config: Typed scenario configuration.
        industry_df: Optional pre-computed industry emissions DataFrame.
            When provided, CSV read is skipped.

    Returns:
        Long-form DataFrame with per-(project, scenario) rows.
    """
    if industry_df is None:
        industry_df = pd.read_csv(scenario_config.emissions_for_industries_path)

    projects_df = _compute_unabated_emissions(projects_df, industry_df, scenario_config)

    scenario_factors_df = compute_scenario_emissions_factors(
        emissions_factors_df, scenario_config
    )
    filtered_industry_df = load_industry_emissions(
        scenario_config, industry_df=industry_df
    )
    merged_df = merge_project_factors(
        projects_df, scenario_factors_df, filtered_industry_df, scenario_config
    )
    return compute_project_emissions(merged_df, scenario_config)


def _compute_unabated_emissions(
    projects_df: pd.DataFrame,
    industry_df: pd.DataFrame,
    scenario_config: ScenarioConfig,
) -> pd.DataFrame:
    """Compute unabated CO2 and CO2e for each project from its annual product
    output and the industry-specific per-unit emission factors.

    For every non-DAC project::

        unabated_mtco2_per_yr  = product_amount × total_tco2_per_unit_product / 1e6
        unabated_mtco2e_per_yr = product_amount × total_tco2e_per_unit_product / 1e6

    Any values previously stored in the input projects CSV's
    ``unabated_mtco2_per_yr`` / ``unabated_mtco2e_per_yr`` columns are
    ignored and overwritten (a WARNING is logged when they are seen). DAC
    projects keep their zero-unabated convention.

    Fails loudly if:
      - A non-DAC project has missing / zero / negative ``product_amount``.
      - A project's product is not present in ``industry_df`` under the
        configured ``industry_emissions_which_flavor``.
      - A project's ``product_unit`` (normalized) does not match the
        industry's ``unit_of_product`` (normalized) from
        ``config_industries.yml``.
    """
    # Warn once per run that stored unabated_* values in the input CSV are
    # being overwritten, so nobody thinks they can hand-edit them. The
    # original values are preserved under `original_input_*` column names
    # so downstream consumers can compare computed vs. as-reported values.
    input_col_map = {
        "unabated_mtco2_per_yr": "original_input_unabated_mtco2_per_yr",
        "unabated_mtco2e_per_yr": "original_input_unabated_mtco2e_per_yr",
    }
    stale_cols = [c for c in input_col_map if c in projects_df.columns]
    if stale_cols:
        logger.warning(
            "Ignoring stored %s from projects CSV — pipeline recomputes "
            "these from product_amount × per-unit factors on every run. "
            "Original values preserved as %s.",
            stale_cols,
            [input_col_map[c] for c in stale_cols],
        )
        projects_df = projects_df.rename(
            columns={c: input_col_map[c] for c in stale_cols}
        )

    # One canonical per-unit row per product (via the configured
    # industrial_product flavor filter).
    factors = (
        industry_df.loc[
            industry_df["industrial_product"].isin(
                scenario_config.industry_emissions_which_flavor
            ),
            [
                "simple_product_name",
                "unit_of_product",
                "total_tco2_per_unit_product",
                "total_tco2e_per_unit_product",
            ],
        ]
        .drop_duplicates(subset="simple_product_name")
        .set_index("simple_product_name")
    )

    df = projects_df.copy()

    # --- Validation: non-DAC facilities must have positive product_amount ---
    if "product_amount" not in df.columns:
        raise ValueError(
            "Projects CSV is missing the required 'product_amount' column. "
            "Pipeline computes unabated emissions from "
            "product_amount × per-unit factors."
        )
    non_dac_mask = df["product"] != "dac"
    missing_mask = non_dac_mask & (
        df["product_amount"].isna() | ~df["product_amount"].gt(0)
    )
    if missing_mask.any():
        missing = df.loc[missing_mask, ["project_name", "product", "product_amount"]]
        raise ValueError(
            f"{len(missing)} non-DAC project(s) have missing / non-positive "
            "product_amount:\n" + missing.to_string(index=False)
        )

    # --- Validation: every non-DAC product must exist in industry factors ---
    missing_products = sorted(set(df.loc[non_dac_mask, "product"]) - set(factors.index))
    if missing_products:
        raise ValueError(
            "Products in projects CSV missing from industrial emissions "
            f"data (check industry_emissions_which_flavor): {missing_products}"
        )

    # --- Validation: unit consistency between projects CSV and industry cfg ---
    def _normalize_unit(u: object) -> str:
        if pd.isna(u):
            return ""
        s = str(u).strip().lower()
        for suffix in ("/yr", "/year", " per yr", " per year"):
            if s.endswith(suffix):
                s = s[: -len(suffix)]
                break
        parts = s.split()
        return parts[0] if parts else s

    if "product_unit" in df.columns:
        inconsistencies = []
        for _, row in df.loc[non_dac_mask].iterrows():
            prod = row["product"]
            cfg_unit = factors.loc[prod, "unit_of_product"]
            if _normalize_unit(cfg_unit) != _normalize_unit(row["product_unit"]):
                inconsistencies.append(
                    (row.get("project_name", "?"), prod, row["product_unit"], cfg_unit)
                )
        if inconsistencies:
            msg = "\n".join(
                f"  {name!r} (product={prod}): "
                f"projects CSV product_unit={pu!r}, "
                f"config unit_of_product={cu!r}"
                for name, prod, pu, cu in inconsistencies
            )
            raise ValueError(
                "product_unit in projects CSV does not match "
                "unit_of_product in config_industries.yml for:\n" + msg
            )

    # --- Compute unabated emissions per facility ---
    df["unabated_mtco2_per_yr"] = 0.0
    df["unabated_mtco2e_per_yr"] = 0.0
    for prod in df.loc[non_dac_mask, "product"].unique():
        pct_co2 = float(factors.loc[prod, "total_tco2_per_unit_product"])
        pct_co2e = float(factors.loc[prod, "total_tco2e_per_unit_product"])
        prod_mask = df["product"] == prod
        df.loc[prod_mask, "unabated_mtco2_per_yr"] = (
            df.loc[prod_mask, "product_amount"] * pct_co2 / 1e6
        )
        df.loc[prod_mask, "unabated_mtco2e_per_yr"] = (
            df.loc[prod_mask, "product_amount"] * pct_co2e / 1e6
        )
    return df


def _append_switching_scenario(
    scenarios_out: pd.DataFrame,
    scenario_config: ScenarioConfig,
    baseline_scenario: str = "baseline",
    tail_scenario: str = "all_eor",
    switching_name: str = "switching",
) -> pd.DataFrame:
    """Append synthetic `switching` rows to the concatenated scenarios frame.

    Each switching row represents a facility that follows its stated CO2
    destination for the first `D = project_duration` years (the 45Q window)
    and then switches to 100% EOR for the remaining `L − D` years of its
    facility lifetime. The row is built from two existing rows for the same
    `(uid, project_state, methane_scenario)` tuple: the baseline row (used
    for years 1..D) and the all_eor row (used for years D+1..L).

    Oil continues to flow after the 45Q window ends, but no additional 45Q
    credit is earned for those tail years (per IRC §45Q eligibility, which
    stops at 12 years). The 45Q PV columns are therefore copied from the
    baseline row without augmentation.

    Columns are populated as follows:
      * ``lifetime_mtco2e``, ``lifetime_mtco2_captured_with_ccs``,
        ``lifetime_million_barrels_oil``, ``lifetime_nominal_million_usd_oil_revenue``:
        exact `D × baseline + (L − D) × all_eor` closed-form sums.
      * ``lifetime_pv_million_usd_oil_revenue``: re-computed by
        :func:`present_value` on the two-period stream (baseline years 1..D
        plus all_eor years D+1..L), with inflation anchored at the baseline
        row's project start year so the tail period continues escalating.
      * Per-year rates (``mtco2e_per_yr_*``, ``million_barrels_oil_per_yr``,
        ``million_usd_per_yr_oil_revenue``, …): time-weighted mean
        `(D × baseline + (L − D) × all_eor) / L`. Synthetic — no actual
        year equals this value — but gives totals that match the exact
        lifetime sums when multiplied by L.
      * ``eor_fraction``: time-weighted mean of the two underlying rates.
      * CCS lifecycle factors (``tco2e_emitted_per_tco2_ccs``, …):
        time-weighted mean.
      * 45Q columns (``avg_million_usd_per_yr_45q``, ``pv_million_usd_credits_45q``,
        ``mwh_solar_from_45q``, ``mtco2e_avoided_from_solar``): copied from
        baseline unchanged (no credit for tail years).
      * ``pv_million_usd_oil_revenue`` (D-year semantics) and
        ``nominal_million_usd_oil_revenue`` (D-year semantics): copied from
        baseline. The 30-year versions live in the new ``lifetime_*``
        columns.
      * Unabated columns and metadata (``project_start``, ``retrofit_newbuild``,
        state tags, methane-scenario tags, etc.): copied from baseline
        (invariant across destination scenarios).
    """
    if baseline_scenario not in scenarios_out["scenario"].unique():
        logger.warning(
            "Skipping switching synthesis: baseline scenario %r not present.",
            baseline_scenario,
        )
        return scenarios_out
    if tail_scenario not in scenarios_out["scenario"].unique():
        logger.warning(
            "Skipping switching synthesis: tail scenario %r not present.",
            tail_scenario,
        )
        return scenarios_out

    D = scenario_config.project_duration
    L = scenario_config.facility_lifetime_years
    tail = L - D
    if tail < 0:
        raise ValueError(f"facility_lifetime_years={L} must be >= project_duration={D}")

    # Key on (uid or project_name, project_state, methane_scenario). `uid` is
    # not yet present at this point in the pipeline; use project_name.
    key_cols = ["project_name", "project_state", "methane_scenario"]
    baseline_df = scenarios_out[scenarios_out["scenario"] == baseline_scenario].copy()
    tail_df = scenarios_out[scenarios_out["scenario"] == tail_scenario].copy()

    merged = baseline_df.merge(
        tail_df,
        on=key_cols,
        suffixes=("_b", "_t"),
        how="inner",
    )
    if merged.empty:
        logger.warning("No matching baseline/all_eor pairs; no switching rows added.")
        return scenarios_out

    switching = pd.DataFrame(index=merged.index)

    # Columns we recognize as time-weighted averages ("per_yr" rates and the
    # CCS lifecycle factors). Everything else without a special rule is
    # copied from the baseline row.
    TIME_WEIGHTED_COLS = {
        "eor_fraction",
        "tco2e_emitted_per_tco2_ccs",
        "tco2_emitted_per_tco2_ccs",
        "net_tco2e_emitted_per_tco2_ccs",
        "net_tco2_emitted_per_tco2_ccs",
        "mtco2e_per_yr_emitted_because_of_ccs",
        "mtco2e_per_yr_emissions_including_ccs",
        "mtco2_per_yr_emitted_because_of_ccs",
        "mtco2_per_yr_emissions_including_ccs",
        "million_barrels_oil_per_yr",
        "million_usd_per_yr_oil_revenue",
        "tch4_per_yr_emissions_including_ccs",
    }

    # Exact two-period lifetime sums for quantities whose per-year rate is a
    # true physical rate (no inflation/escalation involved). The nominal oil-
    # revenue total is NOT in this map because it must track oil-price
    # escalation year-by-year; it is computed separately below (in parallel
    # with the PV computation) via `_lifetime_oil_nominal`.
    EXACT_LIFETIME_COLS = {
        "lifetime_mtco2e": ("mtco2e_per_yr_emissions_including_ccs", True),
        "lifetime_mtco2_captured_with_ccs": ("mtco2_per_yr_captured_with_ccs", True),
        "lifetime_million_barrels_oil": ("million_barrels_oil_per_yr", True),
    }

    # Walk baseline's columns; produce the switching value for each.
    all_cols = [c for c in scenarios_out.columns if c in baseline_df.columns]
    for col in all_cols:
        b_col = f"{col}_b"
        t_col = f"{col}_t"
        if b_col not in merged.columns:
            # Must have been a key column — copy directly.
            switching[col] = merged[col]
            continue

        if col in TIME_WEIGHTED_COLS:
            switching[col] = (D * merged[b_col] + tail * merged[t_col]) / L
        elif col in EXACT_LIFETIME_COLS:
            per_yr_col, _ = EXACT_LIFETIME_COLS[col]
            switching[col] = (
                D * merged[f"{per_yr_col}_b"] + tail * merged[f"{per_yr_col}_t"]
            )
        else:
            # Default: inherit from baseline (metadata, unabated, 45Q, …).
            switching[col] = merged[b_col]

    # Overwrite scalar identifiers.
    switching["scenario"] = switching_name
    if "assumed_co2_destination" in switching.columns:
        switching["assumed_co2_destination"] = switching_name

    # Re-compute PV of the full-lifetime oil stream.
    # Baseline PV already covers years y..y+D-1 → copy it.
    # Add PV of the tail: 100%-EOR annual rate for years y+D..y+L-1,
    # anchored to the baseline row's project_start for inflation escalation.
    cfg = scenario_config
    start_col = cfg.project_start_col  # "project_start"

    def _lifetime_oil_pv(row):
        base_pv = row[f"pv_million_usd_oil_revenue_b"]
        if tail == 0:
            return base_pv
        y_start = int(row[f"{start_col}_b"])
        tail_pv = present_value(
            annual_cost=row["million_usd_per_yr_oil_revenue_t"],
            project_start_year=y_start + D,
            project_duration=tail,
            discount_rate=cfg.discount_rate,
            base_year=cfg.pv_valuation_year,
            escalation_year=y_start + 1,
            inflation_rate=cfg.inflation_rate,
        )
        return base_pv + tail_pv

    switching["lifetime_pv_million_usd_oil_revenue"] = merged.apply(
        _lifetime_oil_pv, axis=1
    )

    # Nominal (un-discounted, but inflation-escalated) oil revenue over the
    # full L-year lifetime. Years 1..D use the baseline destination mix (already
    # captured by the row's D-year `nominal_million_usd_oil_revenue_b` with
    # escalation anchored at y_j+1); years D+1..L use the all_eor base rate,
    # with the *same* escalation anchor so the (1+u)^(y - y_j) multiplier is
    # continuous across the pivot. Matches methods Eq. ~switching_oil_rev.
    def _lifetime_oil_nominal(row):
        base_nominal = row["nominal_million_usd_oil_revenue_b"]
        if tail == 0:
            return base_nominal
        y_start = int(row[f"{start_col}_b"])
        tail_nominal = nominal_total(
            annual_cost=row["million_usd_per_yr_oil_revenue_t"],
            project_start_year=y_start + D,
            project_duration=tail,
            escalation_year=y_start + 1,
            inflation_rate=cfg.inflation_rate,
        )
        return base_nominal + tail_nominal

    switching["lifetime_nominal_million_usd_oil_revenue"] = merged.apply(
        _lifetime_oil_nominal, axis=1
    )

    # For every other scenario, oil only flows during the 45Q window, so
    # the D-year `nominal_*` and `pv_*` oil-revenue columns already equal
    # the full-lifetime total. For `switching`, oil flows for all L years
    # (baseline mix for years 1..D, then 100% EOR for years D+1..L), so
    # these two columns are overwritten with the lifetime totals so that
    # every downstream consumer reads the correct "revenue over this
    # scenario's oil-producing window" semantics without needing a
    # scenario-specific branch.
    switching["nominal_million_usd_oil_revenue"] = switching[
        "lifetime_nominal_million_usd_oil_revenue"
    ]
    switching["pv_million_usd_oil_revenue"] = switching[
        "lifetime_pv_million_usd_oil_revenue"
    ]

    switching = switching[all_cols]
    return pd.concat([scenarios_out, switching], ignore_index=True)


def run_full_pipeline(
    scenario_config: ScenarioConfig,
) -> Dict[str, pd.DataFrame]:
    """Run the full pipeline, returning all intermediate and final DataFrames.

    Loops over (project_state, methane_scenario) combinations. For each state,
    projects are filtered to that state and computed with the state's constants
    from ``cfg.state_constants``; each methane scenario further overrides
    ``gwp_ch4`` and ``methane_leak_rate`` on top of the state constants.
    Results are concatenated with ``project_state``, ``methane_scenario``,
    ``gwp_ch4``, and ``methane_leak_rate`` columns.

    Fails loudly if the projects CSV lacks a ``project_state`` column, or if any
    project references a state not defined in the infrastructure YAML's
    ``states:`` block.

    Args:
        scenario_config: Must have ``use_precomputed=False`` and paths to the
            infrastructure config, industries config, and CH4 assumptions config.

    Returns:
        Dict with keys ``scenarios``, ``ccs_emissions``, ``industrial_emissions``,
        each a long-form DataFrame tagged with state and methane scenario columns.
    """

    # Use latin1: projects CSVs are frequently exported from Excel and contain
    # non-UTF-8 bytes (e.g., em-dashes, smart quotes) that break the default codec.
    projects_df = pd.read_csv(scenario_config.projects_info_path, encoding="latin1")
    if "product_type_simple" in projects_df.columns:
        projects_df = projects_df.rename(columns={"product_type_simple": "product"})

    # Honor the `include` column if present: only project rows with include=1
    # (or any truthy value) count. This lets curators mark projects as
    # tentative / historical / not-yet-vetted without deleting them from the CSV.
    if "include" in projects_df.columns:
        excluded = projects_df[projects_df["include"] != 1]
        if len(excluded):
            logger.info("Excluding %d project(s) with include != 1", len(excluded))
        projects_df = projects_df[projects_df["include"] == 1].copy()

    cfg = CCSProjectConfig.from_yaml(
        scenario_config.infrastructure_config_path,
        scenario_config.industries_config_path,
    )

    if "project_state" not in projects_df.columns:
        raise ValueError(
            f"Projects CSV {scenario_config.projects_info_path} has no "
            "`project_state` column; per-state constants require it."
        )

    # Drop rows with missing project_start (needed for 45Q PV calculation).
    # Warn loudly so the omission is visible each run — the user should fix
    # the source CSV, not have these projects silently disappear.
    start_col = scenario_config.project_start_col
    missing_start = projects_df[projects_df[start_col].isna()]
    if len(missing_start):
        names = missing_start.get("project_name", pd.Series(dtype=object)).tolist()
        logger.warning(
            "Dropping %d project(s) with missing %s (fix source CSV to include): %s",
            len(missing_start),
            start_col,
            names,
        )
        projects_df = projects_df.dropna(subset=[start_col]).copy()

    project_states = set(projects_df["project_state"].dropna().unique())
    unknown = project_states - set(cfg.state_constants)
    if unknown:
        raise KeyError(
            f"Projects reference states with no infrastructure constants: "
            f"{sorted(unknown)}. Add these to the `states:` block in "
            f"{scenario_config.infrastructure_config_path}. "
            f"Currently defined: {sorted(cfg.state_constants)}."
        )

    methane_scenarios = scenario_config.methane_scenarios
    if not methane_scenarios:
        methane_scenarios = [
            MethaneScenario(
                name="baseline",
                gwp_ch4=cfg.constants.gwp_ch4,
                methane_leak_rate=cfg.constants.methane_leak_rate,
            )
        ]

    scenario_frames = []
    ccs_frames = []
    industry_frames = []

    for state_code in sorted(project_states):
        state_constants = cfg.state_constants[state_code]
        state_projects_df = projects_df[
            projects_df["project_state"] == state_code
        ].copy()

        # Compute state-specific unabated and captured CO2 (methane-invariant, but
        # varies with state via grid intensity and coal EF used in industry compute).
        state_cfg_base = replace(cfg, constants=state_constants)
        ie_config_base = IndustrialEmissionsConfig.from_project_config(state_cfg_base)
        ie_base = compute_industrial_emissions(ie_config_base)
        ie_filtered = ie_base.loc[
            ie_base["industrial_product"].isin(
                scenario_config.industry_emissions_which_flavor
            ),
            [
                "simple_product_name",
                "total_tco2_per_unit_product",
                "adjusted_capture_fraction_tco2",
                "is_methane_based",
            ],
        ].rename(columns={"simple_product_name": "product"})
        # Small lookup table used later to tag the ccs_emissions frame.
        _ind_to_methane = dict(
            zip(ie_filtered["product"], ie_filtered["is_methane_based"])
        )

        state_projects_df = state_projects_df.merge(
            ie_filtered, on="product", how="left"
        )
        state_projects_df["unabated_mtco2_per_yr"] = (
            state_projects_df["product_amount"]
            * state_projects_df["total_tco2_per_unit_product"]
            / 1e6
        )
        state_projects_df["computed_captured_mtco2_per_yr"] = (
            state_projects_df["product_amount"]
            * state_projects_df["total_tco2_per_unit_product"]
            * state_projects_df["adjusted_capture_fraction_tco2"]
            / 1e6
        )
        # DAC has no industry-side emission factor (industry lookup yielded NaN).
        # Convention: without CCS, a DAC plant does nothing — it neither emits
        # nor removes CO2 — so `unabated_mtco2_per_yr = 0`. With CCS, it captures
        # `nameplate_captured_mtco2_per_yr` from the atmosphere. Net emissions
        # then = 0 - nameplate + CCS_operational_emissions, correctly reflecting
        # the atmospheric removal offset by fossil energy used to run capture.
        # (Any CSV-provided `unabated_mtco2_per_yr` value for DAC is ignored;
        # the old -nameplate convention double-counted the removal.)
        is_dac = state_projects_df["product"] == "dac"
        state_projects_df.loc[is_dac, "unabated_mtco2_per_yr"] = 0.0
        state_projects_df.loc[
            is_dac, "computed_captured_mtco2_per_yr"
        ] = state_projects_df.loc[is_dac, "nameplate_captured_mtco2_per_yr"]
        state_projects_df = state_projects_df.drop(
            columns=["total_tco2_per_unit_product", "adjusted_capture_fraction_tco2"]
        )

        for ch4_scenario in methane_scenarios:
            modified_constants = replace(
                state_constants,
                gwp_ch4=ch4_scenario.gwp_ch4,
                methane_leak_rate=ch4_scenario.methane_leak_rate,
            )
            modified_cfg = replace(cfg, constants=modified_constants)

            emissions_factors_df = compute_ccs_emissions_factors(modified_cfg)

            ie_config = IndustrialEmissionsConfig.from_project_config(modified_cfg)
            industry_df = compute_industrial_emissions(ie_config)

            result = compute_scenarios(
                projects_df=state_projects_df,
                emissions_factors_df=emissions_factors_df,
                scenario_config=scenario_config,
                industry_df=industry_df,
            )

            # Avoided GHG from the "equivalent solar" counterfactual, computed
            # once the state's grid CO2e intensity is in scope. For each project
            # row: A_j = mwh_solar × bar_e_state / 1e6 (Mt CO2e). bar_e includes
            # fugitive methane from gas-fired generation on the grid.
            bar_e = modified_constants.grid_tco2_per_mwh + (
                modified_constants.methane_leak_rate
                * modified_constants.gwp_ch4
                * modified_constants.grid_tch4_per_mwh
            )
            if "mwh_solar_from_45q" in result.columns:
                result["mtco2e_avoided_from_solar"] = (
                    result["mwh_solar_from_45q"] * bar_e / 1e6
                )

            # Methane mass per year (metric tons CH4), derived from the
            # CO2e − CO2 difference and the methane scenario's GWP:
            # tCH4 = (MtCO2e − MtCO2) × 1e6 / gwp_ch4. Both the net-with-CCS
            # and unabated-counterfactual rates are written for symmetry
            # with the mtco2_* / mtco2e_* column pairs.
            gwp = modified_constants.gwp_ch4
            if gwp and gwp > 0:
                result["tch4_per_yr_emissions_including_ccs"] = (
                    (
                        result["mtco2e_per_yr_emissions_including_ccs"]
                        - result["mtco2_per_yr_emissions_including_ccs"]
                    )
                    * 1e6
                    / gwp
                )
                result["unabated_tch4_per_yr"] = (
                    (result["unabated_mtco2e_per_yr"] - result["unabated_mtco2_per_yr"])
                    * 1e6
                    / gwp
                )

            # Lifetime totals, using the paper's standard L-year facility
            # life with D years of CCS operation (D = project_duration,
            # L = facility_lifetime_years). Written for every project/scenario
            # row so downstream code (notebooks, summary tables) does not have
            # to repeat the arithmetic:
            #   lifetime_unabated_mtco2e            = L × unabated_mtco2e_per_yr
            #   lifetime_mtco2e                     = D × emissions_including_ccs
            #                                         + (L − D) × unabated_mtco2e_per_yr
            #   lifetime_mtco2_captured_with_ccs    = D × mtco2_per_yr_captured_with_ccs
            # The avoided-GHG from the equivalent-solar counterfactual is
            # anchored to the 45Q eligibility window (D years), not the
            # facility lifetime, so it is NOT prefixed with "lifetime_".
            # `mtco2e_avoided_from_solar` already represents the D-year sum.
            facility_lifetime_years = scenario_config.facility_lifetime_years
            ccs_active_years = scenario_config.project_duration
            unabated_tail_years = facility_lifetime_years - ccs_active_years
            if unabated_tail_years < 0:
                raise ValueError(
                    f"facility_lifetime_years={facility_lifetime_years} must be "
                    f">= project_duration={ccs_active_years}; got "
                    f"unabated_tail_years={unabated_tail_years}."
                )
            result["lifetime_unabated_mtco2e"] = (
                facility_lifetime_years * result["unabated_mtco2e_per_yr"]
            )
            result["lifetime_mtco2e"] = (
                ccs_active_years * result["mtco2e_per_yr_emissions_including_ccs"]
                + unabated_tail_years * result["unabated_mtco2e_per_yr"]
            )
            result["lifetime_mtco2_captured_with_ccs"] = (
                ccs_active_years * result["mtco2_per_yr_captured_with_ccs"]
            )

            # Lifetime oil accounting. For all standard destination scenarios
            # (baseline, current_mix, all_eor, all_gs, …), oil flows only
            # during the D-year 45Q window; lifetime totals therefore equal
            # the D-year totals already computed in `_compute_oil_metrics`.
            # The synthesized `switching` scenario (added post-concat) over-
            # writes these three columns with the two-period sum spanning
            # the baseline mix for years 1..D and all_eor for years D+1..L.
            result["lifetime_million_barrels_oil"] = (
                ccs_active_years * result["million_barrels_oil_per_yr"]
            )
            result["lifetime_nominal_million_usd_oil_revenue"] = result[
                "nominal_million_usd_oil_revenue"
            ]
            result["lifetime_pv_million_usd_oil_revenue"] = result[
                "pv_million_usd_oil_revenue"
            ]

            tag_cols = {
                "project_state": state_code,
                "methane_scenario": ch4_scenario.name,
                "gwp_ch4": ch4_scenario.gwp_ch4,
                "methane_leak_rate": ch4_scenario.methane_leak_rate,
            }
            result = result.assign(**tag_cols)
            scenario_frames.append(result)

            # Tag ccs_emissions rows with is_methane_based via co2_source lookup.
            ccs_out = emissions_factors_df.assign(**tag_cols)
            ccs_out["is_methane_based"] = ccs_out["co2_source"].map(_ind_to_methane)
            ccs_frames.append(ccs_out)
            industry_frames.append(industry_df.assign(**tag_cols))

    scenarios_out = pd.concat(scenario_frames, ignore_index=True)

    # Synthesize the `switching` scenario: for each facility × methane
    # scenario, combine the baseline row (years 1..D at stated destinations)
    # with the all_eor row (years D+1..L at 100% EOR). Oil accounting on
    # this row is anchored to the full facility lifetime via the new
    # lifetime_* columns; 45Q PV is inherited from baseline only (no credit
    # for the post-D tail years).
    scenarios_out = _append_switching_scenario(scenarios_out, scenario_config)
    # Second switching variant: start from the current-mix destination
    # (92% EOR / 8% GS, the industry-typical today) for years 1..D, then
    # switch to 100% EOR for years D+1..L.
    scenarios_out = _append_switching_scenario(
        scenarios_out,
        scenario_config,
        baseline_scenario="current_mix",
        switching_name="switching_from_current",
    )

    scenarios_out["uid"] = scenarios_out["project_name"].map(
        lambda n: re.sub(r"[^a-z0-9]+", "-", n.lower()).strip("-")
    )
    methane_part = scenarios_out.get(
        "methane_scenario", pd.Series("", index=scenarios_out.index)
    )
    scenarios_out["scenario_id"] = scenarios_out["scenario"].where(
        methane_part == "", scenarios_out["scenario"] + "-" + methane_part
    )

    result = {
        "scenarios": scenarios_out,
        "ccs_emissions": pd.concat(ccs_frames, ignore_index=True),
        "industrial_emissions": pd.concat(industry_frames, ignore_index=True),
    }

    if scenario_config.ccs_emissions_factors_path:
        result["ccs_emissions"].to_csv(
            scenario_config.ccs_emissions_factors_path, index=False
        )
        logger.info(
            "CCS emissions factors written to %s",
            scenario_config.ccs_emissions_factors_path,
        )
    if scenario_config.emissions_for_industries_path:
        result["industrial_emissions"].to_csv(
            scenario_config.emissions_for_industries_path, index=False
        )
        logger.info(
            "Industrial emissions written to %s",
            scenario_config.emissions_for_industries_path,
        )

    return result


def compute_scenarios_full_pipeline(
    scenario_config: ScenarioConfig,
) -> pd.DataFrame:
    """Run the full pipeline, returning just the scenarios DataFrame.

    Convenience wrapper around :func:`run_full_pipeline`.
    """
    return run_full_pipeline(scenario_config)["scenarios"]


def compute_scenario_summary(
    scenarios_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-scenario totals for oil production, subsidies, and emissions.

    Args:
        scenarios_df: Output of :func:`compute_scenarios` or
            :func:`compute_scenarios_full_pipeline`.

    Returns:
        DataFrame with one row per (scenario, methane_scenario) containing
        aggregate barrels, subsidy dollars, and emissions.
    """
    group_cols = ["scenario"]
    if "methane_scenario" in scenarios_df.columns:
        group_cols.append("methane_scenario")

    agg_spec = {
        "total_nameplate_mtco2_per_yr": ("nameplate_captured_mtco2_per_yr", "sum"),
        "total_million_barrels_oil_per_yr": ("million_barrels_oil_per_yr", "sum"),
        "total_avg_million_usd_per_yr_45q": ("avg_million_usd_per_yr_45q", "sum"),
        "total_pv_million_usd_credits_45q": ("pv_million_usd_credits_45q", "sum"),
        "total_mtco2e_per_yr_emissions": (
            "mtco2e_per_yr_emissions_including_ccs",
            "sum",
        ),
    }
    # Optional solar columns (populated only by run_full_pipeline; the
    # avoided-GHG column needs state grid intensities in scope).
    if "mwh_solar_from_45q" in scenarios_df.columns:
        agg_spec["total_mwh_solar_from_45q"] = ("mwh_solar_from_45q", "sum")
    if "mtco2e_avoided_from_solar" in scenarios_df.columns:
        agg_spec["total_mtco2e_avoided_from_solar"] = (
            "mtco2e_avoided_from_solar",
            "sum",
        )
    return scenarios_df.groupby(group_cols).agg(**agg_spec).reset_index()
