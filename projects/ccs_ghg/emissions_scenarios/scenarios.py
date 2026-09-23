"""Compute total GHG emissions for various industrial CCS scenarios."""

# pylint: disable=cell-var-from-loop
import logging
import re
from dataclasses import replace
from typing import Dict, Optional

import pandas as pd

from projects.ccs_ghg.ccs_emissions.compute import compute_ccs_emissions_factors
from projects.ccs_ghg.config import CCSProjectConfig
from projects.ccs_ghg.emissions_scenarios.config import MethaneScenario, ScenarioConfig
from projects.ccs_ghg.industrial_emissions.compute import compute_industrial_emissions
from projects.ccs_ghg.industrial_emissions.config import IndustrialEmissionsConfig

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
    discount_rate: float = 0.051,
    base_year: int = None,
    escalation_year: int = None,
    inflation_rate: float = 0.0,
) -> float:
    """Present value of a stream of annual payments, optionally escalated for inflation."""
    if base_year is None:
        base_year = project_start_year

    pv = 0.0
    for year in range(project_start_year, project_start_year + project_duration):
        amount = _escalated_amount(annual_cost, year, escalation_year, inflation_rate)
        t = year - base_year + 1
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

    # 45Q tax-credit calculation (blended EOR/GS rate, with inflation escalation)
    subsidy_basis = source_capture
    eligible = df[cfg.project_start_col] >= cfg.forty_five_q_earliest_start

    blended_45q_rate = (
        df["eor_fraction"] * cfg.usd_per_tco2_45q_eor
        + (1 - df["eor_fraction"]) * cfg.usd_per_tco2_45q_gs
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
            base_year=int(cfg.base_year),
            escalation_year=cfg.escalation_year,
            inflation_rate=cfg.inflation_rate,
        ),
        axis=1,
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

    # Oil accounting (barrels + revenue) inherits the same `source_capture`
    # basis so barrels and revenue stay consistent within a run.
    df["million_barrels_oil_per_yr"] = (
        source_capture * df["eor_fraction"] * cfg.recovery_factor_bbl_per_tco2
    )

    df["million_usd_per_yr_oil_revenue"] = (
        source_capture
        * df["eor_fraction"]
        * cfg.recovery_factor_bbl_per_tco2
        * cfg.profit_per_bbl_usd
    )
    df["nominal_million_usd_oil_revenue"] = df.apply(
        lambda row: nominal_total(
            annual_cost=row["million_usd_per_yr_oil_revenue"],
            project_start_year=int(row[cfg.project_start_col]),
            project_duration=cfg.project_duration,
            escalation_year=int(row[cfg.project_start_col]),
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
            base_year=int(cfg.base_year),
            escalation_year=int(row[cfg.project_start_col]),
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

    projects_df = _adjust_unabated_co2e(projects_df, industry_df, scenario_config)

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


def _adjust_unabated_co2e(
    projects_df: pd.DataFrame,
    industry_df: pd.DataFrame,
    scenario_config: ScenarioConfig,
) -> pd.DataFrame:
    """Recompute unabated CO2e using the CO2e/CO2 ratio from industrial emissions.

    The projects CSV stores a fixed ``unabated_mtco2e_per_yr`` computed at one
    GWP/leak-rate.  This rescales it using the per-industry ratio of
    ``total_tco2e / total_tco2`` from the industrial emissions module, which
    reflects the current methane scenario's GWP and leak rate.

    ``unabated_mtco2_per_yr`` (pure CO2, invariant) is preserved; only
    ``unabated_mtco2e_per_yr`` is updated.
    """
    selected = industry_df.loc[
        industry_df["industrial_product"].isin(
            scenario_config.industry_emissions_which_flavor
        ),
        [
            "simple_product_name",
            "total_tco2_per_unit_product",
            "total_tco2e_per_unit_product",
        ],
    ].copy()

    mask = selected["total_tco2_per_unit_product"] != 0
    selected.loc[mask, "co2e_co2_ratio"] = (
        selected.loc[mask, "total_tco2e_per_unit_product"]
        / selected.loc[mask, "total_tco2_per_unit_product"]
    )
    selected["co2e_co2_ratio"] = selected["co2e_co2_ratio"].fillna(1.0)

    ratio_map = selected.set_index("simple_product_name")["co2e_co2_ratio"].to_dict()

    df = projects_df.copy()
    ratio = df["product"].map(ratio_map).fillna(1.0)
    df["unabated_mtco2e_per_yr"] = df["unabated_mtco2_per_yr"] * ratio
    return df


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
        # Preserve pre-existing unabated_mtco2_per_yr values, if the CSV
        # provides them (some fixtures do). Missing column → all-NaN sentinel.
        if "unabated_mtco2_per_yr" in state_projects_df.columns:
            original_unabated_mtco2 = state_projects_df["unabated_mtco2_per_yr"].copy()
        else:
            original_unabated_mtco2 = pd.Series(
                float("nan"), index=state_projects_df.index
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

    return (
        scenarios_df.groupby(group_cols)
        .agg(
            total_nameplate_mtco2_per_yr=(
                "nameplate_captured_mtco2_per_yr",
                "sum",
            ),
            total_million_barrels_oil_per_yr=(
                "million_barrels_oil_per_yr",
                "sum",
            ),
            total_avg_million_usd_per_yr_45q=("avg_million_usd_per_yr_45q", "sum"),
            total_pv_million_usd_credits_45q=(
                "pv_million_usd_credits_45q",
                "sum",
            ),
            total_mtco2e_per_yr_emissions=(
                "mtco2e_per_yr_emissions_including_ccs",
                "sum",
            ),
        )
        .reset_index()
    )
