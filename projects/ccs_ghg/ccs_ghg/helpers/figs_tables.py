"""Figure and table helpers for the Gulf Coast CCS paper.

Single standalone module holding the 24 figure/table builders used by
`notebooks/ccs_ghg.ipynb`. Each function builds one figure or one table
for the final paper (or its SI) from the pipeline output CSVs in
`data/`: facility-level emissions, CCS scenarios, and industry totals.
Figures are returned as matplotlib `Figure`/`Axes` pairs or Plotly
`Figure` objects; tables as pandas DataFrames. Module-level constants
(color palettes, industry markers, scenario-name mappings) are shared
across the builders so styling stays consistent.

The per-figure and per-table source modules alongside this file
(`plot_*.py`, `table*.py`, `summarize_gulf_coast_industries.py`) carry
the same implementations; this module is a standalone copy so the
notebook has one import.
"""
import datetime
import logging
import os
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import click
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Patch

from ccs_ghg._vendored.io import yaml_to_dict

logger = logging.getLogger(__name__)

INDUSTRY_COLORS = {
    "ammonia": "#2a78d6",
    "power_ng": "#1baf7a",
    "power_ng_new_build": "#1baf7a",
    "power_ng_retrofit": "#1baf7a",
    "power_ng_data_center": "#9c27b0",
    "hydrogen": "#eda100",
    "lng": "#008300",
    "power_coal": "#4a3aa7",
    "ethanol": "#e34948",
    "petrochemicals": "#e87ba4",
    "refined_fuels": "#6e6d68",
    "refining": "#6e6d68",
    "refining_and_petrochemicals": "#a3568f",
    "power_generation_gas_and_coal": "#4a3aa7",
    "iron_steel": "#5c4a35",
    "methanol": "#eb6834",
    "ngp": "#888780",
    "dac": "#6b4513",
    "dac_with_ngcc": "#8a5a2a",
}

INDUSTRY_MARKERS = {
    "hydrogen": "^",  # triangle up
    "ammonia": "v",  # triangle down
    "methanol": "<",  # triangle left
    "iron_steel": ">",  # triangle right
    "lng": "s",  # square
    "ngp": "D",  # diamond
    "power_ng": "*",  # star
    "power_ng_new_build": "*",
    "power_ng_retrofit": "*",
    "power_ng_data_center": "h",  # hexagon (pointy top)
    "power_coal": "p",  # pentagon
    "power_generation_gas_and_coal": "*",
    "refined_fuels": "o",  # circle
    "refining": "o",
    "refining_and_petrochemicals": "o",
    "petrochemicals": "P",  # plus (filled)
    "ethanol": "X",  # x (filled)
    "dac": "8",  # octagon
    "dac_with_ngcc": "8",
}

class _IShapeHandler(HandlerBase):
    """Legend handler drawing an 'I' shape: two horizontal bars with an
    upward-facing filled arrow between them. Used by
    `plot_emissions_breakdown_combined` to represent the EOR-increment
    slice on each CCS bar."""

    def __init__(
        self,
        color: str = "#b71c1c",
        border_lw: float = 1.5,
        arrow_lw: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._color = color
        self._border_lw = border_lw
        self._arrow_lw = arrow_lw

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        x0 = xdescent
        x1 = xdescent + width
        y0 = ydescent
        y1 = ydescent + height
        x_mid = xdescent + width / 2
        bottom = Line2D(
            [x0, x1],
            [y0, y0],
            color=self._color,
            lw=self._border_lw,
            transform=trans,
            solid_capstyle="butt",
        )
        top = Line2D(
            [x0, x1],
            [y1, y1],
            color=self._color,
            lw=self._border_lw,
            transform=trans,
            solid_capstyle="butt",
        )
        arrow = FancyArrowPatch(
            (x_mid, y0),
            (x_mid, y1),
            arrowstyle="-|>",
            mutation_scale=8,
            color=self._color,
            lw=self._arrow_lw,
            transform=trans,
        )
        return [bottom, top, arrow]

_FONT_PRESETS: Dict[str, Dict[str, int]] = {
    "report": {
        "title": 22,
        "axis_label": 16,
        "tick": 14,
        "bar_label": 15,
        "legend": 12,
        "legend_title": 13,
        "annotation": 14,
        "sankey_node": 20,
        "sankey_title": 24,
        "sankey_layout": 22,
        "sankey_bracket": 20,
    },
    "presentation": {
        "title": 24,
        "axis_label": 18,
        "tick": 15,
        "bar_label": 16,
        "legend": 14,
        "legend_title": 15,
        "annotation": 16,
        "sankey_node": 26,
        "sankey_title": 30,
        "sankey_layout": 28,
        "sankey_bracket": 26,
    },
}

_DEFAULT_PLOTTING_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "config", "config_plotting.yml"
)

def _resolve_display_mode(display_mode: Optional[str] = None) -> str:
    if display_mode is None:
        try:
            cfg = yaml_to_dict(_DEFAULT_PLOTTING_CONFIG_PATH)
            display_mode = cfg.get("display_mode", "report")
        except Exception:
            display_mode = "report"
    return display_mode

def _resolve_fonts(display_mode: Optional[str] = None) -> Dict[str, int]:
    return _FONT_PRESETS.get(
        _resolve_display_mode(display_mode), _FONT_PRESETS["report"]
    )

def _build_breakdown_data(
    industrial_emissions_df: pd.DataFrame,
    ccs_emissions_df: pd.DataFrame,
    methane_scenario: str,
    co2_destination: str,
    industries: Optional[List[str]] = None,
    which_flavor: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build the per-industry, per-tCO2-captured emissions breakdown used by the
    stacked-bar plots (`plot_emissions_breakdown`, `plot_emissions_breakdown_combined`,
    and their state-total siblings).

    For a single methane scenario and CO2 destination, this filters, dedupes,
    and normalizes the two upstream pipeline outputs so each remaining industry
    is described by a fixed set of components — all expressed in tCO2e per
    tCO2 captured with CCS. That common denominator lets the stacked bars
    compare per-industry emissions "cost" of storing one ton of CO2, regardless
    of facility size or product tonnage.

    Steps:
      1. Filter `industrial_emissions_df` to the requested `methane_scenario`,
         and `ccs_emissions_df` to the requested `methane_scenario` +
         `co2_destination` ('eor' or 'gs').
      2. Optionally narrow to a specific `industrial_product` flavor (e.g.
         "power (CCGT)") and/or a specific list of `simple_product_name`s.
      3. Keep one row per `simple_product_name` (first hit) on each side —
         drops the state-loop duplicates introduced by the post-refactor
         pipeline; callers should pre-filter to a specific state if they want
         state-specific factors rather than the first-state defaults.
      4. Compute `tco2_captured_per_unit = total_tco2 × adjusted_capture_fraction_tco2`
         and discard industries with zero capture (the ratios below are
         undefined for them).
      5. Divide every component by `tco2_captured_per_unit` so every column is
         "tCO2e per tCO2 captured".

    Component columns (all per tCO2 captured):
      - `process_ch4`             CH4 from the industrial process, as CO2e
                                  (i.e. `process_tco2e - process_tco2`).
      - `combustion_ch4`          CH4 from on-site combustion + grid electricity.
      - `uncaptured_process_co2`  Process CO2 that escapes CCS.
      - `uncaptured_combustion_co2`  Combustion CO2 that escapes CCS, plus
                                  ALL grid CO2 (grid emissions are never captured).
      - `captured_process_co2`    Process CO2 stored by CCS. **Negative** so it
                                  stacks downward in the bar chart.
      - `captured_combustion_co2` Combustion CO2 stored by CCS. **Negative**.
      - `ccs_co2` / `ccs_ch4`     CO2 and CH4-as-CO2e emitted by CCS operations
                                  themselves (compression, transport, injection),
                                  from the CCS emissions factors table joined
                                  on `simple_product_name == co2_source`.

    Args:
        industrial_emissions_df: Output of the industrial-emissions pipeline
            (per-industry CO2/CO2e/CH4 columns and capture fractions).
        ccs_emissions_df: Output of the CCS emissions pipeline (per-
            `(co2_source, co2_destination)` lifecycle emission factors).
        methane_scenario: Which methane scenario to select (e.g. "edf20",
            "howarth", "industry_cited_gwp20").
        co2_destination: 'eor' or 'gs'.
        industries: Restrict to these `simple_product_name` values.
        which_flavor: Restrict to these `industrial_product` values
            (e.g. `["power (CCGT)"]`) — useful when one simple product has
            several flavors and only one is wanted.

    Returns:
        DataFrame with one row per remaining industry, carrying the eight
        component columns above plus the original merge keys and factor
        columns for downstream sorting/labeling.
    """

    # ── 1. Filter both inputs to the requested scenario + destination ──────
    ie = industrial_emissions_df.loc[
        industrial_emissions_df["methane_scenario"] == methane_scenario
    ].copy()
    ef = ccs_emissions_df.loc[
        (ccs_emissions_df["methane_scenario"] == methane_scenario)
        & (ccs_emissions_df["co2_destination"] == co2_destination)
    ].copy()

    if which_flavor and "industrial_product" in ie.columns:
        ie = ie[ie["industrial_product"].isin(which_flavor)]
    if industries:
        ie = ie[ie["simple_product_name"].isin(industries)]
        ef = ef[ef["co2_source"].isin(industries)]

    # Dedupe to one row per industry. The pipeline's state loop can produce
    # multiple rows per (industry, scenario, destination); we take the first
    # match. Callers wanting state-specific factors should pre-filter by
    # `project_state` before calling.
    ie = ie.drop_duplicates(subset="simple_product_name", keep="first")
    ef = ef.drop_duplicates(subset=["co2_source", "co2_destination"], keep="first")

    # ── 2. Normalize every component to tCO2e per tCO2 captured with CCS ──
    ie["tco2_captured_per_unit"] = (
        ie["total_tco2_per_unit_product"] * ie["adjusted_capture_fraction_tco2"]
    )
    ie = ie.loc[ie["tco2_captured_per_unit"] > 0].copy()
    if ie.empty:
        return ie
    scale = 1.0 / ie["tco2_captured_per_unit"]

    # ── 3. Compute the eight breakdown components ─────────────────────────
    # Short aliases so the arithmetic below reads as physics rather than
    # DataFrame indexing. `.get` on grid columns lets older CSVs (without
    # grid electricity) work by treating grid contributions as zero.
    p_co2 = ie["process_tco2_per_unit_product"]
    p_co2e = ie["process_tco2e_per_unit_product"]
    c_co2 = ie["combustion_tco2_per_unit_product"]
    c_co2e = ie["combustion_tco2e_per_unit_product"]
    g_co2 = ie.get("grid_tco2_per_unit_product", 0.0)
    g_co2e = ie.get("grid_tco2e_per_unit_product", 0.0)
    p_cf = ie["process_capture_fraction_tco2"]
    c_cf = ie["combustion_capture_fraction_tco2"]

    # CH4-only contribution = CO2e (CO2 + CH4-as-CO2e) minus CO2-only piece.
    # Grid electricity's CH4 is bucketed under combustion since neither is CCS-captured.
    ie["process_ch4"] = (p_co2e - p_co2) * scale
    ie["combustion_ch4"] = ((c_co2e - c_co2) + (g_co2e - g_co2)) * scale

    # Uncaptured CO2 → positive (stacks upward from zero).
    # Grid CO2 is never captured, so it lives entirely in "uncaptured combustion".
    ie["uncaptured_process_co2"] = p_co2 * (1 - p_cf) * scale
    ie["uncaptured_combustion_co2"] = (c_co2 * (1 - c_cf) + g_co2) * scale

    # Captured CO2 → negative (stacks downward from zero in the bar chart).
    ie["captured_process_co2"] = -p_co2 * p_cf * scale
    ie["captured_combustion_co2"] = -c_co2 * c_cf * scale

    # ── 4. Attach CCS-operations emission factors (also per tCO2 captured) ─
    merged = ie.merge(
        ef[["co2_source", "tco2_emitted_per_tco2_ccs", "tco2e_emitted_per_tco2_ccs"]],
        left_on="simple_product_name",
        right_on="co2_source",
        how="left",
    )
    merged["ccs_co2"] = merged["tco2_emitted_per_tco2_ccs"].fillna(0)
    merged["ccs_ch4"] = (
        merged["tco2e_emitted_per_tco2_ccs"].fillna(0) - merged["ccs_co2"]
    )
    return merged

def _compute_dual_axis_records(
    industrial_emissions_df,
    ccs_emissions_df,
    scenarios_df,
    methane_scenario,
    destination_scenarios,
    industries,
    which_flavor,
    component_cols,
):
    """Compute one record per destination scenario for the dual-axis stacked
    bar chart, using the passed dataframes as a single (assumed single-state)
    slice. Returned as a list of dicts aligned with destination_scenarios."""
    eor_df = _build_breakdown_data(
        industrial_emissions_df,
        ccs_emissions_df,
        methane_scenario,
        "eor",
        industries,
        which_flavor,
    ).set_index("simple_product_name")
    gs_df = _build_breakdown_data(
        industrial_emissions_df,
        ccs_emissions_df,
        methane_scenario,
        "gs",
        industries,
        which_flavor,
    ).set_index("simple_product_name")

    common = eor_df.index.intersection(gs_df.index)
    ccs_cols = {"ccs_co2", "ccs_ch4"}

    ref = scenarios_df[
        scenarios_df["methane_scenario"] == methane_scenario
    ].drop_duplicates(subset="uid")
    ref = ref[ref["product"].isin(common)]
    captured_by_industry = ref.groupby("product")[
        "mtco2_per_yr_captured_with_ccs"
    ].sum()
    total_captured = captured_by_industry.sum()

    records = []
    for scen in destination_scenarios:
        row = {"scenario": scen["name"], "label": scen["label"]}
        zero_lc = scen.get("zero_lifecycle", False)

        if scen.get("from_data"):
            scen_df = scenarios_df[
                (scenarios_df["methane_scenario"] == methane_scenario)
                & (scenarios_df["scenario"] == scen["name"])
            ]
            scen_df = scen_df[scen_df["product"].isin(common)]
            for col in component_cols:
                if zero_lc and col in ccs_cols:
                    row[col] = 0.0
                    continue
                weighted = 0.0
                for _, proj in scen_df.iterrows():
                    ef = proj["eor_fraction"]
                    ind = proj["product"]
                    factor = ef * eor_df.loc[ind, col] + (1 - ef) * gs_df.loc[ind, col]
                    weighted += factor * proj["mtco2_per_yr_captured_with_ccs"]
                row[col] = weighted
            row["_captured"] = scen_df["mtco2_per_yr_captured_with_ccs"].sum()
        elif zero_lc:
            for col in component_cols:
                if col in ccs_cols:
                    row[col] = 0.0
                else:
                    row[col] = sum(
                        eor_df.loc[ind, col] * captured_by_industry[ind]
                        for ind in captured_by_industry.index
                    )
            row["_captured"] = total_captured
        else:
            eor_frac = scen["eor_pct"] / 100
            gs_frac = 1 - eor_frac
            for col in component_cols:
                row[col] = sum(
                    (eor_frac * eor_df.loc[ind, col] + gs_frac * gs_df.loc[ind, col])
                    * captured_by_industry[ind]
                    for ind in captured_by_industry.index
                )
            row["_captured"] = total_captured

        row["_net"] = sum(row[c] for c in component_cols)
        records.append(row)
    return records

def _compute_dac_records(
    ccs_emissions_df,
    scenarios_df,
    methane_scenario,
    destination_scenarios,
):
    """Compute DAC's contribution to each destination-scenario stacked bar.

    DAC has zero per-unit-of-product emissions (its "unabated" is 0 by
    convention: no CCS → no operation), so it drops out of the standard
    per-unit-captured math in `_compute_dual_axis_records`. But its captured
    tonnes and CCS-related lifecycle emissions ARE real and belong in the
    stacked bar. Returns a list of dicts aligned with destination_scenarios,
    each carrying (captured_process_co2, ccs_co2, ccs_ch4, _captured)
    contributions for DAC. Other components stay 0."""
    dac_scen = scenarios_df[
        (scenarios_df["methane_scenario"] == methane_scenario)
        & (scenarios_df["product"] == "dac")
    ]
    if dac_scen.empty:
        return [
            {"scenario": s["name"], "_captured": 0.0} for s in destination_scenarios
        ]

    # Per-tCO2-captured CCS factors for DAC, keyed by co2_destination.
    dac_ccs = ccs_emissions_df[
        (ccs_emissions_df["methane_scenario"] == methane_scenario)
        & (ccs_emissions_df["co2_source"] == "dac")
    ]
    if dac_ccs.empty:
        return [
            {"scenario": s["name"], "_captured": 0.0} for s in destination_scenarios
        ]
    # Average across states within this call (caller already state-filtered).
    ccs_by_dest = dac_ccs.groupby("co2_destination").agg(
        tco2=("tco2_emitted_per_tco2_ccs", "mean"),
        tco2e=("tco2e_emitted_per_tco2_ccs", "mean"),
    )

    def _factors_for(destination):
        row = ccs_by_dest.loc[destination]
        return float(row["tco2"]), float(row["tco2e"])

    eor_tco2, eor_tco2e = (
        _factors_for("eor") if "eor" in ccs_by_dest.index else (0.0, 0.0)
    )
    gs_tco2, gs_tco2e = _factors_for("gs") if "gs" in ccs_by_dest.index else (0.0, 0.0)

    out = []
    for scen in destination_scenarios:
        row = {"scenario": scen["name"]}
        if scen.get("from_data"):
            # Weight per project's stated destination
            df = dac_scen[dac_scen["scenario"] == scen["name"]]
            captured = df["mtco2_per_yr_captured_with_ccs"].sum()
            ccs_co2 = sum(
                proj["mtco2_per_yr_captured_with_ccs"]
                * (
                    proj["eor_fraction"] * eor_tco2
                    + (1 - proj["eor_fraction"]) * gs_tco2
                )
                for _, proj in df.iterrows()
            )
            ccs_tco2e = sum(
                proj["mtco2_per_yr_captured_with_ccs"]
                * (
                    proj["eor_fraction"] * eor_tco2e
                    + (1 - proj["eor_fraction"]) * gs_tco2e
                )
                for _, proj in df.iterrows()
            )
        elif scen.get("zero_lifecycle"):
            # Nameplate case: no lifecycle emissions attributed
            df = dac_scen[dac_scen["scenario"] == scen["name"]]
            captured = df["mtco2_per_yr_captured_with_ccs"].sum()
            ccs_co2 = 0.0
            ccs_tco2e = 0.0
        else:
            eor_frac = scen["eor_pct"] / 100.0
            gs_frac = 1 - eor_frac
            # Total captured DAC tonnes (same across weighted scenarios)
            captured = dac_scen.drop_duplicates(subset="uid")[
                "mtco2_per_yr_captured_with_ccs"
            ].sum()
            ccs_co2 = captured * (eor_frac * eor_tco2 + gs_frac * gs_tco2)
            ccs_tco2e = captured * (eor_frac * eor_tco2e + gs_frac * gs_tco2e)
        ccs_ch4 = ccs_tco2e - ccs_co2

        # Populate only the components DAC actually touches:
        # captured_process_co2 is stored as a negative value in the standard
        # breakdown (below-axis "stored" bar); mirror that convention here.
        row["captured_process_co2"] = -captured
        row["ccs_co2"] = ccs_co2
        row["ccs_ch4"] = ccs_ch4
        row["_captured"] = captured
        out.append(row)
    return out

def _compute_unabated_record(
    industrial_emissions_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    methane_scenario: str,
    industries: Optional[List[str]] = None,
    which_flavor: Optional[List[str]] = None,
) -> dict:
    """State-total emissions breakdown for the unabated (no-CCS) scenario.

    Returns a dict shaped like the entries in `_compute_dual_axis_records`
    output, but with only the four non-CCS component columns populated:
    process_ch4, combustion_ch4, uncaptured_process_co2, uncaptured_combustion_co2
    (the "captured_*" and "ccs_*" components are 0 because there is no CCS).

    Per-industry, per-state component *fractions* of total tCO2e/unit product
    are read from industrial_emissions_df; those fractions are multiplied by
    each project's `unabated_mtco2e_per_yr` (from scenarios_df) and summed to
    produce state-total per-component MtCO2e/yr. The four components sum to
    the state-total unabated tonnage.
    """
    ie = industrial_emissions_df[
        industrial_emissions_df["methane_scenario"] == methane_scenario
    ].copy()
    if which_flavor and "industrial_product" in ie.columns:
        ie = ie[ie["industrial_product"].isin(which_flavor)]
    if industries:
        ie = ie[ie["simple_product_name"].isin(industries)]

    # Component fractions of total_tco2e per unit product. Grid CH4/CO2 fold
    # into combustion (they aren't a separate CCS-capturable bucket).
    total = ie["total_tco2e_per_unit_product"]
    safe = total.where(total > 0, np.nan)
    grid_co2 = ie.get("grid_tco2_per_unit_product", 0.0)
    grid_co2e = ie.get("grid_tco2e_per_unit_product", 0.0)
    ie["_frac_process_ch4"] = (
        (ie["process_tco2e_per_unit_product"] - ie["process_tco2_per_unit_product"])
        / safe
    ).fillna(0.0)
    ie["_frac_combustion_ch4"] = (
        (
            (
                ie["combustion_tco2e_per_unit_product"]
                - ie["combustion_tco2_per_unit_product"]
            )
            + (grid_co2e - grid_co2)
        )
        / safe
    ).fillna(0.0)
    ie["_frac_uncaptured_process_co2"] = (
        ie["process_tco2_per_unit_product"] / safe
    ).fillna(0.0)
    ie["_frac_uncaptured_combustion_co2"] = (
        (ie["combustion_tco2_per_unit_product"] + grid_co2) / safe
    ).fillna(0.0)

    # Lookup by (state, product) when both frames carry project_state, else
    # by product alone. Dedup keeps the first industrial_product flavor.
    if "project_state" in ie.columns and "project_state" in scenarios_df.columns:
        key_cols = ["project_state", "simple_product_name"]
    else:
        key_cols = ["simple_product_name"]
    frac_cols = [
        "_frac_process_ch4",
        "_frac_combustion_ch4",
        "_frac_uncaptured_process_co2",
        "_frac_uncaptured_combustion_co2",
    ]
    frac_lookup = ie.drop_duplicates(subset=key_cols, keep="first").set_index(key_cols)[
        frac_cols
    ]

    # Per-project unabated totals, deduped so we don't double-count across
    # destination scenarios (unabated tonnage is destination-independent).
    scen = scenarios_df[
        scenarios_df["methane_scenario"] == methane_scenario
    ].drop_duplicates(subset="uid")
    if industries:
        scen = scen[scen["product"].isin(industries)]

    if "project_state" in key_cols:
        grouped = scen.groupby(["project_state", "product"])[
            "unabated_mtco2e_per_yr"
        ].sum()
    else:
        grouped = scen.groupby("product")["unabated_mtco2e_per_yr"].sum()

    record = {
        "process_ch4": 0.0,
        "combustion_ch4": 0.0,
        "uncaptured_process_co2": 0.0,
        "uncaptured_combustion_co2": 0.0,
        "captured_process_co2": 0.0,
        "captured_combustion_co2": 0.0,
        "ccs_co2": 0.0,
        "ccs_ch4": 0.0,
        "_captured": 0.0,
    }
    for key, unabated_mt in grouped.items():
        if key not in frac_lookup.index:
            continue
        fracs = frac_lookup.loc[key]
        record["process_ch4"] += unabated_mt * fracs["_frac_process_ch4"]
        record["combustion_ch4"] += unabated_mt * fracs["_frac_combustion_ch4"]
        record["uncaptured_process_co2"] += (
            unabated_mt * fracs["_frac_uncaptured_process_co2"]
        )
        record["uncaptured_combustion_co2"] += (
            unabated_mt * fracs["_frac_uncaptured_combustion_co2"]
        )

    record["_net"] = (
        record["process_ch4"]
        + record["combustion_ch4"]
        + record["uncaptured_process_co2"]
        + record["uncaptured_combustion_co2"]
    )
    return record

_BREAKDOWN_COMPONENTS = [
    ("process_ch4", r"Process $\mathbf{CH_4}$", "#e07b39"),
    ("combustion_ch4", r"Combustion $\mathbf{CH_4}$", "#f4a460"),
    ("uncaptured_process_co2", r"Uncaptured process $\mathbf{CO_2}$", "#777777"),
    ("uncaptured_combustion_co2", r"Uncaptured combustion $\mathbf{CO_2}$", "#aaaaaa"),
    ("captured_process_co2", r"Process $\mathbf{CO_2}$ stored with CCS", "#2e7d32"),
    (
        "captured_combustion_co2",
        r"Combustion $\mathbf{CO_2}$ stored with CCS",
        "#66bb6a",
    ),
    ("ccs_ch4", r"CCS-caused $\mathbf{CH_4}$", "#b71c1c"),
    ("ccs_co2", r"CCS-caused $\mathbf{CO_2}$", "#ef5350"),
]

_FACILITY_NAME_DEFAULTS = {
    "ngp": "Natural Gas Processing",
    "dac": "DAC",
    "lng": "LNG",
    "power_ng": "Gas power",
    "power_ng_new_build": "Natural gas power plant",
    "power_ng_retrofit": "Natural gas power plant (retrofit)",
    "power_ng_data_center": "Gas-powered Data Centers",
    "power_coal": "Coal power",
    "refined_fuels": "Refining",
    "petrochemicals": "Petrochemicals",
    "refining_and_petrochemicals": "Refining & Petrochemicals",
    "power_generation_gas_and_coal": "Power generation (gas and coal)",
    "hydrogen": "Hydrogen",
    "ammonia": "Ammonia",
    "ethanol": "Ethanol",
    "methanol": "Methanol",
    "iron_steel": "Iron & steel",
    "dac_with_ngcc": "DAC (NGCC-powered)",
}

def plot_emissions_breakdown_combined(
    industrial_emissions_df: pd.DataFrame,
    ccs_emissions_df: pd.DataFrame,
    methane_scenario: str,
    industries: Optional[List[str]] = None,
    which_flavor: Optional[List[str]] = None,
    name_replacements: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 8),
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    display_mode: Optional[str] = None,
    wrapwidth: int = 14,  # wrap x-tick labels at this many characters
) -> Tuple[plt.Figure, plt.Axes]:
    """Stacked-bar chart of per-tCO2-captured emissions, one bar per industry,
    with a hatched cap showing the extra emissions when the captured CO2 is
    routed to EOR instead of geologic storage.

    Each bar decomposes tCO2e-emitted-per-tCO2-captured into the eight
    components defined in `_BREAKDOWN_COMPONENTS`. Positive components stack
    upward from y=0; the captured-CO2 pieces stack downward (they're the
    physical "injected underground" slice, plotted below the axis). Two
    directional arrows to the left of the first bar label these two regions
    ("Emitted" above zero, "Injected" below).

    On top of every bar, a red-outlined hatched slice shows the *EOR
    increment*: the additional CCS-operations emissions incurred if the
    captured CO2 is sent to EOR rather than geologic storage. Computed as
    `(eor.ccs_co2 + eor.ccs_ch4) - (gs.ccs_co2 + gs.ccs_ch4)` per industry,
    clipped at zero. Two bar-top labels give the "emitted" total for GS
    (below the hatched cap) and for EOR (above the hatched cap); both
    exclude the -1 captured slice so they show emitted-only, not net.

    Bars are ordered by decreasing net emissions under the GS scenario, so
    the largest emitter sits on the left.

    Args:
        industrial_emissions_df: Output of the industrial-emissions pipeline
            (per-industry CO2/CO2e/CH4 columns + capture fractions).
        ccs_emissions_df: Output of the CCS emissions pipeline (per-
            (co2_source, co2_destination) lifecycle factors).
        methane_scenario: Which methane scenario to fix (e.g. "edf20", "howarth").
        industries: Restrict to these `simple_product_name` values.
        which_flavor: Restrict to these `industrial_product` values (useful
            when a single simple_product has multiple flavors and only one
            is wanted).
        name_replacements: Display-only overrides for x-tick labels — does NOT
            merge rows. To collapse multiple industries into one bar, either
            filter to a single representative via `industries=[...]` and
            relabel it here, or pre-aggregate the input dataframes.
        title: Chart title override.
        figsize: Figure size in inches.
        output_path: If given, save the figure to this path at 300 dpi.
        ax: If given, plot onto this existing Axes.
        display_mode: Font preset name (see `_resolve_fonts`).

    Returns:
        (fig, ax) tuple.
    """
    fs = _resolve_fonts(display_mode)

    # Tunables (kept together so they can be adjusted without hunting).
    BAR_WIDTH = 0.55
    Y_HEADROOM = 1.12  # multiplier over tallest bar for ylim top
    Y_FLOOR = -1.5  # captured sums to -1; leave a little margin
    LABEL_X_NUDGE = 0.03  # bar-top label horizontal offset
    ARROW_X_OFFSET = 0.55  # arrows drawn this far left of first bar
    ZERO_LINE_OVERHANG = 0.03  # zero line extends past axis edges

    # ── 1. Build per-destination breakdowns and align them by industry ────
    gs_df = _build_breakdown_data(
        industrial_emissions_df,
        ccs_emissions_df,
        methane_scenario,
        "gs",
        industries,
        which_flavor,
    )
    eor_df = _build_breakdown_data(
        industrial_emissions_df,
        ccs_emissions_df,
        methane_scenario,
        "eor",
        industries,
        which_flavor,
    )

    component_cols = [col for col, _, _ in _BREAKDOWN_COMPONENTS]
    gs_df["_net"] = gs_df[component_cols].sum(axis=1)
    gs_df = gs_df.sort_values("_net", ascending=False)
    industry_order = gs_df["simple_product_name"].tolist()
    eor_df = eor_df.set_index("simple_product_name").loc[industry_order].reset_index()

    # EOR increment: extra CCS-operations emissions when using EOR vs. GS.
    # Clipped at 0 so we never subtract from the base bar.
    eor_increment = (eor_df["ccs_co2"].values + eor_df["ccs_ch4"].values) - (
        gs_df["ccs_co2"].values + gs_df["ccs_ch4"].values
    )
    eor_increment = np.maximum(eor_increment, 0)

    # ── 2. Figure / axes ──────────────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    x = np.arange(len(industry_order))

    # ── 3. Stack the 8 breakdown components ───────────────────────────────
    # Positive contributions stack upward from y=0; negatives (captured CO2)
    # stack downward. Track running tops for each direction.
    pos_bottom = np.zeros(len(x))
    neg_bottom = np.zeros(len(x))
    gs_indexed = gs_df.set_index("simple_product_name")
    base_bar_kwargs = dict(
        width=BAR_WIDTH,
        edgecolor="white",
        linewidth=1.2,
        zorder=2,
    )
    for col, label, color in _BREAKDOWN_COMPONENTS:
        vals = gs_indexed.loc[industry_order, col].values
        pos_vals = np.where(vals > 0, vals, 0)
        neg_vals = np.where(vals < 0, vals, 0)
        if pos_vals.any():
            ax.bar(
                x,
                pos_vals,
                bottom=pos_bottom,
                color=color,
                label=label,
                **base_bar_kwargs,
            )
            pos_bottom += pos_vals
        if neg_vals.any():
            # Don't duplicate a legend entry if this component was already
            # drawn as positive above.
            neg_label = None if pos_vals.any() else label
            ax.bar(
                x,
                neg_vals,
                bottom=neg_bottom,
                color=color,
                label=neg_label,
                **base_bar_kwargs,
            )
            neg_bottom += neg_vals

    # ── 4. EOR-increment cap: red "I" (bottom bar + upward arrow + top bar) ─
    # The bottom and top horizontals mark the GS and EOR ceilings; the arrow
    # between them visually indicates "extra emissions added if using EOR
    # instead of GS". Legend uses _IShapeHandler to render a matching swatch.
    EOR_COLOR = "#b71c1c"
    EOR_BORDER_LW = 1.5
    EOR_ARROW_LW = 2.0
    for j in range(len(x)):
        inc = eor_increment[j]
        if inc <= 0:
            continue
        y_bottom = pos_bottom[j]
        y_top = pos_bottom[j] + inc
        x_left = x[j] - BAR_WIDTH / 2
        x_right = x[j] + BAR_WIDTH / 2
        ax.plot(
            [x_left, x_right],
            [y_bottom, y_bottom],
            color=EOR_COLOR,
            linewidth=EOR_BORDER_LW,
            zorder=2,
            solid_capstyle="butt",
        )
        ax.plot(
            [x_left, x_right],
            [y_top, y_top],
            color=EOR_COLOR,
            linewidth=EOR_BORDER_LW,
            zorder=2,
            solid_capstyle="butt",
        )
        ax.annotate(
            "",
            xy=(x[j], y_top),
            xytext=(x[j], y_bottom),
            arrowprops={
                "arrowstyle": "-|>",
                "color": EOR_COLOR,
                "lw": EOR_ARROW_LW,
                "mutation_scale": 12,
            },
            zorder=2,
        )
    # Proxy artist so this shape gets a labeled entry in the legend, rendered
    # via _IShapeHandler below.
    eor_proxy = Line2D(
        [],
        [],
        color=EOR_COLOR,
        lw=EOR_BORDER_LW,
        label="Additional emissions if CO$_2$ used for EOR",
    )
    ax.add_line(eor_proxy)
    gs_top = pos_bottom.copy()
    eor_top = pos_bottom + eor_increment

    # Annotate the LNG "I" (if present) with plain-English labels next to the
    # two horizontal bars, so a reader can read the concept off the plot
    # without hunting for the legend.
    try:
        lng_j = industry_order.index("lng")
    except ValueError:
        lng_j = None
    if lng_j is not None and eor_increment[lng_j] > 0:
        i_bottom = pos_bottom[lng_j]
        i_top = pos_bottom[lng_j] + eor_increment[lng_j]
        i_right = x[lng_j] + BAR_WIDTH / 2
        I_LABEL_X_NUDGE = 0.08
        ax.text(
            i_right + I_LABEL_X_NUDGE,
            i_top,
            "When CO$_2$ is used for EOR",
            ha="left",
            va="center",
            fontsize=fs["annotation"],
            fontfamily="Arial",
            color="#222222",
            zorder=6,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
        )
        ax.text(
            i_right + I_LABEL_X_NUDGE,
            i_bottom,
            "When CO$_2$ is injected for geologic storage",
            ha="left",
            va="center",
            fontsize=fs["annotation"],
            fontfamily="Arial",
            color="#222222",
            zorder=6,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
        )

    # Zero line — extends slightly past the axes so it wraps the bars.
    ax.axhline(
        0,
        color="black",
        linewidth=1.0,
        zorder=3,
        xmin=-ZERO_LINE_OVERHANG,
        xmax=1 + ZERO_LINE_OVERHANG,
        clip_on=False,
    )

    # ── 5. Bar-top labels (emitted totals for GS and EOR) ─────────────────
    # `_net + 1` removes the -1 from the captured slice, so the label reflects
    # only the emitted portion (positive contributions), not the net.
    # Each numeric label is paired with a smaller "(GS)" or "(EOR)" suffix
    # placed just beneath it so the reader can tell the two totals apart.
    label_x = x - BAR_WIDTH / 2 - LABEL_X_NUDGE
    label_style = dict(
        ha="right",
        va="center",
        fontsize=fs["bar_label"] - 1,
        fontfamily="Arial",
        fontweight="bold",
        zorder=5,
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
    )
    suffix_style = dict(
        ha="right",
        va="top",
        fontsize=fs["bar_label"] - 2,
        fontfamily="Arial",
        fontweight="normal",
        zorder=5,
        xycoords="data",
        textcoords="offset points",
        xytext=(0, -8),
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
    )
    for j in range(len(x)):
        gs_emitted = gs_df.iloc[j]["_net"] + 1
        eor_emitted = gs_emitted + eor_increment[j]
        ax.text(label_x[j], gs_top[j], f"{gs_emitted:.1f}", **label_style)
        ax.text(label_x[j], eor_top[j], f"{eor_emitted:.1f}", **label_style)
        ax.annotate("(GS)", xy=(label_x[j], gs_top[j]), **suffix_style)
        ax.annotate("(EOR)", xy=(label_x[j], eor_top[j]), **suffix_style)

    # ── 6. X-tick labels (with optional display renames + text wrap) ──────
    replacements = {**_FACILITY_NAME_DEFAULTS, **(name_replacements or {})}
    tick_labels = [
        textwrap.fill(
            replacements.get(ind, ind.replace("_", " ").title()),
            width=wrapwidth,
        )
        for ind in industry_order
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=fs["tick"] + 1, fontfamily="Arial")
    ax.set_ylabel(
        "Total facility emissions per tCO$_2$ injected with CCS\n[tCO$_2$e emitted/tCO$_2$ injected]",
        fontsize=fs["axis_label"],
        fontfamily="Arial",
        labelpad=12,
    )

    # ── 7. Y-axis limits, ticks, and zero-tick halo ───────────────────────
    y_max = eor_top.max() * Y_HEADROOM
    ax.set_ylim(Y_FLOOR, y_max)
    yticks = [t for t in ax.get_yticks() if Y_FLOOR <= t <= y_max]
    if -1 not in yticks:
        yticks.append(-1)
    ax.set_yticks(sorted(yticks))
    ax.tick_params(axis="y", labelsize=fs["tick"])
    # Halo the "0" tick so its text stays legible where bars cross it.
    for ytl in ax.get_yticklabels():
        is_zero_tick = ytl.get_text() == "0" or (
            hasattr(ytl, "get_position") and ytl.get_position()[1] == 0
        )
        if is_zero_tick:
            ytl.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
            ytl.set_zorder(4)

    # ── 8. Directional arrows to the left of the first bar ────────────────
    # Down arrow (0 → -1): the "Injected underground" region.
    # Up arrow (0 → +1): the "Emitted to atmosphere" region.
    arrow_x = x[0] - ARROW_X_OFFSET
    for y_end, text_y, va, arrow_label, color in (
        (-1, -1.08, "top", "Injected", "#2e7d32"),
        (1, 1.08, "bottom", "Emitted", "black"),
    ):
        ax.annotate(
            "",
            xy=(arrow_x, y_end),
            xytext=(arrow_x, 0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": 3,
                "mutation_scale": 15,
            },
        )
        ax.text(
            arrow_x,
            text_y,
            arrow_label,
            fontsize=fs["annotation"] + 1,
            fontfamily="Arial",
            weight="semibold",
            color=color,
            ha="left",
            va=va,
        )

    # ── 9. Title, grid, spines ────────────────────────────────────────────
    # Only `title=None` triggers the default; `title=""` is honored as
    # "render no title" so callers can suppress it explicitly.
    if title is None:
        title = r"Greenhouse gas emissions per tCO$_2$ injected with CCS"
    ax.set_title(
        title,
        fontsize=fs["title"],
        fontfamily="Arial",
        pad=20,
    )
    ax.grid(axis="y", color="lightgray", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(bottom=False)

    # ── 10. Legend — dedupe entries introduced by the pos/neg split ───────
    handles, legend_labels = ax.get_legend_handles_labels()
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, legend_labels):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)
    ax.legend(
        unique_handles,
        unique_labels,
        fontsize=fs["legend"] + 3,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        loc="upper right",
        handler_map={eor_proxy: _IShapeHandler(color=EOR_COLOR)},
        handleheight=2.4,  # taller swatch so the "I" reads as I-shaped
    )

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
    return fig, ax

_STATE_DISPLAY_NAMES: Dict[str, str] = {
    "TX": "Texas",
    "LA": "Louisiana",
}

def _format_states_for_title(states: Optional[List[str]]) -> str:
    """Turn a list like ['TX', 'LA'] into 'Texas + Louisiana' for chart titles.

    Falls back to the raw code if unknown, and returns a generic 'multi-state'
    label if states is None."""
    if not states:
        return "multi-state"
    return " + ".join(_STATE_DISPLAY_NAMES.get(s, s) for s in states)

def plot_state_emissions_vs_unabated_and_existing(
    industrial_emissions_df: pd.DataFrame,
    ccs_emissions_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    methane_scenario: str,
    destination_scenarios: List[Union[str, dict]],
    industries: Optional[List[str]] = None,
    which_flavor: Optional[List[str]] = None,
    states: Optional[List[str]] = None,
    title: Optional[str] = None,
    legend_below: bool = True,
    figsize: Tuple[int, int] = (14, 8),
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    display_mode: Optional[str] = None,
    scenario_labels: Optional[Dict[str, str]] = None,
    show_per_tco2_label: bool = True,
    legend_ncol: int = 3,
    annotate_captured_inline: bool = False,
    slimmest_version: bool = False,
    legend_right: bool = False,
    legend_right_axes_fraction: float = 0.72,
    legend_right_valign: str = "top",
    ghost_in_legend: bool = True,
    lifetime: bool = False,
    facility_lifetime_years: int = 30,
    ccs_active_years: int = 12,
) -> Tuple[plt.Figure, plt.Axes]:
    """Stacked-bar chart comparing CCS destination scenarios against an
    unabated reference and (optionally) an "existing facilities" bar.

    Each x-tick is one destination scenario; each stacked bar breaks annual
    emissions into industrial CH₄, industrial CO₂, CCS-caused CH₄,
    CCS-caused CO₂, and (for CCS scenarios) a dashed overlay showing the
    CO₂ captured and stored. The Unabated bar carries a bold
    "Net: X MtCO₂e/yr" label.

    Enable the "existing facilities" bar by including ``"existing"`` in
    `destination_scenarios` (typically as the first entry). An ``"existing"``
    bar sums `unabated_mtco2e_per_yr` across projects whose
    ``retrofit_newbuild`` column is ``"retrofit"``, broken into the same four
    non-CCS components as the Unabated bar. Needs the ``retrofit_newbuild``
    column on `scenarios_df`.

    Args:
        scenario_labels: Optional dict mapping scenario names to custom
            x-tick labels (e.g. ``{"unabated": "Full industrial buildout
            without CCS", "existing": "Already operating"}``). Keys match the
            scenario ``name`` (``"existing"``, ``"unabated"``, ``"all_gs"``,
            ``"baseline"``, ``"current_mix"``, ``"all_eor"``, etc. or any
            custom dict entry's ``"name"``). Unmatched keys are silently
            ignored; scenarios without an override keep their default labels.
            Embed ``\\n`` for explicit line breaks (textwrap still auto-wraps
            each line at width 20). CCS scenarios normally receive an
            ``(X% EOR; Y% GS)`` suffix after rendering; providing a label that
            already contains "EOR" or "GS" suppresses the auto-suffix.
        show_per_tco2_label: If True (default), each CCS bar is annotated with
            both a ``(X.X tCO2e/tCO2 injected)`` parenthetical and a bold
            ``Net: X MtCO2e/yr`` line above it, and the secondary (right-hand)
            y-axis showing tCO2e/tCO2 injected is drawn. If False, the
            parenthetical is dropped, ``Net:`` sits directly above the bar at
            the same offset the Unabated and Existing bars already use, and
            the secondary y-axis is suppressed (the two per-tCO2 conventions
            toggle together).
        legend_ncol: Number of columns in the figure legend. Defaults to 3,
            matching the sibling function. Only used when ``legend_below`` is
            True (the right-side legend is rendered as a single column).
        annotate_captured_inline: If True, remove the "CO₂ captured by CCS"
            entry from the legend and instead draw a green arrow + annotation
            pointing at the green-dashed captured block on the first CCS bar.
            Default False (legend entry present, no inline annotation).
        slimmest_version: If True, collapse the stacked-component palette to
            five entries: "Industrial CH₄" (process + combustion CH₄, orange),
            "Industrial CO₂" (uncaptured process + uncaptured combustion CO₂,
            gray), "CO₂ stored with CCS" (captured process + captured
            combustion CO₂, dark green matching the dashed overlay), plus the
            two CCS-caused components (CH₄ and CO₂, unchanged). Default False.
        legend_right: If True, place the legend to the right of the plot
            instead of below, using `legend_ncol` columns. Takes precedence
            over `legend_below`. Default False.
        legend_right_axes_fraction: Only used when `legend_right=True`.
            Fraction of the figure width given to the axes (the remainder is
            reserved for the right-side legend). Default 0.72. Raise toward
            0.85 if the legend is narrow (few labels, 1 column) and the plot
            looks squished; lower toward 0.6 if the legend has many columns
            or long labels and is being clipped. Scaling the full figure via
            `figsize` scales the plot and the legend reserve together.
        legend_right_valign: Only used when `legend_right=True`. Vertical
            alignment of the right-side legend relative to the plot area.
            One of "top" (default — legend anchored to the plot's upper
            edge), "center" (vertically centered along the plot's right
            edge), or "bottom" (anchored to the plot's lower edge).
        facility_lifetime_years, ccs_active_years: Must match
            ``scenario_config.facility_lifetime_years`` and
            ``scenario_config.project_duration`` used to build ``scenarios_df``,
            otherwise on-chart lifetime totals will not equal the pipeline's
            ``lifetime_mtco2e`` column.
    """
    display_mode = _resolve_display_mode(display_mode)
    fs = _resolve_fonts(display_mode)

    # `ghost_in_legend=False` has the same effect as `annotate_captured_inline=True`
    # — ghost drops out of the legend and gains an inline arrow annotation.
    # Treat the two as a single trigger so either flag works.
    show_ghost_inline = annotate_captured_inline or (not ghost_in_legend)

    if states is not None:
        if "project_state" in scenarios_df.columns:
            scenarios_df = scenarios_df[scenarios_df["project_state"].isin(states)]
        if "project_state" in industrial_emissions_df.columns:
            industrial_emissions_df = industrial_emissions_df[
                industrial_emissions_df["project_state"].isin(states)
            ]
        if "project_state" in ccs_emissions_df.columns:
            ccs_emissions_df = ccs_emissions_df[
                ccs_emissions_df["project_state"].isin(states)
            ]

    known_dest_scenarios = {
        "existing": {
            "name": "existing",
            "label": "Existing facilities\n(no industrial buildout)",
            "is_existing": True,
        },
        "all_gs": {
            "name": "all_gs",
            "label": "With CCS, assuming\n100% Geologic storage",
            "eor_pct": 0,
        },
        "baseline": {
            "name": "baseline",
            "label": "With CCS, stated\ndestination",
            "from_data": True,
        },
        "current_mix": {
            "name": "current_mix",
            "label": "With CCS, current\ndestination",
            "eor_pct": 92,
        },
        "nameplate_implied": {
            "name": "nameplate_implied",
            "label": "Nameplate 'capture capacity'",
            "zero_lifecycle": True,
        },
        "all_eor": {
            "name": "all_eor",
            "label": "With CCS, assuming\n100% EOR",
            "eor_pct": 100,
        },
        "switching": {
            "name": "switching",
            "label": "With CCS, announced\ndestinations, then\nall EOR after 45Q",
            "from_data": True,
            "is_switching": True,
        },
        "switching_from_current": {
            "name": "switching_from_current",
            "label": "With CCS, current U.S.\npractice, then\nall EOR after 45Q",
            "from_data": True,
            "is_switching": True,
        },
        "unabated": {
            "name": "unabated",
            "label": "Unabated after full buildout (no CCS)",
            "is_unabated": True,
        },
    }

    resolved = []
    for s in destination_scenarios:
        if isinstance(s, str):
            if s not in known_dest_scenarios:
                raise ValueError(
                    f"Unknown destination scenario '{s}'. "
                    f"Known: {list(known_dest_scenarios)}"
                )
            resolved.append(known_dest_scenarios[s])
        else:
            resolved.append(s)
    destination_scenarios = resolved

    # Apply user-provided label overrides (copy per-scenario dicts so we don't
    # mutate the shared `known_dest_scenarios` entries). Unmatched keys are
    # silently ignored.
    if scenario_labels:
        destination_scenarios = [
            (
                {**scen, "label": scenario_labels[scen["name"]]}
                if scen.get("name") in scenario_labels
                else scen
            )
            for scen in destination_scenarios
        ]

    component_cols = [col for col, _, _ in _BREAKDOWN_COMPONENTS]

    ccs_scenarios = [
        s
        for s in destination_scenarios
        if not s.get("is_unabated") and not s.get("is_existing")
    ]

    if ccs_scenarios:
        if "project_state" in scenarios_df.columns:
            active_states = sorted(scenarios_df["project_state"].dropna().unique())
        else:
            active_states = [None]
        per_state_records = []
        per_state_dac = []
        for state in active_states:
            if state is None:
                ie_s, ccs_s, scen_s = (
                    industrial_emissions_df,
                    ccs_emissions_df,
                    scenarios_df,
                )
            else:
                ie_s = (
                    industrial_emissions_df[
                        industrial_emissions_df["project_state"] == state
                    ]
                    if "project_state" in industrial_emissions_df.columns
                    else industrial_emissions_df
                )
                ccs_s = (
                    ccs_emissions_df[ccs_emissions_df["project_state"] == state]
                    if "project_state" in ccs_emissions_df.columns
                    else ccs_emissions_df
                )
                scen_s = scenarios_df[scenarios_df["project_state"] == state]
            per_state_records.append(
                _compute_dual_axis_records(
                    ie_s,
                    ccs_s,
                    scen_s,
                    methane_scenario,
                    ccs_scenarios,
                    industries,
                    which_flavor,
                    component_cols,
                )
            )
            per_state_dac.append(
                _compute_dac_records(
                    ccs_s,
                    scen_s,
                    methane_scenario,
                    ccs_scenarios,
                )
            )
    else:
        per_state_records = []
        per_state_dac = []

    unabated_record = None
    if any(s.get("is_unabated") for s in destination_scenarios):
        unabated_record = _compute_unabated_record(
            industrial_emissions_df,
            scenarios_df,
            methane_scenario,
            industries=industries,
            which_flavor=which_flavor,
        )

    # "Existing facilities" bar = unabated breakdown restricted to retrofit
    # projects. Reuses _compute_unabated_record by pre-filtering scenarios_df
    # to retrofit uids first (so the per-project unabated_mtco2e_per_yr sum
    # only picks up retrofits). Requires the retrofit_newbuild column.
    existing_record = None
    if any(s.get("is_existing") for s in destination_scenarios):
        if "retrofit_newbuild" not in scenarios_df.columns:
            raise ValueError(
                "'existing' scenario requested but scenarios_df is missing the "
                "'retrofit_newbuild' column needed to subset retrofit projects."
            )
        retro_mask = (
            scenarios_df["retrofit_newbuild"].astype(str).str.strip().str.lower()
            == "retrofit"
        )
        retro_scen_df = scenarios_df[retro_mask]
        if retro_scen_df.empty:
            existing_record = {
                col: 0.0 for col in component_cols + ["_captured", "_net"]
            }
        else:
            existing_record = _compute_unabated_record(
                industrial_emissions_df,
                retro_scen_df,
                methane_scenario,
                industries=industries,
                which_flavor=which_flavor,
            )

    # Interleave CCS + unabated + existing records back into the caller's order.
    records = []
    ccs_i = 0
    for scen in destination_scenarios:
        row = {
            "scenario": scen["name"],
            "label": scen["label"],
            "_is_unabated": bool(scen.get("is_unabated")),
            "_is_existing": bool(scen.get("is_existing")),
            "_is_switching": bool(scen.get("is_switching")),
        }
        if scen.get("is_unabated"):
            for col in component_cols + ["_captured"]:
                row[col] = unabated_record.get(col, 0.0)
            row["_net"] = sum(row[c] for c in component_cols)
        elif scen.get("is_existing"):
            for col in component_cols + ["_captured"]:
                row[col] = existing_record.get(col, 0.0)
            row["_net"] = sum(row[c] for c in component_cols)
        else:
            for col in component_cols + ["_captured"]:
                row[col] = sum(r[ccs_i][col] for r in per_state_records)
                row[col] += sum(d[ccs_i].get(col, 0.0) for d in per_state_dac)
            row["_net"] = sum(row[c] for c in component_cols)
            ccs_i += 1
        records.append(row)

    result_df = pd.DataFrame(records)

    # ──── Lifetime scaling ─────────────────────────────────────────────────
    # When `lifetime=True`, convert every annual rate into a 30-yr lifetime
    # total. Mapping (per the paper's convention of a 12-yr CCS window + 18-yr
    # unabated tail inside a 30-yr facility life):
    #   Unabated / Existing rows:    all components × facility_lifetime_years
    #   CCS rows:
    #     process_ch4, combustion_ch4         × facility_lifetime_years
    #                                           (facility methane is time-
    #                                           invariant and already matches
    #                                           the unabated-era value)
    #     uncaptured_process_co2,              ccs_active × CCS-era value
    #     uncaptured_combustion_co2          + (facility_lifetime - ccs_active)
    #                                            × unabated-era value
    #     captured_process_co2,                ccs_active × CCS-era value
    #     captured_combustion_co2,             (zero in the unabated tail)
    #     ccs_ch4, ccs_co2
    #     _captured                             ccs_active × annual captured
    # _net is recomputed from the scaled components.
    if lifetime:
        unabated_tail_years = facility_lifetime_years - ccs_active_years
        if unabated_tail_years < 0:
            raise ValueError(
                f"facility_lifetime_years={facility_lifetime_years} must be "
                f">= ccs_active_years={ccs_active_years}."
            )
        # Need the unabated breakdown as the tail-period baseline for CCS
        # rows. Compute it even if no explicit Unabated bar was requested.
        tail_unab = unabated_record
        if tail_unab is None and (
            (result_df["_is_unabated"] | result_df["_is_existing"]).any() is False
            or True  # unconditional for CCS-only lifetime cases
        ):
            tail_unab = _compute_unabated_record(
                industrial_emissions_df,
                scenarios_df,
                methane_scenario,
                industries=industries,
                which_flavor=which_flavor,
            )

        time_invariant = {"process_ch4", "combustion_ch4"}
        ccs_only = {
            "captured_process_co2",
            "captured_combustion_co2",
            "ccs_ch4",
            "ccs_co2",
        }
        blended = {"uncaptured_process_co2", "uncaptured_combustion_co2"}

        for i, row in result_df.iterrows():
            if row["_is_unabated"] or row["_is_existing"]:
                for col in component_cols + ["_captured"]:
                    result_df.at[i, col] = row[col] * facility_lifetime_years
            elif row.get("_is_switching", False):
                # Switching scenarios keep CCS running for the full facility
                # lifetime (destination just pivots to 100% EOR after the 45Q
                # window). Per-year rates are already time-weighted means over
                # L years, so every component — including CCS-only and
                # blended ones — scales by facility_lifetime_years.
                for col in component_cols + ["_captured"]:
                    result_df.at[i, col] = row[col] * facility_lifetime_years
            else:
                for col in component_cols:
                    annual = row[col]
                    if col in time_invariant:
                        result_df.at[i, col] = annual * facility_lifetime_years
                    elif col in ccs_only:
                        result_df.at[i, col] = annual * ccs_active_years
                    elif col in blended:
                        tail = tail_unab.get(col, 0.0) if tail_unab else 0.0
                        result_df.at[i, col] = (
                            annual * ccs_active_years + tail * unabated_tail_years
                        )
                    else:
                        result_df.at[i, col] = annual * facility_lifetime_years
                result_df.at[i, "_captured"] = row["_captured"] * ccs_active_years
            result_df.at[i, "_net"] = sum(result_df.at[i, c] for c in component_cols)

    if ccs_scenarios:
        _truth_scen = scenarios_df[scenarios_df["methane_scenario"] == methane_scenario]
        if states is not None and "project_state" in _truth_scen.columns:
            _truth_scen = _truth_scen[_truth_scen["project_state"].isin(states)]
        per_scen_cap = [
            _truth_scen[_truth_scen["scenario"] == s["name"]][
                "mtco2_per_yr_captured_with_ccs"
            ].sum()
            for s in ccs_scenarios
        ]
        total_captured = sum(per_scen_cap) / max(len(per_scen_cap), 1)
    else:
        total_captured = 0.0
    # Lifetime mode: the right-hand "tCO2e per tCO2 injected" axis stays
    # dimensionally consistent only if the capture denominator is also a
    # lifetime total (12-yr window) matching the y-axis's lifetime units.
    if lifetime:
        total_captured = total_captured * ccs_active_years

    # Auto-append "(X% EOR; Y% GS)" to CCS bar labels — but only when the
    # caller hasn't provided their own labels. If `scenario_labels` is passed,
    # the caller is fully controlling the x-tick labels and expects their
    # strings verbatim across every bar, so we skip the suffix pass entirely.
    if not scenario_labels:
        for i, row in result_df.iterrows():
            if row["_is_unabated"] or row["_is_existing"]:
                continue
            if "EOR" in row["label"] or "GS" in row["label"]:
                continue
            scen_info = next(
                (s for s in destination_scenarios if s["name"] == row["scenario"]),
                None,
            )
            if scen_info is None:
                continue
            if scen_info.get("from_data"):
                scen_df = scenarios_df[
                    (scenarios_df["methane_scenario"] == methane_scenario)
                    & (scenarios_df["scenario"] == row["scenario"])
                ]
                total_cap = scen_df["mtco2_per_yr_captured_with_ccs"].sum()
                if total_cap > 0:
                    w_eor = (
                        scen_df["eor_fraction"]
                        * scen_df["mtco2_per_yr_captured_with_ccs"]
                    ).sum() / total_cap
                    eor_pct = round(w_eor * 100)
                    result_df.at[
                        i, "label"
                    ] = f"{row['label']} ({eor_pct}% EOR; {100 - eor_pct}% GS)"
            elif "eor_pct" in scen_info:
                eor_pct = scen_info["eor_pct"]
                result_df.at[
                    i, "label"
                ] = f"{row['label']} ({eor_pct}% EOR; {100 - eor_pct}% GS)"

    x = np.arange(len(result_df))
    width = 0.45

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    pos_bottom = np.zeros(len(x))
    neg_bottom = np.zeros(len(x))

    # Slimmest-version lumping: fold the industrial-side CH4 sub-components
    # into one orange entry, the uncaptured industrial-side CO2 sub-components
    # into one gray entry, and the two captured-and-stored CO2 sub-components
    # into one dark-green "CO2 stored with CCS" entry (matching the color of
    # the dashed captured overlay). CCS-caused CH4 and CO2 stay separate.
    if slimmest_version:
        result_df["_industrial_ch4"] = (
            result_df["process_ch4"] + result_df["combustion_ch4"]
        )
        result_df["_industrial_co2"] = (
            result_df["uncaptured_process_co2"] + result_df["uncaptured_combustion_co2"]
        )
        result_df["_stored_co2"] = (
            result_df["captured_process_co2"] + result_df["captured_combustion_co2"]
        )
        breakdown_components = [
            ("_industrial_ch4", r"Uncaptured $\mathbf{CH_4}$", "#e07b39"),
            ("_industrial_co2", r"Uncaptured $\mathbf{CO_2}$", "#888888"),
            ("_stored_co2", r"$\mathbf{CO_2}$ stored with CCS", "#2e7d32"),
            ("ccs_ch4", r"CCS-caused $\mathbf{CH_4}$", "#b71c1c"),
            ("ccs_co2", r"CCS-caused $\mathbf{CO_2}$", "#ef5350"),
        ]
    else:
        breakdown_components = _BREAKDOWN_COMPONENTS

    for col, label, color in breakdown_components:
        vals = result_df[col].values
        pos_vals = np.where(vals > 0, vals, 0)
        neg_vals = np.where(vals < 0, vals, 0)
        if pos_vals.any():
            ax.bar(
                x,
                pos_vals,
                width=width,
                bottom=pos_bottom,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                label=label,
                zorder=2,
            )
            pos_bottom += pos_vals
        if neg_vals.any():
            ax.bar(
                x,
                neg_vals,
                width=width,
                bottom=neg_bottom,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                label=label if not pos_vals.any() else None,
                zorder=2,
            )
            neg_bottom += neg_vals

    # Green dashed "captured" overlay only on CCS bars (not unabated or existing).
    captured_vals = result_df["_captured"].values.copy()
    non_ccs_mask = result_df["_is_unabated"].values | result_df["_is_existing"].values
    captured_vals[non_ccs_mask] = 0.0
    ccs_mask = ~non_ccs_mask
    _captured_green = "#2e7d32"
    if ccs_mask.any():
        ax.bar(
            x[ccs_mask],
            captured_vals[ccs_mask],
            width=width,
            bottom=pos_bottom[ccs_mask],
            facecolor="none",
            edgecolor=_captured_green,
            linewidth=1.5,
            linestyle="--",
            zorder=2,
            label=(None if show_ghost_inline else r"$\mathbf{CO_2}$ captured by CCS"),
        )
    ghost_top = pos_bottom + captured_vals

    ax.axhline(
        0,
        color="black",
        linewidth=1.0,
        zorder=3,
        xmin=-0.03,
        xmax=1.03,
        clip_on=False,
    )

    LINE_OVERHANG = 0.04
    for j in range(len(x)):
        ax.plot(
            [x[j] - width / 2 - LINE_OVERHANG, x[j] + width / 2 + LINE_OVERHANG],
            [pos_bottom[j], pos_bottom[j]],
            color="black",
            linewidth=1.75,
            zorder=4,
            solid_capstyle="butt",
        )

    LABEL_GAP = 3
    LABEL_LINE_HEIGHT = 28
    # In slimmest-version mode the bar annotations drop the "Net: " prefix
    # and collapse "MtCO₂e" to a bare "Mt" unit — the slim palette already
    # makes the species and the stacked total unambiguous.
    _net_prefix = "" if slimmest_version else "Net: "
    _mt_token = r" Mt" if slimmest_version else r" MtCO$_2$e"
    # Presentation mode: bump the top-of-bar net-emissions labels so they
    # read from the back of a room.
    _net_label_fs = fs["bar_label"] + (6 if display_mode == "presentation" else 0)
    for j in range(len(x)):
        scen_name = result_df.iloc[j]["scenario"]
        bar_top = pos_bottom[j]
        if result_df.iloc[j]["_is_unabated"] or result_df.iloc[j]["_is_existing"]:
            net_mt = result_df.iloc[j]["_net"]
            _net_unit = (
                _mt_token + "\n" + rf"over {facility_lifetime_years}yr"
                if lifetime
                else _mt_token + "/yr"
            )
            ax.text(
                x[j],
                bar_top + LABEL_GAP,
                f"{_net_prefix}{net_mt:.0f}" + _net_unit,
                ha="center",
                va="bottom",
                fontsize=_net_label_fs,
                fontfamily="Arial",
                fontweight="bold",
                zorder=5,
                path_effects=[pe.withStroke(linewidth=5, foreground="white")],
            )
            continue
        if lifetime:
            # In lifetime mode the Net label and the per-tCO2 ratio must use
            # the already-scaled result_df values so both numerator and
            # denominator are 30-yr/12-yr lifetime totals, matching the
            # y-axis units. Reading scenarios_df here would downgrade them
            # back to annual rates and silently mis-label the bar.
            #
            # `result_df["_net"]` for CCS rows sums all 8 breakdown components
            # — two of which (`captured_process_co2`, `captured_combustion_co2`)
            # are entered as *negative* values. So _net = emitted − captured,
            # a sink-credited quantity. Adding `_captured` back (also already
            # lifetime-scaled) recovers the real "atmospheric emissions" total,
            # keeping "Net:" consistent with the unabated bar and with the
            # annual-mode pipeline column `mtco2e_per_yr_emissions_including_ccs`.
            net_mt = result_df.iloc[j]["_net"] + result_df.iloc[j]["_captured"]
            captured_j = result_df.iloc[j]["_captured"]
        else:
            scen_rows = scenarios_df[
                (scenarios_df["methane_scenario"] == methane_scenario)
                & (scenarios_df["scenario"] == scen_name)
            ]
            net_mt = scen_rows["mtco2e_per_yr_emissions_including_ccs"].sum()
            captured_j = scen_rows["mtco2_per_yr_captured_with_ccs"].sum()
        per_tco2 = net_mt / captured_j if captured_j else 0
        if show_per_tco2_label:
            ax.text(
                x[j],
                bar_top + LABEL_GAP,
                f"({per_tco2:.1f}" + r" tCO$_2$e/tCO$_2$ injected)",
                ha="center",
                va="bottom",
                fontsize=_net_label_fs - 1,
                fontfamily="Arial Narrow",
                fontweight="normal",
                zorder=5,
                path_effects=[pe.withStroke(linewidth=5, foreground="white")],
            )
            net_y = bar_top + LABEL_GAP + LABEL_LINE_HEIGHT
        else:
            net_y = bar_top + LABEL_GAP
        _net_unit_ccs = (
            _mt_token + "\n" + rf"over {facility_lifetime_years}yr"
            if lifetime
            else _mt_token + "/yr"
        )
        ax.text(
            x[j],
            net_y,
            f"{_net_prefix}{net_mt:.0f}" + _net_unit_ccs,
            ha="center",
            va="bottom",
            fontsize=_net_label_fs,
            fontfamily="Arial",
            fontweight="bold",
            zorder=5,
            path_effects=[pe.withStroke(linewidth=5, foreground="white")],
        )

    # Inline arrow+annotation pointing at the green-dashed captured block of
    # the first CCS bar. Only drawn when the caller has turned off the legend
    # entry via `annotate_captured_inline=True` or `ghost_in_legend=False`.
    # The arrow tip sits just on top of the box (at the horizontal midpoint of
    # the top edge); the text sits above-and-to-the-left of the bar so the
    # arrow comes down from the text onto the top of the box.
    if show_ghost_inline:
        _ccs_idx = np.where(ccs_mask)[0]
        if len(_ccs_idx) > 0:
            j_tgt = int(_ccs_idx[0])
            top_of_box_y = pos_bottom[j_tgt] + captured_vals[j_tgt]
            arrow_tip = (x[j_tgt], top_of_box_y)
            text_x = x[j_tgt] - 0.25
            text_y = top_of_box_y + 65
            ax.annotate(
                r"CO$_2$ captured by CCS",
                xy=arrow_tip,
                xytext=(text_x, text_y),
                fontsize=fs["annotation"],
                fontfamily="Arial",
                fontweight="bold",
                color=_captured_green,
                ha="left",
                va="center",
                arrowprops=dict(
                    arrowstyle="->",
                    color=_captured_green,
                    linewidth=1.5,
                    shrinkA=2,
                    shrinkB=0,
                ),
                zorder=5,
                path_effects=[pe.withStroke(linewidth=4, foreground="white")],
            )

    ax.set_xticks(x)
    # Preserve mathtext labels verbatim — textwrap.fill will shred a
    # "$\mathbf{...}$" span across lines and break the delimiters, which
    # makes matplotlib print the raw TeX instead of rendering it. Caller is
    # responsible for inserting their own "\n" if they want line breaks.
    wrapped_labels = [
        lbl
        if "$" in lbl
        else "\n".join(textwrap.fill(line, width=20) for line in lbl.split("\n"))
        for lbl in result_df["label"].tolist()
    ]
    if (slimmest_version == True) & (display_mode == "presentation"):
        ticksize = fs["tick"] + 2
        labelpad = 1.4
        legendfontsize = fs["legend"] + 6
        bbox_y = -0.25
    elif display_mode == "presentation":
        ticksize = fs["tick"]
        labelpad = 1
        legendfontsize = fs["legend"] + 6
        bbox_y = -0.2
    else:
        ticksize = fs["tick"]
        labelpad = 1
        legendfontsize = fs["legend"]
        bbox_y = -0.2
    ax.set_xticklabels(wrapped_labels, fontsize=ticksize, fontfamily="Arial")
    _ylab_species = "Mt" if slimmest_version else r"MtCO$_2$e"
    ax.set_ylabel(
        rf"{_ylab_species} over {facility_lifetime_years}-yr lifetime"
        if lifetime
        else rf"{_ylab_species} / yr",
        fontsize=fs["axis_label"],
        fontfamily="Arial",
        labelpad=labelpad,
    )
    y_min = neg_bottom.min()
    y_max = ghost_top.max() * 1.10
    ax.set_ylim(y_min - 1 if y_min < 0 else -1, y_max)
    ax.tick_params(axis="y", labelsize=ticksize)

    # Force a visible y-tick below zero on the primary (left) axis. Matplotlib
    # with a mostly-positive ylim will usually land its lowest tick at 0; we
    # want one additional labeled tick below the zero line to anchor the
    # downward extent of the stacked bars and the "Net" baseline.
    _current_yticks = np.asarray(ax.get_yticks())
    _pos_ticks = _current_yticks[_current_yticks > 0]
    if len(_pos_ticks) >= 1 and (_current_yticks < 0).sum() == 0:
        if len(_pos_ticks) >= 2:
            tick_spacing = float(_pos_ticks[1] - _pos_ticks[0])
        else:
            tick_spacing = float(_pos_ticks[0])
        negative_tick = -tick_spacing
        new_ticks = np.concatenate(([negative_tick], _current_yticks))
        ax.set_yticks(new_ticks)
        lo, hi = ax.get_ylim()
        if lo > negative_tick - tick_spacing * 0.05:
            ax.set_ylim(negative_tick - tick_spacing * 0.05, hi)

    # Secondary (right) y-axis — "tCO2e emitted per tCO2 injected". Only drawn
    # when the inline per-tCO2 bar labels are also on; the two conventions go
    # together, so a plot without the parenthetical shouldn't carry a lone
    # right-hand axis showing the same ratio.
    if total_captured > 0 and show_per_tco2_label:
        ax2 = ax.secondary_yaxis(
            "right",
            functions=(
                lambda y: y / total_captured,
                lambda y: y * total_captured,
            ),
        )
        ax2.set_ylabel(
            r"tCO$_2$e emitted per tCO$_2$ injected",
            fontsize=fs["axis_label"],
            fontfamily="Arial",
            labelpad=2,
        )
        ax2.tick_params(axis="y", labelsize=fs["tick"])

    if title is None:
        if states is not None:
            title_states = states
        elif "project_state" in scenarios_df.columns:
            title_states = sorted(
                scenarios_df["project_state"].dropna().unique().tolist()
            )
        else:
            title_states = None
        state_label = _format_states_for_title(title_states)
        title = f"{state_label} CCS: state-total emissions vs. unabated (MtCO$_2$e/yr)"
    ax.set_title(title, fontsize=fs["title"], fontfamily="Arial", pad=20)

    ax.grid(axis="y", color="lightgray", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False)
    # When the secondary (right) axis is suppressed, hide the primary's right
    # spine too so there isn't a lone unlabeled vertical border on the right.
    if not (total_captured > 0 and show_per_tco2_label):
        ax.spines["right"].set_visible(False)

    handles, labels_leg = ax.get_legend_handles_labels()
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels_leg):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)

    # In slimmest-version mode the legend keeps the draw/stack order out of
    # it and follows a reader-first ordering: uncaptured species first
    # (CO₂ then CH₄), then CCS-caused species (CO₂ then CH₄), then the two
    # stored/captured entries. Labels not in this map keep their original
    # relative position and are appended at the end.
    if slimmest_version:
        _slimmest_order = [
            r"Uncaptured $\mathbf{CO_2}$",
            r"Uncaptured $\mathbf{CH_4}$",
            r"CCS-caused $\mathbf{CO_2}$",
            r"CCS-caused $\mathbf{CH_4}$",
            r"$\mathbf{CO_2}$ stored with CCS",
            r"$\mathbf{CO_2}$ captured by CCS",
        ]
        _by_label = dict(zip(unique_labels, unique_handles))
        ordered_labels = [l for l in _slimmest_order if l in _by_label]
        ordered_labels += [l for l in unique_labels if l not in _slimmest_order]
        unique_labels = ordered_labels
        unique_handles = [_by_label[l] for l in ordered_labels]

    if legend_right:
        # Right-of-plot legend with caller-controlled column count. Instead
        # of locking tight_layout to a fixed rect (which ignores figsize and
        # produces a squished plot when the caller resizes), use
        # subplots_adjust with a tunable `legend_right_axes_fraction` so the
        # caller can rebalance plot vs. legend width without changing
        # figsize.
        _valign_map = {
            "top": ("upper left", (1.02, 1.0)),
            "center": ("center left", (1.02, 0.5)),
            "bottom": ("lower left", (1.02, 0.0)),
        }
        if legend_right_valign not in _valign_map:
            raise ValueError(
                f"legend_right_valign={legend_right_valign!r} is not valid; "
                f"use one of {list(_valign_map)}."
            )
        _loc, _anchor = _valign_map[legend_right_valign]
        ax.legend(
            unique_handles,
            unique_labels,
            fontsize=legendfontsize,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            loc=_loc,
            bbox_to_anchor=_anchor,
            ncol=legend_ncol,
        )
        fig.subplots_adjust(right=legend_right_axes_fraction)
    elif legend_below:
        ax.legend(
            unique_handles,
            unique_labels,
            fontsize=legendfontsize,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            loc="upper center",
            bbox_to_anchor=(0.5, bbox_y),
            ncol=legend_ncol,
        )
        fig.tight_layout()
    else:
        ax.legend(
            unique_handles,
            unique_labels,
            fontsize=fs["legend"],
            frameon=True,
            facecolor="white",
            edgecolor="none",
            loc="upper left",
            bbox_to_anchor=(1.15, 1),
        )
        fig.tight_layout(rect=[0, 0, 0.68, 1])
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)

    return fig, ax

def _hex_to_rgba(hex_color: str, alpha: float = 0.4) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"

_SANKEY_NODE_COLORS = {
    "total": "#888888",
    "process": "#2d6a2e",
    "process_co2": "#4caf50",
    "process_ch4": "#a5d6a7",
    "combustion": "#bf5b04",
    "combustion_co2": "#fb8c00",
    "combustion_ch4": "#ffcc80",
    "captured": "#1565c0",
    "uncaptured": "#c62828",
    "implementing_ccs": "#8b0000",
    "ccs_co2": "#ef5350",
    "ccs_ch4": "#b71c1c",
    "eor_oil_co2": "#4e342e",
}

def plot_industry_sankey(
    industrial_emissions_df: pd.DataFrame,
    industry: str,
    methane_scenario: str,
    which_flavor: Optional[str] = None,
    minimal_labels: bool = False,
    show_final_values: bool = False,
    verbose: bool = False,
    title: Optional[str] = None,
    width: int = 1400,
    height: int = 620,
    output_path: Optional[str] = None,
    display_mode: Optional[str] = None,
    include_ccs_emissions: bool = False,
    ccs_emissions_df: Optional[pd.DataFrame] = None,
    co2_destination: str = "gs",
):
    """Sankey diagram showing emission sources and destinations for one industry.

    Args:
        industrial_emissions_df: Output of industrial emissions pipeline.
        industry: ``simple_product_name`` value (e.g. 'hydrogen').
        methane_scenario: Which methane scenario row to use.
        which_flavor: Optional ``industrial_product`` value to select
            when multiple flavors map to the same simple_product_name.
        minimal_labels: If True, use short node labels (e.g. 'CO₂', 'CH₄')
            without values. If False (default), show full labels with values.
        title: Chart title override.
        width: Figure width in pixels.
        height: Figure height in pixels.
        output_path: If given, save as image (requires kaleido).
        include_ccs_emissions: If True, add a third source branch
            ("Implementing CCS") into "Total emissions" showing the additional
            emissions caused by running the capture train. Splits into "CCS
            CO₂" and "CCS CH₄" in the next column, both of which flow to
            Uncaptured (CCS-caused emissions are themselves never captured).
            Default False.
        ccs_emissions_df: Required when ``include_ccs_emissions=True``. Output
            of the CCS emissions pipeline, filtered to the relevant
            ``co2_destination``. Uses ``tco2_emitted_per_tco2_ccs`` and
            ``tco2e_emitted_per_tco2_ccs`` factors.
        co2_destination: Either ``"gs"`` (geologic storage, default — the
            cleanest CCS case) or ``"eor"``. Only used when
            ``include_ccs_emissions=True``. When ``"eor"``, a separate
            standalone "EOR CO₂" source node is drawn in the 3rd column,
            labeled "CO₂" with a "combustion of EOR oil" annotation; its
            value is the EOR oil-combustion component (the residual of
            tco2_emitted_per_tco2_ccs after capture + transport + storage),
            and it flows directly to Uncaptured.

    Returns:
        Plotly Figure.
    """
    import plotly.graph_objects as go

    fs = _resolve_fonts(display_mode)
    ie = industrial_emissions_df[
        (industrial_emissions_df["methane_scenario"] == methane_scenario)
        & (industrial_emissions_df["simple_product_name"] == industry)
    ]
    if which_flavor and "industrial_product" in ie.columns:
        ie = ie[ie["industrial_product"] == which_flavor]
    row = ie.iloc[0]

    proc_co2 = row["process_tco2_per_unit_product"]
    comb_co2 = row["combustion_tco2_per_unit_product"]
    proc_tco2e = row["process_tco2e_per_unit_product"]
    comb_tco2e = row["combustion_tco2e_per_unit_product"]
    proc_ch4 = proc_tco2e - proc_co2
    comb_ch4 = comb_tco2e - comb_co2
    proc_cf = row["process_capture_fraction_tco2"]
    comb_cf = row["combustion_capture_fraction_tco2"]
    captured_proc = proc_co2 * proc_cf
    captured_comb = comb_co2 * comb_cf
    uncaptured_proc_co2 = proc_co2 - captured_proc
    uncaptured_comb_co2 = comb_co2 - captured_comb
    industrial_total = proc_tco2e + comb_tco2e
    uncaptured_proc_total = uncaptured_proc_co2 + proc_ch4
    uncaptured_comb_total = uncaptured_comb_co2 + comb_ch4

    eps = 1e-9
    nc = _SANKEY_NODE_COLORS

    captured_total = captured_proc + captured_comb

    # CCS-caused emissions (optional third source branch). Factors in
    # ccs_emissions_df are per tCO2 captured; convert to per-unit-product by
    # multiplying by captured_total.
    ccs_co2_per_unit = 0.0
    ccs_ch4_per_unit = 0.0
    if include_ccs_emissions:
        if ccs_emissions_df is None:
            raise ValueError(
                "include_ccs_emissions=True requires ccs_emissions_df to be "
                "provided."
            )
        ef = ccs_emissions_df[
            (ccs_emissions_df["methane_scenario"] == methane_scenario)
            & (ccs_emissions_df["co2_destination"] == co2_destination)
            & (ccs_emissions_df["co2_source"] == industry)
        ]
        if ef.empty:
            raise ValueError(
                f"No ccs_emissions_df row for industry={industry!r}, "
                f"methane_scenario={methane_scenario!r}, "
                f"co2_destination={co2_destination!r}."
            )
        ef_row = ef.iloc[0]
        ccs_co2_per_tco2 = float(ef_row["tco2_emitted_per_tco2_ccs"])
        ccs_co2e_per_tco2 = float(ef_row["tco2e_emitted_per_tco2_ccs"])
        ccs_co2_per_unit = ccs_co2_per_tco2 * captured_total
        ccs_ch4_per_unit = (ccs_co2e_per_tco2 - ccs_co2_per_tco2) * captured_total

    # EOR-oil CO2 (combustion of oil recovered via EOR). The factor is the
    # residual of tco2_emitted_per_tco2_ccs after subtracting capture +
    # transport + stored components (zero for GS by construction). Split out
    # into its own source node in column 3 so viewers see the oil tail
    # distinctly from the capture/transport/storage tail.
    oil_co2_per_unit = 0.0
    if include_ccs_emissions and co2_destination == "eor":
        try:
            cap_tco2 = float(ef_row["tco2_emitted_per_tco2_captured"])
            tr_tco2 = float(ef_row["tco2_emitted_per_tco2_transported"])
            st_tco2 = float(ef_row["tco2_emitted_per_tco2_stored"])
        except KeyError as exc:
            raise ValueError(
                "co2_destination='eor' requires ccs_emissions_df to include "
                "tco2_emitted_per_tco2_captured, _transported, and _stored "
                "columns so the oil component can be split out."
            ) from exc
        oil_tco2 = ccs_co2_per_tco2 - (cap_tco2 + tr_tco2 + st_tco2)
        oil_co2_per_unit = max(oil_tco2, 0.0) * captured_total
        # Oil CO2 was already included in ccs_co2_per_unit — subtract it so
        # the Implementing-CCS ribbon reflects only capture/transport/storage.
        ccs_co2_per_unit -= oil_co2_per_unit

    ccs_implementing_total = ccs_co2_per_unit + ccs_ch4_per_unit
    uncaptured_total = (
        uncaptured_proc_total
        + uncaptured_comb_total
        + ccs_implementing_total
        + oil_co2_per_unit
    )
    # `total` is the overall throughput (industrial + CCS-caused + EOR oil),
    # used only for bracket height positioning. Percentages are keyed to
    # `industrial_total` (the grey "Total unabated emissions" bar) so CCS and
    # oil emissions read as *additive* to the baseline — uncaptured % can
    # exceed 100%.
    total = industrial_total + ccs_implementing_total + oil_co2_per_unit

    # Grey "Total unabated emissions" bar and all percentages are keyed off
    # the unabated industrial-only total; CCS-caused emissions are a separate
    # source branch that ADDS to the unabated baseline rather than being part
    # of it.
    _total_name = "Total<br>unabated<br>emissions"

    all_nodes = [
        (_total_name, nc["total"], 0.001, 0.5, industrial_total),
        ("Process", nc["process"], 0.20, None, proc_tco2e),
        ("Combustion", nc["combustion"], 0.20, None, comb_tco2e),
    ]
    if include_ccs_emissions and ccs_implementing_total > eps:
        all_nodes.append(
            (
                "Implementing CCS",
                nc["implementing_ccs"],
                0.20,
                None,
                ccs_implementing_total,
            )
        )
    all_nodes.extend(
        [
            ("Process CO₂", nc["process_co2"], 0.45, None, proc_co2),
            ("Process CH₄", nc["process_ch4"], 0.45, None, proc_ch4),
            ("Combustion CO₂", nc["combustion_co2"], 0.45, None, comb_co2),
            ("Combustion CH₄", nc["combustion_ch4"], 0.45, None, comb_ch4),
        ]
    )
    if include_ccs_emissions and ccs_implementing_total > eps:
        all_nodes.extend(
            [
                ("CCS CO₂", nc["ccs_co2"], 0.45, None, ccs_co2_per_unit),
                ("CCS CH₄", nc["ccs_ch4"], 0.45, None, ccs_ch4_per_unit),
            ]
        )
    if include_ccs_emissions and oil_co2_per_unit > eps:
        all_nodes.append(("EOR CO₂", nc["eor_oil_co2"], 0.45, None, oil_co2_per_unit))
    all_nodes.extend(
        [
            ("Captured", nc["captured"], 0.75, None, captured_total),
            ("Uncaptured", nc["uncaptured"], 0.75, None, uncaptured_total),
        ]
    )

    raw_flows = [
        (_total_name, "Process", proc_tco2e),
        (_total_name, "Combustion", comb_tco2e),
        ("Process", "Process CO₂", proc_co2),
        ("Process", "Process CH₄", proc_ch4),
        ("Combustion", "Combustion CO₂", comb_co2),
        ("Combustion", "Combustion CH₄", comb_ch4),
        ("Process CO₂", "Captured", captured_proc),
        ("Process CO₂", "Uncaptured", uncaptured_proc_co2),
        ("Process CH₄", "Uncaptured", proc_ch4),
        ("Combustion CO₂", "Captured", captured_comb),
        ("Combustion CO₂", "Uncaptured", uncaptured_comb_co2),
        ("Combustion CH₄", "Uncaptured", comb_ch4),
    ]
    if include_ccs_emissions and ccs_implementing_total > eps:
        raw_flows.extend(
            [
                ("Implementing CCS", "CCS CO₂", ccs_co2_per_unit),
                ("Implementing CCS", "CCS CH₄", ccs_ch4_per_unit),
                ("CCS CO₂", "Uncaptured", ccs_co2_per_unit),
                ("CCS CH₄", "Uncaptured", ccs_ch4_per_unit),
            ]
        )
    if include_ccs_emissions and oil_co2_per_unit > eps:
        raw_flows.append(("EOR CO₂", "Uncaptured", oil_co2_per_unit))
    flows = [(s, t, v) for s, t, v in raw_flows if v > eps]

    if not flows:
        raise ValueError(
            f"No non-zero emission flows for industry '{industry}' "
            f"(methane_scenario='{methane_scenario}'). "
            "Sankey diagram requires at least one positive flow."
        )

    used_names = set()
    for s, t, _ in flows:
        used_names.update([s, t])

    kept = [
        (name, color, xp, yp, val)
        for name, color, xp, yp, val in all_nodes
        if name in used_names
    ]

    old_to_new = {name: i for i, (name, *_) in enumerate(kept)}
    sources = [old_to_new[s] for s, _, _ in flows]
    targets = [old_to_new[t] for _, t, _ in flows]
    values = [v for _, _, v in flows]

    node_names = [n for n, *_ in kept]
    node_colors_out = [c for _, c, *_ in kept]
    x_pos = [xp for _, _, xp, *_ in kept]
    node_vals = {n: v for n, _, _, _, v in kept}

    col_nodes: Dict[float, List[Tuple[str, float]]] = {}
    for name, _, xp, _, val in kept:
        col_nodes.setdefault(xp, []).append((name, val))

    y_map: Dict[str, float] = {}
    for xp, members in col_nodes.items():
        if len(members) == 1:
            y_map[members[0][0]] = 0.5
        else:
            col_total = sum(v for _, v in members)
            cursor = 0.0
            pad = 0.05
            usable = 1.0 - pad * (len(members) - 1)
            for name, val in members:
                frac = val / col_total if col_total else 1.0 / len(members)
                y_map[name] = cursor + frac * usable / 2
                cursor += frac * usable + pad

    y_pos = [y_map[n] for n in node_names]

    unit = row.get("unit_of_product", "unit product")

    rightmost_nodes = {"Captured", "Uncaptured"}

    _minimal_name = {
        _total_name: _total_name,
        "Process": "Process",
        "Combustion": "Combustion",
        "Implementing CCS": "Implementing CCS",
        "Process CO₂": "CO₂",
        "Process CH₄": "CH₄",
        "Combustion CO₂": "CO₂",
        "Combustion CH₄": "CH₄",
        "CCS CO₂": "CO₂",
        "CCS CH₄": "CH₄",
        "EOR CO₂": "CO₂",
        "Captured": "",
        "Uncaptured": "",
    }

    # Verbose stream names used only for the Process / Combustion mid-column
    # nodes; applied to both minimal_labels paths so the two Process/Combustion
    # bars always show the long phrase when verbose=True.
    _verbose_stream_name = {
        "Process": "Emissions from<br>industrial process<br>itself",
        "Combustion": "Emissions from<br> combustion",
        "Implementing CCS": "Additional emissions<br>from implementing<br>CCS",
    }

    def _node_display_name(n: str) -> str:
        if verbose and n in _verbose_stream_name:
            return _verbose_stream_name[n]
        return n

    if minimal_labels:
        node_labels = [
            _verbose_stream_name[n]
            if verbose and n in _verbose_stream_name
            else _minimal_name.get(n, n)
            for n in node_names
        ]
    else:
        node_labels = [
            ""
            if n in rightmost_nodes
            else f"{_node_display_name(n)}<br>{node_vals[n]:.2f} tCO₂e/{unit}"
            for n in node_names
        ]

    # Presentation mode: bold every non-empty stream label (plotly honors
    # HTML `<b>` tags in Sankey node labels) and bump the node font size so
    # the diagram reads clearly from across a conference room.
    _is_presentation = _resolve_display_mode(display_mode) == "presentation"
    if _is_presentation:
        node_labels = [f"<b>{lab}</b>" if lab else lab for lab in node_labels]
        _sankey_node_size = fs["sankey_node"] + 8
    else:
        _sankey_node_size = fs["sankey_node"]

    link_colors = [_hex_to_rgba(node_colors_out[s], 0.4) for s in sources]

    # chart_title = title or (
    #    f"GHG emission sources and destinations for {display_name} with CCS"
    # )

    fig = go.Figure(
        go.Sankey(
            arrangement="fixed",
            node={
                "label": node_labels,
                "color": node_colors_out,
                "x": x_pos,
                "y": y_pos,
                "pad": 20,
                "thickness": 18,
            },
            textfont={"size": _sankey_node_size, "family": "Arial"},
            link={
                "source": sources,
                "target": targets,
                "value": values,
                "color": link_colors,
            },
        )
    )
    chart_title = ""
    if title is not None:
        chart_title = title
    fig.update_layout(
        title={
            "text": chart_title,
            "font": {"size": fs["sankey_title"], "family": "Arial"},
        },
        font={"size": fs["sankey_layout"], "family": "Arial"},
        width=width,
        height=height,
    )

    captured_nodes = [n for n in node_names if n == "Captured"]
    uncaptured_nodes = [n for n in node_names if n == "Uncaptured"]

    sankey_right = 0.78
    fig.update_layout(margin={"r": 180})

    # Annotation for the "EOR CO₂" node — labels the standalone source stream
    # that starts in column 3 when co2_destination='eor'. Pixel offsets keep
    # the leader line thin and tight regardless of figure size.
    if include_ccs_emissions and oil_co2_per_unit > eps and "EOR CO₂" in y_map:
        _oil_node_y = y_map["EOR CO₂"]
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.50,
            y=1.0 - _oil_node_y,
            text="combustion of<br>EOR oil",
            showarrow=True,
            arrowhead=0,
            arrowwidth=1.0,
            arrowcolor=nc["eor_oil_co2"],
            ax=35,
            ay=-30,
            xanchor="left",
            yanchor="middle",
            align="left",
            font={
                "size": fs["sankey_node"],
                "family": "Arial",
                "color": nc["eor_oil_co2"],
            },
        )

    if show_final_values:
        max_x_val = max(x_pos)
        final_names = [n for n, xp in zip(node_names, x_pos) if xp == max_x_val]
        final_members = [(n, node_vals[n]) for n in final_names]
        final_col_total = sum(v for _, v in final_members)
        n_final = len(final_members)
        pad_col = 0.05
        usable_col = 1.0 - pad_col * (n_final - 1)
        label_x = max_x_val - 0.015

        for name in final_names:
            nv = node_vals[name]
            if nv < eps:
                continue
            frac = nv / final_col_total if final_col_total else 0
            node_height = frac * usable_col
            node_top = y_map[name] - node_height / 2

            node_idx = old_to_new[name]
            incoming = [
                (sources[j], values[j])
                for j in range(len(values))
                if targets[j] == node_idx
            ]

            cursor = node_top
            for _, flow_val in incoming:
                flow_h = (flow_val / nv) * node_height if nv else 0
                flow_center = cursor + flow_h / 2
                cursor += flow_h
                pct = flow_val / industrial_total * 100 if industrial_total > eps else 0
                fig.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=label_x,
                    y=1.0 - flow_center,
                    text=f"<b>{pct:.0f}%</b>",
                    showarrow=False,
                    font={"size": fs["sankey_node"] - 4},
                    xanchor="right",
                    yanchor="middle",
                )

    def _bracket_annotation(fig, node_group, label_text, color):
        if not node_group:
            return
        ys = [y_map[n] for n in node_group]
        fracs = [node_vals[n] / total for n in node_group]
        y_tops = [y - f * 0.35 for y, f in zip(ys, fracs)]
        y_bots = [y + f * 0.35 for y, f in zip(ys, fracs)]
        y_min = min(y_tops)
        y_max = max(y_bots)

        pad_y = (y_max - y_min) * 0.05 + 0.01
        y_top_plot = 1.0 - (y_min - pad_y)
        y_bot_plot = 1.0 - (y_max + pad_y)
        y_mid_plot = (y_top_plot + y_bot_plot) / 2

        bx = sankey_right + 0.04
        tick = 0.015
        shapes = [
            {
                "type": "line",
                "xref": "paper",
                "yref": "paper",
                "x0": bx,
                "y0": y_bot_plot,
                "x1": bx,
                "y1": y_top_plot,
                "line": {"color": color, "width": 2.5},
            },
            {
                "type": "line",
                "xref": "paper",
                "yref": "paper",
                "x0": bx - tick,
                "y0": y_top_plot,
                "x1": bx,
                "y1": y_top_plot,
                "line": {"color": color, "width": 2.5},
            },
            {
                "type": "line",
                "xref": "paper",
                "yref": "paper",
                "x0": bx - tick,
                "y0": y_bot_plot,
                "x1": bx,
                "y1": y_bot_plot,
                "line": {"color": color, "width": 2.5},
            },
        ]
        fig.update_layout(shapes=list(fig.layout.shapes or []) + shapes)

        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=bx + 0.01,
            y=y_mid_plot,
            text=label_text,
            showarrow=False,
            font={"size": fs["sankey_bracket"], "color": color, "family": "Arial"},
            textangle=0,
            xanchor="left",
            yanchor="middle",
            align="left",
        )

    # Percentages for the rightmost bracket labels use the TOTAL emissions
    # denominator (industrial + implementing-CCS + any EOR-oil combustion) so
    # Captured % + Uncaptured % = 100% of what the plant actually emits.
    _pct_denom = total if total > eps else industrial_total
    captured_pct = captured_total / _pct_denom * 100 if _pct_denom > eps else 0
    uncaptured_pct = uncaptured_total / _pct_denom * 100 if _pct_denom > eps else 0

    # Presentation mode: enlarge just the "Captured" / "Uncaptured" header
    # word via a per-span font-size override (plotly annotations honor
    # <span style="font-size:XXpx">). The second line stays at the bracket's
    # base size so captions don't overrun the chart.
    if _is_presentation:
        _hdr_size = fs["sankey_bracket"] + 12

        def _hdr(word):
            return f'<span style="font-size:{_hdr_size}px"><b>{word}</b></span>'

    else:

        def _hdr(word):
            return f"<b>{word}</b>"

    if captured_total > eps:
        if minimal_labels:
            if include_ccs_emissions:
                cap_text = (
                    f"{_hdr('Captured')}<br>{captured_pct:.0f}% of<br>"
                    f"total<br>emissions"
                )
            else:
                cap_text = f"{_hdr('Captured')}<br>{captured_pct:.0f}% of emissions"
        else:
            cap_text = f"{_hdr('Captured')}<br>{captured_total:.2f} tCO₂e/{unit}"
        _bracket_annotation(fig, captured_nodes, cap_text, color=nc["captured"])
    if uncaptured_total > eps:
        if minimal_labels:
            if include_ccs_emissions:
                uncap_text = (
                    f"{_hdr('Uncaptured')}<br>{uncaptured_pct:.0f}% of<br>"
                    f"total<br>emissions"
                )
            else:
                uncap_text = (
                    f"{_hdr('Uncaptured')}<br>" f"{uncaptured_pct:.0f}% of emissions"
                )
        else:
            uncap_text = (
                f"{_hdr('Uncaptured')}<br>" f"{uncaptured_total:.2f} tCO₂e/{unit}"
            )
        _bracket_annotation(fig, uncaptured_nodes, uncap_text, color=nc["uncaptured"])

    if output_path:
        fig.write_image(output_path, scale=2)

    return fig

def plot_state_sankey(
    industrial_emissions_df: pd.DataFrame,
    ccs_emissions_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    methane_scenario: str,
    eor_pct: float = 92,
    industries: Optional[List[str]] = None,
    which_flavor: Optional[List[str]] = None,
    title: Optional[str] = None,
    min_label: bool = False,
    show_final_values: bool = False,
    width: int = 1500,
    height: int = 620,
    output_path: Optional[str] = None,
    display_mode: Optional[str] = None,
    fig: Optional["go.Figure"] = None,
):
    import plotly.graph_objects as go

    fs = _resolve_fonts(display_mode)
    eor_frac = eor_pct / 100
    gs_frac = 1 - eor_frac

    eor_df = _build_breakdown_data(
        industrial_emissions_df,
        ccs_emissions_df,
        methane_scenario,
        "eor",
        industries,
        which_flavor,
    ).set_index("simple_product_name")
    gs_df = _build_breakdown_data(
        industrial_emissions_df,
        ccs_emissions_df,
        methane_scenario,
        "gs",
        industries,
        which_flavor,
    ).set_index("simple_product_name")

    common = eor_df.index.intersection(gs_df.index)
    ref = scenarios_df[
        scenarios_df["methane_scenario"] == methane_scenario
    ].drop_duplicates(subset="uid")
    ref = ref[ref["product"].isin(common)]
    captured_by_ind = ref.groupby("product")["mtco2_per_yr_captured_with_ccs"].sum()
    total_captured = captured_by_ind.sum()

    component_cols = [col for col, _, _ in _BREAKDOWN_COMPONENTS]

    weighted = {}
    for col in component_cols:
        val = sum(
            (eor_frac * eor_df.loc[ind, col] + gs_frac * gs_df.loc[ind, col])
            * captured_by_ind[ind]
            for ind in captured_by_ind.index
        )
        weighted[col] = val / total_captured if total_captured else 0.0

    proc_co2_total = (
        weighted["uncaptured_process_co2"] - weighted["captured_process_co2"]
    )
    comb_co2_total = (
        weighted["uncaptured_combustion_co2"] - weighted["captured_combustion_co2"]
    )
    total_process = weighted["process_ch4"] + proc_co2_total
    total_combustion = weighted["combustion_ch4"] + comb_co2_total
    total_industrial = total_process + total_combustion
    total_ccs = weighted["ccs_co2"] + weighted["ccs_ch4"]
    captured_total = -(
        weighted["captured_process_co2"] + weighted["captured_combustion_co2"]
    )
    uncaptured_total = (
        weighted["uncaptured_process_co2"]
        + weighted["uncaptured_combustion_co2"]
        + weighted["process_ch4"]
        + weighted["combustion_ch4"]
        + total_ccs
    )

    nc = _SANKEY_NODE_COLORS
    ccs_color = "#7b1fa2"
    atmos_color = nc["uncaptured"]
    stored_color = nc["captured"]

    eps = 1e-9

    ccs_label = "Emissions from\nimplementing CCS"
    total_label = "Total unabated\nindustrial emissions"
    process_label = "Industrial\nprocess emissions"
    combustion_label = "Industrial\ncombustion emissions"

    node_locs = [0.001, 0.25, 0.55, 0.78]

    all_nodes = [
        (total_label, nc["total"], node_locs[0], 0.5),
        (process_label, nc["process"], node_locs[1], None),
        (combustion_label, nc["combustion"], node_locs[1], None),
        (ccs_label, ccs_color, node_locs[1], None),
        ("Process CO₂", nc["process_co2"], node_locs[2], None),
        ("Process CH₄", nc["process_ch4"], node_locs[2], None),
        ("Combustion CO₂", nc["combustion_co2"], node_locs[2], None),
        ("Combustion CH₄", nc["combustion_ch4"], node_locs[2], None),
        ("CCS CO₂", "#ce93d8", node_locs[2], None),
        ("CCS CH₄", "#e1bee7", node_locs[2], None),
        ("Stored underground", stored_color, node_locs[3], None),
        ("Emitted to atmosphere", atmos_color, node_locs[3], None),
    ]
    node_vals = {
        total_label: total_industrial,
        process_label: total_process,
        combustion_label: total_combustion,
        ccs_label: total_ccs,
        "Process CO₂": proc_co2_total,
        "Process CH₄": weighted["process_ch4"],
        "Combustion CO₂": comb_co2_total,
        "Combustion CH₄": weighted["combustion_ch4"],
        "CCS CO₂": weighted["ccs_co2"],
        "CCS CH₄": weighted["ccs_ch4"],
        "Stored underground": captured_total,
        "Emitted to atmosphere": uncaptured_total,
    }

    raw_flows = [
        (total_label, process_label, total_process),
        (total_label, combustion_label, total_combustion),
        (process_label, "Process CO₂", proc_co2_total),
        (process_label, "Process CH₄", weighted["process_ch4"]),
        (combustion_label, "Combustion CO₂", comb_co2_total),
        (combustion_label, "Combustion CH₄", weighted["combustion_ch4"]),
        (ccs_label, "CCS CO₂", weighted["ccs_co2"]),
        (ccs_label, "CCS CH₄", weighted["ccs_ch4"]),
        ("Process CO₂", "Stored underground", -weighted["captured_process_co2"]),
        ("Process CO₂", "Emitted to atmosphere", weighted["uncaptured_process_co2"]),
        ("Combustion CO₂", "Stored underground", -weighted["captured_combustion_co2"]),
        (
            "Combustion CO₂",
            "Emitted to atmosphere",
            weighted["uncaptured_combustion_co2"],
        ),
        ("Process CH₄", "Emitted to atmosphere", weighted["process_ch4"]),
        ("Combustion CH₄", "Emitted to atmosphere", weighted["combustion_ch4"]),
        ("CCS CO₂", "Emitted to atmosphere", weighted["ccs_co2"]),
        ("CCS CH₄", "Emitted to atmosphere", weighted["ccs_ch4"]),
    ]
    flows = [(s, t, v) for s, t, v in raw_flows if v > eps]

    used_names = set()
    for s, t, _ in flows:
        used_names.update([s, t])

    kept = [(n, c, xp, yp) for n, c, xp, yp in all_nodes if n in used_names]
    old_to_new = {name: i for i, (name, *_) in enumerate(kept)}
    sources = [old_to_new[s] for s, _, _ in flows]
    targets = [old_to_new[t] for _, t, _ in flows]
    values = [v for _, _, v in flows]
    node_names = [n for n, *_ in kept]
    node_colors_out = [c for _, c, *_ in kept]
    x_pos = [xp for _, _, xp, _ in kept]

    total_all = total_industrial + total_ccs
    col_nodes: Dict[float, List[Tuple[str, float]]] = {}
    for n, _, xp, _ in kept:
        col_nodes.setdefault(xp, []).append((n, node_vals.get(n, 0)))

    y_map: Dict[str, float] = {}
    for xp, members in col_nodes.items():
        if len(members) == 1:
            y_map[members[0][0]] = 0.5
        else:
            col_total = sum(v for _, v in members)
            cursor = 0.0
            pad = 0.05
            usable = 1.0 - pad * (len(members) - 1)
            for name, val in members:
                frac = val / col_total if col_total else 1.0 / len(members)
                y_map[name] = cursor + frac * usable / 2
                cursor += frac * usable + pad

    for n, _, _, yp in kept:
        if yp is not None:
            y_map[n] = yp

    y_pos = [y_map[n] for n in node_names]

    rightmost_nodes = {"Stored underground", "Emitted to atmosphere"}
    mid_col_nodes = {
        "Process CO₂",
        "Process CH₄",
        "Combustion CO₂",
        "Combustion CH₄",
        "CCS CO₂",
        "CCS CH₄",
    }
    if min_label:
        node_labels = []
        for n in node_names:
            if n in rightmost_nodes:
                node_labels.append("")
            elif n in (total_label, ccs_label):
                mt = node_vals[n] * total_captured
                node_labels.append(
                    f"{n.replace(chr(10), '<br>')}<br>{mt:.0f} MtCO₂e/yr"
                )
            elif n in mid_col_nodes:
                short = n.split()[-1]
                node_labels.append(short)
            else:
                node_labels.append(n.replace("\n", "<br>"))
    else:
        node_labels = [
            f"{n.replace(chr(10), '<br>')}<br>{node_vals[n]:.2f} tCO₂e/tCO₂"
            for n in node_names
        ]

    link_colors = [_hex_to_rgba(node_colors_out[s], 0.4) for s in sources]

    sankey_trace = go.Sankey(
        arrangement="fixed",
        node={
            "label": node_labels,
            "color": node_colors_out,
            "x": x_pos,
            "y": y_pos,
            "pad": 20,
            "thickness": 18,
        },
        textfont={"size": fs["sankey_node"], "family": "Arial", "color": "black"},
        link={
            "source": sources,
            "target": targets,
            "value": values,
            "color": link_colors,
        },
    )
    if fig is None:
        fig = go.Figure(sankey_trace)
    else:
        fig.add_trace(sankey_trace)

    eor_label = f"{eor_pct:.0f}% EOR, {100 - eor_pct:.0f}% GS"
    chart_title = (
        title
        if title is not None
        else (f"Texas CCS: emission flows per tCO₂ captured ({eor_label})")
    )
    fig.update_layout(
        title={
            "text": chart_title,
            "font": {"size": fs["sankey_title"], "family": "Arial", "color": "black"},
        },
        font={"size": fs["sankey_layout"], "family": "Arial", "color": "black"},
        width=width,
        height=height,
        margin={"r": 80},
    )

    sankey_right = 0.78

    if show_final_values:
        max_x_val = max(x_pos)
        final_names = [n for n, xp in zip(node_names, x_pos) if xp == max_x_val]
        final_members = [(n, node_vals.get(n, 0)) for n in final_names]
        final_col_total = sum(v for _, v in final_members)
        n_final = len(final_members)
        pad_col = 0.05
        usable_col = 1.0 - pad_col * (n_final - 1)
        label_x = sankey_right - 0.015

        for name in final_names:
            nv = node_vals.get(name, 0)
            if nv < eps:
                continue
            frac = nv / final_col_total if final_col_total else 0
            node_height = frac * usable_col
            node_top = y_map[name] - node_height / 2

            node_idx = old_to_new[name]
            incoming = [
                (sources[j], values[j])
                for j in range(len(values))
                if targets[j] == node_idx
            ]
            if len(incoming) <= 1:
                continue

            cursor = node_top
            for _, flow_val in incoming:
                flow_h = (flow_val / nv) * node_height if nv else 0
                flow_center = cursor + flow_h / 2
                cursor += flow_h
                mt_val = flow_val * total_captured
                fig.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=label_x,
                    y=1.0 - flow_center,
                    text=f"<b>{mt_val:.0f}</b>",
                    showarrow=False,
                    font={"size": fs["sankey_node"] - 4, "color": "black"},
                    xanchor="right",
                    yanchor="middle",
                )

    def _state_bracket(fig, node_group, label_text, color):
        if not node_group:
            return
        ys = [y_map[n] for n in node_group]
        fracs = [node_vals[n] / total_all for n in node_group]
        y_tops = [y - f * 0.35 for y, f in zip(ys, fracs)]
        y_bots = [y + f * 0.35 for y, f in zip(ys, fracs)]
        y_min = min(y_tops)
        y_max = max(y_bots)

        pad_y = (y_max - y_min) * 0.05 + 0.01
        y_top_plot = 1.0 - (y_min - pad_y)
        y_bot_plot = 1.0 - (y_max + pad_y)
        y_mid_plot = (y_top_plot + y_bot_plot) / 2

        bx = sankey_right + 0.04
        tick = 0.015
        shapes = [
            {
                "type": "line",
                "xref": "paper",
                "yref": "paper",
                "x0": bx,
                "y0": y_bot_plot,
                "x1": bx,
                "y1": y_top_plot,
                "line": {"color": color, "width": 2.5},
            },
            {
                "type": "line",
                "xref": "paper",
                "yref": "paper",
                "x0": bx - tick,
                "y0": y_top_plot,
                "x1": bx,
                "y1": y_top_plot,
                "line": {"color": color, "width": 2.5},
            },
            {
                "type": "line",
                "xref": "paper",
                "yref": "paper",
                "x0": bx - tick,
                "y0": y_bot_plot,
                "x1": bx,
                "y1": y_bot_plot,
                "line": {"color": color, "width": 2.5},
            },
        ]
        fig.update_layout(shapes=list(fig.layout.shapes or []) + shapes)
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=bx + 0.01,
            y=y_mid_plot,
            text=label_text,
            showarrow=False,
            font={"size": fs["sankey_bracket"], "color": color, "family": "Arial"},
            xanchor="left",
            yanchor="middle",
            align="left",
        )

    stored_nodes = [n for n in node_names if n == "Stored underground"]
    atmos_nodes = [n for n in node_names if n == "Emitted to atmosphere"]

    if min_label:
        # Both bracket totals are computed from the same weighted factors that
        # drive the per-flow annotations above, so the bracket figure always
        # equals the sum of the flow labels (no "label says 197, flows sum to
        # 199" drift from mixing two computation paths).
        stored_mt = captured_total * total_captured
        emitted_mt = uncaptured_total * total_captured
        stored_text = f"<b>Stored</b><br>{stored_mt:.0f} MtCO₂e/yr"
        emitted_text = f"<b>Emitted</b><br>{emitted_mt:.0f} MtCO₂e/yr"
    else:
        stored_text = f"<b>Stored</b><br>{captured_total:.2f} tCO₂e/tCO₂"
        emitted_text = f"<b>Emitted</b><br>{uncaptured_total:.2f} tCO₂e/tCO₂"

    _state_bracket(fig, stored_nodes, stored_text, stored_color)
    _state_bracket(fig, atmos_nodes, emitted_text, atmos_color)

    if output_path:
        fig.write_image(output_path, scale=3)

    return fig

_DEFAULT_PRODUCT_GROUP_MAP: Dict[str, str] = {
    "refined_fuels": "refining_and_petrochemicals",
    "petrochemicals": "refining_and_petrochemicals",
    "power_ng_new_build": "power_generation_gas_and_coal",
    "power_ng_retrofit": "power_generation_gas_and_coal",
    "power_coal": "power_generation_gas_and_coal",
}

_LEGEND_FALLBACK_PALETTE: List[str] = [
    "#8c564b",  # brown
    "#17becf",  # teal
    "#bcbd22",  # olive
    "#9467bd",  # purple
    "#f77f6f",  # coral
    "#4a7c59",  # forest green
    "#c5b0d5",  # light purple
    "#d4a017",  # amber
]

def plot_products_to_destination_sankey(
    df: pd.DataFrame,
    product_col: str = "product",
    captured_col: str = "mtco2_per_yr_captured_with_ccs",
    net_emissions_col: str = "mtco2e_per_yr_emissions_including_ccs",
    name_replacements: Optional[Dict[str, str]] = None,
    colors: Optional[Dict[str, str]] = None,
    product_group_map: Optional[Dict[str, str]] = None,
    storage_color: str = "#3a3a3a",
    atmosphere_color: str = "#ffffff",
    title: Optional[str] = None,
    show_values: bool = True,
    small_node_threshold: float = 50.0,
    width: int = 1000,
    height: int = 700,
    output_path: Optional[str] = None,
    display_mode: Optional[str] = None,
):
    """Sankey with one node per product on the left and Storage / Atmosphere on the right.

    Each product's total flow equals `captured_col + net_emissions_col`. Product →
    Storage is `captured_col`; Product → Atmosphere is `net_emissions_col`.

    Args:
        df: Wide-format DataFrame with one row per product and the columns
            named by `product_col`, `captured_col`, and `net_emissions_col`.
            Usually pre-aggregated for a single scenario/methane combo.
        product_col: Column holding the product / industry name (default: "product").
        captured_col: Column with CO2 tonnes going to storage.
        net_emissions_col: Column with tCO2e tonnes going to atmosphere.
        name_replacements: Optional mapping to prettify product labels.
        colors: Optional mapping from product name → hex color; falls back to
            INDUSTRY_COLORS then to a grey default.
        title: Chart title. If None, no title is drawn.
        show_values: If True, node labels include their inflow/outflow value
            (no unit label; assumed MtCO2e/yr).

    Returns:
        Plotly Figure.
    """
    import plotly.graph_objects as go

    fs = _resolve_fonts(display_mode)
    color_map = {**INDUSTRY_COLORS, **(colors or {})}
    names = {**_FACILITY_NAME_DEFAULTS, **(name_replacements or {})}

    # Optionally combine products into aggregate groups (e.g., refined_fuels
    # + petrochemicals → "refining_and_petrochemicals"). None ⇒ use built-in
    # defaults; pass {} to disable grouping entirely.
    if product_group_map is None:
        product_group_map = _DEFAULT_PRODUCT_GROUP_MAP
    if product_group_map:
        df = df.copy()
        df[product_col] = df[product_col].replace(product_group_map)
        # Sum grouped rows; preserve the order they first appear in the input.
        first_pos = {v: i for i, v in enumerate(df[product_col].tolist())}
        df = df.groupby(product_col, sort=False).sum(numeric_only=True).reset_index()
        df = df.sort_values(by=product_col, key=lambda s: s.map(first_pos)).reset_index(
            drop=True
        )

    # Preserve caller's ordering (df should already be sorted how you want the
    # products to appear top-to-bottom on the left).
    products = df[product_col].tolist()
    captured_by_product = df.set_index(product_col)[captured_col].to_dict()
    net_by_product = df.set_index(product_col)[net_emissions_col].to_dict()

    # Node index layout: products first, then the two destinations.
    n_products = len(products)
    STORAGE_IDX = n_products
    ATMOS_IDX = n_products + 1

    total_storage = sum(captured_by_product[p] for p in products)
    total_atmos = sum(net_by_product[p] for p in products)

    def _label(name: str, value: float, pretty_override: Optional[str] = None) -> str:
        pretty = pretty_override or names.get(name, name.replace("_", " ").title())
        if show_values:
            return f"{pretty}<br>{value:.0f} MtCO<sub>2</sub>e/yr"
        return pretty

    # Node index layout: products first, then Atmosphere, then Storage. The
    # right-side order (Atmosphere before Storage) is what places Atmosphere
    # above Storage under snap layout.
    ATMOS_IDX = n_products
    STORAGE_IDX = n_products + 1

    # Product labels rendered as manual annotations (see below) so we can
    # anchor them to the LEFT of each node — Plotly's built-in Sankey labels
    # for nodes near x=0 unfortunately spill to the RIGHT of the node, over
    # the flows. Destination labels do render acceptably on the right via
    # Plotly's default, but we mirror the annotation approach for consistency.
    product_annotation_labels = [
        _label(p, captured_by_product[p] + net_by_product[p]) for p in products
    ]
    dest_annotation_labels = [
        _label("atmosphere", total_atmos, pretty_override="Atmosphere"),
        _label("storage", total_storage, pretty_override="Storage (CCS)"),
    ]
    # Blank out the built-in Sankey labels so they don't overlap the annotations.
    node_labels = [""] * (n_products + 2)
    node_colors = [color_map.get(p, "#888888") for p in products] + [
        atmosphere_color,
        storage_color,
    ]

    # Compute y-positions manually so big-flow nodes get vertical space
    # proportional to their flow and don't overlap smaller ones above/below
    # them (which is what happens with `arrangement="fixed"` + evenly-spaced y).
    # Node height in Plotly Sankey is proportional to flow, so we stack from
    # top by accumulating heights + a small gap.
    def _stacked_y(flows: List[float], pad_frac: float = 0.02) -> List[float]:
        total = float(sum(flows)) if any(flows) else 1.0
        n = len(flows)
        # Reserve (n+1) pads of vertical space; the rest is proportional to flows.
        gap = pad_frac
        usable = max(1.0 - gap * (n + 1), 0.05)
        cursor = gap
        centers = []
        for f in flows:
            h = (f / total) * usable
            centers.append(cursor + h / 2)
            cursor += h + gap
        return centers

    product_flows = [captured_by_product[p] + net_by_product[p] for p in products]
    product_y = _stacked_y(product_flows)
    dest_flows = [total_atmos, total_storage]  # Atmosphere first (top)
    dest_y = _stacked_y(dest_flows)

    # Explicit node x-positions: products on the far left (labels appear to
    # their left, vertically centered on each node), destinations on the far
    # right (labels appear to their right).
    node_x = [0.001] * n_products + [0.999, 0.999]
    node_y = product_y + dest_y

    sources: List[int] = []
    targets: List[int] = []
    values: List[float] = []
    link_colors: List[str] = []
    for i, p in enumerate(products):
        cap = captured_by_product[p]
        atm = net_by_product[p]
        if cap > 0:
            sources.append(i)
            targets.append(STORAGE_IDX)
            values.append(float(cap))
            link_colors.append(_hex_to_rgba(color_map.get(p, "#888888"), 0.55))
        if atm > 0:
            sources.append(i)
            targets.append(ATMOS_IDX)
            values.append(float(atm))
            link_colors.append(_hex_to_rgba(color_map.get(p, "#888888"), 0.35))

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="fixed",
                node=dict(
                    label=node_labels,
                    color=node_colors,
                    pad=18,
                    thickness=18,
                    line=dict(color="#333333", width=0.5),
                    x=node_x,
                    y=node_y,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                ),
            )
        ]
    )
    # Split products into "big" (labeled next to the node) and "small"
    # (grouped into a legend block below the plot to avoid overlap). Threshold
    # is on total flow (captured + net).
    big_indices, small_indices = [], []
    for i, p in enumerate(products):
        total_flow = captured_by_product[p] + net_by_product[p]
        if total_flow >= small_node_threshold:
            big_indices.append(i)
        else:
            small_indices.append(i)

    # Ensure each legend (small-node) product has a visually distinct color.
    # If two legend nodes share a color (either from INDUSTRY_COLORS or from a
    # user-supplied `colors` mapping), remap the duplicate to the next unused
    # color from _LEGEND_FALLBACK_PALETTE. This may collide with a big-node
    # color, which is acceptable — legend distinctness is the priority.
    if small_indices:
        used_in_legend: set = set()
        fallback_iter = iter(_LEGEND_FALLBACK_PALETTE)
        for i in small_indices:
            p = products[i]
            current = color_map.get(p, "#888888").lower()
            if current in used_in_legend:
                # Pick the next fallback that isn't already used in the legend.
                for candidate in fallback_iter:
                    if candidate.lower() not in used_in_legend:
                        color_map[p] = candidate
                        used_in_legend.add(candidate.lower())
                        break
                else:
                    # Exhausted palette — leave the duplicate rather than crash.
                    used_in_legend.add(current)
            else:
                used_in_legend.add(current)

    # Bottom margin scales with legend rows (2 items per row — wider entries
    # now that each label includes units).
    legend_entries_per_row = 2
    n_legend_rows = (
        (len(small_indices) + legend_entries_per_row - 1) // legend_entries_per_row
        if small_indices
        else 0
    )
    bottom_margin = 20 + 24 * n_legend_rows

    layout_kwargs = dict(
        width=width,
        height=height,
        font=dict(size=fs["annotation"], family="Arial"),
        # Big left/right margins to accommodate labels sitting outside the
        # Sankey plot area (nodes hug x=0 and x=1).
        margin=dict(l=220, r=220, t=60 if title else 20, b=bottom_margin),
    )
    if title is not None:
        layout_kwargs["title"] = dict(text=title, x=0.5, xanchor="center")
    fig.update_layout(**layout_kwargs)

    # Manual annotations for big-node labels (left) and destinations (right).
    # Plotly annotations use y=0 at bottom while Sankey y=0 is at top, so flip.
    annotation_font_size = fs["annotation"]
    # `label_offset_paper` pushes labels off the plot edge into the margin so
    # they don't butt up against the node itself. Increase if labels still crowd
    # the nodes; decrease if margin space is being wasted.
    label_offset_paper = 0.03
    for i in big_indices:
        fig.add_annotation(
            x=0 - label_offset_paper,
            y=1 - node_y[i],
            xref="paper",
            yref="paper",
            text=product_annotation_labels[i],
            showarrow=False,
            xanchor="right",
            yanchor="middle",
            align="right",
            font=dict(size=annotation_font_size, family="Arial", color="#222222"),
        )
    for j, label in enumerate(dest_annotation_labels):
        fig.add_annotation(
            x=1 + label_offset_paper,
            y=1 - dest_y[j],
            xref="paper",
            yref="paper",
            text=label,
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            align="left",
            font=dict(size=annotation_font_size, family="Arial", color="#222222"),
        )

    # Legend for small-flow products, below the plot area. Each entry: a
    # colored bullet + product name + value with units. Two entries per row
    # keeps long labels ("Refining & Petrochemicals: N MtCO2e/yr") from
    # colliding with the next column.
    if small_indices:
        # Row 0 is closest to the plot (paper y just below 0), row 1 is one line
        # further down; annotations use y<0 to sit in the bottom margin.
        row_height_paper = 24 / height  # 24px per row → paper-unit fraction
        col_x_positions = [0.05, 0.55]  # left edges of the two columns
        for k, i in enumerate(small_indices):
            row = k // legend_entries_per_row
            col = k % legend_entries_per_row
            p = products[i]
            pretty = names.get(p, p.replace("_", " ").title())
            total = captured_by_product[p] + net_by_product[p]
            swatch_color = color_map.get(p, "#888888")
            text = (
                f"<span style='color:{swatch_color}'>&#9679;</span> "
                f"<span style='color:#222222'>{pretty}: {total:.0f} "
                f"MtCO<sub>2</sub>e/yr</span>"
            )
            fig.add_annotation(
                x=col_x_positions[col],
                y=-0.02 - (row + 1) * row_height_paper,
                xref="paper",
                yref="paper",
                text=text,
                showarrow=False,
                xanchor="left",
                yanchor="top",
                align="left",
                font=dict(size=annotation_font_size, family="Arial"),
            )

    if output_path:
        fig.write_image(output_path, scale=2)
    return fig

_STATE_MAP_DEFAULTS: Dict[str, Dict] = {
    "Texas": {
        "project_state_codes": ["TX"],
        "xlim": (-106.8, -93.5),
        "ylim": (25.6, 36.75),
        "cities": [
            (-97.7431, 30.2672, "Austin"),
            (-95.3698, 29.7604, "Houston"),
            (-96.7970, 32.7767, "Dallas"),
        ],
    },
    "Louisiana": {
        "project_state_codes": ["LA"],
        "xlim": (-94.2, -88.7),
        "ylim": (28.7, 33.3),
        "cities": [
            (-90.0715, 29.9511, "New Orleans"),
            (-91.1871, 30.4515, "Baton Rouge"),
            #    (-93.2174, 30.2266, "Lake Charles"),
        ],
    },
}

_CITY_LABEL_OFFSETS: Dict[str, Tuple[int, int]] = {
    "Houston": (-5, 10),
    "Austin": (-8, 10),  # NW
    "Dallas": (8, 8),  # NE
    "New Orleans": (-60, -55),  # SW of marker; leader line drawn
    "Baton Rouge": (6, 36),  # NNE of marker; leader line drawn
    # "Lake Charles": (8, 10),
}

_CITY_LEADER_LINES = {"New Orleans", "Baton Rouge"}

def plot_facility_map(
    scenarios_df: pd.DataFrame,
    projects_df: pd.DataFrame,
    us_states_gdf,
    states: Optional[List[str]] = None,
    methane_scenario: str = "edf20",
    scenario_filter: str = "baseline",
    name_replacements: Optional[Dict[str, str]] = None,
    colors: Optional[Dict[str, str]] = None,
    markers: Optional[Dict[str, str]] = None,
    cities: Optional[List[Tuple[float, float, str]]] = None,
    rivers_gdf=None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    jitter_radius: float = 0.18,
    markersize: int = 150,
    big_markers: bool = False,
    alpha: float = 0.85,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    seed: int = 42,
    display_mode: Optional[str] = None,
    legend_right: bool = False,
    legend_right_valign: str = "center",
    legend_tight: bool = False,
    reduced_legend: bool = False,
    legend_ncol: Optional[int] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Map of CCS-linked facilities in one or more U.S. states.

    Args:
        states: List of full state names (e.g. ["Texas", "Louisiana"]). Highlighted
            white on the map; other states are grey. Facilities are filtered to
            projects whose `project_state` is in the corresponding project state
            codes from _STATE_MAP_DEFAULTS. Defaults to ["Texas"].
        xlim / ylim: Map view bounds. If None, computed from the union of state
            defaults in _STATE_MAP_DEFAULTS.
        cities: Reference cities to label. If None, uses the union of state
            defaults from _STATE_MAP_DEFAULTS.
        rivers_gdf: Optional GeoDataFrame of river/waterway line features to
            draw beneath the state fill (e.g. Natural Earth's 10m rivers layer:
            https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-rivers-lake-centerlines/).
            Reprojected to EPSG:4326 if needed and clipped to the map view.
        markersize: Base size for facility markers (matplotlib s= parameter).
            Actual size is markersize/2 in normal mode, markersize*3 in mini mode.
            Default 140 (≈70pt markers in normal display).
        legend_right: If False (default), the industry legend sits below the
            map in two columns. If True, the legend sits to the right of the
            map in a single column (useful for landscape slide layouts).
        legend_right_valign: Only used when ``legend_right=True``. Vertical
            placement of the right-side legend relative to the map's axes:
            ``"upper"``, ``"center"`` (default), or ``"lower"``.
        legend_tight: If True, the legend's left edge is anchored just east
            of the highlighted state's actual geometry (computed from
            `us_states_gdf`), so the state itself is never overlapped. For a
            Louisiana-centered map this places the legend over Mississippi.
            The right edge of the legend is allowed to extend past the map's
            right xlim bound. Takes precedence over `legend_right`.
            Default False.
        legend_ncol: Number of columns in the industry legend. When None
            (default), each legend placement picks a sensible default:
            ``legend_tight`` and ``legend_right`` use 1 column, the default
            below-the-map placement uses 2. Pass an integer to override.
        (other args unchanged from prior single-state version)
    """
    from scipy.spatial import KDTree

    fs = _resolve_fonts(display_mode)

    if states is None:
        states = ["Texas"]
    unknown = [s for s in states if s not in _STATE_MAP_DEFAULTS]
    if unknown:
        raise ValueError(
            f"Unknown state(s) {unknown}. Known: {list(_STATE_MAP_DEFAULTS)}. "
            "Add an entry to _STATE_MAP_DEFAULTS to extend."
        )
    state_defaults = [_STATE_MAP_DEFAULTS[s] for s in states]
    project_state_codes = {
        code for sd in state_defaults for code in sd["project_state_codes"]
    }

    names = {**_FACILITY_NAME_DEFAULTS, **(name_replacements or {})}
    color_map = {**INDUSTRY_COLORS, **(colors or {})}
    # When `markers` is None, every industry uses a circle (prior behavior).
    # Pass `markers=INDUSTRY_MARKERS` for shape-coded industries (readable in
    # B&W); pass a partial dict to override shapes for selected industries
    # while keeping circles for the rest.
    marker_map = markers or {}

    if cities is None:
        cities = [c for sd in state_defaults for c in sd["cities"]]

    if xlim is None:
        xlim = (
            min(sd["xlim"][0] for sd in state_defaults),
            max(sd["xlim"][1] for sd in state_defaults),
        )
    if ylim is None:
        ylim = (
            min(sd["ylim"][0] for sd in state_defaults),
            max(sd["ylim"][1] for sd in state_defaults),
        )

    filtered_df = scenarios_df.loc[
        (scenarios_df.scenario_id.str.contains(scenario_filter))
        & (scenarios_df.methane_scenario == methane_scenario)
    ]
    # Filter to projects in the requested states. project_state may live on
    # scenarios_df (post-refactor) or on projects_df (pre-refactor).
    if "project_state" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["project_state"].isin(project_state_codes)
        ]
    elif "project_state" in projects_df.columns:
        keep_uids = set(
            projects_df.loc[
                projects_df["project_state"].isin(project_state_codes), "uid"
            ]
        )
        filtered_df = filtered_df[filtered_df["uid"].isin(keep_uids)]
    if "lat_degrees_n" not in filtered_df.columns:
        filtered_df = filtered_df.merge(
            projects_df[["uid", "lat_degrees_n", "lon_degrees_e"]],
            on="uid",
            how="left",
        )

    # One row per facility, regardless of any upstream join cardinality
    # (e.g., multiple emission_profiles sharing an industrial_product name,
    # or multiple destination/distance variants per project). Prevents the
    # legend counts and the plotted markers from being inflated.
    filtered_df = filtered_df.drop_duplicates(subset="uid")

    industry_counts = filtered_df.groupby("product").size().sort_values(ascending=False)
    sorted_products = list(industry_counts.index)

    # Per-product summed product_amount and unit, taken from projects_df and
    # restricted to the uids actually plotted (so sums match the state filter).
    # projects_df stores the industry under `co2_source`; scenarios_df uses
    # `product`. We join on uid to carry product_amount and product_unit
    # alongside the plot's product label.
    _plot_uids = set(filtered_df["uid"].dropna().unique())
    _pdf = projects_df[projects_df["uid"].isin(_plot_uids)].drop_duplicates(
        subset="uid"
    )
    _amount_lookup = _pdf.groupby("co2_source")["product_amount"].sum().to_dict()
    _unit_lookup = (
        _pdf.dropna(subset=["product_unit"])
        .groupby("co2_source")["product_unit"]
        .first()
        .to_dict()
    )
    # (divisor, display_unit, decimal_digits) by product. Tonnage industries
    # scale up by 1e6 → Mt/yr; MWh electricity industries scale up by 1e6 →
    # TWh/yr; refining converts bbl/yr → kbbl/yr. Any product not listed falls
    # back to the raw YAML unit with no scaling.
    _unit_scale = {
        "hydrogen": (1e6, "Mt/yr", 1),
        "ammonia": (1e6, "Mt/yr", 1),
        "methanol": (1e6, "Mt/yr", 1),
        "ethanol": (1e6, "Mt/yr", 2),
        "lng": (1e6, "Mt/yr", 1),
        "dac": (1e6, "Mt/yr", 1),
        "iron_steel": (1e6, "Mt/yr", 1),
        "petrochemicals": (1e6, "Mt/yr", 1),
        "power_ng_retrofit": (1e6, "TWh/yr", 1),
        "power_ng_new_build": (1e6, "TWh/yr", 1),
        "power_ng_data_center": (1e6, "TWh/yr", 1),
        "power_coal": (1e6, "TWh/yr", 1),
        "refined_fuels": (1e3, "kbbl/yr", 0),
        "ngp": (1e9, "Quad/yr", 2),  # MMBtu → Quad (1 Quad = 1e15 Btu = 1e9 MMBtu)
    }

    mappable = filtered_df.dropna(subset=["lat_degrees_n", "lon_degrees_e"])
    lons = mappable["lon_degrees_e"].values.copy()
    lats = mappable["lat_degrees_n"].values.copy()
    prods = mappable["product"].values

    rng = np.random.default_rng(seed)
    orig_lons = lons.copy()
    coords = np.column_stack([lons, lats])
    tree = KDTree(coords)
    neighbor_counts = np.array(
        [len(tree.query_ball_point(pt, r=0.15)) for pt in coords]
    )
    needs_jitter = neighbor_counts > 1
    n_jitter = needs_jitter.sum()
    if n_jitter > 0:
        angles = rng.uniform(0, 2 * np.pi, n_jitter)
        radii = jitter_radius * np.sqrt(rng.uniform(0, 1, n_jitter))
        dx = radii * np.cos(angles)
        dy = radii * np.sin(angles)
        coastal = orig_lons[needs_jitter] > -95.5
        dx[coastal] = -np.abs(dx[coastal])
        lons[needs_jitter] += dx
        lats[needs_jitter] += dy

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    us_crs = us_states_gdf.crs
    if us_crs is not None and str(us_crs) != "EPSG:4326":
        us_plot = us_states_gdf.to_crs("EPSG:4326")
    else:
        us_plot = us_states_gdf

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    is_mini = display_mode == "mini"
    border_lw = 4.5 if is_mini else 2
    marker_s = markersize * 4 if is_mini else markersize  # / 2
    if big_markers:
        marker_s = int(marker_s * 1.75)

    highlighted = set(states)
    us_plot.plot(
        facecolor=us_plot["name"].apply(
            lambda x: "white" if x in highlighted else "#e8e8e8"
        ),
        edgecolor="#222222",
        lw=border_lw,
        ax=ax,
    )

    # Rivers: overlay after state fill, before facility markers. Reproject to
    # EPSG:4326 if needed, then clip to the union of the highlighted state
    # polygons so we don't show rivers in Mexico or in non-featured US states.
    if rivers_gdf is not None:
        import geopandas as gpd  # local import: only needed when rivers requested
        from shapely.ops import unary_union

        rivers = rivers_gdf
        if rivers.crs is not None and str(rivers.crs) != "EPSG:4326":
            rivers = rivers.to_crs("EPSG:4326")
        # Fast prefilter to the map bbox so the geometric clip is cheap.
        pad = 0.5
        rivers = rivers.cx[xlim[0] - pad : xlim[1] + pad, ylim[0] - pad : ylim[1] + pad]

        # Build the mask geometry from the highlighted state polygons.
        highlighted_geoms = us_plot.loc[us_plot["name"].isin(highlighted), "geometry"]
        if len(highlighted_geoms):
            mask = unary_union(highlighted_geoms.values)
            rivers = gpd.clip(rivers, mask)

        river_lw = 1.4 if is_mini else 0.8
        rivers.plot(
            ax=ax,
            color="#4a90d9",
            linewidth=river_lw,
            alpha=0.7,
            zorder=2,
        )

    for product in sorted_products:
        mask = prods == product
        c = color_map.get(product, "#333333")
        label = names.get(product, product.replace("_", " ").title())
        count = int(industry_counts[product])
        amount = _amount_lookup.get(product, 0.0)
        raw_unit = _unit_lookup.get(product, "")
        divisor, display_unit, digits = _unit_scale.get(product, (1.0, raw_unit, 0))
        scaled = amount / divisor
        unit_suffix = f" {display_unit}" if display_unit else ""
        facility_word = "facility" if count == 1 else "facilities"
        ax.scatter(
            lons[mask],
            lats[mask],
            c=c,
            s=marker_s,
            marker=marker_map.get(product, "o"),
            edgecolors="#333333",
            linewidths=1.5 if is_mini else 1,
            alpha=alpha,
            zorder=4,
            label=f"{label} ({count} {facility_word}; {scaled:,.{digits}f}{unit_suffix})",
        )

    if not is_mini:
        for lon, lat, name in cities:
            ax.scatter(
                lon,
                lat,
                color="black",
                s=100,
                zorder=5,
                edgecolor="white",
                lw=1.5,
                marker="s",
            )
            xytext = _CITY_LABEL_OFFSETS.get(name, (8, 8))
            # Anchor the text relative to the offset direction so labels grow
            # away from the marker instead of overlapping it.
            ha = "center"  # ha = "left" if xytext[0] >= 0 else "right"
            va = "bottom" if xytext[1] >= 0 else "top"
            _arrowprops = (
                dict(
                    arrowstyle="-",
                    color="black",
                    linewidth=0.6,
                    shrinkA=0,
                    shrinkB=3,
                )
                if name in _CITY_LEADER_LINES
                else None
            )
            ax.annotate(
                name,
                xy=(lon, lat),
                xytext=xytext,
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=fs["annotation"],
                fontfamily="Arial",
                fontweight="normal",
                color="#222222",
                zorder=6,
                # White halo makes labels legible over rivers/state fill and
                # softens any residual near-collisions.
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                arrowprops=_arrowprops,
            )

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if not is_mini:
        if display_mode == "presentation":
            fontsize = fs["legend"] + 1
        else:
            fontsize = fs["legend"] - 1

        # Legend-marker scale. The handle size is markerscale * plotted-marker-s;
        # when big_markers is on, the on-map markers are 1.75x bigger via
        # `marker_s`, but we also want the legend swatches visibly larger, so
        # bump the scale factor itself.
        _legend_markerscale = 1.0 if big_markers else 0.6

        # Reduced legend: just industry names rendered in each industry's
        # color — no marker swatch, no facility counts, no amount info. Built
        # once here, then threaded into whichever legend-placement branch runs.
        if reduced_legend:
            _reduced_handles = []
            _reduced_labels = []
            _reduced_label_colors = []
            for product in sorted_products:
                c = color_map.get(product, "#333333")
                name = names.get(product, product.replace("_", " ").title())
                # facecolor is irrelevant (handle is hidden) but needs a value
                # so labelcolor plumbing has a color to inherit from; edges
                # must be fully suppressed or matplotlib draws a thin vertical
                # line even when handlelength=0.
                _reduced_handles.append(
                    Patch(facecolor=c, edgecolor="none", linewidth=0, label=name)
                )
                _reduced_labels.append(name)
                _reduced_label_colors.append(c)
            # In reduced-legend mode, the text IS the swatch, so bump its
            # weight and size and switch to a condensed family so more
            # entries fit without overlap.
            _reduced_kwargs = {
                "handles": _reduced_handles,
                "labels": _reduced_labels,
                "handlelength": 0,
                "handleheight": 0,
                "handletextpad": 0,
                "labelcolor": _reduced_label_colors,
                # `prop` wins for the FontProperties object, so set `size`
                # inside it too — otherwise matplotlib's legend builds a
                # FontProperties from the dict without the fontsize kwarg
                # applied, and the bump doesn't actually take effect.
                "prop": {
                    "family": "Arial Narrow",
                    "weight": "bold",
                    "size": fontsize + 6,
                },
                "fontsize": fontsize + 12,
            }
        else:
            _reduced_kwargs = {}

        if legend_tight:
            # Anchor the legend's LEFT edge just east of the highlighted
            # state's geometry so the state itself is never overlapped. The
            # right edge is allowed to extend past the map view bounds; this
            # is what the caller asked for and it keeps the full plot of
            # Louisiana (and its facility markers) unobstructed. If we can't
            # look up the state geometry for any reason, fall back to the
            # eastern edge of the view.
            try:
                highlighted_geoms = us_plot.loc[
                    us_plot["name"].isin(highlighted), "geometry"
                ]
                east_lon = float(highlighted_geoms.total_bounds[2])
            except Exception:
                east_lon = float(ax.get_xlim()[1])
            xlo, xhi = ax.get_xlim()
            ylo, yhi = ax.get_ylim()
            x_pad = (xhi - xlo) * 0.005
            y_pad = (yhi - ylo) * 0.02
            leg = ax.legend(
                **{
                    "loc": "upper left",
                    "bbox_to_anchor": (east_lon + x_pad, yhi - y_pad),
                    "bbox_transform": ax.transData,
                    "fontsize": fontsize,
                    "frameon": True,
                    "facecolor": "white",
                    "edgecolor": "#cccccc",
                    "framealpha": 0.92,
                    "markerscale": _legend_markerscale,
                    "handletextpad": 0.4,
                    "borderpad": 0.5,
                    "labelspacing": 0.4,
                    "ncol": legend_ncol if legend_ncol is not None else 1,
                    "title": "",
                    **_reduced_kwargs,
                }
            )
            leg.set_zorder(10)
        elif legend_right:
            # Vertical placement options for the right-side legend. Each key
            # maps to a (matplotlib legend loc, bbox_to_anchor y-coord) pair.
            _right_valign_map = {
                "upper": ("upper left", 1.0),
                "center": ("center left", 0.5),
                "lower": ("lower left", 0.0),
            }
            if legend_right_valign not in _right_valign_map:
                raise ValueError(
                    f"legend_right_valign={legend_right_valign!r} is not valid; "
                    f"must be one of {sorted(_right_valign_map)}."
                )
            _loc, _y = _right_valign_map[legend_right_valign]
            ax.legend(
                **{
                    "loc": _loc,
                    "bbox_to_anchor": (1.02, _y),
                    "fontsize": fontsize,
                    "frameon": False,
                    "markerscale": _legend_markerscale,
                    "handletextpad": 0.4,
                    "borderpad": 0.2,
                    "labelspacing": 0.5,
                    "ncol": legend_ncol if legend_ncol is not None else 1,
                    "title": "",
                    **_reduced_kwargs,
                }
            )
        else:
            ax.legend(
                **{
                    "loc": "upper center",
                    "bbox_to_anchor": (0.5, -0.02),
                    "fontsize": fontsize,
                    "frameon": False,
                    "markerscale": _legend_markerscale,
                    "handletextpad": 0.4,
                    "borderpad": 0.2,
                    "labelspacing": 0.3,
                    "columnspacing": 0.5,
                    "ncol": legend_ncol if legend_ncol is not None else 2,
                    "title": "",
                    **_reduced_kwargs,
                }
            )

    if title is not None and not is_mini:
        ax.set_title(
            title,
            fontsize=fs["title"],
            fontfamily="Arial",
            fontweight="semibold",
            pad=12,
        )

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)

    return fig, ax

def _annotate_facility_windows(
    scen_df: pd.DataFrame,
    base_year: int,
    facility_lifetime_years: int,
) -> pd.DataFrame:
    """Add `_fac_start` and `_fac_end` columns to `scen_df` based on the per-
    project `retrofit_newbuild` flag (if present).

    For projects flagged "retrofit" (case-insensitive), the facility operates
    `facility_lifetime_years` years starting from `base_year` — i.e. the plant
    already exists today and continues for that long regardless of when CCS
    starts. CCS still activates at `project_start`.

    For projects flagged "new build" or any other value (including missing),
    the facility operates `facility_lifetime_years` years starting from
    `project_start`, matching the pre-flag behavior.
    """
    if "retrofit_newbuild" in scen_df.columns:
        is_retrofit = (
            scen_df["retrofit_newbuild"].astype(str).str.strip().str.lower()
            == "retrofit"
        )
    else:
        is_retrofit = pd.Series(False, index=scen_df.index)
    fac_start = np.where(
        is_retrofit, base_year, scen_df["project_start"].astype(int)
    ).astype(int)
    return scen_df.assign(
        _fac_start=fac_start,
        _fac_end=fac_start + facility_lifetime_years,
    )

def plot_emissions_through_time(
    scenarios_df: pd.DataFrame,
    projects_df: pd.DataFrame,
    methane_scenario: str,
    scenario: str = "baseline",
    ccs_active_years: int = 12,
    facility_lifetime_years: int = 30,
    base_year: Optional[int] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (14, 7),
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    title_fontsize: Optional[int] = None,
    legend_fontsize: Optional[int] = None,
    display_mode: Optional[str] = None,
    output_path: Optional[str] = None,
    plot_existing: bool = False,
    legend_ncol: int = 1,
    label_totals: bool = False,
    totals_offsets: Optional[Dict[str, Tuple[float, float]]] = None,
    totals_anchor_years: Optional[Dict[str, int]] = None,
    totals_fontsize: Optional[float] = None,
    totals_fontweight: str = "semibold",
    totals_leader_linewidth: float = 0.8,
    trajectory_linewidth: float = 2.0,
    stored_label_xy: Optional[Tuple[float, float]] = None,
    uncaptured_label_xy: Optional[Tuple[float, float]] = None,
    implementing_ccs_label_xy: Optional[Tuple[float, float]] = None,
    implementing_ccs_anchor_xy: Optional[Tuple[float, float]] = None,
    region_label_fontsize: Optional[float] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Facility-lifetime emissions timeline, honoring the per-project
    `retrofit_newbuild` flag.

    For a retrofit project, the facility is modeled as already operating in
    `base_year` and continuing for `facility_lifetime_years`; CCS kicks in
    at `project_start` and runs for `ccs_active_years`, with unabated
    emissions bracketing that CCS window on both sides. For a new-build
    project, the facility starts at `project_start` (nothing earlier) and
    runs for `facility_lifetime_years`.

    `base_year` defaults to the current calendar year if not supplied.
    Projects whose rows lack the `retrofit_newbuild` column are treated as
    new builds (back-compat).

    If ``plot_existing=True``, an additional gray fill is drawn from the
    left-x limit to the earliest project_start at the sum of
    ``unabated_mtco2e_per_yr`` across retrofit-flagged projects. This
    represents the "existing" (pre-CCS, pre-buildout) emissions baseline
    from facilities already operating today.
    """
    fs = _resolve_fonts(display_mode)
    if base_year is None:
        base_year = datetime.date.today().year

    scen_df = scenarios_df[
        (scenarios_df["methane_scenario"] == methane_scenario)
        & (scenarios_df["scenario"] == scenario)
    ].drop_duplicates(subset="uid")

    if "project_start" not in scen_df.columns:
        scen_df = scen_df.merge(
            projects_df[["uid", "project_start"]].drop_duplicates(subset="uid"),
            on="uid",
            how="left",
        )
    if (
        "retrofit_newbuild" not in scen_df.columns
        and "retrofit_newbuild" in projects_df.columns
    ):
        scen_df = scen_df.merge(
            projects_df[["uid", "retrofit_newbuild"]].drop_duplicates(subset="uid"),
            on="uid",
            how="left",
        )

    # Drop rows missing any accumulated value — a single NaN propagates into
    # the yearly arrays and breaks ylim computation.
    required = [
        "project_start",
        "unabated_mtco2e_per_yr",
        "mtco2_per_yr_captured_with_ccs",
        "mtco2e_per_yr_emitted_because_of_ccs",
    ]
    missing = scen_df[scen_df[required].isna().any(axis=1)]
    if len(missing):
        names = missing.get("project_name", missing["uid"]).tolist()
        logger.warning(
            "plot_emissions_through_time: dropping %d project(s) with NaN in "
            "required columns (fix upstream CSV): %s",
            len(missing),
            names,
        )
        scen_df = scen_df.dropna(subset=required)

    if scen_df.empty:
        raise ValueError(
            f"No projects with complete data for methane_scenario={methane_scenario!r}, "
            f"scenario={scenario!r}. Nothing to plot."
        )

    scen_df = _annotate_facility_windows(scen_df, base_year, facility_lifetime_years)
    timeline_start = int(scen_df["_fac_start"].min())
    timeline_end = int(scen_df["_fac_end"].max())
    years = np.arange(timeline_start, timeline_end)

    uncaptured_arr = np.zeros(len(years))
    lifecycle_arr = np.zeros(len(years))
    unabated_arr = np.zeros(len(years))
    captured_arr = np.zeros(len(years))

    # Switching scenarios: CCS runs for the full facility lifetime after
    # project_start, with the destination mix pivoting from the first-period
    # mix (baseline or current_mix) to 100% EOR at ps + ccs_active_years.
    # The stored per-year CCS-lifecycle emission rate on a switching row is
    # a time-weighted mean over L years, so for a year-by-year timeline we
    # need the per-period rates from the underlying baseline/current_mix
    # and all_eor rows.
    _SWITCHING_FIRST = {
        "switching": "baseline",
        "switching_from_current": "current_mix",
    }
    is_switching = scenario in _SWITCHING_FIRST
    if is_switching:
        _p1_name = _SWITCHING_FIRST[scenario]
        _p2_name = "all_eor"

        def _ccs_lc_by_uid(sname):
            return (
                scenarios_df[
                    (scenarios_df["methane_scenario"] == methane_scenario)
                    & (scenarios_df["scenario"] == sname)
                ]
                .drop_duplicates(subset="uid")
                .set_index("uid")["mtco2e_per_yr_emitted_because_of_ccs"]
            )

        _ccs_lc_p1 = _ccs_lc_by_uid(_p1_name)
        _ccs_lc_p2 = _ccs_lc_by_uid(_p2_name)

    for _, row in scen_df.iterrows():
        ps = int(row["project_start"])
        fac_start = int(row["_fac_start"])
        fac_end = int(row["_fac_end"])
        ccs_end = ps + ccs_active_years

        unabated = row["unabated_mtco2e_per_yr"]
        captured = row["mtco2_per_yr_captured_with_ccs"]
        ccs_lc = row["mtco2e_per_yr_emitted_because_of_ccs"]

        if is_switching:
            uid = row["uid"]
            ccs_lc_p1 = float(_ccs_lc_p1.get(uid, ccs_lc))
            ccs_lc_p2 = float(_ccs_lc_p2.get(uid, ccs_lc))

            for i, yr in enumerate(years):
                if yr < fac_start or yr >= fac_end:
                    continue
                unabated_arr[i] += unabated
                if yr < ps:
                    # Retrofit pre-CCS: facility already running, unabated.
                    uncaptured_arr[i] += unabated
                elif yr < ccs_end:
                    uncaptured_arr[i] += unabated - captured
                    lifecycle_arr[i] += ccs_lc_p1
                    captured_arr[i] += captured
                else:
                    # Post-45Q tail: CCS keeps running, destination pivots
                    # to 100% EOR until fac_end.
                    uncaptured_arr[i] += unabated - captured
                    lifecycle_arr[i] += ccs_lc_p2
                    captured_arr[i] += captured
        else:
            for i, yr in enumerate(years):
                if yr < fac_start or yr >= fac_end:
                    continue
                unabated_arr[i] += unabated
                if ps <= yr < ccs_end:
                    uncaptured_arr[i] += unabated - captured
                    lifecycle_arr[i] += ccs_lc
                    captured_arr[i] += captured
                else:
                    uncaptured_arr[i] += unabated

    net_arr = uncaptured_arr + lifecycle_arr

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Existing (pre-buildout) emissions from retrofit facilities, drawn from
    # the left x-limit to the earliest project_start of the future buildout
    # (i.e. first project_start >= base_year — ignores pre-existing historical
    # CCS projects). Only visible when the user's xlim starts earlier than
    # that buildout start.
    if plot_existing and "retrofit_newbuild" in scen_df.columns:
        is_retrofit = (
            scen_df["retrofit_newbuild"].astype(str).str.strip().str.lower()
            == "retrofit"
        )
        existing_level = float(scen_df.loc[is_retrofit, "unabated_mtco2e_per_yr"].sum())
        future_starts = scen_df.loc[
            scen_df["project_start"] >= base_year, "project_start"
        ]
        ccs_buildout_start = (
            int(future_starts.min()) if len(future_starts) else base_year
        )
        left_x = xlim[0] if xlim is not None else timeline_start
        if existing_level > 0 and left_x < ccs_buildout_start:
            ax.fill_between(
                [left_x, ccs_buildout_start],
                0,
                existing_level,
                step="post",
                facecolor="#888888",
                edgecolor="#555555",
                linewidth=0.0,
                hatch="///",
                alpha=0.7,
                label="Existing emissions (pre-buildout)",
            )

    ax.fill_between(
        years,
        0,
        uncaptured_arr,
        step="post",
        facecolor="#888888",
        edgecolor="#555555",
        linewidth=0.0,
        hatch="///",
        alpha=0.7,
        label="Uncaptured industrial emissions",
    )
    ax.fill_between(
        years,
        uncaptured_arr,
        net_arr,
        step="post",
        color="#d62728",
        alpha=0.7,
        label="Emissions from implementing CCS",
    )

    stored_arr = captured_arr  # unabated_arr - net_arr
    stored_below = -np.abs(stored_arr)
    ax.fill_between(
        years,
        0,
        stored_below,
        step="post",
        facecolor="#a8d5a2",
        edgecolor="none",
        linewidth=0,
    )

    ax.plot(
        years,
        stored_below,
        color="#2ca02c",
        linestyle="--",
        linewidth=trajectory_linewidth,
        drawstyle="steps-post",
        label="CO₂ stored with CCS",
    )

    ax.plot(
        years,
        unabated_arr,
        color="black",
        linestyle="--",
        linewidth=trajectory_linewidth,
        drawstyle="steps-post",
        label="Unabated emissions",
    )

    ax.plot(
        years,
        net_arr,
        color="#d62728",
        linestyle=(0, (3, 1, 1, 1)),
        linewidth=trajectory_linewidth,
        drawstyle="steps-post",
        label="Net emissions (with CCS)",
    )

    if label_totals:
        # In presentation mode, drop the parenthetical qualifier from each
        # totals label — the trajectory colors and the surrounding slide
        # context already identify which curve each number describes.
        _presenting = _resolve_display_mode(display_mode) == "presentation"
        _totals_cfg = {
            "stored": {
                "value": float(scen_df["lifetime_mtco2_captured_with_ccs"].sum())
                / 1000.0,
                "color": "#2ca02c",
                "arr": stored_below,
                "label": (
                    "{val:.1f} GtCO₂ stored"
                    if _presenting
                    else "{val:.1f} GtCO₂ stored (with CCS)"
                ),
                "default_offset": (20, -22),
            },
            "unabated": {
                "value": float(scen_df["lifetime_unabated_mtco2e"].sum()) / 1000.0,
                "color": "black",
                "arr": unabated_arr,
                "label": (
                    "{val:.1f} GtCO₂e emitted"
                    if _presenting
                    else "{val:.1f} GtCO₂e emitted (unabated)"
                ),
                "default_offset": (20, 24),
            },
            "net": {
                "value": float(scen_df["lifetime_mtco2e"].sum()) / 1000.0,
                "color": "#d62728",
                "arr": net_arr,
                "label": (
                    "{val:.1f} GtCO₂e emitted"
                    if _presenting
                    else "{val:.1f} GtCO₂e emitted (with CCS)"
                ),
                "default_offset": (20, 24),
            },
        }
        _off = totals_offsets or {}
        _anchors = totals_anchor_years or {}
        _totals_fs = (
            totals_fontsize if totals_fontsize is not None else fs["annotation"]
        )
        for _key, _cfg in _totals_cfg.items():
            _arr = _cfg["arr"]
            if _key in _anchors:
                _match = np.where(years == int(_anchors[_key]))[0]
                _idx = int(_match[0]) if len(_match) else int(np.argmax(np.abs(_arr)))
            else:
                _idx = int(np.argmax(np.abs(_arr)))
            _xy = (int(years[_idx]), float(_arr[_idx]))
            _dx, _dy = _off.get(_key, _cfg["default_offset"])
            ax.annotate(
                _cfg["label"].format(val=_cfg["value"]),
                xy=_xy,
                xytext=(_dx, _dy),
                textcoords="offset points",
                color=_cfg["color"],
                fontsize=_totals_fs,
                fontweight=totals_fontweight,
                ha="left" if _dx >= 0 else "right",
                va="center",
                arrowprops=dict(
                    arrowstyle="-",
                    color=_cfg["color"],
                    linewidth=totals_leader_linewidth,
                    shrinkA=0,
                    shrinkB=2,
                    relpos=(0 if _dx >= 0 else 1, 0.5),
                ),
                zorder=10,
            )

    ax.set_xlabel("Year", fontsize=fs["axis_label"])
    ax.set_ylabel("MtCO₂e / yr", fontsize=fs["axis_label"], labelpad=-1)
    if title is None:
        title = "Full buildout: Texas facility emissions through time"
    if title_fontsize is None:
        title_fontsize = fs["title"]
    ax.set_title(
        title,
        fontsize=title_fontsize,
    )
    if legend_fontsize is None:
        legend_fontsize = fs["legend"]
    # In presentation mode, suppress the legend and label the three filled
    # regions in place. Positions come from the caller when provided; the
    # defaults match the standard Gulf-Coast chart layout. "Stored" and
    # "Uncaptured industrial" render as plain black Arial-Narrow text with
    # no halo; "Emissions from implementing CCS" renders in dark red with
    # a leader line pointing to the peak of the thin lifecycle band.
    _presenting_legend = _resolve_display_mode(display_mode) == "presentation"
    if _presenting_legend:
        _region_fs = (
            region_label_fontsize
            if region_label_fontsize is not None
            else fs["annotation"] + 2
        )

        _stored_xy = stored_label_xy if stored_label_xy is not None else (2032, -50)
        ax.text(
            _stored_xy[0],
            _stored_xy[1],
            "Stored with CCS",
            fontsize=_region_fs,
            fontfamily="Arial Narrow",
            fontweight="normal",
            color="black",
            ha="left",
            va="center",
            zorder=11,
        )

        _uncap_xy = (
            uncaptured_label_xy if uncaptured_label_xy is not None else (2032, 100)
        )
        ax.text(
            _uncap_xy[0],
            _uncap_xy[1],
            "Uncaptured industrial",
            fontsize=_region_fs,
            fontfamily="Arial Narrow",
            fontweight="normal",
            color="black",
            ha="left",
            va="center",
            zorder=11,
        )

        # Leader-line annotation for the thin red "Implementing CCS" band.
        # Anchor lands on the band in data coords; label text sits wherever
        # the caller asks (defaults to the upper-left of the plot).
        _ccs_anchor = (
            implementing_ccs_anchor_xy
            if implementing_ccs_anchor_xy is not None
            else (2032.5, 325)
        )
        _ccs_label_xy = (
            implementing_ccs_label_xy
            if implementing_ccs_label_xy is not None
            else (2026, 575)
        )
        # ha="left", va="top" keeps (x, y) at the UPPER-LEFT corner of the
        # text block, `multialignment="right"` right-justifies the three
        # lines inside it, and `relpos=(1.0, 0.0)` attaches the leader to
        # the LOWER-RIGHT corner so the arrow drops from under the last
        # line of text onto the red band at `_ccs_anchor`.
        _ccs_red = "#d62728"
        ax.annotate(
            "From\nimplementing\nCCS",
            xy=_ccs_anchor,
            xytext=_ccs_label_xy,
            textcoords="data",
            color=_ccs_red,
            fontsize=_region_fs,
            fontfamily="Arial Narrow",
            fontweight="normal",
            ha="left",
            va="top",
            multialignment="right",
            zorder=11,
            arrowprops=dict(
                arrowstyle="-",
                color=_ccs_red,
                linewidth=0.9,
                shrinkA=0,
                shrinkB=2,
                relpos=(1.0, 0.0),
            ),
        )
    else:
        ax.legend(
            fontsize=legend_fontsize,
            loc="lower right",
            frameon=True,
            ncol=legend_ncol,
        )
    ax.tick_params(labelsize=fs["tick"])
    ax.set_xlim(xlim if xlim is not None else (timeline_start, timeline_end))
    ax.set_ylim(ylim if ylim is not None else (stored_below.min() * 1.15, None))
    ax.grid(axis="both", alpha=0.5, zorder=-1)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=600)

    return fig, ax

def summarize_scenario_ids(
    scenarios_df: pd.DataFrame,
    projects_df: Optional[pd.DataFrame] = None,
    scenarios: Optional[Sequence[str]] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    solar_lcoe_usd_per_mwh: float = 60.0,
    ccs_active_years: int = 12,
    facility_lifetime_years: int = 30,
    base_year: Optional[int] = None,
) -> pd.DataFrame:
    """One row per scenario_id with the key cross-scenario summary stats.

    'unabated-<methane>' and 'existing-<methane>' are synthesized as
    pseudo-scenarios (no CCS, so no 45Q, no solar). 'unabated' aggregates
    all projects; 'existing' aggregates only retrofit-flagged projects.

    ``total_mtco2e`` is integrated year-by-year over each project's facility
    window (30 yr), inside the 12-yr CCS window using
    ``mtco2e_per_yr_emissions_including_ccs`` and otherwise
    ``unabated_mtco2e_per_yr``. ``start_year`` and ``end_year`` (both
    inclusive, both optional) clip which calendar years count toward the
    integral — e.g. ``start_year=2026, end_year=2050`` sums only year-blocks
    in [2026, 2050] that overlap each project's facility window.

    Parameters
    ----------
    scenarios_df
        Pipeline output with one row per (uid, scenario, methane_scenario).
    projects_df
        Optional. Used only if ``scenarios_df`` lacks ``project_start`` or
        ``retrofit_newbuild``.
    scenarios
        Optional subset of destination scenarios (incl. the pseudo-scenarios
        ``"unabated"`` and ``"existing"``). ``None`` returns every scenario.
    start_year, end_year
        Inclusive bounds on the calendar years included in ``total_mtco2e``.
        Both default to the natural facility window (``start_year=None`` →
        earliest ``_fac_start``; ``end_year=None`` → latest ``_fac_end - 1``).
    solar_lcoe_usd_per_mwh
        Echoed into every row. Does NOT recompute ``mtco2e_avoided_from_solar``
        — pass the same value the pipeline ran with for consistency.
    ccs_active_years, facility_lifetime_years, base_year
        Match :func:`plot_emissions_through_time` defaults. ``base_year``
        defaults to today's calendar year (for retrofit facility windows).

    Columns
    -------
    scenario_id, scenario, methane_scenario, total_mtco2e,
    annual_mtco2e_all_ccs_running, nominal_45q_musd, pv_45q_musd,
    twh_solar_from_45q, mtco2e_avoided_from_solar, gwp_ch4,
    methane_leak_rate, eor_fraction, total_million_bbls_oil,
    max_annual_million_bbls_oil_all_ccs_running,
    solar_lcoe_usd_per_mwh, start_year, end_year.
    """
    required = {
        "scenario_id",
        "scenario",
        "methane_scenario",
        "uid",
        "unabated_mtco2e_per_yr",
        "mtco2e_per_yr_emissions_including_ccs",
        "avg_million_usd_per_yr_45q",
        "pv_million_usd_credits_45q",
        "mwh_solar_from_45q",
        "mtco2e_avoided_from_solar",
        "gwp_ch4",
        "methane_leak_rate",
        "eor_fraction",
        "mtco2_per_yr_captured_with_ccs",
        "million_barrels_oil_per_yr",
    }
    missing_cols = required - set(scenarios_df.columns)
    if missing_cols:
        raise KeyError(f"scenarios_df missing required columns: {sorted(missing_cols)}")
    if base_year is None:
        base_year = datetime.date.today().year

    merge_cols = []
    if "project_start" not in scenarios_df.columns:
        merge_cols.append("project_start")
    if "retrofit_newbuild" not in scenarios_df.columns:
        merge_cols.append("retrofit_newbuild")
    if merge_cols:
        if projects_df is None:
            raise ValueError(
                f"scenarios_df is missing {merge_cols}; pass projects_df so they "
                "can be merged in."
            )
        scenarios_df = scenarios_df.merge(
            projects_df[["uid", *merge_cols]].drop_duplicates(subset="uid"),
            on="uid",
            how="left",
        )

    annotated = _annotate_facility_windows(
        scenarios_df.drop_duplicates(subset=["uid", "scenario_id"]),
        base_year=base_year,
        facility_lifetime_years=facility_lifetime_years,
    )

    def _year_counts(row) -> Tuple[int, int]:
        """Return (ccs_years_in_range, non_ccs_facility_years_in_range) for a row."""
        fs = int(row["_fac_start"])
        fe = int(row["_fac_end"])  # exclusive
        ps = int(row["project_start"])
        ccs_s = ps
        ccs_e = ps + ccs_active_years  # exclusive
        lo = start_year if start_year is not None else fs
        hi_excl = (end_year + 1) if end_year is not None else fe
        fac_lo = max(fs, lo)
        fac_hi = min(fe, hi_excl)
        fac_count = max(0, fac_hi - fac_lo)
        # CCS window must also be inside the clipped facility window
        ccs_lo = max(ccs_s, fac_lo)
        ccs_hi = min(ccs_e, fac_hi)
        ccs_count = max(0, ccs_hi - ccs_lo)
        return ccs_count, fac_count - ccs_count

    def _row(sub: pd.DataFrame, scenario: str, methane: str, is_pseudo: bool) -> dict:
        if len(sub) == 0:
            return None
        counts = sub.apply(_year_counts, axis=1, result_type="expand")
        ccs_yrs = counts[0].to_numpy()
        non_ccs_yrs = counts[1].to_numpy()
        unab = sub["unabated_mtco2e_per_yr"].to_numpy()
        net = sub["mtco2e_per_yr_emissions_including_ccs"].to_numpy()

        oil_per_yr = sub["million_barrels_oil_per_yr"].to_numpy()
        # The `switching*` scenarios keep CCS running for the full facility
        # lifetime (some first-period mix for years 1..D, 100% EOR afterwards).
        # Their lifetime totals live in the `lifetime_*` columns, which already
        # span D + (L−D) years. Use those directly instead of the D-year
        # closed form that applies to the standard destination scenarios.
        is_switching = scenario.startswith("switching")
        if is_pseudo:
            total = float((unab * (ccs_yrs + non_ccs_yrs)).sum())
            annual_all_ccs = float(unab.sum())
            nominal_45q = 0.0
            pv_45q = 0.0
            twh_solar = 0.0
            avoided = 0.0
            eor = 0.0
            total_oil = 0.0
            max_annual_oil = 0.0
        elif is_switching:
            total = float(sub["lifetime_mtco2e"].sum())
            annual_all_ccs = float(net.sum())
            nominal_45q = float(
                sub["avg_million_usd_per_yr_45q"].sum() * ccs_active_years
            )
            pv_45q = float(sub["pv_million_usd_credits_45q"].sum())
            twh_solar = float(sub["mwh_solar_from_45q"].sum() / 1e6)
            avoided = float(sub["mtco2e_avoided_from_solar"].sum())
            cap = float(sub["mtco2_per_yr_captured_with_ccs"].sum())
            eor = (
                float(
                    (sub["eor_fraction"] * sub["mtco2_per_yr_captured_with_ccs"]).sum()
                    / cap
                )
                if cap > 0
                else 0.0
            )
            total_oil = float(sub["lifetime_million_barrels_oil"].sum())
            max_annual_oil = float(oil_per_yr.sum())
        else:
            total = float((net * ccs_yrs + unab * non_ccs_yrs).sum())
            annual_all_ccs = float(net.sum())
            nominal_45q = float(
                sub["avg_million_usd_per_yr_45q"].sum() * ccs_active_years
            )
            pv_45q = float(sub["pv_million_usd_credits_45q"].sum())
            twh_solar = float(sub["mwh_solar_from_45q"].sum() / 1e6)
            avoided = float(sub["mtco2e_avoided_from_solar"].sum())
            cap = float(sub["mtco2_per_yr_captured_with_ccs"].sum())
            eor = (
                float(
                    (sub["eor_fraction"] * sub["mtco2_per_yr_captured_with_ccs"]).sum()
                    / cap
                )
                if cap > 0
                else 0.0
            )
            # Oil is produced only during each project's CCS window (clipped
            # to the summary [start_year, end_year] range).
            total_oil = float((oil_per_yr * ccs_yrs).sum())
            max_annual_oil = float(oil_per_yr.sum())
        return {
            "scenario_id": f"{scenario}-{methane}",
            "scenario": scenario,
            "methane_scenario": methane,
            "total_mtco2e": total,
            "annual_mtco2e_all_ccs_running": annual_all_ccs,
            "nominal_45q_musd": nominal_45q,
            "pv_45q_musd": pv_45q,
            "twh_solar_from_45q": twh_solar,
            "mtco2e_avoided_from_solar": avoided,
            "gwp_ch4": sub["gwp_ch4"].iloc[0],
            "methane_leak_rate": sub["methane_leak_rate"].iloc[0],
            "eor_fraction": eor,
            "total_million_bbls_oil": total_oil,
            "max_annual_million_bbls_oil_all_ccs_running": max_annual_oil,
            "solar_lcoe_usd_per_mwh": solar_lcoe_usd_per_mwh,
            "start_year": start_year,
            "end_year": end_year,
        }

    rows = []
    want = set(scenarios) if scenarios is not None else None
    for sid, sub in annotated.groupby("scenario_id"):
        scenario = sub["scenario"].iloc[0]
        methane = sub["methane_scenario"].iloc[0]
        if want is not None and scenario not in want:
            continue
        rows.append(_row(sub, scenario, methane, is_pseudo=False))

    want_unabated = want is None or "unabated" in want
    want_existing = want is None or "existing" in want
    if want_unabated or want_existing:
        for methane, sub_all in annotated.drop_duplicates(
            subset=["uid", "methane_scenario"]
        ).groupby("methane_scenario"):
            if want_unabated:
                rows.append(_row(sub_all, "unabated", methane, is_pseudo=True))
            if want_existing:
                retro = sub_all[
                    sub_all["retrofit_newbuild"].astype(str).str.strip().str.lower()
                    == "retrofit"
                ]
                r = _row(retro, "existing", methane, is_pseudo=True)
                if r is not None:
                    rows.append(r)

    rows = [r for r in rows if r is not None]
    out = pd.DataFrame(rows)
    return out.sort_values(["methane_scenario", "scenario"]).reset_index(drop=True)

def summarize_state_scenarios(
    scenarios_df: pd.DataFrame,
    methane_scenario: str = "likely",
    scenarios_order: Optional[List[str]] = None,
    solar_lcoe_usd_per_mwh: float = 60.0,
    project_duration_years: int = 12,
    facility_lifetime_years: int = 30,
    return_styled: bool = False,
    states: Optional[List[str]] = None,
):
    """Per-state × per-scenario summary of key CCS metrics.

    Rows: Unabated (no CCS) + CCS scenarios in `scenarios_order`.
    Columns: MultiIndex (metric, state), with Gulf Coast total → Texas →
    Louisiana within each metric. States are TX and LA; "Gulf Coast total"
    is TX + LA (summed per primitive, not per derived ratio).

    Metrics (left → right):
     10. Lifetime total CO2e (Mt)        — full facility-life cumulative
                                           emissions. For the CCS scenarios:
                                           `project_duration_years` × (annual
                                           CO2e with CCS) + (`facility_lifetime_years`
                                           − `project_duration_years`) × (annual
                                           unabated CO2e). For the Unabated row:
                                           `facility_lifetime_years` × annual
                                           unabated CO2e. Defaults are a 12-yr
                                           CCS operation window followed by
                                           18 unabated years, totalling a
                                           30-yr facility life.
      4. CO2 captured (Mt/yr)
      1. Total CO2e emissions (Mt/yr)    — annual rate; unabated row uses
                                           `unabated_mtco2e_per_yr`.
      8. tCO2e emitted per tCO2 injected — Σ `mtco2e_emissions_including_ccs`
                                           / Σ `mtco2_captured_with_ccs`. The
                                           full-system annual CO2e footprint
                                           (post-CCS, including everything —
                                           unabated residual, CCS lifecycle,
                                           EOR oil combustion) divided by the
                                           tCO2 injected that year. NOT the
                                           CCS lifecycle emissions factor ε
                                           (the pipeline column
                                           `tco2e_emitted_per_tco2_ccs`),
                                           which is a different, smaller
                                           quantity. Ratio of sums (not sum
                                           of ratios), so the Gulf Coast
                                           value is a capture-weighted
                                           average across TX + LA. Zero for
                                           the Unabated row (no capture,
                                           undefined denominator).
      7. Fraction warming from CH4       — (CO2e − CO2) / CO2e, computed from
                                           the raw CO2 and CO2e sums (ratio
                                           of sums, not sum of ratios). Unitless
                                           fraction in [0, 1].
      2. EOR-induced oil (kbbl/day)      — thousand barrels/day.
      3. PV oil profit (M$, @ $10/bbl)   — 12-yr PV at the pipeline's real
                                           discount rate (default 5.1%);
                                           profit = bbl × `profit_per_bbl_usd`
                                           ($10/bbl default, escalating with
                                           inflation per the pipeline config).
                                           The pipeline's
                                           `pv_million_usd_oil_revenue` field
                                           is really PV of profit, not gross
                                           revenue.
      5. PV 45Q (M$)                     — 12-yr PV, same discount rate.
      9. Nominal total 45Q (M$)          — undiscounted sum of annual 45Q
                                           payments over the project lifetime,
                                           with each year escalated at the
                                           pipeline's inflation rate (2.5%/yr
                                           starting 2027). Computed as
                                           `avg_million_usd_per_yr_45q` ×
                                           `project_duration_years` (default
                                           12), consistent with how the
                                           pipeline stores the per-year
                                           average.
      6. Equivalent solar (TWh)          — pipeline's `mwh_solar_from_45q`,
                                           summed and converted to TWh (= 10^6
                                           MWh). This is nominal-45Q divided
                                           by nominal-LCOE year-by-year across
                                           the 12-yr window, per the methods
                                           doc §1.5.7 (nominal/nominal, not PV).
     11. CO2e avoided by solar (Mt)      — pipeline's
                                           `mtco2e_avoided_from_solar`, summed
                                           across projects. Counterfactual
                                           lifetime GHG avoided if the 45Q $
                                           bought utility-scale solar MWh
                                           instead: MWh × bar_e_state / 1e6,
                                           where bar_e includes CO2 plus CH4
                                           leakage from the state's gas-fired
                                           generation. Zero for the Unabated
                                           row (no 45Q, no solar).
    """
    if scenarios_order is None:
        scenarios_order = [
            "all_gs",
            "baseline",
            "current_mix",
            "all_eor",
            "switching",
        ]

    _scen_display = {
        "all_gs": "100% Geologic storage",
        "baseline": "Announced destinations",
        "current_mix": "Current U.S. practice (92% EOR)",
        "all_eor": "100% EOR",
        "switching": "Switching (announced destinations → all EOR)",
    }
    _state_display = {"TX": "Texas", "LA": "Louisiana"}

    if states is None:
        states = ["TX", "LA"]
    unknown = [s for s in states if s not in _state_display]
    if unknown:
        raise ValueError(
            f"Unknown state(s) {unknown}; summarize_state_scenarios supports "
            f"{list(_state_display)}."
        )
    include_gulf_coast = len(states) > 1

    solar_label = f"Equivalent solar (TWh @ ${int(solar_lcoe_usd_per_mwh)}/MWh)"
    solar_avoided_label = "CO2e avoided by solar (Mt)"
    lifetime_label = (
        f"Lifetime total CO2e (Mt, {project_duration_years}+"
        f"{facility_lifetime_years - project_duration_years}yr)"
    )
    _metrics = [
        lifetime_label,
        "CO2 captured (Mt/yr)",
        "Total CO2e (Mt/yr)",
        "tCO2e emitted per tCO2 injected",
        "Fraction warming from CH4",
        "EOR oil (kbbl/day)",
        "PV oil profit (M$, @ $10/bbl)",
        "PV 45Q (M$)",
        "Nominal total 45Q (M$)",
        solar_label,
        solar_avoided_label,
    ]

    df = scenarios_df[scenarios_df["methane_scenario"] == methane_scenario].copy()
    df = df[df["project_state"].isin(states)]

    # Per-project unabated is identical across CCS scenarios; take one row per
    # uid so the state totals are not inflated by the scenario join cardinality.
    if "uid" in df.columns:
        unab_src = df.drop_duplicates(subset="uid")
    else:
        unab_src = df[df["scenario"] == scenarios_order[0]]
    unab_co2e_by_state = unab_src.groupby("project_state")[
        "unabated_mtco2e_per_yr"
    ].sum()
    unab_co2_by_state = unab_src.groupby("project_state")["unabated_mtco2_per_yr"].sum()

    # Retrofit-only unabated totals: used for the "Existing facilities only
    # (no buildout; no CCS)" counterfactual row. Retrofit facilities already
    # exist today; this row captures what their 30-year unabated emissions
    # would be if no CCS and no new facilities are built.
    retrofit_src = (
        unab_src[
            unab_src["retrofit_newbuild"].astype(str).str.strip().str.lower()
            == "retrofit"
        ]
        if "retrofit_newbuild" in unab_src.columns
        else unab_src.iloc[0:0]
    )
    retrofit_co2e_by_state = retrofit_src.groupby("project_state")[
        "unabated_mtco2e_per_yr"
    ].sum()
    retrofit_co2_by_state = retrofit_src.groupby("project_state")[
        "unabated_mtco2_per_yr"
    ].sum()

    def _frac_ch4(co2e: float, co2: float) -> float:
        return ((co2e - co2) / co2e) if co2e > 0 else 0.0

    def _total_ef(co2e_mt: float, captured_mt: float) -> float:
        return (co2e_mt / captured_mt) if captured_mt > 0 else 0.0

    unabated_tail_years = facility_lifetime_years - project_duration_years

    def _derive(
        prims: Dict[str, float],
        mode: str = "ccs",
    ) -> Dict[str, float]:
        """Return the metric dict for one (scenario, state) cell.

        `mode` picks the lifetime-emissions formula:
          * "ccs"       — `D × co2e_mt + (L−D) × unabated_co2e_mt` (the
                          standard destination scenarios).
          * "unabated"  — `L × unabated_co2e_mt` (no CCS ever).
          * "switching" — `prims["lifetime_mt"]` read directly from the
                          pipeline's `lifetime_mtco2e` column, which carries
                          the exact `D × baseline_per_yr + (L−D) × all_eor_per_yr`
                          sum for this scenario.
        """
        pv_45q = prims["pv_45q_m"]
        nominal_45q = prims["avg_annual_45q_m"] * project_duration_years
        if mode == "unabated":
            lifetime_mt = facility_lifetime_years * prims["unabated_co2e_mt"]
        elif mode == "switching":
            lifetime_mt = prims["lifetime_mt"]
        else:
            lifetime_mt = (
                project_duration_years * prims["co2e_mt"]
                + unabated_tail_years * prims["unabated_co2e_mt"]
            )
        return {
            lifetime_label: lifetime_mt,
            "CO2 captured (Mt/yr)": prims["captured_mt"],
            "Total CO2e (Mt/yr)": prims["co2e_mt"],
            "tCO2e emitted per tCO2 injected": _total_ef(
                prims["co2e_mt"], prims["captured_mt"]
            ),
            "Fraction warming from CH4": _frac_ch4(prims["co2e_mt"], prims["co2_mt"]),
            "EOR oil (kbbl/day)": prims["oil_bbl_day"],
            "PV oil profit (M$, @ $10/bbl)": prims["pv_oil_m"],
            "PV 45Q (M$)": pv_45q,
            "Nominal total 45Q (M$)": nominal_45q,
            # pipeline's mwh_solar_from_45q is Σ_y nominal_45Q(y) / nominal_LCOE(y)
            # across the 12-yr CCS window (methods doc §1.5.7). ÷ 1e6 → TWh.
            solar_label: prims["mwh_solar"] / 1e6,
            # 12-yr lifetime GHG avoided by substituting that solar MWh for grid
            # generation at the state's bar_e (CO2 + CH4 leakage from gas plants).
            solar_avoided_label: prims["mtco2e_avoided_solar"],
        }

    # Build per-(scenario, state) primitives, then derive per-metric values
    # for the row-dict. For Gulf Coast total we sum TX+LA primitives first
    # and then derive, so the % CH4 ratio uses the full-fleet CO2 and CO2e.
    rows: List[Dict[str, Dict[str, float]]] = []
    row_index: List[str] = []

    def _no_ccs_prims(co2e_by_state, co2_by_state) -> Dict[str, Dict[str, float]]:
        """Build the primitives dict for a no-CCS row (Unabated or
        Existing facilities only)."""
        return {
            state: {
                "captured_mt": 0.0,
                "co2e_mt": float(co2e_by_state.get(state, 0.0)),
                "co2_mt": float(co2_by_state.get(state, 0.0)),
                "unabated_co2e_mt": float(co2e_by_state.get(state, 0.0)),
                "oil_bbl_day": 0.0,
                "pv_oil_m": 0.0,
                "pv_45q_m": 0.0,
                "avg_annual_45q_m": 0.0,
                "mwh_solar": 0.0,
                "mtco2e_avoided_solar": 0.0,
                "lifetime_mt": 0.0,
            }
            for state in states
        }

    def _add_row(label: str, prims_by_state: Dict[str, Dict[str, float]], mode: str):
        row_index.append(label)
        row = {state: _derive(prims_by_state[state], mode=mode) for state in states}
        if include_gulf_coast:
            row["GC"] = _derive(
                {
                    k: sum(prims_by_state[s][k] for s in states)
                    for k in prims_by_state[states[0]]
                },
                mode=mode,
            )
        rows.append(row)

    # Unabated row: use unabated CO2 and CO2e; all CCS-side metrics are 0.
    _add_row(
        "Unabated (no CCS)",
        _no_ccs_prims(unab_co2e_by_state, unab_co2_by_state),
        mode="unabated",
    )

    # Existing-facilities-only row: identical to Unabated but restricted to
    # retrofit facilities. No CCS, no new builds, no 45Q, no oil — just the
    # 30-year unabated emissions from the currently-operating plants.
    _add_row(
        "Existing facilities only (no buildout; no CCS)",
        _no_ccs_prims(retrofit_co2e_by_state, retrofit_co2_by_state),
        mode="unabated",
    )

    for scen in scenarios_order:
        scen_df = df[df["scenario"] == scen]
        # Switching keeps CCS running for the full facility lifetime, so its
        # oil barrels flow for L years rather than D. The pipeline already
        # stores lifetime-semantics `pv_million_usd_oil_revenue` and
        # `nominal_million_usd_oil_revenue` on switching rows (identical to
        # the dedicated `lifetime_*` columns), so no PV-column switch is
        # needed here. The kbbl/day rate still needs a wider denominator
        # for switching so that `rate × days = total barrels`. Matches
        # both switching variants (`switching` and `switching_from_current`).
        is_switching = scen.startswith("switching")
        oil_rate_col = (
            "lifetime_million_barrels_oil"
            if is_switching
            else "million_barrels_oil_per_yr"
        )
        oil_rate_denom_days = facility_lifetime_years * 365.0 if is_switching else 365.0
        prims_by_state: Dict[str, Dict[str, float]] = {}
        for state in states:
            sdf = scen_df[scen_df["project_state"] == state]
            prims_by_state[state] = {
                "captured_mt": sdf["mtco2_per_yr_captured_with_ccs"].sum(),
                "co2e_mt": sdf["mtco2e_per_yr_emissions_including_ccs"].sum(),
                "co2_mt": sdf["mtco2_per_yr_emissions_including_ccs"].sum(),
                "unabated_co2e_mt": float(unab_co2e_by_state.get(state, 0.0)),
                "oil_bbl_day": sdf[oil_rate_col].sum() * 1e3 / oil_rate_denom_days,
                "pv_oil_m": sdf["pv_million_usd_oil_revenue"].sum(),
                "pv_45q_m": sdf["pv_million_usd_credits_45q"].sum(),
                "avg_annual_45q_m": sdf["avg_million_usd_per_yr_45q"].sum(),
                "mwh_solar": sdf["mwh_solar_from_45q"].sum(),
                "mtco2e_avoided_solar": (
                    sdf["mtco2e_avoided_from_solar"].sum()
                    if "mtco2e_avoided_from_solar" in sdf.columns
                    else 0.0
                ),
                "lifetime_mt": (
                    float(sdf["lifetime_mtco2e"].sum())
                    if is_switching and "lifetime_mtco2e" in sdf.columns
                    else 0.0
                ),
            }
        _add_row(
            _scen_display.get(scen, scen),
            prims_by_state,
            mode="switching" if is_switching else "ccs",
        )

    # Flatten into a MultiIndex DataFrame: columns = (metric, state). Gulf Coast
    # total comes first (when multiple states are requested); state columns
    # follow in the order given.
    flat: Dict[Tuple[str, str], List[float]] = {}
    for m in _metrics:
        if include_gulf_coast:
            flat[(m, "Gulf Coast total")] = [r["GC"][m] for r in rows]
        for state in states:
            flat[(m, _state_display[state])] = [r[state][m] for r in rows]

    result = pd.DataFrame(
        flat,
        index=pd.Index(row_index, name="Scenario"),
    )
    result.columns = pd.MultiIndex.from_tuples(
        result.columns, names=["Metric", "State"]
    )

    if not return_styled:
        return result

    _fmt_by_metric = {
        lifetime_label: "{:,.0f}",
        "CO2 captured (Mt/yr)": "{:,.2f}",
        "Total CO2e (Mt/yr)": "{:,.1f}",
        "tCO2e emitted per tCO2 injected": "{:,.2f}",
        "Fraction warming from CH4": "{:,.3f}",
        "EOR oil (kbbl/day)": "{:,.1f}",
        "PV oil profit (M$, @ $10/bbl)": "{:,.0f}",
        "PV 45Q (M$)": "{:,.0f}",
        "Nominal total 45Q (M$)": "{:,.0f}",
        solar_label: "{:,.1f}",
        solar_avoided_label: "{:,.1f}",
    }
    styled = result.style.format(
        {col: _fmt_by_metric[col[0]] for col in result.columns}
    ).set_caption(
        f"CCS scenario summary — methane_scenario='{methane_scenario}', "
        f"PV at pipeline discount rate over 12-yr project lifetime; "
        f"solar from pipeline nominal-45Q ÷ nominal-LCOE year-by-year "
        f"(base LCOE ${int(solar_lcoe_usd_per_mwh)}/MWh)."
    )
    return styled


_DEFAULT_INDUSTRIES = (
    "hydrogen",
    "ammonia",
    "iron_steel",
    "methanol",
    "ethanol",
    "lng",
    "ngp",
    "power_ng_data_center",
    "refined_fuels",
    "petrochemicals",
    "power_coal",
)

_PRODUCT_ALIASES = {
    "power_ng_data_center": ("power_ng_data_center", "power_ng_new_build"),
}

_NAME_REPLACEMENTS = {
    "ngp": "Natural gas processing",
    "iron_steel": "Iron & steel",
    "power_coal": "Power (coal)",
    "lng": "LNG",
    "petrochemicals": "Petrochemicals",
    "power_ng_data_center": "Gas power/data centers",
    "dac": "DAC",
    "refined_fuels": "Refined fuels",
    "hydrogen": "Hydrogen",
    "ammonia": "Ammonia",
    "methanol": "Methanol",
    "ethanol": "Ethanol",
}

COLOR_CAPTURED = "#2E7D32"  # dark green — CO2 that CCS removes

COLOR_UNCAPTURED_CO2 = "#9E9E9E"  # gray — CO2 that escapes CCS

COLOR_METHANE = "#F57C00"  # orange — CH4 as CO2e

COLOR_GS = "#2E7D32"  # dark green — saline storage

COLOR_EOR = "#C62828"  # red — EOR

def _compute_panel_data(
    ie_df: pd.DataFrame,
    ccs_df: pd.DataFrame,
    scen_df: pd.DataFrame,
    methane_scenario: str,
    industries: list,
    states: tuple,
):
    """Return (panel_a_df, panel_b_df) indexed by simple_product_name.

    panel_a columns: captured_share, uncaptured_co2_share, methane_share,
                     buildout_share (all fractions of total CO2e footprint;
                     first three sum to 1).
    panel_b columns: gs_net, eor_net (net tCO2e emitted per tCO2 stored).
    Both are sorted by captured_share DESCENDING, so when used as y-values
    in a horizontal bar chart (where y=0 is the bottom) the LOWEST-capture
    industry (e.g. LNG) sits at the TOP and the highest-capture (e.g. coal)
    sits at the bottom.
    """
    # Defensive check: the requested methane_scenario must exist in every
    # input dataframe. A common silent-failure mode is upstream code that
    # pre-filters one of the three dfs to a single scenario — the compute
    # path then returns an empty or stale frame.
    for name, df in (
        ("industrial_emissions_df", ie_df),
        ("ccs_emissions_df", ccs_df),
        ("scenarios_df", scen_df),
    ):
        present = set(df["methane_scenario"].dropna().unique())
        if methane_scenario not in present:
            raise ValueError(
                f"methane_scenario={methane_scenario!r} not found in {name}. "
                f"Available: {sorted(present)}. Did you pre-filter upstream?"
            )

    # ── Panel (a) composition ───────────────────────────────────────────
    ie = (
        ie_df.query(
            "methane_scenario == @methane_scenario "
            "and simple_product_name in @industries "
            "and project_state in @states"
        )
        .drop_duplicates(subset="simple_product_name", keep="first")
        .set_index("simple_product_name")
    )
    captured = ie["adjusted_capture_fraction_tco2e"]
    co2_share = ie["total_tco2_per_unit_product"] / ie["total_tco2e_per_unit_product"]
    uncaptured = co2_share - captured
    methane = 1.0 - co2_share

    # ── Buildout share: each industry's slice of the Gulf-Coast total
    # unabated emissions (scenario-invariant per facility; dedupe on uid).
    # Industries with aliases in _PRODUCT_ALIASES sum their alias-list
    # products into the displayed row (e.g. power_ng_data_center +
    # power_ng_new_build → single "Gas power & data centers" row).
    buildout_src = scen_df.query(
        "methane_scenario == @methane_scenario and project_state in @states"
    ).drop_duplicates(subset=["uid", "methane_scenario"])
    by_product = buildout_src.groupby("product")["unabated_mtco2e_per_yr"].sum()
    total = float(buildout_src["unabated_mtco2e_per_yr"].sum())

    def _numerator(industry: str) -> float:
        sources = _PRODUCT_ALIASES.get(industry, (industry,))
        return float(by_product.reindex(list(sources)).fillna(0).sum())

    buildout_share = pd.Series(
        {ind: (_numerator(ind) / total if total > 0 else 0.0) for ind in captured.index}
    )

    panel_a = pd.DataFrame(
        {
            "captured_share": captured,
            "uncaptured_co2_share": uncaptured,
            "methane_share": methane,
            "buildout_share": buildout_share.reindex(captured.index).fillna(0.0),
        }
    ).sort_values("captured_share", ascending=False)

    # ── Panel (b) net-per-tonne-stored ──────────────────────────────────
    ccs = ccs_df.query(
        "methane_scenario == @methane_scenario "
        "and co2_source in @industries "
        "and project_state in @states"
    ).drop_duplicates(subset=["co2_source", "co2_destination"], keep="first")
    gs = ccs[ccs["co2_destination"] == "gs"].set_index("co2_source")[
        "net_tco2e_emitted_per_tco2_ccs"
    ]
    eor = ccs[ccs["co2_destination"] == "eor"].set_index("co2_source")[
        "net_tco2e_emitted_per_tco2_ccs"
    ]
    panel_b = pd.DataFrame({"gs_net": gs, "eor_net": eor}).reindex(panel_a.index)

    return panel_a, panel_b

def plot_two_panel(
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    name_replacements: dict = None,
    figsize: tuple = (16.0, 7.0),
):
    """Render the two-panel figure. Industries share a y-axis across panels."""
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.9, 1.0], "wspace": 0.22},
        sharey=True,
    )

    industries = panel_a.index.tolist()
    y_pos = np.arange(len(industries))
    display_names = [(name_replacements or {}).get(ind, ind) for ind in industries]

    # ────────── Panel (a): stacked horizontal bars 0–100% ──────────
    bar_h = 0.72
    captured_pct = panel_a["captured_share"].to_numpy() * 100
    uncaptured_pct = panel_a["uncaptured_co2_share"].to_numpy() * 100
    methane_pct = panel_a["methane_share"].to_numpy() * 100

    ax_a.barh(
        y_pos,
        captured_pct,
        bar_h,
        left=0,
        color=COLOR_CAPTURED,
        label="Captured by CCS",
    )
    ax_a.barh(
        y_pos,
        uncaptured_pct,
        bar_h,
        left=captured_pct,
        facecolor=COLOR_UNCAPTURED_CO2,
        edgecolor="#6a6a6a",
        linewidth=0.0,
        hatch="///",
        label="Uncaptured CO$_2$",
    )
    ax_a.barh(
        y_pos,
        methane_pct,
        bar_h,
        left=captured_pct + uncaptured_pct,
        color=COLOR_METHANE,
        label="Methane (as CO$_2$e)",
    )

    # Numeric labels inside each segment wide enough to read. The threshold
    # is tuned so LNG's 7% captured headline number still renders inside the
    # bar.
    MIN_SEGMENT_PCT_TO_LABEL = 6.0
    for i, (cap, unc, met) in enumerate(zip(captured_pct, uncaptured_pct, methane_pct)):
        if cap >= MIN_SEGMENT_PCT_TO_LABEL:
            ax_a.text(
                cap / 2,
                i,
                f"{cap:.0f}%",
                ha="center",
                va="center",
                fontsize=11,
                color="white",
                fontweight="bold",
            )
        if unc >= MIN_SEGMENT_PCT_TO_LABEL:
            ax_a.text(
                cap + unc / 2,
                i,
                f"{unc:.0f}%",
                ha="center",
                va="center",
                fontsize=11,
                color="white",
                fontweight="bold",
            )
        if met >= MIN_SEGMENT_PCT_TO_LABEL:
            ax_a.text(
                cap + unc + met / 2,
                i,
                f"{met:.0f}%",
                ha="center",
                va="center",
                fontsize=11,
                color="white",
                fontweight="bold",
            )

    # Right-margin: industry's share of total buildout emissions. Sub-1%
    # contributions render as "<1%" instead of rounding to "0%".
    def _fmt_buildout_pct(frac: float) -> str:
        pct = frac * 100
        if frac > 0 and pct < 1.0:
            return "<1%"
        return f"{pct:.0f}%"

    for i, ind in enumerate(industries):
        frac = float(panel_a.loc[ind, "buildout_share"])
        ax_a.text(
            101.5,
            i,
            _fmt_buildout_pct(frac),
            fontsize=13,
            color="black",
            va="center",
            ha="left",
        )
    ax_a.text(
        101.5,
        len(industries) - 1 + 0.5,
        "Share of\nbuildout",
        fontsize=13,
        color="black",
        va="bottom",
        ha="left",
    )

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(display_names, fontsize=14)
    ax_a.set_xlim(0, 100)
    ax_a.set_xlabel(
        "Share of facility emissions before CCS (%)",
        fontsize=15,
    )
    ax_a.set_title(
        "(a) Facility emissions that CCS can capture",
        fontsize=16,
        loc="left",
        pad=10,
    )
    ax_a.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, -0.22),
        frameon=False,
        fontsize=16,
        ncol=3,
    )
    ax_a.grid(True, axis="x", alpha=0.3)
    ax_a.tick_params(axis="x", labelsize=12)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # ────────── Panel (b): dot-and-line plot ──────────
    for i, ind in enumerate(industries):
        gs = panel_b.loc[ind, "gs_net"]
        eor = panel_b.loc[ind, "eor_net"]
        if pd.isna(gs) or pd.isna(eor):
            continue
        ax_b.plot(
            [gs, eor],
            [i, i],
            color="dimgray",
            linewidth=1.5,
            zorder=2,
        )
        ax_b.scatter(
            [gs],
            [i],
            s=110,
            marker="o",
            color=COLOR_GS,
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        ax_b.scatter(
            [eor],
            [i],
            s=110,
            marker="o",
            color=COLOR_EOR,
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )

    ax_b.axvline(0, color="black", linewidth=1.0, zorder=1)
    ax_b.set_xlabel(
        "Net tCO$_2$e emitted per tCO$_2$ stored",
        fontsize=15,
    )
    ax_b.set_title(
        "(b) How CCS alters emissions",
        fontsize=16,
        loc="left",
        pad=10,
    )
    ax_b.tick_params(axis="x", labelsize=12)
    ax_b.grid(True, axis="x", alpha=0.3)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    gs_handle = mlines.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=COLOR_GS,
        markeredgecolor="black",
        markersize=12,
        label="Saline storage",
    )
    eor_handle = mlines.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=COLOR_EOR,
        markeredgecolor="black",
        markersize=12,
        label="EOR",
    )
    ax_b.legend(
        handles=[gs_handle, eor_handle],
        loc="lower right",
        bbox_to_anchor=(1.0, -0.22),
        frameon=False,
        fontsize=16,
        ncol=2,
    )

    fig.subplots_adjust(left=0.18, right=0.95, bottom=0.14, top=0.92)

    # Faint cross-axis connectors so readers can trace each industry's bar in
    # panel (a) to its corresponding dot-pair row in panel (b). Drawn AFTER
    # subplots_adjust so each panel's final data limits are settled.
    xb_left = ax_b.get_xlim()[0]
    for i in range(len(industries)):
        con = ConnectionPatch(
            xyA=(100, i),
            coordsA="data",
            axesA=ax_a,
            xyB=(xb_left, i),
            coordsB="data",
            axesB=ax_b,
            color="lightgray",
            linewidth=0.6,
            linestyle=":",
            zorder=0,
        )
        fig.add_artist(con)

    return fig, (ax_a, ax_b)


_DEFAULT_SCENARIOS = [
    "baseline",
    "current_mix",
    "all_gs",
    "all_eor",
    "switching",
    "switching_from_current",
]

_PRETTY = {
    "existing_only": "Existing\nfacilities only\n(no further\nbuildout,\nno CCS)",
    "unabated": "Full buildout,\nunabated\n(no CCS)",
    "all_gs": "100% Geologic\nstorage\n('best case'\nclimate impact)",
    "baseline": "Announced\ndestinations\n(17% EOR,\n83% GS)",
    "current_mix": "Current U.S.\npractice\n(92% EOR,\n8% GS)",
    "all_eor": "100% EOR\n('worst case'\nwhile 45Q\nis paid)",
    "switching": "Announced\ndestinations,\nthen all EOR\nafter 12-year\n45Q",
    "switching_from_current": "Current U.S.\npractice,\nthen all EOR\nafter 12-year\n45Q",
    "equivalent_solar": "Equivalent\nsolar\n(45Q dollars\n→ utility PV)",
}

_CCS_COLOR_20, _CCS_COLOR_100 = "#C44536", "#F2A65A"
_SOLAR_COLOR_20, _SOLAR_COLOR_100 = "#2E5E4E", "#7FB069"

def _delta_gt(df: pd.DataFrame, scen: str, methane_scen: str) -> float:
    sub = df.loc[(df["scenario"] == scen) & (df["methane_scenario"] == methane_scen)]
    return (
        sub["lifetime_mtco2e"].sum() - sub["lifetime_unabated_mtco2e"].sum()
    ) / 1000.0

def _solar_gt(
    df: pd.DataFrame, methane_scen: str, unabated_source_scenario: str
) -> float:
    sub = df.loc[
        (df["scenario"] == unabated_source_scenario)
        & (df["methane_scenario"] == methane_scen)
    ]
    if "mtco2e_avoided_from_solar" not in sub.columns:
        return float("nan")
    # Negated: avoided emissions sit below the zero line.
    return -sub["mtco2e_avoided_from_solar"].sum() / 1000.0

def plot_lifetime_delta_vs_unabated(
    scenarios_df: pd.DataFrame,
    scenarios: list | None = None,
    methane_scen_gwp20: str = "edf20",
    methane_scen_gwp100: str = "edf100",
    unabated_source_scenario: str = "baseline",
    states: tuple = ("TX", "LA"),
    include_solar: bool = True,
    gwp_horizons: tuple = (20, 100),
    scenario_labels: dict | None = None,
    savepath: str | None = None,
    dpi: int = 600,
    figsize: tuple = (18, 10),
    bar_label_fontsize: float = 18,
    tick_fontsize: float = 20,
    legend_fontsize: float = 22,
    ylabel_fontsize: float = 22,
) -> plt.Figure:
    """Grouped bar chart of lifetime GHG delta vs. a no-CCS buildout.

    Scenarios are sorted ascending by their GWP20 delta (lowest-impact on the
    left). Use `gwp_horizons` to control which horizons are shown:

    * `(20, 100)` (default): paired bars per scenario — GWP20 on the left,
      GWP100 on the right.
    * `(20,)`: single bar per scenario, GWP20 only.
    * `(100,)`: single bar per scenario, GWP100 only.

    `scenario_labels` optionally overrides the x-tick label for one or more
    scenarios (keys match scenario names, e.g. ``"baseline"``,
    ``"current_mix"``, ``"switching"``, ``"equivalent_solar"``). Unmatched
    keys are silently ignored; scenarios not in the dict keep their
    `_PRETTY` default.
    """
    if scenarios is None:
        scenarios = list(_DEFAULT_SCENARIOS)

    # Normalise and validate the horizons selection.
    _h = tuple(int(h) for h in gwp_horizons)
    for h in _h:
        if h not in (20, 100):
            raise ValueError(f"gwp_horizons entries must be 20 or 100, got {h!r}")
    if not _h:
        raise ValueError("gwp_horizons must contain at least one of (20, 100)")

    _methane_for = {20: methane_scen_gwp20, 100: methane_scen_gwp100}
    _ccs_color_for = {20: _CCS_COLOR_20, 100: _CCS_COLOR_100}
    _solar_color_for = {20: _SOLAR_COLOR_20, 100: _SOLAR_COLOR_100}
    _label_for = {20: r"GWP$_{20}$", 100: r"GWP$_{100}$"}

    base = scenarios_df.loc[
        scenarios_df["project_state"].isin(states)
        & scenarios_df["methane_scenario"].isin([_methane_for[h] for h in _h])
    ]

    # Always compute both horizons so sort order (by GWP20 delta) stays
    # stable regardless of which horizons the caller displays.
    items = []
    for s in scenarios:
        items.append(
            {
                "key": s,
                "d20": _delta_gt(base, s, methane_scen_gwp20),
                "d100": _delta_gt(base, s, methane_scen_gwp100),
                "is_solar": False,
            }
        )
    if include_solar:
        items.append(
            {
                "key": "equivalent_solar",
                "d20": _solar_gt(base, methane_scen_gwp20, unabated_source_scenario),
                "d100": _solar_gt(base, methane_scen_gwp100, unabated_source_scenario),
                "is_solar": True,
            }
        )
    # Sort by the lowest-horizon value present (GWP20 if included, else GWP100)
    # so single-horizon charts order by their own displayed deltas.
    _sort_col = "d20" if 20 in _h else "d100"
    g = pd.DataFrame(items)
    g = g.sort_values(_sort_col, ascending=True, kind="stable").reset_index(drop=True)

    n = len(g)
    n_h = len(_h)
    # Paired bars: narrow + offset. Single horizon: a wider centered bar.
    w = 0.38 if n_h == 2 else 0.6
    x = np.arange(n, dtype=float)

    fig, ax = plt.subplots(figsize=figsize)

    bars_by_h = {}
    for i, h in enumerate(_h):
        col = f"d{h}"
        color = np.where(g["is_solar"], _solar_color_for[h], _ccs_color_for[h])
        # Centre offset for paired bars (−w/2, +w/2); zero when single.
        offset = (i - (n_h - 1) / 2) * w if n_h > 1 else 0.0
        bars_by_h[h] = ax.bar(
            x + offset,
            g[col],
            w,
            color=color,
            edgecolor="black",
            linewidth=0.6,
        )

    ax.axhline(0, color="black", linewidth=1.0)

    all_vals = np.concatenate([g[f"d{h}"].to_numpy() for h in _h])
    off = 0.015 * np.nanmax(np.abs(all_vals))
    for h in _h:
        col = f"d{h}"
        for rect, v in zip(bars_by_h[h], g[col]):
            if np.isnan(v):
                continue
            va = "bottom" if v >= 0 else "top"
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                v + (off if v >= 0 else -off),
                f"{v:+.2f}",
                ha="center",
                va=va,
                fontsize=bar_label_fontsize,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            )

    ax.set_xticks(x)
    _label_map = {**_PRETTY, **(scenario_labels or {})}
    ax.set_xticklabels(
        [_label_map.get(k, k) for k in g["key"]],
        fontsize=tick_fontsize,
    )
    ax.tick_params(axis="y", labelsize=tick_fontsize)

    # Legend: one CCS entry (and one solar entry, if shown) per displayed
    # horizon. For a single-horizon chart the legend collapses to one or two
    # swatches since there is nothing to disambiguate.
    legend_handles = []
    for h in _h:
        legend_handles.append(
            Patch(
                facecolor=_ccs_color_for[h],
                edgecolor="black",
                label=(f"CCS scenario, {_label_for[h]}" if n_h > 1 else "CCS scenario"),
            )
        )
    if include_solar:
        for h in _h:
            legend_handles.append(
                Patch(
                    facecolor=_solar_color_for[h],
                    edgecolor="black",
                    label=(
                        f"Equivalent solar, {_label_for[h]}"
                        if n_h > 1
                        else "Equivalent solar"
                    ),
                )
            )

    _ylab = r"Lifetime $\Delta$ emissions (GtCO$_2$e"
    if n_h == 1:
        _ylab += f", {_label_for[_h[0]]})"
    else:
        _ylab += ")"
    ax.set_ylabel(_ylab, fontsize=ylabel_fontsize)
    ax.legend(
        handles=legend_handles, loc="best", frameon=False, fontsize=legend_fontsize
    )
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    return fig


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SCENARIOS_CSV = DATA_DIR / "ccs_scenarios_out.csv"

GWP20_VALUES = {82.5}

GWP100_VALUES = {25.0, 29.8}

LINESTYLE_BY_GWP = {"GWP20": "-", "GWP100": "--"}

_DESTINATION_COLORS = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#ff7f0e",  # orange
    "#17becf",  # cyan
    "#8c564b",  # brown
]

_DESTINATION_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]

_COUNTERFACTUAL_STYLE = {
    "unabated": {"marker": "^", "color": "#4D4D4D"},  # dark gray triangle
    "existing_only": {"marker": "D", "color": "#2CA02C"},  # green diamond
}

BASE_CASE_LEAK_RATE = 0.023

_DESTINATION_LABEL_OVERRIDES = {
    "baseline": r"Announced CO$_2$ destinations (83% GS; 17% EOR)",
    "current_mix": "Current U.S. practice (8% GS; 92% EOR)",
    "all_eor": "100% EOR (worst case while 45Q is paid)",
    "all_gs": "100% geologic storage (best-case climate impact)",
    "switching": "Announced destinations, then all EOR after 45Q",
    "switching_from_current": "Current U.S. practice, then all EOR after 45Q",
    "unabated": "Full buildout, unabated (no CCS)",
    "existing_only": "Existing facilities only (no buildout; no CCS)",
}

_GWP_LEGEND_LABELS = {"GWP20": r"GWP$_{20}$", "GWP100": r"GWP$_{100}$"}

def _gwp_group(gwp: float) -> str:
    if gwp in GWP20_VALUES:
        return "GWP20"
    if gwp in GWP100_VALUES:
        return "GWP100"
    return f"GWP (gwp_ch4={gwp:g})"

def compute_buildout_lifetime(
    scen_df: pd.DataFrame,
    methane_scenarios: list,
    destination_scenarios: list,
    states: tuple,
    include_counterfactuals: bool = True,
) -> pd.DataFrame:
    """One row per (destination scenario × methane scenario) with the
    state-summed net lifetime emissions.

    When `include_counterfactuals` is True, two no-CCS reference lines are
    appended under synthetic scenario names:
      * ``unabated``       — every facility operates unabated for its full
                             lifetime (sum of ``lifetime_unabated_mtco2e``
                             across all facilities in the state set).
      * ``existing_only``  — only retrofit (currently-operating) facilities
                             exist and they run unabated for their lifetime
                             (same sum restricted to
                             ``retrofit_newbuild == 'retrofit'``).
    Both counterfactuals are independent of destination mix (no CCS → no
    EOR/GS distinction).
    """
    missing_m = sorted(
        set(methane_scenarios) - set(scen_df["methane_scenario"].unique())
    )
    if missing_m:
        raise click.ClickException(
            f"Methane scenarios not present in {SCENARIOS_CSV.name}: {missing_m}"
        )
    missing_d = sorted(set(destination_scenarios) - set(scen_df["scenario"].unique()))
    if missing_d:
        raise click.ClickException(
            f"Destination scenarios not present: {missing_d}. "
            f"Available: {sorted(scen_df['scenario'].unique())}"
        )

    sub = scen_df.query(
        "methane_scenario in @methane_scenarios "
        "and scenario in @destination_scenarios "
        "and project_state in @states"
    )
    agg = (
        sub.groupby(
            ["scenario", "methane_scenario", "gwp_ch4", "methane_leak_rate"],
            as_index=False,
        )["lifetime_mtco2e"]
        .sum()
        .sort_values(["scenario", "gwp_ch4", "methane_leak_rate"])
        .reset_index(drop=True)
    )

    if include_counterfactuals:
        # `lifetime_unabated_mtco2e` is scenario-invariant per facility, so
        # dedupe on (uid, methane_scenario) before summing.
        per_facility = scen_df.query(
            "methane_scenario in @methane_scenarios and project_state in @states"
        ).drop_duplicates(subset=["uid", "methane_scenario"])

        unabated = (
            per_facility.groupby(
                ["methane_scenario", "gwp_ch4", "methane_leak_rate"],
                as_index=False,
            )["lifetime_unabated_mtco2e"]
            .sum()
            .rename(columns={"lifetime_unabated_mtco2e": "lifetime_mtco2e"})
        )
        unabated.insert(0, "scenario", "unabated")

        existing = (
            per_facility.query("retrofit_newbuild == 'retrofit'")
            .groupby(
                ["methane_scenario", "gwp_ch4", "methane_leak_rate"],
                as_index=False,
            )["lifetime_unabated_mtco2e"]
            .sum()
            .rename(columns={"lifetime_unabated_mtco2e": "lifetime_mtco2e"})
        )
        existing.insert(0, "scenario", "existing_only")

        agg = pd.concat([agg, unabated, existing], ignore_index=True)

    agg["lifetime_gtco2e"] = agg["lifetime_mtco2e"] / 1000.0
    agg["gwp_group"] = agg["gwp_ch4"].map(_gwp_group)
    return agg

def plot_lifetime_emissions_vs_leak_rate(
    df: pd.DataFrame,
    destination_scenarios: list,
    scenario_labels: dict | None = None,
):
    """Line-and-marker plot of lifetime emissions versus methane leak rate,
    with one line per (destination scenario × GWP horizon).

    Legend sits along the bottom in two columns: a colored-marker row for
    the CCS scenarios (different shapes per scenario) and a dark-gray
    line-with-dot row for the GWP horizons (solid = GWP20, dashed = GWP100).

    `scenario_labels` optionally overrides the legend text for one or more
    scenario names (keys match ``"baseline"``, ``"current_mix"``,
    ``"all_eor"``, ``"all_gs"``, ``"switching"``, ``"switching_from_current"``,
    ``"unabated"``, ``"existing_only"``). Unmatched keys are ignored;
    scenarios without an override keep their built-in display label.
    """
    fig, ax = plt.subplots(figsize=(4.2, 5.2))

    # Rendering order: CCS destinations first, then counterfactuals at the
    # end of the legend. Styles for the counterfactuals are fixed; styles
    # for CCS destinations come from the hand-picked color/marker cycles.
    series_order = list(destination_scenarios) + [
        s for s in ("unabated", "existing_only") if s in df["scenario"].unique()
    ]

    dest_color = {
        d: _DESTINATION_COLORS[i % len(_DESTINATION_COLORS)]
        for i, d in enumerate(destination_scenarios)
    }
    dest_marker = {
        d: _DESTINATION_MARKERS[i % len(_DESTINATION_MARKERS)]
        for i, d in enumerate(destination_scenarios)
    }
    for cf_name, style in _COUNTERFACTUAL_STYLE.items():
        dest_color[cf_name] = style["color"]
        dest_marker[cf_name] = style["marker"]

    for (dest, gwp_group), g in df.groupby(["scenario", "gwp_group"], sort=False):
        g = g.sort_values("methane_leak_rate")
        ax.plot(
            g["methane_leak_rate"] * 100,
            g["lifetime_gtco2e"],
            marker=dest_marker.get(dest, "o"),
            linewidth=1.5,
            markersize=6,
            linestyle=LINESTYLE_BY_GWP.get(gwp_group, ":"),
            color=dest_color.get(dest, "gray"),
        )

    ax.set_xlabel("Supply-chain methane leak rate (%)", fontsize=10)
    ax.set_ylabel("Net lifetime emissions (GtCO$_2$e)", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.set_ylim(0, 18)
    ax.grid(True, alpha=0.3)

    # Base-case assumption marker: a thin vertical dashed line at the
    # paper's default leak rate, with a short label at the top of the axes.
    base_x_pct = BASE_CASE_LEAK_RATE * 100
    ax.axvline(base_x_pct, color="gray", linestyle=":", linewidth=1.2, alpha=0.8)
    ax.annotate(
        f"base case\n({base_x_pct:.1f}%)",
        xy=(base_x_pct, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(4, -4),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8,
        color="gray",
    )

    # ── Legend handles ───────────────────────────────────────────────────
    dest_handles = [
        Line2D(
            [0],
            [0],
            marker=dest_marker[d],
            markersize=7,
            markerfacecolor=dest_color[d],
            markeredgecolor=dest_color[d],
            linestyle="",
            label={**_DESTINATION_LABEL_OVERRIDES, **(scenario_labels or {})}.get(d, d),
        )
        for d in series_order
    ]
    gwp_handles = [
        Line2D(
            [0],
            [0],
            color="dimgray",
            linewidth=1.5,
            linestyle=LINESTYLE_BY_GWP[g],
            label=_GWP_LEGEND_LABELS[g],
        )
        for g in ("GWP20", "GWP100")
    ]

    # Two legends side-by-side along the bottom: CCS scenarios on the left
    # (single column, one row per entry), GWP horizon on the right (single
    # column, two rows). This keeps the vertical footprint small.
    fig.tight_layout()
    n_ccs_rows = len(dest_handles)
    bottom_reserve = 0.14 + 0.035 * n_ccs_rows
    fig.subplots_adjust(bottom=bottom_reserve)

    ccs_legend = fig.legend(
        handles=dest_handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.0),
        title="Scenario",
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        ncol=1,
        handletextpad=0.4,
        alignment="left",
    )
    fig.add_artist(ccs_legend)
    fig.legend(
        handles=gwp_handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.0),
        title="GWP horizon",
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        ncol=1,
        handletextpad=0.5,
        alignment="left",
    )

    return fig, ax


_SCENARIO_LABELS = {
    "baseline": "Announced CO$_2$ destinations\n(83% GS; 17% EOR)",
    "current_mix": "Current U.S. practice\n(8% GS; 92% EOR)",
    "all_eor": "100% EOR",
    "all_gs": "100% GS",
    "switching": "Announced destinations, then\nall EOR after 45Q ends",
    "switching_from_current": "Current U.S. practice, then\nall EOR after 45Q ends",
    "half_eor": "50% EOR",
    "eor30": "30% EOR",
    "eor45": "45% EOR",
    "unabated": "Unabated\n(no CCS buildout)",
    "existing_only": "Existing facilities only\n(no buildout; no CCS)",
}

_SCENARIO_COLORS = {
    "baseline": "#1f77b4",
    "current_mix": "#d62728",
    "all_eor": "#9467bd",
    "all_gs": "#2ca02c",
    "switching": "#ff7f0e",
    "switching_from_current": "#e377c2",
    "half_eor": "#17becf",
    "eor30": "#8c564b",
    "eor45": "#bcbd22",
    "unabated": "#4D4D4D",
    "existing_only": "#808080",  # medium gray to group with the no-CCS line
}

_SCENARIO_MARKERS = {
    "baseline": "o",
    "current_mix": "s",
    "all_eor": "v",
    "all_gs": "P",
    "switching": "D",
    "switching_from_current": "h",
    "half_eor": "X",
    "eor30": "<",
    "eor45": ">",
    "unabated": "^",
    "existing_only": "D",
}

_LABEL_POSITIONS = {
    "existing_only": (9.0, 2.0),
    "all_gs": (3.0, 10.5),
    "baseline": (13.0, 15.0),
    "current_mix": (26.9, 10.5),
    "all_eor": (29.3, 14.6),
    "switching": (41.0, 16.5),
    "switching_from_current": (56.0, 18.0),
}

_LABEL_HALO_SCENARIOS: set = set()

_SCENARIO_MARKER_SIZES = {
    "switching_from_current": 220,
}

_DEFAULT_MARKER_SIZE = 140

def compute_scenario_totals(
    scen_df: pd.DataFrame,
    methane_scenario: str,
    destination_scenarios: list,
    states: tuple,
) -> pd.DataFrame:
    """One row per destination scenario with Σ-of-facilities totals for the
    chosen methane scenario and state set."""
    missing = sorted(set(destination_scenarios) - set(scen_df["scenario"].unique()))
    if missing:
        raise click.ClickException(
            f"Destination scenarios not present: {missing}. "
            f"Available: {sorted(scen_df['scenario'].unique())}"
        )
    if methane_scenario not in scen_df["methane_scenario"].unique():
        raise click.ClickException(
            f"Methane scenario {methane_scenario!r} not present; "
            f"available: {sorted(scen_df['methane_scenario'].unique())}"
        )

    sub = scen_df.query(
        "methane_scenario == @methane_scenario "
        "and scenario in @destination_scenarios "
        "and project_state in @states"
    )
    agg = sub.groupby("scenario", as_index=False).agg(
        pv_profit_musd=("pv_million_usd_oil_revenue", "sum"),
        lifetime_mtco2e=("lifetime_mtco2e", "sum"),
    )
    agg["lifetime_gtco2e"] = agg["lifetime_mtco2e"] / 1000.0
    return agg

def compute_counterfactual_totals(
    scen_df: pd.DataFrame,
    methane_scenario: str,
    states: tuple,
) -> pd.DataFrame:
    """Build (unabated, existing_only) reference rows for the same methane
    scenario and state set. Each facility contributes
    `facility_lifetime_years × unabated_mtco2e_per_yr`, which is already the
    per-row `lifetime_unabated_mtco2e` value; de-dupe to one row per uid
    before summing. Profit is zero by definition (no CCS → no EOR oil)."""
    per_fac = scen_df.query(
        "methane_scenario == @methane_scenario and project_state in @states"
    ).drop_duplicates(subset=["uid", "methane_scenario"])
    rows = []
    rows.append(
        {
            "scenario": "unabated",
            "pv_profit_musd": 0.0,
            "lifetime_mtco2e": float(per_fac["lifetime_unabated_mtco2e"].sum()),
        }
    )
    retro = per_fac[
        per_fac["retrofit_newbuild"].astype(str).str.strip().str.lower() == "retrofit"
    ]
    rows.append(
        {
            "scenario": "existing_only",
            "pv_profit_musd": 0.0,
            "lifetime_mtco2e": float(retro["lifetime_unabated_mtco2e"].sum()),
        }
    )
    out = pd.DataFrame(rows)
    out["lifetime_gtco2e"] = out["lifetime_mtco2e"] / 1000.0
    return out

def plot_profit_vs_emissions(df: pd.DataFrame):
    """Scatter with leader-line callouts labeling each scenario, plus a
    dashed `no CCS` reference line at the unabated emissions level."""
    fig, ax = plt.subplots(figsize=(9.0, 6.5))

    # Reference line at the "no CCS" (unabated) emissions level. This takes
    # the place of plotting the `unabated` point and anchors the comparison:
    # CCS scenarios above the line emit more than doing nothing.
    unabated_rows = df[df["scenario"] == "unabated"]
    unabated_y = (
        float(unabated_rows["lifetime_gtco2e"].iloc[0]) if len(unabated_rows) else None
    )
    if unabated_y is not None:
        ax.axhline(
            unabated_y,
            linestyle="--",
            color="dimgray",
            linewidth=1.2,
            zorder=1,
        )
        ax.text(
            68.5,
            unabated_y + 0.25,
            "no CCS",
            fontsize=12,
            color="dimgray",
            ha="right",
            va="bottom",
            fontstyle="italic",
        )

    # Scatter + leader-line annotation per scenario (skip `unabated` — it's
    # now represented by the dashed reference line above).
    for _, row in df.iterrows():
        scen = row["scenario"]
        if scen == "unabated":
            continue
        x = row["pv_profit_musd"] / 1000.0  # display as B$
        y = row["lifetime_gtco2e"]
        color = _SCENARIO_COLORS.get(scen, "gray")
        marker = _SCENARIO_MARKERS.get(scen, "o")
        ax.scatter(
            x,
            y,
            s=_SCENARIO_MARKER_SIZES.get(scen, _DEFAULT_MARKER_SIZE),
            marker=marker,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        label_pos = _LABEL_POSITIONS.get(scen)
        if label_pos is None:
            continue
        ann = ax.annotate(
            _SCENARIO_LABELS.get(scen, scen),
            xy=(x, y),
            xytext=label_pos,
            textcoords="data",
            fontsize=12,
            color="black",
            ha="center",
            va="center",
            arrowprops={
                "arrowstyle": "-",
                "color": "gray",
                "lw": 0.7,
                "shrinkA": 2,
                "shrinkB": 5,
            },
        )
        if scen in _LABEL_HALO_SCENARIOS:
            # White halo so the label text reads clearly over the "no CCS"
            # dashed reference line that passes right behind it.
            ann.set_path_effects(
                [
                    pe.Stroke(linewidth=3, foreground="white"),
                    pe.Normal(),
                ]
            )

    ax.set_xlabel("Present value of EOR oil profit (\\$ billion)", fontsize=16)
    ax.set_ylabel("Net lifetime emissions (GtCO$_2$e)", fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    ax.set_xlim(-1, 70)
    ax.set_ylim(0, 20)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax


METHANE_SCENARIO = "edf20"

METHANE_SCENARIOS_FOR_COMPARISON = ("edf20", "edf100")

GULF_COAST_STATES = ("TX", "LA")

def build_summary(
    scen_df: pd.DataFrame,
    ie_df: pd.DataFrame,
    methane_scenario: str = METHANE_SCENARIO,
    states: tuple = GULF_COAST_STATES,
    dac_reference_scenario: str = "baseline",
) -> pd.DataFrame:
    """One row per industry with per-unit-product emission factors, capture
    fractions, and total product produced across the requested state set.

    `dac_reference_scenario` selects which destination scenario is used when
    reading DAC's captured tonnage (used only for the `total_amount_of_product`
    entry for DAC, which has no unabated tonnage to back-solve production from).
    """
    raw_scen_df = scen_df  # keep the un-deduped frame around for DAC lookup

    # ── 1. Restrict scenarios to the requested slice ─────────────────────
    scen_df = (
        scen_df.query(
            "methane_scenario == @methane_scenario " "and project_state in @states"
        )
        .drop_duplicates(subset="uid")
        .rename(columns={"product": "simple_product_name"})
    )
    products_in_slice = scen_df["simple_product_name"].unique().tolist()

    # ── 2. Filter industrial-emissions factors to the same slice ──────────
    # Dedupe on (state, product) — a single simple_product_name may have
    # multiple `industrial_product` flavors; take the first.
    ie_df = ie_df.query(
        "methane_scenario == @methane_scenario "
        "and project_state in @states "
        "and simple_product_name in @products_in_slice"
    ).drop_duplicates(subset=["project_state", "simple_product_name"], keep="first")
    leak_rate = float(ie_df["methane_leak_rate"].iloc[0])
    gwp_ch4 = float(ie_df["gwp_ch4"].iloc[0])

    # One row per product (first state seen). Per-unit CO2/CH4 values are
    # config-driven and mostly state-independent, so first-value is exact for
    # nearly everything.
    per_product_df = ie_df.drop_duplicates(
        subset="simple_product_name", keep="first"
    ).copy()

    # ── 3. Compute per-unit CH4 emissions in both tCH4 and tCO2e ──────────
    # Direct-facility CH4 is bucketed with "process" here.
    per_product_df["process_ch4_tCH4_per_unit_product"] = per_product_df[
        "process_tch4_throughput_per_unit_product"
    ].fillna(0) * leak_rate + per_product_df[
        "direct_facility_tch4_per_unit_product"
    ].fillna(
        0
    )
    per_product_df["combustion_ch4_tCH4_per_unit_product"] = (
        per_product_df["combustion_tch4_throughput_per_unit_product"].fillna(0)
        + per_product_df["grid_tch4_per_unit_product"].fillna(0)
    ) * leak_rate
    per_product_df["process_ch4_tCO2e_per_unit_product"] = (
        per_product_df["process_ch4_tCH4_per_unit_product"] * gwp_ch4
    )
    per_product_df["combustion_ch4_tCO2e_per_unit_product"] = (
        per_product_df["combustion_ch4_tCH4_per_unit_product"] * gwp_ch4
    )

    # ── 4. Localized capture rate = fraction of the targeted stream ───────
    per_product_df["localized_capture_rate"] = np.where(
        per_product_df["ccs_applied_to"] == "process",
        per_product_df["process_capture_fraction_tco2"],
        per_product_df["combustion_capture_fraction_tco2"],
    )

    # ── 5. Total product across TX + LA (units match `unit_of_product`) ───
    # Back out per-facility production from unabated_mtco2e_per_yr and each
    # (state, product)'s total_tco2e_per_unit_product. unabated is stored in
    # Mt, so multiply by 1e6 to get absolute tonnes-of-CO2e first.
    tpu_lookup = ie_df.set_index(["project_state", "simple_product_name"])[
        "total_tco2e_per_unit_product"
    ]

    def _tpu(row):
        key = (row["project_state"], row["simple_product_name"])
        return tpu_lookup.get(key, np.nan)

    scen_df["_tpu"] = scen_df.apply(_tpu, axis=1)
    scen_df["_production_units"] = np.where(
        scen_df["_tpu"] > 0,
        scen_df["unabated_mtco2e_per_yr"] * 1e6 / scen_df["_tpu"],
        0.0,
    )
    total_product = scen_df.groupby("simple_product_name")["_production_units"].sum()
    per_product_df["total_amount_of_product"] = (
        per_product_df["simple_product_name"].map(total_product).fillna(0.0)
    )

    # DAC override: DAC has zero unabated tonnage by convention, so the
    # back-solve above yields 0. Its actual "product" is CO2 captured; sum
    # `mtco2_per_yr_captured_with_ccs` across DAC facilities under the
    # reference destination scenario, and convert from Mt to tonnes to match
    # unit_of_product = "ton CO2".
    if "dac" in per_product_df["simple_product_name"].values:
        dac_captured_mt = float(
            raw_scen_df.query(
                "methane_scenario == @methane_scenario "
                "and project_state in @states "
                "and product == 'dac' "
                "and scenario == @dac_reference_scenario"
            )["mtco2_per_yr_captured_with_ccs"].sum()
        )
        per_product_df.loc[
            per_product_df["simple_product_name"] == "dac",
            "total_amount_of_product",
        ] = (
            dac_captured_mt * 1e6
        )

    # ── 6. Order and rename columns for output ────────────────────────────
    summary_df = per_product_df[
        [
            "simple_product_name",
            "unit_of_product",
            "total_amount_of_product",
            "process_tco2_per_unit_product",
            "combustion_tco2_per_unit_product",
            "process_ch4_tCO2e_per_unit_product",
            "process_ch4_tCH4_per_unit_product",
            "combustion_ch4_tCO2e_per_unit_product",
            "combustion_ch4_tCH4_per_unit_product",
            "ccs_applied_to",
            "localized_capture_rate",
            "adjusted_capture_fraction_tco2",
        ]
    ].rename(
        columns={
            "process_tco2_per_unit_product": "process_co2_tCO2_per_unit_product",
            "combustion_tco2_per_unit_product": "combustion_co2_tCO2_per_unit_product",
            "adjusted_capture_fraction_tco2": "effective_capture_rate",
        }
    )

    return summary_df.sort_values(
        "process_co2_tCO2_per_unit_product", ascending=False
    ).reset_index(drop=True)

def build_multi_scenario_summary(
    scen_df: pd.DataFrame,
    ie_df: pd.DataFrame,
    ccs_df: pd.DataFrame,
    methane_scenarios: Iterable[str] = METHANE_SCENARIOS_FOR_COMPARISON,
    states: tuple = GULF_COAST_STATES,
) -> pd.DataFrame:
    """One row per industry comparing effective capture fraction and CCS
    lifecycle emissions factor across multiple methane scenarios.

    For each scenario in `methane_scenarios`, three columns are added:
      effective_capture_rate_<scenario>  — `adjusted_capture_fraction_tco2e`
         (tCO2e basis; varies by scenario via leak rate and GWP).
      ccs_factor_gs_<scenario>           — `tco2e_emitted_per_tco2_ccs` for GS.
      ccs_factor_eor_<scenario>          — `tco2e_emitted_per_tco2_ccs` for EOR.

    Each (scenario, source, destination, state) tuple in `ccs_df` has a single
    `distance` row (the industry's primary pipeline tier assigned upstream),
    so this function takes that row as-is rather than filtering on distance.
    """
    methane_scenarios = list(methane_scenarios)

    products = (
        scen_df.query("project_state in @states")
        .rename(columns={"product": "simple_product_name"})["simple_product_name"]
        .dropna()
        .unique()
        .tolist()
    )
    out = pd.DataFrame({"simple_product_name": products})

    for scen in methane_scenarios:
        ie_slice = (
            ie_df.query(
                "methane_scenario == @scen "
                "and project_state in @states "
                "and simple_product_name in @products"
            )
            .drop_duplicates(subset="simple_product_name", keep="first")[
                ["simple_product_name", "adjusted_capture_fraction_tco2e"]
            ]
            .rename(
                columns={
                    "adjusted_capture_fraction_tco2e": f"effective_capture_rate_{scen}",
                }
            )
        )
        out = out.merge(ie_slice, on="simple_product_name", how="left")

        for dest in ("gs", "eor"):
            ccs_slice = (
                ccs_df.query(
                    "methane_scenario == @scen "
                    "and co2_destination == @dest "
                    "and project_state in @states"
                )
                .drop_duplicates(subset="co2_source", keep="first")[
                    ["co2_source", "tco2e_emitted_per_tco2_ccs"]
                ]
                .rename(
                    columns={
                        "co2_source": "simple_product_name",
                        "tco2e_emitted_per_tco2_ccs": f"ccs_factor_{dest}_{scen}",
                    }
                )
            )
            out = out.merge(ccs_slice, on="simple_product_name", how="left")

    return out.sort_values("simple_product_name").reset_index(drop=True)


_GWP_TO_METHANE = {20: "edf20", 100: "edf100"}

_DEFAULT_SCENARIOS__table5_scenario_emissions = (
    "existing_only",
    "unabated",
    "all_gs",
    "baseline",
    "current_mix",
    "all_eor",
    "switching",
    "switching_from_current",
)

_PRETTY__table5_scenario_emissions = {
    "existing_only": "Existing facilities only (no further buildout, no CCS)",
    "unabated": "Full buildout, unabated (no CCS)",
    "all_gs": "100% Geologic storage ('best case' climate impact)",
    "baseline": "Announced destinations (17% EOR, 83% GS)",
    "current_mix": "Current U.S. practice (92% EOR, 8% GS)",
    "all_eor": "100% EOR ('worst case' while 45Q is paid)",
    "switching": "Announced destinations, then all EOR after 45Q",
    "switching_from_current": "Current U.S. practice, then all EOR after 45Q",
}

def table5_scenario_emissions(
    scenarios_df: pd.DataFrame | None = None,
    gwp: int = 20,
    as_fraction: bool = False,
    scenarios: tuple | list = _DEFAULT_SCENARIOS__table5_scenario_emissions,
    states: tuple = ("TX", "LA"),
    scenario_labels: dict | None = None,
    csv_path: str | Path = SCENARIOS_CSV,
) -> pd.DataFrame:
    """Table 5 for one GWP horizon.

    Parameters
    ----------
    scenarios_df
        Pre-loaded `ccs_scenarios_out.csv`. If None, `csv_path` is read.
    gwp
        20 or 100. Picks the matching methane scenario (`edf` or `edf100`).
    as_fraction
        If False (default), the two Δ-vs-unabated columns are reported in
        percent (e.g. 5.45). If True, they're reported as unitless fractions
        (e.g. 0.0545). The two Δ-vs-existing columns are always absolute
        increases (MtCO₂e/yr and GtCO₂e) regardless of this flag.
    scenarios
        Scenario keys to include, in output row order.
    states
        State codes to aggregate over.
    scenario_labels
        Optional display-name overrides, keyed by scenario key.
    csv_path
        Used only when `scenarios_df` is None.
    """
    if gwp not in _GWP_TO_METHANE:
        raise ValueError(f"gwp must be 20 or 100, got {gwp!r}")
    methane_scenario = _GWP_TO_METHANE[gwp]

    if scenarios_df is None:
        scenarios_df = pd.read_csv(csv_path, encoding="latin-1")

    sub = scenarios_df.loc[
        scenarios_df["project_state"].isin(states)
        & (scenarios_df["methane_scenario"] == methane_scenario)
    ]
    base = sub.loc[sub["scenario"] == "baseline"]
    retro_mask = (
        base["retrofit_newbuild"].astype(str).str.strip().str.lower() == "retrofit"
    )

    # Reference totals for percent-change columns.
    unabated_annual_mt = base["unabated_mtco2e_per_yr"].sum()
    unabated_lifetime_mt = base["lifetime_unabated_mtco2e"].sum()
    existing_annual_mt = base.loc[retro_mask, "unabated_mtco2e_per_yr"].sum()
    existing_lifetime_mt = base.loc[retro_mask, "lifetime_unabated_mtco2e"].sum()

    def _annual_and_lifetime(key: str) -> tuple:
        if key == "unabated":
            return unabated_annual_mt, unabated_lifetime_mt
        if key == "existing_only":
            return existing_annual_mt, existing_lifetime_mt
        ss = sub.loc[sub["scenario"] == key]
        return (
            ss["mtco2e_per_yr_emissions_including_ccs"].sum(),
            ss["lifetime_mtco2e"].sum(),
        )

    scale = 1.0 if as_fraction else 100.0
    unit = "fraction" if as_fraction else "%"

    labels = {**_PRETTY__table5_scenario_emissions, **(scenario_labels or {})}
    rows = []
    for s in scenarios:
        annual_mt, lifetime_mt = _annual_and_lifetime(s)
        rows.append(
            {
                "Scenario": labels.get(s, s),
                f"Annual (MtCO₂e/yr, GWP{gwp})": annual_mt,
                f"Δ Annual vs. existing (MtCO₂e/yr)": annual_mt - existing_annual_mt,
                f"Δ Annual vs. unabated ({unit})": scale
                * (annual_mt - unabated_annual_mt)
                / unabated_annual_mt,
                f"Lifetime (GtCO₂e, GWP{gwp})": lifetime_mt / 1000.0,
                f"Δ Lifetime vs. existing (GtCO₂e)": (
                    lifetime_mt - existing_lifetime_mt
                )
                / 1000.0,
                f"Δ Lifetime vs. unabated ({unit})": scale
                * (lifetime_mt - unabated_lifetime_mt)
                / unabated_lifetime_mt,
            }
        )
    return pd.DataFrame(rows)


_PRETTY__table_scenario_oil_and_emissions = {
    "existing_only": "Existing facilities only (no further buildout, no CCS)",
    "unabated": "Full buildout, unabated (no CCS)",
    "all_gs": "100% Geologic storage",
    "baseline": "Announced destinations (17% EOR, 83% GS)",
    "current_mix": "Current U.S. practice (92% EOR, 8% GS)",
    "all_eor": "100% EOR",
    "switching": "Announced destinations, then all EOR after 45Q",
    "switching_from_current": "Current U.S. practice, then all EOR after 45Q",
}

def scenario_oil_and_emissions_table(
    scenarios_df: pd.DataFrame | None = None,
    scenarios: tuple | list = _DEFAULT_SCENARIOS__table5_scenario_emissions,
    methane_scenario: str = "edf20",
    states: tuple = ("TX", "LA"),
    scenario_labels: dict | None = None,
    csv_path: str | Path = SCENARIOS_CSV,
) -> pd.DataFrame:
    """Return per-scenario oil barrels, oil profit (nominal + PV), and
    lifetime-emissions delta vs. the unabated buildout.

    Parameters
    ----------
    scenarios_df
        Pre-loaded `ccs_scenarios_out.csv`. If None, `csv_path` is read.
    scenarios
        Scenario keys to include, in output order.
    methane_scenario
        Methane scenario to fix. Default `edf` = 2.3% leak, GWP20.
    states
        State codes to aggregate over.
    scenario_labels
        Optional override for display names (keys match scenario keys).
    csv_path
        Used only when `scenarios_df` is None.
    """
    if scenarios_df is None:
        scenarios_df = pd.read_csv(csv_path, encoding="latin-1")

    sub = scenarios_df.loc[
        scenarios_df["project_state"].isin(states)
        & (scenarios_df["methane_scenario"] == methane_scenario)
    ]

    # `unabated` and `existing_only` aren't stored as scenario rows in the CSV;
    # synthesize them from the per-facility unabated column on any other
    # scenario (they're identical across scenarios for a given methane case).
    any_scen = sub.loc[sub["scenario"] == scenarios_df["scenario"].iloc[0]]
    if any_scen.empty:
        any_scen = sub.loc[sub["scenario"] == "baseline"]
    unabated_total_mt = any_scen["lifetime_unabated_mtco2e"].sum()
    retro_total_mt = any_scen.loc[
        any_scen["retrofit_newbuild"].astype(str).str.strip().str.lower() == "retrofit",
        "lifetime_unabated_mtco2e",
    ].sum()

    labels = {**_PRETTY__table_scenario_oil_and_emissions, **(scenario_labels or {})}
    rows = []
    for s in scenarios:
        if s == "unabated":
            rows.append(
                {
                    "Scenario": labels.get(s, s),
                    "EOR oil produced (billion bbl)": 0.0,
                    "Profit, nominal ($ billion)": 0.0,
                    "Profit, present value ($ billion)": 0.0,
                    "Change in lifetime emissions vs. unabated (GtCO₂e, GWP₂₀)": 0.0,
                }
            )
            continue
        if s == "existing_only":
            rows.append(
                {
                    "Scenario": labels.get(s, s),
                    "EOR oil produced (billion bbl)": 0.0,
                    "Profit, nominal ($ billion)": 0.0,
                    "Profit, present value ($ billion)": 0.0,
                    "Change in lifetime emissions vs. unabated (GtCO₂e, GWP₂₀)": (
                        retro_total_mt - unabated_total_mt
                    )
                    / 1000.0,
                }
            )
            continue
        ss = sub.loc[sub["scenario"] == s]
        rows.append(
            {
                "Scenario": labels.get(s, s),
                "EOR oil produced (billion bbl)": ss[
                    "lifetime_million_barrels_oil"
                ].sum()
                / 1000.0,
                "Profit, nominal ($ billion)": ss[
                    "lifetime_nominal_million_usd_oil_revenue"
                ].sum()
                / 1000.0,
                "Profit, present value ($ billion)": ss[
                    "lifetime_pv_million_usd_oil_revenue"
                ].sum()
                / 1000.0,
                "Change in lifetime emissions vs. unabated (GtCO₂e, GWP₂₀)": (
                    ss["lifetime_mtco2e"].sum() - ss["lifetime_unabated_mtco2e"].sum()
                )
                / 1000.0,
            }
        )
    return pd.DataFrame(rows)


_TABLE_S2_DEFAULT_DESCRIPTIONS = {
    "existing_only": "Existing facilities only (no further buildout, no CCS)",
    "unabated": "Full buildout, unabated (no CCS)",
    "all_gs": "100% Geologic storage ('best case' climate impact)",
    "baseline": "Announced destination (17% EOR, 83% GS)",
    "current_mix": "Current practice (92% EOR, 8% GS)",
    "all_eor": "100% EOR ('worst case' while 45Q is paid)",
    "switching": "Announced destination, then 100% EOR after 12-year 45Q project",
    "switching_from_current": "Current practice, then 100% EOR after 12-year 45Q project",
}


def build_table_s2(
    scenarios_df: pd.DataFrame,
    scenarios: list | None = None,
    scenario_descriptions: dict | None = None,
    states: tuple = ("TX", "LA"),
    methane_gwp20: str = "edf20",
    methane_gwp100: str = "edf100",
    facility_lifetime_years: int = 30,
) -> pd.DataFrame:
    """Paper Table S2 — state-split Table 5.

    For every state in `states` and every scenario in `scenarios`, returns:
      * Annual MtCO₂e/yr at GWP₂₀ and GWP₁₀₀
      * Lifetime GtCO₂e over `facility_lifetime_years` at GWP₂₀ and GWP₁₀₀
      * Δ Lifetime vs. full-buildout unabated (GtCO₂e, GWP₂₀)
      * Δ Lifetime vs. existing-facilities-only (GtCO₂e, GWP₂₀)

    Each state contributes a block of six metric columns prefixed by the
    state code (e.g. "TX — Annual (MtCO₂e/yr, GWP₂₀)"); state blocks share
    the single "Scenario" column.

    CCS scenarios (including the switching variants) read
    `mtco2e_per_yr_emissions_including_ccs` and `lifetime_mtco2e`;
    counterfactuals (`unabated`, `existing_only`) use the annual unabated
    rate × `facility_lifetime_years`.
    """
    if scenario_descriptions is None:
        scenario_descriptions = _TABLE_S2_DEFAULT_DESCRIPTIONS
    if scenarios is None:
        scenarios = [
            "existing_only",
            "unabated",
            "all_gs",
            "baseline",
            "current_mix",
            "all_eor",
            "switching",
            "switching_from_current",
        ]
    L = facility_lifetime_years

    def _per_state(state: str) -> pd.DataFrame:
        sdf = scenarios_df[scenarios_df["project_state"] == state]
        retrofit = (
            sdf["retrofit_newbuild"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("retrofit")
        )

        def annual(scen: str, methane: str) -> float:
            if scen == "unabated":
                sel = sdf[
                    (sdf["methane_scenario"] == methane)
                    & (sdf["scenario"] == "baseline")
                ]
                return float(
                    sel.drop_duplicates("uid")["unabated_mtco2e_per_yr"].sum()
                )
            if scen == "existing_only":
                sel = sdf[(sdf["methane_scenario"] == methane) & retrofit]
                return float(
                    sel.drop_duplicates("uid")["unabated_mtco2e_per_yr"].sum()
                )
            sel = sdf[
                (sdf["methane_scenario"] == methane) & (sdf["scenario"] == scen)
            ]
            return float(sel["mtco2e_per_yr_emissions_including_ccs"].sum())

        def lifetime_gt(scen: str, methane: str) -> float:
            if scen in ("unabated", "existing_only"):
                return annual(scen, methane) * L / 1000.0
            sel = sdf[
                (sdf["methane_scenario"] == methane) & (sdf["scenario"] == scen)
            ]
            return float(sel["lifetime_mtco2e"].sum()) / 1000.0

        unabated_gt20 = annual("unabated", methane_gwp20) * L / 1000.0
        existing_gt20 = annual("existing_only", methane_gwp20) * L / 1000.0

        rows = []
        for scen in scenarios:
            life20 = lifetime_gt(scen, methane_gwp20)
            rows.append(
                {
                    "Scenario": scenario_descriptions.get(scen, scen),
                    f"{state} — Annual (MtCO₂e/yr, GWP₂₀)": annual(scen, methane_gwp20),
                    f"{state} — Annual (MtCO₂e/yr, GWP₁₀₀)": annual(scen, methane_gwp100),
                    f"{state} — Lifetime (GtCO₂e, GWP₂₀)": life20,
                    f"{state} — Lifetime (GtCO₂e, GWP₁₀₀)": lifetime_gt(scen, methane_gwp100),
                    f"{state} — Δ Lifetime vs. unabated (GtCO₂e, GWP₂₀)": life20 - unabated_gt20,
                    f"{state} — Δ Lifetime vs. existing (GtCO₂e, GWP₂₀)": life20 - existing_gt20,
                }
            )
        return pd.DataFrame(rows)

    blocks = [_per_state(s) for s in states]
    result = blocks[0]
    for b in blocks[1:]:
        result = result.merge(b, on="Scenario")
    return result


__all__ = [
    "BASE_CASE_LEAK_RATE",
    "COLOR_CAPTURED",
    "COLOR_EOR",
    "COLOR_GS",
    "COLOR_METHANE",
    "COLOR_UNCAPTURED_CO2",
    "DATA_DIR",
    "GULF_COAST_STATES",
    "GWP100_VALUES",
    "GWP20_VALUES",
    "INDUSTRY_COLORS",
    "INDUSTRY_MARKERS",
    "LINESTYLE_BY_GWP",
    "METHANE_SCENARIO",
    "METHANE_SCENARIOS_FOR_COMPARISON",
    "SCENARIOS_CSV",
    "_BREAKDOWN_COMPONENTS",
    "_CCS_COLOR_100",
    "_CCS_COLOR_20",
    "_CITY_LABEL_OFFSETS",
    "_CITY_LEADER_LINES",
    "_COUNTERFACTUAL_STYLE",
    "_DEFAULT_INDUSTRIES",
    "_DEFAULT_MARKER_SIZE",
    "_DEFAULT_PLOTTING_CONFIG_PATH",
    "_DEFAULT_PRODUCT_GROUP_MAP",
    "_DEFAULT_SCENARIOS",
    "_DEFAULT_SCENARIOS__table5_scenario_emissions",
    "_DESTINATION_COLORS",
    "_DESTINATION_LABEL_OVERRIDES",
    "_DESTINATION_MARKERS",
    "_FACILITY_NAME_DEFAULTS",
    "_FONT_PRESETS",
    "_GWP_LEGEND_LABELS",
    "_GWP_TO_METHANE",
    "_IShapeHandler",
    "_LABEL_HALO_SCENARIOS",
    "_LABEL_POSITIONS",
    "_LEGEND_FALLBACK_PALETTE",
    "_NAME_REPLACEMENTS",
    "_PRETTY",
    "_PRETTY__table5_scenario_emissions",
    "_PRETTY__table_scenario_oil_and_emissions",
    "_PRODUCT_ALIASES",
    "_SANKEY_NODE_COLORS",
    "_SCENARIO_COLORS",
    "_SCENARIO_LABELS",
    "_SCENARIO_MARKERS",
    "_SCENARIO_MARKER_SIZES",
    "_SOLAR_COLOR_100",
    "_SOLAR_COLOR_20",
    "_STATE_DISPLAY_NAMES",
    "_STATE_MAP_DEFAULTS",
    "_annotate_facility_windows",
    "_build_breakdown_data",
    "_compute_dac_records",
    "_compute_dual_axis_records",
    "_compute_panel_data",
    "_compute_unabated_record",
    "_delta_gt",
    "_format_states_for_title",
    "_gwp_group",
    "_hex_to_rgba",
    "_resolve_display_mode",
    "_resolve_fonts",
    "_solar_gt",
    "build_multi_scenario_summary",
    "build_summary",
    "build_table_s2",
    "compute_buildout_lifetime",
    "compute_counterfactual_totals",
    "compute_scenario_totals",
    "logger",
    "plot_emissions_breakdown_combined",
    "plot_emissions_through_time",
    "plot_facility_map",
    "plot_industry_sankey",
    "plot_lifetime_delta_vs_unabated",
    "plot_lifetime_emissions_vs_leak_rate",
    "plot_products_to_destination_sankey",
    "plot_profit_vs_emissions",
    "plot_state_emissions_vs_unabated_and_existing",
    "plot_state_sankey",
    "plot_two_panel",
    "scenario_oil_and_emissions_table",
    "summarize_scenario_ids",
    "summarize_state_scenarios",
    "table5_scenario_emissions",
]
