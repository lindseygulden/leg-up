"""Plotting utilities for CCS scenario results."""

# pylint: disable=too-many-arguments,too-many-positional-arguments, consider-using-enumerate, broad-exception-caught, too-many-lines, consider-using-enumerate import-outside-toplevel, too-many-locals, too-many-branches, too-many-statements, missing-function-docstring
import datetime
import logging
import os
import textwrap
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    import plotly.graph_objects as go

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from ccs_ghg.constants import BBL_IN_KBOE, DAYS_PER_YEAR
from ccs_ghg._vendored.io import yaml_to_dict
from ccs_ghg._vendored.plotting import add_brace

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


# Per-industry marker shapes for `plot_facility_map(markers=INDUSTRY_MARKERS)`.
# Shape groups follow industry type: triangles = feedstock conversion
# (hydrogen, ammonia, methanol, iron_steel); squares/diamonds = natural-gas
# handling (lng, ngp); stars/hexagons = power generation; distinct outliers
# for refining, petrochemicals, ethanol, and DAC. Chosen so a B&W print can
# tell the industries apart by shape alone.
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


class _BarWithLineHandler(HandlerBase):
    """Legend handler that draws a filled rectangle with a colored line on top."""

    def __init__(self, bar_color, line_color, alpha=0.9, line_width=3.0, **kwargs):
        super().__init__(**kwargs)
        self._bar_color = bar_color
        self._line_color = line_color
        self._alpha = alpha
        self._line_width = line_width

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        patch = Rectangle(
            [xdescent, ydescent],
            width,
            height,
            facecolor=self._bar_color,
            edgecolor="none",
            alpha=self._alpha,
            transform=trans,
        )
        line = Line2D(
            [xdescent, xdescent + width],
            [ydescent + height, ydescent + height],
            color=self._line_color,
            linewidth=self._line_width,
            transform=trans,
        )
        return [patch, line]


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


def _load_config(config: Union[str, dict]) -> dict:
    if isinstance(config, str):
        return yaml_to_dict(config)
    return config


def _legend_label(case_cfg: dict) -> str:
    label = case_cfg["label"]
    if "bold_prefix" in case_cfg:
        prefix = case_cfg["bold_prefix"].replace(" ", r"\ ")
        return rf"$\bf{{{prefix}}}$ {label}"
    return case_cfg.get("legend", label)


def _build_product_group_map(plotting_config: Union[str, dict, None]) -> Dict[str, str]:
    """Build a reverse map {member_industry: group_name} from plot_groups config."""
    if plotting_config is None:
        return {}
    cfg = _load_config(plotting_config)
    group_map = {}
    for group_name, members in cfg.get("plot_groups", {}).items():
        for member in members:
            group_map[member] = group_name
    return group_map


def _apply_plot_groups(df: pd.DataFrame, group_map: Dict[str, str]) -> pd.DataFrame:
    """Remap product column values according to plot_groups."""
    if not group_map:
        return df
    result = df.copy()
    result["product"] = result["product"].map(lambda p: group_map.get(p, p))
    return result


def apply_plot_groups(
    df: pd.DataFrame,
    plotting_config: Union[str, dict] = _DEFAULT_PLOTTING_CONFIG_PATH,
) -> pd.DataFrame:
    """Remap product column values using plot_groups from the plotting config.

    Use this to collapse related industries (e.g. power_ng and
    power_ng_data_center) into a single product before passing
    scenarios data to plotting functions.
    """
    group_map = _build_product_group_map(plotting_config)
    return _apply_plot_groups(df, group_map)


def _product_label(product: str, overrides: Dict[str, str]) -> str:
    if product in overrides:
        return overrides[product]
    return product.upper().replace("_", " ")


def _build_plot_data(
    scenarios_df: pd.DataFrame,
    methane_scenario: str,
    cases: List[str],
    group_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Aggregate scenarios.csv into long-form (product, case, MtCO2e/yr)."""
    df = scenarios_df[scenarios_df["methane_scenario"] == methane_scenario]
    if group_map:
        df = _apply_plot_groups(df, group_map)

    records: list = []
    for case in cases:
        if case == "unabated":
            vals = (
                df[["uid", "product", "unabated_mtco2e_per_yr"]]
                .drop_duplicates(subset="uid")
                .groupby("product")["unabated_mtco2e_per_yr"]
                .sum()
            )
        else:
            vals = (
                df[df["scenario"] == case]
                .groupby("product")["mtco2e_per_yr_emissions_including_ccs"]
                .sum()
            )
        for product, value in vals.items():
            records.append({"product": product, "case": case, "MtCO2e/yr": value})

    return pd.DataFrame(records)


# custom squarify calculation
def _worst_ratio(row, side):
    s = sum(row)
    if s == 0 or side == 0:
        return float("inf")
    rmax, rmin = max(row), min(row)
    return max(
        (side * side * rmax) / (s * s),
        (s * s) / (side * side * rmin),
    )


def _layout_row(row, x, y, w, h):
    total = sum(row)
    rects = []
    if w <= h:
        rh = total / w if w else 0
        cx = x
        for r in row:
            rw = r / rh if rh else 0
            rects.append({"x": cx, "y": y, "dx": rw, "dy": rh})
            cx += rw
        rem = {"x": x, "y": y + rh, "w": w, "h": h - rh}
    else:
        rw = total / h if h else 0
        cy = y
        for r in row:
            rh2 = r / rw if rw else 0
            rects.append({"x": x, "y": cy, "dx": rw, "dy": rh2})
            cy += rh2
        rem = {"x": x + rw, "y": y, "w": w - rw, "h": h}
    return rects, rem


def squarify(sizes, x, y, w, h):
    area = w * h
    total = sum(sizes)
    normed = [s * area / total for s in sizes]
    result = []
    _sq(normed, x, y, w, h, result)
    return result


def _sq(sizes, x, y, w, h, out):
    if not sizes:
        return
    if len(sizes) == 1:
        out.append({"x": x, "y": y, "dx": w, "dy": h})
        return
    side = min(w, h)
    row = [sizes[0]]
    rest = sizes[1:]
    while rest:
        trial = row + [rest[0]]
        if _worst_ratio(trial, side) <= _worst_ratio(row, side):
            row = trial
            rest = rest[1:]
        else:
            break
    rects, rem = _layout_row(row, x, y, w, h)
    out.extend(rects)
    _sq(rest, rem["x"], rem["y"], rem["w"], rem["h"], out)


def plot_treemap_co2_source(  # pylint: disable=unused-argument
    products: Optional[List[Tuple[str, float]]] = None,
    width: float = 80,
    height: float = 65,
    fontsize: int = 13,
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    # ── data ──────────────────────────────────────────────────────────────
    if products is None:
        products = [
            ("'Clean' ammonia\n(methane feedstock)", 21.772),
            ("Power\n(methane-fired)", 11.607),
            ("'Clean'\nhydrogen\n(methane\nfeedstock)", 8.664),
            ("LNG", 5.086),
            ("Power\n(coal-fired)", 1.396),
            ("Ethanol", 0.705),
            ("Petrochemicals", 0.602),
            ("Methanol", 0.319),
            ("Natural gas processing", 0.244),
            ("Refined fuels", 0.145),
        ]
    names = [p[0] for p in products]
    values = [p[1] for p in products]
    total = sum(values)

    palette = [
        "#2a78d6",
        "#1baf7a",
        "#eda100",
        "#008300",
        "#4a3aa7",
        "#e34948",
        "#e87ba4",
        "#eb6834",
        "#888780",
        "#6e6d68",
    ]

    def luminance(h):
        r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
        return 0.299 * r + 0.587 * g + 0.114 * b

    def fg(h):
        return "#1a1a19" if luminance(h) > 150 else "#ffffff"

    def _fg_sub(h):
        return "#4a4a48" if luminance(h) > 150 else "#ffffffb3"

    # ── layout ────────────────────────────────────────────────────────────
    w, h = width, height  # 130, 60  # ~1.61 aspect ratio matching the widget
    pad = 0.2  # gap between tiles

    rects = squarify(values, 0, 0, w, h)

    # shrink each rect by PAD for gutters
    for r in rects:
        r["x"] += pad / 2
        r["y"] += pad / 2
        r["dx"] -= pad
        r["dy"] -= pad

    # ── figure ────────────────────────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=(13.6, 8.4))
        fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    label_states = []
    for i, (rect, name, val) in enumerate(zip(rects, names, values)):
        x, y, w, h = rect["x"], rect["y"], rect["dx"], rect["dy"]
        pct = val / total * 100
        bg = palette[i]

        # rounded rectangle
        fancy = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0",
            facecolor=bg,
            edgecolor="white",
            linewidth=0,
        )
        ax.add_patch(fancy)

        # label sizing — estimate char width in data coords at each font size
        # use longest line for multi-line names
        longest_line = max(name.split("\n"), key=len)
        n_lines = len(name.split("\n"))
        big = w > 14
        cw = 0.75 if big else 0.62
        name_w = len(longest_line) * cw
        has_room_name = w > name_w + 2.5 and h > (5 + 3.5 * (n_lines - 1))
        has_room_sub = w > 10 and h > (10 + 3.5 * (n_lines - 1))

        # ── track what gets labeled in-box ────────────────────────────────
        label_state = "none"  # "full", "name_only", or "none"

        if has_room_name and has_room_sub:
            label_state = "full"
            name_y = y + h - 5.8 - 3.5 * (n_lines - 1)
            ax.text(
                x + 1.4,
                name_y,
                name,
                fontsize=fontsize + 3 if big else fontsize,
                fontweight=500,
                fontfamily="sans-serif",
                color=fg(bg),
                va="baseline",
                ha="left",
                linespacing=1.15,
            )
            ax.text(
                x + 1.4,
                y + h - 2.2,
                f"{val:.1f} Mt  \u00b7  {pct:.0f}%",
                fontsize=fontsize - 1 if big else fontsize - 3,
                fontweight=600,
                fontfamily="sans-serif",
                color=fg(bg),
                alpha=1,  # 0.85,
                va="baseline",
                ha="left",
            )
        elif has_room_name:
            label_state = "name_only"
            ax.text(
                x + w / 2,
                y + h / 2,
                name,
                fontsize=fontsize if big else fontsize - 2,
                fontweight=500,
                fontfamily="sans-serif",
                color=fg(bg),
                va="center",
                ha="center",
                linespacing=1.15,
            )

        label_states.append(label_state)

    ax.set_xlim(-0.3, w + 0.3)
    ax.set_ylim(-0.3, h + 0.3)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

    # ── legend — only items without full in-box labels ────────────────────
    legend_handles = []
    for i, (name, val) in enumerate(products):
        if label_states[i] == "full":
            continue  # already fully labeled in its tile
        pct = val / total * 100
        # name-only tiles (e.g. Ethanol): include the numeric they're missing
        # unlabeled tiles: include both name and numeric
        name_oneline = name.replace("\n", " ")
        label = f"{name_oneline}  {val:.1f} Mt ({pct:.0f}%)"
        patch = plt.Rectangle((0, 0), 1, 1, fc=palette[i], ec="none")
        legend_handles.append((patch, label))

    if legend_handles:
        ncol = min(len(legend_handles), 2)
        leg = ax.legend(
            [h[0] for h in legend_handles],
            [h[1] for h in legend_handles],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=ncol,
            frameon=False,
            fontsize=fontsize + 1,
            handlelength=1.0,
            handletextpad=0.5,
            columnspacing=1.5,
        )
        for text in leg.get_texts():
            text.set_color("#333333")

    # plt.subplots_adjust(bottom=0.10)
    # plt.tight_layout(pad=0)
    return ax


def plot_treemap_total_emissions(
    scenarios_df: pd.DataFrame,
    methane_scenario: str,
    scenario: str = "current_mix",
    name_replacements: Optional[Dict[str, str]] = None,
    width: float = 80,
    height: float = 65,
    fontsize: int = 13,
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    replacements = {**_FACILITY_NAME_DEFAULTS, **(name_replacements or {})}

    df = scenarios_df[
        (scenarios_df["methane_scenario"] == methane_scenario)
        & (scenarios_df["scenario"] == scenario)
    ].copy()

    grouped = (
        df.groupby("product")["mtco2e_per_yr_emissions_including_ccs"]
        .sum()
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    grouped = grouped[grouped != 0]

    products = []
    raw_values = []
    for ind, val in grouped.items():
        label = replacements.get(ind, ind.replace("_", " ").title())
        products.append((label, val))
        raw_values.append(val)

    names = [p[0] for p in products]
    values = [abs(v) for v in raw_values]
    total = sum(values)

    palette = [
        "#2a78d6",
        "#1baf7a",
        "#eda100",
        "#008300",
        "#4a3aa7",
        "#e34948",
        "#e87ba4",
        "#eb6834",
        "#888780",
        "#6e6d68",
        "#5c6bc0",
        "#26a69a",
    ]

    def luminance(h):
        r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
        return 0.299 * r + 0.587 * g + 0.114 * b

    def fg(h):
        return "#1a1a19" if luminance(h) > 150 else "#ffffff"

    w, h = width, height
    pad = 0.2

    rects = squarify(values, 0, 0, w, h)
    for r in rects:
        r["x"] += pad / 2
        r["y"] += pad / 2
        r["dx"] -= pad
        r["dy"] -= pad

    if ax is None:
        fig, ax = plt.subplots(figsize=(13.6, 8.4))
        fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    label_states = []
    for i, (rect, name, val) in enumerate(zip(rects, names, values)):
        x, y, w, h = rect["x"], rect["y"], rect["dx"], rect["dy"]
        raw_val = raw_values[i]
        pct = val / total * 100
        is_removal = raw_val < 0
        bg = palette[i % len(palette)]

        fancy = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0",
            facecolor=bg,
            edgecolor="white",
            linewidth=0,
        )
        ax.add_patch(fancy)

        if is_removal:
            hatch_patch = FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=0",
                facecolor="none",
                edgecolor=fg(bg),
                linewidth=0,
                hatch="//",
                alpha=0.3,
            )
            ax.add_patch(hatch_patch)

        sign = "−" if is_removal else ""
        sub_text = f"{sign}{val:.1f} Mt  ·  {pct:.0f}%"

        longest_line = max(name.split("\n"), key=len)
        n_lines = len(name.split("\n"))
        big = w > 14
        cw = 0.75 if big else 0.62
        name_w = len(longest_line) * cw
        has_room_name = w > name_w + 2.5 and h > (5 + 3.5 * (n_lines - 1))
        has_room_sub = w > 10 and h > (10 + 3.5 * (n_lines - 1))

        label_state = "none"

        if has_room_name and has_room_sub:
            label_state = "full"
            name_y = y + h - 5.8 - 3.5 * (n_lines - 1)
            ax.text(
                x + 1.4,
                name_y,
                name,
                fontsize=fontsize + 3 if big else fontsize,
                fontweight=500,
                fontfamily="sans-serif",
                color=fg(bg),
                va="baseline",
                ha="left",
                linespacing=1.15,
            )
            ax.text(
                x + 1.4,
                y + h - 2.2,
                sub_text,
                fontsize=fontsize - 1 if big else fontsize - 3,
                fontweight=600,
                fontfamily="sans-serif",
                color=fg(bg),
                alpha=1,
                va="baseline",
                ha="left",
            )
        elif has_room_name:
            label_state = "name_only"
            ax.text(
                x + w / 2,
                y + h / 2,
                name,
                fontsize=fontsize if big else fontsize - 2,
                fontweight=500,
                fontfamily="sans-serif",
                color=fg(bg),
                va="center",
                ha="center",
                linespacing=1.15,
            )

        label_states.append(label_state)

    ax.set_xlim(-0.3, w + 0.3)
    ax.set_ylim(-0.3, h + 0.3)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

    legend_handles = []
    for i, (name, val) in enumerate(products):
        if label_states[i] == "full":
            continue
        abs_val = abs(val)
        pct = abs_val / total * 100
        sign = "−" if val < 0 else ""
        name_oneline = name.replace("\n", " ")
        label = f"{name_oneline}  {sign}{abs_val:.1f} Mt ({pct:.0f}%)"
        patch = plt.Rectangle((0, 0), 1, 1, fc=palette[i % len(palette)], ec="none")
        legend_handles.append((patch, label))

    if legend_handles:
        ncol = min(len(legend_handles), 2)
        leg = ax.legend(
            [h[0] for h in legend_handles],
            [h[1] for h in legend_handles],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=ncol,
            frameon=False,
            fontsize=fontsize + 1,
            handlelength=1.0,
            handletextpad=0.5,
            columnspacing=1.5,
        )
        for text in leg.get_texts():
            text.set_color("#333333")

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)

    return ax


def plot_scenarios_by_product(
    scenarios_df: pd.DataFrame,
    methane_scenario: str,
    plotting_config: Union[str, dict],
    cases: Optional[List[str]] = None,
    include_unabated: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 7),
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    display_mode: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Layered bar chart of per-product emissions across scenario cases.

    Args:
        scenarios_df: Long-form scenarios output (from scenarios CLI).
        methane_scenario: Which methane scenario to fix (e.g. 'gwp20_alvarez').
        plotting_config: Path to plotting config YAML, or a dict.
        cases: Scenario names to include (plus 'unabated' for the no-CCS case).
            Defaults to all cases defined in the plotting config.
        include_unabated: Add the unabated case if not already in *cases*.
            Set False to suppress it.
        title: Override the chart title from the plotting config.
        figsize: Figure size in inches.
        output_path: If given, save figure to this path.
        ax: If given, plot on this existing Axes.

    Returns:
        (fig, ax) tuple.
    """
    cfg = _load_config(plotting_config)
    case_defs = cfg["cases"]
    back_case = cfg["back_case"]
    product_overrides = cfg.get("product_labels", {})
    width = cfg.get("bar_width", 0.65)
    lw = cfg.get("line_width", 3.5)
    group_map = _build_product_group_map(cfg)

    if cases is None:
        cases = list(case_defs.keys())
    else:
        cases = list(cases)

    if include_unabated and "unabated" not in cases and "unabated" in case_defs:
        cases.append("unabated")
    if not include_unabated and "unabated" in cases:
        cases.remove("unabated")

    plot_df = _build_plot_data(scenarios_df, methane_scenario, cases, group_map)

    product_order = (
        plot_df.groupby("product")["MtCO2e/yr"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    x = np.arange(len(product_order))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    fs = _resolve_fonts(display_mode)

    priority_cases = ("baseline", "unabated")
    draw_order = [c for c in cases if c != back_case and c not in priority_cases]
    for pc in priority_cases:
        if pc in cases:
            draw_order.append(pc)
    draw_order = [back_case] + draw_order if back_case in cases else draw_order

    zorder_map = {back_case: 6, "baseline": 4, "unabated": 5}

    for case in draw_order:
        case_cfg = case_defs[case]
        vals = []
        for p in product_order:
            match = plot_df.loc[
                (plot_df["product"] == p) & (plot_df["case"] == case), "MtCO2e/yr"
            ]
            vals.append(match.values[0] if len(match) else 0)

        if case == back_case:
            bar_color = case_cfg.get("bar_color", "#bbbbbb")
            ax.bar(
                x,
                vals,
                width=width,
                color=bar_color,
                edgecolor="none",
                alpha=0.9,
                zorder=2,
            )
            for j, val in enumerate(vals):
                ax.plot(
                    [x[j] - width / 2, x[j] + width / 2],
                    [val, val],
                    color=case_cfg["color"],
                    linewidth=lw,
                    solid_capstyle="butt",
                    zorder=zorder_map[case],
                )
        else:
            z = zorder_map.get(case, 3)
            for j, val in enumerate(vals):
                ax.plot(
                    [x[j] - width / 2, x[j] + width / 2],
                    [val, val],
                    color=case_cfg["color"],
                    linewidth=lw,
                    solid_capstyle="butt",
                    zorder=z,
                )

    for j, product in enumerate(product_order):
        max_val = plot_df.loc[plot_df["product"] == product, "MtCO2e/yr"].max()
        ax.text(
            x[j],
            max_val + 0.15,
            f"{max_val:.1f}",
            ha="center",
            va="bottom",
            fontsize=fs["bar_label"],
            fontfamily="Arial",
            fontweight="bold",
            zorder=6,
        )

    handles = []
    handler_map = {}
    for case in cases:
        case_cfg = case_defs[case]
        label = _legend_label(case_cfg)
        if case == back_case:
            proxy = Line2D([], [], label=label)
            handler_map[proxy] = _BarWithLineHandler(
                bar_color=case_cfg.get("bar_color", "#bbbbbb"),
                line_color=case_cfg["color"],
                alpha=0.9,
                line_width=lw,
            )
            handles.append(proxy)
        else:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=case_cfg["color"],
                    linewidth=lw,
                    label=label,
                )
            )

    ax.legend(
        handles=handles,
        handler_map=handler_map,
        fontsize=fs["legend"],
        frameon=False,
        loc="upper right",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [_product_label(p, product_overrides) for p in product_order],
        ha="center",
        fontsize=fs["tick"],
        fontfamily="Arial Narrow",
    )
    ax.set_ylabel(
        "MtCO2e/yr", fontsize=fs["axis_label"], fontfamily="Arial", labelpad=12
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.tick_params(axis="y", labelsize=fs["tick"])

    chart_title = title or cfg.get(
        "title", "CCS: GHG emissions by product and scenario"
    )
    ax.set_title(chart_title, fontsize=fs["title"], fontfamily="Arial", pad=20)

    ax.grid(axis="y", color="lightgray", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(bottom=False, left=False)
    ax.set_ylim(0, plot_df["MtCO2e/yr"].max() * 1.2)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)

    return fig, ax


def _compute_state_totals(
    scenarios_df: pd.DataFrame,
    methane_scenario: str,
    cases: List[str],
) -> Tuple[Dict[str, float], float]:
    """Sum emissions across all products for each case and for unabated."""
    df = scenarios_df[scenarios_df["methane_scenario"] == methane_scenario]

    no_ccs_value = (
        df[["uid", "unabated_mtco2e_per_yr"]]
        .drop_duplicates(subset="uid")["unabated_mtco2e_per_yr"]
        .sum()
    )

    totals: Dict[str, float] = {}
    for case in cases:
        totals[case] = df[df["scenario"] == case][
            "mtco2e_per_yr_emissions_including_ccs"
        ].sum()

    return totals, no_ccs_value


def plot_ccs_emissions_factors(
    emissions_df: pd.DataFrame,
    plotting_config: Union[str, dict],
    methane_scenario: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 7),
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    display_mode: Optional[str] = None,
):
    fs = _resolve_fonts(display_mode)
    cfg = _load_config(plotting_config)["emissions_factor"]

    methane_case = (
        methane_scenario if methane_scenario is not None else cfg["methane_case"]
    )

    factor_col = "_emitted_per_tco2_ccs"
    ylabel = " emitted per tCO2 injected with CCS"
    if cfg["tco2e"]:
        factor_col = "tco2e" + factor_col
        ylabel = "tCO2e" + ylabel
    else:
        factor_col = "tco2" + factor_col
        ylabel = "tCO2" + ylabel
    if cfg["net"]:
        factor_col = "net_" + factor_col
        ylabel = "Net " + ylabel

    plot_df = emissions_df.loc[
        (emissions_df["methane_scenario"] == methane_case)
        & (emissions_df["co2_source"].isin(cfg["industries"]))
    ][[factor_col, "co2_source", "co2_destination"]]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    plot_df.sort_values(
        by=["co2_destination", factor_col], ascending=False, inplace=True
    )
    order = plot_df.loc[plot_df["co2_destination"] == "eor"]["co2_source"]

    sns.barplot(
        ax=ax,
        data=plot_df,
        x="co2_source",
        y=factor_col,
        hue="co2_destination",
        order=order,
        width=0.4,
    )
    replace_dict = cfg["name_replacements"]
    ax.set_xticklabels(
        [
            (
                replace_dict[i.get_text()]
                if i.get_text() in replace_dict.keys()
                else i.get_text().replace("_", " ").title()
            )
            for i in ax.get_xticklabels()
        ],
        fontsize=fs["tick"],
        fontfamily="Arial",
    )
    ax.tick_params(axis="y", labelsize=fs["tick"])
    ax.plot([-0.5, len(cfg["industries"]) - 0.5], [0, 0], lw=1, color="#555555")

    n_ind = len(cfg["industries"])
    xlim_right = n_ind - 0.5
    ax.set_xlim([-0.5, xlim_right + 0.8])
    ax.annotate(
        "Net zero\nemissions",
        xy=(xlim_right + 0.15, 0),
        xytext=(xlim_right + 0.15, ax.get_ylim()[0] * 0.55),
        fontsize=fs["annotation"],
        fontfamily="Arial",
        fontstyle="italic",
        color="#555555",
        ha="left",
        va="top",
        arrowprops={"arrowstyle": "-|>", "color": "#555555", "lw": 1.2},
    )

    legend = ax.get_legend()
    if legend is not None:
        legend.set_title(r"CO$_2$ destination")
        legend.get_title().set_fontsize(fs["legend_title"])
        hue_labels = {
            "eor": r"CO$_2$ destination is EOR",
            "gs": r"CO$_2$ destination is geologic storage",
        }
        for text in legend.get_texts():
            text.set_text(hue_labels.get(text.get_text(), text.get_text()))
            text.set_fontsize(fs["legend"])

    if title is not None:
        ax.set_title(title, fontsize=fs["title"], fontfamily="Arial", pad=20)
    ax.set_ylabel(ylabel, fontsize=fs["axis_label"], fontfamily="Arial")
    ax.set_xlabel("")
    ax.grid(axis="y", zorder=-1)
    ax.set_axisbelow(True)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=300)

    return fig, ax


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


def plot_emissions_breakdown(
    industrial_emissions_df: pd.DataFrame,
    ccs_emissions_df: pd.DataFrame,
    methane_scenario: str,
    co2_destination: str = "eor",
    industries: Optional[List[str]] = None,
    which_flavor: Optional[List[str]] = None,
    name_replacements: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 8),
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    display_mode: Optional[str] = None,
    ghost_in_legend: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Stacked bar chart breaking down net emissions per tCO2 captured by component.

    Each industry gets one stacked bar. Positive segments stack upward,
    negative segments (captured CO2) stack downward from zero.

    Args:
        industrial_emissions_df: Output of industrial emissions pipeline,
            with per-industry CO2/CO2e/CH4 columns and capture fractions.
        ccs_emissions_df: Output of CCS emissions pipeline, with per-
            (co2_source, co2_destination) lifecycle factors.
        methane_scenario: Which methane scenario to use.
        co2_destination: 'eor' or 'gs'.
        industries: Subset of industries to plot (simple_product_name values).
        which_flavor: Optional list of ``industrial_product`` values to
            select (e.g. ``['power (CCGT)', 'natural gas processing']``).
            When omitted, the first row per ``simple_product_name`` is used.
        name_replacements: Display name overrides (e.g. {'ngp': 'NGP'}).
        title: Chart title override.
        figsize: Figure size.
        output_path: Save path.
        ax: Existing axes.

    Returns:
        (fig, ax) tuple.
    """
    fs = _resolve_fonts(display_mode)
    df = _build_breakdown_data(
        industrial_emissions_df,
        ccs_emissions_df,
        methane_scenario,
        co2_destination,
        industries,
        which_flavor,
    )

    component_cols = [col for col, _, _ in _BREAKDOWN_COMPONENTS]
    df["_net"] = df[component_cols].sum(axis=1)
    df = df.sort_values("_net", ascending=False)
    industry_order = df["simple_product_name"].tolist()

    x = np.arange(len(industry_order))
    width = 0.55

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    pos_bottom = np.zeros(len(x))
    neg_bottom = np.zeros(len(x))

    for col, label, color in _BREAKDOWN_COMPONENTS:
        vals = df.set_index("simple_product_name").loc[industry_order, col].values
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

    ccs_top = pos_bottom.copy()

    ghost_vals = np.ones(len(x))
    ax.bar(
        x,
        ghost_vals,
        width=width,
        bottom=pos_bottom,
        facecolor="#c8e6c9",
        alpha=0.5,
        edgecolor="#1b5e20",
        linewidth=2.5,
        linestyle="--",
        hatch="///",
        label=r"$\mathbf{CO_2}$ captured by CCS" if ghost_in_legend else None,
        zorder=2,
    )
    pos_bottom += ghost_vals

    # In-figure annotation (used when the ghost isn't in the legend). An
    # arrow drops from the label onto the top edge of the first ghost box;
    # the label itself sits above-and-to-the-right of that bar so the arrow
    # points down into the dashed green block.
    if not ghost_in_legend:
        _ghost_green = "#1b5e20"
        j_tgt = 0
        top_of_box_y = pos_bottom[j_tgt]
        text_x = x[j_tgt] + 0.45
        text_y = top_of_box_y + 0.6
        ax.annotate(
            r"CO$_2$ captured by CCS",
            xy=(x[j_tgt], top_of_box_y),
            xytext=(text_x, text_y),
            fontsize=fs["annotation"],
            fontfamily="Arial",
            fontweight="bold",
            color=_ghost_green,
            ha="left",
            va="center",
            arrowprops={
                "arrowstyle": "->",
                "color": _ghost_green,
                "linewidth": 1.5,
                "shrinkA": 2,
                "shrinkB": 0,
            },
            zorder=5,
            path_effects=[pe.withStroke(linewidth=4, foreground="white")],
        )

    ax.axhline(
        0, color="black", linewidth=1.0, zorder=3, xmin=-0.03, xmax=1.03, clip_on=False
    )

    for j in range(len(x)):
        net_val = df.iloc[j]["_net"] + 1
        y_pos = ccs_top[j] + 0.05
        ax.text(
            x[j],
            y_pos,
            f"{net_val:.1f}",
            ha="center",
            va="bottom",
            fontsize=fs["bar_label"],
            fontfamily="Arial",
            fontweight="bold",
            zorder=5,
            path_effects=[
                pe.withStroke(linewidth=2, foreground="white"),
            ],
        )

    replacements = {**_FACILITY_NAME_DEFAULTS, **(name_replacements or {})}
    labels = [
        textwrap.fill(replacements.get(ind, ind.replace("_", " ").title()), width=14)
        for ind in industry_order
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fs["tick"], fontfamily="Arial")
    ax.set_ylabel(
        r"tCO$_2$e per tCO$_2$ injected with CCS",
        fontsize=fs["axis_label"],
        fontfamily="Arial",
        labelpad=12,
    )

    y_min = neg_bottom.min()
    y_max = pos_bottom.max() * 1.08
    ax.set_ylim(y_min - 0.3, y_max)
    yticks = ax.get_yticks()
    if -1 not in yticks:
        yticks = sorted(set(list(yticks) + [-1]))
        ax.set_yticks(yticks)
    ax.tick_params(axis="y", labelsize=fs["tick"])

    chart_title = title or (
        r"Emissions breakdown per tCO$_2$ captured"
        f" ({co2_destination.upper()} destination)"
    )
    ax.set_title(chart_title, fontsize=fs["title"], fontfamily="Arial", pad=20)

    ax.grid(axis="y", color="lightgray", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False)

    handles, labels_leg = ax.get_legend_handles_labels()
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels_leg):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)
    ax.legend(
        unique_handles,
        unique_labels,
        fontsize=fs["legend"],
        frameon=True,
        facecolor="white",
        edgecolor="none",
        loc="upper right",
    )

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)

    return fig, ax


def plot_emissions_breakdown_eor_gs(
    industrial_emissions_df: pd.DataFrame,
    ccs_emissions_df: pd.DataFrame,
    methane_scenario: str,
    industries: Optional[List[str]] = None,
    which_flavor: Optional[List[str]] = None,
    name_replacements: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (24, 8),
    output_path: Optional[str] = None,
    display_mode: Optional[str] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    fig, (ax_gs, ax_eor) = plt.subplots(1, 2, figsize=figsize)

    plot_emissions_breakdown(
        industrial_emissions_df,
        ccs_emissions_df,
        methane_scenario,
        co2_destination="gs",
        industries=industries,
        which_flavor=which_flavor,
        name_replacements=name_replacements,
        title=r"Geologic Storage (GS)",
        ax=ax_gs,
        display_mode=display_mode,
    )

    plot_emissions_breakdown(
        industrial_emissions_df,
        ccs_emissions_df,
        methane_scenario,
        co2_destination="eor",
        industries=industries,
        which_flavor=which_flavor,
        name_replacements=name_replacements,
        title=r"Enhanced Oil Recovery (EOR)",
        ax=ax_eor,
        display_mode=display_mode,
    )

    ax_gs.get_legend().remove()
    handles, labels_leg = ax_eor.get_legend_handles_labels()
    ax_eor.get_legend().remove()

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    fig.legend(
        handles,
        labels_leg,
        loc="lower center",
        ncol=4,
        fontsize=_resolve_fonts(display_mode)["legend"],
        frameon=True,
        facecolor="white",
        edgecolor="none",
        bbox_to_anchor=(0.5, 0.01),
    )

    if title:
        fig.suptitle(
            title,
            fontsize=_resolve_fonts(display_mode)["title"],
            fontfamily="Arial",
            y=1.02,
        )

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=300)

    return fig, (ax_gs, ax_eor)


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
    bar_width = 0.55
    y_headroom = 1.12  # multiplier over tallest bar for ylim top
    y_floor = -1.5  # captured sums to -1; leave a little margin
    label_x_nudge = 0.03  # bar-top label horizontal offset
    arrow_x_offset = 0.55  # arrows drawn this far left of first bar
    zero_line_overhang = 0.03  # zero line extends past axis edges

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
    base_bar_kwargs = {
        "width": bar_width,
        "edgecolor": "white",
        "linewidth": 1.2,
        "zorder": 2,
    }
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
    eor_color = "#b71c1c"
    eor_border_lw = 1.5
    eor_arrow_lw = 2.0
    for j in range(len(x)):
        inc = eor_increment[j]
        if inc <= 0:
            continue
        y_bottom = pos_bottom[j]
        y_top = pos_bottom[j] + inc
        x_left = x[j] - bar_width / 2
        x_right = x[j] + bar_width / 2
        ax.plot(
            [x_left, x_right],
            [y_bottom, y_bottom],
            color=eor_color,
            linewidth=eor_border_lw,
            zorder=2,
            solid_capstyle="butt",
        )
        ax.plot(
            [x_left, x_right],
            [y_top, y_top],
            color=eor_color,
            linewidth=eor_border_lw,
            zorder=2,
            solid_capstyle="butt",
        )
        ax.annotate(
            "",
            xy=(x[j], y_top),
            xytext=(x[j], y_bottom),
            arrowprops={
                "arrowstyle": "-|>",
                "color": eor_color,
                "lw": eor_arrow_lw,
                "mutation_scale": 12,
            },
            zorder=2,
        )
    # Proxy artist so this shape gets a labeled entry in the legend, rendered
    # via _IShapeHandler below.
    eor_proxy = Line2D(
        [],
        [],
        color=eor_color,
        lw=eor_border_lw,
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
        i_right = x[lng_j] + bar_width / 2
        i_label_x_nudge = 0.08
        ax.text(
            i_right + i_label_x_nudge,
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
            i_right + i_label_x_nudge,
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
        xmin=-zero_line_overhang,
        xmax=1 + zero_line_overhang,
        clip_on=False,
    )

    # ── 5. Bar-top labels (emitted totals for GS and EOR) ─────────────────
    # `_net + 1` removes the -1 from the captured slice, so the label reflects
    # only the emitted portion (positive contributions), not the net.
    # Each numeric label is paired with a smaller "(GS)" or "(EOR)" suffix
    # placed just beneath it so the reader can tell the two totals apart.
    label_x = x - bar_width / 2 - label_x_nudge
    label_style = {
        "ha": "right",
        "va": "center",
        "fontsize": fs["bar_label"] - 1,
        "fontfamily": "Arial",
        "fontweight": "bold",
        "zorder": 5,
        "path_effects": [pe.withStroke(linewidth=2, foreground="white")],
    }
    suffix_style = {
        "ha": "right",
        "va": "top",
        "fontsize": fs["bar_label"] - 2,
        "fontfamily": "Arial",
        "fontweight": "normal",
        "zorder": 5,
        "xycoords": "data",
        "textcoords": "offset points",
        "xytext": (0, -8),
        "path_effects": [pe.withStroke(linewidth=2, foreground="white")],
    }
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
    y_max = eor_top.max() * y_headroom
    ax.set_ylim(y_floor, y_max)
    yticks = [t for t in ax.get_yticks() if y_floor <= t <= y_max]
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
    arrow_x = x[0] - arrow_x_offset
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
        handler_map={eor_proxy: _IShapeHandler(color=eor_color)},
        handleheight=2.4,  # taller swatch so the "I" reads as I-shaped
    )

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
    return fig, ax


def plot_state_emissions_breakdown(
    industrial_emissions_df: pd.DataFrame,
    ccs_emissions_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    methane_scenario: str,
    destination_scenarios: Optional[List[Union[str, dict]]] = None,
    industries: Optional[List[str]] = None,
    which_flavor: Optional[List[str]] = None,
    title: Optional[str] = None,
    legend_below: bool = False,
    figsize: Tuple[int, int] = (12, 8),
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    display_mode: Optional[str] = None,
    wrapwidth: int = 14,  # wrap x-tick labels at this many characters
) -> Tuple[plt.Figure, plt.Axes]:
    """Stacked bar chart of state-total emissions breakdown per tCO2 captured.

    Computes a captured-CO2-weighted average of per-industry component
    factors across all projects, with one bar per destination-mix scenario.
    """
    fs = _resolve_fonts(display_mode)

    known_dest_scenarios = {
        "all_gs": {
            "name": "all_gs",
            "label": "100% Geologic Storage",
            "eor_pct": 0,
        },
        "baseline": {
            "name": "baseline",
            "label": "Announced destinations",
            "from_data": True,
        },
        "current_mix": {
            "name": "current_mix",
            "label": "Current U.S. practice",
            "eor_pct": 92,
        },
        "nameplate_implied": {
            "name": "nameplate_implied",
            "label": "Nameplate 'capture capacity'",
            "zero_lifecycle": True,
        },
        "all_eor": {
            "name": "all_eor",
            "label": "100% EOR",
            "eor_pct": 100,
        },
        "switching": {
            "name": "switching",
            "label": "Stated →\n100% EOR after 45Q",
            "from_data": True,
            "is_switching": True,
        },
        "switching_from_current": {
            "name": "switching_from_current",
            "label": "Current →\n100% EOR after 45Q",
            "from_data": True,
            "is_switching": True,
        },
    }

    if destination_scenarios is None:
        destination_scenarios = list(known_dest_scenarios.values())
    else:
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
    component_cols = [col for col, _, _ in _BREAKDOWN_COMPONENTS]
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
            total_cap = scen_df["mtco2_per_yr_captured_with_ccs"].sum()
            if total_cap > 0 and "eor_fraction" in scen_df.columns:
                avg_eor = (
                    scen_df["eor_fraction"] * scen_df["mtco2_per_yr_captured_with_ccs"]
                ).sum() / total_cap
                eor_pct = avg_eor * 100
                row["label"] += f" ({eor_pct:.0f}% EOR; {100 - eor_pct:.0f}% GS)"
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
                row[col] = weighted / total_cap if total_cap else 0.0
        elif zero_lc:
            scen_df = scenarios_df[
                (scenarios_df["methane_scenario"] == methane_scenario)
                & (scenarios_df["scenario"] == scen["name"])
            ]
            scen_df = scen_df[scen_df["product"].isin(common)]
            total_cap = scen_df["mtco2_per_yr_captured_with_ccs"].sum()
            if total_cap > 0 and "eor_fraction" in scen_df.columns:
                avg_eor = (
                    scen_df["eor_fraction"] * scen_df["mtco2_per_yr_captured_with_ccs"]
                ).sum() / total_cap
                eor_pct = avg_eor * 100
                row["label"] += f" ({eor_pct:.0f}% EOR; {100 - eor_pct:.0f}% GS)"
            for col in component_cols:
                if col in ccs_cols:
                    row[col] = 0.0
                else:
                    weighted = sum(
                        eor_df.loc[ind, col] * captured_by_industry[ind]
                        for ind in captured_by_industry.index
                    )
                    row[col] = weighted / total_captured if total_captured else 0.0
        else:
            eor_frac = scen["eor_pct"] / 100
            gs_frac = 1 - eor_frac
            eor_pct = scen["eor_pct"]
            if "EOR" not in row["label"] and "GS" not in row["label"]:
                row["label"] += f" ({eor_pct:.0f}% EOR; {100 - eor_pct:.0f}% GS)"
            for col in component_cols:
                weighted = sum(
                    (eor_frac * eor_df.loc[ind, col] + gs_frac * gs_df.loc[ind, col])
                    * captured_by_industry[ind]
                    for ind in captured_by_industry.index
                )
                row[col] = weighted / total_captured if total_captured else 0.0

        row["_net"] = sum(row[c] for c in component_cols)
        records.append(row)

    result_df = (
        pd.DataFrame(records)
        .sort_values("_net", ascending=False)
        .reset_index(drop=True)
    )

    x = np.arange(len(result_df))
    width = 0.45

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    pos_bottom = np.zeros(len(x))
    neg_bottom = np.zeros(len(x))

    for col, label, color in _BREAKDOWN_COMPONENTS:
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

    ccs_top = pos_bottom.copy()

    ax.axhline(
        0, color="black", linewidth=1.0, zorder=3, xmin=-0.03, xmax=1.03, clip_on=False
    )

    for j in range(len(x)):
        net_val = result_df.iloc[j]["_net"]
        y_pos = ccs_top[j] + 0.05
        ax.text(
            x[j],
            y_pos,
            f"{net_val:.1f}",
            ha="center",
            va="bottom",
            fontsize=fs["bar_label"],
            fontfamily="Arial",
            fontweight="bold",
            zorder=5,
            path_effects=[
                pe.withStroke(linewidth=4, foreground="white"),
            ],
        )

    ax.set_xticks(x)
    wrapped_labels = [
        "\n".join(textwrap.fill(line, width=wrapwidth) for line in lbl.split("\n"))
        for lbl in result_df["label"].tolist()
    ]
    ax.set_xticklabels(
        wrapped_labels,
        fontsize=fs["tick"],
        fontfamily="Arial",
    )
    ax.set_ylabel(
        r"tCO$_2$e emitted per tCO$_2$ injected",
        fontsize=fs["axis_label"],
        fontfamily="Arial",
        labelpad=12,
    )

    ax.set_ylim(-1.5, 5)
    ax.tick_params(axis="y", labelsize=fs["tick"])

    chart_title = title or (r"Texas CCS: state-total emissions per tCO$_2$ injected")
    ax.set_title(chart_title, fontsize=fs["title"], fontfamily="Arial", pad=20)

    ax.grid(axis="y", color="lightgray", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False)

    arrow_x = x[-1] + 0.45
    ax.annotate(
        "",
        xy=(arrow_x, -1),
        xytext=(arrow_x, 0),
        clip_on=False,
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#2e7d32",
            "lw": 3,
            "mutation_scale": 15,
        },
    )
    ax.text(
        arrow_x + 0.08,
        -0.5,
        "Injected",
        fontsize=fs["annotation"],
        fontfamily="Arial",
        weight="semibold",
        color="#2e7d32",
        ha="left",
        va="center",
        clip_on=False,
    )
    ax.annotate(
        "",
        xy=(arrow_x, 1),
        xytext=(arrow_x, 0),
        clip_on=False,
        arrowprops={
            "arrowstyle": "-|>",
            "color": "black",
            "lw": 3,
            "mutation_scale": 15,
        },
    )
    ax.text(
        arrow_x + 0.08,
        0.5,
        "Emitted",
        fontsize=fs["annotation"],
        fontfamily="Arial",
        weight="semibold",
        color="black",
        ha="left",
        va="center",
        clip_on=False,
    )

    handles, labels_leg = ax.get_legend_handles_labels()
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels_leg):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)
    if legend_below:
        ax.legend(
            unique_handles,
            unique_labels,
            fontsize=fs["legend"],
            frameon=True,
            facecolor="white",
            edgecolor="none",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.35),
            ncol=3,
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
            bbox_to_anchor=(1.02, 1),
        )
        fig.tight_layout(rect=[0, 0, 0.72, 1])
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)

    return fig, ax


def plot_state_emissions_breakdown_combined(
    industrial_emissions_df: pd.DataFrame,
    ccs_emissions_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    methane_scenario: str,
    destination_scenarios: List[Union[str, dict]],
    industries: Optional[List[str]] = None,
    which_flavor: Optional[List[str]] = None,
    left_title: Optional[str] = None,
    right_title: Optional[str] = None,
    figsize: Tuple[int, int] = (22, 8),
    output_path: Optional[str] = None,
    display_mode: Optional[str] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Combined 1x2 figure: total MtCO2e/yr (left) and per-tCO2 breakdown (right)."""
    fs = _resolve_fonts(display_mode)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize)

    plot_state_total_emissions_breakdown_lumped_comparison(
        industrial_emissions_df,
        ccs_emissions_df,
        scenarios_df,
        methane_scenario,
        destination_scenarios=destination_scenarios,
        industries=industries,
        which_flavor=which_flavor,
        title=left_title,
        print_mode=True,
        ax=ax_left,
        display_mode=display_mode,
    )

    plot_state_emissions_breakdown(
        industrial_emissions_df,
        ccs_emissions_df,
        scenarios_df,
        methane_scenario,
        destination_scenarios=destination_scenarios,
        industries=industries,
        which_flavor=which_flavor,
        title=right_title,
        ax=ax_right,
        display_mode=display_mode,
    )

    for a in (ax_left, ax_right):
        leg = a.get_legend()
        if leg is not None:
            leg.remove()

    right_lo, right_hi = -1.5, 5.0
    zero_frac = abs(right_lo) / (right_hi - right_lo)
    left_lo, left_hi = ax_left.get_ylim()
    left_range_below = abs(left_lo) if left_lo < 0 else 0
    left_range_above = left_hi
    desired_below = zero_frac * (left_range_above / (1 - zero_frac))
    if desired_below > left_range_below:
        ax_left.set_ylim(-desired_below, left_hi)
    else:
        desired_above = (1 - zero_frac) * (left_range_below / zero_frac)
        ax_left.set_ylim(left_lo, max(left_hi, desired_above))

    handles_all, labels_all = [], []
    for a in (ax_left, ax_right):
        h, l = a.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels_all:
                handles_all.append(hi)
                labels_all.append(li)

    fig.legend(
        handles_all,
        labels_all,
        fontsize=fs["legend"],
        frameon=True,
        facecolor="white",
        edgecolor="none",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)

    return fig, (ax_left, ax_right)


# Map from `project_state` values (in the CSV/scenarios data) to display names
# used in titles and legends. Extend as new states are added.
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


def plot_state_emissions_dual_axis(
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
) -> Tuple[plt.Figure, plt.Axes]:
    """Stacked bars of state-total emissions with dual y-axes (MtCO2e/yr and per-tCO2).

    Args:
        states: Optional filter on `project_state`. If provided, `scenarios_df`,
            `industrial_emissions_df`, and `ccs_emissions_df` are pre-filtered
            to rows whose project_state is in this list (dataframes that lack
            a project_state column are passed through unchanged). If None, all
            rows are used. The chart title is auto-generated from the resolved
            state list unless `title` is explicitly given.
    """
    display_mode = _resolve_display_mode(display_mode)
    fs = _resolve_fonts(display_mode)

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
        "all_gs": {
            "name": "all_gs",
            "label": "100% Geologic storage",
            "eor_pct": 0,
        },
        "baseline": {
            "name": "baseline",
            "label": "Announced destinations",
            "from_data": True,
        },
        "current_mix": {
            "name": "current_mix",
            "label": "Current U.S. practice",
            "eor_pct": 92,
        },
        "nameplate_implied": {
            "name": "nameplate_implied",
            "label": "Nameplate 'capture capacity'",
            "zero_lifecycle": True,
        },
        "all_eor": {
            "name": "all_eor",
            "label": "100% EOR",
            "eor_pct": 100,
        },
        "switching": {
            "name": "switching",
            "label": "Stated →\n100% EOR after 45Q",
            "from_data": True,
            "is_switching": True,
        },
        "switching_from_current": {
            "name": "switching_from_current",
            "label": "Current →\n100% EOR after 45Q",
            "from_data": True,
            "is_switching": True,
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

    component_cols = [col for col, _, _ in _BREAKDOWN_COMPONENTS]

    # Aggregate across states correctly: run the per-scenario record computation
    # once per state so each state's captured tonnes are weighted by its OWN
    # per-industry factors (grid intensity, coal EF, etc.), then sum the
    # per-state results. If project_state is absent, treat all input as one slice.
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
                destination_scenarios,
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
                destination_scenarios,
            )
        )

    # Sum the per-state records into aggregate records (one per destination scenario).
    # Then fold in DAC contributions (which _compute_dual_axis_records excludes
    # because DAC has zero per-unit-of-product emissions — the standard
    # per-unit-captured math can't handle it).
    records = []
    for i, scen in enumerate(destination_scenarios):
        combined = {"scenario": scen["name"], "label": scen["label"]}
        for col in component_cols + ["_captured"]:
            combined[col] = sum(r[i][col] for r in per_state_records)
            combined[col] += sum(d[i].get(col, 0.0) for d in per_state_dac)
        combined["_net"] = sum(combined[c] for c in component_cols)
        records.append(combined)

    # Secondary y-axis denominator = ground-truth captured across the same
    # methane scenario. This includes DAC and matches the per-bar label math
    # below (which reads mtco2_per_yr_captured_with_ccs from scenarios_df).
    _truth_scen = scenarios_df[scenarios_df["methane_scenario"] == methane_scenario]
    if states is not None and "project_state" in _truth_scen.columns:
        _truth_scen = _truth_scen[_truth_scen["project_state"].isin(states)]
    # Average captured across the destination-scenario weightings; they should
    # be equal (same underlying projects) but from_data can differ.
    per_scen_cap = [
        _truth_scen[_truth_scen["scenario"] == s["name"]][
            "mtco2_per_yr_captured_with_ccs"
        ].sum()
        for s in destination_scenarios
    ]
    total_captured = sum(per_scen_cap) / max(len(per_scen_cap), 1)

    result_df = (
        pd.DataFrame(records)
        .sort_values("_net", ascending=False)
        .reset_index(drop=True)
    )

    for i, row in result_df.iterrows():
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
                    scen_df["eor_fraction"] * scen_df["mtco2_per_yr_captured_with_ccs"]
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

    for col, label, color in _BREAKDOWN_COMPONENTS:
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

    captured_vals = result_df["_captured"].values
    ax.bar(
        x,
        captured_vals,
        width=width,
        bottom=pos_bottom,
        facecolor="none",
        edgecolor="#2e7d32",
        linewidth=1.5,
        linestyle="--",
        zorder=2,
        label=r"$\mathbf{CO_2}$ captured by CCS",
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

    unabated_df = scenarios_df[
        scenarios_df["methane_scenario"] == methane_scenario
    ].drop_duplicates(subset="uid")
    baseload_val = unabated_df["unabated_mtco2e_per_yr"].sum()
    ax.axhline(
        baseload_val,
        color="black",
        linewidth=1.5,
        linestyle="--",
        zorder=3,
        label="Total unabated emissions",
    )

    for j in range(len(x)):
        scen_name = result_df.iloc[j]["scenario"]
        scen_rows = scenarios_df[
            (scenarios_df["methane_scenario"] == methane_scenario)
            & (scenarios_df["scenario"] == scen_name)
        ]
        net_mt = scen_rows["mtco2e_per_yr_emissions_including_ccs"].sum()
        captured_j = scen_rows["mtco2_per_yr_captured_with_ccs"].sum()
        per_tco2 = net_mt / captured_j if captured_j else 0
        bar_top = pos_bottom[j]
        ax.text(
            x[j],
            bar_top + 10,
            f"({per_tco2:.1f}" + r" tCO$_2$e/tCO$_2$ injected)",
            ha="center",
            va="bottom",
            fontsize=fs["bar_label"] - 1,
            fontfamily="Arial Narrow",
            fontweight="normal",
            zorder=5,
            path_effects=[pe.withStroke(linewidth=5, foreground="white")],
        )
        ax.text(
            x[j],
            bar_top + 45,
            f"Net: {net_mt:.0f}" + r" MtCO$_2$e/yr",
            ha="center",
            va="bottom",
            fontsize=fs["bar_label"],
            fontfamily="Arial",
            fontweight="bold",
            zorder=5,
            path_effects=[pe.withStroke(linewidth=5, foreground="white")],
        )
    _arrow_x = x[0] - 0.55
    # ax.annotate(
    #    "",
    #    xy=(arrow_x, -40),
    #    xytext=(arrow_x, 0),
    #    clip_on=False,
    #    annotation_clip=False,
    #    arrowprops={
    #        "arrowstyle": "-|>",
    #        "color": "#2e7d32",
    #        "lw": 2,
    #        "mutation_scale": 15,
    #    },
    # )
    # ax.text(
    #    arrow_x+.01,
    #    -15,
    #    "Injected",
    #    fontsize=fs["annotation"],
    #    fontfamily="Arial",
    #    weight="normal",
    #    color="#2e7d32",
    #    ha="left",
    #    va="top",
    #    clip_on=False,path_effects=[pe.withStroke(linewidth=3, foreground="white")]
    # )
    # ax.annotate(
    #    "",
    #    xy=(arrow_x, 40),
    #    xytext=(arrow_x, 0),
    #    clip_on=False,
    #    annotation_clip=False,
    #    arrowprops={
    #        "arrowstyle": "-|>",
    #        "color": "black",
    #        "lw": 2,
    #        "mutation_scale": 15,
    #    },
    # )
    # ax.text(
    #    arrow_x+.01,
    #    15,
    #    "Emitted",
    #    fontsize=fs["annotation"],
    #    fontfamily="Arial",
    #    weight="normal",
    #    color="black",
    #    ha="left",
    #    va="bottom",
    #    clip_on=False,path_effects=[pe.withStroke(linewidth=3, foreground="white")],
    # )
    ax.set_xticks(x)
    wrapped_labels = [
        "\n".join(textwrap.fill(line, width=20) for line in lbl.split("\n"))
        for lbl in result_df["label"].tolist()
    ]
    ax.set_xticklabels(wrapped_labels, fontsize=fs["tick"], fontfamily="Arial")
    ax.set_ylabel(
        r"MtCO$_2$e / yr",
        fontsize=fs["axis_label"],
        fontfamily="Arial",
        labelpad=1,
    )
    y_min = neg_bottom.min()
    y_max = max(ghost_top.max(), baseload_val) * 1.20
    ax.set_ylim(y_min - 1 if y_min < 0 else -1, y_max)
    ax.tick_params(axis="y", labelsize=fs["tick"])

    if total_captured > 0:
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
        # Infer state list from filter (if any) or from the data itself.
        if states is not None:
            title_states = states
        elif "project_state" in scenarios_df.columns:
            title_states = sorted(
                scenarios_df["project_state"].dropna().unique().tolist()
            )
        else:
            title_states = None
        state_label = _format_states_for_title(title_states)
        title = f"{state_label} CCS: state-total emissions (MtCO$_2$e/yr)"
    ax.set_title(title, fontsize=fs["title"], fontfamily="Arial", pad=20)

    ax.grid(axis="y", color="lightgray", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False)

    handles, labels_leg = ax.get_legend_handles_labels()
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels_leg):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)
    if legend_below:
        ax.legend(
            unique_handles,
            unique_labels,
            fontsize=fs["legend"],
            frameon=True,
            facecolor="white",
            edgecolor="none",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=3,
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


def plot_state_emissions_compared_to_unabated_on_dual_axis(
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
) -> Tuple[plt.Figure, plt.Axes]:
    """Same as `plot_state_emissions_dual_axis` but accepts the pseudo-scenario
    keyword ``"unabated"`` in `destination_scenarios`. An unabated bar shows
    the state-total emissions if no CCS were deployed, stacked from the four
    non-CCS components (process CH4, combustion CH4, uncaptured process CO2,
    uncaptured combustion CO2). It skips the green dashed captured-tonnes
    overlay (nothing captured) and the "(N.N tCO2e/tCO2 injected)" subscript
    (no captured tonnes to divide by); only the "Net: MtCO2e/yr" label is
    drawn above the bar.

    Bars are drawn in the order given in `destination_scenarios` (leftmost
    list entry → leftmost bar); no re-sorting by net emissions. The black
    horizontal "Total unabated emissions" reference line from the sibling
    function is removed — the unabated bar itself supersedes it.

    Args:
        destination_scenarios: Same as the sibling function, with the added
            keyword ``"unabated"`` recognized. Any custom dict entry can also
            set ``{"is_unabated": True}`` to mark itself as an unabated bar.
        states: Optional filter on `project_state`. See sibling docstring.
        (all other args behave identically to `plot_state_emissions_dual_axis`)
    """
    display_mode = _resolve_display_mode(display_mode)
    fs = _resolve_fonts(display_mode)

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

    component_cols = [col for col, _, _ in _BREAKDOWN_COMPONENTS]

    # Split scenarios into CCS vs unabated. CCS scenarios go through the
    # per-state _compute_dual_axis_records / _compute_dac_records path; the
    # unabated record is a single aggregate (works across states via its own
    # state-aware fraction lookup).
    ccs_scenarios = [s for s in destination_scenarios if not s.get("is_unabated")]

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

    # Interleave CCS + unabated records back into the caller's order.
    records = []
    ccs_i = 0
    for scen in destination_scenarios:
        row = {
            "scenario": scen["name"],
            "label": scen["label"],
            "_is_unabated": bool(scen.get("is_unabated")),
            "_is_switching": bool(scen.get("is_switching")),
        }
        if scen.get("is_unabated"):
            for col in component_cols + ["_captured"]:
                row[col] = unabated_record.get(col, 0.0)
            row["_net"] = sum(row[c] for c in component_cols)
        else:
            for col in component_cols + ["_captured"]:
                row[col] = sum(r[ccs_i][col] for r in per_state_records)
                row[col] += sum(d[ccs_i].get(col, 0.0) for d in per_state_dac)
            row["_net"] = sum(row[c] for c in component_cols)
            ccs_i += 1
        records.append(row)

    result_df = pd.DataFrame(records)  # keep destination_scenarios order

    # Secondary y-axis denominator (average captured across CCS scenarios).
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

    # EOR%/GS% suffix for CCS labels (unabated left alone).
    for i, row in result_df.iterrows():
        if row["_is_unabated"]:
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
                    scen_df["eor_fraction"] * scen_df["mtco2_per_yr_captured_with_ccs"]
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

    for col, label, color in _BREAKDOWN_COMPONENTS:
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

    # Green dashed "captured" overlay — ONLY on CCS bars. On unabated bars we
    # draw a zero-height placeholder (skipped) so the top of the stack is just
    # pos_bottom itself.
    captured_vals = result_df["_captured"].values.copy()
    captured_vals[result_df["_is_unabated"].values] = 0.0
    ccs_mask = ~result_df["_is_unabated"].values
    if ccs_mask.any():
        ax.bar(
            x[ccs_mask],
            captured_vals[ccs_mask],
            width=width,
            bottom=pos_bottom[ccs_mask],
            facecolor="none",
            edgecolor="#2e7d32",
            linewidth=1.5,
            linestyle="--",
            zorder=2,
            label=r"$\mathbf{CO_2}$ captured by CCS",
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

    # Thick black "Net" line drawn across the top of every bar's emitted
    # stack (below the green dashed captured overlay on CCS bars). Extends
    # slightly beyond each bar's width so it's obviously a marker rather than
    # part of the bar. The Net / per-tCO2 labels sit directly above this line.
    line_overhang = 0.04
    for j in range(len(x)):
        ax.plot(
            [x[j] - width / 2 - line_overhang, x[j] + width / 2 + line_overhang],
            [pos_bottom[j], pos_bottom[j]],
            color="black",
            linewidth=1.75,
            zorder=4,
            solid_capstyle="butt",
        )

    # Black dashed line marking the unabated emissions of retrofit facilities
    # only (the "existing facilities" subset that would be emitting regardless
    # of CCS). Drawn only across the Unabated bar, slightly wider than the
    # solid Net line above. Requires the `retrofit_newbuild` column.
    unabated_rows = np.where(result_df["_is_unabated"].values)[0]
    if len(unabated_rows) > 0 and "retrofit_newbuild" in scenarios_df.columns:
        # Unabated is scenario-invariant, so one scenario-row per project is
        # enough. The function has already filtered scenarios_df by state.
        _scen_names = scenarios_df["scenario"].dropna().unique()
        if len(_scen_names):
            _pick = scenarios_df[
                (scenarios_df["methane_scenario"] == methane_scenario)
                & (scenarios_df["scenario"] == _scen_names[0])
            ].drop_duplicates(subset="uid")
            _is_retro = (
                _pick["retrofit_newbuild"].astype(str).str.strip().str.lower()
                == "retrofit"
            )
            retrofit_unabated_mt = float(
                _pick.loc[_is_retro, "unabated_mtco2e_per_yr"].sum()
            )
        else:
            retrofit_unabated_mt = 0.0

        if retrofit_unabated_mt > 0:
            _unabated_mt = float(_pick.loc[:, "unabated_mtco2e_per_yr"].sum())
            retro_overhang = 0.11  # a bit wider than LINE_OVERHANG (0.04)
            retro_label_gap = 10  # data units below the dashed line
            for j in unabated_rows:
                ax.plot(
                    [
                        x[j] - width / 2 - retro_overhang,
                        x[j] + width / 2 + retro_overhang,
                    ],
                    [retrofit_unabated_mt, retrofit_unabated_mt],
                    color="black",
                    linewidth=2.5,
                    linestyle=(0, (6, 4)),  # longer dashes read as "reference"
                    zorder=4,
                    solid_capstyle="butt",
                )
                ax.text(
                    x[j] + width / 1.9,
                    retrofit_unabated_mt - retro_label_gap,
                    f"Emissions from\nexisting facilities\n({retrofit_unabated_mt:.0f}"
                    + r" MtCO$_2$e/yr)",
                    ha="left",
                    va="top",
                    fontsize=fs["bar_label"] - 3,
                    fontfamily="Arial Narrow",
                    fontweight="normal",
                    color="black",
                    zorder=5,
                    path_effects=[
                        pe.withStroke(linewidth=2, alpha=0.8, foreground="white")
                    ],
                )
                ax.text(
                    x[j] + width / 1.6,
                    retrofit_unabated_mt + (_unabated_mt - retrofit_unabated_mt) / 2,
                    "Additional\nemissions from\nfull buildout\nof new\nfacilities",
                    ha="left",
                    va="top",
                    fontsize=fs["bar_label"] - 3,
                    fontfamily="Arial Narrow",
                    fontweight="normal",
                    color="black",
                    zorder=5,
                    path_effects=[
                        pe.withStroke(linewidth=2, alpha=0.8, foreground="white")
                    ],
                )
                add_brace(
                    ax,
                    x=x[j] + width / 1.9,
                    y1=retrofit_unabated_mt,
                    y2=_unabated_mt,
                    side="right",
                    width=0.05,
                    color="black",
                    lw=1.25,
                )

    # Bar-top labels sit tightly against the black line; CCS bars get the
    # per-tCO2 subscript directly above the line and "Net: X" above that,
    # unabated bars get ONLY the Net line.
    label_gap = 3  # data units between line and first label
    label_line_height = 28  # data units between stacked label lines
    for j in range(len(x)):
        scen_name = result_df.iloc[j]["scenario"]
        bar_top = pos_bottom[j]
        if result_df.iloc[j]["_is_unabated"]:
            net_mt = result_df.iloc[j]["_net"]
            ax.text(
                x[j],
                bar_top + label_gap,
                f"Net: {net_mt:.0f}" + r" MtCO$_2$e/yr",
                ha="center",
                va="bottom",
                fontsize=fs["bar_label"],
                fontfamily="Arial",
                fontweight="bold",
                zorder=5,
                path_effects=[pe.withStroke(linewidth=5, foreground="white")],
            )
            continue
        scen_rows = scenarios_df[
            (scenarios_df["methane_scenario"] == methane_scenario)
            & (scenarios_df["scenario"] == scen_name)
        ]
        net_mt = scen_rows["mtco2e_per_yr_emissions_including_ccs"].sum()
        captured_j = scen_rows["mtco2_per_yr_captured_with_ccs"].sum()
        per_tco2 = net_mt / captured_j if captured_j else 0
        ax.text(
            x[j],
            bar_top + label_gap,
            f"({per_tco2:.1f}" + r" tCO$_2$e/tCO$_2$ injected)",
            ha="center",
            va="bottom",
            fontsize=fs["bar_label"] - 1,
            fontfamily="Arial Narrow",
            fontweight="normal",
            zorder=5,
            path_effects=[pe.withStroke(linewidth=5, foreground="white")],
        )
        ax.text(
            x[j],
            bar_top + label_gap + label_line_height,
            f"Net: {net_mt:.0f}" + r" MtCO$_2$e/yr",
            ha="center",
            va="bottom",
            fontsize=fs["bar_label"],
            fontfamily="Arial",
            fontweight="bold",
            zorder=5,
            path_effects=[pe.withStroke(linewidth=5, foreground="white")],
        )

    ax.set_xticks(x)
    wrapped_labels = [
        "\n".join(textwrap.fill(line, width=20) for line in lbl.split("\n"))
        for lbl in result_df["label"].tolist()
    ]
    ax.set_xticklabels(wrapped_labels, fontsize=fs["tick"], fontfamily="Arial")
    ax.set_ylabel(
        r"MtCO$_2$e / yr",
        fontsize=fs["axis_label"],
        fontfamily="Arial",
        labelpad=1,
    )
    y_min = neg_bottom.min()
    y_max = ghost_top.max() * 1.10
    ax.set_ylim(y_min - 1 if y_min < 0 else -1, y_max)
    ax.tick_params(axis="y", labelsize=fs["tick"])

    if total_captured > 0:
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

    handles, labels_leg = ax.get_legend_handles_labels()
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels_leg):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)
    if legend_below:
        ax.legend(
            unique_handles,
            unique_labels,
            fontsize=fs["legend"],
            frameon=True,
            facecolor="white",
            edgecolor="none",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=3,
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
    """Like `plot_state_emissions_compared_to_unabated_on_dual_axis`, but the
    "existing facilities" (retrofit) emissions appear as their own stacked bar
    on the left of the ``"unabated"`` bar rather than as a dashed overlay on
    top of it. The retrofit-overlay line, the brace, and the two inline
    annotations ("Emissions from existing facilities" and "Additional
    emissions from full buildout of new facilities") are removed; the Unabated
    bar carries only its "Net: X MtCO₂e/yr" label.

    Enable the new bar by including ``"existing"`` in `destination_scenarios`
    (typically as the first entry). An ``"existing"`` bar sums
    `unabated_mtco2e_per_yr` across projects whose ``retrofit_newbuild``
    column is ``"retrofit"``, broken into the same four non-CCS components as
    the Unabated bar. Needs the ``retrofit_newbuild`` column on `scenarios_df`.

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

    line_overhang = 0.04
    for j in range(len(x)):
        ax.plot(
            [x[j] - width / 2 - line_overhang, x[j] + width / 2 + line_overhang],
            [pos_bottom[j], pos_bottom[j]],
            color="black",
            linewidth=1.75,
            zorder=4,
            solid_capstyle="butt",
        )

    label_gap = 3
    label_line_height = 28
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
                bar_top + label_gap,
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
                bar_top + label_gap,
                f"({per_tco2:.1f}" + r" tCO$_2$e/tCO$_2$ injected)",
                ha="center",
                va="bottom",
                fontsize=_net_label_fs - 1,
                fontfamily="Arial Narrow",
                fontweight="normal",
                zorder=5,
                path_effects=[pe.withStroke(linewidth=5, foreground="white")],
            )
            net_y = bar_top + label_gap + label_line_height
        else:
            net_y = bar_top + label_gap
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
                arrowprops={
                    "arrowstyle": "->",
                    "color": _captured_green,
                    "linewidth": 1.5,
                    "shrinkA": 2,
                    "shrinkB": 0,
                },
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
    if (slimmest_version is True) & (display_mode == "presentation"):
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
    if not total_captured > 0 and show_per_tco2_label:
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


def plot_emissions_simplest(
    industrial_emissions_df: pd.DataFrame,
    ccs_emissions_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    methane_scenario: str,
    destination_scenarios: List[Union[str, dict]],
    industries: Optional[List[str]] = None,
    which_flavor: Optional[List[str]] = None,
    states: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 7),
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    display_mode: Optional[str] = None,
    scenario_labels: Optional[Dict[str, str]] = None,
    legend_ncol: int = 2,
    legend_right: bool = False,
    legend_right_axes_fraction: float = 0.72,
    legend_right_valign: str = "top",
    lifetime: bool = False,
    facility_lifetime_years: int = 30,
    ccs_active_years: int = 12,
    industrial_color: str = "#bf5b04",
    ccs_caused_color: str = "#b71c1c",
    stored_color: str = "#2e7d32",
) -> Tuple[plt.Figure, plt.Axes]:
    """Simplest stacked-bar comparison of CCS vs. no-CCS scenarios.

    Each scenario is one bar with up to two stacked positive groups and a
    dashed "ghost" box stacked on top:
      1. Industrial facility emissions  (process + combustion CH4 + CO2 lumped,
         single color)
      2. Emissions caused by implementing CCS  (CCS-caused CH4 + CO2 lumped)
      3. CO2 captured by CCS  (dashed outline, no fill, stacked on top of 1+2
         — represents what WOULD have been emitted but was captured instead)

    CCS scenarios also render a filled green bar below zero showing
    "CO2 injected underground" at the same magnitude as the dashed box above.
    Reference bars (``existing`` and ``unabated``) show only group 1.

    Args:
        destination_scenarios: Same semantics as the sibling function. Strings
            resolve from a known table; dicts are used verbatim. Typical:
            ``["existing", "unabated", "baseline", "current_mix"]``.
        lifetime: When True (default), bars show 30-yr lifetime totals using
            the same component-scaling convention as the sibling function.
        facility_lifetime_years, ccs_active_years: Must match
            ``scenario_config.facility_lifetime_years`` and
            ``scenario_config.project_duration`` used to build ``scenarios_df``,
            otherwise on-chart bar totals will not equal the pipeline's
            ``lifetime_mtco2e`` column.
        industrial_color, ccs_caused_color, stored_color: Colors for the three
        drawn segments (industrial-facility, CCS-caused, stored).
    """
    display_mode = _resolve_display_mode(display_mode)
    fs = _resolve_fonts(display_mode)

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
            "label": "With CCS, 100%\ngeologic storage",
            "eor_pct": 0,
        },
        "baseline": {
            "name": "baseline",
            "label": "With CCS, stated\ndestination",
            "from_data": True,
        },
        "current_mix": {
            "name": "current_mix",
            "label": "With CCS, current\ndestination mix",
            "eor_pct": 92,
        },
        "all_eor": {
            "name": "all_eor",
            "label": "With CCS, 100%\nEOR",
            "eor_pct": 100,
        },
        "switching": {
            "name": "switching",
            "label": "Stated → 100%\nEOR after 45Q",
            "from_data": True,
            "is_switching": True,
        },
        "switching_from_current": {
            "name": "switching_from_current",
            "label": "Current → 100%\nEOR after 45Q",
            "from_data": True,
            "is_switching": True,
        },
        "unabated": {
            "name": "unabated",
            "label": "Full buildout\nno CCS",
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

    existing_record = None
    if any(s.get("is_existing") for s in destination_scenarios):
        if "retrofit_newbuild" not in scenarios_df.columns:
            raise ValueError(
                "'existing' scenario requested but scenarios_df is missing "
                "'retrofit_newbuild'."
            )
        retro_mask = (
            scenarios_df["retrofit_newbuild"].astype(str).str.strip().str.lower()
            == "retrofit"
        )
        retro_scen_df = scenarios_df[retro_mask]
        if retro_scen_df.empty:
            existing_record = {c: 0.0 for c in component_cols + ["_captured", "_net"]}
        else:
            existing_record = _compute_unabated_record(
                industrial_emissions_df,
                retro_scen_df,
                methane_scenario,
                industries=industries,
                which_flavor=which_flavor,
            )

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
        elif scen.get("is_existing"):
            for col in component_cols + ["_captured"]:
                row[col] = existing_record.get(col, 0.0)
        else:
            for col in component_cols + ["_captured"]:
                row[col] = sum(r[ccs_i][col] for r in per_state_records)
                row[col] += sum(d[ccs_i].get(col, 0.0) for d in per_state_dac)
            ccs_i += 1
        row["_net"] = sum(row[c] for c in component_cols)
        records.append(row)

    result_df = pd.DataFrame(records)

    # Lifetime scaling — same math as the sibling function.
    if lifetime:
        unabated_tail_years = facility_lifetime_years - ccs_active_years
        if unabated_tail_years < 0:
            raise ValueError(
                f"facility_lifetime_years={facility_lifetime_years} must be "
                f">= ccs_active_years={ccs_active_years}."
            )
        tail_unab = unabated_record or _compute_unabated_record(
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
                # Switching: CCS runs the full facility lifetime; the stored
                # per-year rates are already time-weighted means over L years.
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

    # Collapse 8 raw components into 2 user-facing filled groups + ghost.
    result_df["_industrial"] = (
        result_df["process_ch4"]
        + result_df["combustion_ch4"]
        + result_df["uncaptured_process_co2"]
        + result_df["uncaptured_combustion_co2"]
    )
    result_df["_ccs_caused"] = result_df["ccs_ch4"] + result_df["ccs_co2"]

    groups = [
        ("_industrial", "Industrial facility\nemissions", industrial_color),
        ("_ccs_caused", "Emissions caused by\nimplementing CCS", ccs_caused_color),
    ]

    x = np.arange(len(result_df))
    width = 0.5

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    pos_bottom = np.zeros(len(x))
    for col, label, color in groups:
        vals = result_df[col].values
        if (vals > 0).any():
            ax.bar(
                x,
                vals,
                width=width,
                bottom=pos_bottom,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                label=label,
                zorder=2,
            )
            pos_bottom += vals

    # CCS scenarios get:
    #   (a) a dashed "ghost" box above zero sitting on top of the two filled
    #       groups — represents what WOULD have gone to atmosphere but was
    #       captured instead.
    #   (b) a filled green bar below zero at the same magnitude — represents
    #       the CO2 physically injected underground.
    captured_vals = result_df["_captured"].values.copy()
    non_ccs_mask = result_df["_is_unabated"].values | result_df["_is_existing"].values
    captured_vals[non_ccs_mask] = 0.0
    ccs_mask = ~non_ccs_mask
    neg_bottom = np.zeros(len(x))
    if ccs_mask.any():
        # (a) Ghost box on top of the stack (above zero).
        ax.bar(
            x[ccs_mask],
            captured_vals[ccs_mask],
            width=width,
            bottom=pos_bottom[ccs_mask],
            facecolor="none",
            edgecolor=stored_color,
            linewidth=1.5,
            linestyle="--",
            zorder=2,
            label=r"$\mathbf{CO_2}$ captured by CCS",
        )
        # (b) Filled green bar below zero.
        ax.bar(
            x[ccs_mask],
            -captured_vals[ccs_mask],
            width=width,
            bottom=0.0,
            color=stored_color,
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
            label=r"$\mathbf{CO_2}$ injected underground",
        )
        neg_bottom[ccs_mask] = -captured_vals[ccs_mask]

    ghost_top = pos_bottom + captured_vals

    ax.axhline(0, color="black", linewidth=1.0, zorder=3)

    # Black "net" line + net label above each bar.
    line_overhang = 0.04
    for j in range(len(x)):
        ax.plot(
            [x[j] - width / 2 - line_overhang, x[j] + width / 2 + line_overhang],
            [pos_bottom[j], pos_bottom[j]],
            color="black",
            linewidth=1.75,
            zorder=4,
            solid_capstyle="butt",
        )
    _unit = (
        r" MtCO$_2$e" + "\n" + rf"over {facility_lifetime_years}yr"
        if lifetime
        else r" MtCO$_2$e/yr"
    )
    for j in range(len(x)):
        row_j = result_df.iloc[j]
        # Mirror `plot_state_emissions_vs_unabated_and_existing` so both plots
        # label "Net:" with the same quantity — atmospheric emissions —
        # across all rows and both modes.
        #
        # Unabated/existing rows:  captured_* = 0, so _net already equals
        #                          atmospheric emissions.
        # Lifetime CCS rows:       _net = sum of 8 components where the two
        #                          captured_* columns are negative, so
        #                          _net = emitted − captured. Add _captured
        #                          back (also lifetime-scaled) to recover
        #                          atmospheric emissions.
        # Annual CCS rows:         Use the pipeline column
        #                          `mtco2e_per_yr_emissions_including_ccs`
        #                          directly — it's authoritative and sidesteps
        #                          `_build_breakdown_data`'s per-industry dedup.
        if row_j["_is_unabated"] or row_j["_is_existing"]:
            net_mt = row_j["_net"]
        elif lifetime:
            net_mt = row_j["_net"] + row_j["_captured"]
        else:
            scen_rows = scenarios_df[
                (scenarios_df["methane_scenario"] == methane_scenario)
                & (scenarios_df["scenario"] == row_j["scenario"])
            ]
            net_mt = scen_rows["mtco2e_per_yr_emissions_including_ccs"].sum()
        ax.text(
            x[j],
            pos_bottom[j] * 1.01,
            f"Net: {net_mt:,.0f}" + _unit,
            ha="center",
            va="bottom",
            fontsize=fs["bar_label"],
            fontfamily="Arial",
            fontweight="bold",
            zorder=5,
            path_effects=[pe.withStroke(linewidth=5, foreground="white")],
        )

    ax.set_xticks(x)
    wrapped_labels = [
        lbl
        if "$" in lbl
        else "\n".join(textwrap.fill(line, width=20) for line in lbl.split("\n"))
        for lbl in result_df["label"].tolist()
    ]
    ax.set_xticklabels(wrapped_labels, fontsize=fs["tick"], fontfamily="Arial")
    ax.set_ylabel(
        rf"MtCO$_2$e over {facility_lifetime_years}-yr lifetime"
        if lifetime
        else r"MtCO$_2$e / yr",
        fontsize=fs["axis_label"],
        fontfamily="Arial",
    )
    y_min = neg_bottom.min() if neg_bottom.min() < 0 else -1
    y_max = max(pos_bottom.max(), ghost_top.max()) * 1.15
    ax.set_ylim(y_min * 1.15, y_max)
    ax.tick_params(axis="y", labelsize=fs["tick"])
    ax.grid(axis="y", color="lightgray", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False)

    if title is None:
        state_label = _format_states_for_title(states) if states else ""
        _unit_suffix = (
            f"over {facility_lifetime_years}-yr lifetime" if lifetime else "per year"
        )
        title = f"{state_label} emissions ({_unit_suffix})".strip()
    ax.set_title(title, fontsize=fs["title"], fontfamily="Arial", pad=16)

    handles, labels_leg = ax.get_legend_handles_labels()
    seen = set()
    unique_h, unique_l = [], []
    for h, l in zip(handles, labels_leg):
        if l not in seen:
            seen.add(l)
            unique_h.append(h)
            unique_l.append(l)
    if legend_right:
        # Right-of-plot legend, matched to the convention used by
        # plot_state_emissions_vs_unabated_and_existing: `legend_right_axes_fraction`
        # rebalances plot vs. legend width without touching figsize, and
        # `legend_right_valign` snaps the legend to top / center / bottom.
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
            unique_h,
            unique_l,
            fontsize=fs["legend"],
            frameon=True,
            facecolor="white",
            edgecolor="none",
            loc=_loc,
            bbox_to_anchor=_anchor,
            ncol=legend_ncol,
        )
        fig.subplots_adjust(right=legend_right_axes_fraction)
    else:
        ax.legend(
            unique_h,
            unique_l,
            fontsize=fs["legend"],
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=legend_ncol,
        )
        fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)

    return fig, ax


def plot_lifetime_emissions_with_ccs_overlay(  # pylint: disable=unused-argument
    industrial_emissions_df: pd.DataFrame,
    ccs_emissions_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    methane_scenario: str,
    ccs_scenarios: Optional[List[str]] = None,
    states: Optional[List[str]] = None,
    industries: Optional[List[str]] = None,
    which_flavor: Optional[List[str]] = None,
    facility_lifetime_years: int = 30,
    ccs_active_years: int = 12,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 7),
    output_path: Optional[str] = None,
    display_mode: Optional[str] = None,
    scenario_labels: Optional[Dict[str, str]] = None,
    ccs_scenario_colors: Optional[Dict[str, str]] = None,
    ccs_line_width: float = 2.5,
    ccs_legend_labels: Optional[Dict[str, str]] = None,
    ccs_annotation_labels: Optional[Dict[str, str]] = None,
    ccs_annotation_offsets: Optional[Dict[str, Tuple[float, float]]] = None,
    bar_line_overhang: float = 0.05,
    net_line_color: str = "black",
    net_line_width: float = 3.0,
    bar_gap: float = 1.0,
    xtick_labels: Optional[List[str]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    unabated_total_offset: Tuple[float, float] = (0.15, 0.0),
    existing_total_offset: Tuple[float, float] = (0.15, 0.0),
    unabated_total_label: Optional[str] = None,
    existing_total_label: Optional[str] = None,
    bar_width: float = 0.55,
    legend_loc: str = "upper left",
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
    legend_ncol: int = 1,
    legend_frameon: bool = False,
    annotation_fontsize: Optional[float] = None,
    annotation_fontweight: str = "semibold",
    unabated_total_fontweight: Optional[str] = None,
    existing_total_fontweight: Optional[str] = None,
    xlim_right_pad: float = 2.8,
) -> Tuple[plt.Figure, "plt.Axes"]:
    """Lifetime-emissions snapshot: stacked Existing and Unabated bars with
    CCS-scenario lifetime totals overlaid as horizontal lines across the
    Unabated bar.

    Two vertical stacked bars per state panel:
      - **Existing**: 30 × unabated breakdown, restricted to retrofit
        facilities (`retrofit_newbuild == "retrofit"`). Shows what's already
        operating today will emit over a 30-yr facility life.
      - **Unabated**: 30 × unabated breakdown across all planned facilities
        (full Gulf Coast CCS buildout, no CCS). The counterfactual.

    For each CCS scenario in ``ccs_scenarios``, the state-aggregate lifetime
    net MtCO₂e (``lifetime_mtco2e`` column from the pipeline, i.e.
    ``12 × mtco2e_per_yr_emissions_including_ccs + 18 × unabated_mtco2e_per_yr``)
    is drawn as a horizontal line spanning the Unabated bar's width, with a
    short label to the right (``"100% EOR: 6,468 MtCO₂e"`` style).

    Args:
        ccs_scenarios: CCS scenarios to overlay, in draw order. Defaults to
            ``["all_gs", "baseline", "all_eor"]``.
        states: State codes; one subplot per state. Defaults to ``["LA"]``.
        scenario_labels: Optional dict mapping scenario name → display label.
            Unmatched scenarios fall back to a built-in table.
        facility_lifetime_years, ccs_active_years: 30/12 by default; the
            unabated/existing bars scale by `facility_lifetime_years`, and the
            CCS lifetime values come from `scenarios_df["lifetime_mtco2e"]`
            which was already built with the pipeline's `project_duration`.
        ccs_line_color, ccs_line_width: Styling for the overlay lines.
    """
    display_mode = _resolve_display_mode(display_mode)
    fs = dict(_resolve_fonts(display_mode))
    # Local presentation-mode bump: annotations, legend, x-tick labels, and
    # axis labels need to read from several feet away. Report mode is
    # untouched.
    if display_mode == "presentation":
        fs["annotation"] = 17
        fs["legend"] = 17
        fs["tick"] = 20
        fs["axis_label"] = 22
    # Caller overrides for annotation styling. These apply to all three
    # annotation groups: the per-CCS-scenario overlay labels on the right, the
    # Existing-bar total, and the Unabated-bar total. Font weight is routed
    # separately because semibold is a strong presentation default but reports
    # often want "normal".
    _annot_fs = (
        annotation_fontsize if annotation_fontsize is not None else fs["annotation"]
    )
    _bar_label_fs = (
        annotation_fontsize if annotation_fontsize is not None else fs["bar_label"]
    )

    if ccs_scenarios is None:
        ccs_scenarios = ["all_gs", "baseline", "all_eor"]
    if states is None:
        states = ["LA"]

    _default_ccs_labels = {
        "all_gs": "100% GS",
        "baseline": "Announced destinations",
        "current_mix": "Current U.S. practice (92% EOR)",
        "all_eor": "100% EOR",
        "nameplate_implied": "Nameplate",
    }
    if scenario_labels is None:
        scenario_labels = {}

    def label_for(s):
        return scenario_labels.get(s, _default_ccs_labels.get(s, s))

    # Legend-vs-annotation label resolution. By default both come from
    # `scenario_labels` + the built-in table above. Callers can override the
    # two independently so (e.g.) the legend reads "100% GS" while the on-plot
    # annotation reads "100% Geologic storage: 6,274 Mt CO2e (12-yr window)".
    if ccs_legend_labels is None:
        ccs_legend_labels = {}
    if ccs_annotation_labels is None:
        ccs_annotation_labels = {}
    if ccs_annotation_offsets is None:
        ccs_annotation_offsets = {}
    # Per-scenario line color. Palette tuned so GS reads "cool" (geologic
    # storage = cleaner), EOR reads "warm" (combusted oil = dirtier), with
    # baseline/current-mix sitting between.
    _default_ccs_colors = {
        "all_gs": "#1f78b4",
        "baseline": "#9b59b6",
        "current_mix": "#f39c12",
        "all_eor": "#b71c1c",
        "nameplate_implied": "#555555",
    }
    if ccs_scenario_colors is None:
        ccs_scenario_colors = {}
    _color_fallback = ["#1f78b4", "#9b59b6", "#f39c12", "#b71c1c", "#2ca02c"]

    def color_for(scen, idx):
        return ccs_scenario_colors.get(
            scen,
            _default_ccs_colors.get(scen, _color_fallback[idx % len(_color_fallback)]),
        )

    if "lifetime_mtco2e" not in scenarios_df.columns:
        raise ValueError(
            "scenarios_df is missing 'lifetime_mtco2e'; re-run the pipeline so "
            "the lifetime_* columns are written."
        )

    component_cols = [col for col, _, _ in _BREAKDOWN_COMPONENTS]

    # Build per-state records once per panel: unabated (all projects) and
    # existing (retrofit-only). Multiply by facility_lifetime_years to get
    # the 30-yr lifetime totals shown on the stacked bars.
    def _state_slice(df, state):
        if "project_state" in df.columns:
            return df[df["project_state"] == state]
        return df

    def _panel_records(state):
        ie_s = _state_slice(industrial_emissions_df, state)
        scen_s = _state_slice(scenarios_df, state)
        scen_s = scen_s[scen_s["methane_scenario"] == methane_scenario]
        unab = _compute_unabated_record(
            ie_s,
            scen_s,
            methane_scenario,
            industries=industries,
            which_flavor=which_flavor,
        )
        if "retrofit_newbuild" in scen_s.columns:
            retro = scen_s[
                scen_s["retrofit_newbuild"].astype(str).str.strip().str.lower()
                == "retrofit"
            ]
            if retro.empty:
                existing = {k: 0.0 for k in unab}
            else:
                existing = _compute_unabated_record(
                    ie_s,
                    retro,
                    methane_scenario,
                    industries=industries,
                    which_flavor=which_flavor,
                )
        else:
            existing = {k: 0.0 for k in unab}
        # Scale annual breakdowns to lifetime (30 yr unabated).
        unab_life = {k: v * facility_lifetime_years for k, v in unab.items()}
        exist_life = {k: v * facility_lifetime_years for k, v in existing.items()}

        # CCS overlay values from pipeline column lifetime_mtco2e.
        ccs_vals = {}
        for scen in ccs_scenarios:
            sub = scen_s[scen_s["scenario"] == scen]
            ccs_vals[scen] = float(sub["lifetime_mtco2e"].sum())
        return unab_life, exist_life, ccs_vals

    panel_data = {state: _panel_records(state) for state in states}

    # Figure layout: one subplot per state, shared y-axis.
    fig, axes_arr = plt.subplots(
        1, len(states), figsize=figsize, sharey=True, squeeze=False
    )
    axes = list(axes_arr[0])

    # Determine global y_max so overlay labels have room to the right.
    y_global_max = 0.0
    for state in states:
        unab, exist, ccs_v = panel_data[state]
        y_global_max = max(
            y_global_max,
            sum(unab.get(c, 0.0) for c in component_cols),
            sum(exist.get(c, 0.0) for c in component_cols),
            max(ccs_v.values()) if ccs_v else 0.0,
        )
    y_top = y_global_max * 1.08

    x_existing = 0.0
    x_unabated = bar_gap
    # Overlay line spans the Unabated bar, with a small overhang on both
    # sides so each CCS line reads as crossing the bar instead of ending
    # flush with the fill. Same overhang is used for the black "net" line
    # across the Existing and Unabated stacks.
    line_x_lo = x_unabated - bar_width / 2 - bar_line_overhang
    line_x_hi = x_unabated + bar_width / 2 + bar_line_overhang
    exist_line_lo = x_existing - bar_width / 2 - bar_line_overhang
    exist_line_hi = x_existing + bar_width / 2 + bar_line_overhang
    label_x = line_x_hi + 0.15

    for panel_i, state in enumerate(states):
        a = axes[panel_i]
        unab, exist, ccs_v = panel_data[state]

        pos_bottom_exist = 0.0
        pos_bottom_unab = 0.0
        for col, label, color in _BREAKDOWN_COMPONENTS:
            exist_val = exist.get(col, 0.0)
            unab_val = unab.get(col, 0.0)
            if exist_val > 0 or unab_val > 0:
                # Draw as two separate bars so the stacking is independent per
                # category; use `label=` only on the first panel so legend has
                # one entry per component.
                a.bar(
                    [x_existing],
                    [exist_val],
                    bottom=[pos_bottom_exist],
                    width=bar_width,
                    color=color,
                    edgecolor="white",
                    linewidth=0.5,
                    label=label if panel_i == 0 else None,
                    zorder=2,
                )
                a.bar(
                    [x_unabated],
                    [unab_val],
                    bottom=[pos_bottom_unab],
                    width=bar_width,
                    color=color,
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=2,
                )
                pos_bottom_exist += exist_val
                pos_bottom_unab += unab_val

        # Black net line across the Existing stack and the Unabated stack,
        # slightly wider than each bar (same overhang as the CCS overlay).
        a.plot(
            [exist_line_lo, exist_line_hi],
            [pos_bottom_exist, pos_bottom_exist],
            color=net_line_color,
            linewidth=net_line_width,
            solid_capstyle="butt",
            zorder=3,
        )
        a.plot(
            [line_x_lo, line_x_hi],
            [pos_bottom_unab, pos_bottom_unab],
            color=net_line_color,
            linewidth=net_line_width,
            solid_capstyle="butt",
            zorder=3,
        )

        # Overlay horizontal line + leader-line annotation for each CCS
        # scenario. Each scenario gets its own color; legend entries are
        # registered only on the first panel so the shared legend lists each
        # scenario once. Annotations are drawn with ax.annotate so each has
        # its own leader line connecting the right end of the colored line
        # to the text. Default annotation y-positions auto-spread when
        # adjacent scenarios' lifetime totals are too close to read; the
        # caller can override any scenario's (dx, dy) via
        # `ccs_annotation_offsets`.
        y_vals = [(scen, ccs_v[scen]) for scen in ccs_scenarios]
        y_axis_span = y_top  # the y_global_max × 1.08 computed below panel loop
        min_gap = 0.04 * y_axis_span
        sorted_scens = sorted(y_vals, key=lambda t: t[1])
        auto_y = dict(sorted_scens)
        for i in range(1, len(sorted_scens)):
            prev_scen = sorted_scens[i - 1][0]
            scen_i = sorted_scens[i][0]
            if auto_y[scen_i] - auto_y[prev_scen] < min_gap:
                auto_y[scen_i] = auto_y[prev_scen] + min_gap

        for scen_idx, scen in enumerate(ccs_scenarios):
            y_val = ccs_v[scen]
            scen_color = color_for(scen, scen_idx)
            # CCS overlay lines are identified by the colored leader-line
            # annotation to the right of each bar, so they intentionally stay
            # out of the legend (which only lists the stacked-component bars).
            a.plot(
                [line_x_lo, line_x_hi],
                [y_val, y_val],
                color=scen_color,
                linewidth=ccs_line_width,
                solid_capstyle="butt",
                zorder=4,
            )
            # Caller-supplied annotation text, or the default
            # "{label}: {X,XXX} MtCO2e". If the caller passes a format string
            # containing `{value}`, the number is substituted; otherwise the
            # string is used verbatim.
            if scen in ccs_annotation_labels:
                raw = ccs_annotation_labels[scen]
                try:
                    text = raw.format(value=y_val, label=label_for(scen))
                except Exception:
                    text = raw
            else:
                text = f"{label_for(scen)}: {y_val:,.0f} MtCO$_2$e"

            dx, dy = ccs_annotation_offsets.get(scen, (0.15, 0.0))
            auto_y_scen = auto_y[scen]
            text_x = line_x_hi + dx
            text_y = auto_y_scen + dy
            a.annotate(
                text,
                xy=(line_x_hi, y_val),
                xytext=(text_x, text_y),
                textcoords="data",
                ha="left",
                va="center",
                fontsize=_annot_fs,
                color=scen_color,
                fontweight=annotation_fontweight,
                arrowprops={
                    "arrowstyle": "-",
                    "color": scen_color,
                    "linewidth": 1.0,
                    "shrinkA": 0,
                    "shrinkB": 2,
                    "relpos": (0, 0.5),
                },
                zorder=5,
            )

        # Baseline annotations next to the black "net" line atop each bar.
        # Placed to the right with a short leader line connecting to the
        # right end of the net line (left-center of the text).
        ex_dx, ex_dy = existing_total_offset
        # Format substitutions:
        #   {value} -> MtCO2e (backward compatible with prior callers)
        #   {gt}    -> GtCO2e (= Mt/1000), one decimal is the usual format
        _ex_gt = pos_bottom_exist / 1000.0
        if existing_total_label is not None:
            try:
                ex_text = existing_total_label.format(value=pos_bottom_exist, gt=_ex_gt)
            except Exception:
                ex_text = existing_total_label
        else:
            ex_text = f"{_ex_gt:.1f} GtCO$_2$e"
        a.annotate(
            ex_text,
            xy=(exist_line_hi, pos_bottom_exist),
            xytext=(exist_line_hi + ex_dx, pos_bottom_exist + ex_dy),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=_bar_label_fs,
            fontweight=(
                existing_total_fontweight
                if existing_total_fontweight is not None
                else annotation_fontweight
            ),
            color=net_line_color,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            arrowprops={
                "arrowstyle": "-",
                "color": net_line_color,
                "linewidth": 1.0,
                "shrinkA": 0,
                "shrinkB": 2,
                "relpos": (0, 0.5),
            },
            zorder=5,
        )
        un_dx, un_dy = unabated_total_offset
        _un_gt = pos_bottom_unab / 1000.0
        if unabated_total_label is not None:
            try:
                un_text = unabated_total_label.format(value=pos_bottom_unab, gt=_un_gt)
            except Exception:
                un_text = unabated_total_label
        else:
            un_text = f"{_un_gt:.1f} GtCO$_2$e"
        a.annotate(
            un_text,
            xy=(line_x_hi, pos_bottom_unab),
            xytext=(line_x_hi + un_dx, pos_bottom_unab + un_dy),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=_bar_label_fs,
            fontweight=(
                unabated_total_fontweight
                if unabated_total_fontweight is not None
                else annotation_fontweight
            ),
            color=net_line_color,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            arrowprops={
                "arrowstyle": "-",
                "color": net_line_color,
                "linewidth": 1.0,
                "shrinkA": 0,
                "shrinkB": 2,
                "relpos": (0, 0.5),
            },
            zorder=5,
        )

        a.set_xticks([x_existing, x_unabated])
        _xtick_labels = (
            xtick_labels
            if xtick_labels is not None
            else ["Existing\nfacilities", "Full buildout\n(unabated)"]
        )
        a.set_xticklabels(_xtick_labels, fontsize=fs["tick"])
        a.set_xlim(-0.6, label_x + xlim_right_pad)
        a.set_ylim(*(ylim if ylim is not None else (0, y_top)))
        if panel_i == 0:
            a.set_ylabel(
                rf"GtCO$_2$e over {facility_lifetime_years}-yr lifetime",
                fontsize=fs["axis_label"],
            )
        a.tick_params(axis="y", labelsize=fs["tick"])
        # Y-axis in Gt (Mt / 1000), labelling every other tick for a cleaner
        # presentation-friendly axis. The ticks themselves stay in Mt so the
        # bar data lines up; only the label display is converted.
        _mt_ticks = a.get_yticks()
        _gt_labels: List[str] = []
        for _i, _m in enumerate(_mt_ticks):
            _g = _m / 1000.0
            if _i % 2 == 0:
                _gt_labels.append(
                    f"{_g:.0f}" if abs(_g - round(_g)) < 1e-6 else f"{_g:.1f}"
                )
            else:
                _gt_labels.append("")
        a.set_yticks(_mt_ticks)
        a.set_yticklabels(_gt_labels)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.grid(axis="y", alpha=0.4)
        a.set_axisbelow(True)
        if len(states) > 1:
            _state_display = {"TX": "Texas", "LA": "Louisiana"}
            a.set_title(_state_display.get(state, state), fontsize=fs["title"])

    _legend_kwargs = {
        "fontsize": fs["legend"],
        "loc": legend_loc,
        "frameon": legend_frameon,
        "ncol": legend_ncol,
    }
    if legend_bbox_to_anchor is not None:
        _legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
    axes[0].legend(**_legend_kwargs)

    if title is None:
        title = (
            f"Lifetime emissions: existing vs. full buildout, with CCS "
            f"scenario totals overlaid (methane_scenario='{methane_scenario}')"
        )
    fig.suptitle(title, fontsize=fs["title"])
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)

    return fig, (axes[0] if len(axes) == 1 else axes)


def plot_state_emissions_by_methane_scenario(
    industrial_emissions_df: pd.DataFrame,
    ccs_emissions_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    methane_scenarios: List[str],
    destination_scenarios: List[Union[str, dict]],
    industries: Optional[List[str]] = None,
    which_flavor: Optional[List[str]] = None,
    states: Optional[List[str]] = None,
    title: Optional[str] = None,
    legend_below: bool = True,
    figsize: Tuple[int, int] = (16, 8),
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    display_mode: Optional[str] = None,
    methane_scenario_labels: Optional[Dict[str, str]] = None,
    destination_scenario_labels: Optional[Dict[str, str]] = None,
    show_per_tco2_label: bool = True,
    legend_ncol: int = 3,
    annotate_captured_inline: bool = False,
    slimmest_version: bool = False,
    legend_right: bool = False,
    legend_right_axes_fraction: float = 0.72,
    legend_right_valign: str = "top",
    group_gap: float = 0.35,
    show_destination_sublabels: bool = True,
    destination_sublabel_rotation: float = 0.0,
) -> Tuple[plt.Figure, plt.Axes]:
    """Grouped-bar version of `plot_state_emissions_vs_unabated_and_existing`
    where the X-axis dimension is methane assumption instead of destination
    scenario.

    Each methane scenario in `methane_scenarios` becomes a group on the
    X-axis; inside each group there is one bar per entry of
    `destination_scenarios` (same recognized values as the sibling —
    ``"existing"``, ``"unabated"``, ``"baseline"``, ``"current_mix"``,
    ``"all_eor"``, ``"all_gs"``, ``"nameplate_implied"``, or a custom dict).
    Typical use: ``destination_scenarios=["current_mix"]`` → one bar per
    methane group; ``["unabated", "current_mix"]`` → two bars per group;
    ``["existing", "unabated", "current_mix"]`` → three bars per group.

    All per-bar styling, lumping, inline annotations, legend positioning,
    and figure-layout flags behave identically to the sibling.

    Args:
        methane_scenarios: Ordered list of methane_scenario names (matching
            values in `scenarios_df['methane_scenario']`) that form the
            x-axis groups.
        destination_scenarios: Ordered list of destination entries (strings
            or dicts) rendered as adjacent bars inside each methane group.
            Same semantics as the sibling, including ``"existing"`` /
            ``"unabated"`` pseudo-scenarios.
        methane_scenario_labels: Optional dict mapping methane_scenario name
            to its x-axis group label. Falls back to the raw name.
        destination_scenario_labels: Optional dict mapping destination name
            to the sub-bar label drawn under each bar in a group. Falls back
            to the known-scenario default label.
        group_gap: Horizontal padding between adjacent methane-scenario
            groups, in units of bar-widths. Default 0.35.
        show_destination_sublabels: If True (default) and more than one bar
            per group, each sub-bar gets a small destination-scenario label
            printed below it so bars within a group can be distinguished at
            a glance. Set False when the whole plot is one bar per group.
        destination_sublabel_rotation: Rotation (degrees) applied to the
            italic destination sub-labels under each bar. Default 0°
            (horizontal). Short abbreviated defaults (``"exist."``,
            ``"unab."``, ``"CCS"``, …) fit comfortably within an 0.8-bar-
            width slot without overlapping neighbors. Set 35–45 if callers
            override via `destination_scenario_labels` with longer labels
            that need to be angled.
        (other args behave identically to
        `plot_state_emissions_vs_unabated_and_existing`.)
    """
    display_mode = _resolve_display_mode(display_mode)
    fs = _resolve_fonts(display_mode)

    if not methane_scenarios:
        raise ValueError("methane_scenarios must contain at least one entry.")

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
            "label": "With CCS,\n100% GS",
            "eor_pct": 0,
        },
        "baseline": {
            "name": "baseline",
            "label": "With CCS,\nstated dest.",
            "from_data": True,
        },
        "current_mix": {
            "name": "current_mix",
            "label": "With CCS,\ncurrent mix",
            "eor_pct": 92,
        },
        "nameplate_implied": {
            "name": "nameplate_implied",
            "label": "Nameplate",
            "zero_lifecycle": True,
        },
        "all_eor": {
            "name": "all_eor",
            "label": "With CCS,\n100% EOR",
            "eor_pct": 100,
        },
        "switching": {
            "name": "switching",
            "label": "Stated →\n100% EOR after 45Q",
            "from_data": True,
            "is_switching": True,
        },
        "switching_from_current": {
            "name": "switching_from_current",
            "label": "Current →\n100% EOR after 45Q",
            "from_data": True,
            "is_switching": True,
        },
        "unabated": {
            "name": "unabated",
            "label": "Unabated after full buildout (no CCS)",
            "is_unabated": True,
        },
    }

    resolved: List[dict] = []
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

    component_cols = [col for col, _, _ in _BREAKDOWN_COMPONENTS]

    # Compute per-bar records: outer loop is methane scenario; inner loop is
    # destination scenario. For each (methane, destination) cell we build the
    # same `row` dict the sibling uses, so the stacking/overlay/label code
    # below is a straight port.
    all_records: List[dict] = []  # one entry per bar; includes methane + dest info
    for m_idx, m_name in enumerate(methane_scenarios):
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
                        m_name,
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
                        m_name,
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
                m_name,
                industries=industries,
                which_flavor=which_flavor,
            )

        existing_record = None
        if any(s.get("is_existing") for s in destination_scenarios):
            if "retrofit_newbuild" not in scenarios_df.columns:
                raise ValueError(
                    "'existing' scenario requested but scenarios_df is missing "
                    "the 'retrofit_newbuild' column needed to subset retrofit "
                    "projects."
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
                    m_name,
                    industries=industries,
                    which_flavor=which_flavor,
                )

        ccs_i = 0
        for d_idx, scen in enumerate(destination_scenarios):
            row = {
                "methane_scenario": m_name,
                "methane_idx": m_idx,
                "destination_idx": d_idx,
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
            all_records.append(row)

    result_df = pd.DataFrame(all_records)

    # Grouped-bar layout: for N methane groups of M destinations each, place
    # bars at x = m_idx * (M + group_gap) + d_idx. Group centers (used for
    # x-ticks) are the arithmetic mean of each group's bar positions.
    n_methane = len(methane_scenarios)
    n_dest = len(destination_scenarios)
    bar_width = 0.8
    group_span = n_dest * bar_width + group_gap
    bar_x = np.array(
        [
            row["methane_idx"] * group_span + row["destination_idx"] * bar_width
            for row in all_records
        ],
        dtype=float,
    )
    # Shift so each group's bars are centered on the integer index m_idx.
    center_offset = (n_dest - 1) * bar_width / 2.0
    group_centers = np.arange(n_methane) * group_span + center_offset
    bar_x = bar_x
    # Just use bar_x as-is; recompute group_centers from the bar positions
    # we actually placed so x-ticks line up exactly with the center of each
    # group of bars.
    group_centers = np.array(
        [
            bar_x[result_df[result_df["methane_idx"] == m].index].mean()
            for m in range(n_methane)
        ]
    )

    # Secondary y-axis denominator: average captured across the CCS bars in
    # the first methane group (keeps the ratio axis anchored to a single
    # methane assumption rather than averaging across inconsistent ones).
    ccs_scenarios_first = [
        s
        for s in destination_scenarios
        if not s.get("is_unabated") and not s.get("is_existing")
    ]
    if ccs_scenarios_first:
        _truth = scenarios_df[scenarios_df["methane_scenario"] == methane_scenarios[0]]
        if states is not None and "project_state" in _truth.columns:
            _truth = _truth[_truth["project_state"].isin(states)]
        per_scen_cap = [
            _truth[_truth["scenario"] == s["name"]][
                "mtco2_per_yr_captured_with_ccs"
            ].sum()
            for s in ccs_scenarios_first
        ]
        total_captured = sum(per_scen_cap) / max(len(per_scen_cap), 1)
    else:
        total_captured = 0.0

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    pos_bottom = np.zeros(len(bar_x))
    neg_bottom = np.zeros(len(bar_x))

    # Slimmest-version lumping — same as sibling, but ordered CO2 → CH4 →
    # CCS so the stack reads bottom-up as industrial CO2, industrial CH4,
    # then everything CCS-related on top.
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
            ("_industrial_co2", r"Industrial $\mathbf{CO_2}$", "#888888"),
            ("_industrial_ch4", r"Industrial $\mathbf{CH_4}$", "#e07b39"),
            ("_stored_co2", r"$\mathbf{CO_2}$ stored with CCS", "#2e7d32"),
            ("ccs_ch4", r"CCS-caused $\mathbf{CH_4}$", "#b71c1c"),
            ("ccs_co2", r"CCS-caused $\mathbf{CO_2}$", "#ef5350"),
        ]
    else:
        # Non-slim: same components as `_BREAKDOWN_COMPONENTS` but reordered
        # bottom-up as uncaptured CO2 → CH4 → captured CO2 → CCS-caused.
        breakdown_components = [
            (
                "uncaptured_process_co2",
                r"Uncaptured process $\mathbf{CO_2}$",
                "#777777",
            ),
            (
                "uncaptured_combustion_co2",
                r"Uncaptured combustion $\mathbf{CO_2}$",
                "#aaaaaa",
            ),
            ("process_ch4", r"Process $\mathbf{CH_4}$", "#e07b39"),
            ("combustion_ch4", r"Combustion $\mathbf{CH_4}$", "#f4a460"),
            (
                "captured_process_co2",
                r"Process $\mathbf{CO_2}$ stored with CCS",
                "#2e7d32",
            ),
            (
                "captured_combustion_co2",
                r"Combustion $\mathbf{CO_2}$ stored with CCS",
                "#66bb6a",
            ),
            ("ccs_ch4", r"CCS-caused $\mathbf{CH_4}$", "#b71c1c"),
            ("ccs_co2", r"CCS-caused $\mathbf{CO_2}$", "#ef5350"),
        ]

    for col, label, color in breakdown_components:
        vals = result_df[col].values
        pos_vals = np.where(vals > 0, vals, 0)
        neg_vals = np.where(vals < 0, vals, 0)
        if pos_vals.any():
            ax.bar(
                bar_x,
                pos_vals,
                width=bar_width,
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
                bar_x,
                neg_vals,
                width=bar_width,
                bottom=neg_bottom,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                label=label if not pos_vals.any() else None,
                zorder=2,
            )
            neg_bottom += neg_vals

    # Green dashed captured overlay only on CCS bars.
    captured_vals = result_df["_captured"].values.copy()
    non_ccs_mask = result_df["_is_unabated"].values | result_df["_is_existing"].values
    captured_vals[non_ccs_mask] = 0.0
    ccs_mask = ~non_ccs_mask
    _captured_green = "#2e7d32"
    if ccs_mask.any():
        ax.bar(
            bar_x[ccs_mask],
            captured_vals[ccs_mask],
            width=bar_width,
            bottom=pos_bottom[ccs_mask],
            facecolor="none",
            edgecolor=_captured_green,
            linewidth=1.5,
            linestyle="--",
            zorder=2,
            label=(
                None if annotate_captured_inline else r"$\mathbf{CO_2}$ captured by CCS"
            ),
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

    # Thick black "Net" line across the top of every bar.
    line_overhang = 0.04
    for j in range(len(bar_x)):
        ax.plot(
            [
                bar_x[j] - bar_width / 2 - line_overhang,
                bar_x[j] + bar_width / 2 + line_overhang,
            ],
            [pos_bottom[j], pos_bottom[j]],
            color="black",
            linewidth=1.75,
            zorder=4,
            solid_capstyle="butt",
        )

    # Per-bar Net label — integer only, centered on top of the bar. For CCS
    # bars we additionally emit the (X.X tCO2e/tCO2 injected) parenthetical
    # below the Net integer, unless `show_per_tco2_label=False`.
    label_gap = 3
    label_line_height = 24
    for j in range(len(bar_x)):
        row = result_df.iloc[j]
        bar_top = pos_bottom[j]
        if row["_is_unabated"] or row["_is_existing"]:
            net_mt = row["_net"]
            ax.text(
                bar_x[j],
                bar_top + label_gap,
                f"{net_mt:.0f}",
                ha="center",
                va="bottom",
                fontsize=fs["bar_label"],
                fontfamily="Arial",
                fontweight="bold",
                zorder=5,
                path_effects=[pe.withStroke(linewidth=5, foreground="white")],
            )
            continue
        scen_rows = scenarios_df[
            (scenarios_df["methane_scenario"] == row["methane_scenario"])
            & (scenarios_df["scenario"] == row["scenario"])
        ]
        net_mt = scen_rows["mtco2e_per_yr_emissions_including_ccs"].sum()
        captured_j = scen_rows["mtco2_per_yr_captured_with_ccs"].sum()
        per_tco2 = net_mt / captured_j if captured_j else 0
        if show_per_tco2_label:
            ax.text(
                bar_x[j],
                bar_top + label_gap,
                f"({per_tco2:.1f}" + r" tCO$_2$e/tCO$_2$ inj.)",
                ha="center",
                va="bottom",
                fontsize=fs["bar_label"] - 2,
                fontfamily="Arial Narrow",
                fontweight="normal",
                zorder=5,
                path_effects=[pe.withStroke(linewidth=5, foreground="white")],
            )
            net_y = bar_top + label_gap + label_line_height
        else:
            net_y = bar_top + label_gap
        ax.text(
            bar_x[j],
            net_y,
            f"{net_mt:.0f}",
            ha="center",
            va="bottom",
            fontsize=fs["bar_label"],
            fontfamily="Arial",
            fontweight="bold",
            zorder=5,
            path_effects=[pe.withStroke(linewidth=5, foreground="white")],
        )

    # Inline captured-CCS annotation — target the first CCS bar in the plot.
    if annotate_captured_inline:
        _ccs_idx = np.where(ccs_mask)[0]
        if len(_ccs_idx) > 0:
            j_tgt = int(_ccs_idx[0])
            top_of_box_y = pos_bottom[j_tgt] + captured_vals[j_tgt]
            arrow_tip = (bar_x[j_tgt], top_of_box_y)
            text_x = bar_x[j_tgt] - 0.25
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
                arrowprops={
                    "arrowstyle": "->",
                    "color": _captured_green,
                    "linewidth": 1.5,
                    "shrinkA": 2,
                    "shrinkB": 0,
                },
                zorder=5,
                path_effects=[pe.withStroke(linewidth=4, foreground="white")],
            )

    # X-tick labels at group centers = methane scenario display names.
    def _methane_label(name: str) -> str:
        if methane_scenario_labels and name in methane_scenario_labels:
            return methane_scenario_labels[name]
        return name

    ax.set_xticks(group_centers)
    tick_labels = [_methane_label(m) for m in methane_scenarios]
    wrapped_labels = [
        "\n".join(textwrap.fill(line, width=22) for line in lbl.split("\n"))
        for lbl in tick_labels
    ]
    ax.set_xticklabels(wrapped_labels, fontsize=fs["tick"], fontfamily="Arial")

    # Sub-bar destination labels — short italic labels placed in the gap
    # between the plot area and the methane-scenario x-tick labels. We
    # bump the main x-tick pad so there is room for the sublabels, then
    # draw them in axes-fraction coords just below the x-axis spine. Short
    # default labels (keyed by scenario name) + a small rotation keep
    # neighboring sublabels from running into each other at typical group
    # widths.
    _default_dest_sublabels = {
        "existing": "exist.",
        "unabated": "unab.",
        "baseline": "stated",
        "current_mix": "CCS",
        "all_eor": "100% EOR",
        "all_gs": "100% GS",
        "nameplate_implied": "name.",
    }
    if show_destination_sublabels and n_dest > 1:

        def _dest_label(scen: dict) -> str:
            name = scen["name"]
            if destination_scenario_labels and name in destination_scenario_labels:
                return destination_scenario_labels[name]
            if name in _default_dest_sublabels:
                return _default_dest_sublabels[name]
            # fall back to the full label with newlines flattened
            return scen["label"].replace("\n", " ")

        # Right-align rotated text so the right edge of each sublabel ends
        # under its own bar (keeps labels anchored visually to the bar
        # above them). Push the main x-tick labels further down to clear
        # the rotated-sublabel row.
        rot = float(destination_sublabel_rotation)
        if abs(rot) < 1e-6:
            ha = "center"
            tick_pad = 24
        else:
            ha = "right"
            tick_pad = 44
        ax.tick_params(axis="x", pad=tick_pad)
        trans = ax.get_xaxis_transform()
        for j in range(len(bar_x)):
            row = result_df.iloc[j]
            scen = destination_scenarios[int(row["destination_idx"])]
            ax.text(
                bar_x[j],
                -0.015,  # axes-fraction y, just below the x-axis spine
                _dest_label(scen),
                transform=trans,
                ha=ha,
                va="top",
                rotation=rot,
                rotation_mode="anchor",
                fontsize=fs["tick"] - 2,
                fontfamily="Arial",
                fontstyle="italic",
                color="#444444",
            )

    ax.set_ylabel(
        r"MtCO$_2$e / yr",
        fontsize=fs["axis_label"],
        fontfamily="Arial",
        labelpad=1,
    )
    y_min = neg_bottom.min()
    y_max = ghost_top.max() * 1.10
    ax.set_ylim(y_min - 1 if y_min < 0 else -1, y_max)
    ax.tick_params(axis="y", labelsize=fs["tick"])

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
        title = (
            f"{state_label} CCS: state-total emissions across methane "
            r"assumptions (MtCO$_2$e/yr)"
        )
    ax.set_title(title, fontsize=fs["title"], fontfamily="Arial", pad=20)

    ax.grid(axis="y", color="lightgray", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False)
    if not total_captured > 0 and show_per_tco2_label:
        ax.spines["right"].set_visible(False)

    handles, labels_leg = ax.get_legend_handles_labels()
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels_leg):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)
    if display_mode == "presentation":
        legendfontsize = fs["legend"] + 1
    else:
        legendfontsize = fs["legend"]

    bbox_y_below = -0.28 if show_destination_sublabels and n_dest > 1 else -0.2

    if legend_right:
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
            bbox_to_anchor=(0.5, bbox_y_below),
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


def plot_state_total_by_methane_scenario_breakdown(
    industrial_emissions_df: pd.DataFrame,
    ccs_emissions_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    destination_scenarios: List[Union[str, dict]],
    methane_scenarios: List[str],
    industries: Optional[List[str]] = None,
    which_flavor: Optional[List[str]] = None,
    states: Optional[List[str]] = None,
    title: Optional[str] = None,
    legend_below: bool = True,
    figsize: Tuple[int, int] = (16, 8),
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    display_mode: Optional[str] = None,
    destination_scenario_labels: Optional[Dict[str, str]] = None,
    methane_scenario_labels: Optional[Dict[str, str]] = None,
    legend_ncol: int = 3,
    slimmest_version: bool = False,
    legend_right: bool = False,
    legend_right_axes_fraction: float = 0.72,
    legend_right_valign: str = "top",
    group_gap: float = 0.5,
    show_methane_sublabels: bool = True,
    methane_sublabel_rotation: float = 25.0,
    show_net_label: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Grouped-bar version of `plot_state_total_by_methane_scenario` with
    the stacked source breakdown preserved on every bar.

    Each destination scenario in `destination_scenarios` becomes a group on
    the X-axis; inside each group there is one stacked bar per entry of
    `methane_scenarios`. The pseudo-scenarios ``"existing"`` and
    ``"unabated"`` are supported in `destination_scenarios` just like the
    sibling functions. Within each bar, components stack bottom-up as
    industrial CO2 → industrial CH4 → CCS-related (same convention as
    `plot_state_emissions_by_methane_scenario`). Methane assumption is
    conveyed by position inside the group plus a small italic sub-label
    under each sub-bar.

    Args:
        destination_scenarios: X-axis groups. Strings (``"existing"``,
            ``"unabated"``, ``"baseline"``, ``"current_mix"``, ``"all_eor"``,
            ``"all_gs"``, ``"nameplate_implied"``) or custom dicts, same
            semantics as the sibling.
        methane_scenarios: Ordered list of methane_scenario names rendered
            as adjacent sub-bars inside each destination group.
        destination_scenario_labels / methane_scenario_labels: Optional
            label-override dicts. Destination labels become the group
            x-tick label; methane labels become the italic sub-bar label.
        show_net_label: Whether to annotate each bar with its net integer
            MtCO2e/yr. Default True.
        (other args mirror `plot_state_emissions_by_methane_scenario`.)
    """
    display_mode = _resolve_display_mode(display_mode)
    fs = _resolve_fonts(display_mode)

    # When BOTH slim mode and presentation mode are on, bump the x-tick, y-
    # axis-label, and legend font sizes extra on top of the preset bump the
    # "presentation" preset already applies. The slim palette has fewer
    # legend entries so there's room to go larger, and presentations
    # typically project onto a screen where even "presentation"-preset
    # fonts can read small.
    _slim_present_boost = slimmest_version and display_mode == "presentation"
    xtick_fs = fs["tick"] + 3 if _slim_present_boost else fs["tick"]
    yaxis_label_fs = fs["axis_label"] + 3 if _slim_present_boost else fs["axis_label"]

    if not methane_scenarios:
        raise ValueError("methane_scenarios must contain at least one entry.")
    if not destination_scenarios:
        raise ValueError("destination_scenarios must contain at least one entry.")

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
        "all_gs": {"name": "all_gs", "label": "100% GS", "eor_pct": 0},
        "baseline": {
            "name": "baseline",
            "label": "Announced\ndestinations",
            "from_data": True,
        },
        "current_mix": {
            "name": "current_mix",
            "label": "Current U.S.\npractice (92% EOR)",
            "eor_pct": 92,
        },
        "nameplate_implied": {
            "name": "nameplate_implied",
            "label": "Nameplate",
            "zero_lifecycle": True,
        },
        "all_eor": {"name": "all_eor", "label": "100% EOR", "eor_pct": 100},
        "switching": {
            "name": "switching",
            "label": "Stated →\n100% EOR",
            "from_data": True,
            "is_switching": True,
        },
        "switching_from_current": {
            "name": "switching_from_current",
            "label": "Current →\n100% EOR",
            "from_data": True,
            "is_switching": True,
        },
        "unabated": {
            "name": "unabated",
            "label": "Unabated\n(no CCS)",
            "is_unabated": True,
        },
    }

    resolved: List[dict] = []
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

    component_cols = [col for col, _, _ in _BREAKDOWN_COMPONENTS]

    # Compute per-bar records: outer loop is destination (group); inner loop
    # is methane (sub-bar).
    all_records: List[dict] = []
    for d_idx, scen in enumerate(destination_scenarios):
        is_unabated = bool(scen.get("is_unabated"))
        is_existing = bool(scen.get("is_existing"))
        for m_idx, m_name in enumerate(methane_scenarios):
            if is_unabated:
                record = _compute_unabated_record(
                    industrial_emissions_df,
                    scenarios_df,
                    m_name,
                    industries=industries,
                    which_flavor=which_flavor,
                )
            elif is_existing:
                if "retrofit_newbuild" not in scenarios_df.columns:
                    raise ValueError(
                        "'existing' scenario requested but scenarios_df is "
                        "missing the 'retrofit_newbuild' column."
                    )
                retro_mask = (
                    scenarios_df["retrofit_newbuild"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    == "retrofit"
                )
                retro_scen_df = scenarios_df[retro_mask]
                if retro_scen_df.empty:
                    record = {
                        col: 0.0 for col in component_cols + ["_captured", "_net"]
                    }
                else:
                    record = _compute_unabated_record(
                        industrial_emissions_df,
                        retro_scen_df,
                        m_name,
                        industries=industries,
                        which_flavor=which_flavor,
                    )
            else:
                # CCS scenario — compute per-state then aggregate
                if "project_state" in scenarios_df.columns:
                    active_states = sorted(
                        scenarios_df["project_state"].dropna().unique()
                    )
                else:
                    active_states = [None]
                per_state_records, per_state_dac = [], []
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
                            m_name,
                            [scen],
                            industries,
                            which_flavor,
                            component_cols,
                        )
                    )
                    per_state_dac.append(
                        _compute_dac_records(ccs_s, scen_s, m_name, [scen])
                    )
                record = {}
                for col in component_cols + ["_captured"]:
                    record[col] = sum(r[0][col] for r in per_state_records)
                    record[col] += sum(d[0].get(col, 0.0) for d in per_state_dac)
                record["_net"] = sum(record[c] for c in component_cols)

            row = {
                "destination": scen["name"],
                "destination_label": scen["label"],
                "destination_idx": d_idx,
                "methane_scenario": m_name,
                "methane_idx": m_idx,
                "_is_unabated": is_unabated,
                "_is_existing": is_existing,
            }
            for col in component_cols + ["_captured"]:
                row[col] = record.get(col, 0.0)
            row["_net"] = sum(row[c] for c in component_cols)
            all_records.append(row)

    result_df = pd.DataFrame(all_records)

    # Grouped-bar layout: destination groups along X, methane sub-bars within.
    n_dest_groups = len(destination_scenarios)
    n_methane = len(methane_scenarios)
    bar_width = 0.8
    group_span = n_methane * bar_width + group_gap
    bar_x = np.array(
        [
            row["destination_idx"] * group_span + row["methane_idx"] * bar_width
            for row in all_records
        ],
        dtype=float,
    )
    group_centers = np.array(
        [
            bar_x[result_df[result_df["destination_idx"] == g].index].mean()
            for g in range(n_dest_groups)
        ]
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    pos_bottom = np.zeros(len(bar_x))
    neg_bottom = np.zeros(len(bar_x))

    # Same bottom-up stacking order as the methane-grouped sibling: industrial
    # CO2 → industrial CH4 → captured CO2 (CCS stored) → CCS-caused.
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
            ("_industrial_co2", r"Industrial $\mathbf{CO_2}$", "#888888"),
            ("_industrial_ch4", r"Industrial $\mathbf{CH_4}$", "#e07b39"),
            ("_stored_co2", r"$\mathbf{CO_2}$ stored with CCS", "#2e7d32"),
            ("ccs_ch4", r"CCS-caused $\mathbf{CH_4}$", "#b71c1c"),
            ("ccs_co2", r"CCS-caused $\mathbf{CO_2}$", "#ef5350"),
        ]
    else:
        breakdown_components = [
            (
                "uncaptured_process_co2",
                r"Uncaptured process $\mathbf{CO_2}$",
                "#777777",
            ),
            (
                "uncaptured_combustion_co2",
                r"Uncaptured combustion $\mathbf{CO_2}$",
                "#aaaaaa",
            ),
            ("process_ch4", r"Process $\mathbf{CH_4}$", "#e07b39"),
            ("combustion_ch4", r"Combustion $\mathbf{CH_4}$", "#f4a460"),
            (
                "captured_process_co2",
                r"Process $\mathbf{CO_2}$ stored with CCS",
                "#2e7d32",
            ),
            (
                "captured_combustion_co2",
                r"Combustion $\mathbf{CO_2}$ stored with CCS",
                "#66bb6a",
            ),
            ("ccs_ch4", r"CCS-caused $\mathbf{CH_4}$", "#b71c1c"),
            ("ccs_co2", r"CCS-caused $\mathbf{CO_2}$", "#ef5350"),
        ]

    for col, label, color in breakdown_components:
        vals = result_df[col].values
        pos_vals = np.where(vals > 0, vals, 0)
        neg_vals = np.where(vals < 0, vals, 0)
        if pos_vals.any():
            ax.bar(
                bar_x,
                pos_vals,
                width=bar_width,
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
                bar_x,
                neg_vals,
                width=bar_width,
                bottom=neg_bottom,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                label=label if not pos_vals.any() else None,
                zorder=2,
            )
            neg_bottom += neg_vals

    # Green dashed captured overlay on CCS bars.
    captured_vals = result_df["_captured"].values.copy()
    non_ccs_mask = result_df["_is_unabated"].values | result_df["_is_existing"].values
    captured_vals[non_ccs_mask] = 0.0
    ccs_mask = ~non_ccs_mask
    if ccs_mask.any():
        ax.bar(
            bar_x[ccs_mask],
            captured_vals[ccs_mask],
            width=bar_width,
            bottom=pos_bottom[ccs_mask],
            facecolor="none",
            edgecolor="#2e7d32",
            linewidth=1.5,
            linestyle="--",
            zorder=2,
            label=r"$\mathbf{CO_2}$ captured by CCS",
        )
    ghost_top = pos_bottom + captured_vals

    ax.axhline(
        0, color="black", linewidth=1.0, zorder=3, xmin=-0.03, xmax=1.03, clip_on=False
    )

    # Net integer label on top of each bar.
    label_gap = 3
    if show_net_label:
        for j in range(len(bar_x)):
            net_mt = result_df.iloc[j]["_net"]
            ax.text(
                bar_x[j],
                pos_bottom[j] + label_gap,
                f"{net_mt:.0f}",
                ha="center",
                va="bottom",
                fontsize=fs["bar_label"] - 1,
                fontfamily="Arial",
                fontweight="bold",
                zorder=5,
                path_effects=[pe.withStroke(linewidth=5, foreground="white")],
            )

    # X-tick labels at group centers = destination scenario labels.
    def _dest_label(scen: dict) -> str:
        if destination_scenario_labels and scen["name"] in destination_scenario_labels:
            return destination_scenario_labels[scen["name"]]
        return scen["label"]

    ax.set_xticks(group_centers)
    tick_labels = [_dest_label(scen) for scen in destination_scenarios]
    wrapped_labels = [
        "\n".join(textwrap.fill(line, width=22) for line in lbl.split("\n"))
        for lbl in tick_labels
    ]
    ax.set_xticklabels(wrapped_labels, fontsize=xtick_fs, fontfamily="Arial")

    # Methane sub-labels under each sub-bar (short defaults; rotation
    # configurable since methane names tend to be longer than destination
    # abbreviations).
    if show_methane_sublabels and n_methane > 1:

        def _m_label(name: str) -> str:
            if methane_scenario_labels and name in methane_scenario_labels:
                return methane_scenario_labels[name]
            return name

        rot = float(methane_sublabel_rotation)
        if abs(rot) < 1e-6:
            ha = "center"
            tick_pad = 24
        else:
            ha = "right"
            tick_pad = 52
        ax.tick_params(axis="x", pad=tick_pad)
        trans = ax.get_xaxis_transform()
        for j in range(len(bar_x)):
            name = result_df.iloc[j]["methane_scenario"]
            ax.text(
                bar_x[j],
                -0.015,
                _m_label(name),
                transform=trans,
                ha=ha,
                va="top",
                rotation=rot,
                rotation_mode="anchor",
                fontsize=fs["tick"] - 2,
                fontfamily="Arial",
                fontstyle="italic",
                color="#444444",
            )

    ax.set_ylabel(
        r"MtCO$_2$e / yr",
        fontsize=yaxis_label_fs,
        fontfamily="Arial",
        labelpad=1,
    )
    y_min = neg_bottom.min()
    y_max = ghost_top.max() * 1.10
    ax.set_ylim(y_min - 1 if y_min < 0 else -1, y_max)
    ax.tick_params(axis="y", labelsize=xtick_fs)

    # Negative tick below zero.
    _current_yticks = np.asarray(ax.get_yticks())
    _pos_ticks = _current_yticks[_current_yticks > 0]
    if len(_pos_ticks) >= 1 and (_current_yticks < 0).sum() == 0:
        tick_spacing = (
            float(_pos_ticks[1] - _pos_ticks[0])
            if len(_pos_ticks) >= 2
            else float(_pos_ticks[0])
        )
        negative_tick = -tick_spacing
        ax.set_yticks(np.concatenate(([negative_tick], _current_yticks)))
        lo, hi = ax.get_ylim()
        if lo > negative_tick - tick_spacing * 0.05:
            ax.set_ylim(negative_tick - tick_spacing * 0.05, hi)

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
        title = (
            f"{state_label} CCS: state-total emissions by destination, "
            r"grouped sub-bars per methane assumption (MtCO$_2$e/yr)"
        )
    ax.set_title(title, fontsize=fs["title"], fontfamily="Arial", pad=20)

    ax.grid(axis="y", color="lightgray", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(bottom=False)

    handles, labels_leg = ax.get_legend_handles_labels()
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels_leg):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)
    if _slim_present_boost:
        legendfontsize = fs["legend"] + 4
    elif display_mode == "presentation":
        legendfontsize = fs["legend"] + 1
    else:
        legendfontsize = fs["legend"]
    bbox_y_below = -0.32 if show_methane_sublabels and n_methane > 1 else -0.2

    if legend_right:
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
            bbox_to_anchor=(0.5, bbox_y_below),
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


def plot_state_emissions_dual_axis_by_state(
    industrial_emissions_df: pd.DataFrame,
    ccs_emissions_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    methane_scenario: str,
    destination_scenarios: List[Union[str, dict]],
    states: List[str],
    industries: Optional[List[str]] = None,
    which_flavor: Optional[List[str]] = None,
    state_hatches: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    legend_below: bool = True,
    figsize: Tuple[int, int] = (16, 8),
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    display_mode: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Same as plot_state_emissions_dual_axis, but with per-state grouped bars.

    Each destination-scenario x-position gets one narrow bar per state, drawn
    side-by-side. Component colors are shared across states (from
    ``_BREAKDOWN_COMPONENTS``); states are distinguished by hatching (default
    None for the first state and "///" for the second). State labels appear
    below each bar; scenario labels below the group.

    Args:
        states: Explicit list of `project_state` values to include (e.g.
            ["TX", "LA"]). Required — this variant is only meaningful with
            multiple states.
        state_hatches: Optional mapping from project_state to matplotlib
            hatch pattern (e.g. {"TX": "", "LA": "///"}). If None, uses
            no-hatch for the first state and "///" for subsequent ones.
    """
    if len(states) < 1:
        raise ValueError("`states` must contain at least one state.")

    display_mode = _resolve_display_mode(display_mode)
    fs = _resolve_fonts(display_mode)

    default_hatches = ["", "///", "\\\\\\", "xxx"]
    if state_hatches is None:
        state_hatches = {
            s: default_hatches[i % len(default_hatches)] for i, s in enumerate(states)
        }

    known_dest_scenarios = {
        "all_gs": {
            "name": "all_gs",
            "label": "100% Geologic storage (0% EOR)",
            "eor_pct": 0,
        },
        "baseline": {
            "name": "baseline",
            "label": "Announced destinations",
            "from_data": True,
        },
        "current_mix": {
            "name": "current_mix",
            "label": "Current U.S. practice",
            "eor_pct": 92,
        },
        "nameplate_implied": {
            "name": "nameplate_implied",
            "label": "Nameplate 'capture capacity'",
            "zero_lifecycle": True,
        },
        "all_eor": {
            "name": "all_eor",
            "label": "100% EOR",
            "eor_pct": 100,
        },
        "switching": {
            "name": "switching",
            "label": "Stated →\n100% EOR after 45Q",
            "from_data": True,
            "is_switching": True,
        },
        "switching_from_current": {
            "name": "switching_from_current",
            "label": "Current →\n100% EOR after 45Q",
            "from_data": True,
            "is_switching": True,
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

    # Compute per-(state, destination_scenario) records. For each state we
    # filter the three dataframes and reuse the same record-assembly logic as
    # plot_state_emissions_dual_axis.
    per_state_records: Dict[str, List[Dict]] = {}
    for state in states:
        ie_s = industrial_emissions_df
        ccs_s = ccs_emissions_df
        scen_s = scenarios_df
        if "project_state" in ie_s.columns:
            ie_s = ie_s[ie_s["project_state"] == state]
        if "project_state" in ccs_s.columns:
            ccs_s = ccs_s[ccs_s["project_state"] == state]
        if "project_state" in scen_s.columns:
            scen_s = scen_s[scen_s["project_state"] == state]

        eor_df = _build_breakdown_data(
            ie_s,
            ccs_s,
            methane_scenario,
            "eor",
            industries,
            which_flavor,
        ).set_index("simple_product_name")
        gs_df = _build_breakdown_data(
            ie_s,
            ccs_s,
            methane_scenario,
            "gs",
            industries,
            which_flavor,
        ).set_index("simple_product_name")

        common = eor_df.index.intersection(gs_df.index)
        component_cols = [col for col, _, _ in _BREAKDOWN_COMPONENTS]
        ccs_cols = {"ccs_co2", "ccs_ch4"}

        ref = scen_s[scen_s["methane_scenario"] == methane_scenario].drop_duplicates(
            subset="uid"
        )
        ref = ref[ref["product"].isin(common)]
        captured_by_industry = ref.groupby("product")[
            "mtco2_per_yr_captured_with_ccs"
        ].sum()
        total_captured = captured_by_industry.sum()

        records = []
        for scen in destination_scenarios:
            row = {"scenario": scen["name"], "label": scen["label"], "state": state}
            zero_lc = scen.get("zero_lifecycle", False)
            if scen.get("from_data"):
                scen_df = scen_s[
                    (scen_s["methane_scenario"] == methane_scenario)
                    & (scen_s["scenario"] == scen["name"])
                ]
                scen_df = scen_df[scen_df["product"].isin(common)]
                # Drop rows with NaN captured/eor_fraction — a single NaN row
                # would make the weighted sum NaN and blank the whole bar.
                # (Typically caused by missing product_amount / unabated in the
                # source projects CSV; user should still fix upstream.)
                scen_df = scen_df.dropna(
                    subset=["mtco2_per_yr_captured_with_ccs", "eor_fraction"]
                )
                for col in component_cols:
                    if zero_lc and col in ccs_cols:
                        row[col] = 0.0
                        continue
                    weighted = 0.0
                    for _, proj in scen_df.iterrows():
                        ef = proj["eor_fraction"]
                        ind = proj["product"]
                        factor = (
                            ef * eor_df.loc[ind, col] + (1 - ef) * gs_df.loc[ind, col]
                        )
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
                        (
                            eor_frac * eor_df.loc[ind, col]
                            + gs_frac * gs_df.loc[ind, col]
                        )
                        * captured_by_industry[ind]
                        for ind in captured_by_industry.index
                    )
                row["_captured"] = total_captured
            row["_net"] = sum(row[c] for c in component_cols)
            records.append(row)
        per_state_records[state] = records

    # Grand total captured (for secondary axis).
    grand_total_captured = sum(
        r["_captured"] for state_recs in per_state_records.values() for r in state_recs
    ) / max(
        len(states), 1
    )  # per-state avg so secondary axis is per-tCO2 per state

    n_scen = len(destination_scenarios)
    n_states = len(states)
    x = np.arange(n_scen)
    group_width = 0.75
    bar_width = group_width / n_states

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    component_cols = [col for col, _, _ in _BREAKDOWN_COMPONENTS]

    max_top = 0.0
    min_bot = 0.0
    seen_component_labels = set()
    _seen_state_labels = set()

    for si, state in enumerate(states):
        offset = (si - (n_states - 1) / 2) * bar_width
        bar_x = x + offset
        state_records_df = pd.DataFrame(per_state_records[state])
        pos_bottom = np.zeros(n_scen)
        neg_bottom = np.zeros(n_scen)

        for col, label, color in _BREAKDOWN_COMPONENTS:
            vals = state_records_df[col].values
            pos_vals = np.where(vals > 0, vals, 0)
            neg_vals = np.where(vals < 0, vals, 0)
            label_to_use = label if label not in seen_component_labels else None
            if pos_vals.any():
                ax.bar(
                    bar_x,
                    pos_vals,
                    width=bar_width,
                    bottom=pos_bottom,
                    color=color,
                    edgecolor="white",
                    linewidth=0.5,
                    hatch=state_hatches.get(state, ""),
                    label=label_to_use,
                    zorder=2,
                )
                pos_bottom += pos_vals
                if label_to_use is not None:
                    seen_component_labels.add(label)
            if neg_vals.any():
                ax.bar(
                    bar_x,
                    neg_vals,
                    width=bar_width,
                    bottom=neg_bottom,
                    color=color,
                    edgecolor="white",
                    linewidth=0.5,
                    hatch=state_hatches.get(state, ""),
                    label=label_to_use if label not in seen_component_labels else None,
                    zorder=2,
                )
                neg_bottom += neg_vals
                if label not in seen_component_labels:
                    seen_component_labels.add(label)

        captured_vals = state_records_df["_captured"].values
        ax.bar(
            bar_x,
            captured_vals,
            width=bar_width,
            bottom=pos_bottom,
            facecolor="none",
            edgecolor="#2e7d32",
            linewidth=1.2,
            linestyle="--",
            zorder=2,
            label=r"$\mathbf{CO_2}$ captured by CCS"
            if "captured" not in seen_component_labels
            else None,
        )
        seen_component_labels.add("captured")
        ghost_top = pos_bottom + captured_vals
        max_top = max(max_top, ghost_top.max())
        min_bot = min(min_bot, neg_bottom.min())

        # Per-bar net label
        for j in range(n_scen):
            net_mt = sum(state_records_df.iloc[j][c] for c in component_cols)
            ax.text(
                bar_x[j],
                pos_bottom[j] + 3,
                f"{net_mt:.0f}",
                ha="center",
                va="bottom",
                fontsize=fs["bar_label"] - 2,
                fontfamily="Arial",
                fontweight="bold",
                zorder=5,
                path_effects=[pe.withStroke(linewidth=4, foreground="white")],
            )
            # State abbreviation directly below the bar
            ax.text(
                bar_x[j],
                min(0, neg_bottom[j]) - (max_top - min_bot) * 0.02,
                state,
                ha="center",
                va="top",
                fontsize=fs["tick"] - 1,
                fontfamily="Arial",
                color="#555555",
                zorder=5,
            )

    ax.axhline(0, color="black", linewidth=1.0, zorder=3)

    # Unabated line: sum across all states/projects for the methane scenario
    unabated_df = scenarios_df[scenarios_df["methane_scenario"] == methane_scenario]
    if "project_state" in unabated_df.columns:
        unabated_df = unabated_df[unabated_df["project_state"].isin(states)]
    unabated_df = unabated_df.drop_duplicates(subset="uid")
    baseload_val = unabated_df["unabated_mtco2e_per_yr"].sum()
    ax.axhline(
        baseload_val,
        color="black",
        linewidth=1.5,
        linestyle="--",
        zorder=3,
        label="Total unabated emissions",
    )
    max_top = max(max_top, baseload_val)

    ax.set_xticks(x)
    wrapped_labels = [
        "\n".join(textwrap.fill(line, width=20) for line in s["label"].split("\n"))
        for s in destination_scenarios
    ]
    # Pad tick labels down so state abbreviations fit between them.
    ax.set_xticklabels(wrapped_labels, fontsize=fs["tick"], fontfamily="Arial")
    ax.tick_params(axis="x", pad=18)

    ax.set_ylabel(
        r"MtCO$_2$e / yr",
        fontsize=fs["axis_label"],
        fontfamily="Arial",
        labelpad=1,
    )
    ax.set_ylim(min_bot - 1 if min_bot < 0 else -1, max_top * 1.20)
    ax.tick_params(axis="y", labelsize=fs["tick"])

    if grand_total_captured > 0:
        ax2 = ax.secondary_yaxis(
            "right",
            functions=(
                lambda y: y / grand_total_captured,
                lambda y: y * grand_total_captured,
            ),
        )
        ax2.set_ylabel(
            r"tCO$_2$e emitted per tCO$_2$ injected (avg across states)",
            fontsize=fs["axis_label"] - 1,
            fontfamily="Arial",
            labelpad=2,
        )
        ax2.tick_params(axis="y", labelsize=fs["tick"])

    if title is None:
        state_label = _format_states_for_title(states)
        title = f"{state_label} CCS: per-state emissions (MtCO$_2$e/yr)"
    ax.set_title(title, fontsize=fs["title"], fontfamily="Arial", pad=20)

    ax.grid(axis="y", color="lightgray", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False)

    # Add a small legend entry showing which hatch = which state.
    from matplotlib.patches import Patch

    state_legend_handles = [
        Patch(
            facecolor="#cccccc",
            edgecolor="#333333",
            hatch=state_hatches.get(s, ""),
            label=_STATE_DISPLAY_NAMES.get(s, s),
        )
        for s in states
    ]

    handles, labels_leg = ax.get_legend_handles_labels()
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels_leg):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)
    # Append state legend entries at the end.
    unique_handles.extend(state_legend_handles)
    unique_labels.extend([h.get_label() for h in state_legend_handles])

    if legend_below:
        ax.legend(
            unique_handles,
            unique_labels,
            fontsize=fs["legend"],
            frameon=True,
            facecolor="white",
            edgecolor="none",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=3,
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
            bbox_to_anchor=(1.02, 1),
        )
        fig.tight_layout(rect=[0, 0, 0.72, 1])
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
    return fig, ax


def plot_emissions_dual_axis_by_methane_flag(
    industrial_emissions_df: pd.DataFrame,
    ccs_emissions_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    methane_scenario: str,
    destination_scenarios: List[Union[str, dict]],
    states: Optional[List[str]] = None,
    industries: Optional[List[str]] = None,
    which_flavor: Optional[List[str]] = None,
    flag_hatches: Optional[Dict[bool, str]] = None,
    flag_labels: Optional[Dict[bool, str]] = None,
    title: Optional[str] = None,
    legend_below: bool = True,
    figsize: Tuple[int, int] = (16, 8),
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    display_mode: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Dual-axis stacked bars grouped by is_methane_based.

    For each destination scenario, draws two adjacent bars — one for
    methane-based industries and one for non-methane-based — each showing the
    stacked component breakdown summed across the requested states. Uses
    hatching to distinguish the two groups (default: solid for methane-based,
    "///" for non-methane).

    Args:
        states: Filter on `project_state`. If None, uses all states in the
            data (each state's captured tonnes are weighted by that state's
            own per-industry factors, then summed).
        flag_hatches: Optional mapping from bool → matplotlib hatch pattern.
            Default: {True: "", False: "///"}.
        flag_labels: Optional mapping from bool → legend label. Default:
            {True: "Methane-based industries", False: "Non-methane-based industries"}.
    """
    display_mode = _resolve_display_mode(display_mode)
    fs = _resolve_fonts(display_mode)

    if flag_hatches is None:
        flag_hatches = {True: "", False: "///"}
    if flag_labels is None:
        flag_labels = {
            True: "Methane-based industries",
            False: "Non-methane-based industries",
        }

    if "is_methane_based" not in scenarios_df.columns:
        raise ValueError(
            "scenarios_df has no 'is_methane_based' column. Re-run the pipeline "
            "with the current CLI so the flag propagates into scenarios output."
        )

    # Apply optional state filter up-front.
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
        "all_gs": {
            "name": "all_gs",
            "label": "100% Geologic storage (0% EOR)",
            "eor_pct": 0,
        },
        "baseline": {
            "name": "baseline",
            "label": "Announced destinations",
            "from_data": True,
        },
        "current_mix": {
            "name": "current_mix",
            "label": "Current U.S. practice",
            "eor_pct": 92,
        },
        "nameplate_implied": {
            "name": "nameplate_implied",
            "label": "Nameplate 'capture capacity'",
            "zero_lifecycle": True,
        },
        "all_eor": {
            "name": "all_eor",
            "label": "100% EOR",
            "eor_pct": 100,
        },
        "switching": {
            "name": "switching",
            "label": "Stated →\n100% EOR after 45Q",
            "from_data": True,
            "is_switching": True,
        },
        "switching_from_current": {
            "name": "switching_from_current",
            "label": "Current →\n100% EOR after 45Q",
            "from_data": True,
            "is_switching": True,
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

    component_cols = [col for col, _, _ in _BREAKDOWN_COMPONENTS]

    # Determine which states to iterate over (for correct per-state factor
    # weighting inside each methane group).
    if "project_state" in scenarios_df.columns:
        active_states = sorted(scenarios_df["project_state"].dropna().unique())
    else:
        active_states = [None]

    flag_values = [True, False]

    def _filter_by_methane(df: pd.DataFrame, flag: bool) -> pd.DataFrame:
        if "is_methane_based" not in df.columns:
            return df
        return df[df["is_methane_based"] == flag]

    # Per (methane_flag, state) records: outer key = flag, inner list aligned to destinations.
    per_flag_records: Dict[bool, List[Dict]] = {}
    for flag in flag_values:
        ie_f = _filter_by_methane(industrial_emissions_df, flag)
        ccs_f = _filter_by_methane(ccs_emissions_df, flag)
        scen_f = _filter_by_methane(scenarios_df, flag)
        if scen_f.empty:
            # No projects match this flag; produce zero-filled records so bars still line up.
            per_flag_records[flag] = [
                {
                    "scenario": s["name"],
                    "label": s["label"],
                    "_captured": 0.0,
                    "_net": 0.0,
                    **{c: 0.0 for c in component_cols},
                }
                for s in destination_scenarios
            ]
            continue

        per_state_lists = []
        per_state_dac_lists = []
        for state in active_states:
            if state is None:
                ie_s, ccs_s, scen_s = ie_f, ccs_f, scen_f
            else:
                ie_s = (
                    ie_f[ie_f["project_state"] == state]
                    if "project_state" in ie_f.columns
                    else ie_f
                )
                ccs_s = (
                    ccs_f[ccs_f["project_state"] == state]
                    if "project_state" in ccs_f.columns
                    else ccs_f
                )
                scen_s = scen_f[scen_f["project_state"] == state]
            if scen_s.empty:
                continue
            per_state_lists.append(
                _compute_dual_axis_records(
                    ie_s,
                    ccs_s,
                    scen_s,
                    methane_scenario,
                    destination_scenarios,
                    industries,
                    which_flavor,
                    component_cols,
                )
            )
            per_state_dac_lists.append(
                _compute_dac_records(
                    ccs_s,
                    scen_s,
                    methane_scenario,
                    destination_scenarios,
                )
            )

        # Sum across states for this flag, folding in DAC contributions
        # (excluded by _compute_dual_axis_records because DAC has zero
        # per-unit-of-product emissions).
        combined = []
        for i, scen in enumerate(destination_scenarios):
            row = {"scenario": scen["name"], "label": scen["label"]}
            for col in component_cols + ["_captured"]:
                base = (
                    sum(r[i][col] for r in per_state_lists) if per_state_lists else 0.0
                )
                dac_add = sum(d[i].get(col, 0.0) for d in per_state_dac_lists)
                row[col] = base + dac_add
            row["_net"] = sum(row[c] for c in component_cols)
            combined.append(row)
        per_flag_records[flag] = combined

    # ---- Plot ----
    n_scen = len(destination_scenarios)
    n_groups = len(flag_values)
    x = np.arange(n_scen)
    group_width = 0.75
    bar_width = group_width / n_groups

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    max_top = 0.0
    min_bot = 0.0
    seen_component_labels: set = set()

    for gi, flag in enumerate(flag_values):
        offset = (gi - (n_groups - 1) / 2) * bar_width
        bar_x = x + offset
        records_df = pd.DataFrame(per_flag_records[flag])
        pos_bottom = np.zeros(n_scen)
        neg_bottom = np.zeros(n_scen)

        for col, label, color in _BREAKDOWN_COMPONENTS:
            vals = records_df[col].values
            pos_vals = np.where(vals > 0, vals, 0)
            neg_vals = np.where(vals < 0, vals, 0)
            lbl_pos = label if label not in seen_component_labels else None
            if pos_vals.any():
                ax.bar(
                    bar_x,
                    pos_vals,
                    width=bar_width,
                    bottom=pos_bottom,
                    color=color,
                    edgecolor="white",
                    linewidth=0.5,
                    hatch=flag_hatches.get(flag, ""),
                    label=lbl_pos,
                    zorder=2,
                )
                pos_bottom += pos_vals
                if lbl_pos is not None:
                    seen_component_labels.add(label)
            if neg_vals.any():
                lbl_neg = label if label not in seen_component_labels else None
                ax.bar(
                    bar_x,
                    neg_vals,
                    width=bar_width,
                    bottom=neg_bottom,
                    color=color,
                    edgecolor="white",
                    linewidth=0.5,
                    hatch=flag_hatches.get(flag, ""),
                    label=lbl_neg,
                    zorder=2,
                )
                neg_bottom += neg_vals
                if lbl_neg is not None:
                    seen_component_labels.add(label)

        captured_vals = records_df["_captured"].values
        cap_label = (
            r"$\mathbf{CO_2}$ captured by CCS"
            if "captured" not in seen_component_labels
            else None
        )
        ax.bar(
            bar_x,
            captured_vals,
            width=bar_width,
            bottom=pos_bottom,
            facecolor="none",
            edgecolor="#2e7d32",
            linewidth=1.2,
            linestyle="--",
            zorder=2,
            label=cap_label,
        )
        seen_component_labels.add("captured")
        ghost_top = pos_bottom + captured_vals
        max_top = max(max_top, ghost_top.max() if len(ghost_top) else 0.0)
        min_bot = min(min_bot, neg_bottom.min() if len(neg_bottom) else 0.0)

        # Per-bar labels: net MtCO2e/yr in bold, per-tCO2 injected rate below.
        # We annotate per-bar because grouped bars have different captured
        # totals — a single secondary y-axis conversion can't represent both.
        # Use ground-truth values from scenarios_df (filtered by is_methane_based)
        # so the labels match the state-plot's semantics and include DAC.
        short_tag = r"CH$_4$" if flag else r"non-CH$_4$"
        _truth_flag = scenarios_df[
            (scenarios_df["methane_scenario"] == methane_scenario)
            & (scenarios_df["is_methane_based"] == flag)
        ]
        for j in range(n_scen):
            scen_name = destination_scenarios[j]["name"]
            _tf = _truth_flag[_truth_flag["scenario"] == scen_name]
            net_mt = _tf["mtco2e_per_yr_emissions_including_ccs"].sum()
            captured = _tf["mtco2_per_yr_captured_with_ccs"].sum()
            per_tco2 = (net_mt / captured) if captured else 0.0
            # Bold net Mt/yr
            ax.text(
                bar_x[j],
                pos_bottom[j] + 3,
                f"{net_mt:.0f}",
                ha="center",
                va="bottom",
                fontsize=fs["bar_label"] - 2,
                fontfamily="Arial",
                fontweight="bold",
                zorder=5,
                path_effects=[pe.withStroke(linewidth=4, foreground="white")],
            )
            # Per-tCO2 injected rate directly under the Mt label (only if this
            # bar actually captured something — nameplate_implied for a
            # methane-flag group with no projects gives captured=0).
            if captured:
                ax.text(
                    bar_x[j],
                    pos_bottom[j] + 3 + (max_top - min_bot) * 0.045,
                    f"({per_tco2:.1f}" + r" tCO$_2$e/tCO$_2$)",
                    ha="center",
                    va="bottom",
                    fontsize=fs["bar_label"] - 3,
                    fontfamily="Arial Narrow",
                    fontweight="normal",
                    color="#333333",
                    zorder=5,
                    path_effects=[pe.withStroke(linewidth=4, foreground="white")],
                )
            # Flag tag under the bar
            ax.text(
                bar_x[j],
                min(0, neg_bottom[j]) - (max_top - min_bot) * 0.02,
                short_tag,
                ha="center",
                va="top",
                fontsize=fs["tick"] - 2,
                fontfamily="Arial",
                color="#555555",
                zorder=5,
            )

    ax.axhline(0, color="black", linewidth=1.0, zorder=3)

    # Unabated line summed across all included projects (both flags, all states).
    unabated_df = scenarios_df[scenarios_df["methane_scenario"] == methane_scenario]
    unabated_df = unabated_df.drop_duplicates(subset="uid")
    baseload_val = unabated_df["unabated_mtco2e_per_yr"].sum()
    ax.axhline(
        baseload_val,
        color="black",
        linewidth=1.5,
        linestyle="--",
        zorder=3,
        label="Total unabated emissions",
    )
    max_top = max(max_top, baseload_val)

    ax.set_xticks(x)
    wrapped_labels = [
        "\n".join(textwrap.fill(line, width=20) for line in s["label"].split("\n"))
        for s in destination_scenarios
    ]
    ax.set_xticklabels(wrapped_labels, fontsize=fs["tick"], fontfamily="Arial")
    ax.tick_params(axis="x", pad=18)

    ax.set_ylabel(
        r"MtCO$_2$e / yr",
        fontsize=fs["axis_label"],
        fontfamily="Arial",
        labelpad=1,
    )
    ax.set_ylim(min_bot - 1 if min_bot < 0 else -1, max_top * 1.20)
    ax.tick_params(axis="y", labelsize=fs["tick"])

    # No secondary y-axis on this variant — the CH4 and non-CH4 groups have
    # different captured totals, so no single MtCO2e/yr → tCO2e/tCO2-injected
    # conversion would be right for both. Per-bar tCO2e/tCO2 annotations above
    # each bar convey the same information without the ambiguity.

    if title is None:
        if states is not None:
            state_label = _format_states_for_title(states)
        elif "project_state" in scenarios_df.columns:
            state_label = _format_states_for_title(
                sorted(scenarios_df["project_state"].dropna().unique().tolist())
            )
        else:
            state_label = "multi-state"
        title = f"{state_label} CCS: methane-based vs. non-methane-based industries (MtCO$_2$e/yr)"
    ax.set_title(title, fontsize=fs["title"], fontfamily="Arial", pad=20)

    ax.grid(axis="y", color="lightgray", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False)

    # Flag legend (methane vs non-methane) as swatches with hatches.
    from matplotlib.patches import Patch

    flag_legend_handles = [
        Patch(
            facecolor="#cccccc",
            edgecolor="#333333",
            hatch=flag_hatches.get(f, ""),
            label=flag_labels.get(f, str(f)),
        )
        for f in flag_values
    ]

    handles, labels_leg = ax.get_legend_handles_labels()
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels_leg):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)
    unique_handles.extend(flag_legend_handles)
    unique_labels.extend([h.get_label() for h in flag_legend_handles])

    if legend_below:
        ax.legend(
            unique_handles,
            unique_labels,
            fontsize=fs["legend"],
            frameon=True,
            facecolor="white",
            edgecolor="none",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=3,
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
            bbox_to_anchor=(1.02, 1),
        )
        fig.tight_layout(rect=[0, 0, 0.72, 1])
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
    return fig, ax


def plot_state_total_emissions_breakdown(
    industrial_emissions_df: pd.DataFrame,
    ccs_emissions_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    methane_scenario: str,
    destination_scenarios: Optional[List[Union[str, dict]]] = None,
    industries: Optional[List[str]] = None,
    which_flavor: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    display_mode: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Stacked bar chart of state-total emissions (MtCO2e/yr) by destination scenario."""
    display_mode = _resolve_display_mode(display_mode)
    fs = _resolve_fonts(display_mode)

    known_dest_scenarios = {
        "all_gs": {
            "name": "all_gs",
            "label": "100% Geologic Storage",
            "eor_pct": 0,
        },
        "baseline": {
            "name": "baseline",
            "label": "Announced destinations",
            "from_data": True,
        },
        "current_mix": {
            "name": "current_mix",
            "label": "Current U.S. practice",
            "eor_pct": 92,
        },
        "nameplate_implied": {
            "name": "nameplate_implied",
            "label": "Nameplate 'capture capacity'",
            "zero_lifecycle": True,
        },
        "all_eor": {
            "name": "all_eor",
            "label": "100% EOR",
            "eor_pct": 100,
        },
        "switching": {
            "name": "switching",
            "label": "Stated →\n100% EOR after 45Q",
            "from_data": True,
            "is_switching": True,
        },
        "switching_from_current": {
            "name": "switching_from_current",
            "label": "Current →\n100% EOR after 45Q",
            "from_data": True,
            "is_switching": True,
        },
    }

    if destination_scenarios is None:
        destination_scenarios = list(known_dest_scenarios.values())
    else:
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
    component_cols = [col for col, _, _ in _BREAKDOWN_COMPONENTS]
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

    result_df = (
        pd.DataFrame(records)
        .sort_values("_net", ascending=False)
        .reset_index(drop=True)
    )

    x = np.arange(len(result_df))
    width = 0.55

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    pos_bottom = np.zeros(len(x))
    neg_bottom = np.zeros(len(x))

    label_bar_idx = 0
    segments = []

    for col, label, color in _BREAKDOWN_COMPONENTS:
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
                zorder=2,
            )
            segments.append(
                (label, color, pos_bottom[label_bar_idx], pos_vals[label_bar_idx])
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
                zorder=2,
            )
            if not pos_vals.any():
                segments.append(
                    (label, color, neg_bottom[label_bar_idx], neg_vals[label_bar_idx])
                )
            neg_bottom += neg_vals

    ccs_top = pos_bottom.copy()

    ghost_vals = result_df["_captured"].values
    ax.bar(
        x,
        ghost_vals,
        width=width,
        bottom=pos_bottom,
        facecolor="none",
        edgecolor="#1b5e20",
        linewidth=3.5,
        linestyle="--",
        zorder=2,
    )
    segments.append(
        (
            r"$\mathbf{CO_2}$ captured by CCS",
            "#1b5e20",
            pos_bottom[label_bar_idx],
            ghost_vals[label_bar_idx],
        )
    )
    pos_bottom += ghost_vals

    ax.plot(
        [0, len(x) - 1],
        [0, 0],
        color="black",
        linewidth=1.0,
        zorder=3,
        clip_on=False,
    )

    total_bar_height = ccs_top[label_bar_idx] - neg_bottom[label_bar_idx]
    label_fontsize = fs["legend"] + 2
    label_x = x[label_bar_idx]
    light_bar_labels = {
        r"Uncaptured combustion $\mathbf{CO_2}$",
        r"Combustion $\mathbf{CH_4}$",
    }

    for seg_label, seg_color, seg_bottom, seg_height in segments:
        if abs(seg_height) < 1e-6:
            continue
        seg_mid = seg_bottom + seg_height / 2
        if seg_label in light_bar_labels:
            text_color = "black"
        else:
            text_color = "white" if seg_height > 0 else "black"
        if abs(seg_height) > total_bar_height * 0.06:
            ax.text(
                label_x,
                seg_mid,
                seg_label,
                ha="center",
                va="center",
                fontsize=label_fontsize,
                fontfamily="Arial",
                fontweight="bold",
                color=text_color,
                zorder=5,
                path_effects=[pe.withStroke(linewidth=3, foreground=seg_color)],
            )
        else:
            ax.annotate(
                seg_label,
                xy=(label_x + width / 2, seg_mid),
                xytext=(label_x + width / 2 + 0.09, seg_mid),
                ha="left",
                va="center",
                fontsize=label_fontsize,
                fontfamily="Arial",
                fontweight="bold",
                color=seg_color,
                zorder=5,
                arrowprops={"arrowstyle": "-|>", "color": seg_color, "lw": 2.2},
            )

    net_val_0 = (
        result_df.iloc[label_bar_idx]["_net"]
        + result_df.iloc[label_bar_idx]["_captured"]
    )
    ax.text(
        label_x + width / 2 + 0.05,
        ccs_top[label_bar_idx],
        f"{net_val_0:.0f}" + r" MtCO$_2$e/yr",
        ha="left",
        va="top",
        fontsize=fs["bar_label"],
        fontfamily="Arial",
        fontweight="bold",
        color="black",
        zorder=5,
    )

    for j in range(1, len(x)):
        net_val = result_df.iloc[j]["_net"] + result_df.iloc[j]["_captured"]
        y_pos = ccs_top[j] + 0.5
        ax.text(
            x[j],
            y_pos,
            f"{net_val:.0f}",
            ha="center",
            va="bottom",
            fontsize=fs["bar_label"],
            fontfamily="Arial",
            fontweight="bold",
            zorder=5,
            path_effects=[pe.withStroke(linewidth=4, foreground="white")],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        result_df["label"].tolist(),
        fontsize=fs["tick"] + 2,
        fontfamily="Arial",
    )
    ax.set_ylabel(
        r"MtCO$_2$e / yr",
        fontsize=fs["axis_label"] + 2,
        fontfamily="Arial",
        labelpad=12,
    )

    y_min = neg_bottom.min()
    y_max = pos_bottom.max() * 1.12
    ax.set_ylim(y_min - 1 if y_min < 0 else -1, y_max)
    ax.tick_params(axis="y", labelsize=fs["tick"] + 1)
    for lbl in ax.get_yticklabels():
        lbl.set_path_effects([pe.withStroke(linewidth=4, foreground="white")])

    chart_title = title or r"Texas CCS: state-total emissions (MtCO$_2$e/yr)"
    ax.set_title(chart_title, fontsize=fs["title"], fontfamily="Arial", pad=20)

    ax.grid(axis="y", color="lightgray", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)

    return fig, ax


_LUMPED_COMPONENTS = [
    (["process_ch4", "combustion_ch4"], r"Industrial $\mathbf{CH_4}$", "#c2600c"),
    (
        ["uncaptured_process_co2", "uncaptured_combustion_co2"],
        r"Uncaptured industrial $\mathbf{CO_2}$",
        "#f4a460",
    ),
    (
        ["captured_process_co2", "captured_combustion_co2"],
        r"Captured $\mathbf{CO_2}$",
        "#1b5e20",
    ),
    (["ccs_co2"], r"CCS-caused $\mathbf{CO_2}$", "#ef5350"),
    (["ccs_ch4"], r"CCS-caused $\mathbf{CH_4}$", "#b71c1c"),
]


def plot_state_total_emissions_breakdown_lumped(
    industrial_emissions_df: pd.DataFrame,
    ccs_emissions_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    methane_scenario: str,
    destination_scenarios: Optional[List[Union[str, dict]]] = None,
    industries: Optional[List[str]] = None,
    which_flavor: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    display_mode: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Stacked bar chart of state-total emissions (MtCO2e/yr) with lumped industrial components."""
    display_mode = _resolve_display_mode(display_mode)
    fs = _resolve_fonts(display_mode)

    known_dest_scenarios = {
        "all_gs": {
            "name": "all_gs",
            "label": "100% Geologic Storage",
            "eor_pct": 0,
        },
        "baseline": {
            "name": "baseline",
            "label": "Announced destinations",
            "from_data": True,
        },
        "current_mix": {
            "name": "current_mix",
            "label": "Current U.S. practice",
            "eor_pct": 92,
        },
        "nameplate_implied": {
            "name": "nameplate_implied",
            "label": "Nameplate 'capture capacity'",
            "zero_lifecycle": True,
        },
        "all_eor": {
            "name": "all_eor",
            "label": "100% EOR",
            "eor_pct": 100,
        },
        "switching": {
            "name": "switching",
            "label": "Stated →\n100% EOR after 45Q",
            "from_data": True,
            "is_switching": True,
        },
        "switching_from_current": {
            "name": "switching_from_current",
            "label": "Current →\n100% EOR after 45Q",
            "from_data": True,
            "is_switching": True,
        },
    }

    if destination_scenarios is None:
        destination_scenarios = list(known_dest_scenarios.values())
    else:
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
    component_cols = [col for col, _, _ in _BREAKDOWN_COMPONENTS]
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

    result_df = (
        pd.DataFrame(records)
        .sort_values("_net", ascending=False)
        .reset_index(drop=True)
    )

    for src_cols, lbl, _ in _LUMPED_COMPONENTS:
        col_name = lbl
        result_df[col_name] = sum(result_df[c] for c in src_cols)

    x = np.arange(len(result_df))
    width = 0.45

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    pos_bottom = np.zeros(len(x))
    neg_bottom = np.zeros(len(x))

    label_bar_idx = 0
    segments = []
    ghost_bar_bottom = None

    for _, label, color in _LUMPED_COMPONENTS:
        vals = result_df[label].values
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
                zorder=2,
            )
            segments.append(
                (label, color, pos_bottom[label_bar_idx], pos_vals[label_bar_idx])
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
                zorder=2,
            )
            if not pos_vals.any():
                segments.append(
                    (label, color, neg_bottom[label_bar_idx], neg_vals[label_bar_idx])
                )
            neg_bottom += neg_vals
        if "Uncaptured" in label:
            ghost_bar_bottom = pos_bottom.copy()

    ccs_top = pos_bottom.copy()

    ghost_vals = result_df["_captured"].values
    ghost_green = "#14532d"
    ax.bar(
        x,
        ghost_vals,
        width=width,
        bottom=ghost_bar_bottom,
        facecolor="none",
        edgecolor=ghost_green,
        linewidth=3.5,
        linestyle="--",
        zorder=2,
    )
    ghost_bottom = ghost_bar_bottom[label_bar_idx]
    ghost_height = ghost_vals[label_bar_idx]
    ghost_bar_bottom += ghost_vals

    ax.plot(
        [0, len(x) - 1],
        [0, 0],
        color="black",
        linewidth=1.0,
        zorder=3,
        clip_on=False,
    )

    total_bar_height = ccs_top[label_bar_idx] - neg_bottom[label_bar_idx]
    label_fontsize = fs["legend"] + 4
    label_x = x[label_bar_idx]

    captured_seg = None
    for seg_label, seg_color, seg_bottom, seg_height in segments:
        if abs(seg_height) < 1e-6:
            continue
        if r"Captured" in seg_label:
            captured_seg = (seg_bottom, seg_height, seg_color)
        seg_mid = seg_bottom + seg_height / 2
        text_color = "white" if seg_height > 0 else "black"
        if "Uncaptured" in seg_label:
            text_color = "black"
        elif "Captured" in seg_label:
            text_color = "white"
        val_int = abs(round(seg_height))
        val_str = f"({val_int} Mt/yr)"
        if abs(seg_height) > total_bar_height * 0.06:
            full_label = f"{seg_label}\n{val_str}"
            ax.text(
                label_x,
                seg_mid,
                full_label,
                ha="center",
                va="center",
                fontsize=label_fontsize,
                fontfamily="Arial",
                fontweight="bold",
                color=text_color,
                zorder=5,
                path_effects=[pe.withStroke(linewidth=3, foreground=seg_color)],
            )
        else:
            full_label = f"{seg_label}\n{val_str}"
            ax.annotate(
                full_label,
                xy=(label_x + width / 2, seg_mid),
                xytext=(label_x + width / 2 + 0.09, seg_mid),
                ha="left",
                va="center",
                fontsize=label_fontsize,
                fontfamily="Arial",
                fontweight="bold",
                color=seg_color,
                zorder=5,
                arrowprops={"arrowstyle": "-|>", "color": seg_color, "lw": 2.2},
            )

    if captured_seg is not None and ghost_height > 1e-6:
        cap_bottom, cap_height, _ = captured_seg
        cap_mid = cap_bottom + cap_height / 2
        ghost_mid = ghost_bottom + ghost_height / 2
        bar_left = label_x - width / 2
        ax.annotate(
            "",
            xy=(bar_left, cap_mid),
            xytext=(bar_left, ghost_mid),
            arrowprops={
                "arrowstyle": "->,head_length=1.5,head_width=0.8",
                "connectionstyle": "arc3,rad=0.4",
                "color": ghost_green,
                "lw": 3.5,
                "linestyle": "--",
            },
            zorder=5,
        )

    scen_name_0 = result_df.iloc[label_bar_idx]["scenario"]
    net_val_0 = scenarios_df[
        (scenarios_df["methane_scenario"] == methane_scenario)
        & (scenarios_df["scenario"] == scen_name_0)
    ]["mtco2e_per_yr_emissions_including_ccs"].sum()
    bar_left = label_x - width / 2
    ax.annotate(
        "Net emissions:\n" + f"{net_val_0:.0f}" + r" MtCO$_2$e/yr",
        xy=(bar_left, ccs_top[label_bar_idx]),
        xytext=(bar_left - 0.09, ccs_top[label_bar_idx]),
        ha="right",
        va="center",
        fontsize=label_fontsize + 2,
        fontfamily="Arial",
        fontweight="bold",
        color="black",
        zorder=5,
        arrowprops={"arrowstyle": "-", "color": "black", "lw": 1.8},
    )

    ghost_top_y = ghost_bottom + ghost_height
    unabated_df = scenarios_df[
        scenarios_df["methane_scenario"] == methane_scenario
    ].drop_duplicates(subset="uid")
    baseload_val = unabated_df["unabated_mtco2e_per_yr"].sum()
    ax.annotate(
        r"'Baseload'"
        + "\n"
        + "emissions:\n"
        + f"{baseload_val:.0f}"
        + r" MtCO$_2$e/yr",
        xy=(label_x + width / 2, ghost_top_y),
        xytext=(label_x + width / 2 + 0.09, ghost_top_y),
        ha="left",
        va="top",
        fontsize=label_fontsize + 2,
        fontfamily="Arial",
        fontweight="bold",
        color=ghost_green,
        zorder=5,
        arrowprops={"arrowstyle": "-", "color": ghost_green, "lw": 1.8},
    )

    for j in range(1, len(x)):
        net_val = result_df.iloc[j]["_net"] + result_df.iloc[j]["_captured"]
        y_pos = ccs_top[j] + 0.5
        ax.text(
            x[j],
            y_pos,
            f"{net_val:.0f}",
            ha="center",
            va="bottom",
            fontsize=fs["bar_label"],
            fontfamily="Arial",
            fontweight="bold",
            zorder=5,
            path_effects=[pe.withStroke(linewidth=4, foreground="white")],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        result_df["label"].tolist(),
        fontsize=fs["tick"] + 2,
        fontfamily="Arial",
    )
    ax.set_ylabel(
        r"MtCO$_2$e / yr",
        fontsize=fs["axis_label"] + 2,
        fontfamily="Arial",
        labelpad=12,
    )

    y_min = neg_bottom.min()
    y_max = np.maximum(ccs_top, ghost_bar_bottom).max() * 1.12
    ax.set_ylim(y_min - 1 if y_min < 0 else -1, y_max)
    ax.set_xlim(-0.55, len(x) - 0.55)
    ax.tick_params(axis="y", labelsize=fs["tick"] + 1)
    for lbl in ax.get_yticklabels():
        lbl.set_path_effects([pe.withStroke(linewidth=5, foreground="white")])

    chart_title = title or r"Texas CCS: state-total emissions (MtCO$_2$e/yr)"
    ax.set_title(chart_title, fontsize=fs["title"], fontfamily="Arial", pad=20)

    ax.grid(axis="y", color="lightgray", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)

    return fig, ax


def plot_state_total_emissions_breakdown_lumped_comparison(
    industrial_emissions_df: pd.DataFrame,
    ccs_emissions_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    methane_scenario: str,
    destination_scenarios: List[Union[str, dict]],
    industries: Optional[List[str]] = None,
    which_flavor: Optional[List[str]] = None,
    title: Optional[str] = None,
    print_mode: bool = False,
    legend_below: bool = False,
    figsize: Tuple[int, int] = (14, 8),
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    display_mode: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Side-by-side stacked bars comparing two destination scenarios with lumped components."""
    display_mode = _resolve_display_mode(display_mode)
    fs = _resolve_fonts(display_mode)

    known_dest_scenarios = {
        "all_gs": {
            "name": "all_gs",
            "label": "100% Geologic Storage",
            "eor_pct": 0,
        },
        "baseline": {
            "name": "baseline",
            "label": "Announced destinations",
            "from_data": True,
        },
        "current_mix": {
            "name": "current_mix",
            "label": "Current U.S. practice",
            "eor_pct": 92,
        },
        "nameplate_implied": {
            "name": "nameplate_implied",
            "label": "Nameplate 'capture capacity'",
            "zero_lifecycle": True,
        },
        "all_eor": {
            "name": "all_eor",
            "label": "100% EOR",
            "eor_pct": 100,
        },
        "switching": {
            "name": "switching",
            "label": "Stated →\n100% EOR after 45Q",
            "from_data": True,
            "is_switching": True,
        },
        "switching_from_current": {
            "name": "switching_from_current",
            "label": "Current →\n100% EOR after 45Q",
            "from_data": True,
            "is_switching": True,
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
    component_cols = [col for col, _, _ in _BREAKDOWN_COMPONENTS]
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

    result_df = (
        pd.DataFrame(records)
        .sort_values("_net", ascending=False)
        .reset_index(drop=True)
    )

    if not print_mode:
        for src_cols, lbl, _ in _LUMPED_COMPONENTS:
            result_df[lbl] = sum(result_df[c] for c in src_cols)

    for i, row in result_df.iterrows():
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
            scen_df = scen_df[scen_df["product"].isin(common)]
            total_cap = scen_df["mtco2_per_yr_captured_with_ccs"].sum()
            if total_cap > 0:
                w_eor = (
                    scen_df["eor_fraction"] * scen_df["mtco2_per_yr_captured_with_ccs"]
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

    if print_mode:
        pos_bottom = np.zeros(len(x))
        neg_bottom = np.zeros(len(x))

        for col, label, color in _BREAKDOWN_COMPONENTS:
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

        ax.axhline(
            0,
            color="black",
            linewidth=1.0,
            zorder=3,
            xmin=-0.03,
            xmax=1.03,
            clip_on=False,
        )

        unabated_df = scenarios_df[
            scenarios_df["methane_scenario"] == methane_scenario
        ].drop_duplicates(subset="uid")
        baseload_val = unabated_df["unabated_mtco2e_per_yr"].sum()
        ax.axhline(
            baseload_val,
            color="black",
            linewidth=1.5,
            linestyle="--",
            zorder=3,
            label=f"Total unabated emissions ({baseload_val:.0f} Mt/yr)",
        )

        for j in range(len(x)):
            scen_name = result_df.iloc[j]["scenario"]
            net_val = scenarios_df[
                (scenarios_df["methane_scenario"] == methane_scenario)
                & (scenarios_df["scenario"] == scen_name)
            ]["mtco2e_per_yr_emissions_including_ccs"].sum()
            y_pos = pos_bottom[j] + 0.05
            ax.text(
                x[j],
                y_pos,
                f"{net_val:.0f}",
                ha="center",
                va="bottom",
                fontsize=fs["bar_label"],
                fontfamily="Arial",
                fontweight="bold",
                zorder=5,
                path_effects=[pe.withStroke(linewidth=4, foreground="white")],
            )

        ax.set_xticks(x)
        wrapped_labels = [
            "\n".join(textwrap.fill(line, width=14) for line in lbl.split("\n"))
            for lbl in result_df["label"].tolist()
        ]
        ax.set_xticklabels(
            wrapped_labels,
            fontsize=fs["tick"],
            fontfamily="Arial",
        )
        ax.set_ylabel(
            r"MtCO$_2$e / yr",
            fontsize=fs["axis_label"],
            fontfamily="Arial",
            labelpad=12,
        )

        y_min = neg_bottom.min()
        y_max = pos_bottom.max() * 1.08
        ax.set_ylim(y_min - 0.15, y_max)
        ax.tick_params(axis="y", labelsize=fs["tick"])

        chart_title = title or r"Texas CCS: state-total emissions (MtCO$_2$e/yr)"
        ax.set_title(chart_title, fontsize=fs["title"], fontfamily="Arial", pad=20)

        ax.grid(axis="y", color="lightgray", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(bottom=False)

        handles, labels_leg = ax.get_legend_handles_labels()
        seen = set()
        unique_handles, unique_labels = [], []
        for h, l in zip(handles, labels_leg):
            if l not in seen:
                seen.add(l)
                unique_handles.append(h)
                unique_labels.append(l)
        if legend_below:
            ax.legend(
                unique_handles,
                unique_labels,
                fontsize=fs["legend"],
                frameon=True,
                facecolor="white",
                edgecolor="none",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.35),
                ncol=3,
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
                bbox_to_anchor=(1.02, 1),
            )
            fig.tight_layout(rect=[0, 0, 0.72, 1])
        if output_path:
            fig.savefig(output_path, bbox_inches="tight", dpi=150)

        return fig, ax

    pos_bottom = np.zeros(len(x))
    neg_bottom = np.zeros(len(x))
    segments_per_bar = [[] for _ in range(len(x))]
    ghost_bar_bottom = None

    for _, label, color in _LUMPED_COMPONENTS:
        vals = result_df[label].values
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
                zorder=2,
            )
            for j in range(len(x)):
                segments_per_bar[j].append((label, color, pos_bottom[j], pos_vals[j]))
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
                zorder=2,
            )
            if not pos_vals.any():
                for j in range(len(x)):
                    segments_per_bar[j].append(
                        (label, color, neg_bottom[j], neg_vals[j])
                    )
            neg_bottom += neg_vals
        if "Uncaptured" in label:
            ghost_bar_bottom = pos_bottom.copy()

    ccs_top = pos_bottom.copy()

    ghost_green = "#14532d"
    ghost_vals = result_df["_captured"].values
    ax.bar(
        x,
        ghost_vals,
        width=width,
        bottom=ghost_bar_bottom,
        facecolor="none",
        edgecolor=ghost_green,
        linewidth=3.5,
        linestyle="--",
        zorder=2,
    )
    ghost_bar_bottom_saved = ghost_bar_bottom.copy()
    ghost_bar_bottom += ghost_vals

    ax.plot(
        [-0.55, len(x) - 0.5],
        [0, 0],
        color="black",
        linewidth=1.0,
        zorder=3,
        clip_on=False,
    )

    label_fontsize = fs["legend"] + 4
    leader_offset = 0.05

    unabated_df = scenarios_df[
        scenarios_df["methane_scenario"] == methane_scenario
    ].drop_duplicates(subset="uid")
    baseload_val = unabated_df["unabated_mtco2e_per_yr"].sum()

    for j in range(len(x)):
        bar_x = x[j]
        bar_left = bar_x - width / 2
        total_bar_height_j = ccs_top[j] - neg_bottom[j]
        ghost_bottom_j = ghost_bar_bottom_saved[j]
        ghost_height_j = ghost_vals[j]

        captured_seg_j = None
        for seg_label, seg_color, seg_bottom, seg_height in segments_per_bar[j]:
            if abs(seg_height) < 1e-6:
                continue
            if "Captured" in seg_label and "Uncaptured" not in seg_label:
                captured_seg_j = (seg_bottom, seg_height, seg_color)
            seg_mid = seg_bottom + seg_height / 2
            text_color = "white" if seg_height > 0 else "black"
            if "Uncaptured" in seg_label:
                text_color = "black"
            elif "Captured" in seg_label:
                text_color = "white"
            val_int = abs(round(seg_height))
            val_str = f"({val_int} Mt/yr)"
            if "Uncaptured" in seg_label:
                display_label = seg_label.replace(
                    "Uncaptured industrial", "Uncaptured\nindustrial"
                )
            elif "CCS-caused" in seg_label and "CH_4" in seg_label:
                display_label = seg_label.replace(r"CCS-caused ", "CCS-caused\n")
            else:
                display_label = seg_label
            if abs(seg_height) > total_bar_height_j * 0.06:
                full_label = f"{display_label}\n{val_str}"
                ax.text(
                    bar_x,
                    seg_mid,
                    full_label,
                    ha="center",
                    va="center",
                    fontsize=label_fontsize + 2,
                    fontfamily="Arial",
                    fontweight="bold",
                    color=text_color,
                    zorder=5,
                    path_effects=[pe.withStroke(linewidth=3, foreground=seg_color)],
                )
            else:
                full_label = f"{display_label} {val_str}"
                if "CCS-caused" in seg_label and "CH_4" in seg_label:
                    ghost_top_check = ghost_bottom_j + ghost_height_j
                    leader_va = "top" if ghost_top_check > ccs_top[j] else "bottom"
                else:
                    leader_va = "center"
                ax.annotate(
                    full_label,
                    xy=(bar_x + width / 2, seg_mid),
                    xytext=(bar_x + width / 2 + leader_offset, seg_mid),
                    ha="left",
                    va=leader_va,
                    fontsize=label_fontsize + 2,
                    fontfamily="Arial",
                    fontweight="bold",
                    color=seg_color,
                    zorder=5,
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": seg_color,
                        "lw": 2.2,
                    },
                )

        if captured_seg_j is not None and ghost_height_j > 1e-6:
            cap_bottom, cap_height, _ = captured_seg_j
            cap_mid = cap_bottom + cap_height / 2
            ghost_mid = ghost_bottom_j + ghost_height_j / 2
            ax.annotate(
                "",
                xy=(bar_left, cap_mid),
                xytext=(bar_left, ghost_mid),
                arrowprops={
                    "arrowstyle": "->,head_length=1.5,head_width=0.8",
                    "connectionstyle": "arc3,rad=0.4",
                    "color": ghost_green,
                    "lw": 3.5,
                    "linestyle": "--",
                },
                zorder=5,
            )

        ghost_top_y = ghost_bottom_j + ghost_height_j
        leaders_up = ghost_top_y > ccs_top[j]
        leader_y_sign = 20 if leaders_up else -20
        leader_va = "bottom" if leaders_up else "top"

        scen_name = result_df.iloc[j]["scenario"]
        net_val = scenarios_df[
            (scenarios_df["methane_scenario"] == methane_scenario)
            & (scenarios_df["scenario"] == scen_name)
        ]["mtco2e_per_yr_emissions_including_ccs"].sum()
        ax.annotate(
            f"{net_val:.0f}\nMt/yr",
            xy=(bar_left, ccs_top[j]),
            xytext=(-20, leader_y_sign),
            textcoords="offset points",
            ha="right",
            va=leader_va,
            fontsize=label_fontsize + 3,
            fontfamily="Arial",
            fontweight="bold",
            color="black",
            zorder=5,
            arrowprops={"arrowstyle": "-", "color": "black", "lw": 1.8},
        )

        ax.annotate(
            f"{baseload_val:.0f}\nMt/yr",
            xy=(bar_x + width / 2, ghost_top_y),
            xytext=(20, leader_y_sign),
            textcoords="offset points",
            ha="left",
            va=leader_va,
            fontsize=label_fontsize + 3,
            fontfamily="Arial",
            fontweight="bold",
            color=ghost_green,
            zorder=5,
            arrowprops={"arrowstyle": "-", "color": ghost_green, "lw": 1.8},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        result_df["label"].tolist(),
        fontsize=fs["tick"] + 2,
        fontfamily="Arial",
    )
    ax.set_ylabel(
        r"MtCO$_2$e / yr",
        fontsize=fs["axis_label"] + 4,
        fontfamily="Arial",
        labelpad=12,
    )
    ax.set_xlabel(
        "CCS Scenario",
        fontsize=fs["axis_label"] + 4,
        fontfamily="Arial",
        labelpad=8,
    )

    y_min = neg_bottom.min()
    y_max = np.maximum(ccs_top, ghost_bar_bottom).max() * 1.12
    ax.set_ylim(y_min - 1 if y_min < 0 else -1, y_max)
    ax.set_xlim(-0.55, len(x) - 0.55)
    ax.tick_params(axis="y", labelsize=fs["tick"] + 1)
    for lbl in ax.get_yticklabels():
        lbl.set_path_effects([pe.withStroke(linewidth=5, foreground="white")])

    chart_title = title or r"Texas CCS: state-total emissions (MtCO$_2$e/yr)"
    ax.set_title(chart_title, fontsize=fs["title"], fontfamily="Arial", pad=20)

    ax.grid(axis="y", color="lightgray", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)

    return fig, ax


def plot_total_emissions_breakdown(
    industrial_emissions_df: pd.DataFrame,
    ccs_emissions_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    scenario: str,
    methane_scenario: str,
    industries: Optional[List[str]] = None,
    which_flavor: Optional[List[str]] = None,
    name_replacements: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 8),
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    display_mode: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Stacked bar chart of total emissions (MtCO2e/yr) by industry for one scenario."""
    fs = _resolve_fonts(display_mode)

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
    component_cols = [col for col, _, _ in _BREAKDOWN_COMPONENTS]

    scen_df = scenarios_df[
        (scenarios_df["methane_scenario"] == methane_scenario)
        & (scenarios_df["scenario"] == scenario)
    ].copy()
    scen_df = scen_df[scen_df["product"].isin(common)]

    industry_totals = {}
    for ind in common:
        proj_rows = scen_df[scen_df["product"] == ind]
        if proj_rows.empty:
            continue
        row_data = {}
        for col in component_cols:
            weighted_sum = 0.0
            for _, proj in proj_rows.iterrows():
                ef = proj["eor_fraction"]
                factor = ef * eor_df.loc[ind, col] + (1 - ef) * gs_df.loc[ind, col]
                weighted_sum += factor * proj["mtco2_per_yr_captured_with_ccs"]
            row_data[col] = weighted_sum
        row_data["_net"] = sum(row_data[c] for c in component_cols)
        industry_totals[ind] = row_data

    if not industry_totals:
        raise ValueError(
            f"No data for scenario='{scenario}', methane_scenario='{methane_scenario}'"
        )

    result_df = pd.DataFrame.from_dict(industry_totals, orient="index")
    result_df = result_df.sort_values("_net", ascending=False)
    industry_order = result_df.index.tolist()

    x = np.arange(len(industry_order))
    width = 0.55

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    pos_bottom = np.zeros(len(x))
    neg_bottom = np.zeros(len(x))

    for col, label, color in _BREAKDOWN_COMPONENTS:
        vals = result_df.loc[industry_order, col].values
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

    ccs_top = pos_bottom.copy()

    captured_by_ind = scen_df.groupby("product")["mtco2_per_yr_captured_with_ccs"].sum()
    ghost_vals = captured_by_ind.reindex(industry_order).fillna(0).values
    ax.bar(
        x,
        ghost_vals,
        width=width,
        bottom=pos_bottom,
        facecolor="#c8e6c9",
        alpha=0.5,
        edgecolor="#1b5e20",
        linewidth=3.5,
        linestyle="--",
        hatch="///",
        label=r"$\mathbf{CO_2}$ captured by CCS",
        zorder=2,
    )
    pos_bottom += ghost_vals

    ax.axhline(
        0, color="black", linewidth=1.0, zorder=3, xmin=-0.03, xmax=1.03, clip_on=False
    )

    for j in range(len(x)):
        net_val = (
            result_df.iloc[j]["_net"]
            + captured_by_ind.reindex(industry_order).fillna(0).iloc[j]
        )
        y_pos = ccs_top[j] + 0.05
        ax.text(
            x[j],
            y_pos,
            f"{net_val:.1f}",
            ha="center",
            va="bottom",
            fontsize=fs["bar_label"] - 4,
            fontfamily="Arial",
            fontweight="bold",
            zorder=5,
        )

    replacements = {**_FACILITY_NAME_DEFAULTS, **(name_replacements or {})}
    labels = [
        textwrap.fill(replacements.get(ind, ind.replace("_", " ").title()), width=14)
        for ind in industry_order
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fs["tick"], fontfamily="Arial Narrow")
    ax.set_ylabel(
        r"MtCO$_2$e / yr",
        fontsize=fs["axis_label"],
        fontfamily="Arial",
        labelpad=12,
    )

    y_min = neg_bottom.min()
    y_max = pos_bottom.max() * 1.08
    ax.set_ylim(y_min - 0.3 if y_min < 0 else -0.1, y_max)
    ax.tick_params(axis="y", labelsize=fs["tick"])

    chart_title = title or (
        r"Total emissions by industry (MtCO$_2$e/yr)" f" — {scenario.replace('_', ' ')}"
    )
    ax.set_title(chart_title, fontsize=fs["title"], fontfamily="Arial", pad=20)

    ax.grid(axis="y", color="lightgray", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False)

    handles, labels_leg = ax.get_legend_handles_labels()
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels_leg):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)
    ax.legend(
        unique_handles,
        unique_labels,
        fontsize=fs["legend"],
        frameon=True,
        facecolor="white",
        edgecolor="none",
        loc="upper right",
    )

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)

    return fig, ax


def plot_unabated_by_industry(
    scenarios_df: pd.DataFrame,
    methane_scenario: str,
    name_replacements: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 10),
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    display_mode: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Single stacked bar showing each industry's contribution to total unabated emissions."""
    fs = _resolve_fonts(display_mode)
    df = scenarios_df[scenarios_df["methane_scenario"] == methane_scenario]
    unabated = (
        df[["uid", "product", "unabated_mtco2e_per_yr"]]
        .drop_duplicates(subset="uid")
        .groupby("product")["unabated_mtco2e_per_yr"]
        .sum()
        .sort_values(ascending=True)
    )
    zero_products = unabated[unabated == 0]
    unabated = unabated[unabated > 0]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    replacements = {**_FACILITY_NAME_DEFAULTS, **(name_replacements or {})}
    total = unabated.sum()
    bottom = 0.0
    bar_width = 1
    small_segments = []

    for product in zero_products.index:
        color = INDUSTRY_COLORS.get(product, "#888888")
        label = replacements.get(product, product.replace("_", " ").title())
        small_segments.append((0.0, label, 0.0, color))

    for _, (product, val) in enumerate(unabated.items()):
        color = INDUSTRY_COLORS.get(product, "#888888")
        label = replacements.get(product, product.replace("_", " ").title())
        ax.bar(
            0,
            val,
            bottom=bottom,
            width=bar_width,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
        )
        mid = bottom + val / 2
        if val > total * 0.05:
            ax.text(
                0,
                mid,
                f"{label}\n({val:.0f})",
                ha="center",
                va="center",
                fontsize=fs["axis_label"] + 2,
                fontfamily="Arial",
                fontweight="semibold",
                color="white",
                zorder=5,
            )
        else:
            small_segments.append((mid, label, val, color))
        bottom += val

    label_x = bar_width / 2 + 0.25
    if small_segments:
        n = len(small_segments)
        seg_top = small_segments[-1][0]
        seg_bottom = small_segments[0][0]
        min_gap = total * 0.09
        needed_span = min_gap * (n - 1)
        seg_center = (seg_top + seg_bottom) / 2
        fan_half = max(needed_span, seg_top - seg_bottom) / 2
        fan_bottom = max(total * 0.005, seg_center - fan_half)
        fan_top = fan_bottom + 2 * fan_half
        text_ys = [
            fan_bottom + i * (fan_top - fan_bottom) / max(n - 1, 1) for i in range(n)
        ]
        for (mid_y, label, val, color), text_y in zip(small_segments, text_ys):
            ax.annotate(
                f"{label} ({val:.1f})",
                xy=(bar_width / 2, mid_y),
                xytext=(label_x, text_y),
                ha="left",
                va="center",
                fontsize=fs["axis_label"] + 2,
                fontfamily="Arial",
                fontweight="semibold",
                color=color,
                arrowprops={
                    "arrowstyle": "-",
                    "color": color,
                    "lw": 1.2,
                    "relpos": (0, 0.5),
                },
                zorder=5,
            )

    ax.text(
        0,
        bottom + total * 0.01,
        f"{total:.0f} MtCO2e/yr",
        ha="center",
        va="bottom",
        fontsize=fs["axis_label"] + 2,
        fontfamily="Arial",
        fontweight="bold",
        zorder=5,
    )

    ax.set_xlim(-0.6, 1.8)
    ax.set_xticks([])
    ax.set_ylabel(
        "MtCO2e/yr", fontsize=fs["axis_label"] + 1, fontfamily="Arial", labelpad=12
    )
    ax.tick_params(axis="y", labelsize=fs["tick"])

    if title is not None:
        ax.set_title(title, fontsize=fs["title"], fontfamily="Arial", pad=20)

    ax.grid(axis="y", color="lightgray", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)

    return fig, ax


def _hex_to_rgba(hex_color: str, alpha: float = 0.4) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _darken_hex(hex_color: str, factor: float = 0.55) -> str:
    """Return a darker shade of `hex_color`. `factor` in (0,1); smaller = darker."""
    hc = hex_color.lstrip("#")
    r, g, b = (int(hc[i : i + 2], 16) for i in (0, 2, 4))
    return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"


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
    storage_idx = n_products
    atmos_idx = n_products + 1

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
    atmos_idx = n_products
    storage_idx = n_products + 1

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
            targets.append(storage_idx)
            values.append(float(cap))
            link_colors.append(_hex_to_rgba(color_map.get(p, "#888888"), 0.55))
        if atm > 0:
            sources.append(i)
            targets.append(atmos_idx)
            values.append(float(atm))
            link_colors.append(_hex_to_rgba(color_map.get(p, "#888888"), 0.35))

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="fixed",
                node={
                    "label": node_labels,
                    "color": node_colors,
                    "pad": 18,
                    "thickness": 18,
                    "line": {"color": "#333333", "width": 0.5},
                    "x": node_x,
                    "y": node_y,
                },
                link={
                    "source": sources,
                    "target": targets,
                    "value": values,
                    "color": link_colors,
                },
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

    layout_kwargs = {
        "width": width,
        "height": height,
        "font": {"size": fs["annotation"], "family": "Arial"},
        "margin": {"l": 220, "r": 220, "t": 60 if title else 20, "b": bottom_margin},
    }
    if title is not None:
        layout_kwargs["title"] = {"text": title, "x": 0.5, "xanchor": "center"}
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
            font={"size": annotation_font_size, "family": "Arial", "color": "#222222"},
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
            font={"size": annotation_font_size, "family": "Arial", "color": "#222222"},
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
                font={"size": annotation_font_size, "family": "Arial"},
            )

    if output_path:
        fig.write_image(output_path, scale=2)
    return fig


def plot_process_and_ccs_sources_sankey(
    df: pd.DataFrame,
    product_col: str = "product",
    unabated_col: str = "unabated_mtco2e_per_yr",
    ccs_emitted_col: str = "mtco2e_per_yr_emitted_because_of_ccs",
    captured_col: str = "mtco2_per_yr_captured_with_ccs",
    net_emissions_col: str = "mtco2e_per_yr_emissions_including_ccs",
    name_replacements: Optional[Dict[str, str]] = None,
    colors: Optional[Dict[str, str]] = None,
    product_group_map: Optional[Dict[str, str]] = None,
    top_n: int = 5,
    ccs_darken_factor: float = 0.35,
    other_color: str = "#a8a8a8",
    storage_color: str = "#3a3a3a",
    atmosphere_color: str = "#ffffff",
    title: Optional[str] = None,
    show_values: bool = True,
    node_pad_px: int = 14,
    width: int = 1200,
    height: int = 950,
    output_path: Optional[str] = None,
    display_mode: Optional[str] = None,
):
    """Three-column Sankey: (process + CCS) sources → products → destinations.

    Companion to `plot_products_to_destination_sankey`. For every product the
    middle-column node has two input streams on the far left:
      * unabated industrial-process emissions (`unabated_col`), colored per
        product, and
      * emissions caused by CCS operations themselves
        (`ccs_emitted_col`), all colored `ccs_source_color`.
    Their sum equals the middle node's thickness. The middle node then splits
    to Atmosphere (`net_emissions_col`) and Storage (`captured_col`), exactly
    as in the sibling plot's right column.

    The `top_n` largest emitters (by unabated + CCS total) get their own row.
    Everything else is aggregated into a single "Other products" row; the
    constituent products are listed in the legend below the plot.

    Args:
        df: Wide-format DataFrame with one row per product and columns named
            by `product_col`, `unabated_col`, `ccs_emitted_col`,
            `captured_col`, and `net_emissions_col`.
        product_col: Column holding the product / industry name.
        unabated_col: Column with unabated industrial-process emissions.
        ccs_emitted_col: Column with emissions caused by CCS itself
            (e.g., added methane leaks, extra combustion for capture energy).
        captured_col: Column with CO2 tonnes routed to storage.
        net_emissions_col: Column with tCO2e tonnes routed to atmosphere.
        top_n: Number of top-emitting products (by unabated + CCS total) to
            display as individual rows. All remaining products are aggregated
            into a single "Other products" row and listed in the legend.
        ccs_darken_factor: Multiplicative factor applied to each product's
            hex color to derive the CCS-source shade (0<factor<1; smaller =
            darker). Both source nodes for a product then share the same
            color family — the darker shade marks CCS-caused emissions.
        other_color: Color of the aggregated "Other products" middle node.
        node_pad_px: Pixel gap enforced between adjacent nodes in each column.
            Larger = more visible separation between all nodes (including
            each product's two source nodes on the left).

    Returns:
        Plotly Figure.
    """
    import plotly.graph_objects as go

    fs = _resolve_fonts(display_mode)
    _is_presentation = _resolve_display_mode(display_mode) == "presentation"
    color_map = {**INDUSTRY_COLORS, **(colors or {})}
    # Local override: this plot uses the shortened "Data centers" label.
    # Any user-supplied `name_replacements` still wins (merged last).
    names = {
        **_FACILITY_NAME_DEFAULTS,
        "power_ng_data_center": "Data centers",
        "ngp": "Nat. Gas Process.",
        **(name_replacements or {}),
    }
    # Font family: narrower than Arial for a denser, more compact look.
    _font_family = "Arial Narrow"

    # Default is NO grouping in this plot so coal/gas/refining/petrochemicals
    # stay individual. Callers can pass a group map to combine other products.
    if product_group_map is None:
        product_group_map = {}
    if product_group_map:
        df = df.copy()
        df[product_col] = df[product_col].replace(product_group_map)
        first_pos = {v: i for i, v in enumerate(df[product_col].tolist())}
        df = df.groupby(product_col, sort=False).sum(numeric_only=True).reset_index()
        df = df.sort_values(by=product_col, key=lambda s: s.map(first_pos)).reset_index(
            drop=True
        )

    # Keep the top_n biggest emitters as individual rows; aggregate the rest
    # into "Other products". Ranking is on total (unabated + CCS).
    row_totals = df[unabated_col].fillna(0) + df[ccs_emitted_col].fillna(0)
    top_idx = row_totals.sort_values(ascending=False).head(top_n).index
    keep_mask = df.index.isin(top_idx)
    big_df = df[keep_mask].copy()
    small_df = df[~keep_mask].copy()
    other_products_list = small_df[product_col].tolist()

    if not small_df.empty:
        other_row = {
            product_col: "other_products",
            unabated_col: small_df[unabated_col].sum(),
            ccs_emitted_col: small_df[ccs_emitted_col].sum(),
            captured_col: small_df[captured_col].sum(),
            net_emissions_col: small_df[net_emissions_col].sum(),
        }
        big_df = pd.concat([big_df, pd.DataFrame([other_row])], ignore_index=True)
        color_map["other_products"] = other_color
        names["other_products"] = "Other products"

    df = big_df
    products = df[product_col].tolist()
    unabated_by_product = df.set_index(product_col)[unabated_col].to_dict()
    ccs_source_by_product = df.set_index(product_col)[ccs_emitted_col].to_dict()
    captured_by_product = df.set_index(product_col)[captured_col].to_dict()
    net_by_product = df.set_index(product_col)[net_emissions_col].to_dict()

    n = len(products)
    total_atmos = sum(net_by_product[p] for p in products)
    total_storage = sum(captured_by_product[p] for p in products)
    product_totals = [
        unabated_by_product[p] + ccs_source_by_product[p] for p in products
    ]

    def _label(name: str, value: float, pretty_override: Optional[str] = None) -> str:
        pretty = pretty_override or names.get(name, name.replace("_", " ").title())
        if show_values:
            return f"{pretty}<br>{value:.0f} Mt/yr"
        return pretty

    product_colors_list = [color_map.get(p, "#888888") for p in products]
    ccs_colors_list = [_darken_hex(c, ccs_darken_factor) for c in product_colors_list]

    # ── Left-column layout with SPACER nodes between adjacent product pairs ──
    # `node.pad` is a single global value in plotly, so within-pair vs
    # between-pair gaps can't be set independently. Workaround: insert an
    # invisible "spacer" node between each pair; those spacers each take up
    # `node.pad + spacer_h + node.pad ≈ 2*node.pad + spacer_h` of vertical
    # space, giving between-pair gaps that are visibly larger than the
    # within-pair `node.pad`. Spacers route to a hidden sink at the right so
    # plotly accepts them as valid nodes.
    n_spacers = max(n - 1, 0)
    _has_sink = False  # legacy — sink no longer used; spacers route into atmos
    # Spacer flow value tuned so each spacer creates a small extra band on
    # the left column (roughly 2–4 px tall). Kept small so the transparent
    # routing into atmosphere only inflates the atmos node by ~1%.
    spacer_flow = max(sum(product_totals) * 0.003, 0.5)

    # Left-column node metadata, in top-to-bottom visual order.
    # Each entry: (kind, product_index_or_None, flow, node_color, line_color)
    line_visible = "#333333"
    line_invisible = "rgba(0,0,0,0)"
    color_invisible = "rgba(255,255,255,0)"

    left_entries: List[Tuple[str, Optional[int], float, str, str]] = []
    for i in range(n):
        left_entries.append(
            (
                "unabated",
                i,
                unabated_by_product[products[i]],
                product_colors_list[i],
                line_visible,
            )
        )
        left_entries.append(
            (
                "ccs",
                i,
                ccs_source_by_product[products[i]],
                ccs_colors_list[i],
                line_visible,
            )
        )
        if i < n - 1:
            left_entries.append(
                (
                    "spacer",
                    i,
                    spacer_flow,
                    color_invisible,
                    line_invisible,
                )
            )

    # Fast lookup from (kind, i) → node index in the left column.
    left_key_to_idx = {(e[0], e[1]): idx for idx, e in enumerate(left_entries)}

    left_len = len(left_entries)
    middle_start = left_len
    atmos_idx = middle_start + n
    storage_idx = atmos_idx + 1

    node_labels = [""] * (left_len + n + 2)
    node_colors: List[str] = (
        [e[3] for e in left_entries]
        + product_colors_list
        + [atmosphere_color, storage_color]
    )
    node_line_colors: List[str] = (
        [e[4] for e in left_entries] + [line_visible] * n + [line_visible, line_visible]
    )

    # Compute y-positions with a plotly-consistent formula so annotations line
    # up with the rendered nodes. Also return top/bottom paper coords so we
    # can compute pair centers for label placement.
    def _stack_positions(
        flows: List[float], total_flow: float
    ) -> Tuple[List[float], List[float], List[float]]:
        n_ = len(flows)
        total_pad_px = max(n_ - 1, 0) * node_pad_px
        usable_px = max(height - total_pad_px, 1)
        cursor_px = 0.0
        centers, tops, bottoms = [], [], []
        for f in flows:
            h_px = (f / total_flow) * usable_px
            tops.append(cursor_px / height)
            centers.append((cursor_px + h_px / 2) / height)
            bottoms.append((cursor_px + h_px) / height)
            cursor_px += h_px + node_pad_px
        return centers, tops, bottoms

    left_flows = [e[2] for e in left_entries]
    total_left_flow = float(sum(left_flows)) or 1.0
    left_centers, left_tops, left_bottoms = _stack_positions(
        left_flows, total_left_flow
    )

    total_middle_flow = float(sum(product_totals)) or 1.0
    product_y, _, _ = _stack_positions(product_totals, total_middle_flow)

    # Right column: atmos receives spacer flow (via transparent links) so
    # there's no separate sink node. Reflect that in the flow used to size
    # the atmos node so annotations line up with the rendered node.
    atmos_flow_rendered = total_atmos + spacer_flow * n_spacers
    right_flows = [atmos_flow_rendered, total_storage]
    total_right_flow = float(sum(right_flows)) or 1.0
    right_centers, _, _ = _stack_positions(right_flows, total_right_flow)
    dest_y = right_centers

    node_x = [0.001] * left_len + [0.5] * n + [0.999, 0.999]
    node_y = left_centers + product_y + right_centers

    # Pair centers (visible top of unabated → visible bottom of CCS), in
    # annotation paper y (1 = top of plot). Labels are anchored here.
    pair_label_ys: List[float] = []
    for i in range(n):
        u_idx = left_key_to_idx[("unabated", i)]
        c_idx = left_key_to_idx[("ccs", i)]
        pair_top = left_tops[u_idx]
        pair_bottom = left_bottoms[c_idx]
        pair_center_plotly = (pair_top + pair_bottom) / 2
        pair_label_ys.append(1.0 - pair_center_plotly)

    sources: List[int] = []
    targets: List[int] = []
    values: List[float] = []
    link_colors: List[str] = []

    # Column 1 → Column 2 (sources → product middle). Both source links use
    # the same product color family: unabated at the product's base color,
    # CCS at the darker shade — matching their source nodes' colors.
    for i, p in enumerate(products):
        middle_idx = middle_start + i
        u = unabated_by_product[p]
        c = ccs_source_by_product[p]
        base_color = product_colors_list[i]
        ccs_color_i = ccs_colors_list[i]
        if u > 0:
            sources.append(left_key_to_idx[("unabated", i)])
            targets.append(middle_idx)
            values.append(float(u))
            link_colors.append(_hex_to_rgba(base_color, 0.55))
        if c > 0:
            sources.append(left_key_to_idx[("ccs", i)])
            targets.append(middle_idx)
            values.append(float(c))
            link_colors.append(_hex_to_rgba(ccs_color_i, 0.55))

    # Column 2 → Column 3 (product middle → destinations)
    for i, p in enumerate(products):
        middle_idx = middle_start + i
        cap = captured_by_product[p]
        atm = net_by_product[p]
        if cap > 0:
            sources.append(middle_idx)
            targets.append(storage_idx)
            values.append(float(cap))
            link_colors.append(_hex_to_rgba(product_colors_list[i], 0.55))
        if atm > 0:
            sources.append(middle_idx)
            targets.append(atmos_idx)
            values.append(float(atm))
            link_colors.append(_hex_to_rgba(product_colors_list[i], 0.35))

    # Spacer → atmosphere with a fully-transparent link. Routes the spacer
    # flow through the diagram without adding a visible sink node; atmos
    # renders ~1% taller than its true value but the label still shows the
    # true total_atmos.
    for i in range(n_spacers):
        sources.append(left_key_to_idx[("spacer", i)])
        targets.append(atmos_idx)
        values.append(spacer_flow)
        link_colors.append(color_invisible)

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="fixed",
                node={
                    "label": node_labels,
                    "color": node_colors,
                    "pad": node_pad_px,
                    "thickness": 18,
                    "line": {"color": node_line_colors, "width": 0.5},
                    "x": node_x,
                    "y": node_y,
                },
                link={
                    "source": sources,
                    "target": targets,
                    "value": values,
                    "color": link_colors,
                },
            )
        ]
    )

    # After aggregation every remaining row gets an inline label. Legend below
    # only carries: (a) the CCS color key, (b) what's inside "Other products".
    n_other_rows = (len(other_products_list) + 1) // 2 if other_products_list else 0
    # 1 row for CCS key + 1 row for "Other products includes:" header +
    # n_other_rows for the list.
    n_legend_rows = 1 + (1 + n_other_rows if other_products_list else 0)
    bottom_margin = 20 + 24 * n_legend_rows
    # Extra top margin so the column headers ("Source of emissions" /
    # "Total emissions" / "Emissions destination") have room above the plot.
    top_margin = (60 if title else 20) + 40

    layout_kwargs = {
        "width": width,
        "height": height,
        "font": {"size": fs["annotation"], "family": _font_family},
        "margin": {"l": 280, "r": 260, "t": top_margin, "b": bottom_margin},
    }
    if title is not None:
        layout_kwargs["title"] = {"text": title, "x": 0.5, "xanchor": "center"}
    fig.update_layout(**layout_kwargs)

    # Font sizes for the left-margin product labels and right-margin
    # destination labels. Bumped up a few points from `fs["annotation"]` since
    # they carry the primary product/destination identification. Presentation
    # mode adds a further bump so labels read from the back of the room.
    _pres_bump = 5 if _is_presentation else 0
    node_label_font_size = fs["annotation"] + 3 + _pres_bump
    stream_annotation_font_size = fs["annotation"] + 1 + _pres_bump
    legend_font_size = fs["annotation"]  # bottom legend / "Other products" list
    column_header_font_size = fs["annotation"] + (8 if _is_presentation else 4)

    # Column headers sit in the top margin, centered on each column's x.
    # The center column intentionally has no header — per-industry labels
    # already identify what each middle node represents, and dropping the
    # header keeps vertical space available for the (now larger) left and
    # right headers in presentation mode.
    for col_x, header_text in (
        (0.0, "Emissions source"),
        (1.0, "Emissions destination"),
    ):
        fig.add_annotation(
            x=col_x,
            y=1.04,
            xref="paper",
            yref="paper",
            text=f"<b>{header_text}</b>",
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            align="center",
            font={
                "size": column_header_font_size,
                "family": _font_family,
                "color": "#222222",
            },
        )
    label_offset_paper = 0.03

    # LEFT margin: one label per product pair, centered vertically between the
    # top of the pair's unabated node and the bottom of its CCS node.
    # Small consecutive nodes may crowd their labels together, so enforce a
    # minimum vertical spacing between adjacent labels; when a label would
    # overlap the one above it, push it down and draw a thin leader line back
    # to the pair-center so the mapping stays clear.
    min_label_spacing = 48 / height  # ~2 text lines at the bumped font size
    annotation_ys: List[float] = []  # paper y (1=top)
    for i in range(n):
        desired = pair_label_ys[i]
        if annotation_ys:
            prev = annotation_ys[-1]
            if prev - desired < min_label_spacing:
                desired = prev - min_label_spacing
        annotation_ys.append(desired)

    for i, p in enumerate(products):
        pretty = names.get(p, p.replace("_", " ").title())
        node_y_ann = pair_label_ys[i]
        label_y_ann = annotation_ys[i]
        was_shifted = abs(label_y_ann - node_y_ann) > 1e-4
        # Industry name in bold with value in parentheses on the same line.
        if show_values:
            left_label_text = f"<b>{pretty}</b> ({product_totals[i]:.0f} Mt/yr)"
        else:
            left_label_text = f"<b>{pretty}</b>"
        fig.add_annotation(
            x=0 - label_offset_paper,
            y=label_y_ann,
            xref="paper",
            yref="paper",
            text=left_label_text,
            showarrow=False,
            xanchor="right",
            yanchor="middle",
            align="right",
            font={
                "size": node_label_font_size,
                "family": _font_family,
                "color": "#222222",
            },
        )
        if was_shifted:
            fig.add_shape(
                type="line",
                xref="paper",
                yref="paper",
                x0=0 - label_offset_paper + 0.003,
                y0=label_y_ann,
                x1=0.0,
                y1=node_y_ann,
                line={"color": "#888888", "width": 0.6},
            )

    dest_labels = [
        _label(
            "atmosphere",
            total_atmos,
            pretty_override="<b>Emitted to the<br>atmosphere</b>",
        ),
        _label(
            "storage",
            total_storage,
            pretty_override="<b>Injected<br>underground</b><br>(CCS)",
        ),
    ]
    for j, label in enumerate(dest_labels):
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
            font={
                "size": node_label_font_size,
                "family": _font_family,
                "color": "#222222",
            },
        )

    # Per-stream annotations on the Ammonia bar (if Ammonia is one of the
    # displayed products): label the two source flows so a reader can identify
    # what each shade in the left column represents without hunting through
    # the bottom legend. In presentation mode only the TOP-MOST stream is
    # labeled (less visual noise on-screen); report mode labels all three.
    _stream_candidates = ["ammonia", "lng", "hydrogen"]
    if _is_presentation:
        _topmost = next((p for p in products if p in _stream_candidates), None)
        _stream_candidates = [_topmost] if _topmost else []
    for indust in _stream_candidates:
        if indust in products:
            amm_i = products.index(indust)
            u_idx = left_key_to_idx[("unabated", amm_i)]
            c_idx = left_key_to_idx[("ccs", amm_i)]
            # Both labels centered on their respective flows, semibold black
            # for consistency and readability.
            stream_label_x = 0.03
            # Presentation mode: labels one point bigger than report mode;
            # report mode keeps the compact -1 shave.
            _stream_label_size = stream_annotation_font_size + (
                0 if _is_presentation else -1
            )
            # Extra downward shift for the CCS stream label so it clears the
            # (now-darker) CCS flow and reads as its own line beneath the
            # "From unabated industry" label.
            _ccs_label_shift = 0.04
            fig.add_annotation(
                x=stream_label_x,
                y=1.0 - left_centers[u_idx],
                xref="paper",
                yref="paper",
                text="<b>From unabated industry</b>",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                align="left",
                font={
                    "size": _stream_label_size,
                    "family": _font_family,
                    "color": "#222222",
                },
                bgcolor="rgba(255,255,255,0)",
            )
            fig.add_annotation(
                x=stream_label_x,
                y=1.0 - left_centers[c_idx] - _ccs_label_shift,
                xref="paper",
                yref="paper",
                text="<b>From implementing CCS</b>",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                align="left",
                font={
                    "size": _stream_label_size,
                    "family": _font_family,
                    "color": "#222222",
                },
                bgcolor="rgba(255,255,255,0)",
            )

    # Legend below plot.
    row_height_paper = 24 / height
    col_x_positions = [0.05, 0.55]

    # (The "darker shade = CCS operations" color-key text is now redundant
    # with the per-stream annotation on the Ammonia bar, so it's omitted.)

    if other_products_list:
        # Header row
        fig.add_annotation(
            x=col_x_positions[0],
            y=-0.02 - 2 * row_height_paper,
            xref="paper",
            yref="paper",
            text=(
                f"<span style='color:{other_color}'>&#9679;</span> "
                f"<span style='color:#222222'>'Other products' contains:</span>"
            ),
            showarrow=False,
            xanchor="left",
            yanchor="top",
            align="left",
            font={"size": legend_font_size + 1, "family": _font_family},
        )
        for k, prod in enumerate(other_products_list):
            row = 2 + (k // 2)
            col = k % 2
            pretty = names.get(prod, prod.replace("_", " ").title())
            u = float(small_df.loc[small_df[product_col] == prod, unabated_col].sum())
            c = float(
                small_df.loc[small_df[product_col] == prod, ccs_emitted_col].sum()
            )
            total = u + c
            text = f"<span style='color:#222222'>&#8226; {pretty}: {total:.0f} Mt/yr</span>"
            fig.add_annotation(
                x=col_x_positions[col] + 0.02,  # slight indent under the header
                y=-0.02 - (row + 1) * row_height_paper,
                xref="paper",
                yref="paper",
                text=text,
                showarrow=False,
                xanchor="left",
                yanchor="top",
                align="left",
                font={"size": legend_font_size - 1, "family": _font_family},
            )

    if output_path:
        fig.write_image(output_path, scale=3)
    return fig


def plot_state_total(
    scenarios_df: pd.DataFrame,
    methane_scenario: str,
    plotting_config: Union[str, dict],
    cases: Optional[List[str]] = None,
    include_unabated: bool = True,
    no_ccs_position: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 7),
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    display_mode: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """State-total bar chart comparing aggregate emissions across scenario cases.

    Each bar is the sum of emissions across all products for one scenario.
    The "Without CCS" bar sits in the middle as a reference.  CCS bars
    below it show hatched avoided-emissions; bars above show excess in a
    darker color.

    Args:
        scenarios_df: Long-form scenarios output (from scenarios CLI).
        methane_scenario: Which methane scenario to fix.
        plotting_config: Path to plotting config YAML, or a dict.
        cases: Scenario names to include (excluding 'unabated').
            Defaults to all non-unabated cases in the plotting config.
        include_unabated: Show the "Without CCS" reference bar.
        no_ccs_position: Bar position for "Without CCS" (default: middle).
        title: Override the chart title.
        figsize: Figure size in inches.
        output_path: If given, save figure to this path.
        ax: If given, plot on this existing Axes.

    Returns:
        (fig, ax) tuple.
    """
    cfg = _load_config(plotting_config)
    case_defs = cfg["cases"]
    st = cfg.get("state_total", {})

    no_ccs_color = st.get("no_ccs_color", "#e07b39")
    more_color = st.get("more_emissions_color", "#aa4400")
    less_color = st.get("less_emissions_color", "#aaaaaa")
    width = st.get("bar_width", 0.8)
    fs = _resolve_fonts(display_mode)
    labelsize = fs["axis_label"]

    if cases is None:
        cases = [c for c in case_defs if c != "unabated"]
    else:
        cases = [c for c in cases if c != "unabated"]

    totals, no_ccs_value = _compute_state_totals(scenarios_df, methane_scenario, cases)

    sorted_cases = sorted(cases, key=lambda c: totals[c], reverse=True)

    if include_unabated and no_ccs_position is None:
        all_vals_sorted = sorted(
            [(c, totals[c]) for c in sorted_cases] + [("unabated", no_ccs_value)],
            key=lambda x: x[1],
            reverse=True,
        )
        no_ccs_pos = next(
            i for i, (c, _) in enumerate(all_vals_sorted) if c == "unabated"
        )
    elif no_ccs_position is not None:
        no_ccs_pos = no_ccs_position
    else:
        no_ccs_pos = len(sorted_cases) // 2

    n_bars = len(sorted_cases) + (1 if include_unabated else 0)
    ccs_positions = list(range(n_bars))
    if include_unabated:
        ccs_positions.pop(no_ccs_pos)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    plt.rcParams["hatch.linewidth"] = 1
    all_vals = list(totals.values()) + ([no_ccs_value] if include_unabated else [])
    max_val = max(all_vals)

    for pos, case in zip(ccs_positions, sorted_cases):
        case_val = totals[case]
        case_cfg = case_defs[case]
        bar_text = case_cfg.get("bar_label", case_cfg["label"])

        if no_ccs_value >= case_val:
            ax.bar(
                pos,
                no_ccs_value,
                color=less_color,
                width=width,
                edgecolor=no_ccs_color,
                hatch="xx",
                zorder=2,
                alpha=0.85,
            )
            ax.bar(pos, case_val, color=no_ccs_color, width=width, zorder=3)
            val_y = case_val - max_val * 0.07
        else:
            ax.bar(
                pos,
                case_val,
                color=more_color,
                width=width,
                edgecolor="none",
                zorder=2,
            )
            ax.bar(
                pos,
                no_ccs_value,
                color=no_ccs_color,
                width=width,
                edgecolor="none",
                zorder=3,
            )
            val_y = case_val + max_val * 0.002

        above_bar = case_val > no_ccs_value
        label_kwargs = {
            "ha": "center",
            "va": "bottom",
            "fontsize": labelsize + 3,
            "fontfamily": "Arial",
            "fontweight": "bold",
            "color": "black",
            "zorder": 5,
        }
        if above_bar:
            label_kwargs["path_effects"] = [
                pe.withStroke(linewidth=3, foreground="white")
            ]
        ax.text(
            pos,
            val_y,
            f"{case_val:.0f}" + r" MtCO$_2$e/yr",
            **label_kwargs,
        )

        pct = (case_val - no_ccs_value) / no_ccs_value * 100
        if pct > 0:
            excess = case_val - no_ccs_value
            if excess > max_val * 0.08:
                pct_y = no_ccs_value + 0.3 * excess
                ax.text(
                    pos,
                    pct_y,
                    f"(+{pct:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=labelsize - 1,
                    fontfamily="Arial",
                    color="white",
                    zorder=5,
                )

        ax.text(
            pos,
            case_val / 2,
            bar_text,
            ha="center",
            va="center",
            fontsize=labelsize,
            weight="semibold",
            fontfamily="Arial",
            color="white",
            zorder=5,
        )

        if "annotation" in case_cfg:
            ann = case_cfg["annotation"]
            ann_base = max(case_val, no_ccs_value)
            ax.annotate(
                ann["text"],
                xy=(pos, ann_base + max_val * 0.03),
                xytext=(pos, ann_base + max_val * 0.16),
                ha="center",
                fontsize=labelsize + 6,
                fontfamily="Arial",
                fontweight="bold",
                color=ann["color"],
                zorder=6,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": ann["color"],
                    "lw": 2.5,
                    "mutation_scale": 20,
                },
                path_effects=[
                    pe.withStroke(linewidth=4, foreground="white"),
                ],
            )

    if include_unabated:
        ax.bar(
            no_ccs_pos,
            no_ccs_value,
            color=no_ccs_color,
            width=width,
            edgecolor="none",
            zorder=2,
        )
        ax.text(
            no_ccs_pos,
            no_ccs_value + max_val * 0.002,
            f"{no_ccs_value:.0f}" + r" MtCO$_2$e/yr",
            ha="center",
            va="bottom",
            fontsize=labelsize + 3,
            fontfamily="Arial",
            fontweight="bold",
            color="black",
            zorder=5,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
        )
        ax.text(
            no_ccs_pos,
            no_ccs_value / 2,
            "Without\nCCS",
            ha="center",
            va="center",
            fontsize=labelsize,
            fontfamily="Arial",
            weight="semibold",
            color="white",
            zorder=5,
        )

    ax.set_ylim(0, max(all_vals) * 1.2)
    ax.set_ylabel(
        "MtCO2e/yr", fontsize=fs["axis_label"], fontfamily="Arial", labelpad=12
    )
    ax.set_xticks([])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.tick_params(axis="y", labelsize=fs["tick"])

    chart_title = title or st.get("title", "Total GHG emissions by CCS scenario")
    ax.set_title(chart_title, fontsize=fs["title"], fontfamily="Arial", pad=30)

    ax.grid(axis="y", color="#444444", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(bottom=False, left=False)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)

    return fig, ax


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


# Default product-combining rules for `plot_products_to_destination_sankey`.
# The user can override or disable by passing product_group_map={} or a
# different mapping.
_DEFAULT_PRODUCT_GROUP_MAP: Dict[str, str] = {
    "refined_fuels": "refining_and_petrochemicals",
    "petrochemicals": "refining_and_petrochemicals",
    "power_ng_new_build": "power_generation_gas_and_coal",
    "power_ng_retrofit": "power_generation_gas_and_coal",
    "power_coal": "power_generation_gas_and_coal",
}


# Fallback palette used to disambiguate legend (small-node) colors when two
# products naturally share a color. Chosen to avoid clashing with the main
# INDUSTRY_COLORS entries. Cycled through as needed.
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


# Per-state defaults for plot_facility_map. Keyed by full state name (matching
# `us_states_gdf["name"]`). `project_state_codes` are the values used in the
# projects CSV `project_state` column; `xlim`/`ylim` frame the state (with a
# small buffer); `cities` are labeled reference points.
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

# City-specific label offsets (matched by name for backwards compatibility).
# City label offsets, in matplotlib "offset points" from the city marker.
# Tuned to avoid neighbor-label collision in the combined TX+LA view where
# Baton Rouge / New Orleans / Lake Charles are all clustered along the coast.
_CITY_LABEL_OFFSETS: Dict[str, Tuple[int, int]] = {
    "Houston": (-5, 10),
    "Austin": (-8, 10),  # NW
    "Dallas": (8, 8),  # NE
    "New Orleans": (-60, -55),  # SW of marker; leader line drawn
    "Baton Rouge": (6, 36),  # NNE of marker; leader line drawn
    # "Lake Charles": (8, 10),
}

# Cities that get a thin leader line from the marker to the label. Used for
# coastal LA cities where the label has to sit far enough away to avoid the
# dense facility cluster around New Orleans / Baton Rouge.
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
                {
                    "arrowstyle": "-",
                    "color": "black",
                    "linewidth": 0.6,
                    "shrinkA": 0,
                    "shrinkB": 3,
                }
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
            from matplotlib.patches import Patch

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


def plot_texas_facility_map(*args, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Backwards-compat wrapper: plot_facility_map(states=["Texas"], ...).

    Prefer calling plot_facility_map directly with an explicit `states` argument."""
    kwargs.setdefault("states", ["Texas"])
    return plot_facility_map(*args, **kwargs)


def plot_oil_production_by_scenario(
    scenarios_df: pd.DataFrame,
    methane_scenario: str,
    scenarios: Optional[List[str]] = None,
    scenario_labels: Optional[Dict[str, str]] = None,
    state_total: Optional[float] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 7),
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    display_mode: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    fs = _resolve_fonts(display_mode)
    df = scenarios_df[scenarios_df["methane_scenario"] == methane_scenario].copy()

    totals_mmbbl_yr = df.groupby("scenario")["million_barrels_oil_per_yr"].sum()
    totals = totals_mmbbl_yr * BBL_IN_KBOE / DAYS_PER_YEAR

    if scenarios is not None:
        totals = totals.reindex(scenarios).dropna()
    totals = totals.sort_values(ascending=False)

    default_labels = {
        "all_eor": "100% EOR",
        "current_mix": "Current U.S. practice",
        "baseline": "Stated EOR/GS mix",
        "nameplate_implied": "Nameplate\n'capture capacity'",
        "nameplate_operator": "Nameplate\n(operator stated)",
        "all_gs": "100% Geologic\nStorage",
    }
    labels = {**default_labels, **(scenario_labels or {})}

    for ws_name in totals.index:
        if ws_name in (scenario_labels or {}):
            continue
        ws_df = df[df["scenario"] == ws_name]
        total_cap = ws_df["mtco2_per_yr_captured_with_ccs"].sum()
        if total_cap > 0:
            w_eor = (
                ws_df["eor_fraction"] * ws_df["mtco2_per_yr_captured_with_ccs"]
            ).sum() / total_cap
            eor_pct = round(w_eor * 100)
            base = labels.get(ws_name, ws_name)
            labels[ws_name] = f"{base}\n({eor_pct}% EOR; {100 - eor_pct}% GS)"

    bar_color = "#3d2b1f"
    edge_color = "#1a1108"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = np.arange(len(totals))
    bars = ax.bar(
        x,
        totals.values,
        width=0.6,
        color=bar_color,
        edgecolor=edge_color,
        linewidth=1.2,
        zorder=3,
    )

    for this_bar, val in zip(bars, totals.values):
        if val > 0:
            ax.text(
                this_bar.get_x() + this_bar.get_width() / 2,
                val + totals.max() * 0.015,
                f"{val:,.0f} kbbl/day",
                ha="center",
                va="bottom",
                fontsize=fs["bar_label"] + 6,
                fontfamily="Arial",
                fontweight="bold",
                color=bar_color,
            )
            if state_total is not None:
                pct = 100 * val / state_total
                ax.text(
                    this_bar.get_x() + this_bar.get_width() / 2,
                    val / 2,
                    f"{pct:.1f}%\nof state total",
                    ha="center",
                    va="center",
                    fontsize=fs["bar_label"] + 6,
                    fontfamily="Arial",
                    fontweight="bold",
                    color="white",
                )

    tick_labels = [labels.get(s, s) for s in totals.index]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=fs["tick"] + 4, fontfamily="Arial")
    ax.set_xlabel("CCS Scenario", fontsize=fs["axis_label"], fontfamily="Arial")
    ax.set_ylabel(
        "Thousand barrels of oil per day",
        fontsize=fs["axis_label"],
        fontfamily="Arial",
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.tick_params(axis="y", labelsize=fs["tick"])
    ax.set_ylim(0, totals.max() + 100)

    ax.set_title(
        title or "Oil production from EOR enabled by CCS in Texas",
        fontsize=fs["title"],
        fontfamily="Arial",
        fontweight="semibold",
        pad=24,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)

    return fig, ax


def plot_state_total_by_methane_scenario(
    scenarios_df: pd.DataFrame,
    scenarios_config: Optional[Union[str, dict]] = None,
    destination_scenarios: Optional[List[str]] = None,
    methane_scenarios: Optional[List[str]] = None,
    scenario_labels: Optional[Dict[str, str]] = None,
    methane_labels: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 7),
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    display_mode: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    fs = _resolve_fonts(display_mode)
    df = scenarios_df.copy()

    if destination_scenarios is not None:
        df = df[df["scenario"].isin(destination_scenarios)]
    if methane_scenarios is not None:
        df = df[df["methane_scenario"].isin(methane_scenarios)]

    totals = (
        df.groupby(["scenario", "methane_scenario"])[
            "mtco2e_per_yr_emissions_including_ccs"
        ]
        .sum()
        .reset_index()
    )

    default_scen_labels = {
        "all_eor": "100% EOR",
        "current_mix": "Current U.S. practice\n(92% EOR, 8% GS)",
        "baseline": "Stated\nEOR/GS mix",
        "nameplate_implied": "Nameplate\n'capture capacity'",
        "nameplate_operator": "Nameplate\n(operator stated)",
        "all_gs": "100% Geologic\nStorage",
    }
    labels = {**default_scen_labels, **(scenario_labels or {})}

    cfg_methane_labels: Dict[str, str] = {}
    scfg_source = scenarios_config or os.path.join(
        os.path.dirname(__file__), "config", "config_scenarios.yml"
    )
    try:
        scfg = _load_config(scfg_source)
        for ms in scfg.get("methane_scenarios", []):
            cfg_methane_labels[
                ms["name"]
            ] = f"{ms.get('leak_plotting_name', ms['name'])}, {ms.get('gwp_plotting_name', '')}"
    except Exception:
        pass
    m_labels = {**cfg_methane_labels, **(methane_labels or {})}

    scenario_order = (
        totals.groupby("scenario")["mtco2e_per_yr_emissions_including_ccs"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    )

    totals["scenario_label"] = totals["scenario"].map(lambda s: labels.get(s, s))
    totals["methane_label"] = totals["methane_scenario"].map(
        lambda s: m_labels.get(s, s)
    )

    methane_rank = (
        totals.groupby("methane_label")["mtco2e_per_yr_emissions_including_ccs"]
        .mean()
        .sort_values()
    )
    methane_order = methane_rank.index.tolist()

    cat_type = pd.CategoricalDtype(
        categories=[labels.get(s, s) for s in scenario_order], ordered=True
    )
    totals["scenario_label"] = totals["scenario_label"].astype(cat_type)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    palette = sns.color_palette("YlOrRd_r", n_colors=len(methane_order))

    sns.barplot(
        data=totals,
        x="scenario_label",
        y="mtco2e_per_yr_emissions_including_ccs",
        hue="methane_label",
        hue_order=methane_order,
        order=[labels.get(s, s) for s in scenario_order],
        palette=palette,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel(
        r"Total MtCO$_2$e / yr",
        fontsize=fs["axis_label"],
        fontfamily="Arial",
    )
    ax.tick_params(axis="x", labelsize=fs["tick"])
    ax.tick_params(axis="y", labelsize=fs["tick"])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    if title is not None:
        ax.set_title(
            title,
            fontsize=fs["title"],
            fontfamily="Arial",
            fontweight="semibold",
            pad=14,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(
        title="Methane leak rate and GWP",
        fontsize=fs["legend"],
        title_fontsize=fs["legend_title"],
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
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


_CCS_WINDOW_ONLY_COLUMNS = frozenset(
    {
        "mtco2_per_yr_captured_with_ccs",
        "mtco2e_per_yr_emitted_because_of_ccs",
    }
)
_CCS_NET_COLUMNS = frozenset(
    {
        "mtco2e_per_yr_emissions_including_ccs",
    }
)


def compute_unabated_emissions_timeseries(
    scenarios_df: pd.DataFrame,
    projects_df: pd.DataFrame,
    scenario_id: str,
    column: Union[str, Sequence[str]] = "unabated_mtco2e_per_yr",
    ccs_active_years: int = 12,
    facility_lifetime_years: int = 30,
    base_year: Optional[int] = None,
) -> pd.DataFrame:
    """Yearly per-project time series summed across the scenario, matching the
    window semantics in :func:`plot_emissions_through_time`.

    Facility windows are set the same way as in :func:`plot_emissions_through_time`:
    retrofits start at ``base_year``, new builds start at ``project_start``, and
    both run for ``facility_lifetime_years``. The CCS window runs for
    ``ccs_active_years`` starting at ``project_start`` and is nested inside
    the facility window.

    Each requested column is aggregated using one of three semantics, chosen
    automatically from the column name:

    - **facility-window** (default for most columns, e.g.
      ``unabated_mtco2e_per_yr``) — value applies for every year inside the
      facility window.
    - **CCS-window-only** (``mtco2_per_yr_captured_with_ccs``,
      ``mtco2e_per_yr_emitted_because_of_ccs``) — value applies only during
      the 12-year CCS window; zero for the remaining facility years.
    - **net-with-CCS** (``mtco2e_per_yr_emissions_including_ccs``) — value
      applies during the CCS window; outside the CCS window but still inside
      the facility window the project contributes ``unabated_mtco2e_per_yr``
      (because the plant keeps emitting after CCS stops). This matches the
      ``net_arr`` series used by :func:`plot_emissions_through_time`.

    Parameters
    ----------
    scenarios_df, projects_df
        Same frames consumed by :func:`plot_emissions_through_time`. The
        scenario frame must contain either a ``scenario_id`` column or the
        pair (``scenario``, ``methane_scenario``) from which it was built.
    scenario_id
        Compound id ``"<scenario>-<methane_scenario>"`` (or just
        ``"<scenario>"`` for scenarios with no methane variant, e.g.
        ``"existing"`` or ``"unabated"``).
    column
        One column name, or a list/tuple of column names, from
        ``scenarios_df``.
    ccs_active_years
        Length (in years) of the 45Q CCS window. Defaults to 12.
    facility_lifetime_years, base_year
        Match the semantics in :func:`plot_emissions_through_time`.
        ``base_year`` defaults to today.

    Returns
    -------
    pandas.DataFrame
        Column ``year`` plus one column per requested input column (same
        names as the inputs), one row per year from the earliest facility
        start through the latest facility end.
    """
    if base_year is None:
        base_year = datetime.date.today().year

    if "scenario_id" in scenarios_df.columns:
        scen_df = scenarios_df[scenarios_df["scenario_id"] == scenario_id]
    else:
        if "-" in scenario_id:
            scen_name, methane = scenario_id.split("-", 1)
        else:
            scen_name, methane = scenario_id, ""
        scen_df = scenarios_df[scenarios_df["scenario"] == scen_name]
        if methane and "methane_scenario" in scenarios_df.columns:
            scen_df = scen_df[scen_df["methane_scenario"] == methane]
    scen_df = scen_df.drop_duplicates(subset="uid")

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

    columns = [column] if isinstance(column, str) else list(column)
    if not columns:
        raise ValueError("`column` must be a non-empty string or list of strings.")
    bad = [c for c in columns if c not in scen_df.columns]
    if bad:
        raise KeyError(
            f"column(s) {bad!r} not found in scenarios_df. Available columns: "
            f"{sorted(scen_df.columns.tolist())}"
        )

    # "net" columns need to fall back to unabated_mtco2e_per_yr outside the
    # CCS window, so that column must be present (and non-null) for every row.
    needs_unabated = any(c in _CCS_NET_COLUMNS for c in columns)
    required = ["project_start", *columns]
    if needs_unabated and "unabated_mtco2e_per_yr" not in required:
        if "unabated_mtco2e_per_yr" not in scen_df.columns:
            raise KeyError(
                "'unabated_mtco2e_per_yr' is required for net-CCS columns "
                f"({sorted(_CCS_NET_COLUMNS)}) but is missing from scenarios_df."
            )
        required.append("unabated_mtco2e_per_yr")
    missing = scen_df[scen_df[required].isna().any(axis=1)]
    if len(missing):
        names = missing.get("project_name", missing["uid"]).tolist()
        logger.warning(
            "compute_unabated_emissions_timeseries: dropping %d project(s) with "
            "NaN in required columns: %s",
            len(missing),
            names,
        )
        scen_df = scen_df.dropna(subset=required)

    if scen_df.empty:
        available = []
        if "scenario_id" in scenarios_df.columns:
            available = sorted(scenarios_df["scenario_id"].dropna().unique().tolist())
        hint = ""
        if (
            "methane_scenario" in scenarios_df.columns
            and scenario_id in scenarios_df["methane_scenario"].unique()
        ):
            hint = (
                f" (Note: {scenario_id!r} is a methane_scenario, not a scenario_id. "
                "The scenario_id format is '<scenario>-<methane_scenario>', "
                f"e.g. 'baseline-{scenario_id}'.)"
            )
        raise ValueError(
            f"No projects with complete data for scenario_id={scenario_id!r}. "
            f"Nothing to compute.{hint} Available scenario_ids: {available}"
        )

    scen_df = _annotate_facility_windows(scen_df, base_year, facility_lifetime_years)
    timeline_start = int(scen_df["_fac_start"].min())
    timeline_end = int(scen_df["_fac_end"].max())
    years = np.arange(timeline_start, timeline_end)
    totals = {c: np.zeros(len(years)) for c in columns}

    for _, row in scen_df.iterrows():
        ps = int(row["project_start"])
        fac_start = int(row["_fac_start"])
        fac_end = int(row["_fac_end"])
        ccs_end = ps + ccs_active_years
        unabated_val = (
            float(row["unabated_mtco2e_per_yr"])
            if "unabated_mtco2e_per_yr" in row.index
            and not pd.isna(row["unabated_mtco2e_per_yr"])
            else 0.0
        )
        vals = {c: row[c] for c in columns}
        for i, yr in enumerate(years):
            if not fac_start <= yr < fac_end:
                continue
            in_ccs = ps <= yr < ccs_end
            for c in columns:
                if c in _CCS_WINDOW_ONLY_COLUMNS:
                    if in_ccs:
                        totals[c][i] += vals[c]
                elif c in _CCS_NET_COLUMNS:
                    totals[c][i] += vals[c] if in_ccs else unabated_val
                else:
                    totals[c][i] += vals[c]

    return pd.DataFrame({"year": years, **totals})


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
    _switching_first = {
        "switching": "baseline",
        "switching_from_current": "current_mix",
    }
    is_switching = scenario in _switching_first
    if is_switching:
        _p1_name = _switching_first[scenario]
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
                arrowprops={
                    "arrowstyle": "-",
                    "color": _cfg["color"],
                    "linewidth": totals_leader_linewidth,
                    "shrinkA": 0,
                    "shrinkB": 2,
                    "relpos": (0 if _dx >= 0 else 1, 0.5),
                },
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
            arrowprops={
                "arrowstyle": "-",
                "color": _ccs_red,
                "linewidth": 0.9,
                "shrinkA": 0,
                "shrinkB": 2,
                "relpos": (1.0, 0.0),
            },
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


def plot_emissions_through_time_with_solar(
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
) -> Tuple[plt.Figure, plt.Axes]:
    """Like `plot_emissions_through_time`, but stacks an additional negative-y
    block for the equivalent-solar GHG avoidance (`mtco2e_avoided_from_solar`,
    §1.5.7). The per-facility total avoided GHG is distributed evenly across
    the 12-year 45Q window — matching the flat-rate convention used for the
    positive-y unabated and CCS-lifecycle layers — and plotted below the
    existing CO₂-stored block, so the full "GHG benefit if 45Q dollars went to
    solar instead" is visible as deeper negative area.
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

    if "mtco2e_avoided_from_solar" not in scen_df.columns:
        raise ValueError(
            "scenarios_df is missing 'mtco2e_avoided_from_solar'; re-run the "
            "pipeline with the solar-counterfactual columns enabled."
        )

    required = [
        "project_start",
        "unabated_mtco2e_per_yr",
        "mtco2_per_yr_captured_with_ccs",
        "mtco2e_per_yr_emitted_because_of_ccs",
        "mtco2e_avoided_from_solar",
    ]
    missing = scen_df[scen_df[required].isna().any(axis=1)]
    if len(missing):
        names = missing.get("project_name", missing["uid"]).tolist()
        logger.warning(
            "plot_emissions_through_time_with_solar: dropping %d project(s) "
            "with NaN in required columns (fix upstream CSV): %s",
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
    solar_avoided_arr = np.zeros(len(years))

    for _, row in scen_df.iterrows():
        ps = int(row["project_start"])
        fac_start = int(row["_fac_start"])
        fac_end = int(row["_fac_end"])
        ccs_end = ps + ccs_active_years

        unabated = row["unabated_mtco2e_per_yr"]
        captured = row["mtco2_per_yr_captured_with_ccs"]
        ccs_lc = row["mtco2e_per_yr_emitted_because_of_ccs"]
        solar_per_yr = row["mtco2e_avoided_from_solar"] / ccs_active_years

        for i, yr in enumerate(years):
            if yr < fac_start or yr >= fac_end:
                continue
            unabated_arr[i] += unabated
            if ps <= yr < ccs_end:
                uncaptured_arr[i] += unabated - captured
                lifecycle_arr[i] += ccs_lc
                captured_arr[i] += captured
                solar_avoided_arr[i] += solar_per_yr
            else:
                uncaptured_arr[i] += unabated

    net_arr = uncaptured_arr + lifecycle_arr

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.fill_between(
        years,
        0,
        uncaptured_arr,
        step="post",
        color="#888888",
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

    stored_arr = captured_arr
    stored_below = -np.abs(stored_arr)
    stored_fill = ax.fill_between(
        years,
        0,
        stored_below,
        step="post",
        facecolor="#a8d5a2",
        edgecolor="#2ca02c",
        linewidth=0,
    )
    stored_fill.set_hatch("//")
    stored_fill.set_edgecolor("#1a7a1a")

    solar_bottom = stored_below - np.abs(solar_avoided_arr)
    solar_fill = ax.fill_between(
        years,
        stored_below,
        solar_bottom,
        step="post",
        facecolor="#ffd166",
        edgecolor="#e09f00",
        linewidth=0,
        label="GHG avoided if 45Q spent on solar",
    )
    solar_fill.set_hatch("..")
    solar_fill.set_edgecolor("#b37c00")

    ax.plot(
        years,
        stored_below,
        color="#2ca02c",
        linestyle="--",
        linewidth=2,
        drawstyle="steps-post",
        label="CO₂ stored with CCS",
    )

    ax.plot(
        years,
        solar_bottom,
        color="#e09f00",
        linestyle=(0, (3, 1, 1, 1)),
        linewidth=2,
        drawstyle="steps-post",
        label="Stored CO₂ + solar-equivalent GHG avoided",
    )

    ax.plot(
        years,
        unabated_arr,
        color="black",
        linestyle="--",
        linewidth=2,
        drawstyle="steps-post",
        label="Unabated emissions",
    )

    ax.plot(
        years,
        net_arr,
        color="#d62728",
        linestyle=(0, (3, 1, 1, 1)),
        linewidth=2,
        drawstyle="steps-post",
        label="Net emissions (with CCS)",
    )

    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_xlabel("Year", fontsize=fs["axis_label"])
    ax.set_ylabel("MtCO₂e / yr", fontsize=fs["axis_label"], labelpad=-1)
    if title is None:
        title = (
            "Full buildout: Texas facility emissions through time, "
            "with equivalent-solar GHG avoidance"
        )
    if title_fontsize is None:
        title_fontsize = fs["title"]
    ax.set_title(title, fontsize=title_fontsize)
    if legend_fontsize is None:
        legend_fontsize = fs["legend"]
    ax.legend(fontsize=legend_fontsize, loc="lower right", frameon=True)
    ax.tick_params(labelsize=fs["tick"])
    ax.set_xlim(xlim if xlim is not None else (timeline_start, timeline_end))
    ax.set_ylim(ylim if ylim is not None else (solar_bottom.min() * 1.15, None))
    ax.grid(axis="both", alpha=0.5, zorder=-1)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=600)

    return fig, ax


_KNOWN_THROUGH_TIME_SCENARIO_LABELS: Dict[str, str] = {
    "existing": "Existing facilities (no new buildout)",
    "unabated": "Unabated after full buildout (no CCS)",
    "all_gs": "With CCS, 100% geologic storage",
    "baseline": "With CCS, stated destination",
    "current_mix": "With CCS, current mix (92% EOR)",
    "nameplate_implied": "Nameplate 'capture capacity'",
    "all_eor": "With CCS, 100% EOR",
}


def plot_net_emissions_through_time(
    scenarios_df: pd.DataFrame,
    projects_df: pd.DataFrame,
    methane_scenario: str,
    destination_scenarios: List[Union[str, dict]],
    scenario_labels: Optional[Dict[str, str]] = None,
    scenario_colors: Optional[Dict[str, str]] = None,
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
    linewidth: float = 2.5,
    legend_loc: str = "upper right",
    fill: bool = True,
    fill_alpha: float = 1.0,
    scenario_linestyles: Optional[Dict[str, str]] = None,
    scenario_linewidths: Optional[Dict[str, float]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Per-scenario net CO₂e-emitted-to-atmosphere timeline.

    One line per scenario in `destination_scenarios`, each summing per-year
    net atmospheric emissions across the full facility fleet. Timing matches
    `plot_emissions_through_time`: retrofit facilities operate from
    `base_year`, new-builds from `project_start`, and for CCS scenarios the
    CCS window runs for `ccs_active_years` starting at each facility's
    `project_start` before reverting to unabated emissions for the rest of
    the 30-yr (`facility_lifetime_years`) life.

    Per-facility annual net emissions:
     - "existing" pseudo-scenario: `unabated_mtco2e_per_yr` for retrofit
       facilities only, across their 30-yr window.
     - "unabated" pseudo-scenario: `unabated_mtco2e_per_yr` for every
       facility across its 30-yr window.
     - CCS scenarios (baseline / current_mix / all_eor / all_gs /
       nameplate_implied, or any custom scenario-name row in scenarios_df):
       within the facility's CCS window →
       (unabated_mtco2e_per_yr − mtco2_per_yr_captured_with_ccs)
       + mtco2e_per_yr_emitted_because_of_ccs;
       outside the window (but still within facility life) → the facility
       just emits its unabated rate.

    Args:
        destination_scenarios: Ordered list of scenario names (and/or dicts)
            to plot. The pseudo-names "existing" and "unabated" are
            recognized; any other string must match a `scenario` value in
            `scenarios_df`. A dict entry can carry `{"name": "...", "label":
            "...", "is_unabated": True, "is_existing": True}` for custom
            scenarios not covered by the known names.
        scenario_labels: Optional per-scenario legend-label overrides keyed
            by scenario name. Unmatched scenarios fall back to the default
            label (strip of the sibling's x-tick label; see
            `_KNOWN_THROUGH_TIME_SCENARIO_LABELS`).
        scenario_colors: Optional per-scenario color overrides keyed by
            scenario name. Scenarios without an entry pick up the next color
            from matplotlib's default property cycle.
        legend_loc: Pass-through to `ax.legend(loc=...)`. Default
            "upper right" (anchored to the subplot's upper-right corner).
        fill: If True (default), the "unabated after full buildout" scenario
            is rendered as a soft-gray filled polygon acting as a reference
            baseline, and every other scenario is drawn as an unfilled line
            on top of it. Setting `fill=False` leaves every scenario,
            including unabated, as a line.
        fill_alpha: Opacity of the unabated reference fill. Default 1.0
            (solid gray). Drop to ~0.6 if you want gridlines to show through.
        scenario_linestyles: Optional per-scenario linestyle overrides keyed
            by scenario name. Accepts any matplotlib linestyle spec
            ("-", "--", "-.", ":", or a `(offset, (dash, gap, ...))` tuple).
            Scenarios without an entry draw as a solid line. For the
            unabated reference fill, the override also applies to the fill's
            border and the line drawn on top of it, so the legend swatch
            picks up the dashing automatically.
        scenario_linewidths: Optional per-scenario linewidth overrides keyed
            by scenario name. Scenarios without an entry inherit the global
            `linewidth` argument.
    """
    fs = _resolve_fonts(display_mode)
    if base_year is None:
        base_year = datetime.date.today().year

    # Resolve the destination_scenarios list to a list of dicts carrying at
    # least {"name", "label", "is_unabated", "is_existing"}.
    resolved: List[dict] = []
    for s in destination_scenarios:
        if isinstance(s, str):
            resolved.append(
                {
                    "name": s,
                    "label": _KNOWN_THROUGH_TIME_SCENARIO_LABELS.get(s, s),
                    "is_unabated": s == "unabated",
                    "is_existing": s == "existing",
                }
            )
        else:
            name = s["name"]
            resolved.append(
                {
                    "name": name,
                    "label": s.get(
                        "label",
                        _KNOWN_THROUGH_TIME_SCENARIO_LABELS.get(name, name),
                    ),
                    "is_unabated": bool(s.get("is_unabated")),
                    "is_existing": bool(s.get("is_existing")),
                }
            )
    if scenario_labels:
        for d in resolved:
            if d["name"] in scenario_labels:
                d["label"] = scenario_labels[d["name"]]

    # One scenario row per uid; unabated is destination-independent so a
    # single row carries all we need for the "unabated" / "existing" lines.
    scen_methane_df = scenarios_df[scenarios_df["methane_scenario"] == methane_scenario]
    if scen_methane_df.empty:
        raise ValueError(
            f"No scenarios_df rows for methane_scenario={methane_scenario!r}."
        )

    def _ensure_timing_cols(df: pd.DataFrame) -> pd.DataFrame:
        """Return `df` with project_start and retrofit_newbuild filled in from projects_df."""
        if "project_start" not in df.columns:
            df = df.merge(
                projects_df[["uid", "project_start"]].drop_duplicates(subset="uid"),
                on="uid",
                how="left",
            )
        if (
            "retrofit_newbuild" not in df.columns
            and "retrofit_newbuild" in projects_df.columns
        ):
            df = df.merge(
                projects_df[["uid", "retrofit_newbuild"]].drop_duplicates(subset="uid"),
                on="uid",
                how="left",
            )
        return df

    def _compute_net_series(scen_entry: dict) -> Tuple[np.ndarray, np.ndarray]:
        name = scen_entry["name"]
        is_unabated = scen_entry["is_unabated"]
        is_existing = scen_entry["is_existing"]

        if is_existing or is_unabated:
            df = scen_methane_df.drop_duplicates(subset="uid").copy()
        else:
            df = (
                scen_methane_df[scen_methane_df["scenario"] == name]
                .drop_duplicates(subset="uid")
                .copy()
            )
            if df.empty:
                raise ValueError(
                    f"No rows for scenario={name!r} under methane_scenario="
                    f"{methane_scenario!r}."
                )
        df = _ensure_timing_cols(df)

        if is_existing:
            if "retrofit_newbuild" not in df.columns:
                raise ValueError(
                    "'existing' scenario requested but retrofit_newbuild "
                    "column is not available on scenarios_df/projects_df."
                )
            mask = (
                df["retrofit_newbuild"].astype(str).str.strip().str.lower()
                == "retrofit"
            )
            df = df[mask]

        required = ["project_start", "unabated_mtco2e_per_yr"]
        if not is_unabated or is_existing:
            required += [
                "mtco2_per_yr_captured_with_ccs",
                "mtco2e_per_yr_emitted_because_of_ccs",
            ]
        df = df.dropna(subset=required)
        if df.empty:
            years_empty = np.arange(base_year, base_year + facility_lifetime_years)
            return years_empty, np.zeros(len(years_empty))

        df = _annotate_facility_windows(df, base_year, facility_lifetime_years)
        _timeline_start = int(df["_fac_start"].min())
        _timeline_end = int(df["_fac_end"].max())
        years_local = np.arange(_timeline_start, _timeline_end)
        net = np.zeros(len(years_local))

        for _, row in df.iterrows():
            ps = int(row["project_start"])
            fac_start = int(row["_fac_start"])
            fac_end = int(row["_fac_end"])
            ccs_end = ps + ccs_active_years
            unabated = row["unabated_mtco2e_per_yr"]
            if is_unabated or is_existing:
                captured = 0.0
                ccs_lc = 0.0
            else:
                captured = row["mtco2_per_yr_captured_with_ccs"]
                ccs_lc = row["mtco2e_per_yr_emitted_because_of_ccs"]
            for i, yr in enumerate(years_local):
                if yr < fac_start or yr >= fac_end:
                    continue
                if (not is_unabated and not is_existing) and ps <= yr < ccs_end:
                    net[i] += (unabated - captured) + ccs_lc
                else:
                    net[i] += unabated
        # Append (last_year + 1, 0) so steps-post rendering shows a clean
        # vertical drop to zero when the facility fleet retires at the end
        # of its 30-yr life (otherwise the line just terminates without
        # visibly falling off).
        if len(years_local):
            years_local = np.append(years_local, years_local[-1] + 1)
            net = np.append(net, 0.0)
        return years_local, net

    series: List[Tuple[dict, np.ndarray, np.ndarray]] = []
    for scen_entry in resolved:
        yrs, net = _compute_net_series(scen_entry)
        series.append((scen_entry, yrs, net))

    # Union timeline so every line can be plotted over the same x extent.
    all_starts = [yrs[0] for _, yrs, _ in series if len(yrs)]
    all_ends = [yrs[-1] + 1 for _, yrs, _ in series if len(yrs)]
    if not all_starts:
        raise ValueError("All resolved scenarios produced empty data.")
    timeline_start = min(all_starts)
    timeline_end = max(all_ends)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    default_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_cycle_iter = iter(default_cycle)
    assigned_colors: Dict[str, str] = {}
    for scen_entry, _yrs, _net in series:
        name = scen_entry["name"]
        if scenario_colors and name in scenario_colors:
            assigned_colors[name] = scenario_colors[name]
        else:
            try:
                assigned_colors[name] = next(color_cycle_iter)
            except StopIteration:
                color_cycle_iter = iter(default_cycle)
                assigned_colors[name] = next(color_cycle_iter)

    # Render pass: unabated gets a soft-gray reference fill underneath; every
    # other scenario is a line on top. The fill carries an edgecolor matching
    # the unabated line color so the legend swatch's bounding box reads in
    # that color instead of the gray fill color.
    unabated_fill_color = "#d0d0d0"

    def _lw_for(name: str) -> float:
        if scenario_linewidths and name in scenario_linewidths:
            return float(scenario_linewidths[name])
        return linewidth

    def _ls_for(name: str):
        if scenario_linestyles and name in scenario_linestyles:
            return scenario_linestyles[name]
        return "-"

    for scen_entry, yrs, net in series:
        name = scen_entry["name"]
        color = assigned_colors[name]
        lw = _lw_for(name)
        ls = _ls_for(name)
        if fill and scen_entry["is_unabated"]:
            ax.fill_between(
                yrs,
                0,
                net,
                step="post",
                facecolor=unabated_fill_color,
                edgecolor=color,
                alpha=fill_alpha,
                linewidth=lw,
                linestyle=ls,
                zorder=1,
                label=scen_entry["label"],
            )
            ax.plot(
                yrs,
                net,
                color=color,
                linewidth=lw,
                linestyle=ls,
                drawstyle="steps-post",
                zorder=3,
            )
        else:
            ax.plot(
                yrs,
                net,
                color=color,
                linewidth=lw,
                linestyle=ls,
                drawstyle="steps-post",
                label=scen_entry["label"],
                zorder=3,
            )

    ax.axhline(0, color="black", linewidth=0.75, alpha=0.6, zorder=1)

    ax.set_xlabel("Year", fontsize=fs["axis_label"])
    ax.set_ylabel(
        r"Net MtCO$_2$e emitted to atmosphere / yr",
        fontsize=fs["axis_label"],
        labelpad=-1,
    )
    if title is None:
        title = "Net atmospheric CO₂e per scenario, through time"
    ax.set_title(
        title,
        fontsize=title_fontsize if title_fontsize is not None else fs["title"],
    )
    # Legend entries follow the caller's destination_scenarios order, not the
    # back-to-front draw order used for the fills.
    handles, labels_leg = ax.get_legend_handles_labels()
    _label_to_handle = {l: h for h, l in zip(handles, labels_leg)}
    ordered_handles = [
        _label_to_handle[s["label"]] for s in resolved if s["label"] in _label_to_handle
    ]
    ordered_labels = [s["label"] for s in resolved if s["label"] in _label_to_handle]
    ax.legend(
        ordered_handles,
        ordered_labels,
        fontsize=legend_fontsize if legend_fontsize is not None else fs["legend"],
        loc=legend_loc,
        frameon=True,
    )
    ax.tick_params(labelsize=fs["tick"])
    ax.set_xlim(xlim if xlim is not None else (timeline_start, timeline_end))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="both", alpha=0.4, zorder=-1)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=600)

    return fig, ax


def plot_unabated_through_time(
    scenarios_df: pd.DataFrame,
    projects_df: pd.DataFrame,
    methane_scenario: str,
    scenario: str = "baseline",
    facility_lifetime_years: int = 30,
    base_year: Optional[int] = None,
    name_replacements: Optional[Dict[str, str]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (14, 7),
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    display_mode: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Facility-lifetime unabated-emissions timeline by industry, honoring
    the per-project `retrofit_newbuild` flag (retrofit facilities start
    emitting in `base_year`; new-build facilities start at `project_start`).
    `base_year` defaults to the current calendar year.
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

    required = ["project_start", "unabated_mtco2e_per_yr"]
    missing = scen_df[scen_df[required].isna().any(axis=1)]
    if len(missing):
        names = missing.get("project_name", missing["uid"]).tolist()
        logger.warning(
            "plot_unabated_through_time: dropping %d project(s) with NaN in "
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

    industries = (
        scen_df.groupby("product")["unabated_mtco2e_per_yr"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    industry_arrs = {}
    for ind in industries:
        arr = np.zeros(len(years))
        ind_df = scen_df[scen_df["product"] == ind]
        for _, row in ind_df.iterrows():
            fac_start = int(row["_fac_start"])
            fac_end = int(row["_fac_end"])
            for i, yr in enumerate(years):
                if fac_start <= yr < fac_end:
                    arr[i] += row["unabated_mtco2e_per_yr"]
        industry_arrs[ind] = arr

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if name_replacements is None:
        name_replacements = {}

    inline_industries = {"ammonia", "lng", "hydrogen", "power_ng"}

    bottom = np.zeros(len(years))
    legend_handles = []
    legend_labels = []
    for ind in industries:
        arr = industry_arrs[ind]
        color = INDUSTRY_COLORS.get(ind, "#cccccc")
        label = _product_label(ind, name_replacements)
        fill = ax.fill_between(
            years,
            bottom,
            bottom + arr,
            step="post",
            color=color,
            alpha=0.85,
        )
        if ind in inline_industries:
            mid_y = (bottom + bottom + arr) / 2
            peak_idx = int(np.argmax(arr))
            ax.text(
                years[peak_idx],
                mid_y[peak_idx],
                label,
                ha="center",
                va="center",
                fontsize=fs["annotation"] - 1,
                fontweight="bold",
                color="white",
                path_effects=[
                    pe.withStroke(linewidth=2.5, foreground="black", alpha=0.7)
                ],
            )
        else:
            legend_handles.append(fill)
            legend_labels.append(label)
        bottom = bottom + arr

    ax.legend(
        legend_handles,
        legend_labels,
        fontsize=fs["legend"],
        loc="upper right",
        frameon=True,
    )

    ax.set_xlabel("Year", fontsize=fs["axis_label"])
    ax.set_ylabel("MtCO₂e / yr", fontsize=fs["axis_label"])
    ax.set_title(
        title or "Unabated industrial emissions through time",
        fontsize=fs["title"],
    )
    ax.tick_params(labelsize=fs["tick"])
    ax.set_xlim(xlim if xlim is not None else (timeline_start, timeline_end))
    ax.set_ylim(ylim if ylim is not None else (0, None))
    ax.set_axisbelow(True)
    ax.grid(axis="both", alpha=0.5)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=600)

    return fig, ax


def plot_emissions_components_through_time(
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
    display_mode: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Stacked-component emissions timeline, honoring the per-project
    `retrofit_newbuild` flag (retrofit facilities start emitting in
    `base_year`; new-build facilities start at `project_start`).
    `base_year` defaults to the current calendar year.
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

    # Drop rows missing any of the values we accumulate — a single NaN would
    # propagate into every year of the stacked arrays and make ylim NaN.
    required = [
        "project_start",
        "unabated_mtco2_per_yr",
        "unabated_mtco2e_per_yr",
        "mtco2_per_yr_captured_with_ccs",
        "mtco2_per_yr_emitted_because_of_ccs",
        "mtco2e_per_yr_emitted_because_of_ccs",
    ]
    missing = scen_df[scen_df[required].isna().any(axis=1)]
    if len(missing):
        names = missing.get("project_name", missing["uid"]).tolist()
        logger.warning(
            "plot_emissions_components_through_time: dropping %d project(s) with "
            "NaN in required columns (fix upstream CSV): %s",
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
    n = len(years)

    ind_co2_arr = np.zeros(n)
    ind_ch4_arr = np.zeros(n)
    ccs_co2_arr = np.zeros(n)
    ccs_ch4_arr = np.zeros(n)
    stored_arr = np.zeros(n)

    for _, row in scen_df.iterrows():
        ps = int(row["project_start"])
        fac_start = int(row["_fac_start"])
        fac_end = int(row["_fac_end"])
        ccs_end = ps + ccs_active_years

        unabated_co2 = row["unabated_mtco2_per_yr"]
        unabated_ch4_co2e = row["unabated_mtco2e_per_yr"] - unabated_co2
        captured = row["mtco2_per_yr_captured_with_ccs"]
        ccs_co2 = row["mtco2_per_yr_emitted_because_of_ccs"]
        ccs_ch4_co2e = row["mtco2e_per_yr_emitted_because_of_ccs"] - ccs_co2

        for i, yr in enumerate(years):
            if yr < fac_start or yr >= fac_end:
                continue
            ind_ch4_arr[i] += unabated_ch4_co2e
            if ps <= yr < ccs_end:
                ind_co2_arr[i] += unabated_co2 - captured
                ccs_co2_arr[i] += ccs_co2
                ccs_ch4_arr[i] += ccs_ch4_co2e
                stored_arr[i] += captured
            else:
                ind_co2_arr[i] += unabated_co2

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    components = [
        (ind_co2_arr, r"Industrial CO$_2$ emitted", "#888888"),
        (ind_ch4_arr, r"Industrial CH$_4$ emitted", "#e07b39"),
        (ccs_co2_arr, r"CCS-caused CO$_2$ emitted", "#ef5350"),
        (ccs_ch4_arr, r"CCS-caused CH$_4$ emitted", "#b71c1c"),
    ]

    bottom = np.zeros(n)
    for arr, label, color in components:
        ax.fill_between(
            years,
            bottom,
            bottom + arr,
            step="post",
            color=color,
            alpha=0.8,
            label=label,
        )
        bottom = bottom + arr

    stored_below = -stored_arr
    stored_fill = ax.fill_between(
        years,
        0,
        stored_below,
        step="post",
        facecolor="#a8d5a2",
        edgecolor="#2ca02c",
        linewidth=0,
    )
    stored_fill.set_hatch("//")
    stored_fill.set_edgecolor("#1a7a1a")

    ax.plot(
        years,
        stored_below,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.5,
        drawstyle="steps-post",
        label=r"CO$_2$ captured and stored",
    )

    ax.set_xlabel("Year", fontsize=fs["axis_label"])
    ax.set_ylabel("MtCO₂e / yr", fontsize=fs["axis_label"], labelpad=-1)
    ax.set_title(
        title or "Total buildout emissions by component through time",
        fontsize=fs["title"],
    )
    ax.legend(fontsize=fs["legend"], loc="lower right", frameon=True)
    ax.tick_params(labelsize=fs["tick"])
    ax.set_xlim(xlim if xlim is not None else (timeline_start, timeline_end))
    ax.set_ylim(ylim if ylim is not None else (stored_below.min() * 1.15, None))
    ax.set_axisbelow(True)
    ax.grid(axis="both", alpha=0.5)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=600)

    return fig, ax


def plot_45q_tradeoff(
    scenarios_df: pd.DataFrame,
    methane_scenario: str = "likely",
    scenarios_order: Optional[List[str]] = None,
    states: Optional[List[str]] = None,
    project_duration_years: int = 12,
    figsize: Tuple[float, float] = (12, 6),
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    display_mode: Optional[str] = None,
    output_path: Optional[str] = None,
    ccs_color: str = "#d62728",
    solar_color: str = "#f4b942",
    show_dollar_annotations: bool = True,
    legend_loc: str = "upper right",
    legend_ncol: int = 1,
    xtick_labels: Optional[Sequence[str]] = None,
    ylabel: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    label_bars: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """Grouped-bar comparison of CCS vs equivalent-solar GHG benefit.

    For each CCS scenario, two side-by-side bars show the 12-yr lifetime
    MtCO2e benefit of two uses of the same nominal 45Q dollars:

    - **CCS**: net GHG reduction vs the unabated counterfactual, i.e.
      ``(unabated_mtco2e_per_yr − mtco2e_per_yr_emissions_including_ccs)
      × project_duration_years``. This subtracts CCS lifecycle emissions
      and EOR-induced oil combustion from gross capture.
    - **Solar**: ``mtco2e_avoided_from_solar`` (pipeline column) summed
      across projects — the 12-yr lifetime GHG avoided if the same 45Q $
      bought utility-scale solar MWh at the state's grid intensity.

    When multiple states are passed, each state becomes a subplot panel.
    The nominal 45Q $ driving both bars is annotated below each group
    (same dollars, two different uses).

    Args:
        scenarios_df: Output of ``run_full_pipeline`` → "scenarios" frame.
        methane_scenario: Filters to one methane scenario (e.g. "likely",
            "edf20", "industry_cited_low_gwp").
        scenarios_order: CCS scenarios to plot, left-to-right. Defaults
            to ``["all_gs", "baseline", "current_mix", "all_eor"]``.
        states: State codes to include. Defaults to ``["TX", "LA"]``;
            each becomes a subplot.
        project_duration_years: CCS operation window length (12 by default).
        ccs_color: Fill color for CCS bars.
        solar_color: Fill color for solar bars (drawn with a hatch so it
            is distinguishable in grayscale print).
        show_dollar_annotations: If True, write the shared 45Q $ (nominal,
            millions) below each scenario group.
    """
    fs = _resolve_fonts(display_mode)

    if scenarios_order is None:
        scenarios_order = ["all_gs", "baseline", "current_mix", "all_eor"]
    if states is None:
        states = ["TX", "LA"]

    _scen_display = {
        "all_gs": "100% GS",
        "baseline": "Announced\ndestinations",
        "current_mix": "Current U.S.\npractice (92% EOR)",
        "all_eor": "100% EOR",
    }
    _state_display = {"TX": "Texas", "LA": "Louisiana"}

    df = scenarios_df[scenarios_df["methane_scenario"] == methane_scenario].copy()
    df = df[df["project_state"].isin(states)]

    # One row per uid for the unabated baseline (identical across CCS scenarios).
    if "uid" in df.columns:
        unab_src = df.drop_duplicates(subset="uid")
    else:
        unab_src = df[df["scenario"] == scenarios_order[0]]
    unab_by_state = unab_src.groupby("project_state")["unabated_mtco2e_per_yr"].sum()

    # Assemble per-(state, scenario) bar heights and shared $-labels.
    data: Dict[str, Dict[str, Dict[str, float]]] = {}
    for state in states:
        data[state] = {}
        sdf_state = df[df["project_state"] == state]
        unabated_total = float(unab_by_state.get(state, 0.0))
        for scen in scenarios_order:
            sdf = sdf_state[sdf_state["scenario"] == scen]
            net_with_ccs = float(sdf["mtco2e_per_yr_emissions_including_ccs"].sum())
            ccs_avoided_mt = (unabated_total - net_with_ccs) * project_duration_years
            solar_avoided_mt = (
                float(sdf["mtco2e_avoided_from_solar"].sum())
                if "mtco2e_avoided_from_solar" in sdf.columns
                else 0.0
            )
            nominal_45q_m = (
                float(sdf["avg_million_usd_per_yr_45q"].sum()) * project_duration_years
            )
            data[state][scen] = {
                "ccs": ccs_avoided_mt,
                "solar": solar_avoided_mt,
                "nominal_45q_m": nominal_45q_m,
            }

    # Build the figure. One panel per state when multiple requested; otherwise
    # honor the caller's `ax` or create a single panel.
    if ax is not None and len(states) == 1:
        axes = [ax]
        fig = ax.figure
    else:
        fig, axes_arr = plt.subplots(
            1, len(states), figsize=figsize, sharey=True, squeeze=False
        )
        axes = list(axes_arr[0])

    x = np.arange(len(scenarios_order))
    bar_w = 0.22
    xtick_fontsize = fs["tick"] + 3
    for panel_i, state in enumerate(states):
        a = axes[panel_i]
        # Avoided (positive data) is plotted negative so it extends downward,
        # matching "kept out of the atmosphere." Positive y is reserved for
        # emitted-to-atmosphere semantics (shown via arrow annotation).
        ccs_h = [-data[state][s]["ccs"] for s in scenarios_order]
        sol_h = [-data[state][s]["solar"] for s in scenarios_order]
        dollars = [data[state][s]["nominal_45q_m"] for s in scenarios_order]

        bars_ccs = a.bar(
            x - bar_w / 2,
            ccs_h,
            bar_w,
            color=ccs_color,
            edgecolor="#333333",
            linewidth=0.5,
            label="CCS (net GHG avoided)",
        )
        bars_solar = a.bar(
            x + bar_w / 2,
            sol_h,
            bar_w,
            facecolor=solar_color,
            edgecolor="#333333",
            linewidth=0.5,
            label="Solar from same 45Q $",
        )
        for b in bars_solar:
            b.set_hatch("//")

        # Optional on-bar labels (used in place of a legend when caller passes
        # label_bars=True). "CCS" in bold white; "Solar" in bold black with a
        # thin white halo so it reads against the hatched yellow fill.
        if label_bars:
            for b, h in zip(bars_ccs, ccs_h):
                a.text(
                    b.get_x() + b.get_width() / 2,
                    h / 2,
                    "CCS",
                    ha="center",
                    va="center",
                    fontsize=fs["legend"] + 6,
                    fontweight="bold",
                    color="white",
                )
            for b, h in zip(bars_solar, sol_h):
                t = a.text(
                    b.get_x() + b.get_width() / 2,
                    h / 2,
                    "Solar",
                    ha="center",
                    va="center",
                    fontsize=fs["legend"] + 6,
                    fontweight="bold",
                    color="black",
                )
                t.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])

        a.set_xticks(x)
        if xtick_labels is not None:
            if len(xtick_labels) != len(scenarios_order):
                raise ValueError(
                    f"xtick_labels has {len(xtick_labels)} entries but "
                    f"scenarios_order has {len(scenarios_order)}"
                )
            tick_text = list(xtick_labels)
        else:
            tick_text = [_scen_display.get(s, s) for s in scenarios_order]
        a.set_xticklabels(tick_text, fontsize=xtick_fontsize)
        a.tick_params(axis="y", labelsize=fs["tick"])
        a.tick_params(axis="x", labelsize=xtick_fontsize)
        a.axhline(0, color="#333333", linewidth=0.8)
        a.set_axisbelow(True)
        a.grid(axis="y", alpha=0.4)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        # Y tick labels show absolute MtCO2e (the sign carries direction, not
        # magnitude), so a bar reaching y=-500 reads as "500 MtCO2e."
        a.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _pos: f"{abs(v):g}")
        )
        if panel_i == 0:
            a.set_ylabel(
                ylabel
                if ylabel is not None
                else f"MtCO₂e over {project_duration_years}-yr lifetime",
                fontsize=fs["axis_label"],
            )

        # Expand ylim so bars (below zero) + headroom (above zero) for the
        # "emitted" arrow annotation both fit. Caller's ylim wins when given.
        if ylim is not None:
            a.set_ylim(ylim)
        else:
            data_min = min(ccs_h + sol_h + [0])
            a.set_ylim(data_min * 1.25, abs(data_min) * 0.55)

        # Arrow annotations centered between the groups of bars: up (dark
        # gray) = "emitted to atmosphere"; down (dark green) = "kept out of
        # atmosphere". Horizontal labels placed next to each arrow tip.
        ymin_a, ymax_a = a.get_ylim()
        arrow_x = float(np.mean(x))  # horizontal center of the bar groups
        xmin_a, xmax_a = a.get_xlim()
        emitted_color = "#4a4a4a"
        avoided_color = "#1a7a1a"

        up_tip_y = ymax_a * 0.92
        up_tail_y = ymax_a * 0.10
        a.annotate(
            "",
            xy=(arrow_x, up_tip_y),
            xytext=(arrow_x, up_tail_y),
            arrowprops={"arrowstyle": "->", "color": emitted_color, "lw": 1.6},
        )
        a.text(
            arrow_x + 0.02 * (xmax_a - xmin_a),
            up_tip_y,
            "emitted to\natmosphere",
            ha="left",
            va="top",
            fontsize=fs["tick"],
            color=emitted_color,
        )

        down_tip_y = ymin_a * 0.92
        down_tail_y = ymin_a * 0.10
        a.annotate(
            "",
            xy=(arrow_x, down_tip_y),
            xytext=(arrow_x, down_tail_y),
            arrowprops={"arrowstyle": "->", "color": avoided_color, "lw": 1.6},
        )
        a.text(
            arrow_x + 0.02 * (xmax_a - xmin_a),
            down_tip_y,
            "kept out of\natmosphere",
            ha="left",
            va="bottom",
            fontsize=fs["tick"],
            color=avoided_color,
        )

        if show_dollar_annotations:
            # Bars now extend downward, so $-labels sit just below each pair.
            ymin_b, _ymax_b = a.get_ylim()
            y_label = ymin_b + 0.02 * abs(ymin_b)
            for xi, d in zip(x, dollars):
                a.text(
                    xi,
                    y_label,
                    f"${d/1000:,.1f}B",
                    ha="center",
                    va="top",
                    fontsize=fs["tick"] - 1,
                    color="#444444",
                )

    if not label_bars:
        axes[0].legend(
            fontsize=fs["legend"], loc=legend_loc, frameon=True, ncol=legend_ncol
        )

    if title is None:
        title = (
            f"45Q tradeoff: CCS vs equivalent solar "
            f"(methane_scenario='{methane_scenario}')"
        )
    fig.suptitle(title, fontsize=fs["title"])
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=600)

    return fig, (axes[0] if len(axes) == 1 else axes)


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
    for _sid, sub in annotated.groupby("scenario_id"):
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


# -----------------------------------------------------------------------------
# Per-project summary table
# -----------------------------------------------------------------------------

_PROJECT_INDUSTRY_DISPLAY = {
    "hydrogen": "Hydrogen",
    "ammonia": "Ammonia",
    "methanol": "Methanol",
    "ethanol": "Ethanol",
    "lng": "LNG",
    "ngp": "NGP",
    "refined_fuels": "Refining",
    "petrochemicals": "Petrochemical",
    "iron_steel": "Iron/Steel",
    "power_coal": "Power (coal)",
    "power_ng": "Power (NG)",
    "power_ng_retrofit": "Power (NG)",
    "power_ng_new_build": "Power (NG)",
    "power_ng_data_center": "Power (NG)",
    "dac": "DAC",
}

# (divisor, display_unit) applied to raw product_amount by co2_source.
# Industries not listed keep the native product_unit from the input CSV.
_PROJECT_UNIT_SCALE = {
    "refined_fuels": (1e3, "kbbl/yr"),
    "power_coal": (1e3, "GWh/yr"),
    "power_ng": (1e3, "GWh/yr"),
    "power_ng_retrofit": (1e3, "GWh/yr"),
    "power_ng_new_build": (1e3, "GWh/yr"),
    "power_ng_data_center": (1e3, "GWh/yr"),
    "ngp": (1e3, "Bbtu/yr"),
    "hydrogen": (1e3, "kton/yr"),
    "ammonia": (1e3, "kton/yr"),
    "methanol": (1e3, "kton/yr"),
    "ethanol": (1e3, "kton/yr"),
    "iron_steel": (1e3, "kton/yr"),
    "lng": (1e3, "kton/yr"),
    "dac": (1e6, "MtCO2/yr"),
}


def _project_status_code(raw: Optional[str]) -> str:
    """Normalize messy project_status to {P, O, C}.

    - Operational → O
    - In/under construction → C
    - Planning / paused / on hold / anything else → P
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "P"
    s = str(raw).strip().lower()
    if s == "operational":
        return "O"
    if "construction" in s:
        return "C"
    # planning, in planning, paused, on hold, anything else → treat as planning
    return "P"


def summarize_projects(
    projects_df: pd.DataFrame,
    project_info_df: Optional[pd.DataFrame] = None,
    scenario: Optional[str] = None,
    methane_scenario: Optional[str] = None,
    co2_source_col: str = "product",
    captured_col: Optional[str] = None,
    return_styled: bool = False,
):
    """Per-project summary table (one row per facility).

    Columns, left to right:
      - State
      - Project Name (status*)         "{name} (X)" where X ∈ {P, O, C}
      - Industry                        display name (Refining, Power (NG), …)
      - Approx. location (lat, lon)     "({lat:.2f}, {lon:.2f})"
      - Use of CO2                      GS or EOR
      - Product produced                "{int(amount)} {unit}" with the unit
                                        conversions specified in the paper
      - Unabated facility emissions [MtCO2e/yr]
      - Carbon captured with CCS [MtCO2/yr]
      - Project start                   int year

    Args:
        projects_df: One row per facility; must carry project_name,
            project_status, project_state, lat_degrees_n, lon_degrees_e,
            carbon_fate, project_start, product_amount, product_unit,
            an industry-tag column (see `co2_source_col`),
            unabated_mtco2e_per_yr, and some column with annual captured CO2.
        co2_source_col: Column name that holds the industry tag used for
            display mapping and unit conversion (one of 'hydrogen',
            'ammonia', 'power_coal', 'refined_fuels', etc.). Defaults to
            `'product'` since both scenarios_df and the pipeline's
            projects_df output expose it under that name; pass
            `'co2_source'` if the input uses the raw CSV convention.
        captured_col: Which column holds the capture tonnage. Defaults to
            the first of `mtco2_per_yr_captured_with_ccs`,
            `nameplate_captured_mtco2_per_yr`,
            `computed_captured_mtco2_per_yr` that is present in the input.
        return_styled: If True, return a pandas Styler with per-column
            number formatting; otherwise return the raw DataFrame.
    """
    df = projects_df.copy()

    # Scenario filtering (if the input carries scenario columns)
    if scenario is not None and "scenario" in df.columns:
        df = df[df["scenario"] == scenario]
    if methane_scenario is not None and "methane_scenario" in df.columns:
        df = df[df["methane_scenario"] == methane_scenario]

    # Auto-merge metadata from project_info_df if any is missing. The CLI
    # emits two CSVs (ccs_projects_out.csv for metadata, ccs_scenarios_out.csv
    # for emissions); passing both lets us assemble the full view.
    _meta_cols = [
        "project_name",
        "project_status",
        "project_state",
        "lat_degrees_n",
        "lon_degrees_e",
        "carbon_fate",
        "project_start",
        "product_amount",
        "product_unit",
        "co2_source",
        "retrofit_newbuild",
    ]
    if (
        project_info_df is not None
        and "uid" in df.columns
        and "uid" in project_info_df.columns
    ):
        missing = [c for c in _meta_cols if c not in df.columns]
        if missing:
            info_cols = ["uid"] + [c for c in missing if c in project_info_df.columns]
            info = project_info_df[info_cols].drop_duplicates(subset="uid")
            df = df.merge(info, on="uid", how="left")

    # Collapse to one row per uid if present (scenarios_df-shaped input).
    if "uid" in df.columns:
        df = df.drop_duplicates(subset="uid").reset_index(drop=True)

    if co2_source_col not in df.columns:
        raise KeyError(
            f"co2_source_col={co2_source_col!r} not in df columns "
            f"(after optional merge). Pass co2_source_col='co2_source' if "
            f"your DataFrame uses that name, or supply project_info_df "
            f"with the metadata."
        )
    cap_candidates = [
        "mtco2_per_yr_captured_with_ccs",
        "nameplate_captured_mtco2_per_yr",
        "computed_captured_mtco2_per_yr",
    ]
    if captured_col is None:
        for c in cap_candidates:
            if c in df.columns:
                captured_col = c
                break
    if captured_col is None or captured_col not in df.columns:
        raise KeyError(
            "No capture-tonnage column found. Expected one of: " f"{cap_candidates}"
        )

    def _industry(row):
        return _PROJECT_INDUSTRY_DISPLAY.get(
            row[co2_source_col], str(row[co2_source_col]).title()
        )

    def _location(row):
        lat = row.get("lat_degrees_n")
        lon = row.get("lon_degrees_e")
        if pd.isna(lat) or pd.isna(lon):
            return ""
        return f"({float(lat):.2f}, {float(lon):.2f})"

    def _product(row):
        divisor, display_unit = _PROJECT_UNIT_SCALE.get(
            row[co2_source_col], (1.0, row.get("product_unit", ""))
        )
        amt = row.get("product_amount")
        if pd.isna(amt):
            return ""
        return f"{int(round(float(amt) / divisor)):,} {display_unit}"

    def _project_name_with_status(row):
        name = str(row.get("project_name", ""))
        return f"{name} ({_project_status_code(row.get('project_status'))})"

    def _carbon_fate(row):
        f = row.get("carbon_fate")
        if pd.isna(f):
            return ""
        return str(f).strip().upper()

    def _year(row):
        y = row.get("project_start")
        if pd.isna(y):
            return ""
        return int(float(y))

    def _new_build_flag(row):
        v = row.get("retrofit_newbuild")
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        return "x" if str(v).strip().lower() == "new build" else ""

    out = pd.DataFrame(
        {
            "State": df["project_state"].values,
            "Project Name (status*)": df.apply(
                _project_name_with_status, axis=1
            ).values,
            "Industry": df.apply(_industry, axis=1).values,
            "New build": df.apply(_new_build_flag, axis=1).values,
            "Approx. location (lat, lon)": df.apply(_location, axis=1).values,
            "Use of CO2": df.apply(_carbon_fate, axis=1).values,
            "Product produced": df.apply(_product, axis=1).values,
            "Unabated facility emissions [MtCO2e/yr]": df[
                "unabated_mtco2e_per_yr"
            ].values,
            "Carbon captured with CCS [MtCO2/yr]": df[captured_col].values,
            "Project start": df.apply(_year, axis=1).values,
        }
    )

    if not return_styled:
        return out

    styled = out.style.format(
        {
            "Unabated facility emissions [MtCO2e/yr]": "{:,.2f}",
            "Carbon captured with CCS [MtCO2/yr]": "{:,.2f}",
            "Project start": "{:.0f}",
        }
    ).set_caption(
        "Per-project summary. Status key: P = in planning / paused / on hold; "
        "O = operational; C = under construction."
    )
    return styled
