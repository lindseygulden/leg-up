"""Vendored subset of `utils.plotting` — brace-drawing helper."""
from matplotlib.patches import PathPatch
from matplotlib.path import Path


def add_brace(ax, x, y1, y2, side="right", width=0.2, **kwargs):
    """
    Add a thin curly brace { or } spanning exactly y1 to y2.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes.

    x : float
        Position of the inner point of the brace.

    y1, y2 : float
        Exact bottom and top of the brace.

    side : {"left", "right"}
        "left"  -> {
        "right" -> }

    width : float
        Horizontal width of the brace in data coordinates.

    **kwargs
        Passed to PathPatch, e.g. lw=1.5, color="black".
    """
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")

    if y2 < y1:
        y1, y2 = y2, y1

    direction = 1 if side == "right" else -1

    h = y2 - y1
    ym = (y1 + y2) / 2

    # Fraction of the height used for each curved section.
    q = 0.25 * h

    verts = [
        (x, y2),  # top
        (x + direction * width, y2),  # control
        (x + direction * width, ym + q),  # control
        (x, ym),  # center point
        (x + direction * width, ym - q),  # control
        (x + direction * width, y1),  # control
        (x, y1),  # bottom
    ]

    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
    ]

    patch = PathPatch(
        Path(verts, codes),
        fill=False,
        **kwargs,
    )

    ax.add_patch(patch)

    return patch
