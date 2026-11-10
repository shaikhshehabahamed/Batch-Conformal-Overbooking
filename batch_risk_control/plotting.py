"""Shared matplotlib style for all paper figures: a validated, colorblind-
safe categorical palette (fixed hue-per-method assignment, consistent
across every figure), clean typography, and print-appropriate chrome.
Adapted from a general-purpose data-viz palette for static PDF/PNG output
rather than an interactive web chart -- hover/tooltip guidance doesn't
apply here, but the color-assignment and accessibility rules do.
"""
import matplotlib as mpl

# Validated categorical palette (light-mode steps; adjacent pairs clear
# CVD deltaE >= 8 and normal-vision deltaE >= 15 in the source validation).
METHOD_COLORS = {
    "ours": "#2a78d6",          # slot 1 blue -- the proposed method, primary
    "ours_static": "#2a78d6",
    "ours_ACI": "#2a78d6",
    "naive": "#eb6834",         # slot 2 orange -- flawed status-quo baseline
    "bonferroni": "#1baf7a",    # slot 3 aqua -- safe, valid fallback
    "dro_plugin": "#eda100",    # slot 4 yellow -- plug-in DRO baseline
    "dro_calib": "#e87ba4",     # slot 5 magenta -- calibrated DRO baseline
}

METHOD_LABELS = {
    "ours": "Ours",
    "ours_static": "Ours (static)",
    "ours_ACI": "Ours (ACI)",
    "naive": "Naive parametric",
    "bonferroni": "Bonferroni",
    "dro_plugin": "DRO (plug-in)",
    "dro_calib": "DRO (calibrated)",
}

# Consistent draw order (legend and z-order): hero method last-drawn (on
# top) but first-listed in the legend.
METHOD_ORDER = ["ours", "bonferroni", "dro_plugin", "dro_calib", "naive"]

# Distinct marker shapes per method -- a second (shape) channel on top of
# color, so lines that happen to nearly overlap (e.g. naive and dro_plugin
# in the decision-layer figure) stay distinguishable, and identity survives
# grayscale/print.
METHOD_MARKERS = {
    "ours": "o", "ours_static": "o", "ours_ACI": "o",
    "bonferroni": "s",
    "dro_plugin": "^",
    "dro_calib": "D",
    "naive": "v",
}

# Colors for non-method categorical contrasts (e.g. time periods, before/
# after) -- deliberately NOT drawn from METHOD_COLORS, so a color never
# means "Bonferroni" in one figure and "test period" in another.
PERIOD_COLORS = {"before": "#52514e", "after": "#e34948"}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"
SURFACE = "#ffffff"
TARGET_LINE = "#898781"


def setup_style():
    """Call once at the top of each figure script."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.labelcolor": INK_PRIMARY,
        "axes.edgecolor": AXIS_LINE,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "axes.axisbelow": True,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "text.color": INK_PRIMARY,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,   # embed as real text, not outlines/curves
        "ps.fonttype": 42,
    })


def clean_spines(ax):
    """Keep only the left/bottom axis lines, matching the 'recessive
    grid/axes' rule -- data ink, not chart furniture, should dominate."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_LINE)
    ax.spines["bottom"].set_color(AXIS_LINE)


def save_fig(fig, path_stem):
    """Save both a vector PDF (for LaTeX inclusion) and a high-res PNG
    (for quick viewing / non-LaTeX use), matching the project's established
    dual-format convention.
    """
    fig.savefig(f"{path_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{path_stem}.png", bbox_inches="tight", dpi=300)
