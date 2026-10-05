"""Shared matplotlib style for report figures (light surface, validated categorical palette)."""
import matplotlib as mpl

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ_CMAP = "Blues"
# Fixed colour per method so a method keeps its colour in every figure.
METHOD_COLORS = {
    "Original": "#52514e",
    "Task-AE": SERIES[0],
    "AE-MSE": SERIES[1],
    "DCT": SERIES[2],
    "Low-Mel": SERIES[3],
    "INT8": SERIES[6],
}


def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=SERIES), "lines.linewidth": 2,
        "lines.markersize": 7, "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })
