#!/usr/bin/env python3
# =============================================================================
# Result 2, Fig. a: DE-SWAN significant-feature counts across age, 5 omics layers
#
# Input  (written by 01-plot_deswan_omics_one.R, one run per omics):
#   outputs/deswan-window_<W>/<omics>_window_counts_window_<tag>.csv
#   outputs/deswan-window_<W>/<omics>_main_peaks_window_<tag>.csv
# Output:
#   figures/deswan-window_<W>/panel_a_deswan_omics_window_<tag>.{pdf,svg,png}
#   figures/deswan-window_<W>/panel_a_deswan_omics_window_<tag>_source_data.csv
#
# Canvas 183 x 50 mm (Nature double column). Axes are placed at absolute mm
# positions with fig.add_axes; no tight_layout / bbox_inches="tight", so the
# exported size is exact. Text stays editable (pdf.fonttype 42, svg.fonttype none).
#
# Usage:  python 01-plot_deswan_all.py [--window 7.5] [--output-dir D] [--figure-dir D]
#         (called automatically at the end of 01-plot_deswan_omics_one.R in Result-02)
#         env DESWAN_OUTPUT_ROOT / DESWAN_FIGURE_ROOT override the default dirs.
# =============================================================================

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------
# 1. Content
# ----------------------------------------------------------------------------
OMICS = [  # (file key, panel title, colour)
    ("transcriptomics", "Transcriptomics", "#CB6E70"),
    ("metabolomics", "Metabolomics", "#68AADD"),
    ("lipidomics", "Lipidomics", "#54AD94"),
    ("gut_microbiome", "Gut microbiome", "#DDA062"),
    ("oral_microbiome", "Oral microbiome", "#896EAE"),
]
AGE_MIN, AGE_MAX = 40, 70

# ----------------------------------------------------------------------------
# 2. Style
# ----------------------------------------------------------------------------
TEXT = "#222222"
GREY = "#6B6B6B"
LIGHT = "#D9D9D9"

FS_TICK, FS_AXIS, FS_TITLE, FS_ANNOT, FS_LETTER = 5.5, 6.0, 6.5, 5.3, 8.0
FS_SEC_TICK, FS_SEC_LABEL = 5.0, 5.5

LINE_W = 0.9
FILL_ALPHA = 0.18
GUIDE_W = 0.5
GUIDE_DASH = (0, (2, 1.5))
PEAK_S = 14
PEAK_EDGE_W = 0.6
Y_HEADROOM = 1.45

# ----------------------------------------------------------------------------
# 3. Geometry (mm)
# ----------------------------------------------------------------------------
FIG_W_MM, FIG_H_MM = 183.0, 50.0
AX_W_MM, AX_H_MM = 23.4, 28.0
AX_LEFT_MM = [11.3, 45.8, 80.4, 114.9, 149.4]
AX_BOTTOM_MM = 8.6          # room for tick labels + "Age (years)"
LETTER_XY_MM = (1.0, FIG_H_MM - 4.0)

MM = 1 / 25.4
FONT = "Arial"


def configure_fonts() -> str:
    """Register Arial from ~/.fonts and require it (same approach as Result-01/01-cohort_feature.ipynb)."""
    for f in glob.glob(os.path.expanduser("~/.fonts/*.ttf")):
        font_manager.fontManager.addfont(f)

    font_manager.findfont(font_manager.FontProperties(family=FONT), fallback_to_default=False)
    font_manager.findfont(font_manager.FontProperties(family=FONT, weight="bold"),
                          fallback_to_default=False)

    plt.rcParams.update({
        "font.family": FONT,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
        "text.color": TEXT,
        "axes.edgecolor": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "axes.linewidth": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "axes.labelsize": FS_AXIS,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    return FONT


def window_tag(window: float) -> str:
    return "window_" + f"{window:g}".replace(".", "p")


def load_omics(output_dir: Path, key: str, tag: str):
    counts = pd.read_csv(output_dir / f"{key}_window_counts_{tag}.csv").sort_values("age")
    peaks = pd.read_csv(output_dir / f"{key}_main_peaks_{tag}.csv").sort_values("peak_age")
    n_total = int(counts["n_tested"].max())
    if counts["n_tested"].nunique() != 1:
        print(f"[warn] {key}: n_tested varies by age; % axis uses the maximum ({n_total:,}).")
    lookup = counts.set_index("age")["n_significant"]
    for age, n in zip(peaks["peak_age"], peaks["peak_n"]):
        if int(lookup.loc[age]) != int(n):
            raise ValueError(f"{key}: main-peak count at {age} y disagrees with window counts.")
    return counts, peaks, n_total


def peak_label_layout(age: float, peak_n: float, ymax: float):
    """Return (x, y, ha, va) for a peak label following the panel-a placement rules."""
    if 48 <= age <= 50:      # early wave: to the right of the point
        return age + 1.3, peak_n + 0.015 * ymax, "left", "bottom"
    if age >= 60:            # late transcript peak: left of the point, slightly below the top
        return age - 1.2, peak_n - 0.02 * ymax, "right", "top"
    return age, peak_n + 0.05 * ymax, "center", "bottom"   # small 52-y peak: centred above


def draw_panel(ax, key, title, colour, counts, peaks, n_total, first, last):
    x = counts["age"].to_numpy(float)
    y = counts["n_significant"].to_numpy(float)
    peak_max = float(peaks["peak_n"].max()) if len(peaks) else float(y.max())
    ymax = max(peak_max, 1.0) * Y_HEADROOM

    ax.fill_between(x, 0, y, color=colour, alpha=FILL_ALPHA, lw=0, zorder=1)
    ax.plot(x, y, color=colour, lw=LINE_W, solid_joinstyle="round",
            solid_capstyle="round", zorder=3)

    for age, n in zip(peaks["peak_age"], peaks["peak_n"]):
        pct = 100 * n / n_total
        ax.plot([age, age], [0, n], color=colour, lw=GUIDE_W, ls=GUIDE_DASH, zorder=2)
        ax.scatter([age], [n], s=PEAK_S, color=colour, edgecolors="white",
                   linewidths=PEAK_EDGE_W, zorder=6, clip_on=False)
        lx, ly, ha, va = peak_label_layout(age, n, ymax)
        ax.text(lx, ly, f"{age:g} y\n{int(n):,} ({pct:.1f}%)", fontsize=FS_ANNOT,
                color=TEXT, ha=ha, va=va, linespacing=1.05, multialignment=ha if ha != "center" else "center",
                fontweight="bold" if pct > 10 else "normal", zorder=7)

    # Axes
    ax.set_xlim(AGE_MIN, AGE_MAX)
    ax.set_xticks([40, 50, 60, 70])
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("Age (years)", labelpad=1)
    if first:
        ax.set_ylabel("No. significant features", labelpad=2)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", color="black", pad=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.5)

    # Secondary axis: % of all features in this omics layer
    sec = ax.secondary_yaxis("right", functions=(lambda v: 100 * v / n_total,
                                                lambda p: p * n_total / 100))
    sec.spines["right"].set_color(LIGHT)
    sec.spines["right"].set_linewidth(0.5)
    sec.tick_params(axis="y", labelsize=FS_SEC_TICK, colors=GREY, length=1.5, width=0.4, pad=1.2)
    sec.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    sec.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    if last:
        sec.set_ylabel("% of features", fontsize=FS_SEC_LABEL, color=GREY, labelpad=2)
    return ymax



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=float, default=float(os.environ.get("DESWAN_BUCKET_SIZE", 7.5)))
    parser.add_argument("--output-dir", help="Folder with *_window_counts_* and *_main_peaks_* CSVs")
    parser.add_argument("--figure-dir", help="Folder for the exported panel")
    args = parser.parse_args()

    tag = window_tag(args.window)
    sub = f"deswan-window_{args.window:g}"
    output_dir = Path(args.output_dir) if args.output_dir else \
        Path(os.environ.get("DESWAN_OUTPUT_ROOT", SCRIPT_DIR / "outputs")) / sub
    figure_dir = Path(args.figure_dir) if args.figure_dir else \
        Path(os.environ.get("DESWAN_FIGURE_ROOT", SCRIPT_DIR / "figures")) / sub
    figure_dir.mkdir(parents=True, exist_ok=True)

    configure_fonts()
    fig = plt.figure(figsize=(FIG_W_MM * MM, FIG_H_MM * MM))
    source = []
    for i, (key, title, colour) in enumerate(OMICS):
        counts, peaks, n_total = load_omics(output_dir, key, tag)
        ax = fig.add_axes([AX_LEFT_MM[i] / FIG_W_MM, AX_BOTTOM_MM / FIG_H_MM,
                           AX_W_MM / FIG_W_MM, AX_H_MM / FIG_H_MM])
        draw_panel(ax, key, title, colour, counts, peaks, n_total,
                   first=i == 0, last=i == len(OMICS) - 1)
        d = counts[["age", "n_significant", "n_tested"]].copy()
        d.insert(0, "omics", title)
        d["percent_of_features"] = 100 * d["n_significant"] / n_total
        d["is_main_peak"] = d["age"].isin(peaks["peak_age"])
        source.append(d)

    fig.text(LETTER_XY_MM[0] / FIG_W_MM, LETTER_XY_MM[1] / FIG_H_MM, "a", fontsize=FS_LETTER,
             fontweight="bold", ha="left", va="center", color="black")

    stem = figure_dir / f"panel_a_deswan_omics_{tag}"
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.svg")
    fig.savefig(f"{stem}.png", dpi=600)
    plt.close(fig)
    pd.concat(source).to_csv(f"{stem}_source_data.csv", index=False)
    print(f"Saved {stem}.pdf / .svg / .png (600 dpi) and _source_data.csv")


if __name__ == "__main__":
    main()
