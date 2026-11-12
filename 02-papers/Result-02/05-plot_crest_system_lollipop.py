#!/usr/bin/env python3
"""Grouped lollipop plots: proportion of crest features significantly associated
with each clinical indicator, Crest1 vs Crest2, 12 systems in two 6-panel figures.

Input (default): outputs/05-crest_system_association/top10_plot_data.csv written by
05-crest_system_association.py (long format: system, indicator_id, indicator_label,
rank_within_system, crest, proportion in 0-1). A wide table with columns
System, Feature, Crest1, Crest2 (percent, 0-100) is also accepted (--input).

Outputs per figure: <figure-dir>/crest_system_lollipop_<group>.pdf/.svg/.png (300 dpi)
and _source_data.csv. Text stays editable (pdf.fonttype 42, svg.fonttype none).

    python 05-plot_crest_system_lollipop.py
    python 05-plot_crest_system_lollipop.py --input my_table.csv --groups group1

Values above 50% are drawn as an upward triangle at 50%; NA (no valid test) is
written as "NA". Only labels that would reach the next row, the canvas edge or the
y title are split onto two lines (never truncated); the script exits non-zero if
that is still not enough. Panels with fewer than 10 features keep the 10-feature
bar width and dot offset in physical size.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Systems and panel order (row-major: top row left->right, then bottom row)
# ---------------------------------------------------------------------------
SYSTEM_LABELS = {
    "cardio": "Cardiovascular", "body_comp": "Body composition", "bone_density": "Bone density",
    "diet": "Diet", "sleep": "Sleep", "liver_health": "Liver health", "cgm": "Insulin resistance",
    "frailty": "Frailty", "life_style": "Lifestyle", "immune": "Immune system",
    "hema": "Hematopoietic system", "renal_function": "Renal function",
}
GROUPS = {
    "group1": ["Body composition", "Immune system", "Insulin resistance",
               "Cardiovascular", "Bone density", "Hematopoietic system"],
    "group2": ["Diet", "Sleep", "Liver health", "Frailty", "Lifestyle", "Renal function"],
}

# ---------------------------------------------------------------------------
# Layout (inches) and style
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 12.0, 7.0
AX_W, AX_H = 3.52, 1.56
COL_LEFT = [0.78, 4.58, 8.38]
ROW_BOTTOM = [4.76, 1.54]
Y_TITLE_X, Y_TITLE_Y = 0.12, 3.9
LETTER_POS = {0: (-0.105, 1.10), 1: (-0.05, 1.10), 2: (-0.05, 1.10)}
LEGEND_ANCHOR = (0.525, 1.0)
X_OFFSET, BAR_W, DOT_S = 0.17, 0.17, 46
REF_N = 10                   # panels with fewer features keep the 10-feature bar width
YLIM, YTICKS, Y_CAP = (-1.5, 51), [0, 10, 20, 30, 40, 50], 50

# Internal keys match the "crest" values written by 05-crest_system_association.py
# (Crest1/Crest2); CREST_LABELS is the display text used in the legend.
CREST_COLORS = {"Crest1": "#4E89C7", "Crest2": "#F5A05A"}
CREST_LABELS = {"Crest1": "Crest 1",
                "Crest2": "Crest 2"}
BAR, TEXT, NA_GREY = "#C8C8C8", "#222222", "#6B6B6B"
FS_TICK, FS_TITLE, FS_LEGEND, FS_YTITLE, FS_LETTER = 7, 8, 8, 8, 10
CLEARANCE_IN = 0.04          # minimum gap between rotated labels and the next element


def configure_fonts():
    for f in glob.glob(os.path.expanduser("~/.fonts/*.ttf")):
        font_manager.fontManager.addfont(f)
    for family in ("Arial", "Liberation Sans", "Helvetica"):
        try:
            for w in ("normal", "bold"):
                font_manager.findfont(font_manager.FontProperties(family=family, weight=w),
                                      fallback_to_default=False)
            break
        except ValueError:
            continue
    else:
        sys.exit("None of Arial, Liberation Sans or Helvetica is available.")
    if family != "Arial":
        warnings.warn(f"Arial not found; using {family}.")
    plt.rcParams.update({
        "font.family": family, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.unicode_minus": True, "axes.linewidth": 0.6,
        "text.color": TEXT, "axes.edgecolor": TEXT, "axes.labelcolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT,
        "xtick.major.size": 2.5, "xtick.major.width": 0.6,
        "ytick.major.size": 2.5, "ytick.major.width": 0.6,
        "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "figure.autolayout": False,
    })
    return family


# ---------------------------------------------------------------------------
# Data: returns long table System, Feature, order, Crest1, Crest2 (percent)
# ---------------------------------------------------------------------------
def load_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        sys.exit(f"Input not found: {path}\nRun 05-crest_system_association.py first or pass --input.")
    d = pd.read_csv(path, na_values=["NA", ""])
    # Accept "Crest 1"/"Crest 2" as well as "Crest1"/"Crest2" (column names or crest values).
    d = d.rename(columns=lambda c: c.replace("Crest ", "Crest") if str(c).startswith("Crest ") else c)
    if "crest" in d.columns:
        d["crest"] = d["crest"].astype(str).str.replace(r"^Crest\s+", "Crest", regex=True)
    if {"System", "Feature", "Crest1", "Crest2"} <= set(d.columns):          # wide, percent
        out = d[["System", "Feature", "Crest1", "Crest2"]].copy()
        out["System"] = out.System.map(lambda s: SYSTEM_LABELS.get(s, s))
        out["order"] = out.groupby("System", sort=False).cumcount()
        for c in ("Crest1", "Crest2"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out
    need = {"system", "indicator_id", "indicator_label", "rank_within_system", "crest", "proportion"}
    if not need <= set(d.columns):
        sys.exit(f"Input needs System/Feature/Crest1/Crest2 (or 'Crest 1'/'Crest 2') or {sorted(need)}; found {list(d.columns)}")
    d["proportion"] = pd.to_numeric(d.proportion, errors="coerce") * 100       # 0-1 -> percent
    if "selected_top10" in d.columns:
        d = d[d.selected_top10.astype(str).str.lower().isin(["true", "1"])]
    keys = ["system", "indicator_id", "indicator_label", "rank_within_system"]
    if d.duplicated(keys + ["crest"]).any():
        sys.exit("Duplicate system/indicator/crest rows in input.")
    wide = d.set_index(keys + ["crest"]).proportion.unstack("crest")   # keeps NA cells
    wide.columns.name = None
    wide = wide.reset_index().sort_values(["system", "rank_within_system"], kind="stable")
    wide["System"] = wide.system.map(lambda s: SYSTEM_LABELS.get(s, s))
    wide["order"] = wide.groupby("System", sort=False).cumcount()
    for c in ("Crest1", "Crest2"):
        if c not in wide.columns:
            wide[c] = np.nan
    return wide.rename(columns={"indicator_label": "Feature"})[
        ["System", "Feature", "order", "Crest1", "Crest2"]]


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def axes_rect(row, col):
    return [COL_LEFT[col] / FIG_W, ROW_BOTTOM[row] / FIG_H, AX_W / FIG_W, AX_H / FIG_H]


def tidy(label):
    """Display-only clean-up: collapse repeated spaces (original text kept in Source Data)."""
    return re.sub(r"\s+", " ", str(label)).strip()


def two_lines(label):
    """Split at the space closest to the middle (never truncates)."""
    if "\n" in label or " " not in label:
        return label
    mid = len(label) / 2
    cut = min((i for i, ch in enumerate(label) if ch == " "), key=lambda i: abs(i - mid))
    return label[:cut] + "\n" + label[cut + 1:]


def slot_scale(n):
    """Category-unit scale that keeps bar width/offset physically constant (reference: 10 features)."""
    return min((n + 0.2) / (REF_N + 0.2), 1.0) if n else 1.0


def draw(table, systems, display, fixture=False):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    axes, first = [], None
    for k, system in enumerate(systems):
        row, col = divmod(k, 3)
        ax = fig.add_axes(axes_rect(row, col), sharey=first)
        first = first or ax
        axes.append(ax)
        d = table[table.System == system].sort_values("order")
        n = len(d)
        x = np.arange(n, dtype=float)
        sc = slot_scale(n)
        for crest, dx, z in (("Crest1", -X_OFFSET * sc, 3), ("Crest2", X_OFFSET * sc, 4)):
            y = d[crest].to_numpy(float)
            ok = np.isfinite(y)
            yc = np.minimum(y, Y_CAP)
            ax.bar(x[ok] + dx, yc[ok], width=BAR_W * sc, bottom=0, color=BAR, linewidth=0, zorder=1)
            within = ok & (y <= Y_CAP)
            ax.scatter(x[within] + dx, y[within], s=DOT_S, color=CREST_COLORS[crest],
                       linewidths=0, zorder=z, clip_on=False)
            above = ok & (y > Y_CAP)
            if above.any():
                ax.scatter(x[above] + dx, np.full(above.sum(), Y_CAP), s=DOT_S, marker="^",
                           color=CREST_COLORS[crest], linewidths=0, zorder=z, clip_on=False)
            for xi in x[~ok]:
                ax.text(xi + dx, 1.0, "NA", rotation=90, ha="center", va="bottom", fontsize=FS_TICK,
                        color=NA_GREY, zorder=z)
        ax.set_xlim(-0.6, max(n - 0.4, 0.6))
        ax.set_ylim(*YLIM)
        ax.set_yticks(YTICKS)
        ax.set_yticklabels([f"{t}%" for t in YTICKS])
        ax.tick_params(axis="y", labelsize=FS_TICK, labelleft=(col == 0))
        ax.set_xticks(x)
        ax.set_xticklabels([display[(system, f)] for f in d.Feature], rotation=45, ha="right",
                           rotation_mode="anchor", fontsize=FS_TICK, multialignment="right")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_zorder(0)
        ax.set_title(system, loc="left", pad=6, fontsize=FS_TITLE, fontweight="bold", color=TEXT)
        lx, ly = LETTER_POS[col]
        ax.text(lx, ly, "abcdef"[k], transform=ax.transAxes, ha="left", va="bottom",
                fontsize=FS_LETTER, fontweight="bold", color=TEXT)
        if n == 0:
            ax.text(0.5, 0.5, "No indicators with valid tests", transform=ax.transAxes,
                    ha="center", va="center", fontsize=FS_TICK, color=NA_GREY)
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=7, markeredgewidth=0,
                      color=CREST_COLORS[c]) for c in CREST_COLORS]
    fig.legend(handles, [CREST_LABELS[c] for c in CREST_COLORS], loc="upper center",
               bbox_to_anchor=LEGEND_ANCHOR, ncol=2, frameon=False, fontsize=FS_LEGEND,
               columnspacing=2.5, handletextpad=0.4)
    ytitle = fig.text(Y_TITLE_X / FIG_W, Y_TITLE_Y / FIG_H,
                      "Proportion of significantly associated features", rotation=90,
                      ha="center", va="center", fontsize=FS_YTITLE)
    if fixture:
        fig.text(0.995, 0.5, "TEST FIXTURE - not empirical results", rotation=90, ha="right",
                 va="center", fontsize=7, color=NA_GREY)
    return fig, axes, ytitle


def check(fig, axes, ytitle, table, systems):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    dpi = fig.dpi
    tol = CLEARANCE_IN * dpi
    problems = []
    labels = [[t for t in ax.get_xticklabels() if t.get_text()] for ax in axes]
    boxes = [[t.get_window_extent(r) for t in row] for row in labels]
    top_row_bottom = min((b.y0 for bs in boxes[:3] for b in bs), default=np.inf)
    second_row_tops = [ax.title.get_window_extent(r).y1 for ax in axes[3:]]
    second_row_tops += [t.get_window_extent(r).y1 for ax in axes[3:] for t in ax.texts[-1:]]
    if second_row_tops and top_row_bottom - max(second_row_tops) < tol:
        problems.append(f"row-1 labels reach {top_row_bottom / dpi:.2f} in, row-2 titles/letters top at "
                        f"{max(second_row_tops) / dpi:.2f} in")
    for bs in boxes[3:]:
        for b in bs:
            if b.y0 < tol:
                problems.append(f"row-2 label leaves the canvas bottom ({b.y0 / dpi:.2f} in)")
                break
    yt = ytitle.get_window_extent(r)
    for bs in boxes:
        for b in bs:
            if b.x0 < tol or b.overlaps(yt):
                problems.append(f"a rotated label reaches the left edge / y title ({b.x0 / dpi:.2f} in)")
                break
    for ax in (axes[0], axes[3] if len(axes) > 3 else None):
        if ax is not None and ax.texts and ax.texts[-1].get_window_extent(r).overlaps(yt):
            problems.append("y-axis title overlaps a panel letter")
    for ax in axes:
        for coll in ax.collections:
            for xpix, ypix in ax.transData.transform(coll.get_offsets()):
                if ypix - np.sqrt(DOT_S) / 2 * dpi / 72 < ax.spines["bottom"].get_window_extent(r).y0 - 1:
                    problems.append(f"{ax.get_title()}: a 0% point is cut below the axis")
    # Order: panels and features must match the data table.
    for ax, system in zip(axes, systems):
        want = [str(f) for f in table[table.System == system].sort_values("order").Feature]
        want = [tidy(f) for f in want]
        got = [t.get_text().replace("\n", " ") for t in ax.get_xticklabels()]
        if ax.get_title(loc="left") != system or got != want:
            problems.append(f"order mismatch in panel {system}")
    return problems


def offending_labels(fig, axes, ytitle):
    """Labels that reach the next row, the canvas bottom/left edge or the y title."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tol = CLEARANCE_IN * fig.dpi
    limit_row1 = max([ax.title.get_window_extent(r).y1 for ax in axes[3:]] +
                     [ax.texts[-1].get_window_extent(r).y1 for ax in axes[3:] if ax.texts] + [0]) + tol
    yt = ytitle.get_window_extent(r)
    bad = []
    for k, ax in enumerate(axes):
        floor = limit_row1 if k < 3 else tol
        for t in ax.get_xticklabels():
            b = t.get_window_extent(r)
            if t.get_text() and (b.y0 < floor or b.x0 < tol or b.overlaps(yt)):
                bad.append((ax.get_title(loc="left"), t.get_text()))
    return bad


def build(table, group, fixture):
    systems = GROUPS[group]
    missing = [s for s in systems if s not in set(table.System)]
    if missing:
        warnings.warn(f"{group}: systems absent from input, drawn empty: {missing}")
    rows = table[table.System.isin(systems)]
    display = {(s, f): tidy(f) for s, f in zip(rows.System, rows.Feature)}
    original = {(s, tidy(f)): f for s, f in zip(rows.System, rows.Feature)}
    wrapped = []
    for _ in range(3):
        fig, axes, ytitle = draw(table, systems, display, fixture)
        bad = [(s, original[(s, txt)]) for s, txt in offending_labels(fig, axes, ytitle)
               if "\n" not in txt]
        if not bad:
            break
        plt.close(fig)
        for key in bad:
            display[key] = two_lines(display[key])
            wrapped.append(key[1])
    problems = check(fig, axes, ytitle, table, systems)
    # Two-line labels need room between neighbouring 45-degree labels.
    for ax, system in zip(axes, systems):
        n = len(ax.get_xticks())
        if n and any("\n" in t.get_text() for t in ax.get_xticklabels()):
            spacing_pt = AX_W / (n + 0.2) * np.sin(np.pi / 4) * 72
            if spacing_pt < 2 * 1.2 * FS_TICK:
                problems.append(f"{system}: two-line labels too dense ({spacing_pt:.1f} pt spacing)")
    return fig, systems, wrapped, display, problems


def make_figures(input_path: Path, figure_dir: Path | None = None, groups=None,
                 stem: str = "crest_system_lollipop", fixture: bool = False) -> int:
    """Draw every requested group from a plot-data CSV; returns the exit status.

    Shared by main() and 05b-plot_crest_lollipop_from_min_data.py so both draw the same figure.
    """
    input_path = Path(input_path)
    figure_dir = Path(figure_dir) if figure_dir is not None else HERE / "figures" / "crest_system_association"
    groups = list(GROUPS) if groups is None else list(groups)
    family = configure_fonts()
    table = load_table(input_path)
    unknown = sorted(set(table.System) - {s for g in GROUPS.values() for s in g})
    if unknown:
        warnings.warn(f"Systems not in any figure group are ignored: {unknown}")
    figure_dir.mkdir(parents=True, exist_ok=True)
    failed = False
    for group in groups:
        fig, systems, wrapped, display, problems = build(table, group, fixture)
        base = figure_dir / f"{stem}_{group}"
        fig.savefig(f"{base}.pdf")
        fig.savefig(f"{base}.svg")
        fig.savefig(f"{base}.png", dpi=300)
        src = table[table.System.isin(systems)].copy()
        src["panel"] = src.System.map({s: "abcdef"[i] for i, s in enumerate(systems)})
        src["display_label"] = [display.get((s_, f), f).replace("\n", " / ")
                                for s_, f in zip(src.System, src.Feature)]
        src["label_wrapped"] = src.Feature.isin(wrapped)
        src.sort_values(["panel", "order"]).to_csv(f"{base}_source_data.csv", index=False)
        for p in problems:
            warnings.warn(f"{group}: {p}")
        failed |= bool(problems)
        print(f"{group}: {', '.join(systems)}; {len(wrapped)} label(s) wrapped onto two lines"
              f"{': ' + '; '.join(wrapped) if wrapped else ''}; "
              f"font {family}; saved {base}.pdf/.svg/.png")
        plt.close(fig)
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", type=Path,
                    default=HERE / "outputs" / "05-crest_system_association" / "top10_plot_data.csv")
    ap.add_argument("--figure-dir", type=Path, default=HERE / "figures" / "crest_system_association")
    ap.add_argument("--groups", nargs="+", choices=list(GROUPS), default=list(GROUPS))
    ap.add_argument("--stem", default="crest_system_lollipop")
    ap.add_argument("--fixture", action="store_true", help="Mark output as a layout test fixture.")
    a = ap.parse_args()

    return make_figures(a.input, a.figure_dir, a.groups, a.stem, a.fixture)


if __name__ == "__main__":
    sys.exit(main())
