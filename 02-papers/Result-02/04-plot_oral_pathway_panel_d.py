#!/usr/bin/env python3
"""Panel d: smoothed age profiles of the 48-year oral-crest pathways.

Pathway x age heatmap (one block per age, 41-69 years); dashed line, 48 years.
The mean z (+/- 1.96 x SE across pathways) is still exported in Source Data.

Inputs are the exports of 04-oral_microbiome_pathway.ipynb also used by
04-oral_microbiome_plot_pathway.py:
  <prefix>_eligible_pathway_age_heatmap_sample_level_rpk.csv   (--input)
  <prefix>_eligible_pathway_overview_plot_data.csv             (--pathway-metadata)
with prefix age48_oral_microbiome_window_7p5 by default.

Processing (as in 04-oral_microbiome_plot_pathway.py and the Methods):
log(1 + RPK) per sample -> Gaussian kernel mean (bandwidth 2 years) evaluated
at each integer age 41..69 -> z-score within pathway across the grid.

Outputs (editable text): <stem>.pdf, .svg, .png (600 dpi), _source_data.csv,
_legend_abbreviations.txt.

    python 04-plot_oral_pathway_panel_d.py
    python 04-plot_oral_pathway_panel_d.py --row-order onset
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
MM = 1 / 25.4

# ---------------------------------------------------------------------------
# Layout (mm, origin bottom-left) and style
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 104, 66
# The mean-z trajectory panel is not shown; the heatmap takes over its height
# (top at 58 mm, where the trajectory used to end).
HEAT = (9, 10, 47.6, 48.0)          # left, bottom, width, height
CBAR = (68, 4, 25.6, 1.7)
LABEL_GAP_MM = 1.0
SHOW_TITLE = False  # panel title off; the layer and n go in the figure legend
CANVAS_RIGHT_LIMIT_MM = 103
XLIM = (40.5, 69.5)
XTICKS = [41, 45, 50, 55, 60, 65, 69]
TARGET_AGE, LATE_AGE = 48, 61
PHASES = [("low", 40.5, 48), ("near average", 48, 61), ("elevated", 61, 69.5)]

TEXT, GREY, LIGHT = "#222222", "#6B6B6B", "#D9D9D9"
CMAP = LinearSegmentedColormap.from_list(
    "div", ["#2166AC", "#67A9CF", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#EF8A62", "#B2182B"])
NORM = TwoSlopeNorm(vmin=-2.2, vcenter=0, vmax=2.6)
DASH_48 = dict(color=TEXT, lw=0.7, ls=(0, (3, 1.5)))

FS_LETTER, FS_TITLE, FS_PHASE, FS_TICK = 8, 6.5, 5, 5
FS_ROW, FS_48, FS_AXIS, FS_CBAR_TITLE = 5, 5.4, 5.5, 5.3

# Abbreviations (applied in order); each used one is reported for the legend.
ABBREVIATIONS = [
    ("5-aminoimidazole ribonucleotide", "AIR", "AIR, 5-aminoimidazole ribonucleotide"),
    ("meso-diaminopimelate", "meso-DAP", "meso-DAP, meso-diaminopimelate"),
    ("biosynthesis", "biosynth.", "biosynth., biosynthesis"),
]

# Functional groups for row order: BCAA -> purine/nucleotide precursors ->
# cell wall -> sugar metabolism -> other; UNINTEGRATED always last.
FUNCTION_GROUPS = [
    ("Branched-chain amino acids", r"valine|isoleucine|leucine|branched"),
    ("Purine/nucleotide precursors",
     r"ribonucleotide|ribonucleoside|aminoimidazole|\bAIR\b|purine|inosine|guanosine|adenosine|"
     r"pyrimidine|PRPP"),
    ("Cell wall", r"peptidoglycan|diaminopimelate|meso-DAP|lysine|teichoic|lipid A|"
                  r"O-antigen|UDP-N-acetyl|cell wall|murein"),
    ("Sugar metabolism", r"glycogen|lactose|galactose|glucose|sucrose|starch|glycoly|"
                         r"fructose|mannose|rhamnose|fucose|sugar|trehalose|maltose"),
]


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
        "font.family": family, "mathtext.default": "regular",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.unicode_minus": True, "axes.linewidth": 0.5,
        "text.color": TEXT, "axes.edgecolor": TEXT, "axes.labelcolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT,
        "xtick.major.size": 2, "xtick.major.width": 0.5, "xtick.major.pad": 1.2,
        "ytick.major.size": 2, "ytick.major.width": 0.5, "ytick.major.pad": 1.2,
        "axes.grid": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
    })
    return family


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--microbiome-key", choices=["oral_microbiome", "gut_microbiome"],
                    default="oral_microbiome")
    ap.add_argument("--deswan-window", type=float, default=7.5)
    ap.add_argument("--target-age", type=float, default=48)
    ap.add_argument("--input", type=Path)
    ap.add_argument("--pathway-metadata", type=Path)
    ap.add_argument("--output-dir", type=Path,
                    default=SCRIPT_DIR / "figures" / "microbiome_pathway_enrichment")
    ap.add_argument("--output-stem", default=None)
    ap.add_argument("--bandwidth", type=float, default=2.0)
    ap.add_argument("--row-order", choices=["function", "onset", "metadata"], default="function",
                    help="function: BCAA -> purine -> cell wall -> sugar; onset: first age with z > 1.")
    a = ap.parse_args()
    prefix = (f"age{a.target_age:g}_{a.microbiome_key}_window_"
              f"{a.deswan_window:g}".replace(".", "p"))
    data_dir = SCRIPT_DIR / "outputs" / "microbiome_pathway_enrichment"
    if a.input is None:
        a.input = data_dir / f"{prefix}_eligible_pathway_age_heatmap_sample_level_rpk.csv"
    if a.pathway_metadata is None:
        stem = a.input.name.replace("_eligible_pathway_age_heatmap_sample_level_rpk.csv", "")
        cand = a.input.with_name(f"{stem}_eligible_pathway_overview_plot_data.csv")
        a.pathway_metadata = cand if cand.is_file() else None
    if a.output_stem is None:
        a.output_stem = f"{prefix}_panel_d_pathway_profiles"
    return a


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def smoothed_row_z(path: Path, bandwidth: float, grid: np.ndarray):
    if not path.is_file():
        sys.exit(f"Input not found: {path}\nRun 04-oral_microbiome_pathway.ipynb first or pass --input.")
    d = pd.read_csv(path)
    meta = {"participant_id", "age", "age_bin_index", "_age_bin", "research_stage"}
    if not {"participant_id", "age"} <= set(d.columns):
        sys.exit("Input needs participant_id and age columns.")
    pathways = [c for c in d.columns if c not in meta]
    x = d[pathways].apply(pd.to_numeric, errors="raise")
    if x.isna().any().any() or (x < 0).any().any():
        sys.exit("Pathway RPK must be finite and non-negative (no silent zero-filling).")
    age = pd.to_numeric(d["age"], errors="coerce")
    keep = age.between(40, 70)
    age, logx = age[keep].to_numpy(float), np.log1p(x[keep].to_numpy(float))
    w = np.exp(-0.5 * ((age[None, :] - grid[:, None]) / bandwidth) ** 2)   # grid x samples
    smooth = (w @ logx) / w.sum(axis=1, keepdims=True)                    # grid x pathways
    eff_n = w.sum(axis=1) ** 2 / (w ** 2).sum(axis=1)
    sm = pd.DataFrame(smooth.T, index=pathways, columns=grid)
    sd = sm.std(axis=1, ddof=0).replace(0, np.nan)
    z = sm.sub(sm.mean(axis=1), axis=0).div(sd, axis=0).fillna(0.0)
    return z, sm, eff_n, int(keep.sum()), int(d.loc[keep, "participant_id"].nunique())


def pathway_names(pathways, meta_path):
    names = {p: p for p in pathways}
    rank = {p: i for i, p in enumerate(pathways)}
    if meta_path is not None:
        m = pd.read_csv(meta_path)
        for i, r in enumerate(m.itertuples(index=False)):
            pid = str(r.pathway_id)
            if pid in names:
                nm = str(getattr(r, "pathway_name", "") or "").strip()
                names[pid] = nm if nm and nm.lower() != "nan" else pid
                rank[pid] = i
    return names, rank


def display_name(pid, name, used):
    if pid.upper() == "UNINTEGRATED":
        return "Unintegrated"
    out = name
    for long, short, note in ABBREVIATIONS:
        if re.search(re.escape(long), out, flags=re.I):
            out = re.sub(re.escape(long), short, out, flags=re.I)
            used.append(note)
    out = re.sub(r"\(from ADP-D-Glucose\)", "(from ADP-D-glucose)", out)
    return out[:1].upper() + out[1:] if not re.match(r"^[A-Z]-|^\d", out) else out


def function_group(pid, name):
    if pid.upper() == "UNINTEGRATED":
        return len(FUNCTION_GROUPS) + 1, "Unintegrated"
    for i, (label, pat) in enumerate(FUNCTION_GROUPS):
        if re.search(pat, name, flags=re.I):
            return i, label
    return len(FUNCTION_GROUPS), "Other"


def order_rows(z, names, rank, how):
    pids = list(z.index)
    if how == "metadata":
        return sorted(pids, key=lambda p: rank[p]), {p: "" for p in pids}
    groups = {p: function_group(p, names[p]) for p in pids}
    if how == "function":
        return sorted(pids, key=lambda p: (groups[p][0], rank[p])), {p: groups[p][1] for p in pids}
    grid = z.columns.to_numpy(float)

    def onset(p):
        above = np.flatnonzero(z.loc[p].to_numpy() > 1)
        return grid[above[0]] if above.size else np.inf
    return (sorted(pids, key=lambda p: (p.upper() == "UNINTEGRATED", onset(p), -z.loc[p].max())),
            {p: groups[p][1] for p in pids})


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def add_axes_mm(fig, box):
    l, b, w, h = box
    return fig.add_axes([l / FIG_W, b / FIG_H, w / FIG_W, h / FIG_H])


def draw(z, labels, title):
    grid = z.columns.to_numpy(float)
    n = len(z)
    zmat = z.to_numpy(float)
    mean = zmat.mean(axis=0)
    se = zmat.std(axis=0, ddof=1) / np.sqrt(n)
    lo, hi = mean - 1.96 * se, mean + 1.96 * se

    fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM))
    ax_h = add_axes_mm(fig, HEAT)

    # Heatmap; the dashed line marks 48 years (stated in the legend, not labelled here).
    ax_h.imshow(zmat, cmap=CMAP, norm=NORM, aspect="auto", interpolation="nearest",
                extent=(grid[0] - 0.5, grid[-1] + 0.5, n - 0.5, -0.5))
    ax_h.axvline(TARGET_AGE, zorder=3, **DASH_48)
    if SHOW_TITLE:
        ax_h.set_title(title, fontsize=FS_TITLE, fontweight="normal", pad=4)
    ax_h.set_xlim(*XLIM)
    ax_h.set_ylim(n - 0.5, -0.5)
    ax_h.set_xticks(XTICKS)
    ax_h.set_xticklabels([str(t) for t in XTICKS], fontsize=FS_TICK)
    ax_h.set_xlabel("Age (years)", fontsize=FS_AXIS, labelpad=1)
    ax_h.yaxis.tick_right()
    ax_h.set_yticks(np.arange(n))
    ax_h.set_yticklabels(labels, fontsize=FS_ROW, color=TEXT, ha="left")
    ax_h.tick_params(axis="y", length=0, pad=LABEL_GAP_MM * 72 / 25.4)  # mm -> points
    for s in ax_h.spines.values():
        s.set_visible(False)

    # --- colour bar
    ax_c = add_axes_mm(fig, CBAR)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=ax_c,
                      orientation="horizontal")
    cb.set_ticks([-2, 0, 2])
    cb.set_ticklabels(["−2", "0", "2"])
    ax_c.xaxis.set_ticks_position("top")
    ax_c.tick_params(labelsize=FS_TICK, length=1.5, width=0.3, pad=0.8)
    cb.outline.set_linewidth(0.3)
    cb_title = fig.text((CBAR[0] + CBAR[2] / 2) / FIG_W, (CBAR[1] - 0.6) / FIG_H,
                        "Pathway abundance (row z-score)", ha="center", va="top",
                        fontsize=FS_CBAR_TITLE)

    # fig.text(1 / FIG_W, 1 - 1 / FIG_H, "d", ha="left", va="top", fontsize=FS_LETTER,
    #          fontweight="bold")
    return fig, ax_h, ax_c, cb_title, (mean, se, lo, hi)


def check(fig, ax_h, ax_c, cb_title):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    px_mm = 25.4 / fig.dpi
    problems = []
    labels = ax_h.get_yticklabels()
    cbar_boxes = [ax_c.get_window_extent(r), cb_title.get_window_extent(r)]
    cbar_boxes += [t.get_window_extent(r) for t in ax_c.get_xticklabels() if t.get_text()]
    for t in labels:
        b = t.get_window_extent(r)
        if "\n" in t.get_text():
            problems.append(f"wrapped label: {t.get_text()!r}")
        if b.x1 * px_mm > CANVAS_RIGHT_LIMIT_MM + 0.05:
            problems.append(f"label beyond {CANVAS_RIGHT_LIMIT_MM} mm: {t.get_text()} "
                            f"({b.x1 * px_mm:.1f} mm)")
        if any(b.overlaps(c) for c in cbar_boxes):
            problems.append(f"colour bar overlaps label: {t.get_text()}")
    for t in [cb_title, ax_h.xaxis.label]:
        b = t.get_window_extent(r)
        if b.x0 < 0 or b.y0 < 0 or b.x1 > fig.bbox.x1 or b.y1 > fig.bbox.y1:
            problems.append(f"clipped: {t.get_text()}")
    return problems


def main():
    a = parse_args()
    family = configure_fonts()
    grid = np.arange(41, 70, dtype=float)       # one block per age: 41..69
    z, smooth, eff_n, n_samples, n_participants = smoothed_row_z(a.input, a.bandwidth, grid)
    names, rank = pathway_names(list(z.index), a.pathway_metadata)
    order, groups = order_rows(z, names, rank, a.row_order)
    z = z.loc[order]
    used = []
    labels = [display_name(p, names[p], used) for p in order]
    used = list(dict.fromkeys(used))
    title_layer = "Oral" if a.microbiome_key == "oral_microbiome" else "Gut"
    title = f"{title_layer} microbial MetaCyc pathways (n = {len(z)})"

    fig, ax_h, ax_c, cb_title, (mean, se, lo, hi) = draw(z, labels, title)
    problems = check(fig, ax_h, ax_c, cb_title)
    for p in problems:
        warnings.warn(p)

    a.output_dir.mkdir(parents=True, exist_ok=True)
    base = a.output_dir / a.output_stem
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.svg")
    fig.savefig(f"{base}.png", dpi=600)

    long = z.reset_index(names="pathway_id").melt(id_vars="pathway_id", var_name="age",
                                                  value_name="row_z")
    long["row"] = long.pathway_id.map({p: i + 1 for i, p in enumerate(order)})
    long["pathway_name"] = long.pathway_id.map(names)
    long["display_label"] = long.pathway_id.map(dict(zip(order, labels)))
    long["function_group"] = long.pathway_id.map(groups)
    long["smoothed_mean_log1p_rpk"] = [smooth.loc[p, g] for p, g in zip(long.pathway_id, long.age)]
    summary = pd.DataFrame({"pathway_id": "MEAN_OF_PATHWAYS", "age": grid, "row_z": mean,
                            "se_across_pathways": se, "ci95_low": lo, "ci95_high": hi,
                            "kernel_effective_n": eff_n})
    pd.concat([long, summary], ignore_index=True).to_csv(f"{base}_source_data.csv", index=False)
    with open(f"{base}_legend_abbreviations.txt", "w", encoding="utf-8") as fh:
        fh.write("; ".join(used) + ".\n" if used else "No abbreviations.\n")
        fh.write(f"n = {n_samples} samples from {n_participants} participants aged 40-70 years; "
                 f"Gaussian kernel bandwidth {a.bandwidth:g} years; row order: {a.row_order}.\n")
    print(f"Font: {family}; {len(z)} pathways; {n_samples} samples; saved {base}.pdf/.svg/.png")
    print("Abbreviations: " + "; ".join(used))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
