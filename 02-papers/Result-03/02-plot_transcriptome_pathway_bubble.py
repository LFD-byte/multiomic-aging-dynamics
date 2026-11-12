#!/usr/bin/env python3
"""Figure 4b · pathway enrichment of transcriptomic genes in each trajectory cluster.

Grouped bubble chart drawn from the results of
``02-transcriptome_cluster_pathway_landscape.R``. The R script writes
``outputs/transcriptome_cluster_pathway_landscape/figure_source_data.csv``;
this script converts it into the compact plotting table ``b_enrichment.csv``

    term, database, term_cluster, test_cluster, gene_ratio, fdr

(one row per displayed term × tested cluster C1–C5) and draws the figure.

Usage (headless, e.g. on a remote server)::

    python 02-plot_transcriptome_pathway_bubble.py
    python 02-plot_transcriptome_pathway_bubble.py --r-source path/to/figure_source_data.csv
    python 02-plot_transcriptome_pathway_bubble.py --csv path/to/b_enrichment.csv   # plot only

Outputs: vector PDF (editable text, fonttype 42) and 600-dpi RGB PNG, 88 mm wide,
height fitted to the content; plus ``*_database_note.txt`` listing the source
database of every displayed term for the figure legend.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402
from PIL import Image  # noqa: E402


HERE = Path(__file__).resolve().parent
R_OUTPUT_DIR = HERE / "outputs" / "transcriptome_cluster_pathway_landscape"
DEFAULT_R_SOURCE = R_OUTPUT_DIR / "figure_source_data.csv"
DEFAULT_CSV = R_OUTPUT_DIR / "b_enrichment.csv"
DEFAULT_STEM = (
    HERE / "figures" / "transcriptome_cluster_pathway_landscape"
    / "fig4b_transcriptome_pathway_bubble"
)

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
MM_PER_PT = 25.4 / 72
FIG_W = 88.0                      # mm
CLUSTERS = (1, 2, 3, 4, 5)
CLUSTER_COLORS = {
    1: "#2E3A59", 2: "#7F93AE", 3: "#9AAE8C", 4: "#C49A6C", 5: "#7E5A78",
}
CLUSTER_NAMES = {
    1: "Late accelerating rise", 2: "Gradual rise", 3: "Accelerating decline",
    4: "Midlife trough", 5: "Midlife peak",
}
FDR_THRESHOLD = 0.05
FDR_CMAP = LinearSegmentedColormap.from_list(
    "fdr_reds", ["#FBEAE3", "#F4C3AC", "#E58A6E", "#C9504A", "#9E1B2F"]
)
FDR_MIN, FDR_MAX = 1.3, 20.0      # −log10 FDR colour range (clipped)
FDR_TICKS = (2, 5, 10, 20)

TEXT = "#222222"
MUTED = "#666666"
NOTE = "#777777"

FS = {                            # pt, final print size
    "panel": 8.0,
    "group_id": 5.6,
    "group_name": 5.0,
    "term": 5.3,
    "col_header": 5.3,
    "legend_title": 5.2,
    "legend_label": 5.0,
    "cbar_tick": 4.6,
    "note": 4.8,
}
MIN_TEXT_PT = 4.5

# Layout (mm)
EDGE = 0.8                        # outer margin
BAND_RADIUS = 1.0
BAND_ALPHA = 0.14
GROUP_GAP = 1.3
TITLE_ROW_H = 3.4
ROW_H = 3.1
LABEL_PAD = 2.0                   # band left edge → label
NAME_GAP = 1.0                    # "Cluster 1" → trajectory name
LABEL_TO_COL = 2.2                # label right end → first column edge
COL_W = 8.6
HEADER_GAP = 2.0                  # header text bottom → first band top
LEGEND_GAP = 3.6                  # last band bottom → legend row top
LINE_SPACING = 1.15


def bubble_area(gene_ratio):
    """Marker area in pt² for ``plt.scatter``."""
    return 2.5 + np.asarray(gene_ratio, dtype=float) * 200.0


def bubble_diameter_mm(gene_ratio) -> float:
    return float(np.sqrt(bubble_area(gene_ratio))) * MM_PER_PT


# --------------------------------------------------------------------------- #
# Fonts and text helpers
# --------------------------------------------------------------------------- #
def setup_style() -> str:
    """Register ~/.fonts, pick Arial → Helvetica → Liberation Sans, set rcParams."""
    for f in glob.glob(os.path.expanduser("~/.fonts/*.ttf")):
        font_manager.fontManager.addfont(f)
    family = None
    for candidate in ("Arial", "Helvetica", "Liberation Sans"):
        try:
            for weight in ("normal", "bold"):
                font_manager.findfont(
                    font_manager.FontProperties(family=candidate, weight=weight),
                    fallback_to_default=False,
                )
            family = candidate
            break
        except ValueError:
            continue
    if family is None:
        raise RuntimeError("None of Arial / Helvetica / Liberation Sans is installed")
    mpl.rcParams.update({
        "font.family": family,
        "font.size": FS["term"],
        "text.color": TEXT,
        "axes.edgecolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })
    return family


class TextMeter:
    """Measure rendered text extents in mm with a throw-away Agg canvas."""

    def __init__(self) -> None:
        self.fig = plt.figure(figsize=(4, 4), dpi=600)
        self.renderer = self.fig.canvas.get_renderer()

    def size(self, text: str, size: float, weight: str = "normal",
             linespacing: float = LINE_SPACING) -> tuple[float, float]:
        t = self.fig.text(0, 0, text, fontsize=size, fontweight=weight,
                          linespacing=linespacing)
        box = t.get_window_extent(self.renderer)
        t.remove()
        return box.width / 600 * 25.4, box.height / 600 * 25.4

    def width(self, text: str, size: float, weight: str = "normal") -> float:
        return self.size(text, size, weight)[0]

    def close(self) -> None:
        plt.close(self.fig)


def sentence_case(term: str) -> str:
    """Sentence case that keeps acronyms / mixed-case tokens (MHC, I, tRNA, DNA)."""
    words = str(term).strip().split()
    out = []
    for i, word in enumerate(words):
        keep = any(c.isupper() for c in word[1:]) or (len(word) == 1 and word.isupper() and i > 0)
        if keep:
            out.append(word)
        elif i == 0:
            out.append(word[:1].upper() + word[1:].lower())
        else:
            out.append(word.lower())
    return " ".join(out)


def wrap_to_width(text: str, max_mm: float, meter: TextMeter, size: float) -> str:
    """Greedy word wrap using measured widths."""
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and meter.width(candidate, size) > max_mm:
            lines.append(current)
            current = word
        else:
            current = candidate
    lines.append(current)
    too_wide = [ln for ln in lines if meter.width(ln, size) > max_mm]
    if too_wide:
        raise ValueError(f"A single word does not fit in {max_mm:.1f} mm: {too_wide}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def export_b_enrichment(r_source: Path, out_csv: Path) -> Path:
    """Convert the R ``figure_source_data.csv`` into ``b_enrichment.csv``."""
    raw = pd.read_csv(r_source)
    raw = raw.sort_values(["display_order", "cluster_id"], kind="stable")
    table = pd.DataFrame({
        "term": raw["term"],
        "database": raw["source"],
        "term_cluster": raw["source_cluster"].astype(int),
        "test_cluster": raw["cluster_id"].astype(int),
        "gene_ratio": raw["gene_ratio"].astype(float),
        "fdr": raw["fdr"].astype(float),
    })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({table['term'].nunique()} terms × {table['test_cluster'].nunique()} clusters)")
    return out_csv


def load_b_enrichment(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"term", "database", "term_cluster", "test_cluster", "gene_ratio", "fdr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    df["term_cluster"] = df["term_cluster"].astype(int)
    df["test_cluster"] = df["test_cluster"].astype(int)
    counts = df.groupby(["term_cluster", "term"]).size()
    if not (counts == len(CLUSTERS)).all():
        raise ValueError("Every term must have exactly one row per tested cluster C1–C5")
    if ((df["gene_ratio"] < 0) | (df["gene_ratio"] > 1)).any():
        raise ValueError("gene_ratio must lie in [0, 1]")
    return df


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    png = stem.with_suffix(".png")
    fig.savefig(png, dpi=600)
    plt.close(fig)
    with Image.open(png) as image:
        rgb = image.convert("RGB")
    rgb.save(png, dpi=(600, 600), optimize=True)
    print(f"Saved {stem}.[pdf|png]")


def _check_layout(fig: plt.Figure) -> list[str]:
    """Return problems: text outside the canvas, overlapping or too-small text."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    texts = [t for t in fig.findobj(plt.Text) if t.get_visible() and t.get_text().strip()]
    boxes = [(t.get_text().replace("\n", " ")[:30], t.get_window_extent(renderer), t) for t in texts]
    problems = []
    for label, box, t in boxes:
        if box.x0 < -0.5 or box.y0 < -0.5 or box.x1 > W + 0.5 or box.y1 > H + 0.5:
            problems.append(f"clipped: {label!r}")
        if t.get_fontsize() < MIN_TEXT_PT - 1e-6:
            problems.append(f"< {MIN_TEXT_PT} pt: {label!r} ({t.get_fontsize()} pt)")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                problems.append(f"overlap: {boxes[i][0]!r} × {boxes[j][0]!r}")
    return problems


def plot_pathway_bubble(csv_path: str | Path, out_stem: str | Path = DEFAULT_STEM) -> Path:
    """Draw Figure 4b from ``b_enrichment.csv``; return the output stem."""
    csv_path, out_stem = Path(csv_path), Path(out_stem)
    setup_style()
    df = load_b_enrichment(csv_path)
    meter = TextMeter()

    groups = [c for c in CLUSTERS if (df["term_cluster"] == c).any()]
    empty_clusters = [c for c in CLUSTERS if c not in groups]

    # ---- horizontal layout ------------------------------------------------ #
    band_left = EDGE
    label_x = band_left + LABEL_PAD
    # widest label allowed so that the last column still fits inside 88 mm
    max_label_w = (FIG_W - EDGE) - (len(CLUSTERS) * COL_W) - LABEL_TO_COL - label_x
    rows = []                                    # one entry per displayed term
    for c in groups:
        sub = df[df["term_cluster"] == c]
        for term in dict.fromkeys(sub["term"]):  # keep CSV order
            label = wrap_to_width(sentence_case(term), max_label_w, meter, FS["term"])
            rows.append({"cluster": c, "term": term, "label": label,
                         "n_lines": label.count("\n") + 1,
                         "width": meter.size(label, FS["term"])[0]})
    label_right = label_x + max(r["width"] for r in rows)
    col_x = {c: label_right + LABEL_TO_COL + COL_W / 2 + i * COL_W
             for i, c in enumerate(CLUSTERS)}
    band_right = col_x[CLUSTERS[-1]] + COL_W / 2

    # ---- vertical layout (depth from the top, mm) --------------------------- #
    panel_h = meter.size("b", FS["panel"], "bold")[1]
    header_h = meter.size("Cluster 1", FS["col_header"])[1]
    line_step = FS["term"] * MM_PER_PT * LINE_SPACING
    y = EDGE + max(panel_h, header_h + HEADER_GAP)
    first_band_top = y
    bands = []
    for c in groups:
        top = y
        title_y = top + TITLE_ROW_H / 2
        y = top + TITLE_ROW_H
        for r in (r for r in rows if r["cluster"] == c):
            h = ROW_H + (r["n_lines"] - 1) * line_step
            r["y"] = y + h / 2
            y += h
        bands.append({"cluster": c, "top": top, "bottom": y, "title_y": title_y})
        y += GROUP_GAP
    last_band_bottom = bands[-1]["bottom"]

    legend_ratios = (0.05, 0.10, 0.20)
    max_d = max(bubble_diameter_mm(r) for r in legend_ratios)
    legend_y = last_band_bottom + LEGEND_GAP + max_d / 2
    cbar_h, cbar_w = 1.6, 13.0
    tick_h = meter.size("20", FS["cbar_tick"])[1]
    legend_bottom = max(legend_y + max_d / 2, legend_y + cbar_h / 2 + 0.5 + tick_h)
    note_h = meter.size("Cluster", FS["note"])[1]
    note_y = legend_bottom + 1.2 + note_h / 2 if empty_clusters else legend_bottom
    fig_h = (note_y + note_h / 2 if empty_clusters else legend_bottom) + EDGE

    # ---- canvas: one axes in mm, y pointing down ---------------------------- #
    fig = plt.figure(figsize=(FIG_W / 25.4, fig_h / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(fig_h, 0)
    ax.axis("off")

    # ax.text(EDGE, EDGE, "b", fontsize=FS["panel"], fontweight="bold",
    #         ha="left", va="top")

    for c in CLUSTERS:
        ax.text(col_x[c], first_band_top - HEADER_GAP, f"Cluster {c}",
                fontsize=FS["col_header"], ha="center", va="bottom")

    for band in bands:
        c = band["cluster"]
        ax.add_patch(FancyBboxPatch(
            (band_left, band["top"]), band_right - band_left, band["bottom"] - band["top"],
            boxstyle=f"round,pad=0,rounding_size={BAND_RADIUS}",
            facecolor=CLUSTER_COLORS[c], alpha=BAND_ALPHA, edgecolor="none", zorder=0,
        ))
        ax.text(label_x, band["title_y"], f"Cluster {c}", fontsize=FS["group_id"],
                fontweight="bold", ha="left", va="center")
        id_w = meter.width(f"Cluster {c}", FS["group_id"], "bold")
        ax.text(label_x + id_w + NAME_GAP, band["title_y"], CLUSTER_NAMES[c].lower(),
                fontsize=FS["group_name"], color=MUTED, ha="left", va="center")

    norm = LogNorm(FDR_MIN, FDR_MAX)
    for r in rows:
        ax.text(label_x, r["y"], r["label"], fontsize=FS["term"], ha="left",
                va="center", linespacing=LINE_SPACING)
        cells = df[(df["term_cluster"] == r["cluster"]) & (df["term"] == r["term"])]
        for _, cell in cells.iterrows():
            x = col_x[int(cell["test_cluster"])]
            s = bubble_area(cell["gene_ratio"])
            if cell["fdr"] < FDR_THRESHOLD:
                score = np.clip(-np.log10(max(cell["fdr"], 1e-300)), FDR_MIN, FDR_MAX)
                ax.scatter(x, r["y"], s=s, color=FDR_CMAP(norm(score)),
                           edgecolors="#444444", linewidths=0.3, zorder=3)
            else:
                ax.scatter(x, r["y"], s=s, color="white",
                           edgecolors="#A8A8A8", linewidths=0.4, zorder=3)

    # ---- legend row ------------------------------------------------------- #
    x = band_left
    ax.text(x, legend_y, "Gene ratio", fontsize=FS["legend_title"], ha="left", va="center")
    x += meter.width("Gene ratio", FS["legend_title"]) + 1.5
    for ratio in legend_ratios:
        d = bubble_diameter_mm(ratio)
        ax.scatter(x + d / 2, legend_y, s=bubble_area(ratio), color="#D9D9D9",
                   edgecolors="#666666", linewidths=0.3, zorder=3)
        x += d + 0.7
        label = f"{ratio * 100:.0f}%"
        ax.text(x, legend_y, label, fontsize=FS["legend_label"], ha="left", va="center")
        x += meter.width(label, FS["legend_label"]) + 1.6
    x += 1.6

    # "−log10 FDR" with the subscript drawn at ≥ 4.6 pt rather than mathtext's ~70 %.
    for piece, size, dy in (("−log", FS["legend_title"], 0.0),
                            ("10", FS["cbar_tick"], 0.55),
                            (" FDR", FS["legend_title"], 0.0)):
        ax.text(x, legend_y + dy, piece, fontsize=size, ha="left", va="center")
        x += meter.width(piece, size)
    x += 1.2
    cax = fig.add_axes([x / FIG_W, 1 - (legend_y + cbar_h / 2) / fig_h,
                        cbar_w / FIG_W, cbar_h / fig_h])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=FDR_CMAP), cax=cax,
                        orientation="horizontal")
    cbar.set_ticks(list(FDR_TICKS))
    cbar.set_ticklabels([str(t) for t in FDR_TICKS])
    cax.xaxis.set_minor_locator(NullLocator())
    cax.tick_params(axis="x", which="major", labelsize=FS["cbar_tick"], length=1.0,
                    width=0.3, pad=0.6, color=TEXT, labelcolor=TEXT)
    cbar.outline.set_linewidth(0.3)
    x += cbar_w + 3.0

    d = bubble_diameter_mm(0.10)
    ax.scatter(x + d / 2, legend_y, s=bubble_area(0.10), color="white",
               edgecolors="#A8A8A8", linewidths=0.4, zorder=3)
    x += d + 0.7
    ax.text(x, legend_y, "FDR ≥ 0.05", fontsize=FS["legend_label"], ha="left", va="center")
    x += meter.width("FDR ≥ 0.05", FS["legend_label"])
    if x > FIG_W - EDGE:
        raise ValueError(f"Legend row is {x:.1f} mm wide; exceeds {FIG_W - EDGE:.1f} mm")

    if empty_clusters:
        names = [f"Cluster {c}" for c in empty_clusters]
        who = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        # ax.text(band_left, note_y, f"{who}: no term at FDR < 0.05", fontsize=FS["note"],
        #         color=NOTE, ha="left", va="center")

    meter.close()
    problems = _check_layout(fig)
    for p in problems:
        print("LAYOUT WARNING:", p)
    if not problems:
        print("Layout check passed: no clipped, overlapping or < 4.5 pt text")
    print(f"Figure size: {FIG_W:.0f} × {fig_h:.1f} mm")

    _save(fig, out_stem)
    _write_database_note(df, groups, out_stem)
    return out_stem


def _write_database_note(df: pd.DataFrame, groups: list[int], stem: Path) -> None:
    """Database of each displayed term, for the figure legend (not drawn)."""
    terms = df.drop_duplicates(["term_cluster", "term"])
    lines = ["Source database of the displayed terms (for the figure legend):"]
    for c in groups:
        sub = terms[terms["term_cluster"] == c]
        items = "; ".join(f"{sentence_case(t)} ({db})" for t, db in zip(sub["term"], sub["database"]))
        lines.append(f"Cluster {c}: {items}")
    path = stem.with_name(stem.name + "_database_note.txt")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--r-source", type=Path, default=DEFAULT_R_SOURCE,
                        help="figure_source_data.csv written by the R script")
    parser.add_argument("--csv", type=Path, default=None,
                        help="plot an existing b_enrichment.csv (skips conversion)")
    parser.add_argument("--out", type=Path, default=DEFAULT_STEM, help="output stem")
    args = parser.parse_args()

    csv_path = args.csv or export_b_enrichment(args.r_source, DEFAULT_CSV)
    plot_pathway_bubble(csv_path, args.out)


if __name__ == "__main__":
    main()
