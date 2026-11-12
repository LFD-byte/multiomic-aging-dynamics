#!/usr/bin/env python3
"""Figure 4c · LION lipid-ontology ranked enrichment in each trajectory cluster.

Bubble chart drawn from the LION/web ranking-mode results produced by the
03-* scripts (``outputs/lion_web_five_patterns``). The plotting table is
``c_lion.csv``::

    term, cluster, signed_log10q, pct_annotated, q

one row per LION term × cluster (C1–C5); row order = top-to-bottom order.

Direction (sign of ``signed_log10q``)
    LION/web does not export a direction. It is derived here from the same
    inputs that were submitted to LION/web: for each cluster, the 90 lipids are
    ranked from most to least similar to the cluster pattern (the submitted
    custom statistic, high → low). For each term we compare the empirical
    distribution of its annotated lipids along that ranking with the uniform
    distribution of all lipids (the Kolmogorov–Smirnov running difference):
    if the largest deviation lies at the top of the ranking the sign is "+"
    (enriched among the most similar lipids), otherwise "−" (least similar).
    The q value itself is the unchanged LION/web FDR q.

Usage (headless)::

    python 03-plot_lion_bubble.py                          # build c_lion.csv from LION outputs, plot
    python 03-plot_lion_bubble.py --csv c_lion.csv         # plot an existing table
    python 03-plot_lion_bubble.py --demo                   # demonstration data
    python 03-plot_lion_bubble.py --band-height 54.7       # match the height of Figure 4b

Outputs: vector PDF (editable text, fonttype 42) and 600-dpi RGB PNG.
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
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402
from PIL import Image  # noqa: E402


HERE = Path(__file__).resolve().parent
DEFAULT_LION_ROOT = HERE / "outputs" / "lion_web_five_patterns"
DEFAULT_CSV = DEFAULT_LION_ROOT / "visualization" / "c_lion.csv"
DEFAULT_STEM = HERE / "figures" / "lion_web_five_patterns" / "fig4c_lion_ranked_bubble"

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
MM_PER_PT = 25.4 / 72
FIG_W = 85.0                      # mm (target width)
CLUSTERS = (1, 2, 3, 4, 5)
Q_THRESHOLD = 0.05
CMAP = LinearSegmentedColormap.from_list(   # shared with Figure 4b
    "lion_reds", ["#FBEAE3", "#F4C3AC", "#E58A6E", "#C9504A", "#9E1B2F"]
)
Q_MAX = 3.5                       # |signed −log10 q| colour cap
Q_TICKS = (0, 1, 2, 3)

TEXT = "#222222"
NOTE = "#777777"
BAND_FILL = "#F1F1F1"
EDGE_NS = ("#C8C8C8", 0.3)        # q ≥ 0.05
EDGE_POS = ("#111111", 0.75)      # q < 0.05, most-similar end
EDGE_NEG = ("#2166AC", 0.75)      # q < 0.05, least-similar end

FS = {                            # pt, final print size
    "panel": 8.0,
    "term": 5.3,
    "col_header": 5.3,
    "legend_title": 5.2,
    "legend_label": 5.0,
    "cbar_tick": 4.6,
    "note": 4.8,
}
MIN_TEXT_PT = 4.5

# Layout (mm)
EDGE = 0.8
BAND_HEIGHT = 54.7                # default: equal to the band area of Figure 4b
BAND_RADIUS = 1.0
BAND_TOP_PAD = 0.6
LABEL_PAD = 2.0
LABEL_TO_COL = 2.2
COL_W = 8.6
HEADER_GAP = 2.0
LEGEND_GAP = 3.6
LEGEND_ROW_GAP = 1.4
LINE_SPACING = 1.1


def bubble_area(pct):
    """Marker area in pt² for ``plt.scatter``."""
    return 2.0 + np.asarray(pct, dtype=float) * 0.55


def bubble_diameter_mm(pct) -> float:
    return float(np.sqrt(bubble_area(pct))) * MM_PER_PT


# --------------------------------------------------------------------------- #
# Fonts and text helpers
# --------------------------------------------------------------------------- #
def setup_style() -> str:
    """Register ~/.fonts and use Arial (Helvetica / Liberation Sans as fallback)."""
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
        raise RuntimeError("Arial (or Helvetica / Liberation Sans) is not installed")
    if family != "Arial":
        print(f"NOTE: Arial not found, using {family}")
    mpl.rcParams.update({
        "font.family": family,
        "font.size": FS["term"],
        "text.color": TEXT,
        "xtick.color": TEXT,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })
    return family


class TextMeter:
    """Measure rendered text extents in mm."""

    def __init__(self) -> None:
        self.fig = plt.figure(figsize=(4, 4), dpi=600)
        self.renderer = self.fig.canvas.get_renderer()

    def size(self, text: str, size: float, weight: str = "normal") -> tuple[float, float]:
        t = self.fig.text(0, 0, text, fontsize=size, fontweight=weight,
                          linespacing=LINE_SPACING)
        box = t.get_window_extent(self.renderer)
        t.remove()
        return box.width / 600 * 25.4, box.height / 600 * 25.4

    def width(self, text: str, size: float, weight: str = "normal") -> float:
        return self.size(text, size, weight)[0]

    def close(self) -> None:
        plt.close(self.fig)


def wrap_to_width(text: str, max_mm: float, meter: TextMeter, size: float) -> str:
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and meter.width(candidate, size) > max_mm:
            lines.append(current)
            current = word
        else:
            current = candidate
    lines.append(current)
    if any(meter.width(ln, size) > max_mm for ln in lines):
        raise ValueError(f"A single word of {text!r} does not fit in {max_mm:.1f} mm")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _ks_direction(ranked_names: list[str], annotated: set[str]) -> tuple[int, float, float]:
    """+1 if annotated lipids concentrate at the top of the ranking, −1 if at the bottom."""
    n = len(ranked_names)
    hits = np.array([name in annotated for name in ranked_names], dtype=float)
    k = hits.sum()
    if k == 0 or k == n:
        return 1, 0.0, 0.0
    diff = np.cumsum(hits) / k - np.arange(1, n + 1) / n
    d_top, d_bottom = float(diff.max()), float(-diff.min())
    return (1 if d_top >= d_bottom else -1), d_top, d_bottom


def export_c_lion(lion_root: Path, out_csv: Path) -> tuple[Path, int]:
    """Build ``c_lion.csv`` from the LION/web outputs; return (path, n lipids)."""
    dot = pd.read_csv(lion_root / "visualization" / "lion_dot_plot_source_data.csv")
    dot = dot[dot["selected_for_display"].astype(str).str.lower() == "true"]
    dot = dot.sort_values(["display_order", "pattern_id"], kind="stable")
    membership = pd.read_csv(lion_root / "visualization" / "lion_lipid_term_membership.csv")
    annotated = (
        membership[membership["is_annotated"].astype(str).str.lower() == "true"]
        .groupby("term_id")["standard_name"].apply(set).to_dict()
    )
    manifest = pd.read_csv(lion_root / "lion_web_run_manifest.csv")
    manifest = manifest[manifest["priority"] == "primary"]
    rankings = {}
    for row in manifest.itertuples(index=False):
        if str(row.ranking_direction) != "high_to_low":
            raise ValueError(f"Unexpected ranking direction for pattern {row.pattern_id}")
        ranking = pd.read_csv(lion_root / row.primary_input, sep="\t", header=None,
                              names=["standard_name", "statistic"])
        ranking = ranking.sort_values("statistic", ascending=False, kind="stable")
        rankings[int(row.pattern_id)] = ranking["standard_name"].tolist()

    rows, audit = [], []
    for r in dot.itertuples(index=False):
        sign, d_top, d_bottom = _ks_direction(rankings[int(r.pattern_id)],
                                              annotated.get(r.term_id, set()))
        q = float(r.q_value)
        magnitude = -np.log10(max(q, np.finfo(float).tiny))
        rows.append({
            "term": r.term_label, "cluster": int(r.pattern_id),
            "signed_log10q": sign * magnitude + 0.0,
            "pct_annotated": float(r.annotated_percentage), "q": q,
        })
        audit.append({"term_id": r.term_id, "term": r.term_label, "cluster": int(r.pattern_id),
                      "q": q, "ks_top_deviation": d_top, "ks_bottom_deviation": d_bottom,
                      "direction": "most similar" if sign > 0 else "least similar"})
    table = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    pd.DataFrame(audit).to_csv(out_csv.with_name("c_lion_direction_audit.csv"), index=False)
    n_lipids = int(dot["submitted_lipids"].iloc[0])
    print(f"Wrote {out_csv} ({table['term'].nunique()} terms × {table['cluster'].nunique()} clusters, "
          f"n = {n_lipids} ranked lipids)")
    return out_csv, n_lipids


# Demonstration table: significant (term, cluster, q); every other cell is random.
DEMO_SIGNIFICANT = [
    ("Glycerolipids", 2, 0.000442), ("Triacylglycerols", 2, 0.000442),
    ("Lipid storage", 2, 0.000442), ("Lipid droplet", 2, 0.000442),
    ("Headgroup with negative charge", 3, 0.00112), ("Neutral intrinsic curvature", 3, 0.00112),
    ("Neutral intrinsic curvature", 5, 0.0061), ("Membrane component", 3, 0.00112),
    ("Membrane component", 5, 0.0082), ("Headgroup with neutral charge", 2, 0.0016),
    ("Glycerophospholipids", 3, 0.00185), ("Glycerophospholipids", 5, 0.0061),
    ("Fatty acid with less than 2 double bonds", 3, 0.00194),
    ("Glycerophosphocholines", 3, 0.00418), ("Glycerophosphocholines", 5, 0.0082),
    ("Fatty acids", 3, 0.00807), ("Fatty acids", 4, 0.0222),
]


def make_demo_csv(out_csv: Path, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    terms = list(dict.fromkeys(t for t, _, _ in DEMO_SIGNIFICANT))
    sig = {(t, c): q for t, c, q in DEMO_SIGNIFICANT}
    rows = []
    for term in terms:
        pct = rng.uniform(8, 55)
        for c in CLUSTERS:
            if (term, c) in sig:
                q = sig[(term, c)]
                signed = -np.log10(q)
            else:
                q = rng.uniform(0.15, 1.0)
                signed = rng.choice([-1, 1]) * -np.log10(q)
            rows.append({"term": term, "cluster": c, "signed_log10q": signed,
                         "pct_annotated": pct, "q": q})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote demonstration data {out_csv}")
    return out_csv


def load_c_lion(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"term", "cluster", "signed_log10q", "pct_annotated", "q"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
    df["cluster"] = df["cluster"].astype(int)
    if not (df.groupby("term").size() == len(CLUSTERS)).all():
        raise ValueError("Every term needs exactly one row per cluster C1–C5")
    if not ((df["pct_annotated"] >= 0) & (df["pct_annotated"] <= 100)).all():
        raise ValueError("pct_annotated must lie in [0, 100]")
    return df


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def _check_layout(fig: plt.Figure) -> list[str]:
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
            problems.append(f"< {MIN_TEXT_PT} pt: {label!r}")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                problems.append(f"overlap: {boxes[i][0]!r} × {boxes[j][0]!r}")
    return problems


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


def plot_lion_bubble(
    csv_path: str | Path,
    band_height: float = BAND_HEIGHT,
    out_stem: str | Path = DEFAULT_STEM,
    n_lipids: int = 90,
    fig_width: float = FIG_W,
) -> Path:
    """Draw Figure 4c from ``c_lion.csv``.

    ``band_height`` (mm) is the total height of the grey block; the row height
    is ``(band_height − 0.6) / number of terms`` so that the panel lines up
    with Figure 4b.
    """
    csv_path, out_stem = Path(csv_path), Path(out_stem)
    setup_style()
    df = load_c_lion(csv_path)
    meter = TextMeter()
    terms = list(dict.fromkeys(df["term"]))
    row_h = (band_height - BAND_TOP_PAD) / len(terms)

    # ---- horizontal layout ------------------------------------------------ #
    band_left = EDGE
    label_x = band_left + LABEL_PAD
    max_label_w = (fig_width - EDGE) - len(CLUSTERS) * COL_W - LABEL_TO_COL - label_x
    labels = {t: wrap_to_width(t, max_label_w, meter, FS["term"]) for t in terms}
    for t, lab in labels.items():
        if meter.size(lab, FS["term"])[1] > row_h:
            raise ValueError(f"Label {t!r} needs more than the {row_h:.2f}-mm row height")
    label_right = label_x + max(meter.size(lab, FS["term"])[0] for lab in labels.values())
    col_x = {c: label_right + LABEL_TO_COL + COL_W / 2 + i * COL_W
             for i, c in enumerate(CLUSTERS)}
    band_right = col_x[CLUSTERS[-1]] + COL_W / 2

    # ---- vertical layout (depth from the top, mm) ------------------------- #
    panel_h = meter.size("c", FS["panel"], "bold")[1]
    header_h = meter.size("Cluster 1", FS["col_header"])[1]
    band_top = EDGE + max(panel_h, header_h + HEADER_GAP)
    band_bottom = band_top + band_height
    row_y = {t: band_top + BAND_TOP_PAD + (i + 0.5) * row_h for i, t in enumerate(terms)}

    legend_pcts = (10, 30, 50)
    max_d = max(bubble_diameter_mm(p) for p in legend_pcts)
    cbar_w, cbar_h = 13.0, 1.6
    tick_h = meter.size("0", FS["cbar_tick"])[1]
    row1_y = band_bottom + LEGEND_GAP + max(max_d, cbar_h) / 2
    row1_bottom = max(row1_y + max_d / 2, row1_y + cbar_h / 2 + 0.5 + tick_h)

    # Only explain the blue outline when the data actually contain one.
    has_negative = bool(((df["q"] < Q_THRESHOLD) & (df["signed_log10q"] < 0)).any())
    note = "Blue outline, enriched among least similar lipids" if has_negative else ""
    sig_d = bubble_diameter_mm(30)
    sig_label = "q < 0.05"
    note_x = band_left + sig_d + 0.7 + meter.width(sig_label, FS["legend_label"]) + 2.5
    note_text = wrap_to_width(note, fig_width - EDGE - note_x, meter, FS["note"]) if note else ""
    note_h = meter.size(note_text, FS["note"])[1] if note else 0.0
    row2_top = row1_bottom + LEGEND_ROW_GAP
    row2_y = row2_top + max(sig_d, meter.size(sig_label, FS["legend_label"])[1]) / 2
    fig_h = max(row2_y + sig_d / 2, row2_top + note_h) + EDGE

    # ---- canvas: one axes in mm, y pointing down -------------------------- #
    fig = plt.figure(figsize=(fig_width / 25.4, fig_h / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_width)
    ax.set_ylim(fig_h, 0)
    ax.axis("off")

    # ax.text(EDGE, EDGE, "c", fontsize=FS["panel"], fontweight="bold", ha="left", va="top")
    for c in CLUSTERS:
        ax.text(col_x[c], band_top - HEADER_GAP, f"Cluster {c}",
                fontsize=FS["col_header"], ha="center", va="bottom")
    ax.add_patch(FancyBboxPatch(
        (band_left, band_top), band_right - band_left, band_height,
        boxstyle=f"round,pad=0,rounding_size={BAND_RADIUS}",
        facecolor=BAND_FILL, edgecolor="none", zorder=0,
    ))

    norm = Normalize(0, Q_MAX, clip=True)
    for t in terms:
        y = row_y[t]
        ax.text(label_x, y, labels[t], fontsize=FS["term"], ha="left", va="center",
                linespacing=LINE_SPACING)
        for cell in df[df["term"] == t].itertuples(index=False):
            if cell.q < Q_THRESHOLD:
                edge, lw = EDGE_POS if cell.signed_log10q >= 0 else EDGE_NEG
            else:
                edge, lw = EDGE_NS
            ax.scatter(col_x[int(cell.cluster)], y, s=bubble_area(cell.pct_annotated),
                       color=CMAP(norm(abs(cell.signed_log10q))), edgecolors=edge,
                       linewidths=lw, zorder=3)

    # ---- legend row 1: bubble size + colour bar --------------------------- #
    x = band_left
    ax.text(x, row1_y, "Annotated lipids", fontsize=FS["legend_title"], ha="left", va="center")
    x += meter.width("Annotated lipids", FS["legend_title"]) + 1.5
    for pct in legend_pcts:
        d = bubble_diameter_mm(pct)
        ax.scatter(x + d / 2, row1_y, s=bubble_area(pct), color="#D9D9D9",
                   edgecolors="#666666", linewidths=0.3, zorder=3)
        x += d + 0.7
        label = f"{pct}%"
        ax.text(x, row1_y, label, fontsize=FS["legend_label"], ha="left", va="center")
        x += meter.width(label, FS["legend_label"]) + 1.6
    x += 1.6
    # "−log10 q" with the subscript at 4.6 pt (mathtext would shrink it below 4.5 pt).
    for piece, size, dy in (("−log", FS["legend_title"], 0.0),
                            ("10", FS["cbar_tick"], 0.55),
                            (" q", FS["legend_title"], 0.0)):
        ax.text(x, row1_y + dy, piece, fontsize=size, ha="left", va="center")
        x += meter.width(piece, size)
    x += 1.2
    cax = fig.add_axes([x / fig_width, 1 - (row1_y + cbar_h / 2) / fig_h,
                        cbar_w / fig_width, cbar_h / fig_h])
    cbar = fig.colorbar(ScalarMappable(norm=Normalize(0, Q_MAX), cmap=CMAP), cax=cax,
                        orientation="horizontal")
    cbar.set_ticks(list(Q_TICKS))
    cbar.set_ticklabels([str(t) for t in Q_TICKS])
    cax.xaxis.set_minor_locator(NullLocator())
    cax.tick_params(axis="x", which="major", labelsize=FS["cbar_tick"], length=1.0,
                    width=0.3, pad=0.6, color=TEXT, labelcolor=TEXT)
    cbar.outline.set_linewidth(0.3)
    x += cbar_w + meter.width("3", FS["cbar_tick"]) / 2
    if x > fig_width - EDGE:
        raise ValueError(f"Legend row 1 is {x:.1f} mm wide; exceeds {fig_width - EDGE:.1f} mm")

    # ---- legend row 2: significance outline + note ------------------------ #
    ax.scatter(band_left + sig_d / 2, row2_y, s=bubble_area(30), color="#F4C3AC",
               edgecolors=EDGE_POS[0], linewidths=EDGE_POS[1], zorder=3)
    ax.text(band_left + sig_d + 0.7, row2_y, sig_label, fontsize=FS["legend_label"],
            ha="left", va="center")
    if note_text:
        ax.text(note_x, row2_y, note_text, fontsize=FS["note"], color=NOTE,
                ha="left", va="center", linespacing=LINE_SPACING)

    meter.close()
    problems = _check_layout(fig)
    for p in problems:
        print("LAYOUT WARNING:", p)
    if not problems:
        print("Layout check passed: no clipped, overlapping or < 4.5 pt text")
    print(f"Figure size: {fig_width:.0f} × {fig_h:.1f} mm; row height {row_h:.2f} mm; "
          f"grey block {band_height:.1f} mm")
    _save(fig, out_stem)
    return out_stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lion-root", type=Path, default=DEFAULT_LION_ROOT)
    parser.add_argument("--csv", type=Path, default=None, help="plot an existing c_lion.csv")
    parser.add_argument("--demo", action="store_true", help="use demonstration data")
    parser.add_argument("--band-height", type=float, default=BAND_HEIGHT,
                        help="total height of the grey block (mm)")
    parser.add_argument("--n-lipids", type=int, default=None,
                        help="number of ranked lipids shown in the note (default: from data, else 90)")
    parser.add_argument("--out", type=Path, default=DEFAULT_STEM, help="output stem")
    args = parser.parse_args()

    n_lipids = args.n_lipids or 90
    if args.csv is not None:
        csv_path = args.csv
    elif args.demo:
        csv_path = make_demo_csv(DEFAULT_CSV.with_name("c_lion_demo.csv"))
    else:
        csv_path, n_found = export_c_lion(args.lion_root, DEFAULT_CSV)
        n_lipids = args.n_lipids or n_found
    plot_lion_bubble(csv_path, args.band_height, args.out, n_lipids=n_lipids)


if __name__ == "__main__":
    main()
