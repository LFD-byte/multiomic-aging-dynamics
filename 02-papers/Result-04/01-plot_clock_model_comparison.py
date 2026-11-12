"""Fig. 6a - Elastic Net vs Spline GAM vs LightGBM aging clocks across five omic layers.

5 columns (omic layers) x 3 rows (MAE, R², Pearson's r) small multiples of outer-CV
test-fold metrics (5-fold x 10 repeats = 50 folds per model and layer).

Canvas is exactly 124 x 64 mm. Every axes is placed in absolute millimetres measured from
the canvas TOP-LEFT corner, and figures are saved WITHOUT bbox_inches="tight", so the
exported size and all font sizes are the true printed sizes.

Inputs (default: ./outputs, written by 01-nonlinearity_model_comparison.py; read as is,
no intermediate files are written)
    nonlinearity_fold_metrics.csv   omics, repeat, fold, model, MAE, R2, PearsonR, ...
        -> only Elastic Net / Spline GAM / LightGBM are plotted

Outputs (default: ./figures/aging_clock_comparison)
    Fig6a_clock_model_comparison.pdf   vector, editable text (Type 42)
    Fig6a_clock_model_comparison.png   600 dpi
    Fig6a_qc_report.txt                font-size / alignment / overlap checks

Usage
    python 01-plot_clock_model_comparison.py [--data-dir DIR] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import glob
import itertools
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter

HERE = Path(__file__).resolve().parent
STEM = "Fig6a_clock_model_comparison"

# --------------------------------------------------------------------------- #
# Font (same approach as Result-01/01-cohort_feature.ipynb)
# --------------------------------------------------------------------------- #
FONT = os.environ.get("FIG_FONT", "Arial")   # override only for local previews
for f in glob.glob(os.path.expanduser("~/.fonts/*.ttf")):
    font_manager.fontManager.addfont(f)
for w in ("normal", "bold"):   # fail loudly instead of silently substituting a font
    font_manager.findfont(font_manager.FontProperties(family=FONT, weight=w),
                          fallback_to_default=False)

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
MM = 1 / 25.4
FIG_W, FIG_H = 124.0, 64.0                       # mm
FS, FS_TICK, FS_SMALL, FS_COL, FS_PANEL = 6.0, 5.5, 5.2, 6.5, 8.0   # pt
C_TEXT, C_TEXT2, C_AXIS, C_REF, C_MED = "#222222", "#666666", "#444444", "#bdbdbd", "#333333"
LW = 0.5

plt.rcParams.update({
    "font.family": FONT,
    "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    "text.color": C_TEXT, "axes.labelcolor": C_TEXT,
    "axes.edgecolor": C_AXIS, "xtick.color": C_AXIS, "ytick.color": C_AXIS,
    "xtick.labelcolor": C_TEXT, "ytick.labelcolor": C_TEXT,
    "axes.linewidth": LW,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.major.width": LW, "ytick.major.width": LW,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "xtick.minor.visible": False, "ytick.minor.visible": False,
    "axes.labelpad": 2.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "axes.facecolor": "white", "figure.facecolor": "white",
    "legend.frameon": False,
    "axes.unicode_minus": True, "mathtext.default": "regular",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.dpi": 600, "savefig.facecolor": "white",
})

# --------------------------------------------------------------------------- #
# Content
# --------------------------------------------------------------------------- #
LAYERS = ["Transcriptomics", "Metabolomics", "Lipidomics", "Gut microbiome", "Oral microbiome"]
COLOR = {"Transcriptomics": "#c45c62", "Metabolomics": "#5a9fd6", "Lipidomics": "#4aa48a",
         "Gut microbiome": "#d99a55", "Oral microbiome": "#8566ab"}
MODELS = ["Elastic Net", "Spline GAM", "LightGBM"]
FILL_ALPHA = {"Elastic Net": 0.85, "Spline GAM": 0.30, "LightGBM": None}   # None = white
MEDIAN_COL = {"Elastic Net": "white", "Spline GAM": C_MED, "LightGBM": C_MED}

# (column, y label, ylim, ticks, zero reference line)
METRICS = [
    ("MAE", "MAE (years)", (4.95, 6.55), [5.0, 5.5, 6.0, 6.5], False),
    ("R2", "R²", (-0.06, 0.40), [0.0, 0.2, 0.4], True),
    ("pearson_r", "Pearson's r", (0.12, 0.65), [0.2, 0.4, 0.6], False),
]

# Layout (mm, origin = canvas top-left)
COL_X0, COL_W, COL_GAP = 12.0, 19.5, 2.6
ROW_Y0, ROW_H, ROW_GAP = 8.5, 12.8, 2.4
TITLE_GAP = 1.2
PANEL_XY = (1.0, 1.0)

# Per-panel drawing
XLIM = (-0.6, 2.75)
BOX_HALF = 0.28
PT_DX, PT_JIT, PT_S, PT_ALPHA = 0.42, 0.07, 0.6, 0.6
LW_BOX, LW_WHISK, LW_MED = 0.6, 0.5, 0.9
OFF_EDGE, OFF_TEXT = 0.07, 0.12    # fraction of the y range


def umin(s: str) -> str:
    return s.replace("-", "−")


# --------------------------------------------------------------------------- #
# Inputs (read directly from the outputs of 01-nonlinearity_model_comparison.py)
# --------------------------------------------------------------------------- #
UPSTREAM_LAYER = {"Transcriptomics": "Transcriptomics", "Metabolomics": "Metabolomics",
                  "Lipidomics": "Lipidomics", "Gut Microbiome": "Gut microbiome",
                  "Oral Microbiome": "Oral microbiome"}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"{path} not found - run 01-nonlinearity_model_comparison.py first")
    return pd.read_csv(path)


def load_inputs(data_dir: Path):
    """Return the fold-level metrics in memory; nothing is written to disk."""
    fm = _read(data_dir / "nonlinearity_fold_metrics.csv")
    unknown = set(fm.omics) - set(UPSTREAM_LAYER)
    if unknown:
        sys.exit(f"unexpected omics values in nonlinearity_fold_metrics.csv: {unknown}")
    fm = fm[fm.model.isin(MODELS)].copy()
    fm["layer"] = fm.omics.map(UPSTREAM_LAYER)
    return fm.rename(columns={"PearsonR": "pearson_r"})


# --------------------------------------------------------------------------- #
# Geometry helpers (mm from top-left)
# --------------------------------------------------------------------------- #
def add_ax_mm(fig, x, y, w, h):
    return fig.add_axes([x / FIG_W, 1 - (y + h) / FIG_H, w / FIG_W, h / FIG_H])


def text_mm(fig, x, y, s, **kw):
    return fig.text(x / FIG_W, 1 - y / FIG_H, s, **kw)


def box_stats(v):
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    lo = v[v >= q1 - 1.5 * iqr].min()
    hi = v[v <= q3 + 1.5 * iqr].max()
    return q1, med, q3, lo, hi


# --------------------------------------------------------------------------- #
# Panel
# --------------------------------------------------------------------------- #
def draw_panel(ax, sub, layer, metric, ylim, rng, log):
    col = COLOR[layer]
    span = ylim[1] - ylim[0]
    for x, model in enumerate(MODELS):
        v = sub.loc[sub.model == model, metric].dropna().to_numpy()
        if len(v) == 0:
            raise ValueError(f"no data for {layer} / {model} / {metric}")
        q1, med, q3, lo, hi = box_stats(v)
        n_out = int(((v < ylim[0]) | (v > ylim[1])).sum())

        if med > ylim[1] or med < ylim[0]:          # off-scale: triangle + value only
            up = med > ylim[1]
            ty = ylim[1] - OFF_EDGE * span if up else ylim[0] + OFF_EDGE * span
            ly = ty - OFF_TEXT * span if up else ty + OFF_TEXT * span
            ax.plot([x], [ty], marker="^" if up else "v", ms=3, color=col, mec=col,
                    mew=0, ls="", zorder=5, clip_on=False, gid="data")
            ax.text(x, ly, umin(f"{med:.2f}"), fontsize=FS_SMALL, color=C_TEXT2,
                    ha="center", va="center", zorder=5, gid="offscale",
                    # white knockout: the R² zero line runs through this position
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
            log.append(f"  off-scale median: {layer} / {model} / {metric} = {med:.3f} "
                       f"(box, whiskers and points not drawn; {n_out}/{len(v)} folds outside ylim)")
            continue
        if n_out:
            log.append(f"  note: {layer} / {model} / {metric}: {n_out}/{len(v)} fold values "
                       f"outside ylim {ylim} (clipped)")

        a = FILL_ALPHA[model]
        face = "white" if a is None else to_rgba(col, a)
        ax.add_patch(Rectangle((x - BOX_HALF, q1), 2 * BOX_HALF, q3 - q1, facecolor=face,
                               edgecolor=col, lw=LW_BOX, zorder=3, gid="data"))
        ax.plot([x, x], [q3, hi], color=col, lw=LW_WHISK, zorder=2, solid_capstyle="butt",
                gid="data")
        ax.plot([x, x], [lo, q1], color=col, lw=LW_WHISK, zorder=2, solid_capstyle="butt",
                gid="data")
        ax.plot([x - BOX_HALF, x + BOX_HALF], [med, med], color=MEDIAN_COL[model], lw=LW_MED,
                solid_capstyle="butt", zorder=4, gid="data")
        ax.scatter(x + PT_DX + rng.uniform(-PT_JIT, PT_JIT, len(v)), v, s=PT_S, color=col,
                   alpha=PT_ALPHA, linewidths=0, edgecolors="none", zorder=2,
                   rasterized=True, gid="data")


def build(fm, rng_seed=0):
    log = []
    fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM))
    rng = np.random.default_rng(rng_seed)
    axes = np.empty((len(METRICS), len(LAYERS)), dtype=object)
    for j, layer in enumerate(LAYERS):
        x0 = COL_X0 + j * (COL_W + COL_GAP)
        text_mm(fig, x0 + COL_W / 2, ROW_Y0 - TITLE_GAP, layer, fontsize=FS_COL,
                fontweight="bold", ha="center", va="baseline", color=C_TEXT)
        sub = fm[fm.layer == layer]
        for i, (metric, ylab, ylim, ticks, zero) in enumerate(METRICS):
            y0 = ROW_Y0 + i * (ROW_H + ROW_GAP)
            ax = add_ax_mm(fig, x0, y0, COL_W, ROW_H)
            axes[i, j] = ax
            if zero:
                ax.axhline(0, color=C_REF, lw=0.5, ls=(0, (2, 1.5)), zorder=1, gid="data")
            draw_panel(ax, sub, layer, metric, ylim, rng, log)
            ax.set_xlim(XLIM)
            ax.set_ylim(ylim)
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: umin(f"{v:.1f}")))
            ax.set_xticks(range(len(MODELS)))
            if i == len(METRICS) - 1:
                ax.set_xticklabels(MODELS, rotation=40, ha="right", rotation_mode="anchor")
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(ylab)
            else:
                ax.set_yticklabels([])
    # align the three y labels of column 1 on one vertical line
    fig.align_ylabels(axes[:, 0])
    # text_mm(fig, *PANEL_XY, "a", fontsize=FS_PANEL, fontweight="bold", ha="left", va="top",
    #         color=C_TEXT)
    return fig, axes, log


def data_bboxes(axes, r):
    """Display-space bboxes of all data marks (gid='data'), clipped to their axes."""
    px2pt = 72 / axes[0].figure.dpi
    out = []
    for ax in axes:
        abox = ax.get_window_extent(r)
        for art in ax.get_children():
            if art.get_gid() != "data":
                continue
            if isinstance(art, matplotlib.collections.PathCollection):
                rad = np.sqrt(PT_S) / 2 / px2pt
                for (px, py) in art.get_offset_transform().transform(art.get_offsets()):
                    bb = matplotlib.transforms.Bbox([[px - rad, py - rad], [px + rad, py + rad]])
                    if bb.overlaps(abox):
                        out.append((ax, bb))
            else:
                bb = art.get_window_extent(r)
                if art.get_clip_on():
                    bb = matplotlib.transforms.Bbox.intersection(bb, abox)
                if bb is not None:
                    out.append((ax, bb))
    return out


# --------------------------------------------------------------------------- #
# QC
# --------------------------------------------------------------------------- #
def text_poly(t, r):
    """Corner polygon (display px) of a text, exact for rotated tick labels."""
    ang = t.get_rotation() % 360
    if ang % 90 == 0 or t.get_rotation_mode() != "anchor":
        b = t.get_window_extent(r)
        return np.array([[b.x0, b.y0], [b.x1, b.y0], [b.x1, b.y1], [b.x0, b.y1]])
    w, h, d = r.get_text_width_height_descent(t.get_text(), t.get_fontproperties(), ismath=False)
    ha, va = t.get_ha(), t.get_va()
    xs = {"left": (0, w), "center": (-w / 2, w / 2), "right": (-w, 0)}[ha]
    ys = {"top": (-h, 0), "bottom": (0, h), "center": (-h / 2, h / 2),
          "baseline": (-d, h - d), "center_baseline": (-h / 2, h / 2)}[va]
    c = np.array([[xs[0], ys[0]], [xs[1], ys[0]], [xs[1], ys[1]], [xs[0], ys[1]]])
    th = np.deg2rad(ang)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    anchor = t.get_transform().transform(t.get_unitless_position())
    return c @ R.T + anchor


def polys_overlap(p, q):
    """Separating-axis test for two convex polygons."""
    for poly in (p, q):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]])
            a, b = p @ n, q @ n
            if a.max() <= b.min() or b.max() <= a.min():
                return False
    return True


def bbox_poly(b):
    return np.array([[b.x0, b.y0], [b.x1, b.y0], [b.x1, b.y1], [b.x0, b.y1]])


def pdf_font_sizes(path):
    """Effective glyph sizes in the PDF (font size x text-matrix scale; correct for rotation)."""
    import math
    from pdfminer.converter import PDFLayoutAnalyzer
    from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
    from pdfminer.pdfpage import PDFPage

    sizes, fonts = [], set()

    class Probe(PDFLayoutAnalyzer):
        def render_char(self, matrix, font, fontsize, scaling, rise, cid, ncs, graphicstate):
            a, b, c, d = matrix[:4]
            sizes.append(fontsize * math.sqrt(abs(a * d - b * c)))
            fonts.add(getattr(font, "fontname", "?"))
            return font.char_width(cid) * fontsize * scaling

    rm = PDFResourceManager()
    dev = Probe(rm)
    with open(path, "rb") as fh:
        for page in PDFPage.get_pages(fh):
            box = page.mediabox
            PDFPageInterpreter(rm, dev).process_page(page)
    return sizes, fonts, box
def _visible_texts(fig):
    out = []
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text().strip() and t.get_figure() is not None:
            out.append(t)
    return out


def qc(fig, axes, pdf_path, log):
    rep = []
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    px2pt = 72 / fig.dpi

    # 1. font sizes (matplotlib objects + glyphs embedded in the PDF)
    sizes = [t.get_fontsize() for t in _visible_texts(fig)]
    rep.append(f"[1] min font size (figure objects): {min(sizes):.2f} pt  (max {max(sizes):.2f})")
    try:
        psz, fonts, box = pdf_font_sizes(pdf_path)
        rep.append(f"    PDF page {box[2] / 72 * 25.4:.2f} x {box[3] / 72 * 25.4:.2f} mm; "
                   f"{len(psz)} glyphs; min {min(psz):.2f} pt, max {max(psz):.2f} pt; "
                   f"sizes used {sorted({round(x, 2) for x in psz})}; fonts {sorted(fonts)}")
        ok1 = min(psz) >= 5.0 - 1e-6
    except ImportError:
        rep.append("    (pdfminer.six not installed: PDF glyph sizes not checked)")
        ok1 = min(sizes) >= 5.0
    rep.append(f"    -> {'PASS' if ok1 else 'FAIL'} (>= 5 pt)")

    # 2. alignment of plotting areas (in pt)
    W, H = fig.get_size_inches() * 72
    pos = np.array([[[a.get_position().x0 * W, a.get_position().x1 * W,
                      a.get_position().y0 * H, a.get_position().y1 * H]
                     for a in row] for row in axes])
    dev_row = max(np.ptp(pos[i, :, 2:], axis=0).max() for i in range(pos.shape[0]))
    dev_col = max(np.ptp(pos[:, j, :2], axis=0).max() for j in range(pos.shape[1]))
    ok2 = max(dev_row, dev_col) <= 1.5
    rep.append(f"[2] max top/bottom deviation within rows: {dev_row:.4f} pt; "
               f"max left/right deviation within columns: {dev_col:.4f} pt "
               f"-> {'PASS' if ok2 else 'FAIL'} (<= 1.5 pt)")

    # 3. overlaps (rotated text uses its true rotated outline)
    texts = _visible_texts(fig)
    tp = [(t, text_poly(t, r)) for t in texts]
    issues = []
    for (t1, p1), (t2, p2) in itertools.combinations(tp, 2):
        if polys_overlap(p1, p2):
            issues.append(f"text/text: {t1.get_text()!r} x {t2.get_text()!r}")
    data = data_bboxes(list(axes.ravel()), r)
    for t, pt in tp:
        hits = sum(polys_overlap(pt, bbox_poly(db)) for _, db in data)
        if hits:
            issues.append(f"text/data: {t.get_text()!r} overlaps {hits} data element(s)")
    # text vs other axes' frames (e.g. rotated x labels running into a neighbouring panel)
    for t, pt in tp:
        for ax in axes.ravel():
            if t.axes is ax or t.axes is None:
                continue
            if polys_overlap(pt, bbox_poly(ax.get_window_extent(r))):
                issues.append(f"text/axes: {t.get_text()!r} enters another panel")
    rep.append(f"[3] overlaps: {len(issues)} " + ("-> PASS" if not issues else "-> CHECK"))
    rep += [f"    {s}" for s in issues]
    tb = [(t, t.get_window_extent(r)) for t in texts]

    # clearance of in-axes annotations
    for t, b in tb:
        if t.axes is not None and t.get_text() not in [l.get_text() for l in t.axes.get_yticklabels()]:
            if t in t.axes.texts:
                d = min((_bbox_gap(b, db) for ax, db in data if ax is t.axes), default=np.inf)
                rep.append(f"    clearance {t.get_text()!r}: {d * px2pt:.2f} pt to nearest data mark")
    rep.append("[data notes]")
    rep += log
    return rep


def _bbox_gap(a, b):
    dx = max(b.x0 - a.x1, a.x0 - b.x1, 0)
    dy = max(b.y0 - a.y1, a.y0 - b.y1, 0)
    return float(np.hypot(dx, dy))


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=HERE / "outputs",
                   help="folder with nonlinearity_fold_metrics.csv")
    p.add_argument("--out-dir", type=Path, default=HERE / "figures" / "aging_clock_comparison")
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    fm = load_inputs(a.data_dir)
    print(f"data: {a.data_dir}")
    bad = set(fm.layer) - set(LAYERS) | set(fm.model) - set(MODELS)
    if bad:
        sys.exit(f"unexpected layer/model values: {bad}")
    n = fm.groupby(["layer", "model"]).size()
    print("folds per layer x model:", sorted(set(n)))

    fig, axes, log = build(fm)
    pdf, png = a.out_dir / f"{STEM}.pdf", a.out_dir / f"{STEM}.png"
    fig.savefig(pdf)                 # no bbox_inches="tight": keep exact 124 x 64 mm
    fig.savefig(png, dpi=600)
    rep = [f"Fig. 6a QC  ({FIG_W:.0f} x {FIG_H:.0f} mm, font {FONT})"] + qc(fig, axes, pdf, log)
    (a.out_dir / "Fig6a_qc_report.txt").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))
    print(f"saved {pdf}\nsaved {png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
