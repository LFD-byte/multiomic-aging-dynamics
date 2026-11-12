"""Fig. 6b - Four-omics stacked clock: chronological vs OOF predicted age (hexbin).

Reads the server output of 02-aging_clock.py
    outputs/panel_a_clock_predictions.parquet
    rows clock_name == "Multi-Omics (4-omics stacking)": one row per participant, pred_age =
    mean of the 10 repeated outer-fold OOF predictions, true_age = chronological age.

Canvas is exactly 54 x 58 mm; the axes is placed in absolute mm from the canvas TOP-LEFT
corner and nothing is saved with bbox_inches="tight", so sizes are true printed sizes.
Not drawn: colour bar, slope label, "y = x" label, panel title and panel letter (these go in
the composite Fig. 6 / legend; the fitted slope is printed in the QC report).

Outputs (default: ./figures/stack_aging_clock)
    Fig6b_four_omics_clock.pdf   vector, editable text; only the hexbin is rasterized
    Fig6b_four_omics_clock.png   600 dpi
    Fig6b_qc_report.txt          font size / overlap / displayed-value checks

Usage
    python 02-plot_fig6b_four_omics_clock.py [--data-dir DIR] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import glob
import itertools
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path as MPath
from matplotlib.ticker import FixedLocator
from scipy import stats

HERE = Path(__file__).resolve().parent
STEM = "Fig6b_four_omics_clock"
CLOCK = "Multi-Omics (4-omics stacking)"
N_REPEATS = 10

# --------------------------------------------------------------------------- #
# Font: Arial / Helvetica, else Liberation Sans (metric-compatible)
# --------------------------------------------------------------------------- #
for f in glob.glob(os.path.expanduser("~/.fonts/*.ttf")):
    font_manager.fontManager.addfont(f)


def _pick_font():
    for fam in ("Arial", "Helvetica", "Liberation Sans"):
        try:
            for w in ("normal", "bold"):
                font_manager.findfont(font_manager.FontProperties(family=fam, weight=w),
                                      fallback_to_default=False)
            return fam
        except ValueError:
            continue
    sys.exit("None of Arial / Helvetica / Liberation Sans is installed")


FONT = _pick_font()

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
MM = 1 / 25.4
FIG_W, FIG_H = 54.0, 58.0                       # mm
AX_X, AX_Y, AX_W, AX_H = 11.0, 8.5, 40.0, 40.0  # mm, top-left origin
FS, FS_TICK = 6.0, 5.5

C_CLOCK = "#17605c"
CMAP = LinearSegmentedColormap.from_list("four_omics",
                                         ["#f5f8f8", "#c5d7d6", "#74a09d", "#3a7a76"])
C_REF, C_TEXT, C_AXIS, C_TICKLAB = "#9a9a9a", "#222222", "#777777", "#333333"
LIM, TICKS = (40, 70), [40, 50, 60, 70]

plt.rcParams.update({
    "font.family": FONT,
    "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    "text.color": C_TEXT, "axes.labelcolor": C_TEXT,
    "axes.edgecolor": C_AXIS, "xtick.color": C_AXIS, "ytick.color": C_AXIS,
    "xtick.labelcolor": C_TICKLAB, "ytick.labelcolor": C_TICKLAB,
    "axes.linewidth": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "xtick.minor.visible": False, "ytick.minor.visible": False,
    "axes.labelpad": 2.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "axes.facecolor": "white", "figure.facecolor": "white",
    "legend.frameon": False, "axes.unicode_minus": True, "mathtext.default": "regular",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.dpi": 600, "savefig.facecolor": "white",
})


def umin(s: str) -> str:
    return s.replace("-", "−")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"{path} not found - run 02-aging_clock.py first")
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    df = df[df.clock_name == CLOCK].copy()
    if df.empty:
        sys.exit(f"no rows with clock_name == {CLOCK!r} in {path}")
    df["participant_id"] = df.participant_id.astype(str)
    if df.participant_id.duplicated().any():
        sys.exit("duplicate participants: expected one repeat-averaged row per participant")
    if "n_repeats" in df and not df.n_repeats.eq(N_REPEATS).all():
        sys.exit(f"n_repeats != {N_REPEATS}: {df.n_repeats.value_counts().to_dict()}")
    return df[["participant_id", "true_age", "pred_age"]].astype(
        {"true_age": float, "pred_age": float})


def compute_stats(x, y):
    lr = stats.linregress(x, y)
    return {"n": len(x),
            "mae": float(np.mean(np.abs(y - x))),
            "r2": float(1 - np.sum((y - x) ** 2) / np.sum((x - x.mean()) ** 2)),
            "r": float(stats.pearsonr(x, y)[0]),
            "slope": float(lr.slope), "intercept": float(lr.intercept)}


def stat_lines(s):
    return [f"n = {s['n']:,}",
            umin(f"MAE = {s['mae']:.2f} years"),
            umin(f"R² = {s['r2']:.3f}"),
            umin(f"Pearson's r = {s['r']:.3f}")]


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def build(df):
    x, y = df.true_age.to_numpy(), df.pred_age.to_numpy()
    s = compute_stats(x, y)
    fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM))
    ax = fig.add_axes([AX_X / FIG_W, 1 - (AX_Y + AX_H) / FIG_H, AX_W / FIG_W, AX_H / FIG_H])

    hb = ax.hexbin(x, y, gridsize=19, extent=(*LIM, *LIM), mincnt=1, vmin=0, cmap=CMAP,
                   edgecolors="white", linewidths=0.15, rasterized=True, zorder=1)
    ref, = ax.plot(LIM, LIM, color=C_REF, lw=0.6, ls=(0, (2, 1.5)), zorder=2)
    xf = np.array(LIM, float)
    fit, = ax.plot(xf, s["intercept"] + s["slope"] * xf, color=C_CLOCK, lw=0.9,
                   solid_capstyle="butt", zorder=3)

    ax.set_xlim(LIM); ax.set_ylim(LIM); ax.set_aspect("equal")
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_locator(FixedLocator(TICKS))
    ax.set_xlabel("Chronological age (years)")
    ax.set_ylabel("OOF predicted age (years)")

    texts = {}
    texts["stats"] = ax.text(41.3, 69.0, "\n".join(stat_lines(s)), va="top", ha="left",
                             fontsize=FS_TICK, color=C_TEXT, linespacing=1.3, zorder=5)

    log = []
    n_out = int(((x < LIM[0]) | (x > LIM[1]) | (y < LIM[0]) | (y > LIM[1])).sum())
    log.append(f"participants outside the 40-70 window (not shown in hexbin, "
               f"included in all statistics): {n_out}")
    return fig, ax, hb, [ref, fit], texts, s, log


# --------------------------------------------------------------------------- #
# Geometry for QC
# --------------------------------------------------------------------------- #
def text_poly(t, r):
    ang = t.get_rotation() % 360
    if ang % 90 == 0 or t.get_rotation_mode() != "anchor":
        b = t.get_window_extent(r)
        return np.array([[b.x0, b.y0], [b.x1, b.y0], [b.x1, b.y1], [b.x0, b.y1]])
    w, h, d = r.get_text_width_height_descent(t.get_text(), t.get_fontproperties(), ismath=False)
    xs = {"left": (0, w), "center": (-w / 2, w / 2), "right": (-w, 0)}[t.get_ha()]
    ys = {"top": (-h, 0), "bottom": (0, h), "center": (-h / 2, h / 2),
          "baseline": (-d, h - d), "center_baseline": (-h / 2, h / 2)}[t.get_va()]
    c = np.array([[xs[0], ys[0]], [xs[1], ys[0]], [xs[1], ys[1]], [xs[0], ys[1]]])
    th = np.deg2rad(ang)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return c @ R.T + t.get_transform().transform(t.get_unitless_position())


def bbox_poly(b):
    return np.array([[b.x0, b.y0], [b.x1, b.y0], [b.x1, b.y1], [b.x0, b.y1]])


def polys_overlap(p, q):
    for poly in (p, q):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]])
            a, b = p @ n, q @ n
            if a.max() <= b.min() or b.max() <= a.min():
                return False
    return True


def hex_polys(ax, hb):
    base = hb.get_paths()[0].vertices[:-1]
    return [ax.transData.transform(base + c) for c in hb.get_offsets()]


def line_hits_poly(ax, line, poly, r, px_per_pt):
    """True if a line (with its stroke width) touches a polygon in display space."""
    xd, yd = line.get_data()
    pts = ax.transData.transform(np.column_stack([np.linspace(xd[0], xd[-1], 4000),
                                                  np.linspace(yd[0], yd[-1], 4000)]))
    rad = line.get_linewidth() / 2 * px_per_pt
    return MPath(np.vstack([poly, poly[:1]])).contains_points(pts, radius=2 * rad).any() or \
        MPath(np.vstack([poly, poly[:1]])).contains_points(pts, radius=-2 * rad).any()


def line_clearance(ax, line, poly, px_per_pt):
    xd, yd = line.get_data()
    pts = ax.transData.transform(np.column_stack([np.linspace(xd[0], xd[-1], 4000),
                                                  np.linspace(yd[0], yd[-1], 4000)]))
    d = np.inf
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        ab = b - a
        tt = np.clip(((pts - a) @ ab) / (ab @ ab), 0, 1)
        d = min(d, np.min(np.linalg.norm(pts - (a + np.outer(tt, ab)), axis=1)))
    inside = MPath(np.vstack([poly, poly[:1]])).contains_points(pts).any()
    return (0.0 if inside else d / px_per_pt) - line.get_linewidth() / 2


# --------------------------------------------------------------------------- #
# QC
# --------------------------------------------------------------------------- #
def pdf_font_sizes(path):
    from pdfminer.converter import PDFLayoutAnalyzer
    from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
    from pdfminer.pdfpage import PDFPage
    sizes, fonts, chars = [], set(), []

    class Probe(PDFLayoutAnalyzer):
        def render_char(self, matrix, font, fontsize, scaling, rise, cid, ncs, graphicstate):
            a, b, c, d = matrix[:4]
            sizes.append(fontsize * math.sqrt(abs(a * d - b * c)))
            fonts.add(getattr(font, "fontname", "?"))
            try:
                chars.append(font.to_unichr(cid))
            except Exception:
                chars.append("?")
            return font.char_width(cid) * fontsize * scaling

    rm = PDFResourceManager()
    dev = Probe(rm)
    with open(path, "rb") as fh:
        for page in PDFPage.get_pages(fh):
            box = page.mediabox
            PDFPageInterpreter(rm, dev).process_page(page)
    return sizes, fonts, "".join(chars), box


def qc(fig, ax, hb, lines, texts, s, pdf_path, log):
    rep = []
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ppt = fig.dpi / 72

    # 1. font sizes
    try:
        psz, fonts, pdf_chars, box = pdf_font_sizes(pdf_path)
        rep.append(f"[1] PDF page {box[2] / 72 * 25.4:.2f} x {box[3] / 72 * 25.4:.2f} mm; "
                   f"{len(psz)} glyphs; min {min(psz):.2f} pt, max {max(psz):.2f} pt; "
                   f"sizes {sorted({round(v, 2) for v in psz})}; fonts {sorted(fonts)}")
        ok = min(psz) >= 5.0 - 1e-6
    except ImportError:
        pdf_chars = None
        sz = [t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_text()]
        rep.append(f"[1] (pdfminer.six not installed) figure text min {min(sz):.2f} pt")
        ok = min(sz) >= 5.0
    rep.append(f"    -> {'PASS' if ok else 'FAIL'} (>= 5 pt)")

    # 2. annotation overlaps
    items = [(k, text_poly(t, r)) for k, t in texts.items()]
    issues, notes = [], []
    for (n1, p1), (n2, p2) in itertools.combinations(items, 2):
        if polys_overlap(p1, p2):
            issues.append(f"{n1} x {n2}")
    for name, p in items:
        for ln, lname in zip(lines, ["y = x reference line", "fit line"]):
            c = line_clearance(ax, ln, p, ppt)
            if c <= 0:
                issues.append(f"{name} x {lname}")
            notes.append(f"    clearance {name} -> {lname}: {c:.2f} pt")
    hexes = hex_polys(ax, hb)
    for name, p in items:
        k = sum(polys_overlap(p, h) for h in hexes)
        if k:
            notes.append(f"    info: {name} lies over {k} hexagon(s)")
    rep.append(f"[2] overlaps among labels / lines: {len(issues)} -> "
               + ("PASS" if not issues else "CHECK"))
    rep += [f"    {i}" for i in issues] + notes

    # 3. displayed values vs computed values
    shown = stat_lines(s)
    rep.append(f"[3] computed from data: n = {s['n']}, MAE = {s['mae']:.6f} years, "
               f"R² = {s['r2']:.6f}, Pearson's r = {s['r']:.6f}, slope = {s['slope']:.6f}, "
               f"intercept = {s['intercept']:.4f}")
    fig_txt = texts["stats"].get_text().splitlines()
    ok3 = fig_txt == shown
    if pdf_chars is not None:
        flat = pdf_chars.replace(" ", "")
        ok3 &= all(line.replace(" ", "") in flat for line in shown)
    rep.append("    shown: " + " | ".join(fig_txt) + f"  -> {'PASS' if ok3 else 'FAIL'}"
               + (" (figure objects and PDF text)" if pdf_chars is not None else ""))
    rep.append("[notes]")
    rep += [f"    {l}" for l in log]
    return rep


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=HERE / "outputs",
                   help="folder with panel_a_clock_predictions.parquet")
    p.add_argument("--pred-file", type=Path, default=None,
                   help="override the predictions file (parquet, or csv with the same columns)")
    p.add_argument("--out-dir", type=Path, default=HERE / "figures" / "stack_aging_clock")
    a = p.parse_args()
    src = a.pred_file or a.data_dir / "panel_a_clock_predictions.parquet"
    a.out_dir.mkdir(parents=True, exist_ok=True)

    df = load(src)
    fig, ax, hb, lines, texts, s, log = build(df)
    out = {e: a.out_dir / f"{STEM}.{e}" for e in ("pdf", "png")}
    fig.savefig(out["pdf"])                    # no bbox_inches="tight": exact 54 x 58 mm
    fig.savefig(out["png"], dpi=600)
    rep = [f"Fig. 6b QC  ({FIG_W:.0f} x {FIG_H:.0f} mm, font {FONT}, source {src})"]
    rep += qc(fig, ax, hb, lines, texts, s, out["pdf"], log)
    (a.out_dir / "Fig6b_qc_report.txt").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))
    for f in out.values():
        print(f"saved {f}")
    plt.close(fig)


if __name__ == "__main__":
    main()
