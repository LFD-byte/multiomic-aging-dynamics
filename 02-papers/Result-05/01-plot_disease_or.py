"""Fig. 6c - AgeAccel of each clock vs baseline diagnoses (OR per 1-s.d., 95% CI).

Reads the server output of disease_odds_ratio.py directly (no intermediate files):
    outputs/disease_or_results.csv
        outcome, score, cohort, n, n_cases, odds_ratio, or_ci_low, or_ci_high, q_value, ...
    * primary cohort (four_omics_common, n = 1,848): 4-omics stack + 4 single-omic clocks,
      q_value = BH-FDR over the 40 primary tests (8 diagnoses x 5 scores);
    * Metabolomics† (metabolomics_available, n = 1,438): descriptive only, never coloured by q.
Two figures with identical style are written from the same CSV:
  * Fig. 6c (main): diagnoses with a significant four-omics stack association (q < 0.05)
    minus --exclude (default: Hypercholesterolaemia) -> Obesity, Fatty liver,
    Hyperglycaemia, Hypertension; or an explicit --conditions list.
  * Supplementary: every remaining diagnosis (default: Hypercholesterolaemia,
    Hypothyroidism, Cholelithiasis, Asthma); no panel letter; wider x range because some
    CIs reach ~0.68.
Within each figure diagnoses are ordered left -> right by the four-omics stack OR (largest
first). q values are not recomputed: they remain BH-FDR over all 40 primary tests, as in
disease_or_results.csv, so the two figures together show all 40 primary tests.

Canvas is exactly 183 x 46 mm; axes are placed in absolute mm from the canvas TOP-LEFT
corner and figures are saved without bbox_inches="tight".

Outputs (default: ./figures/disease_or)
    Fig6c_disease_or.pdf / .png                    main panel c (PNG 600 dpi)
    Fig6c_qc_report.txt
    SuppFig_disease_or_other_diagnoses.pdf / .png  supplementary figure
    SuppFig_disease_or_other_diagnoses_qc_report.txt

Usage
    python 01-plot_disease_or.py [--data-dir DIR] [--out-dir DIR]
                                 [--conditions NAME [NAME ...]]
                                 [--exclude NAME [NAME ...]]
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
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

HERE = Path(__file__).resolve().parent
STEM = "Fig6c_disease_or"
SUPP_STEM = "SuppFig_disease_or_other_diagnoses"

# --------------------------------------------------------------------------- #
# Font (same approach as Result-01/01-cohort_feature.ipynb)
# --------------------------------------------------------------------------- #
FONT = os.environ.get("FIG_FONT", "Arial")   # override only for local previews
for f in glob.glob(os.path.expanduser("~/.fonts/*.ttf")):
    font_manager.fontManager.addfont(f)
for w in ("normal", "bold"):
    font_manager.findfont(font_manager.FontProperties(family=FONT, weight=w),
                          fallback_to_default=False)

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
MM = 1 / 25.4
FIG_W, FIG_H = 183.0, 46.0
X0, X1, GAP = 25.0, 180.0, 2.6


def panel_width(n):
    return (X1 - X0 - (n - 1) * GAP) / n                   # 4 panels -> 36.8 mm


EXCLUDE = ["Hypercholesterolaemia"]
PY, PH = 15.0, 23.0
NAME_BASE, CASES_BASE = 10.6, 13.7
PANEL_XY = (1.0, 1.5)
XTITLE_X, XTITLE_TOP = (X0 + X1) / 2, 42.4
LEGEND_RIGHT = 180.0          # legend sits on the x-title row, right edge at x = 180 mm

FS, FS_TICK, FS_LEG, FS_SMALL, FS_NAME, FS_PANEL = 6.0, 5.5, 5.5, 5.2, 6.5, 8.0
C_TEXT, C_TEXT2, C_AXIS, C_TICKLAB = "#222222", "#666666", "#777777", "#333333"
C_NS, C_REF = "#bdbdbd", "#9a9a9a"

plt.rcParams.update({
    "font.family": FONT,
    "font.size": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEG,
    "text.color": C_TEXT, "axes.labelcolor": C_TEXT,
    "axes.edgecolor": C_AXIS, "xtick.color": C_AXIS, "ytick.color": C_AXIS,
    "xtick.labelcolor": C_TICKLAB, "ytick.labelcolor": C_TICKLAB,
    "axes.linewidth": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "xtick.minor.visible": False, "ytick.minor.visible": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "axes.facecolor": "white", "figure.facecolor": "white",
    "legend.frameon": False, "axes.unicode_minus": True, "mathtext.default": "regular",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.dpi": 600, "savefig.facecolor": "white",
})

# rows: (key in CSV after renaming, label, colour, y)
STACK = "Four-omics stack"
MET = "Metabolomics"
ROWS = [
    (STACK, "Four-omics stack", "#17605c", 0.0),
    ("Transcriptomics", "Transcriptomics", "#c45c62", 1.0),
    ("Lipidomics", "Lipidomics", "#4aa48a", 2.0),
    ("Gut microbiome", "Gut microbiome", "#d99a55", 3.0),
    ("Oral microbiome", "Oral microbiome", "#8566ab", 4.0),
    (MET, "Metabolomics†", "#5a9fd6", 5.45),
]
SCORE_MAP = {"4-omics stack": STACK, "Transcriptomics": "Transcriptomics",
             "Lipidomics": "Lipidomics", "Gut Microbiome": "Gut microbiome",
             "Oral Microbiome": "Oral microbiome", "Metabolomics†": MET}
PRIMARY_COHORT = "four_omics_common"
YLIM = (5.95, -0.5)
XLIM = (0.78, 2.2)                                         # main figure
XTICKS, XTICKLABELS = [0.8, 1, 1.5, 2], ["0.8", "1", "1.5", "2"]
SUPP_XLIM = (0.64, 2.2)                                    # supplementary (CIs down to ~0.68)
SUPP_XTICKS, SUPP_XTICKLABELS = [0.7, 1, 1.5, 2], ["0.7", "1", "1.5", "2"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"{path} not found - run disease_odds_ratio.py first")
    d = pd.read_csv(path)
    need = {"outcome", "score", "cohort", "n", "n_cases", "odds_ratio", "or_ci_low",
            "or_ci_high", "q_value"}
    if miss := need - set(d.columns):
        sys.exit(f"{path} lacks columns {sorted(miss)}")
    if unknown := set(d.score) - set(SCORE_MAP):
        sys.exit(f"unexpected score labels: {sorted(unknown)}")
    if "fit_status" in d and not d.fit_status.eq("ok").all():
        bad = d.loc[~d.fit_status.eq("ok"), ["outcome", "score", "fit_status"]]
        sys.exit(f"models that did not converge:\n{bad.to_string(index=False)}")
    d = d.rename(columns={"outcome": "condition", "n_cases": "cases", "odds_ratio": "OR",
                          "or_ci_low": "ci_low", "or_ci_high": "ci_high", "q_value": "q"})
    d["score"] = d.score.map(SCORE_MAP)
    return d


def choose_conditions(d, explicit, exclude):
    stack = d[d.score == STACK].set_index("condition")
    if explicit:
        if miss := set(explicit) - set(stack.index):
            sys.exit(f"unknown conditions: {sorted(miss)}")
        sel = stack.loc[explicit]
    else:
        sel = stack[(stack.q < 0.05) & ~stack.index.isin(exclude)]
    order = sel.sort_values("OR", ascending=False).index.tolist()
    if not order:
        sys.exit("no diagnoses selected for the main figure")
    rest = stack.drop(index=order).sort_values("OR", ascending=False).index.tolist()
    return order, rest


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def text_mm(fig, x, y, s, **kw):
    return fig.text(x / FIG_W, 1 - y / FIG_H, s, **kw)


def build(d, conditions, xlim, xticks, xticklabels, letter):
    PW = panel_width(len(conditions))
    fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM))
    axes, titles, log = [], [], []
    for j, cond in enumerate(conditions):
        x = X0 + j * (PW + GAP)
        ax = fig.add_axes([x / FIG_W, 1 - (PY + PH) / FIG_H, PW / FIG_W, PH / FIG_H])
        axes.append(ax)
        sub = d[d.condition == cond].set_index("score")
        prim = sub.loc[STACK]
        titles.append(text_mm(fig, x + PW / 2, NAME_BASE, cond, fontsize=FS_NAME,
                              fontweight="bold", color=C_TEXT, ha="center", va="baseline"))
        titles.append(text_mm(fig, x + PW / 2, CASES_BASE,
                              f"{int(prim.cases):,} cases / {int(prim.n):,}",
                              fontsize=FS_SMALL, color=C_TEXT2, ha="center", va="baseline"))
        ax.axvline(1.0, color=C_REF, lw=0.5, zorder=0)
        for key, _, col, y in ROWS:
            if key not in sub.index:
                sys.exit(f"{cond}: no row for {key}")
            r = sub.loc[key]
            big = key == STACK
            if key == MET:                              # descriptive: always open, blue
                lc, fc = col, "white"
            elif r.q < 0.05:
                lc, fc = col, col
            else:
                lc, fc = C_NS, "white"
            ax.plot([r.ci_low, r.ci_high], [y, y], color=lc, lw=1.0 if big else 0.7,
                    solid_capstyle="butt", zorder=2)
            ax.plot([r.OR], [y], ls="", marker="o", ms=3.8 if big else 3.0, mfc=fc, mec=lc,
                    mew=0.7, zorder=3)
            if r.ci_low < xlim[0] or r.ci_high > xlim[1]:
                log.append(f"  WARNING {cond} / {key}: CI {r.ci_low:.3f}-{r.ci_high:.3f} "
                           f"clipped by xlim {xlim}")
        ax.set_xscale("log")
        ax.set_xlim(xlim)
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        ax.xaxis.set_major_formatter(FixedFormatter(xticklabels))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(YLIM)
        ax.spines["left"].set_visible(False)
        ax.set_yticks([y for *_, y in ROWS])
        ax.tick_params(axis="y", length=0)
        if j == 0:
            ax.set_yticklabels([lab for _, lab, *_ in ROWS])
            for t, (key, _, col, _) in zip(ax.get_yticklabels(), ROWS):
                t.set_color(col)
                if key == STACK:
                    t.set_fontweight("bold")
        else:
            ax.set_yticklabels([])

    if letter:
        text_mm(fig, *PANEL_XY, letter, fontsize=FS_PANEL, fontweight="bold", color=C_TEXT,
                ha="left", va="top")
    xtitle = text_mm(fig, XTITLE_X, XTITLE_TOP,
                     "OR and 95% CI",
                     fontsize=FS, color=C_TEXT, ha="center", va="top")
    handles = [
        Line2D([], [], ls="", marker="o", ms=3.0, mfc=C_TEXT2, mec=C_TEXT2, mew=0.7,
               label="BH q < 0.05"),
        Line2D([], [], ls="", marker="o", ms=3.0, mfc="white", mec=C_NS, mew=0.7,
               label="q ≥ 0.05"),
    ]
    # bottom row: vertically centred on the x title, right edge at LEGEND_RIGHT
    fig.canvas.draw()
    tb = xtitle.get_window_extent(fig.canvas.get_renderer())
    y_mid = (tb.y0 + tb.y1) / 2 / fig.bbox.height
    leg = fig.legend(handles=handles, loc="center right", ncol=2, frameon=False,
                     bbox_to_anchor=(LEGEND_RIGHT / FIG_W, y_mid),
                     borderaxespad=0, borderpad=0, handletextpad=0.1, columnspacing=1.0,
                     handlelength=1.0, fontsize=FS_LEG)
    for t in leg.get_texts():
        t.set_color(C_TEXT)
    met_n = d.loc[d.score == MET, "n"].unique()
    log.append(f"  Metabolomics† cohort n = {', '.join(f'{int(v):,}' for v in met_n)} "
               f"(state in the figure legend; '†' is no longer explained in the panel)")
    return fig, axes, titles, xtitle, leg, log


# --------------------------------------------------------------------------- #
# QC
# --------------------------------------------------------------------------- #
def pdf_font_sizes(path):
    from pdfminer.converter import PDFLayoutAnalyzer
    from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
    from pdfminer.pdfpage import PDFPage
    sizes, fonts = [], set()

    class Probe(PDFLayoutAnalyzer):
        def render_char(self, matrix, font, fontsize, scaling, rise, cid, ncs, graphicstate):
            a, b, c, dd = matrix[:4]
            sizes.append(fontsize * math.sqrt(abs(a * dd - b * c)))
            fonts.add(getattr(font, "fontname", "?"))
            return font.char_width(cid) * fontsize * scaling

    rm = PDFResourceManager()
    dev = Probe(rm)
    with open(path, "rb") as fh:
        for page in PDFPage.get_pages(fh):
            box = page.mediabox
            PDFPageInterpreter(rm, dev).process_page(page)
    return sizes, fonts, box


def qc(fig, axes, titles, xtitle, leg, d, conditions, pdf_path, log):
    rep = []
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    px2pt = 72 / fig.dpi

    # 1. font size
    try:
        psz, fonts, box = pdf_font_sizes(pdf_path)
        rep.append(f"[1] PDF page {box[2] / 72 * 25.4:.2f} x {box[3] / 72 * 25.4:.2f} mm; "
                   f"{len(psz)} glyphs; min {min(psz):.2f} pt, max {max(psz):.2f} pt; "
                   f"sizes {sorted({round(v, 2) for v in psz})}; fonts {sorted(fonts)}")
        ok = min(psz) >= 5.0 - 1e-6
    except ImportError:
        sz = [t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_text()]
        rep.append(f"[1] (pdfminer.six not installed) figure text min {min(sz):.2f} pt")
        ok = min(sz) >= 5.0
    rep.append(f"    -> {'PASS' if ok else 'FAIL'} (>= 5 pt)")

    # 2. alignment
    W, H = fig.get_size_inches() * 72
    pos = np.array([[a.get_position().x0 * W, a.get_position().x1 * W,
                     a.get_position().y0 * H, a.get_position().y1 * H] for a in axes])
    dev_tb = max(np.ptp(pos[:, 2]), np.ptp(pos[:, 3]))
    widths = pos[:, 1] - pos[:, 0]
    gaps = pos[1:, 0] - pos[:-1, 1]
    dev_w, dev_g = np.ptp(widths), np.ptp(gaps)
    ok2 = max(dev_tb, dev_w, dev_g) <= 1.5
    rep.append(f"[2] top/bottom deviation {dev_tb:.4f} pt; width {widths.mean() / 72 * 25.4:.3f} mm "
               f"(spread {dev_w:.4f} pt); gap {gaps.mean() / 72 * 25.4:.3f} mm "
               f"(spread {dev_g:.4f} pt) -> {'PASS' if ok2 else 'FAIL'} (<= 1.5 pt)")

    # 3. overlaps of header / legend / x title / panel letter, and names inside their panel
    items = [(t.get_text(), t.get_window_extent(r)) for t in titles]
    items += [(f"legend: {t.get_text()}", t.get_window_extent(r)) for t in leg.get_texts()]
    for h in leg.legend_handles:        # the drawn marker only (centre +/- ms/2)
        if h.get_marker() in ("", "None", None):
            continue
        xd, yd = h.get_xdata(), h.get_ydata()
        cx, cy = h.get_transform().transform((np.mean(xd), np.mean(yd)))
        rad = (h.get_markersize() + h.get_markeredgewidth()) / 2 / px2pt
        items.append((f"legend marker {h.get_label()}",
                      matplotlib.transforms.Bbox([[cx - rad, cy - rad], [cx + rad, cy + rad]])))
    items += [("x title", xtitle.get_window_extent(r))]
    items += [(t.get_text(), t.get_window_extent(r)) for t in fig.texts
              if t.get_fontsize() == FS_PANEL]
    issues = []
    for (n1, b1), (n2, b2) in itertools.combinations(items, 2):
        if b1.overlaps(b2):
            issues.append(f"{n1!r} x {n2!r}")
    for ax in axes:   # header texts vs plotting areas / tick labels
        abox = ax.get_window_extent(r)
        for n, b in items:
            if b.overlaps(abox):
                issues.append(f"{n!r} enters a plotting area")
        for t in ax.get_xticklabels():   # tick labels vs x title and legend
            if not t.get_text():
                continue
            tbx = t.get_window_extent(r)
            for n, b in items:
                if tbx.overlaps(b):
                    issues.append(f"x tick {t.get_text()!r} x {n!r}")
    fig_box = fig.bbox
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_text() and t.get_visible():
            b = t.get_window_extent(r)
            if b.x0 < fig_box.x0 or b.x1 > fig_box.x1 or b.y0 < fig_box.y0 or b.y1 > fig_box.y1:
                issues.append(f"{t.get_text()!r} outside the canvas")
    rep.append(f"[3] overlaps: {len(issues)} -> " + ("PASS" if not issues else "CHECK"))
    rep += [f"    {s}" for s in issues]
    for j, ax in enumerate(axes):
        abox = ax.get_window_extent(r)
        nb = titles[2 * j].get_window_extent(r)
        margin = min(nb.x0 - abox.x0, abox.x1 - nb.x1) * px2pt
        rep.append(f"    {titles[2 * j].get_text()!r}: width {nb.width * px2pt / 72 * 25.4:.2f} mm "
                   f"of {panel_width(len(axes)):.2f} mm panel; side margin {margin:.2f} pt -> "
                   f"{'inside' if margin >= 0 else 'EXCEEDS PANEL'}")
    hdr = sorted([b for _, b in items if _ != "x title"], key=lambda b: b.y0)

    # 4. significance counts
    prim = d[(d.cohort == PRIMARY_COHORT)]
    shown = prim[prim.condition.isin(conditions)]
    by = shown[shown.q < 0.05].groupby("score").size().reindex(
        [k for k, *_ in ROWS if k != MET], fill_value=0)
    rep.append(f"[4] primary-cohort tests with q < 0.05 in the figure: {int(by.sum())} "
               f"({', '.join(f'{k} {v}' for k, v in by.items())}); "
               f"in all {len(prim)} primary tests: {int((prim.q < 0.05).sum())}")
    met = d[(d.score == MET) & d.condition.isin(conditions)]
    rep.append(f"    Metabolomics† (descriptive, own 8-test FDR family) q < 0.05 in figure: "
               f"{int((met.q < 0.05).sum())} - drawn open regardless")
    rep.append("[order] " + " > ".join(
        f"{c} ({d[(d.condition == c) & (d.score == STACK)].OR.iloc[0]:.3f})" for c in conditions))
    rep.append("[notes]")
    rep += log or ["  none"]
    return rep


# --------------------------------------------------------------------------- #
def render(d, conditions, out_dir, stem, label, xlim, xticks, xticklabels, letter, src):
    fig, axes, titles, xtitle, leg, log = build(d, conditions, xlim, xticks, xticklabels,
                                                letter)
    pdf, png = out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"
    fig.savefig(pdf)                  # no bbox_inches="tight": exact 183 x 46 mm
    fig.savefig(png, dpi=600)
    rep = [f"{label} QC  ({FIG_W:.0f} x {FIG_H:.0f} mm, font {FONT}, source {src})",
           f"[diagnoses] {', '.join(conditions)}"]
    rep += qc(fig, axes, titles, xtitle, leg, d, conditions, pdf, log)
    (out_dir / f"{stem}_qc_report.txt" if stem != STEM else out_dir / "Fig6c_qc_report.txt"
     ).write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))
    print(f"saved {pdf}\nsaved {png}\n")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=HERE / "outputs",
                   help="folder with disease_or_results.csv")
    p.add_argument("--out-dir", type=Path, default=HERE / "figures" / "disease_or")
    p.add_argument("--conditions", nargs="+", default=None,
                   help="explicit diagnoses for the main figure "
                        "(default: four-omics q < 0.05 minus --exclude)")
    p.add_argument("--exclude", nargs="*", default=EXCLUDE,
                   help=f"diagnoses moved from the main to the supplementary figure "
                        f"(default: {EXCLUDE})")
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    src = a.data_dir / "disease_or_results.csv"
    d = load(src)
    main_conds, supp_conds = choose_conditions(d, a.conditions, a.exclude)
    render(d, main_conds, a.out_dir, STEM, "Fig. 6c", XLIM, XTICKS, XTICKLABELS, "c", src)
    if supp_conds:
        render(d, supp_conds, a.out_dir, SUPP_STEM, "Supplementary (other diagnoses)",
               SUPP_XLIM, SUPP_XTICKS, SUPP_XTICKLABELS, None, src)
    shown = len(main_conds) + len(supp_conds)
    print(f"diagnoses: {len(main_conds)} main + {len(supp_conds)} supplementary = {shown} "
          f"of {d.condition.nunique()}")


if __name__ == "__main__":
    main()
