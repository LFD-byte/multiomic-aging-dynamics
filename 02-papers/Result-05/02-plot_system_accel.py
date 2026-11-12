"""Fig. 6d - four-omics AgeAccel vs continuous physiological phenotypes (standardized beta).

Reads the server output of dose_response_forest.py / 02-dose_response_forest.ipynb directly
(no intermediate files):
    outputs/panel_c_dose_response_effects.csv
      (only if it is absent, the legacy name outputs/dose_response_effects.csv is used, with a
       warning on stderr and in the QC report, because that file may be stale)
        system, feature, label, outcome_type, n, effect, ci_low, ci_high, p_value,
        q_value_global, ...
    effect = standardized beta per 1-s.d. higher four-omics AgeAccel from
    OLS  phenotype ~ AgeAccel + age + sex;  q_value_global = BH-FDR over all tests in the CSV.
Selection (never refits or recomputes FDR):
    * no phenotype is excluded here: instrument fields (e.g. DXA scan dose) are removed upstream
      by Result-01/00-filter_system_features.py; only rows without finite effect/CI/P are dropped;
    * each system shows its TOP_N = 10 phenotypes with the smallest P value (all of them if
      it has fewer), smallest P at the top; row pitch is identical in every panel;
    * Fig. 6d: the six prespecified MAIN_SYSTEMS;
      Supplementary: the remaining six systems, same style, no panel letter;
    * "N phenotypes tested" = phenotypes of that system in the CSV.

Canvas is 183 mm wide; its height follows from TOP_N (row pitch ROW_PITCH mm; 82 mm for
TOP_N = 10). Axes are placed in absolute mm from the canvas TOP-LEFT corner and figures
are saved without bbox_inches="tight".

Outputs (default: ./figures/system_accel)
    Fig6d_system_accel.pdf / .png                     main panel d (PNG 600 dpi)
    Fig6d_qc_report.txt
    SuppFig_system_accel_other_systems.pdf / .png     supplementary figure
    SuppFig_system_accel_other_systems_qc_report.txt

Usage
    python 02-plot_system_accel.py [--data-dir DIR] [--out-dir DIR]
                                         [--systems S1 ... S6] [--top-n 10]
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
STEM = "Fig6d_system_accel"
SUPP_STEM = "SuppFig_system_accel_other_systems"

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
# Layout (mm, origin = canvas top-left)
# --------------------------------------------------------------------------- #
MM = 1 / 25.4
FIG_W = 183.0
CELL_W, CELL_GAP, CELL_X0 = 58.3, 2.55, 1.5
CELL_XS = [CELL_X0 + k * (CELL_W + CELL_GAP) for k in range(3)]     # 1.5, 62.35, 123.2
CELL_Y0 = 6.0
LABEL_W, LABEL_PAD = 27.0, 1.6
PLOT_W, PLOT_DY = 27.5, 5.6                                        # plot top = cell top + 5.6
ROW_PITCH, MIN_PLOT_H = 2.5, 12.8      # mm per phenotype row (5.5-pt labels need >= ~2.3)
CELL_VGAP = 6.5                        # plot bottom of row 1 -> cell top of row 2
XTITLE_GAP, BOTTOM_PAD = 5.2, 3.1      # plot bottom of row 2 -> x-title top; title -> edge
TITLE_DX, TITLE_DY, SUB_DY = 1.8, 1.2, 1.4
PANEL_XY = (1.0, 1.5)
LEGEND_RIGHT = 181.5         # legend sits on the x-title row, right edge at x = 181.5 mm
XTITLE_X = FIG_W / 2


def layout(top_n):
    """Plot height, the two cell-row tops, x-title top and canvas height (mm)."""
    plot_h = max(MIN_PLOT_H, ROW_PITCH * top_n)
    y2 = CELL_Y0 + PLOT_DY + plot_h + CELL_VGAP
    xtitle_top = y2 + PLOT_DY + plot_h + XTITLE_GAP
    return {"plot_h": plot_h, "cell_ys": [CELL_Y0, y2], "xtitle_top": xtitle_top,
            "fig_h": xtitle_top + BOTTOM_PAD}


FS, FS_TICK, FS_LEG, FS_SMALL, FS_PANEL = 6.0, 5.5, 5.5, 5.2, 8.0
C_TEXT, C_TEXT2, C_AXIS, C_TICKLAB = "#222222", "#666666", "#777777", "#333333"
C_POS, C_NEG, C_NS, C_REF = "#c0392b", "#2c6ca0", "#bdbdbd", "#9a9a9a"

plt.rcParams.update({
    "font.family": FONT,
    "font.size": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEG,
    "text.color": C_TEXT, "axes.labelcolor": C_TEXT,
    "axes.edgecolor": C_AXIS, "xtick.color": C_AXIS, "ytick.color": C_AXIS,
    "xtick.labelcolor": C_TICKLAB, "ytick.labelcolor": C_TEXT,
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

# --------------------------------------------------------------------------- #
# Content (system keys, labels, order)
# --------------------------------------------------------------------------- #
MAIN_SYSTEMS = ["cardio", "body_comp", "sleep", "diet", "bone_density", "immune"]
SYSTEM_LABELS = {
    "cardio": "Cardiovascular", "body_comp": "Body composition", "sleep": "Sleep",
    "diet": "Diet", "bone_density": "Bone density", "liver_health": "Liver health",
    "cgm": "Glucose regulation", "frailty": "Frailty", "life_style": "Lifestyle",
    "immune": "Immune", "hema": "Hematopoietic", "renal_function": "Renal function",
}
EFFECTS_CSV = "panel_c_dose_response_effects.csv"
LEGACY_EFFECTS_CSV = "dose_response_effects.csv"

# Display-only names for phenotypes whose CSV label does not fit the 27-mm label area at
# 5.5 pt. Keyed by the `feature` code, so they stay correct if the label column changes;
# the CSV and all statistics are untouched.
DISPLAY_LABELS = {
    "standing_one_min_blood_pressure_diastolic": "Standing 1-min diastolic BP",
    "lying_blood_pressure_pulse_rate": "Lying pulse rate",
    "number_of_apnea_hypopnea_events": "Apnoea–hypopnoea events",
    "diet_vegetable_raw_intake_number": "Raw vegetable intake",
}
# Liver elastography "<metric>_<across>_<within>_of_qboxes" (30 features) is renamed by
# rule to "<Metric>, <across> of <within>s", e.g. elasticity_median_median_of_qboxes ->
# "Elasticity, median of medians" (= median across Q-boxes of the per-Q-box medians).
QBOX_WITHIN = {"minimum": "minima", "maximum": "maxima", "mean": "means",
               "median": "medians", "standard_deviation": "SDs", "iqr": "IQRs"}


def qbox_label(feature):
    if not feature.endswith("_of_qboxes"):
        return None
    parts = feature[: -len("_of_qboxes")].split("_", 2)
    if len(parts) != 3 or parts[1] not in {"mean", "median"} or parts[2] not in QBOX_WITHIN:
        return None
    return f"{parts[0].capitalize()}, {parts[1]} of {QBOX_WITHIN[parts[2]]}"


def display_label(row):
    f = str(row.feature)
    return DISPLAY_LABELS.get(f) or qbox_label(f) or str(row.label)


SUPP_SYSTEMS = ["liver_health", "cgm", "frailty", "life_style", "hema", "renal_function"]
TOP_N = 10
XLIM = (-0.2, 0.25)
XTICKS = [-0.2, -0.1, 0.0, 0.1, 0.2]
XTICKLABELS = ["\u22120.2", "\u22120.1", "0", "0.1", "0.2"]
LW_CI, MS, MEW = 0.8, 3.0, 0.7


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def resolve_source(data_dir: Path) -> tuple[Path, bool]:
    """Return (csv path, is_legacy). The current file name always wins."""
    current, legacy = data_dir / EFFECTS_CSV, data_dir / LEGACY_EFFECTS_CSV
    if current.exists():
        return current, False
    if legacy.exists():
        print(f"WARNING: {current.name} not found; falling back to legacy {legacy} "
              "(may be stale - rerun 02-dose_response_forest.ipynb)", file=sys.stderr)
        return legacy, True
    sys.exit(f"{current} not found - run 02-dose_response_forest.ipynb or "
             "dose_response_forest.py first")


def load(path: Path):
    d = pd.read_csv(path)
    need = {"system", "feature", "label", "outcome_type", "n", "effect", "ci_low", "ci_high",
            "p_value", "q_value_global"}
    if miss := need - set(d.columns):
        sys.exit(f"{path} lacks columns {sorted(miss)}")
    n_csv = len(d)
    d = d.copy()
    for c in ("n", "effect", "ci_low", "ci_high", "p_value", "q_value_global"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["effect", "ci_low", "ci_high", "p_value", "q_value_global"])
    return d, n_csv, n_csv - len(d)


def select(d, systems, top_n):
    if miss := set(systems) - set(d.system):
        sys.exit(f"systems not in the CSV: {sorted(miss)}")
    out = {}
    for s in systems:
        part = d[d.system == s].sort_values(["p_value", "feature"])
        top = part.head(top_n).reset_index(drop=True)
        if (top.outcome_type != "continuous").any():
            sys.exit(f"{s}: selected rows include non-continuous outcomes; this panel plots "
                     f"standardized beta only")
        out[s] = (top, len(part))
    return out


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def _fig_h(fig):
    return fig.get_size_inches()[1] * 25.4


def text_mm(fig, x, y, s, **kw):
    return fig.text(x / FIG_W, 1 - y / _fig_h(fig), s, **kw)


def add_ax_mm(fig, x, y, w, h):
    H = _fig_h(fig)
    return fig.add_axes([x / FIG_W, 1 - (y + h) / H, w / FIG_W, h / H])


def style(r):
    if r.q_value_global < 0.05:
        c = C_POS if r.effect > 0 else C_NEG
        return c, c
    return C_NS, "white"


def build(sel, systems, top_n, letter):
    L = layout(top_n)
    fig = plt.figure(figsize=(FIG_W * MM, L["fig_h"] * MM))
    cells, headers, log = [], [], []
    for k, s in enumerate(systems):
        row, col = divmod(k, 3)
        cx, cy = CELL_XS[col], L["cell_ys"][row]
        top, n_tested = sel[s]
        headers.append(text_mm(fig, cx + TITLE_DX, cy + TITLE_DY, SYSTEM_LABELS.get(s, s),
                               fontsize=FS, fontweight="bold", color=C_TEXT,
                               ha="left", va="top"))
        headers.append(text_mm(fig, cx + CELL_W - TITLE_DX, cy + SUB_DY,
                               f"{n_tested:,} phenotypes tested", fontsize=FS_SMALL,
                               color=C_TEXT2, ha="right", va="top"))
        ax = add_ax_mm(fig, cx + LABEL_PAD + LABEL_W, cy + PLOT_DY, PLOT_W, L["plot_h"])
        ax.patch.set_alpha(0)
        ax.axvline(0, color=C_REF, lw=0.5, ls=(0, (2, 1.5)), zorder=0)
        for i, r in top.iterrows():
            lc, fc = style(r)
            ax.plot([r.ci_low, r.ci_high], [i, i], color=lc, lw=LW_CI,
                    solid_capstyle="butt", zorder=2)
            ax.plot([r.effect], [i], ls="", marker="o", ms=MS, mfc=fc, mec=lc, mew=MEW,
                    zorder=3)
            if r.ci_low < XLIM[0] or r.ci_high > XLIM[1]:
                log.append(f"  WARNING {s} / {r.label}: CI {r.ci_low:.3f}-{r.ci_high:.3f} "
                           f"clipped by xlim {XLIM}")
        ax.set_ylim(top_n - 0.5, -0.5)          # same pitch in every panel, rows from the top
        ax.set_xlim(XLIM)
        ax.xaxis.set_major_locator(FixedLocator(XTICKS))
        ax.xaxis.set_major_formatter(FixedFormatter(XTICKLABELS))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.spines["left"].set_visible(False)
        ax.set_yticks(range(len(top)))
        names = [display_label(r) for r in top.itertuples()]
        ax.set_yticklabels(names)
        for r, nm in zip(top.itertuples(), names):
            if nm != str(r.label):
                log.append(f"  display name: {r.label!r} -> {nm!r}")
        ax.tick_params(axis="y", length=0, pad=2)
        cells.append((s, cx, cy, ax, top))

    if letter:
        text_mm(fig, *PANEL_XY, letter, fontsize=FS_PANEL, fontweight="bold", color=C_TEXT,
                ha="left", va="top")
    xtitle = text_mm(fig, XTITLE_X, L["xtitle_top"],
                     "Standardized β and 95% CI", fontsize=FS, color=C_TEXT,
                     ha="center", va="top")
    handles = [
        Line2D([], [], ls="", marker="o", ms=MS, mfc=C_POS, mec=C_POS, mew=MEW,
               label="Positive, BH q < 0.05"),
        Line2D([], [], ls="", marker="o", ms=MS, mfc=C_NEG, mec=C_NEG, mew=MEW,
               label="Negative, BH q < 0.05"),
        Line2D([], [], ls="", marker="o", ms=MS, mfc="white", mec=C_NS, mew=MEW,
               label="q ≥ 0.05"),
    ]
    # bottom row: vertically centred on the x title, right edge at LEGEND_RIGHT
    fig.canvas.draw()
    tb = xtitle.get_window_extent(fig.canvas.get_renderer())
    y_mid = (tb.y0 + tb.y1) / 2 / fig.bbox.height
    leg = fig.legend(handles=handles, loc="center right", ncol=3, frameon=False,
                     bbox_to_anchor=(LEGEND_RIGHT / FIG_W, y_mid),
                     borderaxespad=0, borderpad=0, handletextpad=0.1, columnspacing=1.0,
                     handlelength=1.0, fontsize=FS_LEG)
    for t in leg.get_texts():
        t.set_color(C_TEXT)
    return fig, cells, headers, xtitle, leg, log


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


def mm_box(fig, x0, y0, x1, y1):
    """Display-space Bbox of a canvas rectangle given in mm (top-left origin)."""
    s = fig.dpi / 25.4
    H = fig.bbox.height
    return matplotlib.transforms.Bbox([[x0 * s, H - y1 * s], [x1 * s, H - y0 * s]])


def row_spacing_ok(cells, r, px2mm):
    """Smallest vertical gap between adjacent labels within a panel (mm)."""
    gaps = []
    for *_, ax, _ in cells:
        bs = sorted((t.get_window_extent(r) for t in ax.get_yticklabels()), key=lambda b: -b.y1)
        gaps += [(a.y0 - b.y1) * px2mm for a, b in zip(bs, bs[1:])]
    return min(gaps) if gaps else float("nan")


def qc(fig, cells, headers, xtitle, leg, d, sel, pdf_path, log, n_csv, n_excl):
    rep = []
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    px2pt = 72 / fig.dpi
    px2mm = 25.4 / fig.dpi

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
    pos = np.array([[c[3].get_position().x0 * W, c[3].get_position().x1 * W,
                     c[3].get_position().y0 * H, c[3].get_position().y1 * H] for c in cells])
    pos = pos.reshape(2, 3, 4)
    dev_row = max(np.ptp(pos[i, :, 2:], axis=0).max() for i in range(2))
    dev_w = np.ptp(pos[:, :, 1] - pos[:, :, 0])
    dev_col = max(np.ptp(pos[:, j, :2], axis=0).max() for j in range(3))
    ok2 = max(dev_row, dev_w, dev_col) <= 1.5
    rep.append(f"[2] row top/bottom deviation {dev_row:.4f} pt; width spread {dev_w:.4f} pt; "
               f"column left/right deviation {dev_col:.4f} pt -> {'PASS' if ok2 else 'FAIL'}")

    # 3. phenotype labels inside their 27-mm label area and clear of neighbours
    issues3, widest = [], (0, "")
    all_labels = []
    for s, cx, cy, ax, top in cells:
        area = mm_box(fig, cx + LABEL_PAD, cy, cx + LABEL_PAD + LABEL_W, cy + 60)
        cell = mm_box(fig, cx, cy, cx + CELL_W, cy + 60)
        for t in ax.get_yticklabels():
            b = t.get_window_extent(r)
            all_labels.append((s, t.get_text(), b))
            w = b.width * px2mm
            if w > widest[0]:
                widest = (w, t.get_text())
            if b.x0 < area.x0 - 0.5:
                issues3.append(f"{s}: {t.get_text()!r} ({w:.1f} mm) starts "
                               f"{(area.x0 - b.x0) * px2mm:.2f} mm left of its label area")
            if b.x0 < cell.x0:
                issues3.append(f"{s}: {t.get_text()!r} crosses the cell's left edge")
    for (s1, t1, b1), (s2, t2, b2) in itertools.combinations(all_labels, 2):
        if b1.overlaps(b2):
            issues3.append(f"label overlap: {t1!r} ({s1}) x {t2!r} ({s2})")
    for s, cx, cy, ax, top in cells:   # labels vs other cells' plotting areas
        for s2, t, b in all_labels:
            if s2 != s and b.overlaps(ax.get_window_extent(r)):
                issues3.append(f"{t!r} ({s2}) enters the {s} plot")
    rep.append(f"[3] phenotype labels: {len(issues3)} problem(s) -> "
               + ("PASS" if not issues3 else "CHECK")
               + f"; widest label {widest[1]!r} = {widest[0]:.2f} mm of {LABEL_W:.0f} mm")
    rep += [f"    {x}" for x in issues3]
    rep.append(f"    smallest vertical gap between adjacent labels: "
               f"{row_spacing_ok(cells, r, px2mm):.2f} mm (row pitch {ROW_PITCH} mm)")

    # 4. headers / legend / x title
    items = [(t.get_text(), t.get_window_extent(r)) for t in headers]
    items += [(f"legend: {t.get_text()}", t.get_window_extent(r)) for t in leg.get_texts()]
    for h in leg.legend_handles:
        xd, yd = h.get_xdata(), h.get_ydata()
        cx_, cy_ = h.get_transform().transform((np.mean(xd), np.mean(yd)))
        rad = (h.get_markersize() + h.get_markeredgewidth()) / 2 / px2pt
        items.append((f"legend marker {h.get_label()}",
                      matplotlib.transforms.Bbox([[cx_ - rad, cy_ - rad], [cx_ + rad, cy_ + rad]])))
    items += [("x title", xtitle.get_window_extent(r))]
    items += [("d", t.get_window_extent(r)) for t in fig.texts if t.get_fontsize() == FS_PANEL]
    issues4 = []
    for (n1, b1), (n2, b2) in itertools.combinations(items, 2):
        if b1.overlaps(b2):
            issues4.append(f"{n1!r} x {n2!r}")
    ticklabels = [(f"x tick {t.get_text()}", t.get_window_extent(r))
                  for c in cells for t in c[3].get_xticklabels() if t.get_text()]
    for n, b in items:
        for c in cells:
            if b.overlaps(c[3].get_window_extent(r)):
                issues4.append(f"{n!r} enters the {c[0]} plot")
        for n2, b2 in ticklabels + [(f"label {t}", b3) for _, t, b3 in all_labels]:
            if b.overlaps(b2):
                issues4.append(f"{n!r} x {n2!r}")
    fb = fig.bbox
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_text() and t.get_visible():
            b = t.get_window_extent(r)
            if b.x0 < fb.x0 or b.x1 > fb.x1 or b.y0 < fb.y0 or b.y1 > fb.y1:
                issues4.append(f"{t.get_text()!r} outside the canvas")
    rep.append(f"[4] titles / legend / x title: {len(issues4)} overlap(s) -> "
               + ("PASS" if not issues4 else "CHECK"))
    rep += [f"    {x}" for x in issues4]

    # data summary for the legend
    shown = pd.concat([t for t, _ in sel.values()])
    rep.append(f"[data] CSV rows {n_csv} (BH family of q_value_global), "
               f"{n_excl} rows without finite effect/CI/P dropped -> {len(d)} phenotypes; "
               f"q < 0.05 overall: {int((d.q_value_global < 0.05).sum())}")
    rep.append(f"       shown {len(shown)}: positive sig {int(((shown.q_value_global < .05) & (shown.effect > 0)).sum())}, "
               f"negative sig {int(((shown.q_value_global < .05) & (shown.effect < 0)).sum())}, "
               f"n.s. {int((shown.q_value_global >= .05).sum())}; "
               f"n per test {int(d.n.min()):,}-{int(d.n.max()):,} (all), "
               f"{int(shown.n.min()):,}-{int(shown.n.max()):,} (shown)")
    for s, (top, n_tested) in sel.items():
        rows = "; ".join(f"{r.label} β={r.effect:+.3f} q={r.q_value_global:.2g}"
                         for r in top.itertuples())
        rep.append(f"       {SYSTEM_LABELS.get(s, s)} ({n_tested} tested): {rows}")
    rep.append("[notes]")
    rep += log or ["  none"]
    return rep


# --------------------------------------------------------------------------- #
def render(d, systems, top_n, out_dir, stem, report_name, label, letter, src, n_csv, n_excl):
    sel = select(d, systems, top_n)
    fig, cells, headers, xtitle, leg, log = build(sel, systems, top_n, letter)
    pdf, png = out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"
    fig.savefig(pdf)                  # no bbox_inches="tight": exact mm size
    fig.savefig(png, dpi=600)
    w, h = fig.get_size_inches() * 25.4
    rep = [f"{label} QC  ({w:.1f} x {h:.1f} mm, font {FONT}, source {src})"]
    if Path(src).name == LEGACY_EFFECTS_CSV:
        rep.append(f"WARNING: legacy input {LEGACY_EFFECTS_CSV} used because {EFFECTS_CSV} "
                   "was absent; it may predate the current dose_response_forest run")
    rep += qc(fig, cells, headers, xtitle, leg, d, sel, pdf, log, n_csv, n_excl)
    (out_dir / report_name).write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))
    print(f"saved {pdf}\nsaved {png}\n")
    plt.close(fig)


def make_figures(src: Path, systems=None, top_n: int = TOP_N, out_dir: Path | None = None):
    """Render Fig. 6d and the supplementary figure from an effects CSV.

    Shared by main() and 02b-plot_fig6d_from_min_data.py so both produce the same figure.
    """
    systems = list(MAIN_SYSTEMS if systems is None else systems)
    out_dir = Path(out_dir) if out_dir is not None else HERE / "figures" / "system_accel"
    out_dir.mkdir(parents=True, exist_ok=True)
    d, n_csv, n_excl = load(Path(src))
    render(d, systems, top_n, out_dir, STEM, "Fig6d_qc_report.txt", "Fig. 6d", "d",
           src, n_csv, n_excl)
    rest = [s for s in SUPP_SYSTEMS if s in set(d.system) and s not in systems]
    rest += sorted(set(d.system) - set(systems) - set(rest))
    if len(rest) > 6:
        sys.exit(f"{len(rest)} remaining systems do not fit one 2 x 3 supplementary figure")
    if rest:
        render(d, rest, top_n, out_dir, SUPP_STEM, f"{SUPP_STEM}_qc_report.txt",
               "Supplementary (other systems)", None, src, n_csv, n_excl)
    print(f"systems: {len(systems)} main + {len(rest)} supplementary = "
          f"{len(systems) + len(rest)} of {d.system.nunique()}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=HERE / "outputs",
                   help=f"folder with {EFFECTS_CSV}")
    p.add_argument("--out-dir", type=Path, default=HERE / "figures" / "system_accel")
    p.add_argument("--systems", nargs=6, default=MAIN_SYSTEMS, metavar="SYSTEM",
                   help=f"six main-figure systems, row-major (default: {' '.join(MAIN_SYSTEMS)}); "
                        "all other systems go to the supplementary figure")
    p.add_argument("--top-n", type=int, default=TOP_N)
    a = p.parse_args()
    src, _ = resolve_source(a.data_dir)
    make_figures(src, a.systems, a.top_n, a.out_dir)


if __name__ == "__main__":
    main()
