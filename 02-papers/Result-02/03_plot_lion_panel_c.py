#!/usr/bin/env python3
"""Panel c: LION lipid classes at the 48-year lipidomic peak (lollipop).

Input: the LION/web two-sided ranking result downloaded after running
03_prepare_lion_custom_statistics.R (default:
outputs/lipid_pathway/lion_age_48_two_sided_result.csv), or any table with a
term column, a direction (UP/DOWN) or signed column, and a q-value column.
Only terms with q < --fdr are drawn. signed = sign x (-log10 q).

Output: <output>/<stem>.pdf (vector, editable text), .svg, .png (600 dpi).
Canvas 80 x 60 mm; axes placed with fig.add_axes (no tight_layout).

    python 03_plot_lion_panel_c.py
    python 03_plot_lion_panel_c.py --input=path/to/result.csv --output=figures/lipid_pathway

Fonts follow 02-plot_transcriptomics_no_direction.py: Arial is required
(registered from ~/.fonts). --allow-font-fallback uses Liberation Sans
(metric-compatible with Arial) for previews only.
"""
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

SCRIPT_DIR = Path(__file__).resolve().parent
MM = 1 / 25.4

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
UP, DOWN = "#C0392B", "#2E6DA4"
TEXT, GREY, BAND = "#222222", "#6B6B6B", "#F0F0F0"
FS_LETTER, FS_TITLE, FS_GROUP, FS_ROW = 8, 6, 5.6, 5.5
FS_TICK, FS_AXIS, FS_NOTE = 5.5, 6, 5  # 5 pt is the minimum font size
FIG_W, FIG_H = 80, 60                    # mm
AX_L, AX_W, AX_B, AX_H = 28.4, 44.8, 9, 44  # mm
XLIM, XTICKS = (-7, 7), [-6, -3, 0, 3, 6]
Q_LINE = -np.log10(0.05)                 # 1.30
GROUP_GAP = 0.7                           # extra row spacing between groups
DOT_AREA = 40
MAX_LABEL = 35
SHOW_GROUP_TITLES = False  # direction is already given by colour and the Down/Up axis cues

# Display names: first letter capitalised, LIPID MAPS codes kept.
# FA, fatty acid; PC/PE/PS, phosphatidyl-choline/-ethanolamine/-serine.
ABBREV = [
    (r"^fatty acid", "FA"),
    (r"^saturated fatty acid", "Saturated FA"),
    (r"less than (\d+)", r"<\1"),
    (r"more than (\d+)", r">\1"),
    (r"(\d+)-(\d+)", "\\1\u2013\\2"),
    (r"^monoacylglycerophosphocholines", "Monoacyl-PC"),
    (r"^diacylglycerophosphoethanolamines", "Diacyl-PE"),
    (r"^diacylglycerophosphoserines", "Diacyl-PS"),
    (r"^headgroup with positive charge / zwitter-ion", "Cationic/zwitterionic headgroup"),
    (r"^headgroup with neutral charge", "Neutral headgroup"),
    (r"^simple glc series", "Simple Glc series"),
    (r"^glycerophosphocholines", "Phosphocholines"),
]


def configure_fonts(allow_fallback):
    for f in glob.glob(os.path.expanduser("~/.fonts/*.ttf")):
        font_manager.fontManager.addfont(f)
    family = "Arial"
    try:
        for w in ("normal", "bold"):
            font_manager.findfont(font_manager.FontProperties(family=family, weight=w),
                                  fallback_to_default=False)
    except ValueError:
        if not allow_fallback:
            sys.exit("Arial (regular + bold) not found. Put arial.ttf/arialbd.ttf in ~/.fonts "
                     "or pass --allow-font-fallback for a preview.")
        family = "Liberation Sans"
        warnings.warn("Arial not found; using Liberation Sans (metric-compatible) for preview.")
    plt.rcParams.update({
        "font.family": family, "mathtext.default": "regular",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.unicode_minus": True,
        "text.color": TEXT, "axes.edgecolor": TEXT, "axes.labelcolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT,
        "axes.linewidth": 0.5,
        "xtick.major.size": 2, "xtick.major.width": 0.5, "xtick.major.pad": 1.5,
        "ytick.major.size": 0, "ytick.major.pad": 2,
        "axes.grid": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
    })
    return family


def display_label(term):
    m = re.match(r"^(.*?)\s*(\[[^\]]+\])?\s*$", str(term).strip())
    name, code = m.group(1), m.group(2)
    for pat, rep in ABBREV:
        name = re.sub(pat, rep, name, flags=re.I)
    name = name[:1].upper() + name[1:]
    label = f"{name} {code}" if code else name
    if len(label) > MAX_LABEL:
        warnings.warn(f"Label longer than {MAX_LABEL} characters: {label}")
    return label


def pick(df, names):
    for n in names:
        for c in df.columns:
            if c.strip().lower() == n:
                return c
    return None


def load(path, fdr):
    df = pd.read_csv(path, encoding="utf-8-sig")
    term = pick(df, ["term", "description", "discription", "term name"])
    qcol = pick(df, ["q", "q_value", "fdr q-value", "fdr", "padj", "q-value"])
    dcol = pick(df, ["direction", "regulated"])
    scol = pick(df, ["signed", "es", "enrichment_score"])
    if term is None or qcol is None or (dcol is None and scol is None):
        sys.exit(f"Need term, q and direction/signed columns; found {list(df.columns)}")
    q = pd.to_numeric(df[qcol], errors="raise")
    if dcol is not None:
        d = df[dcol].astype(str).str.strip().str.upper()
        if not d.isin(["UP", "DOWN"]).all():
            sys.exit("Direction must be UP or DOWN.")
        sign = np.where(d == "UP", 1.0, -1.0)
        if scol is not None:  # consistency check against ES when present
            es = pd.to_numeric(df[scol])
            if ((es > 0) & (sign < 0)).any() or ((es < 0) & (sign > 0)).any():
                sys.exit("Direction and signed/ES columns disagree.")
    else:
        sign = np.sign(pd.to_numeric(df[scol]))
    pcol = pick(df, ["p-value", "p_value", "p"])
    out = pd.DataFrame({
        "term": df[term].astype(str).str.strip(),
        "q": q, "sign": sign,
        "p": pd.to_numeric(df[pcol]) if pcol else q,
    })
    out = out[(out.q < fdr) & (out.sign != 0)].copy()
    out["signed"] = out.sign * -np.log10(out.q)
    out["label"] = out.term.map(display_label)
    return out


def layout(d):
    """Return up (top, strongest first) and down (weakest first) with y."""
    up = d[d.sign > 0].sort_values(["signed", "p"], ascending=[True, False])     # weak -> strong
    dn = d[d.sign < 0].sort_values(["signed", "p"], ascending=[True, True])      # strong -> weak
    dn = dn.assign(y=np.arange(len(dn), dtype=float))                         # strongest at y=0
    start = (len(dn) - 1 if len(dn) else -1) + 1 + GROUP_GAP
    up = up.assign(y=start + np.arange(len(up), dtype=float))                 # strongest on top
    return up, dn


def draw(d):
    up, dn = layout(d)
    rows = pd.concat([up, dn])
    fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM))
    ax = fig.add_axes([AX_L / FIG_W, AX_B / FIG_H, AX_W / FIG_W, AX_H / FIG_H])
    top = rows.y.max()
    ax.set_xlim(*XLIM)
    ax.set_ylim(-0.8, top + 1.2)

    # Only q < fdr terms are drawn, so no q > 0.05 band is shown (it would be empty).
    ax.axvline(0, color=TEXT, lw=0.5, zorder=1)

    for grp, col in ((up, UP), (dn, DOWN)):
        ax.hlines(grp.y, 0, grp.signed, color=col, lw=1.0, zorder=3, capstyle="butt")
        ax.scatter(grp.signed, grp.y, s=DOT_AREA, c=col, edgecolors="white",
                   linewidths=0.5, zorder=4, clip_on=False)

    if SHOW_GROUP_TITLES and len(up):
        ax.text(XLIM[1] - 0.15, up.y.max() + 0.62, "Storage lipids \u2191", ha="right",
                va="bottom", fontsize=FS_GROUP, fontweight="bold", color=UP, zorder=5)
    if SHOW_GROUP_TITLES and len(dn):
        # Starts slightly left of the axes (no row label sits in the group gap) so the
        # box clears the weakest up-regulated point; vertically centred in the gap.
        # zorder 1.2: above the band (0) and zero line (1), below the y tick labels
        # (axis zorder 1.5), so the box never masks a row label.
        ax.text(XLIM[0] - 0.35, dn.y.max() + 1.0, "Membrane / long-chain FA \u2193",
                ha="left", va="center", fontsize=FS_GROUP, fontweight="bold", color=DOWN,
                zorder=1.2, bbox=dict(facecolor="white", edgecolor="none", pad=0.3))

    ax.set_yticks(rows.y)
    ax.set_yticklabels(rows.label, fontsize=FS_ROW)
    for lab, s in zip(ax.get_yticklabels(), rows.sign):
        lab.set_color(UP if s > 0 else DOWN)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([f"{t:d}".replace("-", "\u2212") for t in XTICKS], fontsize=FS_TICK)
    ax.set_xlabel("Signed \u2212log$_{10}$ q", fontsize=FS_AXIS, labelpad=1)
    # Direction cues at the two ends of the axis-title line, coloured by direction.
    fig.canvas.draw()
    lab = ax.xaxis.label.get_window_extent(fig.canvas.get_renderer())
    y_lab = ax.transAxes.inverted().transform(((lab.x0 + lab.x1) / 2, (lab.y0 + lab.y1) / 2))[1]
    ax.text(0, y_lab, "\u2190 Down", transform=ax.transAxes, ha="left", va="center",
            fontsize=FS_TICK, color=DOWN, clip_on=False)
    ax.text(1, y_lab, "Up \u2192", transform=ax.transAxes, ha="right", va="center",
            fontsize=FS_TICK, color=UP, clip_on=False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    # ax.set_title("Lipid classes at the 48 y lipidomic peak", fontsize=FS_TITLE,
    #              fontweight="normal", pad=4, loc="center")
    # fig.text(1 / FIG_W, 1 - 1 / FIG_H, "c", ha="left", va="top", fontsize=FS_LETTER,
    #          fontweight="bold")
    return fig, ax, rows


def check(fig, ax):
    """Fail loudly if labels are clipped or the group titles collide."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    problems = []
    for t in ax.get_yticklabels() + [ax.title, ax.xaxis.label]:
        b = t.get_window_extent(r)
        if b.x0 < fb.x0 - 0.5 or b.x1 > fb.x1 + 0.5 or b.y0 < fb.y0 - 0.5 or b.y1 > fb.y1 + 0.5:
            problems.append(f"clipped: {t.get_text()}")
    texts = [t for t in ax.texts if "lipids" in t.get_text() or "Membrane" in t.get_text()]
    from matplotlib.collections import PathCollection
    dots = [c for c in ax.collections if isinstance(c, PathCollection)]
    for t in texts:
        tb = t.get_window_extent(r)
        for c in dots:
            for x, y in ax.transData.transform(c.get_offsets()):
                rad = np.sqrt(DOT_AREA) / 2 * fig.dpi / 72
                dx = max(tb.x0 - x, 0, x - tb.x1)
                dy = max(tb.y0 - y, 0, y - tb.y1)
                if np.hypot(dx, dy) < rad:  # circle-rectangle intersection
                    problems.append(f"'{t.get_text()}' overlaps a point")
    ab = ax.get_window_extent(r)
    for c in dots:
        for x, y in ax.transData.transform(c.get_offsets()):
            if not (ab.x0 <= x <= ab.x1 and ab.y0 <= y <= ab.y1):
                problems.append("a point lies outside the axes")
    if any(t.get_window_extent(r).x1 > ab.x0 for t in ax.get_yticklabels()):
        problems.append("a row label runs into the plotting area")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(SCRIPT_DIR / "outputs" / "lipid_pathway" /
                                           "lion_age_48_two_sided_result.csv"))
    ap.add_argument("--output", default=str(SCRIPT_DIR / "figures" / "lipid_pathway"))
    ap.add_argument("--stem", default="lion_48y_lollipop")
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--allow-font-fallback", action="store_true")
    a = ap.parse_args()

    family = configure_fonts(a.allow_font_fallback)
    d = load(a.input, a.fdr)
    if d.empty:
        sys.exit("No terms with q < fdr.")
    if d.signed.abs().max() > XLIM[1] - 0.3:
        sys.exit("A signed value exceeds the fixed x range of +/-7; widen XLIM.")
    fig, ax, rows = draw(d)
    problems = check(fig, ax)
    for p in problems:
        warnings.warn(p)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / a.stem
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.svg")
    fig.savefig(f"{stem}.png", dpi=600)
    rows[["term", "label", "q", "sign", "signed", "y"]].sort_values("y", ascending=False).to_csv(
        f"{stem}_source_data.csv", index=False)
    print(f"Font: {family}; terms: {len(rows)} ({(rows.sign > 0).sum()} up, "
          f"{(rows.sign < 0).sum()} down); saved {stem}.pdf/.svg/.png")
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
