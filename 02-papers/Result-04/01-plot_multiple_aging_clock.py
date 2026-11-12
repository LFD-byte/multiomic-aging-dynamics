"""Nature-portfolio-style figures for the linear vs. nonlinear aging-clock comparison.

Reads the tables written by 01-nonlinearity_model_comparison.py (./outputs) and
writes vector PDF + 600-dpi PNG figures to ./figures/nature.

Every figure is exactly PAGE_W = 140 mm wide. Axes are placed in absolute
millimetres (see ``new_fig`` / ``add_ax``) and figures are saved WITHOUT
bbox_inches="tight", so the exported page width is exactly 140 mm and the
text sizes (5-7 pt, Nature guideline) are the true printed sizes.

Figures
-------
Figure 1        Main figure (140 mm)
                a-c  MAE / R2 / Pearson r, Elastic Net vs nonlinear (repeat level)
                d    Participant-cluster bootstrap 95% CI of MAE improvement (%)
                e    Inner-CV winner frequency (Spline GAM vs LightGBM)
                f    Cohort size and feature count per omic
Figure 2        Chronological vs predicted age (repeat-averaged OOF predictions)
Extended Data 1 Fold-level distributions of all three base models
Extended Data 2 Paired fold-level MAE differences, inner-CV MAE, age-dependent bias
Table           Summary table (mean +/- s.d., improvement, BH-FDR q) as CSV

Usage: python 01-plot_multiple_aging_clock.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy.stats import pearsonr

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ANALYSIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ANALYSIS_DIR / "outputs"
FIG_DIR = ANALYSIS_DIR / "figures" / "nature"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Nature style (sizes are true printed sizes at PAGE_W = 140 mm)
# --------------------------------------------------------------------------- #
MM = 1 / 25.4
PAGE_W = 140.0          # mm, page width of every figure
FS, FS_TICK, FS_SMALL, FS_LABEL = 6.0, 5.5, 5.0, 8.0   # pt: text / ticks / annotations / panel letters

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_TICK,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelpad": 1.5, "xtick.major.pad": 1.2, "ytick.major.pad": 1.2,
    "axes.titlepad": 2.0,
    "lines.linewidth": 0.7, "patch.linewidth": 0.5,
    "legend.frameon": False, "legend.handlelength": 1.0, "legend.handletextpad": 0.3,
    "legend.borderpad": 0.2, "legend.borderaxespad": 0.2, "legend.columnspacing": 1.0,
    "figure.dpi": 150, "savefig.dpi": 600,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "mathtext.default": "regular",
})

OMICS = ["Metabolomics", "Lipidomics", "Gut Microbiome", "Oral Microbiome", "Transcriptomics"]
LABEL = {"Metabolomics": "Metabolomics", "Lipidomics": "Lipidomics",
         "Gut Microbiome": "Gut\nmicrobiome", "Oral Microbiome": "Oral\nmicrobiome",
         "Transcriptomics": "RNA"}
LABEL1 = {k: v.replace("\n", " ") for k, v in LABEL.items()}
EN, GAM, LGB, NL = "Elastic Net", "Spline GAM", "LightGBM", "Nonlinear representative"
COL = {EN: "#7A7F87", GAM: "#377EB8", LGB: "#E66C2C", NL: "#7B3294"}
NL_NAME = "Nonlinear (inner-CV best)"
METRICS = [("MAE", "MAE (years)"), ("R2", r"$R^2$"), ("PearsonR", "Pearson $r$")]
GRID = dict(color="#E8E8E8", lw=0.4)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def read(name, parquet=False):
    p = OUTPUT_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run 01-nonlinearity_model_comparison.py first")
    return pd.read_parquet(p) if parquet else pd.read_csv(p)


def new_fig(height_mm):
    """Figure of exactly PAGE_W x height_mm."""
    return plt.figure(figsize=(PAGE_W * MM, height_mm * MM))


def add_ax(fig, x, y, w, h, **kw):
    """Axes at (x, y) = lower-left corner, size w x h, all in mm from the figure's lower-left."""
    W, H = fig.get_size_inches() / MM
    return fig.add_axes([x / W, y / H, w / W, h / H], **kw)


def fig_text(fig, x, y, s, **kw):
    """Text at (x, y) in mm from the figure's lower-left."""
    W, H = fig.get_size_inches() / MM
    return fig.text(x / W, y / H, s, **kw)


def panel_label(fig, x, y, s):
    """Bold panel letter; (x, y) = top-left corner in mm."""
    fig_text(fig, x, y, s, fontsize=FS_LABEL, fontweight="bold", va="top", ha="left")


def star(q):
    return "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else "n.s."


def save(fig, stem):
    # No bbox_inches="tight": it would crop/grow the canvas and break the 140 mm page width.
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}")
    w, h = fig.get_size_inches() / MM
    plt.close(fig)
    print(f"saved {FIG_DIR / stem}.pdf/.png  ({w:.0f} x {h:.0f} mm)")


def xticks_omics(ax, multiline=True, rotate=0):
    ax.set_xticks(range(len(OMICS)))
    ax.set_xticklabels([(LABEL if multiline else LABEL1)[o] for o in OMICS],
                       rotation=rotate, ha="right" if rotate else "center",
                       rotation_mode="anchor" if rotate else "default")


def jitter(rng, n, w=0.06):
    return rng.uniform(-w, w, n)


# --------------------------------------------------------------------------- #
# Figure 1
# --------------------------------------------------------------------------- #
def metric_panel(ax, rm, tests, metric, ylabel, rng):
    tops, bottoms = [], []
    for i, o in enumerate(OMICS):
        sub = rm[rm.omics == o]
        mus = {}
        for m, dx in [(EN, -0.17), (NL, 0.17)]:
            v = sub.loc[sub.model == m, metric].to_numpy()
            ax.scatter(i + dx + jitter(rng, len(v)), v, s=2, color=COL[m],
                       alpha=0.35, lw=0, zorder=2)
            mu, sd = v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)
            ax.errorbar(i + dx, mu, yerr=sd, fmt="o", ms=3.0, color=COL[m], mec="white",
                        mew=0.35, capsize=1.2, elinewidth=0.6, capthick=0.5, zorder=4)
            mus[m] = mu
            tops.append(v.max()); bottoms.append(v.min())
        ax.plot([i - 0.17, i + 0.17], [mus[EN], mus[NL]], color="#A0A0A0", lw=0.5, zorder=1)
    lo, hi = min(bottoms), max(tops)
    rng_y = max(hi - lo, 1e-6)
    for i, o in enumerate(OMICS):
        q = tests.query("omics==@o and metric==@metric").q_value_fdr_bh.iloc[0]
        top = rm[(rm.omics == o) & rm.model.isin([EN, NL])][metric].max()
        ax.text(i, top + 0.03 * rng_y, star(q), ha="center", va="bottom",
                fontsize=FS if q < .05 else FS_SMALL, color="#222")
    ax.set_ylim(lo - 0.06 * rng_y, hi + 0.16 * rng_y)
    ax.set_xlim(-0.55, len(OMICS) - 0.45)
    ax.set_ylabel(ylabel)
    xticks_omics(ax, multiline=False, rotate=40)
    ax.tick_params(axis="x", pad=1.0)
    ax.grid(axis="y", **GRID, zorder=0)
    ax.set_axisbelow(True)
    if metric in ("R2", "PearsonR"):
        ax.axhline(0, color="#666", lw=0.4)


def figure1():
    rm = read("nonlinearity_repeat_metrics.csv")
    tests = read("nonlinearity_paired_tests.csv")
    boot = read("nonlinearity_bootstrap_improvement.csv")
    tuning = read("nonlinearity_inner_tuning.csv")
    cohorts = read("nonlinearity_cohort_sizes.csv")
    rng = np.random.default_rng(1)

    fig = new_fig(104)
    # ---- row 1: a-c (three metric panels) ----------------------------------------- #
    top_w, top_h, top_y = 35.0, 28.0, 65.5
    top_x = [10.0, 57.0, 104.0]
    for x, (m, yl), lab in zip(top_x, METRICS, "abc"):
        ax = add_ax(fig, x, top_y, top_w, top_h)
        metric_panel(ax, rm, tests, m, yl, rng)
        panel_label(fig, x - 9.5, top_y + top_h + 4.5, lab)
    handles = [Line2D([0], [0], marker="o", ls="", ms=3.0, color=COL[EN], label=EN),
               Line2D([0], [0], marker="o", ls="", ms=3.0, color=COL[NL], label=NL_NAME)]
    W, H = fig.get_size_inches() / MM
    fig.legend(handles=handles, loc="upper center", ncol=2, columnspacing=2.0,
               bbox_to_anchor=(0.5, 1 - 0.3 / H))

    # ---- row 2: d-f (share the omics y axis; names shown once on d) ---------------- #
    bot_y, bot_h = 15.0, 28.0
    y = np.arange(len(OMICS))[::-1]

    # d: forest plot of MAE improvement (%)
    ax = add_ax(fig, 18.0, bot_y, 36.0, bot_h)
    part = boot[boot.contrast == "MAE improvement (%)"].set_index("omics").loc[OMICS]
    ax.errorbar(part.value, y, xerr=[part.value - part.ci_low, part.ci_high - part.value],
                fmt="o", ms=3.0, color=COL[NL], ecolor="#555", elinewidth=0.6,
                capsize=1.2, capthick=0.5, mec="white", mew=0.35)
    ax.axvline(0, color="#777", ls="--", lw=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL1[o] for o in OMICS])
    ax.set_ylim(-0.6, len(OMICS) - 0.4)
    ax.set_xlabel("MAE improvement vs Elastic Net (%)")
    ax.grid(axis="x", **GRID)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    xr = ax.get_xlim()
    ax.set_xlim(xr[0], xr[1] + 1.0 * (xr[1] - xr[0]))
    ax.set_xticks([t for t in ax.get_xticks() if xr[0] <= t <= xr[1]])
    for yi, (_, r) in zip(y, part.iterrows()):
        ax.text(xr[1] + 0.05 * (xr[1] - xr[0]), yi,
                f"{r.value:.1f} [{r.ci_low:.1f}, {r.ci_high:.1f}]",
                va="center", ha="left", fontsize=FS_SMALL, color="#333")
    panel_label(fig, 1.0, bot_y + bot_h + 7.0, "d")

    # e: inner-CV winner frequency
    ax = add_ax(fig, 60.0, bot_y, 34.0, bot_h)
    freq = (tuning.groupby("omics").nonlinear_representative.value_counts(normalize=True)
            .unstack(fill_value=0).reindex(OMICS).fillna(0))
    for c in (GAM, LGB):
        if c not in freq:
            freq[c] = 0.0
    ax.barh(y, freq[GAM] * 100, color=COL[GAM], height=0.62, label=GAM)
    ax.barh(y, freq[LGB] * 100, left=freq[GAM] * 100, color=COL[LGB], height=0.62, label=LGB)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_ylim(-0.6, len(OMICS) - 0.4)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Selected in outer folds (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=0.8)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    panel_label(fig, 56.0, bot_y + bot_h + 7.0, "e")

    # f: cohort size and feature count
    ax = add_ax(fig, 101.0, bot_y, 37.0, bot_h)
    co = cohorts.set_index("omics").loc[OMICS]
    ax.barh(y, co.n_participants, color="#B9C0CA", height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_ylim(-0.6, len(OMICS) - 0.4)
    ax.set_xlabel("Participants (×1,000)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}"))
    xm = co.n_participants.max()
    for yi, (_, r) in zip(y, co.iterrows()):
        ax.text(r.n_participants + 0.03 * xm, yi,
                f"n={int(r.n_participants):,}; p={int(r.n_candidate_features):,}",
                va="center", fontsize=FS_SMALL, color="#333")
    ax.set_xlim(0, xm * 1.9)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    panel_label(fig, 97.0, bot_y + bot_h + 7.0, "f")

    fig_text(fig, 1.0, 1.0,
             "a–c, points: 10 repeats of pooled outer-CV performance (mean ± s.d.). Stars: one-sided "
             "paired fold-wise Wilcoxon test (nonlinear better than Elastic Net),\nBH-FDR across five "
             "omics. d, participant-cluster bootstrap 95% CI. f, n, participants; p, input features.",
             fontsize=FS_SMALL, color="#555", ha="left", va="bottom", linespacing=1.25)
    save(fig, "Figure1_linear_vs_nonlinear")


# --------------------------------------------------------------------------- #
# Figure 2: predicted vs chronological age
# --------------------------------------------------------------------------- #
def figure2():
    pred = read("nonlinearity_oof_predictions.parquet", parquet=True)
    m_l, m_r, gap_x, gap_y = 14.0, 1.5, 2.8, 2.5          # mm
    m_b, m_t = 8.5, 9.0                                    # mm (m_t: column titles + note)
    s = (PAGE_W - m_l - m_r - 4 * gap_x) / 5               # square panel side
    fig = new_fig(m_b + 2 * s + gap_y + m_t)
    lim = (pred.age.min() - 2, pred.age.max() + 2)
    ticks = [t for t in range(0, 130, 10) if lim[0] < t < lim[1]]
    cmaps = {EN: "Greys", NL: "Purples"}
    for j, o in enumerate(OMICS):
        part = pred[pred.omics == o]
        avg = part.groupby("row_id").agg(age=("age", "first"), **{EN: (EN, "mean"), NL: (NL, "mean")})
        for i, m in enumerate([EN, NL]):
            ax = add_ax(fig, m_l + j * (s + gap_x), m_b + (1 - i) * (s + gap_y), s, s)
            ax.hexbin(avg.age, avg[m], gridsize=22, extent=(*lim, *lim), mincnt=1,
                      cmap=cmaps[m], linewidths=0.0, bins="log")
            ax.plot(lim, lim, color="#333", lw=0.5, ls="--")
            mae = np.mean(np.abs(avg.age - avg[m]))
            r = pearsonr(avg.age, avg[m])[0]
            ax.text(0.05, 0.97, f"MAE = {mae:.2f}\n$r$ = {r:.2f}", transform=ax.transAxes,
                    va="top", ha="left", fontsize=FS_SMALL, linespacing=1.15)
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xticks(ticks); ax.set_yticks(ticks)
            if i == 0:
                ax.set_title(LABEL1[o])
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(f"{EN if i == 0 else NL_NAME}\npredicted age (years)")
            else:
                ax.set_yticklabels([])
    H = fig.get_size_inches()[1] / MM
    for k, lab in enumerate("ab"):
        panel_label(fig, 0.5, m_b + (2 - k) * s + (1 - k) * gap_y + 1.0 + (3.5 if k == 0 else 0), lab)
    fig_text(fig, m_l + (PAGE_W - m_l - m_r) / 2, 0.8, "Chronological age (years)",
             ha="center", va="bottom")
    fig_text(fig, PAGE_W - m_r, H - 0.8,
             "Repeat-averaged out-of-fold predictions; colour, log count per hexagon",
             ha="right", va="top", fontsize=FS_SMALL, color="#555")
    save(fig, "Figure2_predicted_vs_chronological_age")


# --------------------------------------------------------------------------- #
# Extended Data 1: fold-level distributions
# --------------------------------------------------------------------------- #
def nice_five_ticks(lo, hi):
    """Five evenly spaced round ticks that together cover [lo, hi] (smallest nice step)."""
    for k in range(-6, 4):
        for base in (1, 1.5, 2, 2.5, 5):
            step = base * 10.0 ** k
            start = np.floor(lo / step) * step
            if start + 4 * step >= hi - 1e-12:
                # slide the window up as far as possible while still covering lo
                while start + step <= lo + 1e-12 and start + 5 * step >= hi - 1e-12:
                    start += step
                return np.round(start + step * np.arange(5), 10), step
    return np.linspace(lo, hi, 5), (hi - lo) / 4


def extended1():
    fm = read("nonlinearity_fold_metrics.csv")
    models = [EN, GAM, LGB]
    # Fixed 140 mm page width; height follows from panel count.
    fig_w, panel_h, gap = PAGE_W, 24.0, 2.5                  # mm
    m_l, m_r, m_t, m_b = 10.5, 1.0, 5.5, 5.0                 # mm margins
    n_p = len(METRICS)
    fig_h = m_t + m_b + n_p * panel_h + (n_p - 1) * gap
    dark = {EN: "#565B63", GAM: "#20628F", LGB: "#B54D14"}
    fig, axes = plt.subplots(n_p, 1, figsize=(fig_w * MM, fig_h * MM), sharex=True)
    rng = np.random.default_rng(2)
    w = 0.26
    for ax, (metric, yl), lab in zip(axes, METRICS, "abc"):
        # Robust axis range: distributions whose median is far from the rest (unstable fits)
        # are drawn as off-scale arrows so they cannot squash the informative panels.
        grp = {(o, m): fm[(fm.omics == o) & (fm.model == m)][metric].dropna().to_numpy()
               for o in OMICS for m in models}
        meds = np.array([np.median(v) for v in grp.values()])
        centre = np.median(meds)
        scale = max(1.4826 * np.median(np.abs(meds - centre)), 1e-6)
        off = {k: abs(np.median(v) - centre) > 4.5 * scale for k, v in grp.items()}
        kept = np.concatenate([v for k, v in grp.items() if not off[k]])
        ticks, step = nice_five_ticks(kept.min(), kept.max())
        edge = 0.04 * (ticks[-1] - ticks[0])
        ylo, yhi = ticks[0] - edge, ticks[-1] + edge
        span_y = yhi - ylo
        for i in range(len(OMICS)):
            if i % 2 == 0:
                ax.axvspan(i - 0.5, i + 0.5, color="#F4F6F8", lw=0, zorder=0)
        for i, o in enumerate(OMICS):
            for j, m in enumerate(models):
                v = grp[(o, m)]
                pos = i + (j - 1) * w
                if off[(o, m)]:
                    med = float(np.median(v))
                    up = med > centre
                    tip = yhi - 0.03 * span_y if up else ylo + 0.03 * span_y
                    tail = tip - 0.22 * span_y if up else tip + 0.22 * span_y
                    ax.annotate("", xy=(pos, tip), xytext=(pos, tail),
                                arrowprops=dict(arrowstyle="-|>", color=dark[m], lw=0.6,
                                                mutation_scale=4.5), zorder=6)
                    ax.text(pos + 0.05, (tail + tip) / 2, f"{med:.2f}", ha="left", va="center",
                            fontsize=FS_SMALL, color=dark[m], zorder=6)
                    continue
                ax.scatter(pos + jitter(rng, len(v), w * 0.28), v, s=0.9, color=COL[m],
                           alpha=0.5, lw=0, zorder=2)
                bp = ax.boxplot(v, positions=[pos], widths=w * 0.66, patch_artist=True,
                                showfliers=False, manage_ticks=False, zorder=3)
                bp["boxes"][0].set(facecolor=COL[m], edgecolor=dark[m], alpha=0.30, lw=0.5)
                for k in ("whiskers", "caps"):
                    for ln in bp[k]:
                        ln.set(color=dark[m], lw=0.5)
                bp["medians"][0].set(color=dark[m], lw=1.0)
        ax.set_xlim(-0.5, len(OMICS) - 0.5)
        ax.set_ylim(ylo, yhi)
        ax.set_yticks(ticks)
        dec = next(d for d in range(6) if abs(round(step, d) - step) < 1e-9 or d == 5)
        ax.set_yticklabels([f"{t:.{dec}f}" for t in ticks])
        ax.set_ylabel(yl, labelpad=1.5)
        ax.grid(axis="y", color="#E3E6EA", lw=0.4, zorder=1)
        ax.set_axisbelow(True)
        ax.spines["bottom"].set_visible(lab == "c")
        ax.tick_params(axis="x", length=0, pad=2)
        if metric != "MAE" and ylo < 0 < yhi:
            ax.axhline(0, color="#8A8F98", lw=0.5, ls=(0, (3, 2)), zorder=1)
    xticks_omics(axes[-1], multiline=False)
    axes[-1].tick_params(axis="x", labelsize=FS)
    fig.legend(handles=[Line2D([0], [0], marker="s", ls="", color=COL[m], alpha=0.8, label=m,
                               ms=4) for m in models],
               ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), columnspacing=2.0,
               handletextpad=0.2, fontsize=FS)
    fig.subplots_adjust(left=m_l / fig_w, right=1 - m_r / fig_w, top=1 - m_t / fig_h,
                        bottom=m_b / fig_h, hspace=gap / panel_h)
    save(fig, "ExtendedData1_fold_level_distributions")


# --------------------------------------------------------------------------- #
# Extended Data 2: paired differences, inner-CV, age-dependent bias
# --------------------------------------------------------------------------- #
def extended2():
    fm = read("nonlinearity_fold_metrics.csv")
    tuning = read("nonlinearity_inner_tuning.csv")
    pred = read("nonlinearity_oof_predictions.parquet", parquet=True)
    rng = np.random.default_rng(3)

    fig = new_fig(92)
    # a: paired fold-level MAE difference
    ax = add_ax(fig, 14.0, 58.0, 68.0, 28.0)
    wide = fm.pivot_table(index=["omics", "repeat", "fold"], columns="model", values="MAE")
    for i, o in enumerate(OMICS):
        d = (wide.loc[o][EN] - wide.loc[o][NL]).to_numpy()
        ax.scatter(i + jitter(rng, len(d), 0.18), d, s=1.5, color=COL[NL], alpha=0.4, lw=0)
        ax.hlines(np.median(d), i - 0.28, i + 0.28, color="k", lw=0.9, zorder=4)
    ax.axhline(0, color="#777", ls="--", lw=0.5)
    xticks_omics(ax, multiline=True)
    ax.set_ylabel("Fold-level MAE difference\n(Elastic Net − nonlinear, years)")
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    panel_label(fig, 1.0, 91.0, "a")

    # b: inner-CV MAE of GAM vs LightGBM
    ax = add_ax(fig, 96.0, 58.0, 42.0, 28.0)
    for i, o in enumerate(OMICS):
        t = tuning[tuning.omics == o]
        for m, col, dx in [(GAM, "gam_inner_mae", -0.18), (LGB, "lightgbm_inner_mae", 0.18)]:
            v = t[col].dropna().to_numpy()
            bp = ax.boxplot(v, positions=[i + dx], widths=0.28, patch_artist=True,
                            showfliers=False, manage_ticks=False)
            bp["boxes"][0].set(facecolor=COL[m], edgecolor="none", alpha=0.85)
            bp["medians"][0].set(color="white", lw=0.8)
            for k in ("whiskers", "caps"):
                for ln in bp[k]:
                    ln.set(color="#555", lw=0.5)
    ax.set_xlim(-0.55, len(OMICS) - 0.45)
    xticks_omics(ax, multiline=False, rotate=40)
    ax.set_ylabel("Inner-CV MAE (years)")
    ax.legend(handles=[Line2D([0], [0], marker="s", ls="", color=COL[m], label=m, ms=3.5)
                       for m in (GAM, LGB)], loc="upper left", ncol=1)
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    panel_label(fig, 86.0, 91.0, "b")

    # c: mean residual vs. chronological age (regression-to-the-mean check)
    edges = np.arange(40, 71, 5)
    centers = (edges[:-1] + edges[1:]) / 2
    c_l, c_r, c_gap, c_y, c_h = 14.0, 1.5, 2.5, 9.0, 28.0
    c_w = (PAGE_W - c_l - c_r - 4 * c_gap) / 5
    axes_c = [add_ax(fig, c_l + j * (c_w + c_gap), c_y, c_w, c_h) for j in range(5)]
    for j, (ax, o) in enumerate(zip(axes_c, OMICS)):
        part = pred[pred.omics == o]
        avg = part.groupby("row_id").agg(age=("age", "first"), **{EN: (EN, "mean"), NL: (NL, "mean")})
        bins = pd.cut(avg.age, edges, include_lowest=True)
        for m in (EN, NL):
            res = (avg[m] - avg.age).groupby(bins, observed=False)
            mu, se = res.mean().to_numpy(), res.sem().to_numpy()
            ax.plot(centers, mu, color=COL[m], marker="o", ms=1.8, lw=0.7)
            ax.fill_between(centers, mu - 1.96 * se, mu + 1.96 * se, color=COL[m],
                            alpha=0.2, lw=0)
        ax.axhline(0, color="#666", ls="--", lw=0.5)
        ax.set_title(LABEL1[o])
        ax.set_xticks([45, 55, 65])
        ax.set_xlim(edges[0], edges[-1])
        if j == 0:
            ax.set_ylabel("Mean residual\n(predicted − age, years)")
        ax.grid(axis="y", **GRID)
    lo = min(a.get_ylim()[0] for a in axes_c); hi = max(a.get_ylim()[1] for a in axes_c)
    for j, a in enumerate(axes_c):
        a.set_ylim(lo, hi)
        if j:
            a.set_yticklabels([])
    panel_label(fig, 1.0, c_y + c_h + 8.0, "c")
    fig_text(fig, c_l + (PAGE_W - c_l - c_r) / 2, 0.8, "Chronological age (years)",
             ha="center", va="bottom")
    W, H = fig.get_size_inches() / MM
    fig.legend(handles=[Line2D([0], [0], color=COL[EN], marker="o", ms=1.8, label=EN),
                        Line2D([0], [0], color=COL[NL], marker="o", ms=1.8, label=NL_NAME)],
               loc="upper right", ncol=2, columnspacing=1.5,
               bbox_to_anchor=(1 - c_r / W, (c_y + c_h + 8.3) / H), borderaxespad=0)
    save(fig, "ExtendedData2_paired_tuning_bias")


# ---------------------------------------------------------------------------
# Summary table
# --------------------------------------------------------------------------- #
def summary_table():
    s = read("nonlinearity_model_metrics.csv")
    t = read("nonlinearity_paired_tests.csv")
    b = read("nonlinearity_bootstrap_improvement.csv")
    rows = []
    for o in OMICS:
        row = {"Omics": LABEL1[o]}
        for m, nm in [(EN, "EN"), (NL, "NL")]:
            r = s[(s.omics == o) & (s.model == m)].iloc[0]
            row[f"MAE {nm}"] = f"{r.MAE:.2f} ± {r.MAE_sd:.2f}"
            row[f"R2 {nm}"] = f"{r.R2:.3f} ± {r.R2_sd:.3f}"
            row[f"r {nm}"] = f"{r.PearsonR:.3f} ± {r.PearsonR_sd:.3f}"
        bb = b[(b.omics == o) & (b.contrast == "MAE improvement (%)")].iloc[0]
        row["MAE improvement % [95% CI]"] = f"{bb.value:.1f} [{bb.ci_low:.1f}, {bb.ci_high:.1f}]"
        tm = t[(t.omics == o) & (t.metric == "MAE")].iloc[0]
        row["q (MAE, BH-FDR)"] = f"{tm.q_value_fdr_bh:.2g}"
        row["n participants"] = int(s[(s.omics == o)].n_participants.max())
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(FIG_DIR / "Table1_summary.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    print(f"Reading from {OUTPUT_DIR}\nWriting to {FIG_DIR}")
    figure1()
    figure2()
    extended1()
    extended2()
    summary_table()