#!/usr/bin/env python3
# =============================================================================
# VARIANT: signed coherence score ("no_direction").
#   Same layout as 02-plot_transcriptomics.py, but each heatmap row is the
#   signed coherence score of the pathway's crest transcripts instead of their
#   plain mean, so transcripts moving in opposite directions at the crest no
#   longer cancel:
#     s_g   = sign of the DE-SWAN contrast at the crest: OLS  z_g ~ older + sex
#             in [peak - W, peak + W), older = age >= peak (W = 7.5 y, as DE-SWAN)
#     score = mean_g( s_g * z_g ) per sample  ->  LOESS (span 0.34, degree 1)
#             on the same 0.25-y grid as the R heatmap  ->  row-centred, NOT
#             row-scaled (units: SD of expression), one colour scale for all rows.
#   Row scaling is deliberately not used: because s_g is estimated in the crest
#   window, any gene set (even random genes) shows a step at the crest after
#   row z-scoring. The amplitude is what separates crest pathways from noise,
#   so it is kept and tested against random non-crest gene sets of equal size
#   (empirical P; '*' marks P < 0.05).
#   Extra input: ../01-interim_data/omics_data/transcriptomics_with_metadata.csv
#   (--expression), pathway_molecular_expression_heatmap_pathway_key.csv and
#   kegg_enrichment/crest_associated_transcripts.csv.
# -----------------------------------------------------------------------------
# Result 2, Fig. b: pathway trajectories (heatmaps) and enrichment networks for
# the two transcriptomic crests (Crest1, 52 y; Crest2, 64 y).
#
# Inputs (written on the server by the two Result-02 R scripts):
#   02-transcriptomics_pathway_molecular_expression_heatmap.R
#     <base>/pathway_expression_heatmap/pathway_molecular_expression_heatmap_source_data.csv
#     <base>/kegg_enrichment/all_pathway_enrichment_results.csv   (gene counts)
#   02-transcriptomics_pathway_similarity_network.R
#     <base>/similarity_network/figure_network_plot_nodes.csv
#     <base>/similarity_network/figure_network_plot_edges.csv
#   <base> defaults to ./outputs/transcriptomics_pathway
#
# Output: figures/transcriptomics/transcriptomics_crests_signed_coherence.{pdf,svg,png}
#         outputs/transcriptomics_pathway/transcriptomics_crests_signed_coherence_*.csv / _layout_check.txt
#         + _source_data_heatmap.csv, _source_data_network.csv, _layout_check.txt
#
# Canvas 183 x 115 mm; every axes is placed at absolute mm positions with
# fig.add_axes (no tight_layout). Text stays editable (pdf.fonttype 42,
# svg.fonttype none). Arial is registered from ~/.fonts and required, as in
# Result-01/01-cohort_feature.ipynb.
#
# Usage: python 02-plot_transcriptomics_no_direction.py [--expression F] [--null-sets N]
#        (other options as in) python 02-plot_transcriptomics.py [--base-dir D] [--figure-dir D]
# =============================================================================

from __future__ import annotations

import argparse
import functools
import glob
import math
import os
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
FONT = "Arial"

# ----------------------------------------------------------------------------
# 1. Pathway-name shortening (GO conventions; applied in order, case-insensitive)
#    Figure legend: "Pathway names are abbreviated: Pos. reg., positive regulation of;
#    Neg. reg., negative regulation of; Reg., regulation of; TNF, tumour necrosis
#    factor; TLR, Toll-like receptor; IFN, interferon; resp., response;
#    ext., external."
# ----------------------------------------------------------------------------
SHORTEN_RULES = [
    (r"positive regulation of\s+", "Pos. reg. "),
    (r"negative regulation of\s+", "Neg. reg. "),
    (r"regulation of\s+", "Reg. "),
    (r"tumou?r necrosis factor", "TNF"),
    (r"toll-like receptor", "TLR"),
    (r"type II interferon", "IFN-II"),
    (r"type I interferon", "IFN-I"),
    (r"interferon alpha/beta", "IFN-α/β"),
    (r"interferon alpha", "IFN-α"),
    (r"interferon beta", "IFN-β"),
    (r"interferon gamma", "IFN-γ"),
    (r"signaling pathway", "signalling"),
    (r"\bsignaling\b", "signalling"),   # keep British spelling consistent
    (r"immune response", "immune resp."),
    (r"external stimulus", "ext. stimulus"),
    (r"\binvolved in\b", "in"),
]
# Entries still longer than ~38 characters after the rules (matched on the result).
MANUAL_SHORT = {
    "Immune resp.-regulating cell surface receptor signalling":
        "Immune cell-surface receptor signalling",
}
MAX_LABEL_CHARS = 39   # "about 38"
GREEK = "αβγδκ"


@functools.lru_cache(maxsize=None)
def shorten(name: str) -> str:
    s = re.sub(r"\s+", " ", str(name).replace("\n", " ")).strip()
    for pattern, repl in SHORTEN_RULES:
        s = re.sub(pattern, repl, s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    first = s.split(" ")[0]
    keep_case = s[:1] in GREEK or re.match(r"^[a-z]+[A-Z0-9]", first)  # mRNA, tRNA, α/β
    if not keep_case:
        s = s[:1].upper() + s[1:]
    s = MANUAL_SHORT.get(s, s)
    if len(s) > MAX_LABEL_CHARS:
        print(f"[warn] label still {len(s)} chars, add to MANUAL_SHORT: {s!r}")
    return s


# Module titles: first rule whose keywords hit the most member terms wins;
# otherwise the most significant member term (shortened) is used.
MODULE_TITLE_RULES = [
    (("interferon",), "IFN signalling"),
    (("autophagy", "vesicle"), "Autophagy &\nvesicle organization"),
    (("virus", "viral", "innate", "cytokine", "toll-like", "leukocyte"), "Innate antiviral\nimmunity"),
    (("retinoid", "hormone"), "Hormone & retinoid\nmetabolism"),
    (("t cell", "adhesion"), "T-cell differentiation\n& adhesion"),
    (("phagocytosis", "endocytosis", "activation of immune"), "Immune activation\n& phagocytosis"),
]

# ----------------------------------------------------------------------------
# 2. Style
# ----------------------------------------------------------------------------
TEXT, GREY, ISOLATED = "#222222", "#6B6B6B", "#9E9E9E"
HEAT_COLOURS = ["#2166AC", "#67A9CF", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#EF8A62", "#B2182B"]
HEAT_NORM = dict(vmin=-2.2, vcenter=0.0, vmax=2.6)   # unused in this variant
SIGNED_HALF_WINDOW = 7.5      # DE-SWAN buckets.size used for the crests
LOESS_SPAN = 0.34             # as PATHWAY_HEATMAP_LOESS_SPAN in the R script
AGE_MIN, AGE_MAX = 40.0, 70.0
MIN_GENE_SAMPLES = 10
STEP_WINDOW = (0.5, 2.0)      # crest step: mean(peak+0.5..+2) - mean(peak-2..-0.5), as crest_delta_z
SHOW_62_LINE = False          # the late-life reference no longer applies to signed scores
SHOW_MODULE_TITLES = False    # coloured module titles in the networks (names from MODULE_TITLE_RULES)
HEAT_FRAME_LW = 0.5           # heatmap frame line width (pt); 0 = no frame
HEAT_FRAME_COLOUR = "#222222"
HEAT_SIGNED_NORM = None       # set in main() from the data (symmetric, shared by all rows)
FDR_COLOURS = ["#FFF1C9", "#F7B267", "#E4572E", "#8E1C2E"]
FDR_RANGE = (2.0, 14.0)
MODULE_COLOURS = ["#3D6FA8", "#3E8E6B", "#7A5AA6", "#B8741A"]
EDGE_COLOUR, EDGE_ALPHA = "#9A9A9A", 0.55
GENE_RANGE, AREA_RANGE = (10, 80), (14, 80)

FS_LETTER, FS_TITLE, FS_ROW, FS_TICK, FS_AXIS = 8.0, 6.0, 5.0, 5.5, 6.0
FS_NOTE, FS_ROWNUM, FS_NODE, FS_MODULE, FS_LEG = 5.0, 4.6, 5.0, 5.2, 5.0

# ----------------------------------------------------------------------------
# 3. Geometry (mm, measured from the TOP of the panel)
# ----------------------------------------------------------------------------
FIG_W, FIG_H = 183.0, 115.0
HEAT_LEFT, HEAT_W = 33.9, 75.0
HEAT_Y = {1: (9.7, 46.3), 2: (58.5, 95.1)}
NET_LEFT, NET_W = 114.4, 65.9
NET_Y = {1: (6.1, 51.2), 2: (54.9, 100.0)}
NET_YLIM = 1.85
MODULE_CENTRES = {1: [(-1.7, 0.0), (0.2, 0.2), (2.0, -0.4)],
                  2: [(-1.55, 0.0), (0.75, 0.55), (0.85, -0.75)]}
ISOLATED_CENTRE = {1: (2.25, -0.35), 2: (2.25, -0.35)}
LEG_TITLE_Y, LEG_BAR_Y, LEG_BAR_H = 106.6, 108.3, 1.7
HEAT_CB_X, HEAT_CB_W = 37.5, 25.6
FDR_CB_X, FDR_CB_W = 120.8, 20.1
SIZE_LEG_X = 146.0
ROW_LABEL_MARGIN = 1.0       # mm kept free at the canvas edge
MM = 1 / 25.4


def fx(mm):
    return mm / FIG_W


def fy_top(mm):          # mm from top -> figure fraction from bottom
    return 1 - mm / FIG_H


def rect(left, top, width, bottom):
    return [fx(left), fy_top(bottom), fx(width), (bottom - top) / FIG_H]


# ----------------------------------------------------------------------------
# 4. Fonts (same approach as Result-01/01-cohort_feature.ipynb)
# ----------------------------------------------------------------------------
def configure_fonts():
    for f in glob.glob(os.path.expanduser("~/.fonts/*.ttf")):
        font_manager.fontManager.addfont(f)
    font_manager.findfont(font_manager.FontProperties(family=FONT), fallback_to_default=False)
    font_manager.findfont(font_manager.FontProperties(family=FONT, weight="bold"),
                          fallback_to_default=False)
    plt.rcParams.update({
        "font.family": FONT,
        "mathtext.default": "regular",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.unicode_minus": False,
        "text.color": TEXT, "axes.edgecolor": TEXT, "axes.labelcolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT,
        "xtick.major.size": 1.5, "xtick.major.width": 0.5, "xtick.major.pad": 1.2,
        "xtick.labelsize": FS_TICK, "axes.labelsize": FS_AXIS,
        "axes.grid": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
    })


# ----------------------------------------------------------------------------
# 5. Data
# ----------------------------------------------------------------------------
def load_inputs(base: Path):
    heat = pd.read_csv(base / "pathway_expression_heatmap" /
                       "pathway_molecular_expression_heatmap_source_data.csv")
    nodes = pd.read_csv(base / "similarity_network" / "figure_network_plot_nodes.csv")
    edges = pd.read_csv(base / "similarity_network" / "figure_network_plot_edges.csv")
    enrich = pd.read_csv(base / "kegg_enrichment" / "all_pathway_enrichment_results.csv",
                         usecols=["ID", "database", "crest_id", "Count"])
    enrich = enrich.rename(columns={"ID": "term_id", "Count": "gene_count"})
    nodes = nodes.merge(enrich.drop_duplicates(["crest_id", "database", "term_id"]),
                        on=["crest_id", "database", "term_id"], how="left")
    if nodes["gene_count"].isna().any():
        raise ValueError("Gene counts missing for network nodes: " +
                         ", ".join(nodes.loc[nodes.gene_count.isna(), "term_uid"]))
    return heat, nodes, edges


def r_loess(x, y, grid, span=LOESS_SPAN):
    """Exact equivalent of R stats::loess(degree = 1, family = "gaussian", surface = "direct")."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    q = int(np.floor(len(x) * span))
    out = np.empty(len(grid))
    for i, g in enumerate(grid):
        d = np.abs(x - g)
        dq = np.partition(d, q - 1)[q - 1]
        w = np.clip(1 - (d / dq) ** 3, 0, None) ** 3
        sw = w.sum()
        xm, ym = (w * x).sum() / sw, (w * y).sum() / sw
        sxx = (w * (x - xm) ** 2).sum()
        b = (w * (x - xm) * (y - ym)).sum() / sxx if sxx > 0 else 0.0
        out[i] = ym + b * (g - xm)
    return out


def load_expression(path: Path, genes):
    """Age 40-70 samples; transcripts standardised over those samples (as the R script)."""
    header = pd.read_csv(path, nrows=0).columns
    wanted = [g for g in dict.fromkeys(genes) if g in header]
    cols = ["age"] + (["sex"] if "sex" in header else []) + wanted
    e = pd.read_csv(path, usecols=cols)
    e["age"] = pd.to_numeric(e["age"], errors="coerce")
    e = e[(e.age >= AGE_MIN) & (e.age <= AGE_MAX)].reset_index(drop=True)
    x = e[wanted].apply(pd.to_numeric, errors="coerce")
    sd = x.std()
    keep = sd.index[(x.notna().sum() >= MIN_GENE_SAMPLES) & (sd > 1e-10)]
    z = (x[keep] - x[keep].mean()) / sd[keep]
    sex = pd.to_numeric(e["sex"], errors="coerce").to_numpy(float) if "sex" in e else None
    return e.age.to_numpy(float), sex, z


def crest_direction(age, sex, z: pd.DataFrame, peak: float, half: float) -> pd.Series:
    """DE-SWAN contrast at the crest: coefficient of 'older' in z ~ older + sex."""
    m = (age >= peak - half) & (age < peak + half)
    if sex is not None:
        m &= np.isfinite(sex)
    cols = [np.ones(m.sum()), (age[m] >= peak).astype(float)]
    if sex is not None and np.unique(sex[m]).size > 1:
        cols.append(sex[m])
    X = np.column_stack(cols)
    Y = z.loc[m].to_numpy()
    beta = np.full(Y.shape[1], np.nan)
    full = np.isfinite(Y).all(axis=0)
    if full.any():
        beta[full] = np.linalg.lstsq(X, Y[:, full], rcond=None)[0][1]
    for j in np.where(~full)[0]:
        ok = np.isfinite(Y[:, j])
        if ok.sum() >= MIN_GENE_SAMPLES:
            beta[j] = np.linalg.lstsq(X[ok], Y[ok, j], rcond=None)[0][1]
    return pd.Series(beta, index=z.columns)


def crest_step(curve, grid, peak):
    lo, hi = STEP_WINDOW
    pre = (grid >= peak - hi) & (grid <= peak - lo)
    post = (grid >= peak + lo) & (grid <= peak + hi)
    return float(curve[post].mean() - curve[pre].mean())


def signed_curve(age, z, genes, signs, grid):
    score = (z[genes] * signs[genes].to_numpy()).mean(axis=1, skipna=True).to_numpy()
    curve = r_loess(age, score, grid)
    return curve - curve.mean()


def build_signed(heat, key, crest_genes, expression, half, n_null, pool_size, seed, report):
    """Signed coherence trajectories, gene directions, per-pathway statistics and null test."""
    key = key.copy()
    key["genes"] = key["qc_transcripts"].fillna("").map(lambda v: [g for g in v.split(";") if g])
    path_genes = sorted({g for gs in key.genes for g in gs})
    header = pd.read_csv(expression, nrows=0).columns
    excluded = set(crest_genes) | set(path_genes) | {"participant_id", "cohort", "research_stage",
                                                    "array_index", "collection_date",
                                                    "collection_timestamp", "batch", "sex", "age", "bmi"}
    pool = [g for g in header if g not in excluded]
    rng = np.random.default_rng(seed)
    null_genes = list(rng.choice(pool, size=min(pool_size, len(pool)), replace=False)) if n_null else []
    age, sex, z = load_expression(expression, path_genes + null_genes)
    null_genes = [g for g in null_genes if g in z.columns]

    mats, long_rows, gene_rows, stat_rows = {}, [], [], []
    for c in sorted(heat.crest_id.unique()):
        d = heat[heat.crest_id == c]
        grid = np.sort(d.age.unique()).astype(float)
        peak = float(d.peak_age.iloc[0])
        beta = crest_direction(age, sex, z, peak, half)
        signs = np.sign(beta).replace(0, 1.0)
        order = d.drop_duplicates("pathway_rank").sort_values("pathway_rank")
        k = key[key.crest_id == c].set_index("pathway_uid")
        rows, stars = [], []
        for _, p in order.iterrows():
            genes = [g for g in k.loc[p.pathway_uid, "genes"] if g in z.columns]
            sc = signed_curve(age, z, genes, signs, grid)
            plain = r_loess(age, z[genes].mean(axis=1, skipna=True).to_numpy(), grid)
            plain = plain - plain.mean()
            step = crest_step(sc, grid, peak)
            null = []
            for _ in range(n_null):
                pick = list(rng.choice(null_genes, size=len(genes), replace=False))
                null.append(crest_step(signed_curve(age, z, pick, np.sign(beta[pick]).replace(0, 1.0),
                                                    grid), grid, peak))
            p_null = (1 + sum(v >= step for v in null)) / (1 + n_null) if n_null else np.nan
            stars.append(bool(n_null) and p_null < 0.05)
            rows.append(sc)
            up = int((signs[genes] > 0).sum())
            stat_rows.append(dict(
                crest_id=c, peak_age=peak, pathway_rank=p.pathway_rank, pathway_uid=p.pathway_uid,
                pathway_label=p.pathway_label, label=shorten(p.pathway_label), n_genes=len(genes),
                n_up=up, n_down=len(genes) - up,
                abs_mean_beta=abs(float(beta[genes].mean())), mean_abs_beta=float(beta[genes].abs().mean()),
                step_unsigned_sd=crest_step(plain, grid, peak), step_signed_sd=step,
                late_unsigned_sd=float(plain[grid >= 66].mean()), late_signed_sd=float(sc[grid >= 66].mean()),
                amplitude_signed_sd=float(sc.max() - sc.min()),
                null_step_median=float(np.median(null)) if null else np.nan,
                null_step_q95=float(np.quantile(null, 0.95)) if null else np.nan,
                p_null=p_null))
            long_rows.append(pd.DataFrame(dict(crest_id=c, pathway_rank=p.pathway_rank,
                                               pathway_label=p.pathway_label, label=shorten(p.pathway_label),
                                               age=grid, signed_coherence_sd=sc,
                                               unsigned_mean_sd=plain)))
        gene_rows.append(pd.DataFrame(dict(crest_id=c, peak_age=peak, transcript=beta.index,
                                           crest_contrast=beta.values, direction=signs.values,
                                           in_pathways=beta.index.isin(path_genes))))
        labels = [shorten(x) for x in order.pathway_label]
        terms = set(order["pathway_uid"].str.split("::", n=1).str[1])
        mats[c] = (np.vstack(rows), grid, labels, peak, terms, stars)
    stats = pd.DataFrame(stat_rows)
    report.append("signed coherence: " + "; ".join(
        f"Crest {c}: step {g.step_signed_sd.min():.3f}-{g.step_signed_sd.max():.3f} SD "
        f"(unsigned {g.step_unsigned_sd.min():.3f}-{g.step_unsigned_sd.max():.3f}), "
        f"{int((g.p_null < 0.05).sum())}/{len(g)} rows P_null < 0.05"
        for c, g in stats.groupby("crest_id")))
    return mats, pd.concat(long_rows), pd.concat(gene_rows), stats


# ----------------------------------------------------------------------------
# 6. Heatmap
# ----------------------------------------------------------------------------
HEAT_CMAP = LinearSegmentedColormap.from_list("pathway_z", HEAT_COLOURS)


def draw_heatmap(ax, mat, ages, labels, peak, crest, numbered, first):
    n = mat.shape[0]
    step = float(np.median(np.diff(ages)))
    fine = step < 0.99
    # Pixel edges sit half a step outside the first/last age so that each column
    # is centred on its age (peak line and title align exactly with the data).
    x0, x1 = (ages[0] - step / 2, ages[-1] + step / 2) if fine else (39.5, 70.5)
    ax.imshow(mat, cmap=HEAT_CMAP, norm=HEAT_SIGNED_NORM, aspect="auto",
              interpolation="bilinear" if fine else "nearest",
              extent=(x0, x1, n - 0.5, -0.5))
    ax.set_xlim(x0, x1)
    ax.set_ylim(n - 0.5, -0.5)
    if SHOW_62_LINE:
        ax.axvline(62, color=GREY, lw=0.45, ls=(0, (0.6, 1.2)), zorder=3)
    ax.axvline(peak, color=TEXT, lw=0.7, ls=(0, (3, 1.5)), zorder=4)
    ax.annotate(f"Crest {crest} · {peak:g} y", xy=(peak, 1), xycoords=("data", "axes fraction"),
                xytext=(0, 2.6), textcoords="offset points", ha="center", va="bottom",
                fontsize=FS_TITLE, fontweight="bold", color=TEXT)
    if first and SHOW_62_LINE:
        ax.annotate("~62 y", xy=(62, 1), xycoords=("data", "axes fraction"),
                    xytext=(0, 0.9), textcoords="offset points", ha="center", va="bottom",
                    fontsize=FS_NOTE, color=GREY)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=FS_ROW, color=TEXT)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.set_xticks(np.arange(40, 71, 5))
    ax.set_xlabel("Age (years)", labelpad=1)
    for s in ax.spines.values():          # thin frame around the heatmap
        s.set_visible(HEAT_FRAME_LW > 0)
        s.set_linewidth(HEAT_FRAME_LW)
        s.set_color(HEAT_FRAME_COLOUR)
        s.set_zorder(5)
    if numbered:
        for i in range(n):
            ax.annotate(str(i + 1), xy=(1, i), xycoords=("axes fraction", "data"),
                        xytext=(1.5, 0), textcoords="offset points", ha="left", va="center",
                        fontsize=FS_ROWNUM, color=GREY)


# ----------------------------------------------------------------------------
# 7. Network
# ----------------------------------------------------------------------------
FDR_CMAP = LinearSegmentedColormap.from_list("fdr", FDR_COLOURS)
FDR_NORM = Normalize(*FDR_RANGE, clip=True)


def node_area(count):
    c = np.clip(count, *GENE_RANGE)
    return AREA_RANGE[0] + (AREA_RANGE[1] - AREA_RANGE[0]) * (c - GENE_RANGE[0]) / (GENE_RANGE[1] - GENE_RANGE[0])


def module_title(members: pd.DataFrame) -> str:
    descs = [d.lower() for d in members["description"]]
    best, hits = None, 0
    for keys, title in MODULE_TITLE_RULES:
        h = sum(any(k in d for k in keys) for d in descs)
        if h > hits:
            best, hits = title, h
    if best is None:
        top = members.sort_values("minus_log10_fdr", ascending=False).iloc[0]
        best = shorten(top["description"])
    return best


def ring_offsets(n):
    if n == 1:
        return [(0.0, 0.0)]
    if n == 2:
        return [(-0.36, 0.0), (0.36, 0.0)]
    if n <= 6:
        r = 0.34 if n == 3 else 0.46
        return [(r * math.cos(math.pi / 2 + 2 * math.pi * k / n),
                 r * math.sin(math.pi / 2 + 2 * math.pi * k / n)) for k in range(n)]
    out = []   # > 6 nodes: alternate inner / outer ring
    for k in range(n):
        r = 0.52 if k % 2 == 0 else 1.0
        a = math.pi / 2 + 2 * math.pi * k / n
        out.append((r * math.cos(a), r * math.sin(a)))
    return out


def module_extent(n):
    offs = np.array(ring_offsets(n))
    return np.abs(offs[:, 0]).max() + 0.3, np.abs(offs[:, 1]).max() + 0.3


def network_layout(nodes: pd.DataFrame, crest: int, xlim: float):
    d = nodes[nodes.crest_id == crest].copy()
    d["community"] = d["display_community"].fillna("")
    sizes = d.groupby("community").size()
    d["isolated"] = (d["community"] == "") | d["community"].map(sizes).eq(1) | ~d["is_connected"].astype(bool)
    modules = (d[~d.isolated].groupby("community")
               .agg(n=("term_uid", "size"), best=("minus_log10_fdr", "max"))
               .sort_values(["n", "best"], ascending=False).index.tolist())
    centres = list(MODULE_CENTRES.get(crest, []))
    while len(centres) < len(modules):          # fallback: spread along x
        centres.append((-2.0 + 4.0 * len(centres) / max(1, len(modules) - 1), 0.0))
    info = []
    for m, (cx, cy), colour in zip(modules, centres, MODULE_COLOURS):
        members = d[d.community == m].sort_values("minus_log10_fdr", ascending=False)
        rx, ry = module_extent(len(members))
        cx = float(np.clip(cx, -xlim + rx + 0.02, xlim - rx - 0.02))
        cy = float(np.clip(cy, -NET_YLIM + ry + 0.02, NET_YLIM - ry - 0.02))
        for (dx, dy), idx in zip(ring_offsets(len(members)), members.index):
            d.loc[idx, ["x", "y"]] = cx + dx, cy + dy
        d.loc[members.index, "module_colour"] = colour
        info.append(dict(community=m, cx=cx, cy=cy, colour=colour, title=module_title(members),
                         members=members.index.tolist()))
    iso = d[d.isolated].sort_values("minus_log10_fdr", ascending=False)
    ix, iy = ISOLATED_CENTRE.get(crest, (2.25, -0.35))
    for k, idx in enumerate(iso.index):
        d.loc[idx, ["x", "y"]] = ix, iy - 0.55 * k
        d.loc[idx, "module_colour"] = ISOLATED
    return d, info


class LabelPlacer:
    """Greedy label placement in display space: no overlaps, stays inside the axes."""

    def __init__(self, fig, ax):
        self.fig, self.ax = fig, ax
        self.r = fig.canvas.get_renderer()
        self.occupied: list[Bbox] = []
        self.centres: dict = {}        # node id -> (px, py, radius_px)
        self.ax_box = ax.get_window_extent(self.r)
        self.pad = 0.6 * fig.dpi / 72

    def bbox(self, artist):
        return artist.get_window_extent(self.r).expanded(1.0, 1.0)

    def free(self, box):
        b = self.ax_box
        inside = box.x0 >= b.x0 + 1 and box.x1 <= b.x1 - 1 and box.y0 >= b.y0 + 1 and box.y1 <= b.y1 - 1
        if not inside:
            return False
        grown = Bbox.from_extents(box.x0 - self.pad, box.y0 - self.pad, box.x1 + self.pad, box.y1 + self.pad)
        return not any(grown.overlaps(o) for o in self.occupied)

    def add_circle(self, x, y, area, key=None):
        px, py = self.ax.transData.transform((x, y))
        rad = math.sqrt(area / math.pi) * self.fig.dpi / 72 + 0.45 * self.fig.dpi / 72
        self.occupied.append(Bbox.from_extents(px - rad, py - rad, px + rad, py + rad))
        if key is not None:
            self.centres[key] = (px, py, rad)
        return rad

    @staticmethod
    def gap_to(box, px, py, rad):
        dx = max(box.x0 - px, 0, px - box.x1)
        dy = max(box.y0 - py, 0, py - box.y1)
        return math.hypot(dx, dy) - rad

    def owns(self, box, key):
        own = self.gap_to(box, *self.centres[key])
        return all(own < self.gap_to(box, *c) - 0.3 * self.fig.dpi / 72
                   for k, c in self.centres.items() if k != key)

    def place(self, x, y, rad_px, texts, angles, key=None, **kw):
        """Try each text variant at each angle; keep the first free spot."""
        px, py = self.ax.transData.transform((x, y))
        gap = rad_px + 0.8 * self.fig.dpi / 72
        inv = self.ax.transData.inverted()
        best = None                      # (overlap, txt, tx, ty, ha, va) inside the axes
        passes = [(e, True) for e in (0.0, 1.5)] + [(e, False) for e in (0.0, 1.5, 3.0, 5.0)]
        for extra, strict in passes:
            for txt in texts:
                for a in angles:
                    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
                    ha = "left" if c > 0.3 else "right" if c < -0.3 else "center"
                    va = "bottom" if s > 0.3 else "top" if s < -0.3 else "center"
                    g = gap + extra * self.fig.dpi / 72
                    tx, ty = inv.transform((px + c * g, py + s * g))
                    t = self.ax.text(tx, ty, txt, ha=ha, va=va,
                                     multialignment=ha if ha != "center" else "center",
                                     linespacing=1.0, zorder=8, **kw)
                    box = self.bbox(t)
                    t.remove()
                    if self.free(box) and (not strict or key is None or self.owns(box, key)):
                        t = self.ax.text(tx, ty, txt, ha=ha, va=va,
                                         multialignment=ha if ha != "center" else "center",
                                         linespacing=1.0, zorder=8, **kw)
                        self.occupied.append(box)
                        if key is not None and not self.owns(box, key):
                            self.leader(key, box)
                        return t, True
                    b = self.ax_box
                    if box.x0 >= b.x0 and box.x1 <= b.x1 and box.y0 >= b.y0 and box.y1 <= b.y1:
                        ov = sum(i.width * i.height for o in self.occupied
                                 if (i := Bbox.intersection(box, o)) is not None)
                        if best is None or ov < best[0]:
                            best = (ov, txt, tx, ty, ha, va)
        if best is None:
            raise RuntimeError(f"label cannot fit inside the network axes: {texts[0]!r}")
        _, txt, tx, ty, ha, va = best
        t = self.ax.text(tx, ty, txt, ha=ha, va=va, linespacing=1.0, zorder=8, **kw)
        self.occupied.append(self.bbox(t))
        if key is not None:
            self.leader(key, self.bbox(t))
        return t, False

    def leader(self, key, box):
        """Thin grey line from the node edge to the nearest point of its label."""
        px, py, rad = self.centres[key]
        qx, qy = min(max(px, box.x0), box.x1), min(max(py, box.y0), box.y1)
        L = math.hypot(qx - px, qy - py)
        if L <= rad:
            return
        sx, sy = px + (qx - px) * rad / L, py + (qy - py) * rad / L
        inv = self.ax.transData.inverted()
        (x0, y0), (x1, y1) = inv.transform((sx, sy)), inv.transform((qx, qy))
        self.ax.plot([x0, x1], [y0, y1], color=GREY, lw=0.3, zorder=6)


def wrap_variants(label, width=20):
    out = [label]
    if len(label) > 14:
        out.append("\n".join(textwrap.wrap(label, width=width, break_on_hyphens=False)))
        out.append("\n".join(textwrap.wrap(label, width=14, break_on_hyphens=False)))
    return list(dict.fromkeys(out))


def draw_network(fig, ax, nodes, edges, crest, numbered, row_index, report):
    ax_w, ax_h = NET_W, NET_Y[crest][1] - NET_Y[crest][0]
    xlim = NET_YLIM * ax_w / ax_h
    d, modules = network_layout(nodes, crest, xlim)
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-NET_YLIM, NET_YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    d["area"] = node_area(d["gene_count"].to_numpy(float))

    # Module backgrounds: ellipse covering all nodes + 0.3 units
    for m in modules:
        pts = d.loc[m["members"], ["x", "y"]].to_numpy()
        rx = max(np.abs(pts[:, 0] - m["cx"]).max(), 0.0) + 0.3
        ry = max(np.abs(pts[:, 1] - m["cy"]).max(), 0.0) + 0.3
        m.update(rx=rx, ry=ry)
        ax.add_patch(Ellipse((m["cx"], m["cy"]), 2 * rx, 2 * ry, facecolor=m["colour"],
                             alpha=0.13, edgecolor="none", zorder=0))

    # Edges
    pos = d.set_index("term_uid")[["x", "y"]]
    e = edges[(edges.crest_id == crest) & edges.from_uid.isin(pos.index) & edges.to_uid.isin(pos.index)]
    for _, r in e.iterrows():
        (x0, y0), (x1, y1) = pos.loc[r.from_uid], pos.loc[r.to_uid]
        ax.plot([x0, x1], [y0, y1], color=EDGE_COLOUR, alpha=EDGE_ALPHA,
                lw=0.25 + 0.9 * float(r.similarity), solid_capstyle="round", zorder=1)

    # Nodes
    fill = FDR_CMAP(FDR_NORM(d["minus_log10_fdr"].to_numpy(float)))
    ax.scatter(d.x, d.y, s=d.area, c=fill, edgecolors=d.module_colour, linewidths=0.9, zorder=5)

    fig.canvas.draw()
    placer = LabelPlacer(fig, ax)
    rad = {idx: placer.add_circle(r.x, r.y, r.area, key=idx) for idx, r in d.iterrows()}

    # Module titles first (above modules in the upper half, below otherwise); off by default
    for m in (modules if SHOW_MODULE_TITLES else []):
        above = m["cy"] >= 0
        ty = m["cy"] + m["ry"] + 0.06 if above else m["cy"] - m["ry"] - 0.06
        t = ax.text(m["cx"], ty, m["title"], ha="center", va="bottom" if above else "top",
                    fontsize=FS_MODULE, fontweight="bold", color=m["colour"], linespacing=1.0, zorder=9)
        box = placer.bbox(t)
        b = placer.ax_box
        shift = max(0, b.x0 + 2 - box.x0) - max(0, box.x1 - (b.x1 - 2))   # keep inside horizontally
        if shift:
            px, py = ax.transData.transform((m["cx"], ty))
            t.set_position(ax.transData.inverted().transform((px + shift, py)))
            box = placer.bbox(t)
        if not placer.free(box):
            report.append(f"Crest {crest}: module title '{m['title']}' touches another element")
        placer.occupied.append(box)

    # Node labels
    # Nodes closest to their module centre have the fewest free spots: label them first.
    centre = {i: (mm["cx"], mm["cy"]) for mm in modules for i in mm["members"]}
    d["_r"] = [math.hypot(r.x - centre[i][0], r.y - centre[i][1]) if i in centre else 9.0
               for i, r in d.iterrows()]
    order = d.sort_values(["_r", "minus_log10_fdr"], ascending=[True, False]).index
    for idx in order:
        r = d.loc[idx]
        if numbered:
            k = row_index[r.term_uid]
            dark = FDR_NORM(r.minus_log10_fdr) > 0.55
            ax.text(r.x, r.y, str(k), ha="center", va="center", fontsize=4.2,
                    color="white" if dark else TEXT, zorder=7)
            continue
        label = shorten(r.description)
        if r.isolated:
            angles = [-90, -60, -120, 0, 180]
            colour = GREY
        else:
            m = next(mm for mm in modules if idx in mm["members"])
            base = math.degrees(math.atan2(r.y - m["cy"], r.x - m["cx"])) if (r.x, r.y) != (m["cx"], m["cy"]) else -90
            angles = [base + k for k in (0, 20, -20, 40, -40, 60, -60, 80, -80, 100, -100,
                                         120, -120, 140, -140, 160, -160, 180)]
            colour = TEXT
        _, ok = placer.place(r.x, r.y, rad[idx], wrap_variants(label), angles, key=idx,
                             fontsize=FS_NODE, color=colour)
        if not ok:
            report.append(f"Crest {crest}: no free spot for '{label}'")
    return d, modules


# ----------------------------------------------------------------------------
# 8. Legends
# ----------------------------------------------------------------------------
def draw_legends(fig):
    # Heatmap colour bar
    cax = fig.add_axes(rect(HEAT_CB_X, LEG_BAR_Y, HEAT_CB_W, LEG_BAR_Y + LEG_BAR_H))
    lim = HEAT_SIGNED_NORM.vmax
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=HEAT_SIGNED_NORM, cmap=HEAT_CMAP),
                      cax=cax, orientation="horizontal",
                      ticks=_nice_ticks(lim))
    style_cbar(cb)
    fig.text(fx(HEAT_CB_X), fy_top(LEG_TITLE_Y), "Signed coherence score (SD, row-centred)", fontsize=FS_LEG,
             ha="left", va="bottom", color=TEXT)
    # fig.text(fx(HEAT_CB_X + HEAT_CB_W + 4.0), fy_top(LEG_BAR_Y + LEG_BAR_H / 2),
    #          "* crest step > random gene sets (P < 0.05)", fontsize=FS_NOTE,
    #          ha="left", va="center", color=GREY)
    # FDR colour bar
    fax = fig.add_axes(rect(FDR_CB_X, LEG_BAR_Y, FDR_CB_W, LEG_BAR_Y + LEG_BAR_H))
    cb2 = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(*FDR_RANGE), cmap=FDR_CMAP),
                       cax=fax, orientation="horizontal", ticks=[2, 6, 10, 14])
    style_cbar(cb2)
    fig.text(fx(FDR_CB_X), fy_top(LEG_TITLE_Y), "−log$_{10}$ FDR", fontsize=FS_LEG,
             ha="left", va="bottom", color=TEXT)
    # Node-size legend
    fig.text(fx(SIZE_LEG_X), fy_top(LEG_TITLE_Y), "Genes per term", fontsize=FS_LEG,
             ha="left", va="bottom", color=TEXT)
    sax = fig.add_axes(rect(SIZE_LEG_X, LEG_BAR_Y - 0.6, 24.0, LEG_BAR_Y + LEG_BAR_H + 0.6))
    sax.set_xlim(0, 24.0)
    sax.set_ylim(-1, 1)
    sax.axis("off")
    for x, c in zip((1.6, 8.8, 16.4), (10, 40, 80)):
        sax.scatter([x], [0], s=node_area(c), facecolor="#D9D9D9", edgecolor="#8C8C8C",
                    linewidths=0.5, clip_on=False)
        sax.text(x + 2.5, 0, str(c), fontsize=FS_LEG, ha="left", va="center", color=TEXT)


def _nice_ticks(lim):
    step = 0.1 if lim >= 0.2 else 0.05 if lim >= 0.1 else 0.02
    k = int(np.floor(lim / step + 1e-9))
    return [round(i * step, 3) for i in range(-k, k + 1)]


def style_cbar(cb):
    cb.outline.set_linewidth(0.3)
    cb.ax.tick_params(labelsize=5, length=1.5, width=0.3, pad=1)
    cb.ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: (f"{v:.2f}".rstrip("0").rstrip(".") if abs(v) < 1 else f"{v:g}").replace("-", "\u2212")))


# ----------------------------------------------------------------------------
# 9. Checks
# ----------------------------------------------------------------------------
def check_layout(fig, report):
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    boxes = []
    for t in texts:
        b = t.get_window_extent(r)
        if b.width == 0:
            continue
        if b.x0 < fb.x0 - 0.5 or b.x1 > fb.x1 + 0.5 or b.y0 < fb.y0 - 0.5 or b.y1 > fb.y1 + 0.5:
            report.append(f"clipped by canvas: {t.get_text()!r}")
        boxes.append((t, b))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (t1, b1), (t2, b2) = boxes[i], boxes[j]
            if b1.overlaps(b2):
                inter = Bbox.intersection(b1, b2)
                if inter is not None and inter.width > 0.5 and inter.height > 0.5:
                    report.append(f"text overlap: {t1.get_text()!r} / {t2.get_text()!r}")


# ----------------------------------------------------------------------------
# 10. Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=str(SCRIPT_DIR / "outputs" / "transcriptomics_pathway"))
    ap.add_argument("--figure-dir", default=str(SCRIPT_DIR / "figures" / "transcriptomics"))
    ap.add_argument("--output-dir", default=str(SCRIPT_DIR / "outputs" / "transcriptomics_pathway"),
                    help="folder for the result tables (CSV) and the layout check")
    ap.add_argument("--expression", default=str(SCRIPT_DIR.parent / "01-interim_data" / "omics_data" /
                                                "transcriptomics_with_metadata.csv"))
    ap.add_argument("--half-window", type=float, default=SIGNED_HALF_WINDOW)
    ap.add_argument("--null-sets", type=int, default=200, help="random gene sets per pathway (0 = skip)")
    ap.add_argument("--null-pool", type=int, default=1000, help="random non-crest transcripts to draw from")
    ap.add_argument("--seed", type=int, default=20261023)
    args = ap.parse_args()
    base, figdir, outdir = Path(args.base_dir), Path(args.figure_dir), Path(args.output_dir)
    figdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    configure_fonts()
    heat, nodes, edges = load_inputs(base)
    fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM), dpi=600)
    report: list[str] = []

    crests = sorted(heat.crest_id.unique())
    key = pd.read_csv(base / "pathway_expression_heatmap" /
                      "pathway_molecular_expression_heatmap_pathway_key.csv")
    crest_genes = pd.read_csv(base / "kegg_enrichment" / "crest_associated_transcripts.csv")["feature"]
    mats, signed_long, gene_dirs, signed_stats = build_signed(
        heat, key, crest_genes.astype(str), Path(args.expression), args.half_window,
        args.null_sets, args.null_pool, args.seed, report)
    global HEAT_SIGNED_NORM
    allv = np.concatenate([np.abs(m[0]).ravel() for m in mats.values()])
    lim = max(0.05, math.ceil(np.quantile(allv, 0.99) / 0.05) * 0.05)
    HEAT_SIGNED_NORM = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    # Row-label space: measure the longest label; shift/narrow heatmaps if needed.
    r = fig.canvas.get_renderer()
    probe = fig.text(0, 0, max((l for c in crests for l in mats[c][2]), key=len), fontsize=FS_ROW)
    widest = max(fig.text(0, 0, l, fontsize=FS_ROW).get_window_extent(r).width
                 for c in crests for l in mats[c][2]) / fig.dpi * 25.4
    for t in list(fig.texts):
        t.remove()
    del probe
    need = ROW_LABEL_MARGIN + widest + 1.5 / 72 * 25.4 + 0.3
    left = max(HEAT_LEFT, need)
    width = HEAT_W - (left - HEAT_LEFT)
    if left > HEAT_LEFT:
        report.append(f"heatmaps shifted right to {left:.1f} mm (width {width:.1f} mm) for row labels")

    net_sources, heat_sources = [], []
    for i, c in enumerate(crests):
        mat, ages, labels, peak, heat_terms, stars = mats[c]
        net_terms = set(nodes.loc[nodes.crest_id == c, "term_uid"])
        numbered = net_terms == heat_terms
        hx = fig.add_axes(rect(left, HEAT_Y[c][0], width, HEAT_Y[c][1]))
        draw_heatmap(hx, mat, ages, labels, peak, c, numbered, first=i == 0)
        # for row, star in enumerate(stars):
        #     if star:
        #         hx.annotate("*", xy=(1, row), xycoords=("axes fraction", "data"), xytext=(1.2, -0.8),
        #                     textcoords="offset points", ha="left", va="center",
        #                     fontsize=6, color=TEXT)
        nx = fig.add_axes(rect(NET_LEFT, NET_Y[c][0], NET_W, NET_Y[c][1]))
        row_index = {}
        if numbered:
            order = heat[heat.crest_id == c].drop_duplicates("pathway_rank").sort_values("pathway_rank")
            row_index = {u.split("::", 1)[1]: k + 1 for k, u in enumerate(order.pathway_uid)}
        d, _ = draw_network(fig, nx, nodes, edges, c, numbered, row_index, report)
        d = d.assign(label=d.description.map(shorten))
        net_sources.append(d[["crest_id", "term_uid", "description", "label", "community", "isolated",
                              "minus_log10_fdr", "gene_count", "x", "y"]])

    # fig.text(fx(1.0), fy_top(0.0), "b", fontsize=FS_LETTER, fontweight="bold", ha="left", va="top",
    #          color="black")
    draw_legends(fig)
    fig.canvas.draw()
    check_layout(fig, report)

    stem = figdir / "transcriptomics_crests_signed_coherence"            # figures
    table_stem = outdir / "transcriptomics_crests_signed_coherence"      # result tables
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.svg")
    fig.savefig(f"{stem}.png", dpi=600)
    plt.close(fig)
    signed_long.to_csv(f"{table_stem}_source_data_heatmap.csv", index=False)
    gene_dirs.to_csv(f"{table_stem}_gene_directions.csv", index=False)
    signed_stats.to_csv(f"{table_stem}_pathway_statistics.csv", index=False)
    pd.concat(net_sources).to_csv(f"{table_stem}_source_data_network.csv", index=False)
    Path(f"{table_stem}_layout_check.txt").write_text("\n".join(report) or "no layout issues found\n",
                                               encoding="utf-8")
    print("\n".join(report) if report else "layout check: no issues")
    print(f"Saved {stem}.pdf / .svg / .png (600 dpi)")
    print(f"Saved tables to {outdir}")


if __name__ == "__main__":
    main()
