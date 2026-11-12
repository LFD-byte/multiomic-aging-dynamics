#!/usr/bin/env python3
# =============================================================================
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
# Output: figures/transcriptomics/transcriptomics_crests.{pdf,svg,png}
#         + _source_data_heatmap.csv, _source_data_network.csv, _layout_check.txt
#
# Canvas 183 x 115 mm; every axes is placed at absolute mm positions with
# fig.add_axes (no tight_layout). Text stays editable (pdf.fonttype 42,
# svg.fonttype none). Arial is registered from ~/.fonts and required, as in
# Result-01/01-cohort_feature.ipynb.
#
# Usage: python 02-plot_transcriptomics.py [--base-dir D] [--figure-dir D]
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
HEAT_NORM = dict(vmin=-2.2, vcenter=0.0, vmax=2.6)
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


def heat_matrix(heat: pd.DataFrame, crest: int):
    d = heat[heat.crest_id == crest]
    wide = d.pivot_table(index="pathway_rank", columns="age",
                         values="pathway_trajectory_z", aggfunc="first").sort_index()
    labels = (d.drop_duplicates("pathway_rank").set_index("pathway_rank")
              .loc[wide.index, "pathway_label"])
    return (wide.to_numpy(float), wide.columns.to_numpy(float),
            [shorten(x) for x in labels], float(d.peak_age.iloc[0]),
            set(d["pathway_uid"].str.split("::", n=1).str[1]))


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
    ax.imshow(mat, cmap=HEAT_CMAP, norm=TwoSlopeNorm(**HEAT_NORM), aspect="auto",
              interpolation="bilinear" if fine else "nearest",
              extent=(x0, x1, n - 0.5, -0.5))
    ax.set_xlim(x0, x1)
    ax.set_ylim(n - 0.5, -0.5)
    ax.axvline(62, color=GREY, lw=0.45, ls=(0, (0.6, 1.2)), zorder=3)
    ax.axvline(peak, color=TEXT, lw=0.7, ls=(0, (3, 1.5)), zorder=4)
    ax.annotate(f"Crest{crest} · {peak:g} y", xy=(peak, 1), xycoords=("data", "axes fraction"),
                xytext=(0, 2.6), textcoords="offset points", ha="center", va="bottom",
                fontsize=FS_TITLE, fontweight="bold", color=TEXT)
    if first:
        ax.annotate("~62 y", xy=(62, 1), xycoords=("data", "axes fraction"),
                    xytext=(0, 0.9), textcoords="offset points", ha="center", va="bottom",
                    fontsize=FS_NOTE, color=GREY)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=FS_ROW, color=TEXT)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.set_xticks(np.arange(40, 71, 5))
    ax.set_xlabel("Age (years)", labelpad=1)
    for s in ax.spines.values():
        s.set_visible(False)
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

    # Module titles first (above modules in the upper half, below otherwise)
    for m in modules:
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
            report.append(f"Crest{crest}: module title '{m['title']}' touches another element")
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
            report.append(f"Crest{crest}: no free spot for '{label}'")
    return d, modules


# ----------------------------------------------------------------------------
# 8. Legends
# ----------------------------------------------------------------------------
def draw_legends(fig):
    # Heatmap colour bar
    cax = fig.add_axes(rect(HEAT_CB_X, LEG_BAR_Y, HEAT_CB_W, LEG_BAR_Y + LEG_BAR_H))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=TwoSlopeNorm(**HEAT_NORM), cmap=HEAT_CMAP),
                      cax=cax, orientation="horizontal", ticks=[-2, -1, 0, 1, 2])
    style_cbar(cb)
    fig.text(fx(HEAT_CB_X), fy_top(LEG_TITLE_Y), "Pathway activity (row z-score)", fontsize=FS_LEG,
             ha="left", va="bottom", color=TEXT)
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


def style_cbar(cb):
    cb.outline.set_linewidth(0.3)
    cb.ax.tick_params(labelsize=5, length=1.5, width=0.3, pad=1)
    cb.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−")))


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
    args = ap.parse_args()
    base, figdir = Path(args.base_dir), Path(args.figure_dir)
    figdir.mkdir(parents=True, exist_ok=True)

    configure_fonts()
    heat, nodes, edges = load_inputs(base)
    fig = plt.figure(figsize=(FIG_W * MM, FIG_H * MM), dpi=600)
    report: list[str] = []

    crests = sorted(heat.crest_id.unique())
    mats = {c: heat_matrix(heat, c) for c in crests}

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
        mat, ages, labels, peak, heat_terms = mats[c]
        net_terms = set(nodes.loc[nodes.crest_id == c, "term_uid"])
        numbered = net_terms == heat_terms
        hx = fig.add_axes(rect(left, HEAT_Y[c][0], width, HEAT_Y[c][1]))
        draw_heatmap(hx, mat, ages, labels, peak, c, numbered, first=i == 0)
        nx = fig.add_axes(rect(NET_LEFT, NET_Y[c][0], NET_W, NET_Y[c][1]))
        row_index = {}
        if numbered:
            order = heat[heat.crest_id == c].drop_duplicates("pathway_rank").sort_values("pathway_rank")
            row_index = {u.split("::", 1)[1]: k + 1 for k, u in enumerate(order.pathway_uid)}
        d, _ = draw_network(fig, nx, nodes, edges, c, numbered, row_index, report)
        d = d.assign(label=d.description.map(shorten))
        net_sources.append(d[["crest_id", "term_uid", "description", "label", "community", "isolated",
                              "minus_log10_fdr", "gene_count", "x", "y"]])
        h = heat[heat.crest_id == c][["crest_id", "pathway_rank", "pathway_label", "age", "pathway_trajectory_z"]]
        heat_sources.append(h.assign(label=h.pathway_label.map(shorten)))

    fig.text(fx(1.0), fy_top(0.0), "b", fontsize=FS_LETTER, fontweight="bold", ha="left", va="top",
             color="black")
    draw_legends(fig)
    fig.canvas.draw()
    check_layout(fig, report)

    stem = figdir / "transcriptomics_crests"
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.svg")
    fig.savefig(f"{stem}.png", dpi=600)
    plt.close(fig)
    pd.concat(heat_sources).to_csv(f"{stem}_source_data_heatmap.csv", index=False)
    pd.concat(net_sources).to_csv(f"{stem}_source_data_network.csv", index=False)
    Path(f"{stem}_layout_check.txt").write_text("\n".join(report) or "no layout issues found\n",
                                               encoding="utf-8")
    print("\n".join(report) if report else "layout check: no issues")
    print(f"Saved {stem}.pdf / .svg / .png (600 dpi)")


if __name__ == "__main__":
    main()
