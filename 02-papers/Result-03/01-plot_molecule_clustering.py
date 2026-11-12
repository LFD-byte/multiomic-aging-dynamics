"""Step 2 · Draw the molecular trajectory-cluster figures from saved results.

Reads the results written by ``01-molecule_clustering.py`` from
``outputs/molecule_clustering/`` — no GAM refitting, no re-clustering — and
exports two 210 mm wide figures (no title, no panel letter) to
``figures/molecular_age_patterns/`` as PDF and 600-dpi PNG:

    molecular_age_patterns_{method}_ncomms_1x5_210mm                     (210 × 65 mm)
    molecular_age_patterns_{method}_cluster_solution_supplement_210mm    (210 × 70 mm)

Usage::

    python 01-plot_molecule_clustering.py
    python 01-plot_molecule_clustering.py --method kmeans --height-main 85 --height-supp 70

Colours: ``OMICS_COLORS``; font sizes: ``FS``.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np
import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs" / "molecule_clustering"
FIGURE_DIR = HERE / "figures" / "molecular_age_patterns"

# Must match 01-molecule_clustering.py
AGE_MIN, AGE_MAX = 40.0, 70.0
N_GRID = 100
AGE_GRID = np.linspace(AGE_MIN, AGE_MAX, N_GRID)

OMICS_COLORS = {
    "Transcriptomics": "#CB6E70",
    "Metabolomics": "#68AADD",
    "Lipidomics": "#54AD94",
    "Gut microbiome": "#DDA062",
    "Oral microbiome": "#896EAE",
}
OMICS_ORDER = [
    "Transcriptomics", "Lipidomics", "Metabolomics",
    "Gut microbiome", "Oral microbiome",
]


FONT = "Arial"


def _register_arial() -> None:
    """Register Arial from ~/.fonts and fail loudly instead of silently substituting."""
    for f in glob.glob(os.path.expanduser("~/.fonts/*.ttf")):
        font_manager.fontManager.addfont(f)
    for weight in ("normal", "bold"):
        font_manager.findfont(
            font_manager.FontProperties(family=FONT, weight=weight),
            fallback_to_default=False,
        )


def set_ncomms_style() -> None:
    _register_arial()
    mpl.rcParams.update({
        "font.family": FONT,
        "font.size": 7,
        "axes.linewidth": 0.65,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 6.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def _nice_symmetric_limit(curves: np.ndarray) -> float:
    q = float(np.nanpercentile(np.abs(curves), 99.5))
    return max(1.0, np.ceil(q * 2) / 2)


def _save_publication_figure(fig: plt.Figure, stem: Path) -> None:
    """Save a vector PDF and a 600-dpi RGB PNG of a Matplotlib figure."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    # Member trajectories are deliberately rasterized to avoid a huge vector
    # file. Explicit DPI is essential: PDF otherwise uses the low figure DPI,
    # making very transparent individual curves disappear in some viewers.
    fig.savefig(stem.with_suffix(".pdf"), dpi=600)
    png_path = stem.with_suffix(".png")
    fig.savefig(png_path, dpi=600)
    plt.close(fig)
    # Nature requests RGB artwork; Matplotlib otherwise writes an alpha channel.
    with Image.open(png_path) as image:
        png_rgb = image.convert("RGB")
    png_rgb.save(png_path, dpi=(600, 600), optimize=True)
    print(f"Saved figure to {stem}.[pdf|png]")


def _cluster_omics_heat(info: dict[str, object]) -> np.ndarray:
    members = np.asarray(info["members"])
    omics = np.asarray(info["omics"])
    heat = np.full((len(OMICS_ORDER), N_GRID), np.nan, dtype=float)
    for row, omic in enumerate(OMICS_ORDER):
        mask = omics == omic
        if mask.any():
            heat[row] = members[mask].mean(axis=0)
    return heat


# --------------------------------------------------------------------------- #
# Publication figures (fixed 210 mm width)
# --------------------------------------------------------------------------- #
MM = 1 / 25.4
FIG_WIDTH_MM = 210.0
FIG_WIDTH = FIG_WIDTH_MM * MM  # inches

# Font sizes (pt) at final print size. Nature Portfolio: 5–7 pt for all text.
FS = {
    "cluster_id": 7.0,     # "Cluster 1"
    "cluster_n": 6.0,      # "(2,180 molecules)"
    "bar_pct": 5.5,
    "heat_ytick": 6.0,
    "tick": 6.0,
    "axis_label": 7.0,
    "legend": 6.0,
    "legend_title": 6.5,
    "colorbar": 5.5,
    "annot": 6.0,
}


def load_results(method: str = "agglomerative"):
    """Read saved clustering results; return curves, assignments, silhouette, cluster_info."""
    curves = np.load(OUTPUT_DIR / "gam_curves.npz")["curves"]
    assignments = pd.read_csv(OUTPUT_DIR / f"molecule_cluster_assignments_{method}.csv")
    silhouette_scores = pd.read_csv(OUTPUT_DIR / f"silhouette_scores_{method}.csv")
    saved_centers = pd.read_csv(OUTPUT_DIR / f"cluster_center_curves_{method}.csv")
    metadata = pd.read_csv(OUTPUT_DIR / "gam_curve_metadata.csv")

    assert len(curves) == len(assignments), (curves.shape, assignments.shape)
    assert curves.shape[1] == len(AGE_GRID)
    assert metadata[["feature", "omics"]].equals(assignments[["feature", "omics"]]), \
        "Assignment table is not in the same order as gam_curves.npz"

    cluster_info: list[dict[str, object]] = []
    for pid, sub in assignments.groupby("pattern_id", sort=True):
        mask = (assignments["pattern_id"] == pid).to_numpy()
        members = curves[mask]
        center = members.mean(axis=0)
        assert np.allclose(center, saved_centers[f"pattern_{pid}"].to_numpy(), atol=1e-4), pid
        cluster_info.append({
            "cluster_id": int(sub["cluster_id"].iloc[0]),
            "pattern_id": int(pid),
            "members": members,
            "center": center,
            "omics": assignments.loc[mask, "omics"].to_numpy(),
            "n": int(mask.sum()),
            "label": str(sub["pattern"].iloc[0]),
        })
    return curves, assignments, silhouette_scores, cluster_info


def _label_color(hex_color: str) -> str:
    """White or dark text, whichever contrasts better with ``hex_color``."""
    rgb = [int(hex_color.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "white" if (1.05 / (lum + 0.05)) >= ((lum + 0.05) / 0.0715) else "#303237"


def _draw_composition_bar(
    ax: plt.Axes, omics: np.ndarray, label_threshold: float = 0.12
) -> None:
    track = FancyBboxPatch(
        (0, -0.37), 1, 0.74, boxstyle="round,pad=0,rounding_size=0.075",
        facecolor="#ECEFF2", edgecolor="none", transform=ax.transData,
    )
    ax.add_patch(track)
    left = 0.0
    for omic in OMICS_ORDER:
        fraction = float(np.mean(omics == omic))
        if fraction <= 0:
            continue
        segment = ax.barh(
            0, fraction, left=left, height=0.74, color=OMICS_COLORS[omic],
            edgecolor="white", linewidth=0.4, rasterized=True, zorder=1,
        ).patches[0]
        segment.set_clip_path(track)
        if fraction >= label_threshold:
            ax.text(left + fraction / 2, 0, f"{fraction * 100:.0f}%",
                    ha="center", va="center", fontsize=FS["bar_pct"],
                    fontweight="bold", color=_label_color(OMICS_COLORS[omic]), zorder=2)
        left += fraction
    ax.set(xlim=(0, 1), ylim=(-0.5, 0.5))
    ax.axis("off")


def plot_clusters_1x5(
    curves: np.ndarray,
    cluster_info: list[dict[str, object]],
    method: str = "agglomerative",
    height_mm: float = 65.0,
    stem_suffix: str = "_210mm",
) -> Path:
    """Main-text 1 × 5 trajectory figure, 210 mm wide, no title / panel letter.

    Vertical layout is fixed in millimetres so that spacing stays even when
    ``height_mm`` changes: header row (molecule-class legend + colour bar),
    five cluster panels, footer row (age label + line legend).
    """
    set_ncomms_style()
    if len(cluster_info) != 5:
        raise ValueError("The 1 × 5 layout expects five clusters.")

    fig_w_mm, fig_h_mm = FIG_WIDTH_MM, height_mm
    fig = plt.figure(figsize=(FIG_WIDTH, fig_h_mm * MM))
    renderer = fig.canvas.get_renderer()
    fx = lambda mm: mm / fig_w_mm  # noqa: E731  (mm → figure fraction)
    fy = lambda mm: mm / fig_h_mm  # noqa: E731

    def text_width_mm(text: str, size: float, weight: str = "normal") -> float:
        t = fig.text(0, 0, text, fontsize=size, fontweight=weight)
        width = t.get_window_extent(renderer).width / fig.dpi * 25.4
        t.remove()
        return width

    # ---- layout (mm) ------------------------------------------------------ #
    edge = 2.0                               # outer margin
    header_y = fig_h_mm - edge - 1.6         # centre line of the header row
    panel_top = fig_h_mm - 8.0               # top of the cluster titles
    footer_y = edge + 1.4                    # centre line of the footer row
    panel_bottom = 10.0                      # bottom of the trajectory axes
    label_w = max(text_width_mm(o, FS["heat_ytick"]) for o in OMICS_ORDER)
    panel_left = edge + label_w + 1.2        # room for the heatmap row names
    panel_right = fig_w_mm - 3.0             # room for the last "70" tick

    outer = gridspec.GridSpec(
        1, 5, figure=fig, left=fx(panel_left), right=fx(panel_right),
        bottom=fy(panel_bottom), top=fy(panel_top), wspace=0.16,
    )
    y_limit = _nice_symmetric_limit(curves)
    heats = [_cluster_omics_heat(info) for info in cluster_info]
    heat_values = np.concatenate([x[np.isfinite(x)] for x in heats])
    heat_limit = max(1.0, np.ceil(np.percentile(np.abs(heat_values), 98) * 2) / 2)
    heat_cmap = LinearSegmentedColormap.from_list(
        "omics_mean_1x5", ["#3F6596", "#F7F3DF", "#C75C4D"]
    )
    heat_cmap.set_bad("#ECEDEF")
    heat_norm = TwoSlopeNorm(vmin=-heat_limit, vcenter=0, vmax=heat_limit)
    heat_image = None

    for i, info in enumerate(cluster_info):
        inner = gridspec.GridSpecFromSubplotSpec(
            4, 1, subplot_spec=outer[0, i],
            height_ratios=[0.075, 0.08, 0.26, 0.585], hspace=0.10,
        )
        ax_title = fig.add_subplot(inner[0])
        ax_bar = fig.add_subplot(inner[1])
        ax_heat = fig.add_subplot(inner[2])
        ax = fig.add_subplot(inner[3])

        # "Cluster 1" in bold, followed by "(2,180 molecules)" in regular weight.
        ax_title.axis("off")
        cluster_text = ax_title.text(
            0, 0.5, f'Cluster {info["pattern_id"]}', ha="left", va="center",
            fontsize=FS["cluster_id"], fontweight="bold",
        )
        ax_title.annotate(
            f'({info["n"]:,} molecules)', xy=(1, 0.5), xycoords=cluster_text,
            xytext=(2.0, 0), textcoords="offset points", ha="left", va="center",
            fontsize=FS["cluster_n"], color="#303237",
        )

        _draw_composition_bar(ax_bar, np.asarray(info["omics"]))

        heat_image = ax_heat.imshow(
            heats[i], aspect="auto", interpolation="nearest",
            extent=(AGE_MIN, AGE_MAX, len(OMICS_ORDER) - 0.5, -0.5),
            cmap=heat_cmap, norm=heat_norm, rasterized=True,
        )
        ax_heat.set(xlim=(AGE_MIN, AGE_MAX), xticks=[])
        ax_heat.set_yticks(np.arange(len(OMICS_ORDER)))
        ax_heat.set_yticklabels(OMICS_ORDER if i == 0 else [])
        ax_heat.tick_params(axis="y", length=0, pad=1.5, labelsize=FS["heat_ytick"])
        for boundary in np.arange(0.5, len(OMICS_ORDER), 1):
            ax_heat.axhline(boundary, color="white", lw=0.45)
        for spine in ax_heat.spines.values():
            spine.set_visible(False)

        center = np.asarray(info["center"])
        members = np.asarray(info["members"])
        ax.plot(AGE_GRID, members.T, color="#9EA6AF", lw=0.26,
                alpha=min(0.14, max(0.012, 7 / max(len(members), 1))),
                rasterized=True, zorder=1)
        ax.plot(AGE_GRID, center, color="#151515", lw=1.5, zorder=3)
        ax.axhline(0, color="#C8CCD0", lw=0.5, ls=(0, (2, 2)), zorder=0)
        ax.set_facecolor("#FCFCFC")
        ax.set(xlim=(AGE_MIN, AGE_MAX), ylim=(-y_limit, y_limit))
        ax.set_xticks([40, 55, 70])
        ax.set_yticks([-3, 0, 3])
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", pad=1.5, labelsize=FS["tick"])
        if i == 0:
            ax.set_ylabel("Standardized GAM fit", fontsize=FS["axis_label"], labelpad=2)
        else:
            ax.set_yticklabels([])

    # ---- header: molecule-class legend (left) + colour bar (right) -------- #
    title_w = text_width_mm("Molecule class", FS["legend_title"], "bold")
    fig.text(fx(edge), fy(header_y), "Molecule class", ha="left", va="center",
             fontsize=FS["legend_title"], fontweight="bold")
    handles = [Patch(facecolor=OMICS_COLORS[o], edgecolor="none", label=o) for o in OMICS_ORDER]
    fig.legend(
        handles=handles, loc="center left",
        bbox_to_anchor=(fx(edge + title_w + 1.5), fy(header_y)),
        ncol=5, frameon=False, handlelength=0.8, handleheight=0.8,
        handletextpad=0.4, columnspacing=1.0, borderaxespad=0, borderpad=0,
        fontsize=FS["legend"],
    )

    if heat_image is not None:
        cbar_w, cbar_h = 22.0, 1.8
        cbar_x = panel_right - cbar_w
        cax = fig.add_axes([fx(cbar_x), fy(header_y - cbar_h / 2), fx(cbar_w), fy(cbar_h)])
        colorbar = fig.colorbar(heat_image, cax=cax, orientation="horizontal")
        colorbar.set_ticks([-heat_limit, 0, heat_limit])
        colorbar.ax.tick_params(labelsize=FS["colorbar"], length=1.5, width=0.4, pad=0.8)
        colorbar.outline.set_linewidth(0.4)
        fig.text(fx(cbar_x - 1.5), fy(header_y), "Omics mean GAM fit",
                 ha="right", va="center", fontsize=FS["legend"])

    # ---- footer: age label (centred) + line legend (right) ---------------- #
    fig.text(fx((panel_left + panel_right) / 2), fy(footer_y), "Age (years)",
             ha="center", va="center", fontsize=FS["axis_label"])
    line_handles = [
        Line2D([0], [0], color="#9EA6AF", lw=0.8, label="Individual molecule"),
        Line2D([0], [0], color="#151515", lw=1.5, label="Cluster mean"),
    ]
    fig.legend(handles=line_handles, loc="center right",
               bbox_to_anchor=(fx(panel_right), fy(footer_y)),
               ncol=2, frameon=False, handlelength=1.6, handletextpad=0.5,
               columnspacing=1.2, borderaxespad=0, borderpad=0, fontsize=FS["legend"])

    stem = FIGURE_DIR / f"molecular_age_patterns_{method}_ncomms_1x5{stem_suffix}"
    _save_publication_figure(fig, stem)
    return stem


def plot_cluster_solution_supplement(
    cluster_info: list[dict[str, object]],
    silhouette_scores: pd.DataFrame,
    method: str = "agglomerative",
    height_mm: float = 70.0,
    stem_suffix: str = "_210mm",
) -> Path:
    """Silhouette score and pattern prevalence, 210 mm wide, no titles."""
    set_ncomms_style()
    fig, (ax_sil, ax_prev) = plt.subplots(1, 2, figsize=(FIG_WIDTH, height_mm * MM))
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.19, top=0.96, wspace=0.30)

    xk = silhouette_scores["k"].to_numpy()
    ys = silhouette_scores["silhouette_score"].to_numpy()
    best = int(np.argmax(ys))
    ax_sil.plot(xk, ys, color="#45484D", lw=1.4, marker="o", ms=4.5)
    ax_sil.scatter(xk[best], ys[best], s=55, color="#5B79C7", zorder=3)
    ax_sil.annotate(f"selected k = {xk[best]}", xy=(xk[best], ys[best]),
                    xytext=(9, -4), textcoords="offset points", color="#5B79C7",
                    fontsize=FS["annot"], va="top")
    pad = max(0.008, (ys.max() - ys.min()) * 0.2)
    ax_sil.set(xlim=(xk.min() - 0.25, xk.max() + 0.25),
               ylim=(ys.min() - pad, ys.max() + pad))
    ax_sil.set_xlabel("Number of clusters", fontsize=FS["axis_label"])
    ax_sil.set_ylabel("Silhouette score", fontsize=FS["axis_label"])
    ax_sil.set_xticks(xk)
    ax_sil.grid(axis="y", color="#E2E4E7", lw=0.5)
    ax_sil.spines[["top", "right"]].set_visible(False)
    ax_sil.tick_params(labelsize=FS["tick"])

    total_n = sum(int(x["n"]) for x in cluster_info)
    prevalence = np.array([100 * int(x["n"]) / total_n for x in cluster_info])
    ypos = np.arange(len(cluster_info))
    ax_prev.barh(ypos, prevalence, color="#A9B9DE", height=0.62, edgecolor="none")
    for y, value in zip(ypos, prevalence):
        ax_prev.text(value + 0.7, y, f"{value:.1f}%", va="center", fontsize=FS["annot"])
    ax_prev.set_yticks(ypos, [f'Cluster {x["pattern_id"]}' for x in cluster_info])
    ax_prev.invert_yaxis()
    ax_prev.set_xlim(0, max(prevalence) * 1.18)
    ax_prev.set_xlabel("Molecules in pattern (%)", fontsize=FS["axis_label"])
    ax_prev.spines[["top", "right"]].set_visible(False)
    ax_prev.tick_params(labelsize=FS["tick"])

    stem = FIGURE_DIR / f"molecular_age_patterns_{method}_cluster_solution_supplement{stem_suffix}"
    _save_publication_figure(fig, stem)
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=["agglomerative", "kmeans"], default="agglomerative")
    parser.add_argument("--height-main", type=float, default=65.0, help="1 × 5 figure height (mm)")
    parser.add_argument("--height-supp", type=float, default=70.0, help="supplement height (mm)")
    args = parser.parse_args()

    curves, assignments, silhouette_scores, cluster_info = load_results(args.method)
    print(f"{len(curves):,} molecules × {curves.shape[1]} age points")
    print(assignments.groupby(["pattern_id", "pattern"]).size().rename("n").to_string())
    plot_clusters_1x5(curves, cluster_info, args.method, height_mm=args.height_main)
    plot_cluster_solution_supplement(cluster_info, silhouette_scores, args.method,
                                     height_mm=args.height_supp)


if __name__ == "__main__":
    main()
