"""Circular heatmap of nonlinear molecular clusters versus system phenotypes.

Input is the ``cluster_system_associations.csv`` written by
``cluster_system_association.py``.  Sectors are systems, angular cells are
phenotypes, concentric tracks are molecular clusters, cell colour is partial
correlation, and a black dot marks an FDR-significant association.
"""

from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FIGURE_WIDTH_MM = 210.0

SYSTEM_ORDER = [
    "cardio", "body_comp", "bone_density", "diet", "sleep",
    "liver_health", "cgm", "frailty", "life_style", "immune", "hema",
    "renal_function",
]

SYSTEM_LABELS = {
    "cardio": "Cardiovascular\nsystem",
    "body_comp": "Body\ncomposition",
    "bone_density": "Bone\ndensity",
    "diet": "Diet",
    "sleep": "Sleep",
    "liver_health": "Liver\nhealth",
    "cgm": "CGM",
    "frailty": "Frailty",
    "life_style": "Lifestyle",
    "immune": "Immune\nsystem",
    "hema": "Hematopoietic\nsystem",
    "renal_function": "Renal\nfunction",
}

SYSTEM_COLORS = {
    "cardio": "#91BAB8",
    "body_comp": "#49ADD5",
    "bone_density": "#EA843E",
    "diet": "#94C26A",
    "sleep": "#E14434",
    "liver_health": "#EFA06B",
    "cgm": "#9D83C0",
    "frailty": "#E38086",
    "life_style": "#49ADD5",
    "immune": "#EFA06B",
    "hema": "#94C26A",
    "renal_function": "#666666",
}

def _choose_column(frame: pd.DataFrame, requested: str | None, candidates: Sequence[str]) -> str:
    if requested is not None:
        if requested not in frame:
            raise ValueError(f"Requested column {requested!r} is absent")
        return requested
    for column in candidates:
        if column in frame:
            return column
    raise ValueError(f"None of the required columns is present: {list(candidates)}")


def _bh(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid):
        order = valid[np.argsort(p[valid])]
        ranked = p[order] * len(valid) / np.arange(1, len(valid) + 1)
        out[order] = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    return pd.Series(out, index=p_values.index)


def prepare_circle_data(
    associations: pd.DataFrame,
    *,
    effect_col: str | None = None,
    fdr_col: str | None = None,
    top_features_per_system: int | None = 10,
) -> tuple[pd.DataFrame, str, str]:
    """Validate input and optionally retain the most informative phenotypes."""
    required = {"system", "feature", "cluster_id"}
    missing = required.difference(associations.columns)
    if missing:
        raise ValueError(f"Association table is missing columns: {sorted(missing)}")
    effect_col = _choose_column(
        associations, effect_col, ("partial_r", "standardized_beta")
    )
    if fdr_col is None and not any(c in associations for c in ("fdr_global", "fdr_within_system")):
        if "p_value" not in associations:
            raise ValueError("Need fdr_global/fdr_within_system or p_value")
        associations = associations.copy()
        associations["fdr_global"] = _bh(associations["p_value"])
    fdr_col = _choose_column(
        associations, fdr_col, ("fdr_global", "fdr_within_system")
    )

    columns = ["system", "feature", "cluster_id", effect_col, fdr_col]
    columns += [c for c in ("pattern_id", "pattern", "n") if c in associations]
    data = associations[columns].copy()
    data["system"] = data["system"].astype(str)
    data["feature"] = data["feature"].astype(str)
    data[effect_col] = pd.to_numeric(data[effect_col], errors="coerce")
    data[fdr_col] = pd.to_numeric(data[fdr_col], errors="coerce")
    data = data.dropna(subset=["system", "feature", "cluster_id", effect_col])

    duplicates = data.duplicated(["system", "feature", "cluster_id"], keep=False)
    if duplicates.any():
        example = data.loc[duplicates, ["system", "feature", "cluster_id"]].head(3)
        raise ValueError(
            "Duplicate system-feature-cluster rows found; aggregate upstream first. "
            f"Examples: {example.to_dict('records')}"
        )

    if top_features_per_system is not None and top_features_per_system > 0:
        # Rank a phenotype by its best cluster: FDR first, then absolute effect.
        feature_rank = (
            data.assign(_abs_effect=data[effect_col].abs())
            .groupby(["system", "feature"], as_index=False)
            .agg(_min_fdr=(fdr_col, "min"), _max_abs_effect=("_abs_effect", "max"))
            .sort_values(
                ["system", "_min_fdr", "_max_abs_effect", "feature"],
                ascending=[True, True, False, True],
            )
            .groupby("system", sort=False)
            .head(top_features_per_system)
        )
        keep = pd.MultiIndex.from_frame(feature_rank[["system", "feature"]])
        index = pd.MultiIndex.from_frame(data[["system", "feature"]])
        data = data.loc[index.isin(keep)].copy()
    if data.empty:
        raise ValueError("No plottable associations remain")
    return data, effect_col, fdr_col


def _ordered_systems(data: pd.DataFrame) -> list[str]:
    present = data["system"].drop_duplicates().tolist()
    return [s for s in SYSTEM_ORDER if s in present] + [s for s in present if s not in SYSTEM_ORDER]


def _cluster_sort_key(value):
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _cluster_labels(clusters: Sequence) -> dict:
    """Fallback when pattern_id is unavailable: number tracks from one by cluster_id order."""
    return {cluster: f"Cluster {index}" for index, cluster in enumerate(clusters, start=1)}


def _pattern_order_and_labels(data: pd.DataFrame) -> tuple[list, dict]:
    """Order and label tracks by pattern_id (C1-C5), the numbering used in the manuscript.

    cluster_id is the raw, arbitrary clustering label; pattern_id is the curated
    trajectory number shared by every other Result-03 figure.
    """
    mapping = data[["cluster_id", "pattern_id"]].drop_duplicates()
    if mapping["cluster_id"].duplicated().any() or mapping["pattern_id"].duplicated().any():
        raise ValueError(f"cluster_id and pattern_id are not one-to-one: {mapping.to_dict('records')}")
    pattern_of = dict(zip(mapping["cluster_id"], mapping["pattern_id"].astype(int)))
    clusters = sorted(pattern_of, key=pattern_of.get)
    return clusters, {cluster: f"Cluster {pattern_of[cluster]}" for cluster in clusters}


def plot_circle_heatmap(
    associations: pd.DataFrame,
    *,
    effect_col: str | None = None,
    fdr_col: str | None = None,
    fdr_threshold: float = 0.05,
    top_features_per_system: int | None = 10,
    vlim: float | None = None,
    cmap: str = "RdBu_r",
    figsize: tuple[float, float] = (FIGURE_WIDTH_MM / 25.4, FIGURE_WIDTH_MM / 25.4),
    overall_gap_deg: float = 50.0,
    sector_gap_deg: float = 2.0,
    feature_fontsize: float | None = None,
    output_prefix: str | Path | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Draw the publication-oriented circle heatmap and optionally save it."""
    data, effect_col, fdr_col = prepare_circle_data(
        associations,
        effect_col=effect_col,
        fdr_col=fdr_col,
        top_features_per_system=top_features_per_system,
    )
    systems = _ordered_systems(data)
    if "pattern_id" in data:
        clusters, cluster_labels = _pattern_order_and_labels(data)
    else:
        clusters = sorted(data["cluster_id"].drop_duplicates(), key=_cluster_sort_key)
        cluster_labels = _cluster_labels(clusters)

    feature_order: dict[str, list[str]] = {}
    for system in systems:
        feature_order[system] = data.loc[data["system"].eq(system), "feature"].drop_duplicates().tolist()
    n_features = sum(map(len, feature_order.values()))
    if n_features == 0:
        raise ValueError("No phenotype features to plot")

    overall_gap = math.radians(overall_gap_deg)
    internal_gaps = max(0, len(systems) - 1) * math.radians(sector_gap_deg)
    usable = 2 * math.pi - overall_gap - internal_gaps
    if usable <= 0:
        raise ValueError("Angular gaps leave no room for heatmap cells")
    cell_width = usable / n_features
    # Traverse clockwise from the gap to 360 degrees (north / Cartesian 90
    # degrees), leaving the opening to the right of the vertical ending edge.
    theta_cursor = overall_gap
    positions: dict[tuple[str, str], float] = {}
    sector_bounds = {}
    ordered_rows = []
    for system_index, system in enumerate(systems):
        start = theta_cursor
        for feature in feature_order[system]:
            theta = theta_cursor + cell_width / 2
            positions[(system, feature)] = theta
            ordered_rows.append({"system": system, "feature": feature, "theta": theta})
            theta_cursor += cell_width
        sector_bounds[system] = (start, theta_cursor)
        if system_index < len(systems) - 1:
            theta_cursor += math.radians(sector_gap_deg)

    lookup = data.set_index(["system", "feature", "cluster_id"])
    effects = data[effect_col].abs()
    if vlim is None:
        maximum = float(effects.max())
        vlim = max(0.1, math.ceil(maximum * 20) / 20)
    if not np.isfinite(vlim) or vlim <= 0:
        raise ValueError("vlim must be a positive finite number")
    norm = Normalize(vmin=-vlim, vmax=vlim)
    colour_map = mpl.colormaps[cmap]

    inner_radius = 4.0
    system_band_height = 0.25
    ring_gap = 0.035
    ring_height = 0.52
    outer_radius = inner_radius + len(clusters) * (ring_height + ring_gap)

    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, outer_radius + 1.65)
    ax.set_axis_off()

    # Inner coloured bands identify the 12 systems.
    for system in systems:
        start, end = sector_bounds[system]
        ax.bar(
            (start + end) / 2,
            system_band_height,
            width=end - start,
            bottom=inner_radius - system_band_height - 0.08,
            color=SYSTEM_COLORS.get(system, "#999999"),
            edgecolor="white",
            linewidth=0.8,
            align="center",
            zorder=2,
        )
        # Stagger labels for narrow sectors to clear neighbouring system names.
        system_label_radius = inner_radius - system_band_height - 1.0
        if end - start < math.radians(15):
            system_label_radius += 0.4
        ax.text(
            (start + end) / 2,
            system_label_radius,
            SYSTEM_LABELS.get(system, "\n".join(textwrap.wrap(system, 14))),
            ha="center", va="center", fontsize=5.5, color="#111111",
        )

    # Heatmap tracks: cluster 0 is innermost and the last cluster is outermost.
    significant_points = []
    for ring_index, cluster in enumerate(clusters):
        bottom = inner_radius + ring_index * (ring_height + ring_gap)
        for (system, feature), theta in positions.items():
            key = (system, feature, cluster)
            if key in lookup.index:
                row = lookup.loc[key]
                effect = float(row[effect_col])
                facecolor = colour_map(norm(np.clip(effect, -vlim, vlim)))
                significant = np.isfinite(row[fdr_col]) and float(row[fdr_col]) < fdr_threshold
            else:
                effect = np.nan
                facecolor = "#F1F1F1"
                significant = False
            ax.bar(
                theta, ring_height,
                width=cell_width * 0.96,
                bottom=bottom,
                color=facecolor,
                edgecolor="white",
                linewidth=0.32,
                align="center",
                zorder=3,
            )
            if significant:
                significant_points.append((theta, bottom + ring_height / 2))
    if significant_points:
        theta_sig, radius_sig = np.asarray(significant_points).T
        ax.scatter(theta_sig, radius_sig, s=5.5, c="#111111", edgecolors="none", zorder=5)

    # Right half: read outward. Left half: read inward, ending at the heatmap.
    if feature_fontsize is None:
        feature_fontsize = max(5.0, min(7.2, 620 / n_features * min(1.0, figsize[0] / 18)))
    label_radius = outer_radius + 0.12
    for (system, feature), theta in positions.items():
        angle = math.degrees(theta) % 360
        rotation = 90 - angle
        left_half = 180 < angle < 360
        if left_half:
            rotation += 180
        ax.text(
            theta, label_radius, feature,
            rotation=rotation,
            rotation_mode="anchor",
            ha="right" if left_half else "left", va="center",
            fontsize=feature_fontsize, color="#111111",
            gid="phenotype-label",
        )

    # Label each track immediately to the right of the vertical ending edge.
    cluster_texts = []
    for ring_index, cluster in enumerate(clusters):
        radius = inner_radius + ring_index * (ring_height + ring_gap) + ring_height / 2
        label = ax.annotate(
            cluster_labels[cluster], xy=(0, radius), xytext=(3, 0),
            textcoords="offset points", ha="left", va="center", fontsize=6,
            color="#111111", gid="cluster-label",
        )
        cluster_texts.append(label)
    fig.legend(
        handles=[Line2D([0], [0], marker="o", linestyle="none", color="#111111", markersize=4,
                        label=f"{fdr_col} < {fdr_threshold:g}")],
        loc="upper right", bbox_to_anchor=(0.965, 0.965), frameon=False, fontsize=6,
    )
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.97)

    # Fit the complete radial labels inside the fixed-size canvas.
    bounds = fig.bbox
    margin = 2.0 / 25.4 * fig.dpi
    center = ax.transData.transform((0, 0))
    available = (margin, margin, bounds.width - margin, bounds.height - margin)
    for _ in range(20):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [text.get_window_extent(renderer) for text in ax.texts]
        x0 = min(box.x0 for box in boxes)
        y0 = min(box.y0 for box in boxes)
        x1 = max(box.x1 for box in boxes)
        y1 = max(box.y1 for box in boxes)
        shrink = min(
            (center[0] - available[0]) / (center[0] - x0),
            (center[1] - available[1]) / (center[1] - y0),
            (available[2] - center[0]) / (x1 - center[0]),
            (available[3] - center[1]) / (y1 - center[1]),
        )
        if shrink >= 0.999:
            break
        ax.set_ylim(0, ax.get_ylim()[1] / (shrink * 0.98))

    # Position the colour bar beside the cluster IDs after the polar scale is final.
    fig.canvas.draw()
    polar_to_figure = ax.transData + fig.transFigure.inverted()
    gap_bottom = polar_to_figure.transform((0, inner_radius))
    gap_top = polar_to_figure.transform((0, outer_radius - ring_gap))
    point_width = 1 / (fig.get_figwidth() * 72)
    renderer = fig.canvas.get_renderer()
    label_right = max(label.get_window_extent(renderer).x1 for label in cluster_texts)
    colorbar_ax = fig.add_axes([
        label_right / fig.bbox.width + 6 * point_width,
        gap_bottom[1],
        4.5 * point_width,
        gap_top[1] - gap_bottom[1],
    ])
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=colour_map), cax=colorbar_ax)
    colorbar.set_label(
        "Partial correlation" if effect_col == "partial_r" else "Standardized β",
        fontsize=6, labelpad=2.5,
    )
    colorbar.ax.tick_params(labelsize=5, length=2, pad=1.5)
    colorbar.set_ticks([-vlim, 0, vlim])

    order_frame = pd.DataFrame(ordered_rows)
    if output_prefix is not None:
        prefix = Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        # Keep the physical canvas size: tight cropping would change its width.
        with mpl.rc_context({"savefig.bbox": None, "pdf.fonttype": 42}):
            fig.savefig(prefix.with_suffix(".png"), dpi=300, facecolor="white")
            fig.savefig(prefix.with_suffix(".pdf"), facecolor="white")
            fig.savefig(prefix.with_suffix(".svg"), facecolor="white")
        order_frame.to_csv(prefix.with_name(prefix.name + "_feature_order.csv"), index=False)
    return fig, order_frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=HERE / "outputs" / "cluster_system_association" / "cluster_system_associations.csv",
    )
    parser.add_argument(
        "--output-prefix", type=Path,
        default=HERE / "figures" / "cluster_system_circle_heatmap",
    )
    parser.add_argument("--effect-col", default=None)
    parser.add_argument("--fdr-col", default=None)
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--vlim", type=float, default=None)
    parser.add_argument("--width-mm", type=float, default=FIGURE_WIDTH_MM,
                        help="Width and height of the square output in mm (default: 210)")
    parser.add_argument(
        "--top-features-per-system", type=int, default=10,
        help="Maximum phenotypes per system (default: 10; use 0 to show all)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    associations = pd.read_csv(args.input)
    fig, order = plot_circle_heatmap(
        associations,
        effect_col=args.effect_col,
        fdr_col=args.fdr_col,
        fdr_threshold=args.fdr_threshold,
        top_features_per_system=args.top_features_per_system,
        vlim=args.vlim,
        figsize=(args.width_mm / 25.4, args.width_mm / 25.4),
        output_prefix=args.output_prefix,
    )
    plt.close(fig)
    print(
        f"Plotted {order['feature'].nunique():,} phenotypes from "
        f"{order['system'].nunique()} systems; saved to {args.output_prefix}.png/.pdf"
    )


if __name__ == "__main__":
    main()
