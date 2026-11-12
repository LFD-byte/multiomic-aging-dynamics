#!/usr/bin/env python3
"""Reproduce the smoothed pathway-age heatmap from a sample-level RPK CSV.

The input CSV must contain ``participant_id``, ``age`` and one or more pathway
RPK columns. ``age_bin_index`` and ``research_stage`` are accepted as optional
metadata columns and are not treated as pathways.

Run ``04-oral_microbiome_pathway.ipynb`` first with the new
``outputs/deswan-window_7.5`` inputs. This script consumes its sample-level
pathway RPK and overview exports, not the deSWAN taxon tables directly.
Defaults: oral microbiome, window 7.5 (filename tag ``7p5``), target age 48.
Default paths are relative to this script, independent of the working directory.

Example
-------
python 04-oral_microbiome_plot_pathway.py --deswan-window 7.5 --target-age 48

Outputs are editable SVG, PDF, a 300-dpi PNG preview, Source Data and QA JSON.
No TIFF is generated.
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as error:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "This script requires matplotlib. Install dependencies with: "
        "python -m pip install numpy pandas matplotlib"
    ) from error


HEATMAP_LOW = "#0072B2"       # colour-blind-safe blue
HEATMAP_NEUTRAL = "#F7F7F7"   # print-safe midpoint
HEATMAP_HIGH = "#D55E00"      # colour-blind-safe vermillion
HEATMAP_MISSING = "#D9D9D9"
TEXT = "#272727"

OPTIONAL_METADATA_COLUMNS = {
    "age_bin_index",
    "_age_bin",
    "research_stage",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "DejaVu Sans",
            "Liberation Sans",
            "sans-serif",
        ],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    result_dir = Path(
        os.environ.get("RESULT_DIR", str(Path(__file__).resolve().parent))
    ).expanduser()
    data_dir = Path(os.environ.get(
        "MICROBIOME_PATHWAY_OUTPUT_DIR",
        str(result_dir / "outputs" / "microbiome_pathway_enrichment"),
    )).expanduser()
    figure_dir = Path(os.environ.get(
        "MICROBIOME_PATHWAY_FIGURE_DIR",
        str(result_dir / "figures" / "microbiome_pathway_enrichment"),
    )).expanduser()
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce a Nature-style, Gaussian-smoothed pathway-age heatmap "
            "from sample-level pathway RPK values."
        )
    )
    parser.add_argument(
        "--deswan-window",
        type=float,
        default=os.environ.get("DESWAN_WINDOW", "7.5"),
        help="deSWAN half-window width in years; default: 7.5.",
    )
    parser.add_argument(
        "--microbiome-key",
        choices=["oral_microbiome", "gut_microbiome"],
        default=os.environ.get("MICROBIOME_KEY", "oral_microbiome"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Sample-level RPK CSV; defaults to the selected window/age notebook export.",
    )
    parser.add_argument(
        "--pathway-metadata",
        type=Path,
        help=(
            "Optional CSV containing pathway_id and pathway_name or "
            "display_label. Defaults to the matching overview beside --input, "
            "when available. Its row order controls the heatmap row order."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=figure_dir,
        help="Output directory; default: figures/microbiome_pathway_enrichment beside this script.",
    )
    parser.add_argument(
        "--output-stem",
        default=None,
        help="Output filename stem without extension.",
    )
    parser.add_argument(
        "--microbiome-label",
        default=None,
        help="Dataset label shown in the figure title.",
    )
    parser.add_argument(
        "--target-age", type=float, default=os.environ.get("TARGET_AGE", "48")
    )
    parser.add_argument("--age-min", type=float, default=40.0)
    parser.add_argument("--age-max", type=float, default=70.0)
    parser.add_argument(
        "--grid-width",
        type=float,
        default=2.0,
        help="Spacing of displayed age-grid points in years.",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=2.0,
        help="Gaussian-kernel bandwidth in years.",
    )
    parser.add_argument(
        "--z-limit",
        type=float,
        default=2.0,
        help="Symmetric row-z-score colour limit.",
    )
    args = parser.parse_args(argv)
    validate_positive_finite("deswan_window", args.deswan_window)
    validate_positive_finite("target_age", args.target_age)
    window_tag = f"{args.deswan_window:g}".replace(".", "p")
    age_tag = f"{args.target_age:g}".replace(".", "p")
    output_prefix = f"age{age_tag}_{args.microbiome_key}_window_{window_tag}"
    if args.input is None:
        args.input = (
            data_dir / f"{output_prefix}_eligible_pathway_age_heatmap_sample_level_rpk.csv"
        )
    if args.pathway_metadata is None:
        sample_suffix = "_eligible_pathway_age_heatmap_sample_level_rpk"
        if args.input.stem.endswith(sample_suffix):
            prefix = args.input.stem[:-len(sample_suffix)]
            companion = args.input.with_name(
                f"{prefix}_eligible_pathway_overview_plot_data.csv"
            )
            if companion.is_file():
                args.pathway_metadata = companion
    if args.microbiome_label is None:
        args.microbiome_label = {
            "oral_microbiome": "Oral microbiome",
            "gut_microbiome": "Gut microbiome",
        }[args.microbiome_key]
    return args


def validate_positive_finite(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive; got {value!r}")


def make_age_grid(
    age_min: float,
    age_max: float,
    grid_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(age_min) or not np.isfinite(age_max):
        raise ValueError("age_min and age_max must be finite")
    if age_max <= age_min:
        raise ValueError("age_max must be greater than age_min")
    validate_positive_finite("grid_width", grid_width)

    interval_count = (age_max - age_min) / grid_width
    rounded_interval_count = int(round(interval_count))
    if not np.isclose(interval_count, rounded_interval_count):
        raise ValueError(
            "The age range must be exactly divisible by grid_width so that "
            "all raw bins have the same width."
        )
    age_bin_edges = np.linspace(
        age_min,
        age_max,
        rounded_interval_count + 1,
    )
    age_grid = (age_bin_edges[:-1] + age_bin_edges[1:]) / 2
    if not np.all(np.diff(age_grid) > 0):
        raise RuntimeError("The derived age grid is not strictly increasing")
    return age_bin_edges, age_grid


def gaussian_kernel_smooth_by_age(
    ages: np.ndarray,
    values: np.ndarray,
    age_grid: np.ndarray,
    bandwidth_years: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pathway-by-grid kernel means and Kish effective sample sizes."""
    ages = np.asarray(ages, dtype=float)
    values = np.asarray(values, dtype=float)
    age_grid = np.asarray(age_grid, dtype=float)
    validate_positive_finite("bandwidth", bandwidth_years)

    if ages.ndim != 1:
        raise ValueError("ages must be one-dimensional")
    if values.ndim != 2 or values.shape[0] != ages.size:
        raise ValueError(
            "values must be a samples-by-pathways matrix aligned to ages"
        )
    if age_grid.ndim != 1 or age_grid.size == 0:
        raise ValueError("age_grid must be a non-empty vector")
    if not np.all(np.diff(age_grid) > 0):
        raise ValueError("age_grid must be strictly increasing")
    if not np.isfinite(ages).all() or not np.isfinite(values).all():
        raise ValueError("ages and pathway values must all be finite")

    smoothed_columns: list[np.ndarray] = []
    effective_sample_sizes: list[float] = []
    for target_age in age_grid:
        scaled_distance = (ages - target_age) / bandwidth_years
        weights = np.exp(-0.5 * np.square(scaled_distance))
        weight_sum = float(weights.sum())
        squared_weight_sum = float(np.square(weights).sum())
        if weight_sum <= 0 or squared_weight_sum <= 0:
            raise RuntimeError(
                f"Gaussian kernel has zero support at age {target_age:g}"
            )
        smoothed_columns.append(
            (weights[:, None] * values).sum(axis=0) / weight_sum
        )
        effective_sample_sizes.append(
            np.square(weight_sum) / squared_weight_sum
        )

    return (
        np.column_stack(smoothed_columns),
        np.asarray(effective_sample_sizes, dtype=float),
    )


def default_output_stem(input_file: Path) -> str:
    stem = input_file.stem
    suffix = "_sample_level_rpk"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return f"{stem}_local_reproduction"


def load_sample_data(
    input_file: Path,
    age_min: float,
    age_max: float,
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_file}\n"
            "Run 04-oral_microbiome_pathway.ipynb with the same DESWAN_WINDOW "
            "and TARGET_AGE to export sample-level RPK first, or pass --input."
        )

    data = pd.read_csv(input_file)
    required_columns = {"participant_id", "age"}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise KeyError(
            "Input CSV lacks required columns: " + ", ".join(missing_columns)
        )

    pathway_columns = [
        column
        for column in data.columns
        if column
        not in required_columns | OPTIONAL_METADATA_COLUMNS
    ]
    if not pathway_columns:
        raise RuntimeError("No pathway RPK columns were detected")

    data["age"] = pd.to_numeric(data["age"], errors="coerce")
    numeric_pathways = data[pathway_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    invalid_pathway_cell_count = int(numeric_pathways.isna().sum().sum())
    if invalid_pathway_cell_count:
        raise RuntimeError(
            f"Pathway matrix contains {invalid_pathway_cell_count:,} missing "
            "or non-numeric cells; no silent zero-filling was performed."
        )
    if (numeric_pathways < 0).any().any():
        negative_count = int((numeric_pathways < 0).sum().sum())
        raise RuntimeError(
            f"Pathway matrix contains {negative_count:,} negative RPK values."
        )
    data[pathway_columns] = numeric_pathways

    input_row_count = len(data)
    valid_age_mask = data["age"].notna() & data["age"].between(
        age_min,
        age_max,
        inclusive="both",
    )
    invalid_age_row_count = int((~valid_age_mask).sum())
    analysis_data = data.loc[valid_age_mask].copy()
    if analysis_data.empty:
        raise RuntimeError(
            f"No samples have valid ages within [{age_min:g}, {age_max:g}]"
        )

    audit_counts = {
        "input_rows": int(input_row_count),
        "rows_excluded_for_missing_or_out_of_range_age": invalid_age_row_count,
        "analysis_rows": int(len(analysis_data)),
        "unique_participants": int(
            analysis_data["participant_id"].astype("string").nunique()
        ),
        "pathway_columns": int(len(pathway_columns)),
    }
    return analysis_data, pathway_columns, audit_counts


def load_pathway_labels(
    pathway_columns: list[str],
    metadata_file: Path | None,
) -> tuple[list[str], dict[str, str], list[str]]:
    if metadata_file is None:
        labels = {
            pathway_id: textwrap.fill(pathway_id, width=47)
            for pathway_id in pathway_columns
        }
        return pathway_columns, labels, pathway_columns.copy()
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Pathway metadata CSV not found: {metadata_file}"
        )

    metadata = pd.read_csv(metadata_file)
    if "pathway_id" not in metadata.columns:
        raise KeyError("Pathway metadata CSV lacks pathway_id")
    metadata["pathway_id"] = metadata["pathway_id"].astype(str)
    if metadata["pathway_id"].duplicated().any():
        raise RuntimeError("Pathway metadata contains duplicate pathway_id values")

    label_map: dict[str, str] = {}
    for row in metadata.itertuples(index=False):
        pathway_id = str(row.pathway_id)
        display_label = str(getattr(row, "display_label", "")).strip()
        if display_label.casefold() == "nan":
            display_label = ""
        pathway_name = str(getattr(row, "pathway_name", "")).strip()
        if pathway_name.casefold() == "nan":
            pathway_name = ""
        combined = display_label or (
            f"{pathway_id}  {pathway_name}" if pathway_name else pathway_id
        )
        label_map[pathway_id] = textwrap.fill(combined, width=47)

    ordered_pathways = [
        pathway_id
        for pathway_id in metadata["pathway_id"].tolist()
        if pathway_id in pathway_columns
    ]
    unlabelled_pathways = [
        pathway_id
        for pathway_id in pathway_columns
        if pathway_id not in label_map
    ]
    ordered_pathways.extend(unlabelled_pathways)
    for pathway_id in unlabelled_pathways:
        label_map[pathway_id] = textwrap.fill(pathway_id, width=47)

    if set(ordered_pathways) != set(pathway_columns):
        raise RuntimeError("Pathway ordering did not preserve all input columns")
    return ordered_pathways, label_map, unlabelled_pathways


def find_text_collisions(
    text_objects: list[mpl.text.Text],
    renderer: mpl.backend_bases.RendererBase,
) -> list[tuple[str, str]]:
    boxes = [
        text_object.get_window_extent(renderer).expanded(1.01, 1.03)
        for text_object in text_objects
    ]
    collisions: list[tuple[str, str]] = []
    for left_index in range(len(boxes)):
        for right_index in range(left_index + 1, len(boxes)):
            if boxes[left_index].overlaps(boxes[right_index]):
                collisions.append(
                    (
                        text_objects[left_index].get_text(),
                        text_objects[right_index].get_text(),
                    )
                )
    return collisions


def build_source_data(
    pathway_order: list[str],
    age_bin_edges: np.ndarray,
    age_grid: np.ndarray,
    sample_n: pd.Series,
    participant_n: pd.Series,
    effective_n: np.ndarray,
    raw_mean_rpk: pd.DataFrame,
    raw_mean_log1p_rpk: pd.DataFrame,
    smoothed_mean_log1p_rpk: pd.DataFrame,
    row_z_score: pd.DataFrame,
    bandwidth: float,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for pathway_id in pathway_order:
        for grid_index, grid_age in enumerate(age_grid):
            records.append(
                {
                    "pathway_id": pathway_id,
                    "age_bin_start": age_bin_edges[grid_index],
                    "age_bin_end": age_bin_edges[grid_index + 1],
                    "age_grid_years": grid_age,
                    "n_samples_in_raw_bin": int(sample_n.loc[grid_index]),
                    "n_participants_in_raw_bin": int(
                        participant_n.loc[grid_index]
                    ),
                    "smoothing_method": (
                        "sample-level Gaussian-kernel weighted mean"
                    ),
                    "smoothing_bandwidth_years": bandwidth,
                    "kernel_effective_sample_n": effective_n[grid_index],
                    "raw_bin_mean_summed_rpk": raw_mean_rpk.loc[
                        pathway_id,
                        grid_index,
                    ],
                    "raw_bin_mean_log1p_summed_rpk": raw_mean_log1p_rpk.loc[
                        pathway_id,
                        grid_index,
                    ],
                    "smoothed_mean_log1p_summed_rpk": (
                        smoothed_mean_log1p_rpk.loc[
                            pathway_id,
                            grid_index,
                        ]
                    ),
                    "smoothed_row_z_score": row_z_score.loc[
                        pathway_id,
                        grid_index,
                    ],
                }
            )
    return pd.DataFrame(records)


def main() -> int:
    args = parse_args()
    validate_positive_finite("bandwidth", args.bandwidth)
    validate_positive_finite("z_limit", args.z_limit)
    age_bin_edges, age_grid = make_age_grid(
        args.age_min,
        args.age_max,
        args.grid_width,
    )
    if not np.isfinite(args.target_age) or not (
        args.age_min <= args.target_age <= args.age_max
    ):
        raise ValueError(
            f"target_age must be within [{args.age_min:g}, {args.age_max:g}]"
        )

    analysis_data, detected_pathways, audit_counts = load_sample_data(
        args.input,
        args.age_min,
        args.age_max,
    )
    pathway_order, label_map, unlabelled_pathways = load_pathway_labels(
        detected_pathways,
        args.pathway_metadata,
    )

    age_bin_index = pd.cut(
        analysis_data["age"],
        bins=age_bin_edges,
        labels=False,
        include_lowest=True,
        right=True,
    )
    if age_bin_index.isna().any():
        raise RuntimeError(
            "At least one retained age was not assigned to a raw age bin"
        )
    age_bin_index = age_bin_index.astype(int)
    if age_bin_index.nunique() < 3:
        raise RuntimeError("Fewer than three non-empty raw age bins are present")

    saved_bin_mismatch_count = 0
    if "age_bin_index" in analysis_data.columns:
        saved_bin_index = pd.to_numeric(
            analysis_data["age_bin_index"],
            errors="coerce",
        )
        comparable = saved_bin_index.notna()
        saved_bin_mismatch_count = int(
            (
                saved_bin_index.loc[comparable].astype(int)
                != age_bin_index.loc[comparable]
            ).sum()
        )

    analysis_data = analysis_data.copy()
    analysis_data["_local_age_bin"] = age_bin_index.to_numpy()
    bin_index = pd.Index(
        np.arange(len(age_grid)),
        name="_local_age_bin",
    )
    sample_n = (
        analysis_data.groupby("_local_age_bin")
        .size()
        .reindex(bin_index, fill_value=0)
        .astype(int)
    )
    participant_n = (
        analysis_data.groupby("_local_age_bin")["participant_id"]
        .nunique()
        .reindex(bin_index, fill_value=0)
        .astype(int)
    )

    raw_mean_rpk = (
        analysis_data.groupby("_local_age_bin")[pathway_order]
        .mean()
        .reindex(bin_index)
        .T
    )
    sample_log1p_rpk = np.log1p(analysis_data[pathway_order])
    raw_log1p_with_bin = sample_log1p_rpk.copy()
    raw_log1p_with_bin["_local_age_bin"] = age_bin_index.to_numpy()
    raw_mean_log1p_rpk = (
        raw_log1p_with_bin.groupby("_local_age_bin")[pathway_order]
        .mean()
        .reindex(bin_index)
        .T
    )

    smoothed_values, effective_n = gaussian_kernel_smooth_by_age(
        ages=analysis_data["age"].to_numpy(dtype=float),
        values=sample_log1p_rpk.to_numpy(dtype=float),
        age_grid=age_grid,
        bandwidth_years=args.bandwidth,
    )
    smoothed_mean_log1p_rpk = pd.DataFrame(
        smoothed_values,
        index=pathway_order,
        columns=bin_index,
    )
    row_mean = smoothed_mean_log1p_rpk.mean(axis=1)
    row_sd = smoothed_mean_log1p_rpk.std(axis=1, ddof=0).replace(0, np.nan)
    row_z_score = smoothed_mean_log1p_rpk.sub(
        row_mean,
        axis=0,
    ).div(
        row_sd,
        axis=0,
    )
    constant_rows = row_z_score.isna().all(axis=1)
    row_z_score.loc[constant_rows] = 0.0

    output_dir = args.output_dir or args.input.parent / "local_reproduction"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_stem or default_output_stem(args.input)
    output_base = output_dir / output_stem

    source_data = build_source_data(
        pathway_order=pathway_order,
        age_bin_edges=age_bin_edges,
        age_grid=age_grid,
        sample_n=sample_n,
        participant_n=participant_n,
        effective_n=effective_n,
        raw_mean_rpk=raw_mean_rpk,
        raw_mean_log1p_rpk=raw_mean_log1p_rpk,
        smoothed_mean_log1p_rpk=smoothed_mean_log1p_rpk,
        row_z_score=row_z_score,
        bandwidth=args.bandwidth,
    )
    source_data_file = output_base.with_name(
        f"{output_stem}_source_data.csv"
    )
    source_data.to_csv(source_data_file, index=False)

    figure_width_in = 183 / 25.4
    figure_height_in = min(
        170 / 25.4,
        max(4.0, 1.65 + 0.28 * len(pathway_order)),
    )
    fig, ax = plt.subplots(
        figsize=(figure_width_in, figure_height_in),
        facecolor="white",
    )
    heatmap_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "nature_blue_neutral_vermillion",
        [HEATMAP_LOW, HEATMAP_NEUTRAL, HEATMAP_HIGH],
        N=256,
    )
    heatmap_cmap.set_bad(HEATMAP_MISSING)
    heatmap_values = np.ma.masked_invalid(
        row_z_score.loc[pathway_order].to_numpy(dtype=float)
    )
    image = ax.imshow(
        heatmap_values,
        aspect="auto",
        interpolation="nearest",
        cmap=heatmap_cmap,
        norm=mpl.colors.TwoSlopeNorm(
            vmin=-args.z_limit,
            vcenter=0,
            vmax=args.z_limit,
        ),
    )

    ax.set_yticks(np.arange(len(pathway_order)))
    ax.set_yticklabels(
        [label_map[pathway_id] for pathway_id in pathway_order],
        fontsize=5.8,
        color=TEXT,
    )
    tick_positions = np.unique(
        np.linspace(
            0,
            len(age_grid) - 1,
            min(8, len(age_grid)),
            dtype=int,
        )
    )
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [f"{age_grid[position]:g}" for position in tick_positions],
        fontsize=5.8,
        color=TEXT,
    )
    ax.set_xlabel(
        f"Age (years; {args.grid_width:g}-year display grid)",
        fontsize=6.8,
        color=TEXT,
        labelpad=5,
    )
    ax.tick_params(axis="y", length=0, pad=5)
    ax.tick_params(axis="x", length=3, width=0.7, colors=TEXT)
    ax.set_xticks(np.arange(-0.5, len(age_grid), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pathway_order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.45)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Preserve the original grid and smoothing. Mark 48 y at its actual age,
    # between the 47/49-y grid points, rather than relabelling either midpoint.
    target_grid_position = (args.target_age - args.age_min) / args.grid_width - 0.5
    if np.isclose(target_grid_position, round(target_grid_position)):
        target_outline = mpl.patches.Rectangle(
            (target_grid_position - 0.5, -0.5),
            1,
            len(pathway_order),
            fill=False,
            edgecolor=TEXT,
            linewidth=1.0,
            zorder=5,
        )
        ax.add_patch(target_outline)
    else:
        ax.vlines(
            target_grid_position, -0.5, len(pathway_order) - 0.5,
            color=TEXT, linewidth=1.0, zorder=5,
        )
    ax.text(
        target_grid_position,
        1.01,
        f"{args.target_age:g} y",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=5.5,
        fontweight="bold",
        color=TEXT,
        clip_on=False,
    )
    ax.set_ylim(len(pathway_order) - 0.5, -1.15)

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.032,
        pad=0.025,
        ticks=[-args.z_limit, 0, args.z_limit],
    )
    colorbar.set_label(
        "Smoothed mean log1p RPK\n(row z-score)",
        fontsize=5.8,
        color=TEXT,
        labelpad=4,
    )
    colorbar.ax.tick_params(
        labelsize=5.2,
        length=2.5,
        width=0.6,
        colors=TEXT,
    )
    colorbar.outline.set_linewidth(0.6)

    title = fig.text(
        0.02,
        0.982,
        f"{args.microbiome_label}: smoothed age-resolved pathway profiles",
        ha="left",
        va="top",
        fontsize=7.0,
        fontweight="bold",
        color=TEXT,
    )
    subtitle = fig.text(
        0.02,
        0.938,
        (
            "Sample-level log1p RPK smoothed with a Gaussian kernel; "
            "rows standardized across the age grid"
        ),
        ha="left",
        va="top",
        fontsize=6.0,
        color="#5A5A5A",
    )
    fig.text(
        0.02,
        0.018,
        (
            f"Gaussian bandwidth = {args.bandwidth:g} y; colour limits = "
            f"±{args.z_limit:g}.\nDescriptive display; no age-association "
            "test was performed."
        ),
        ha="left",
        va="bottom",
        fontsize=5.4,
        color="#666666",
    )
    fig.subplots_adjust(left=0.42, right=0.88, top=0.84, bottom=0.15)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    visible_text = [
        text_object
        for text_object in fig.findobj(mpl.text.Text)
        if text_object.get_visible() and text_object.get_text().strip()
    ]
    minimum_font_pt = min(
        text_object.get_fontsize() for text_object in visible_text
    )
    if minimum_font_pt < 5:
        raise RuntimeError(
            f"Figure contains text below 5 pt: {minimum_font_pt:.2f} pt"
        )
    collision_groups = {
        "pathway_labels": find_text_collisions(
            [label for label in ax.get_yticklabels() if label.get_visible()],
            renderer,
        ),
        "age_ticks": find_text_collisions(
            [label for label in ax.get_xticklabels() if label.get_visible()],
            renderer,
        ),
        "title_and_subtitle": find_text_collisions(
            [title, subtitle],
            renderer,
        ),
    }
    collision_count = sum(
        len(collisions) for collisions in collision_groups.values()
    )
    if collision_count:
        raise RuntimeError(
            "Text collision audit failed: "
            + json.dumps(collision_groups, ensure_ascii=False)
        )

    figure_paths = {
        "svg": output_base.with_name(f"{output_stem}.svg"),
        "pdf": output_base.with_name(f"{output_stem}.pdf"),
        "png": output_base.with_name(f"{output_stem}.png"),
    }
    fig.savefig(
        figure_paths["svg"],
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
    )
    fig.savefig(
        figure_paths["pdf"],
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
    )
    fig.savefig(
        figure_paths["png"],
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
    )
    plt.close(fig)

    svg_text = figure_paths["svg"].read_text(encoding="utf-8")
    if "<text" not in svg_text:
        raise RuntimeError("SVG text was converted to paths")

    qa = {
        "figure_contract": {
            "core_conclusion": (
                "Threshold-qualified pathway contributions show heterogeneous "
                "smoothed average profiles across age."
            ),
            "evidence_role": "descriptive secondary main-figure subpanel",
            "archetype": "single-panel quantitative heatmap",
        },
        "input_file": str(args.input.resolve()),
        "deswan_window": args.deswan_window,
        "target_age": args.target_age,
        "target_age_grid_position": target_grid_position,
        "pathway_metadata_file": (
            str(args.pathway_metadata.resolve())
            if args.pathway_metadata is not None
            else None
        ),
        "data_counts": audit_counts,
        "saved_age_bin_mismatch_count": saved_bin_mismatch_count,
        "unlabelled_pathways": unlabelled_pathways,
        "age_range": [args.age_min, args.age_max],
        "age_grid_width_years": args.grid_width,
        "smoothing_method": "Nadaraya-Watson Gaussian-kernel weighted mean",
        "smoothing_bandwidth_years": args.bandwidth,
        "kernel_effective_sample_n_range": [
            float(effective_n.min()),
            float(effective_n.max()),
        ],
        "normalization": "row z-score calculated after smoothing",
        "colour_scale": {
            "type": "symmetric diverging",
            "limits": [-args.z_limit, args.z_limit],
            "low": HEATMAP_LOW,
            "midpoint": HEATMAP_NEUTRAL,
            "high": HEATMAP_HIGH,
            "missing": HEATMAP_MISSING,
        },
        "minimum_source_font_pt": minimum_font_pt,
        "collision_audit": {
            "status": "passed",
            "n_detected": collision_count,
            "groups": collision_groups,
        },
        "editable_svg_text": True,
        "vector_exports": ["svg", "pdf"],
        "raster_exports": ["png_300_dpi"],
        "tiff_export": "disabled by design",
        "statistics_in_figure": "none; descriptive smoothing only",
        "review_risks": [
            "trajectory shape depends on the prespecified bandwidth",
            "row z-scores suppress between-pathway absolute differences",
            "edge estimates can have lower effective support",
        ],
        "source_data_file": str(source_data_file.resolve()),
    }
    qa_file = output_base.with_name(f"{output_stem}_figure_qa.json")
    qa_file.write_text(
        json.dumps(qa, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Local heatmap reproduction completed")
    print(f"  Input rows: {audit_counts['input_rows']:,}")
    print(f"  Analysis rows: {audit_counts['analysis_rows']:,}")
    print(f"  Pathways: {len(pathway_order):,}")
    print(
        "  Kernel effective sample size: "
        f"{effective_n.min():.1f}–{effective_n.max():.1f}"
    )
    for format_name, path in figure_paths.items():
        print(f"  {format_name.upper()}: {path}")
    print(f"  Source data: {source_data_file}")
    print(f"  Figure QA: {qa_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
