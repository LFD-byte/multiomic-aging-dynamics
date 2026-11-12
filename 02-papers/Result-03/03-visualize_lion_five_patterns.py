#!/usr/bin/env python
"""Export two aligned, title-free LION figures, each exactly 100 mm wide.

Figure a is a ranked-LION dot plot. Colour encodes enrichment
evidence (-log10 BH FDR q) and dot area encodes the percentage of submitted
lipids annotated to a term. Terms are selected automatically: by default,
take up to five terms with q < 0.05 and the lowest q values in each pattern,
then display their union across all five patterns. Original q values are
retained; display selection does not rerun multiple-testing correction.
An effect-size ranking requires an explicitly supplied numeric effect column.
The standard LION/web CSV files do not contain a
directional enrichment statistic, so this figure deliberately does not label
terms as activated, increased or decreased.

Figure b uses exactly the same ontology terms and row order. Cell text is
the percentage of submitted, hard-assigned cluster lipids annotated to each
term; colour is the within-row z score of those percentages. Official LION
annotations are reconstructed locally and checked against every exported
term count. Unmatched lipids remain in the denominator, so this measures
annotation coverage, not complete biological class membership. No inferential
test is applied to this descriptive panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

from lion_ontology_composition import reconstruct_annotations, summarize_ontology_coverage


# Editable vector text and a conservative Nature-family typography floor.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "Helvetica",
    "DejaVu Sans",
    "Liberation Sans",
]
plt.rcParams.update({
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})
plt.rcParams["font.size"] = 7
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["axes.labelsize"] = 7
plt.rcParams["xtick.labelsize"] = 6.5
plt.rcParams["ytick.labelsize"] = 6.5
plt.rcParams["axes.linewidth"] = 0.7
plt.rcParams["legend.frameon"] = False
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"


HERE = Path(__file__).resolve().parent
DEFAULT_LION_ROOT = HERE / "outputs" / "lion_web_five_patterns"
DEFAULT_FIGURE_DIR = HERE / "figures" / "lion_web_five_patterns"
DEFAULT_SOURCE_DIR = DEFAULT_LION_ROOT / "visualization"
DEFAULT_MAPPING = (
    DEFAULT_LION_ROOT / "audit" / "lipid_mapping_and_pattern_scores.csv"
)
DEFAULT_ANNOTATION_REFERENCE = DEFAULT_LION_ROOT / "annotation_reference"

PATTERN_IDS = (1, 2, 3, 4, 5)
FDR_THRESHOLD = 0.05
FIGURE_WIDTH_MM = 100.0
# Match the default export height in 02-transcriptome_cluster_pathway_landscape.R.
FIGURE_A_HEIGHT_MM = 100.0
MM_PER_INCH = 25.4
# Reuse the overview figure's lipidomics accent and omics heatmap endpoints.
EVIDENCE_COLOURS = ("#FCF3F2", "#EED4D3", "#DFA5A8", "#C96F75")
COVERAGE_COLOURS = ("#3F6596", "#F7F3DF", "#C75C4D")
TEXT_COLOUR = "#2E3543"
SECONDARY_TEXT_COLOUR = "#566171"
SIGNIFICANCE_EDGE = "#394A63"


@dataclass(frozen=True)
class TermSelection:
    mode: str = "top-n"
    q_threshold: float = 0.05
    top_n: int = 5
    effect_column: str | None = None
    effect_order: str = "larger"

    def __post_init__(self) -> None:
        if self.mode not in {"top-n", "q-value", "effect-size"}:
            raise ValueError(f"Unknown selection mode: {self.mode}")
        if not np.isfinite(self.q_threshold) or not 0 < self.q_threshold <= 1:
            raise ValueError("Selection q threshold must be finite and in (0, 1].")
        if isinstance(self.top_n, bool) or not isinstance(self.top_n, int) or self.top_n < 1:
            raise ValueError("Top N must be a positive integer.")
        if self.effect_order not in {"larger", "smaller", "absolute"}:
            raise ValueError(f"Unknown effect ordering: {self.effect_order}")
        if self.mode == "effect-size":
            if not self.effect_column or not self.effect_column.strip():
                raise ValueError("Effect-size selection requires --effect-column.")
            reserved = {
                "pvalue", "fdrqvalue", "qvalue", "neglog10q", "annotated",
                "annotatedpercentage", "submittedlipids", "inputlipids",
            }
            normalized = "".join(c for c in self.effect_column.lower() if c.isalnum())
            if normalized in reserved:
                raise ValueError("P/q values and annotation counts are not effect sizes.")
        elif self.effect_column is not None:
            raise ValueError("--effect-column is only used with --selection effect-size.")


def selection_description(selection: TermSelection) -> str:
    threshold = f"q < {selection.q_threshold:g}"
    if selection.mode == "q-value":
        return f"all terms satisfying {threshold} in at least one pattern"
    ranking = "the lowest q values (ties resolved by p value, then term ID)"
    if selection.mode == "effect-size":
        ordering = {"larger": "largest", "smaller": "smallest", "absolute": "largest absolute"}
        ranking = (
            f"the {ordering[selection.effect_order]} values of {selection.effect_column} "
            "(ties resolved by q value, p value, then term ID)"
        )
    return (
        f"the union of up to {selection.top_n} terms per pattern satisfying {threshold}, "
        f"ranked by {ranking}; patterns with no eligible terms contribute no terms"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(
    frame: pd.DataFrame, required: Iterable[str], label: str
) -> None:
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def as_boolean(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0", "yes", "no"}
    unexpected = sorted(set(values).difference(allowed))
    if unexpected:
        raise ValueError(f"Unexpected Boolean values: {unexpected}")
    return values.isin({"true", "1", "yes"})


def clean_description(value: str) -> str:
    text = str(value).strip()
    # Bracketed ontology-class codes are useful in the audit CSV but visually
    # redundant in display labels; full descriptions remain in the audit CSV.
    return text.rsplit(" [", maxsplit=1)[0]


def load_ranked_lion_results(
    lion_root: Path,
    effect_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    manifest_path = lion_root / "lion_web_run_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"LION run manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    require_columns(
        manifest,
        {
            "pattern_id",
            "pattern",
            "mode",
            "input_lipids",
            "expected_result_csv",
        },
        "LION run manifest",
    )
    ranked_manifest = manifest.loc[manifest["mode"].eq("ranked")].copy()
    ranked_manifest["pattern_id"] = pd.to_numeric(
        ranked_manifest["pattern_id"], errors="raise"
    ).astype(int)
    if tuple(sorted(ranked_manifest["pattern_id"].tolist())) != PATTERN_IDS:
        raise ValueError("The manifest must contain one ranked run for patterns 1-5.")
    if ranked_manifest["pattern_id"].duplicated().any():
        raise ValueError("The ranked-run manifest contains duplicate pattern IDs.")

    frames: list[pd.DataFrame] = []
    result_paths: list[Path] = []
    for run in ranked_manifest.sort_values("pattern_id").itertuples(index=False):
        result_path = lion_root / "results" / str(run.expected_result_csv)
        if not result_path.exists():
            raise FileNotFoundError(f"Ranked LION result not found: {result_path}")
        result = pd.read_csv(result_path)
        description_column = next(
            (
                name
                for name in ("Discription", "Description")
                if name in result.columns
            ),
            None,
        )
        if description_column is None:
            raise ValueError(
                f"No description column was found in LION result: {result_path}"
            )
        require_columns(
            result,
            {"Term ID", "Annotated", "p-value", "FDR q-value"},
            result_path.name,
        )
        if effect_column is not None:
            if effect_column not in result.columns:
                raise ValueError(
                    f"Effect column {effect_column!r} is absent from {result_path.name}. "
                    "Standard LION exports contain no effect size; use q-value or top-n "
                    "selection, or supply a real effect statistic in every result file."
                )
            result["effect_value"] = pd.to_numeric(result[effect_column], errors="raise")
            if not np.isfinite(result["effect_value"]).all():
                raise ValueError(f"Non-finite effect values in {result_path.name}.")
        result = result.rename(
            columns={
                "Term ID": "term_id",
                description_column: "description",
                "Annotated": "annotated",
                "p-value": "p_value",
                "FDR q-value": "q_value",
            }
        )
        for column in ("annotated", "p_value", "q_value"):
            result[column] = pd.to_numeric(result[column], errors="raise")
        if result["term_id"].duplicated().any():
            raise ValueError(f"Duplicate LION term IDs in {result_path.name}.")
        if not result["p_value"].between(0, 1).all():
            raise ValueError(f"p-value is outside [0, 1] in {result_path.name}.")
        if not result["q_value"].between(0, 1).all():
            raise ValueError(f"FDR q-value is outside [0, 1] in {result_path.name}.")
        if (result["annotated"] < 0).any():
            raise ValueError(f"Annotated count is negative in {result_path.name}.")
        input_lipids = int(run.input_lipids)
        if input_lipids <= 0:
            raise ValueError(f"Invalid input_lipids for pattern {run.pattern_id}.")
        result["pattern_id"] = int(run.pattern_id)
        result["pattern"] = str(run.pattern)
        result["cluster_label"] = f"Cluster {int(run.pattern_id)}"
        result["submitted_lipids"] = input_lipids
        result["annotated_percentage"] = (
            100.0 * result["annotated"] / input_lipids
        )
        result["neg_log10_q"] = -np.log10(
            result["q_value"].clip(lower=np.finfo(float).tiny)
        )
        result["significant"] = result["q_value"] < FDR_THRESHOLD
        frames.append(result)
        result_paths.append(result_path)

    return pd.concat(frames, ignore_index=True), ranked_manifest, [manifest_path, *result_paths]


def select_lion_terms(
    all_results: pd.DataFrame, selection: TermSelection
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the display grid, selected-term summary and complete selection audit.

    Each pattern nominates terms independently; all observed results for the
    union are then shown. Minima across patterns are display-order statistics,
    not newly calibrated or pooled p/q values.
    """
    require_columns(
        all_results, {"pattern_id", "term_id", "description", "p_value", "q_value"},
        "LION selection input",
    )
    audit = all_results.copy().reset_index(drop=True)
    if audit.duplicated(["pattern_id", "term_id"]).any():
        raise ValueError("Duplicate pattern/term results cannot be selected.")
    if tuple(sorted(audit["pattern_id"].unique())) != PATTERN_IDS:
        raise ValueError("Term selection requires results for all five patterns.")
    if not audit[["p_value", "q_value"]].apply(lambda values: values.between(0, 1)).all().all():
        raise ValueError("Selection requires finite p/q values in [0, 1].")
    descriptions = audit["description"].fillna("").astype(str).str.strip()
    if descriptions.eq("").any() or audit["term_id"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("LION term IDs and descriptions must not be blank.")
    audit["description"] = descriptions
    if audit.groupby("term_id")["description"].nunique().gt(1).any():
        raise ValueError("LION descriptions for the same term differ between patterns.")
    audit["eligible_by_q"] = audit["q_value"] < selection.q_threshold
    audit["significant"] = audit["q_value"] < FDR_THRESHOLD
    sort_columns = ["pattern_id", "q_value", "p_value", "term_id"]
    ascending = [True, True, True, True]
    if selection.mode == "effect-size":
        require_columns(audit, {"effect_value"}, "Effect-size selection input")
        if not np.isfinite(audit["effect_value"]).all():
            raise ValueError("Effect-size selection requires finite effect values.")
        effect = audit["effect_value"]
        audit["effect_ranking_score"] = (
            effect.abs() if selection.effect_order == "absolute"
            else -effect if selection.effect_order == "smaller" else effect
        )
        sort_columns.insert(1, "effect_ranking_score")
        ascending.insert(1, False)
    eligible = audit.loc[audit["eligible_by_q"]].sort_values(sort_columns, ascending=ascending)
    eligible["rank_within_pattern"] = eligible.groupby("pattern_id").cumcount() + 1
    audit["rank_within_pattern"] = pd.Series(pd.NA, index=audit.index, dtype="Int64")
    audit.loc[eligible.index, "rank_within_pattern"] = eligible["rank_within_pattern"]
    nominated = eligible if selection.mode == "q-value" else eligible.loc[
        eligible["rank_within_pattern"] <= selection.top_n
    ]
    if nominated.empty:
        raise ValueError(
            f"No LION terms satisfy q < {selection.q_threshold:g}; "
            "no nonsignificant fallback terms were selected."
        )
    audit["selected_for_pattern"] = audit.index.isin(nominated.index)
    audit["selected_for_display"] = audit["term_id"].isin(nominated["term_id"])
    audit["selection_reason"] = np.select(
        [~audit["eligible_by_q"], audit["selected_for_pattern"]],
        ["q_threshold_not_met", "selected_for_pattern"],
        default="outside_top_n",
    )
    terms = audit.loc[audit["selected_for_display"]].groupby("term_id", as_index=False).agg(
        description=("description", "first"),
        min_q_value=("q_value", "min"),
        min_p_value=("p_value", "min"),
        significant_patterns=("significant", "sum"),
    )
    nomination_summary = nominated.groupby("term_id", as_index=False).agg(
        selected_by_patterns=("pattern_id", lambda values: ";".join(map(str, sorted(values)))),
        best_rank_within_pattern=("rank_within_pattern", "min"),
    )
    terms = terms.merge(nomination_summary, on="term_id", validate="one_to_one")
    term_sort = ["min_q_value", "min_p_value", "term_id"]
    term_ascending = [True, True, True]
    if selection.mode == "effect-size":
        # Use eligible nominations, not a large effect failing the q threshold.
        best_effect = nominated.groupby("term_id")["effect_ranking_score"].max()
        terms["best_effect_ranking_score"] = terms["term_id"].map(best_effect)
        term_sort.insert(0, "best_effect_ranking_score")
        term_ascending.insert(0, False)
    terms = terms.sort_values(term_sort, ascending=term_ascending).reset_index(drop=True)
    terms["term_label"] = terms["description"].map(clean_description).map(
        lambda label: label[:1].upper() + label[1:]
    )
    repeated_labels = terms["term_label"].duplicated(keep=False)
    terms.loc[repeated_labels, "term_label"] += " (" + terms.loc[repeated_labels, "term_id"] + ")"
    terms["display_order"] = np.arange(len(terms), dtype=int)
    selected = audit.loc[audit["selected_for_display"]].merge(
        terms[["term_id", "term_label", "display_order"]], on="term_id", validate="many_to_one"
    )
    expected_rows = len(PATTERN_IDS) * len(terms)
    if len(selected) != expected_rows:
        observed = selected.groupby("term_id")["pattern_id"].nunique()
        missing_terms = terms.loc[
            ~terms["term_id"].isin(observed[observed.eq(5)].index),
            "term_id",
        ].tolist()
        raise ValueError(
            "Automatically selected terms are missing from one or more result tables; "
            "supply complete exports rather than imputing nonsignificance: "
            f"{missing_terms}"
        )
    selected["description_clean"] = selected["description"].map(
        clean_description
    )
    selected["direction_available"] = False
    selected = selected.sort_values(["display_order", "pattern_id"])
    return selected.reset_index(drop=True), terms, audit.sort_values(["pattern_id", "term_id"])


def build_ontology_summary(
    mapping_path: Path,
    reference_dir: Path,
    all_results: pd.DataFrame,
    selected_terms: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Path]]:
    data = pd.read_csv(mapping_path)
    require_columns(data, {"standard_name", "pattern_id", "pattern", "ora_eligible"}, "Lipid mapping")
    eligible = data.loc[as_boolean(data["ora_eligible"])].copy()
    eligible["pattern_id"] = pd.to_numeric(eligible["pattern_id"], errors="raise").astype(int)
    if eligible["standard_name"].duplicated().any():
        raise ValueError("Duplicate submitted lipid names in the cluster mapping.")
    annotations, reference_paths = reconstruct_annotations(eligible, reference_dir)
    summary, memberships, count_audit = summarize_ontology_coverage(
        annotations, all_results, selected_terms, PATTERN_IDS
    )
    submitted = all_results["submitted_lipids"].unique()
    if len(submitted) != 1 or len(eligible) != submitted[0]:
        raise ValueError("The cluster mapping and ranked results must cover the same submitted lipid set.")
    return summary, memberships, count_audit, [mapping_path, *reference_paths]


def bubble_area(annotated_percentage: np.ndarray | float) -> np.ndarray:
    # Marker area is proportional to annotation coverage, in points squared.
    return 2.0 * np.asarray(annotated_percentage, dtype=float)


def lion_term_layout(source: pd.DataFrame) -> pd.DataFrame:
    """Shared term order and relative row spacing for both files."""
    rows = source[["display_order", "term_label"]].drop_duplicates().sort_values("display_order").copy()
    rows["wrapped_label"] = rows["term_label"].map(
        lambda label: textwrap.fill(label, width=28, break_long_words=False, break_on_hyphens=False)
    )
    rows["row_height"] = rows["wrapped_label"].map(
        lambda label: max(1.0, 0.74 * (label.count("\n") + 1))
    )
    rows["plot_y"] = rows["row_height"].cumsum() - rows["row_height"] / 2
    return rows


def figure_dimensions(source: pd.DataFrame, kind: str) -> tuple[float, float, float]:
    row_units = float(lion_term_layout(source)["row_height"].sum())
    if kind == "a":
        # The current 14 terms fit at 100 mm. Larger selections may grow to
        # retain readable labels rather than dropping terms or shrinking fonts.
        outer_height_mm = 28.0
        plot_height_mm = max(FIGURE_A_HEIGHT_MM - outer_height_mm, row_units * 3.5)
    elif kind == "b":
        outer_height_mm = 35.0
        plot_height_mm = max(42.0, row_units * 5.2)
    else:
        raise ValueError(f"Unknown standalone figure kind: {kind}")
    return FIGURE_WIDTH_MM, plot_height_mm + outer_height_mm, plot_height_mm


def configure_term_axes(ax: mpl.axes.Axes, layout: pd.DataFrame, cluster_labels: list[str]) -> None:
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(float(layout["row_height"].sum()), 0)
    ax.set_yticks(layout["plot_y"])
    ax.set_yticklabels(layout["wrapped_label"], fontsize=6.2, color=TEXT_COLOUR)
    ax.set_xticks(range(5))
    ax.set_xticklabels(cluster_labels, fontsize=6.0, color=TEXT_COLOUR)
    ax.tick_params(axis="both", length=0, pad=4)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_horizontal_colorbar(
    fig: mpl.figure.Figure,
    cax: mpl.axes.Axes,
    scalar: mpl.cm.ScalarMappable,
    label: str,
    ticks: list[float] | None = None,
) -> None:
    bar = fig.colorbar(scalar, cax=cax, orientation="horizontal", ticks=ticks)
    bar.set_label(label, fontsize=6.2, color=TEXT_COLOUR, labelpad=3)
    bar.ax.tick_params(labelsize=5.8, length=2, width=0.45, pad=2, colors=SECONDARY_TEXT_COLOUR)
    bar.outline.set_visible(False)


def draw_dot_plot(ax: mpl.axes.Axes, cax: mpl.axes.Axes, source: pd.DataFrame) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "lipidomics_rose_evidence", EVIDENCE_COLOURS
    )
    evidence_max = max(2.0, float(source["neg_log10_q"].max()))
    norm = Normalize(vmin=0.0, vmax=evidence_max)
    layout = lion_term_layout(source)
    for row in layout.itertuples(index=False):
        if row.display_order % 2 == 0:
            ax.axhspan(row.plot_y - row.row_height / 2, row.plot_y + row.row_height / 2,
                       color="#F7F8FA", linewidth=0, zorder=0)
    positions = layout.set_index("display_order")["plot_y"]
    ax.scatter(
        source["pattern_id"].to_numpy() - 1,
        source["display_order"].map(positions),
        s=bubble_area(source["annotated_percentage"]),
        c=source["neg_log10_q"], cmap=cmap, norm=norm,
        edgecolors=np.where(source["significant"], SIGNIFICANCE_EDGE, "#B7BDC8"),
        linewidths=np.where(source["significant"], 0.85, 0.4), zorder=3,
    )
    configure_term_axes(ax, layout, [f"Cluster {pattern}" for pattern in PATTERN_IDS])
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E3E7ED", linewidth=0.45)
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    add_horizontal_colorbar(ax.figure, cax, scalar, "−log10(FDR q)")


def draw_ontology_heatmap(ax: mpl.axes.Axes, cax: mpl.axes.Axes, source: pd.DataFrame) -> None:
    layout = lion_term_layout(source)
    z_matrix = source.pivot(index="display_order", columns="pattern_id", values="row_z_score").reindex(
        index=layout["display_order"], columns=PATTERN_IDS
    ).to_numpy(dtype=float)
    text_matrix = source.pivot(index="display_order", columns="pattern_id", values="display_value").reindex(
        index=layout["display_order"], columns=PATTERN_IDS
    ).to_numpy()
    if not np.isfinite(z_matrix).all():
        raise ValueError("Non-finite ontology-coverage heatmap values.")
    colour_limit = max(1.5, math.ceil(float(np.abs(z_matrix).max()) * 2) / 2)
    cmap = LinearSegmentedColormap.from_list(
        "overview_blue_cream_coral", COVERAGE_COLOURS
    )
    norm = TwoSlopeNorm(vmin=-colour_limit, vcenter=0.0, vmax=colour_limit)
    y_edges = np.r_[0.0, layout["row_height"].cumsum().to_numpy()]
    mesh = ax.pcolormesh(
        np.arange(6) - 0.5, y_edges, z_matrix,
        cmap=cmap, norm=norm, shading="flat", edgecolors="white", linewidth=0.7,
    )
    for row_index, y_position in enumerate(layout["plot_y"]):
        for column in range(5):
            # Choose legible text independently of the value's significance.
            rgb = np.asarray(cmap(norm(z_matrix[row_index, column]))[:3])
            linear_rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
            luminance = float(linear_rgb @ np.array([0.2126, 0.7152, 0.0722]))
            text_colour = "white" if luminance < 0.179 else "black"
            ax.text(column, y_position, str(text_matrix[row_index, column]),
                    ha="center", va="center", fontsize=6.1, color=text_colour)
    counts = source[["pattern_id", "assigned_lipids"]].drop_duplicates().set_index("pattern_id")["assigned_lipids"]
    configure_term_axes(ax, layout, [f"Cluster {pattern}\n{int(counts[pattern])} lipids" for pattern in PATTERN_IDS])
    add_horizontal_colorbar(ax.figure, cax, mesh, "Within-row z score", [-colour_limit, 0.0, colour_limit])


def make_figure(
    source: pd.DataFrame,
    output_stem: Path,
    kind: str,
) -> tuple[list[Path], dict[str, object]]:
    """Export one standalone panel without a title or panel letter."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    width_mm, height_mm, plot_height_mm = figure_dimensions(source, kind)
    plot_bottom_mm = 25.0 if kind == "a" else 32.0
    fig = plt.figure(figsize=(width_mm / MM_PER_INCH, height_mm / MM_PER_INCH), facecolor="white")
    # Reserve the full width for the term labels and five equally spaced columns.
    # Horizontal keys below the plot retain identical column positions.
    ax = fig.add_axes([43.0 / width_mm, plot_bottom_mm / height_mm, 55.0 / width_mm, plot_height_mm / height_mm])
    cax = fig.add_axes([43.0 / width_mm, 8.0 / height_mm, 55.0 / width_mm, 2.2 / height_mm])
    if kind == "a":
        draw_dot_plot(ax, cax, source)
        key_ax = fig.add_axes([0, 15.0 / height_mm, 1, 7.5 / height_mm])
        key_ax.set(xlim=(0, 100), ylim=(0, 1))
        key_ax.set_axis_off()
        key_ax.text(3, 0.5, "Annotated lipids (%)", ha="left", va="center", fontsize=6.0, color=SECONDARY_TEXT_COLOUR)
        for x, coverage in zip((32.5, 47.0, 61.5), (10, 30, 50)):
            key_ax.scatter([x], [0.5], s=bubble_area(coverage), facecolor="#E8EBF1", edgecolor="#9BA7B7", linewidth=0.45)
            key_ax.text(x + 3.0, 0.5, str(coverage), ha="left", va="center", fontsize=6.0, color=SECONDARY_TEXT_COLOUR)
        key_ax.scatter([78.0], [0.5], s=43, facecolor="white", edgecolor=SIGNIFICANCE_EDGE, linewidth=0.85)
        key_ax.text(81.0, 0.5, "q < 0.05", ha="left", va="center", fontsize=6.0, color=SECONDARY_TEXT_COLOUR)
        submitted = source["submitted_lipids"].unique()
        fig.text(3.0 / width_mm, 8.2 / height_mm, f"{int(submitted[0])} submitted lipids",
                 ha="left", va="center", fontsize=6.0, color=SECONDARY_TEXT_COLOUR)
    elif kind == "b":
        draw_ontology_heatmap(ax, cax, source)
        fig.text(43.0 / width_mm, 18.0 / height_mm, "Cell values: annotated lipids (%)",
                 ha="left", va="center", fontsize=6.0, color=SECONDARY_TEXT_COLOUR)
        fig.text(3.0 / width_mm, 8.2 / height_mm, "Descriptive coverage",
                 ha="left", va="center", fontsize=6.0, color=SECONDARY_TEXT_COLOUR)
    else:
        raise ValueError(f"Unknown standalone figure kind: {kind}")
    geometry = {
        "width_mm": width_mm, "height_mm": height_mm,
        "plot_area_mm": [value for value in (43.0, plot_bottom_mm, 55.0, plot_height_mm)],
        "row_centres": lion_term_layout(source)["plot_y"].tolist(),
        "titles": False,
    }
    # No tight cropping: preserve the exact requested physical page width.
    svg_path = output_stem.with_suffix(".svg")
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    tiff_path = output_stem.with_suffix(".tiff")
    fig.savefig(svg_path, facecolor="white", edgecolor="none")
    fig.savefig(pdf_path, facecolor="white", edgecolor="none")
    fig.savefig(png_path, dpi=600, facecolor="white", edgecolor="none")
    fig.savefig(tiff_path, dpi=600, facecolor="white", edgecolor="none",
                pil_kwargs={"compression": "tiff_lzw"})
    outputs = [svg_path, pdf_path, png_path, tiff_path]
    plt.close(fig)
    return outputs, geometry


def write_supporting_files(
    source_dir: Path,
    lion_source: pd.DataFrame,
    ontology_source: pd.DataFrame,
    membership_audit: pd.DataFrame,
    count_audit: pd.DataFrame,
    ranked_manifest: pd.DataFrame,
    selected_terms: pd.DataFrame,
    selection_audit: pd.DataFrame,
    selection: TermSelection,
    input_paths: list[Path],
    figure_paths: list[Path],
    geometry: dict[str, object],
) -> list[Path]:
    source_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "lion_dot_plot_source_data.csv": lion_source,
        "lion_ontology_heatmap_source_data.csv": ontology_source,
        "lion_lipid_term_membership.csv": membership_audit,
        "lion_annotation_count_validation.csv": count_audit,
        "selected_lion_terms.csv": selected_terms,
        "lion_term_selection_audit.csv": selection_audit,
    }
    outputs = []
    for name, table in tables.items():
        path = source_dir / name
        table.to_csv(path, index=False)
        outputs.append(path)
    rule = selection_description(selection)
    count_rows = ontology_source[["pattern_id", "assigned_lipids", "ontology_matched_lipids", "unmatched_lipids"]].drop_duplicates().sort_values("pattern_id")
    count_text = ", ".join(f"cluster {row.pattern_id}: {row.assigned_lipids} lipid species" for row in count_rows.itertuples(index=False))
    unmatched_names = sorted(membership_audit.loc[membership_audit["annotation_status"].eq("unmatched"), "standard_name"].unique())
    nominal_n = sorted(ranked_manifest["input_lipids"].astype(int).unique())
    submitted_text = "/".join(map(str, nominal_n))
    legend_a = (
        f"Ranked LION enrichment across five molecular-age clusters. The {len(selected_terms)} "
        f"displayed terms are {rule}. All five clusters are displayed for every selected term. "
        "Original per-run Benjamini–Hochberg q values are retained without recalculation "
        "after display selection. Colour denotes −log10(FDR q); marker area is proportional "
        f"to the percentage of the {submitted_text} submitted lipids annotated to each term. "
        f"Dark outlines indicate q < {FDR_THRESHOLD:g}. Each ranking contains "
        f"{submitted_text} submitted input lipids; the hard-assigned cluster counts in figure b "
        "are not the sample sizes of these ranked tests. Colour does not indicate "
        "activation, abundance direction or effect size."
    )
    legend_b = (
        "LION annotation coverage within hard-assigned molecular-age clusters. Terms and "
        "row order are identical to figure a. Cell labels report 100 × the number of lipids "
        "annotated to a term / the number of submitted lipids assigned to that cluster. "
        "Colour shows the population z score of these percentages within each term across "
        f"the five clusters. Denominators are {count_text}. Annotations were reconstructed "
        "from the versioned official LION database and LION/web direct/smart-matching "
        f"rules; all {len(count_audit)} exported term totals match. No fatty-acid chains "
        "were predicted from summed compositions. "
        f"Unmatched submitted lipids ({', '.join(unmatched_names) if unmatched_names else 'none'}) "
        "remain in the percentage denominators, with unknown memberships explicitly "
        "recorded in Source Data. Thus percentages describe annotation coverage, not "
        "complete biological class membership. LION terms overlap and their percentages "
        "need not sum to 100%. No inferential test was performed for figure b."
    )
    legend_path = source_dir / "figure_legend.txt"
    legend_path.write_text("Figure a\n" + legend_a + "\n\nFigure b\n" + legend_b + "\n", encoding="utf-8")
    outputs.append(legend_path)
    nomination_counts = {
        str(pattern): int((selection_audit["pattern_id"].eq(pattern) & selection_audit["selected_for_pattern"]).sum())
        for pattern in PATTERN_IDS
    }
    notes = [
        "Figure contract",
        "- Question: which LION properties characterize the five molecular-age patterns?",
        "- Figure a: ranked enrichment evidence; figure b: descriptive within-cluster annotation coverage.",
        "- Two separate quantitative grids; the same terms, row order, column positions and cluster order, with independent vertical spacing.",
        "- Figure a targets 100 mm height to match the transcriptomic pathway landscape; larger term selections grow only if needed for readable labels.",
        "- Python/matplotlib; white background, sequential lipidomics rose evidence palette, overview blue-cream-coral descriptive palette.",
        "- Overview palette reused: Lipidomics #C96F75; heatmap #3F6596 / #F7F3DF / #C75C4D. No green accents.",
        "- No main title, subtitle or panel letter; cluster labels and statistical keys retained.",
        f"- Selection: {rule}.",
        f"- {len(selection_audit)} input results, {len(selected_terms)} selected terms; nominations {nomination_counts}.",
        "- Original per-run FDR retained; no new inferential comparison or multiple-testing correction.",
        "- Minimum q is a display statistic, not a global q across the five runs.",
        f"- {len(count_audit)} / {len(count_audit)} reconstructed annotation totals match the original exports.",
        "- All submitted cluster members contribute to the denominator; numerator counts unique annotated species.",
        f"- Unmatched names: {unmatched_names}; membership is missing in the audit, not a claim of biological absence.",
        "- Fatty-acid chain prediction is disabled; exact matches, published aliases, smart matching and ontology ancestry only.",
        "- A single lipid may carry several overlapping ontology annotations; percentages are not parts of a single whole.",
        "- Heatmap colour is row-standardized annotation coverage, not statistical significance; all labels have equal font weight.",
        "- n labels count lipid species, not independent human samples.",
        "- Term selection and the complete lipid-membership audit are exported.",
        f"- Export geometry: {json.dumps(geometry)}.",
        "- Each figure is exactly 100 mm wide; no tight-cropping resize is applied.",
        "- Separate 600 dpi PNG/TIFF and editable PDF/SVG.",
        "- Annotation source: https://github.com/martijnmolenaar/topOnto.LION2.db and https://github.com/martijnmolenaar/LION-web.",
    ]
    qa_path = source_dir / "figure_qa_notes.txt"
    qa_path.write_text("\n".join(notes) + "\n", encoding="utf-8")
    outputs.append(qa_path)
    manifest = {
        "analysis": "Separate LION enrichment and matched-term annotation-coverage figures",
        "backend": "python-matplotlib", "figure_geometry": geometry,
        "palette": {"ranked_evidence": list(EVIDENCE_COLOURS), "ontology_coverage": list(COVERAGE_COLOURS)},
        "fdr_threshold": FDR_THRESHOLD, "pattern_order": list(PATTERN_IDS),
        "term_selection": asdict(selection), "selection_description": rule,
        "selected_term_count": len(selected_terms), "terms_nominated_per_pattern": nomination_counts,
        "q_values_recomputed": False,
        "bubble_area": "proportional to percentage of all submitted lipids annotated to the term",
        "heatmap_colour": "within-row population z score of within-cluster annotation percentages",
        "heatmap_cell_text": "percentage of submitted, hard-assigned cluster lipids annotated to the term",
        "unmatched_lipids": unmatched_names, "unmatched_kept_in_denominator": True,
        "annotation_validation": {"terms_checked": len(count_audit), "all_match": bool(count_audit["matches_web_export"].all())},
        "cluster_counts": count_rows.to_dict(orient="records"),
        "selected_terms": selected_terms.to_dict(orient="records"),
        "input_sha256": {path.as_posix(): sha256(path) for path in input_paths},
        "figure_outputs": [path.as_posix() for path in figure_paths],
        "source_outputs": [path.as_posix() for path in outputs],
    }
    manifest_path = source_dir / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    outputs.append(manifest_path)
    return outputs


def run(
    lion_root: Path,
    mapping_path: Path,
    figure_dir: Path,
    source_dir: Path,
    selection: TermSelection = TermSelection(),
    annotation_reference: Path = DEFAULT_ANNOTATION_REFERENCE,
) -> dict[str, object]:
    all_results, ranked_manifest, lion_inputs = load_ranked_lion_results(lion_root, effect_column=selection.effect_column)
    lion_source, selected_terms, selection_audit = select_lion_terms(all_results, selection)
    ontology_source, membership_audit, count_audit, annotation_inputs = build_ontology_summary(
        mapping_path, annotation_reference, all_results, selected_terms
    )
    a_stem = figure_dir / "lion_a_ranked_enrichment_five_clusters"
    b_stem = figure_dir / "lion_b_ontology_coverage_five_clusters"
    a_paths, a_geometry = make_figure(lion_source, a_stem, "a")
    b_paths, b_geometry = make_figure(ontology_source, b_stem, "b")
    if (a_geometry["width_mm"] != b_geometry["width_mm"]
            or a_geometry["row_centres"] != b_geometry["row_centres"]
            or a_geometry["plot_area_mm"][::2] != b_geometry["plot_area_mm"][::2]):
        raise ValueError("The two output files must retain common term order and column positions.")
    geometry = {"a": a_geometry, "b": b_geometry}
    figure_paths = [*a_paths, *b_paths]
    support_paths = write_supporting_files(
        source_dir, lion_source, ontology_source, membership_audit, count_audit,
        ranked_manifest, selected_terms, selection_audit, selection,
        [*lion_inputs, *annotation_inputs], figure_paths, geometry,
    )
    return {
        "figure_files": [str(path) for path in figure_paths],
        "source_files": [str(path) for path in support_paths],
        "geometry": geometry, "selected_terms": len(selected_terms),
        "annotation_counts_verified": len(count_audit),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lion-root", type=Path, default=DEFAULT_LION_ROOT)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--annotation-reference-dir", type=Path, default=DEFAULT_ANNOTATION_REFERENCE)
    parser.add_argument("--selection", choices=("top-n", "q-value", "effect-size"), default="top-n")
    parser.add_argument("--top-n", type=int, default=5, help="Maximum nominations per pattern (default: 5).")
    parser.add_argument("--q-threshold", type=float, default=0.05, help="Strict q-value selection cutoff.")
    parser.add_argument("--effect-column", help="Actual effect-size column; required for effect-size selection.")
    parser.add_argument("--effect-order", choices=("larger", "smaller", "absolute"), default="larger")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        args.lion_root.resolve(), args.mapping.resolve(), args.figure_dir.resolve(), args.source_dir.resolve(),
        TermSelection(mode=args.selection, q_threshold=args.q_threshold, top_n=args.top_n,
                      effect_column=args.effect_column, effect_order=args.effect_order),
        args.annotation_reference_dir.resolve(),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
