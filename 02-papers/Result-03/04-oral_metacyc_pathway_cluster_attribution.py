#!/usr/bin/env python3
"""Map age-associated oral MetaCyc pathways to five fixed oral-taxon patterns.

This script is an independent downstream bridge between two completed analyses:

1. ``04-oral_metacyc_pathway_age_association.py`` supplies participant-level
   community-pathway age associations and the age-bin display matrix;
2. ``04-oral_microbiome_pathway_analysis.ipynb`` supplies the complete frozen
   oral taxon-to-pattern universe; current-cohort prevalence is audit metadata
   and never changes cluster membership; and
3. HUMAnN ``pathway_abundance_microbe_level.arrow`` supplies species-stratified
   pathway RPK contributions.

Clustering is never recomputed. For each displayed pathway and participant,
species-stratified RPK is summed within Cluster1-Cluster5. Two estimands are kept separate:
cluster coverage is fixed-cluster RPK divided by all species-stratified RPK,
whereas within-cluster composition divides each Cluster1-Cluster5 contribution by the sum
of Cluster1-Cluster5 only. Resolved species outside the cluster universe and unresolved
taxa remain in Source Data and the coverage denominator, but they are not
displayed as a fictitious sixth cluster. Participant-level fractions are
averaged so each participant has equal weight. A pathway receives a descriptive
within-cluster label only when one pattern accounts for at least 50% of the
Cluster1-Cluster5 contribution; otherwise it is labelled ``Mixed``. This label never means
that the pattern dominates the whole-community pathway.

The 100-mm main figure combines full pathway names, continuous-age effects
with 95% CIs, and Cluster1-Cluster5 within-cluster composition. A separate 100-mm
supplementary figure retains age-bin abundance, cluster coverage, multiplicity
statistics, representative taxa and cohort information. Both retain the same
pathway order and omit region titles. Taxon patterns are feature clusters, not
participant groups, and attribution is not an independent validation of the
age association or a decomposition of the age regression coefficient.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


ANALYSIS_VERSION = "2.1.2-compact-rows"
PATTERN_IDS = (1, 2, 3, 4, 5)
FIGURE_WIDTH_MM = 100.0
TEXT = "#283B40"
MUTED = "#617579"
MISSING = "#E7ECEC"
EFFECT_COLOR = "#316F78"
AGE_LOW = "#6495AF"
AGE_MID = "#F8FAF9"
AGE_HIGH = "#CD8975"
PATTERN_COLORS = {
    1: "#80ACC8",
    2: "#E0BD74",
    3: "#7AB49C",
    4: "#B39BC8",
    5: "#D89181",
    0: "#9E9E9E",
}
PATTERN_TEXT_COLORS = {
    1: "#005A8D",
    2: "#8A5A00",
    3: "#006B4F",
    4: "#8D456F",
    5: "#A64100",
    0: "#666666",
}
REQUIRED_UNIVERSE_COLUMNS = {
    "species_key",
    "species_display",
    "pattern_id",
    "pattern",
    "species_prevalence_status",
}
REQUIRED_SELECTED_COLUMNS = {
    "pathway_id",
    "pathway_name",
    "n",
    "beta_pathway_sd_per_10y",
    "ci95_low_pathway_sd_per_10y",
    "ci95_high_pathway_sd_per_10y",
    "p_bonferroni_global",
    "q_bh_global",
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
        "font.size": 7,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


class AnalysisSkip(RuntimeError):
    """A required evidence link is absent, so no combined figure is made."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path, max_bytes: int = 128 * 1024 * 1024) -> str | None:
    if path.stat().st_size > max_bytes:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = sha256_file(path)
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime,
            tz=timezone.utc,
        ).isoformat(),
        "sha256": digest,
        "sha256_status": "computed" if digest else "skipped_for_large_file",
    }


def split_csv(value: str) -> list[str]:
    columns = [item.strip() for item in value.split(",") if item.strip()]
    if not columns:
        raise ValueError("At least one sample-key column is required")
    if len(columns) != len(set(columns)):
        raise ValueError(f"Duplicate columns in comma-separated list: {columns}")
    return columns


def normalize_key_component(
    values: pd.Series,
    column_name: str,
    source_label: str,
) -> pd.Series:
    normalized = values.astype("string").str.strip()
    missing_tokens = {"", "nan", "none", "null", "na", "n/a", "<na>"}
    normalized = normalized.mask(normalized.str.casefold().isin(missing_tokens))
    if normalized.isna().any():
        raise ValueError(
            f"{source_label} sample-key column {column_name!r} contains "
            f"{int(normalized.isna().sum())} missing or empty values"
        )
    integer_like = normalized.str.fullmatch(r"[+-]?\d+\.0+", na=False)
    normalized = normalized.mask(
        integer_like,
        normalized.str.replace(r"\.0+$", "", regex=True),
    )
    return normalized.astype(str)


def build_sample_keys(
    frame: pd.DataFrame,
    key_columns: Iterable[str],
    source_label: str,
) -> pd.Series:
    columns = list(key_columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{source_label} lacks sample-key columns {missing}; "
            f"available columns begin with {list(map(str, frame.columns[:20]))}"
        )
    components = [
        normalize_key_component(frame[column], column, source_label)
        for column in columns
    ]
    if len(components) == 1:
        keys = components[0]
    else:
        escaped = [
            component.str.replace("%", "%25", regex=False)
            .str.replace("|", "%7C", regex=False)
            .str.replace("=", "%3D", regex=False)
            for component in components
        ]
        keys = pd.Series("", index=frame.index, dtype="string")
        for column, component in zip(columns, escaped):
            prefix = "" if keys.eq("").all() else "|"
            keys = keys + prefix + column + "=" + component
        keys = keys.astype(str)
    duplicate = keys.duplicated(keep=False)
    if duplicate.any():
        examples = sorted(keys.loc[duplicate].unique())[:5]
        raise ValueError(
            f"{source_label} sample key is not unique: {int(duplicate.sum())} "
            f"rows participate in duplicates; examples: {examples}"
        )
    return keys.reset_index(drop=True)


def read_arrow_schema(path: Path) -> pa.Schema:
    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_file(source)
        return reader.schema


def split_pathway_label(raw_label: object) -> tuple[str, str]:
    raw = str(raw_label).strip()
    if ":" in raw:
        pathway_id, pathway_name = raw.split(":", 1)
        return pathway_id.strip(), pathway_name.strip()
    return raw, ""


def canonical_species(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    tail = str(value).strip().split("|")[-1]
    if "s__" not in tail:
        return None
    species = tail.rsplit("s__", 1)[1]
    species = species.split(".t__", 1)[0]
    species = re.sub(r"[^A-Za-z0-9]+", "_", species).strip("_").casefold()
    return species or None


def numeric_column(values: pd.Series, column_name: str) -> np.ndarray:
    source_missing = values.isna().to_numpy(dtype=bool)
    converted = pd.to_numeric(values, errors="coerce")
    converted_missing = converted.isna().to_numpy(dtype=bool)
    conversion_failure = (~source_missing) & converted_missing
    if conversion_failure.any():
        raise ValueError(
            f"HUMAnN column {column_name!r} contains "
            f"{int(conversion_failure.sum())} non-numeric values"
        )
    numeric = converted.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(numeric).any():
        raise ValueError(f"HUMAnN column {column_name!r} contains infinite values")
    finite = numeric[np.isfinite(numeric)]
    if finite.size and np.any(finite < 0):
        raise ValueError(f"HUMAnN column {column_name!r} contains negative RPK")
    return numeric


def require_matplotlib_panel_alignment(
    fig: mpl.figure.Figure,
    axes: dict[str, mpl.axes.Axes],
    json_out: Path,
    tolerance_pt: float = 1.5,
) -> dict[str, object]:
    """Block export unless all row-aligned data rectangles share their edges."""

    fig.canvas.draw()
    positions = {name: axis.get_position() for name, axis in axes.items()}
    height_points = fig.get_figheight() * 72.0
    top_spread_pt = (
        max(position.y1 for position in positions.values())
        - min(position.y1 for position in positions.values())
    ) * height_points
    bottom_spread_pt = (
        max(position.y0 for position in positions.values())
        - min(position.y0 for position in positions.values())
    ) * height_points
    ordered = sorted(positions.values(), key=lambda position: position.x0)
    width_points = fig.get_figwidth() * 72.0
    gutters_pt = [
        (right.x0 - left.x1) * width_points
        for left, right in zip(ordered[:-1], ordered[1:])
    ]
    gutter_spread_pt = max(gutters_pt, default=0) - min(gutters_pt, default=0)
    passed = (
        top_spread_pt <= tolerance_pt and bottom_spread_pt <= tolerance_pt
        and gutter_spread_pt <= tolerance_pt
        and all(gutter >= 0 for gutter in gutters_pt)
    )
    payload = {
        "status": "PASS" if passed else "FIX BEFORE DELIVERY",
        "tolerance_pt": float(tolerance_pt),
        "top_edge_spread_pt": float(top_spread_pt),
        "bottom_edge_spread_pt": float(bottom_spread_pt),
        "gutter_spread_pt": float(gutter_spread_pt),
        "gutters_pt": gutters_pt,
        "intentional_unequal_panel_widths": True,
        "unequal_width_reason": (
            "full pathway labels, effect or heatmap evidence, composition and "
            "multiplicity text need different physical widths at 100 mm"
        ),
        "panels": {
            name: {
                "x0": float(position.x0),
                "y0": float(position.y0),
                "x1": float(position.x1),
                "y1": float(position.y1),
            }
            for name, position in positions.items()
        },
    }
    write_json(json_out, payload)
    if not passed:
        raise RuntimeError(
            f"Final panel alignment exceeds {tolerance_pt:g} pt: "
            f"top={top_spread_pt:.3f}, bottom={bottom_spread_pt:.3f}, "
            f"gutters={gutter_spread_pt:.3f}"
        )
    return payload


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    source_label: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source_label} lacks required columns {missing}")


def load_age_analysis(
    age_dir: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
    dict[str, Path],
]:
    paths = {
        "manifest": age_dir / "oral_metacyc_pathway_age_association_manifest.json",
        "selected": age_dir / "figure_pathways.csv",
        "age_heatmap": age_dir / "figure_heatmap_row_zscores.csv",
        "age_bin_source": age_dir / "figure_age_bin_source_data.csv",
        "sample_audit": age_dir / "metacyc_sample_preprocessing_audit.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise AnalysisSkip(
            "The completed age-association output is incomplete; missing: "
            + "; ".join(missing)
        )
    age_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if age_manifest.get("status") != "completed":
        raise AnalysisSkip(
            "The age-association manifest is not completed with a figure: "
            f"status={age_manifest.get('status')!r}"
        )

    selected = pd.read_csv(paths["selected"])
    require_columns(selected, REQUIRED_SELECTED_COLUMNS, "age figure pathway table")
    if selected.empty:
        raise AnalysisSkip("The age-association figure contains no pathway")
    if selected["pathway_id"].astype(str).duplicated().any():
        raise ValueError("Age figure pathway IDs are not unique")

    heatmap = pd.read_csv(paths["age_heatmap"])
    require_columns(heatmap, {"pathway_id"}, "age heatmap table")
    heatmap["pathway_id"] = heatmap["pathway_id"].astype(str)
    if heatmap["pathway_id"].duplicated().any():
        raise ValueError("Age heatmap pathway IDs are not unique")
    heatmap = heatmap.set_index("pathway_id")
    missing_heat = sorted(set(selected["pathway_id"].astype(str)) - set(heatmap.index))
    if missing_heat:
        raise ValueError(
            "Age heatmap is missing selected pathway IDs: " + ", ".join(missing_heat)
        )
    heatmap = heatmap.apply(pd.to_numeric, errors="raise")

    age_bin_source = pd.read_csv(paths["age_bin_source"])
    require_columns(age_bin_source, {"pathway_id", "age_bin", "n"}, "age-bin source")
    sample_audit = pd.read_csv(paths["sample_audit"])
    require_columns(
        sample_audit,
        {
            "analysis_row_index_after_qc",
            "sample_key_sha256",
            "sample_preprocessing_status",
        },
        "age sample preprocessing audit",
    )
    return selected, heatmap, age_bin_source, sample_audit, age_manifest, paths


def load_pattern_universe(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    universe = pd.read_csv(path)
    if "species_prevalence_status" not in universe.columns:
        raise AnalysisSkip(
            "The taxon-pattern universe was generated by the earlier "
            "post-clustering prevalence-filter workflow. Rerun the revised 08 "
            "analysis so the file contains the complete frozen cluster universe."
        )
    require_columns(universe, REQUIRED_UNIVERSE_COLUMNS, "oral taxon pattern universe")
    universe = universe.copy()
    universe["species_key"] = universe["species_key"].astype(str).str.strip()
    universe["species_display"] = universe["species_display"].astype(str).str.strip()
    universe["pattern_id"] = pd.to_numeric(
        universe["pattern_id"], errors="raise"
    ).astype(int)
    universe["pattern"] = universe["pattern"].astype(str).str.strip()
    universe["species_prevalence_status"] = (
        universe["species_prevalence_status"].astype(str).str.strip()
    )
    if universe["species_key"].eq("").any():
        raise ValueError("Oral taxon pattern universe contains an empty species key")
    if universe["species_key"].duplicated().any():
        duplicate = universe.loc[
            universe["species_key"].duplicated(keep=False), "species_key"
        ].head(5)
        raise ValueError(
            "Oral taxon pattern universe has duplicate species keys: "
            + ", ".join(duplicate)
        )
    unexpected = sorted(set(universe["pattern_id"]) - set(PATTERN_IDS))
    if unexpected:
        raise ValueError(f"Unexpected oral pattern IDs: {unexpected}")
    missing_patterns = sorted(set(PATTERN_IDS) - set(universe["pattern_id"]))
    if missing_patterns:
        raise AnalysisSkip(
            "The fixed oral taxon universe lacks patterns: "
            + ", ".join(f"Cluster{value}" for value in missing_patterns)
        )
    label_counts = universe.groupby("pattern_id")["pattern"].nunique()
    if (label_counts != 1).any():
        raise ValueError("Each pattern ID must map to exactly one pattern label")
    universe = universe.sort_values(
        ["pattern_id", "species_key"], kind="mergesort"
    ).reset_index(drop=True)
    counts = {
        str(pattern_id): int((universe["pattern_id"] == pattern_id).sum())
        for pattern_id in PATTERN_IDS
    }
    labels = {
        str(pattern_id): str(
            universe.loc[universe["pattern_id"] == pattern_id, "pattern"].iloc[0]
        )
        for pattern_id in PATTERN_IDS
    }
    prevalence_status_counts = {
        str(key): int(value)
        for key, value in universe["species_prevalence_status"].value_counts().items()
    }
    audit = {
        "taxa_n": int(len(universe)),
        "pattern_taxa_counts": counts,
        "pattern_labels": labels,
        "prevalence_audit_status_counts": prevalence_status_counts,
        "universe_definition": (
            "complete frozen oral-taxon cluster assignment; prevalence audit "
            "metadata does not alter membership"
        ),
        "inference_unit": "oral microbial taxon",
        "clustering_recomputed": False,
        "post_clustering_taxon_exclusion": False,
    }
    return universe, audit


def resolve_sample_key_columns(
    requested: str,
    age_manifest: dict[str, object],
) -> list[str]:
    manifest_value = age_manifest.get("arguments", {}).get(
        "sample_key_columns", "participant_id"
    )
    if isinstance(manifest_value, (list, tuple)):
        manifest_columns = [str(value) for value in manifest_value]
    else:
        manifest_columns = split_csv(str(manifest_value))
    if requested.strip():
        requested_columns = split_csv(requested)
        if requested_columns != manifest_columns:
            raise ValueError(
                "--sample-key-columns does not match the completed age analysis: "
                f"requested={requested_columns}, age analysis={manifest_columns}"
            )
        return requested_columns
    return manifest_columns


def align_microbe_samples(
    microbe_path: Path,
    schema: pa.Schema,
    sample_key_columns: list[str],
    sample_audit: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    schema_names = set(schema.names)
    missing = [column for column in sample_key_columns if column not in schema_names]
    if missing:
        raise ValueError(
            "Microbe-level HUMAnN Arrow lacks sample-key columns "
            f"{missing}; positional matching is prohibited"
        )
    key_frame = feather.read_table(
        microbe_path,
        columns=sample_key_columns,
        memory_map=True,
    ).to_pandas(ignore_metadata=True)
    microbe_keys = build_sample_keys(
        key_frame,
        sample_key_columns,
        "microbe-level HUMAnN Arrow",
    )
    microbe_hashes = pd.Series(
        [hashlib.sha256(value.encode("utf-8")).hexdigest() for value in microbe_keys],
        dtype="string",
    )
    if microbe_hashes.duplicated().any():
        raise RuntimeError("Hashed microbe sample keys are unexpectedly non-unique")

    retained = sample_audit.loc[
        sample_audit["sample_preprocessing_status"]
        .astype(str)
        .eq("retained_for_age_association")
    ].copy()
    retained["analysis_row_index_after_qc"] = pd.to_numeric(
        retained["analysis_row_index_after_qc"], errors="raise"
    ).astype(int)
    retained = retained.sort_values("analysis_row_index_after_qc", kind="mergesort")
    retained_hashes = retained["sample_key_sha256"].astype(str).str.strip()
    if retained_hashes.duplicated().any():
        raise ValueError("Age sample preprocessing audit has duplicate retained hashes")

    microbe_index = pd.Index(microbe_hashes.astype(str))
    row_indices = microbe_index.get_indexer(retained_hashes)
    present = row_indices >= 0
    selected_rows = row_indices[present].astype(np.int64)
    if selected_rows.size == 0:
        raise AnalysisSkip(
            "No age-analysis participant is present in the microbe-level HUMAnN Arrow"
        )

    alignment_table = pd.DataFrame(
        [
            {
                "source": "age_association_final_cohort",
                "input_samples": int(len(retained_hashes)),
                "shared_samples": int(present.sum()),
                "source_only_samples": int((~present).sum()),
            },
            {
                "source": "microbe_level_humann_arrow",
                "input_samples": int(len(microbe_hashes)),
                "shared_samples": int(present.sum()),
                "source_only_samples": int(
                    (~microbe_hashes.isin(set(retained_hashes))).sum()
                ),
            },
        ]
    )
    shared_hash = hashlib.sha256(
        "\n".join(sorted(retained_hashes.loc[present].astype(str))).encode("utf-8")
    ).hexdigest()
    audit = {
        "sample_key_columns": sample_key_columns,
        "age_analysis_samples": int(len(retained_hashes)),
        "microbe_arrow_samples": int(len(microbe_hashes)),
        "shared_samples_for_pattern_attribution": int(present.sum()),
        "age_samples_missing_from_microbe_arrow": int((~present).sum()),
        "microbe_samples_outside_age_cohort": int(
            (~microbe_hashes.isin(set(retained_hashes))).sum()
        ),
        "shared_sample_key_set_sha256": shared_hash,
        "identifiers_exported": False,
        "alignment_policy": (
            "SHA-256 hashes of normalized unique sample keys from the completed "
            "age analysis; no positional or row-count matching"
        ),
    }
    return selected_rows, alignment_table, audit


def parse_microbe_columns(
    schema: pa.Schema,
    pathway_ids: list[str],
    universe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    selected_set = set(pathway_ids)
    pattern_map = universe.set_index("species_key")["pattern_id"].to_dict()
    pattern_name_map = universe.set_index("species_key")["pattern"].to_dict()
    display_map = universe.set_index("species_key")["species_display"].to_dict()
    rows: list[dict[str, object]] = []
    all_selected_stratified = 0
    for field in schema:
        column = str(field.name)
        if "|" not in column:
            continue
        pathway_raw, taxon_raw = column.rsplit("|", 1)
        pathway_id, pathway_name = split_pathway_label(pathway_raw)
        if pathway_id not in selected_set:
            continue
        all_selected_stratified += 1
        if not (
            pa.types.is_integer(field.type)
            or pa.types.is_floating(field.type)
            or pa.types.is_decimal(field.type)
        ):
            raise ValueError(
                f"Selected HUMAnN contribution column {column!r} is non-numeric"
            )
        species_key = canonical_species(taxon_raw)
        if species_key is None:
            assignment_status = "unassigned_non_species_taxon"
            pattern_id = 0
            pattern_name = "Unassigned"
            species_name = str(taxon_raw).strip() or "Unresolved taxon"
        elif species_key not in pattern_map:
            assignment_status = "unassigned_outside_fixed_oral_universe"
            pattern_id = 0
            pattern_name = "Unassigned"
            species_name = species_key.replace("_", " ")
        else:
            assignment_status = "assigned_to_fixed_oral_pattern"
            pattern_id = int(pattern_map[species_key])
            pattern_name = str(pattern_name_map[species_key])
            species_name = str(display_map[species_key])
        rows.append(
            {
                "column": column,
                "arrow_type": str(field.type),
                "pathway_id": pathway_id,
                "pathway_name_from_microbe_arrow": pathway_name,
                "taxon_raw": taxon_raw,
                "species_key": species_key or "",
                "species_display": species_name,
                "pattern_id": pattern_id,
                "pattern": pattern_name,
                "assignment_status": assignment_status,
            }
        )
    mapping = pd.DataFrame(rows)
    if mapping.empty:
        raise AnalysisSkip(
            "No species-stratified HUMAnN column matches a displayed age-associated "
            "MetaCyc pathway"
        )
    represented = set(mapping["pathway_id"].astype(str))
    audit = {
        "selected_age_pathways": int(len(pathway_ids)),
        "selected_pathways_with_stratified_columns": int(len(represented)),
        "selected_pathways_without_stratified_columns": sorted(
            selected_set - represented
        ),
        "selected_stratified_columns": int(all_selected_stratified),
        "assigned_stratified_columns": int((mapping["pattern_id"] > 0).sum()),
        "unassigned_stratified_columns": int((mapping["pattern_id"] == 0).sum()),
        "resolved_species_outside_fixed_cluster_columns": int(
            mapping["assignment_status"]
            .eq("unassigned_outside_fixed_oral_universe")
            .sum()
        ),
        "unresolved_non_species_columns": int(
            mapping["assignment_status"].eq("unassigned_non_species_taxon").sum()
        ),
        "resolved_species_outside_fixed_cluster_taxa": int(
            mapping.loc[
                mapping["assignment_status"].eq(
                    "unassigned_outside_fixed_oral_universe"
                ),
                "species_key",
            ].nunique()
        ),
        "matched_pattern_taxa": int(
            mapping.loc[mapping["pattern_id"] > 0, "species_key"].nunique()
        ),
        "fixed_pattern_taxa": int(len(universe)),
    }
    return mapping, audit


def summarize_pattern_contributions(
    microbe_path: Path,
    row_indices: np.ndarray,
    mapping: pd.DataFrame,
    selected: pd.DataFrame,
    universe: pd.DataFrame,
    batch_columns: int,
    min_contribution_samples: int,
    dominant_fraction_threshold: float,
    representative_taxa_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    pathway_ids = selected["pathway_id"].astype(str).tolist()
    pathway_index = {value: index for index, value in enumerate(pathway_ids)}
    n_samples = int(len(row_indices))
    n_pathways = int(len(pathway_ids))
    total = np.zeros((n_samples, n_pathways), dtype=float)
    pattern_totals = np.zeros((n_samples, n_pathways, len(PATTERN_IDS)), dtype=float)
    outside_cluster_total = np.zeros((n_samples, n_pathways), dtype=float)
    unresolved_taxon_total = np.zeros((n_samples, n_pathways), dtype=float)
    pair_values: dict[tuple[str, str], np.ndarray] = {}
    source_missing_values = 0

    mapping_index = mapping.set_index("column", drop=False)
    columns = mapping["column"].astype(str).tolist()
    for start in range(0, len(columns), batch_columns):
        batch = columns[start : start + batch_columns]
        frame = feather.read_table(
            microbe_path,
            columns=batch,
            memory_map=True,
        ).to_pandas(ignore_metadata=True)
        frame = frame.iloc[row_indices].reset_index(drop=True)
        for column in batch:
            info = mapping_index.loc[column]
            if isinstance(info, pd.DataFrame):
                raise RuntimeError(f"Microbe mapping is non-unique for {column!r}")
            values = numeric_column(frame[column], column)
            missing = np.isnan(values)
            source_missing_values += int(missing.sum())
            zero_filled = values.copy()
            zero_filled[missing] = 0.0
            p_index = pathway_index[str(info["pathway_id"])]
            total[:, p_index] += zero_filled
            pattern_id = int(info["pattern_id"])
            if pattern_id in PATTERN_IDS:
                pattern_totals[:, p_index, pattern_id - 1] += zero_filled
                pair_key = (str(info["pathway_id"]), str(info["species_key"]))
                if pair_key not in pair_values:
                    pair_values[pair_key] = np.zeros(n_samples, dtype=float)
                pair_values[pair_key] += zero_filled
            elif str(info["assignment_status"]) == (
                "unassigned_outside_fixed_oral_universe"
            ):
                outside_cluster_total[:, p_index] += zero_filled
            else:
                unresolved_taxon_total[:, p_index] += zero_filled

    assigned_total = pattern_totals.sum(axis=2)
    unassigned_total = outside_cluster_total + unresolved_taxon_total
    tolerance = np.maximum(1e-10, np.abs(total) * 1e-10)
    partition_difference = total - (assigned_total + unassigned_total)
    if np.any(np.abs(partition_difference) > tolerance):
        raise RuntimeError(
            "Cluster1-Cluster5, outside-cluster species and unresolved taxa do not partition "
            "all stratified RPK"
        )

    universe_index = universe.set_index("species_key")
    mapping_source_counts = (
        mapping.loc[mapping["pattern_id"] > 0]
        .groupby(["pathway_id", "species_key"], dropna=False)["column"]
        .nunique()
        .to_dict()
    )
    pair_rows: list[dict[str, object]] = []
    for (pathway_id, species_key), values in pair_values.items():
        taxon = universe_index.loc[species_key]
        detected = int(np.count_nonzero(values > 0))
        pair_rows.append(
            {
                "pathway_id": pathway_id,
                "species_key": species_key,
                "species_display": str(taxon["species_display"]),
                "pattern_id": int(taxon["pattern_id"]),
                "pattern": str(taxon["pattern"]),
                "detected_samples": detected,
                "mean_rpk_all_aligned_samples": float(values.mean()),
                "total_rpk_all_aligned_samples": float(values.sum()),
                "source_columns": int(
                    mapping_source_counts.get((pathway_id, species_key), 0)
                ),
                "stable_contributor": bool(detected >= min_contribution_samples),
            }
        )
    pair_table = pd.DataFrame(pair_rows)
    if pair_table.empty:
        pair_table = pd.DataFrame(
            columns=[
                "pathway_id",
                "species_key",
                "species_display",
                "pattern_id",
                "pattern",
                "detected_samples",
                "mean_rpk_all_aligned_samples",
                "total_rpk_all_aligned_samples",
                "source_columns",
                "stable_contributor",
            ]
        )

    selected_names = selected.set_index("pathway_id")["pathway_name"].to_dict()
    pattern_counts = universe["pattern_id"].value_counts().to_dict()
    pattern_labels = (
        universe.drop_duplicates("pattern_id")
        .set_index("pattern_id")["pattern"]
        .to_dict()
    )
    long_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for pathway_id in pathway_ids:
        p_index = pathway_index[pathway_id]
        valid_total = total[:, p_index] > 0
        valid_cluster = assigned_total[:, p_index] > 0
        n_valid_total = int(valid_total.sum())
        n_valid_cluster = int(valid_cluster.sum())
        component_values = np.column_stack(
            [
                pattern_totals[:, p_index, :],
                outside_cluster_total[:, p_index],
                unresolved_taxon_total[:, p_index],
            ]
        )
        if n_valid_total:
            total_fractions = (
                component_values[valid_total] / total[valid_total, p_index, None]
            )
            mean_total_fraction = total_fractions.mean(axis=0)
            median_total_fraction = np.median(total_fractions, axis=0)
            pooled_total_fraction = (
                component_values.sum(axis=0) / total[:, p_index].sum()
            )
            coverage_values = (
                assigned_total[valid_total, p_index] / total[valid_total, p_index]
            )
            cluster_coverage_mean = float(coverage_values.mean())
            cluster_coverage_median = float(np.median(coverage_values))
            cluster_coverage_q25 = float(np.quantile(coverage_values, 0.25))
            cluster_coverage_q75 = float(np.quantile(coverage_values, 0.75))
            cluster_coverage_pooled = float(
                assigned_total[:, p_index].sum() / total[:, p_index].sum()
            )
        else:
            mean_total_fraction = np.full(7, np.nan)
            median_total_fraction = np.full(7, np.nan)
            pooled_total_fraction = np.full(7, np.nan)
            cluster_coverage_mean = np.nan
            cluster_coverage_median = np.nan
            cluster_coverage_q25 = np.nan
            cluster_coverage_q75 = np.nan
            cluster_coverage_pooled = np.nan

        if n_valid_cluster:
            conditional_fractions = (
                pattern_totals[valid_cluster, p_index, :]
                / assigned_total[valid_cluster, p_index, None]
            )
            mean_within_cluster = conditional_fractions.mean(axis=0)
            median_within_cluster = np.median(conditional_fractions, axis=0)
            pooled_within_cluster = (
                pattern_totals[:, p_index, :].sum(axis=0)
                / assigned_total[:, p_index].sum()
            )
        else:
            mean_within_cluster = np.full(5, np.nan)
            median_within_cluster = np.full(5, np.nan)
            pooled_within_cluster = np.full(5, np.nan)

        if n_valid_cluster and np.isfinite(mean_within_cluster).any():
            maximum = float(np.nanmax(mean_within_cluster))
            tied = np.flatnonzero(np.isclose(mean_within_cluster, maximum, atol=1e-12))
            dominant_pattern = int(tied[0] + 1) if len(tied) == 1 else 0
            if dominant_pattern and maximum >= dominant_fraction_threshold:
                classification = f"Cluster{dominant_pattern}"
                classification_status = "predominant_within_fixed_cluster_taxa"
            else:
                classification = "Mixed"
                classification_status = (
                    "mixed_within_fixed_cluster_taxa_no_pattern_reaches_threshold"
                )
                dominant_pattern = 0
            if dominant_pattern:
                sample_patterns = pattern_totals[valid_cluster, p_index, :]
                row_max = sample_patterns.max(axis=1)
                unique_max = (
                    np.isclose(sample_patterns, row_max[:, None]).sum(axis=1) == 1
                )
                row_dominant = sample_patterns.argmax(axis=1) + 1
                stable_mask = unique_max & (row_dominant == dominant_pattern)
                sample_dominance_support = float(stable_mask.mean())
            else:
                sample_dominance_support = np.nan
        else:
            maximum = np.nan
            dominant_pattern = 0
            classification = "No clustered contribution"
            classification_status = "no_fixed_cluster_taxon_contribution"
            sample_dominance_support = np.nan

        stable_pairs = pair_table.loc[
            pair_table["pathway_id"].astype(str).eq(pathway_id)
            & pair_table["stable_contributor"].astype(bool)
        ].sort_values(
            ["mean_rpk_all_aligned_samples", "detected_samples", "species_key"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        representative = stable_pairs.head(representative_taxa_n)
        representative_taxa = "; ".join(
            f"{row.species_display} (Cluster{int(row.pattern_id)})"
            for row in representative.itertuples(index=False)
        )
        stable_counts = stable_pairs.groupby("pattern_id")["species_key"].nunique()

        components: list[dict[str, object]] = []
        for pattern_id in PATTERN_IDS:
            components.append(
                {
                    "component_id": pattern_id,
                    "component": f"Cluster{pattern_id}",
                    "component_scope": "fixed_cluster_pattern",
                    "pattern_name": str(pattern_labels[pattern_id]),
                    "pattern_taxa_n": int(pattern_counts.get(pattern_id, 0)),
                    "stable_contributor_taxa_n": int(stable_counts.get(pattern_id, 0)),
                    "component_rpk": pattern_totals[:, p_index, pattern_id - 1],
                    "within_cluster_index": pattern_id - 1,
                }
            )
        components.extend(
            [
                {
                    "component_id": 0,
                    "component": "Outside fixed cluster",
                    "component_scope": "resolved_species_outside_fixed_cluster",
                    "pattern_name": "Resolved species outside the frozen Cluster1-Cluster5 universe",
                    "pattern_taxa_n": np.nan,
                    "stable_contributor_taxa_n": np.nan,
                    "component_rpk": outside_cluster_total[:, p_index],
                    "within_cluster_index": None,
                },
                {
                    "component_id": -1,
                    "component": "Unresolved taxon",
                    "component_scope": "non_species_or_unresolved_taxon",
                    "pattern_name": "HUMAnN contribution not resolved to a matched species",
                    "pattern_taxa_n": np.nan,
                    "stable_contributor_taxa_n": np.nan,
                    "component_rpk": unresolved_taxon_total[:, p_index],
                    "within_cluster_index": None,
                },
            ]
        )
        for component_index, component in enumerate(components):
            component_rpk = np.asarray(component["component_rpk"], dtype=float)
            within_index = component["within_cluster_index"]
            long_rows.append(
                {
                    "pathway_id": pathway_id,
                    "pathway_name": str(selected_names[pathway_id]),
                    "component_id": component["component_id"],
                    "component": component["component"],
                    "component_scope": component["component_scope"],
                    "pattern_name": component["pattern_name"],
                    "pattern_taxa_n": component["pattern_taxa_n"],
                    "stable_contributor_taxa_n": component["stable_contributor_taxa_n"],
                    "age_analysis_samples": int(selected["n"].iloc[0]),
                    "microbe_attribution_samples": n_samples,
                    "samples_with_any_stratified_contribution": n_valid_total,
                    "samples_with_any_fixed_cluster_contribution": n_valid_cluster,
                    "samples_with_component_contribution": int(
                        np.count_nonzero(component_rpk > 0)
                    ),
                    "mean_fraction_of_all_stratified_rpk": float(
                        mean_total_fraction[component_index]
                    ),
                    "median_fraction_of_all_stratified_rpk": float(
                        median_total_fraction[component_index]
                    ),
                    "pooled_fraction_of_all_stratified_rpk": float(
                        pooled_total_fraction[component_index]
                    ),
                    "mean_within_cluster_fraction": (
                        float(mean_within_cluster[int(within_index)])
                        if within_index is not None
                        else np.nan
                    ),
                    "median_within_cluster_fraction": (
                        float(median_within_cluster[int(within_index)])
                        if within_index is not None
                        else np.nan
                    ),
                    "pooled_within_cluster_fraction": (
                        float(pooled_within_cluster[int(within_index)])
                        if within_index is not None
                        else np.nan
                    ),
                    "mean_rpk_all_aligned_samples": float(component_rpk.mean()),
                    "total_rpk_all_aligned_samples": float(component_rpk.sum()),
                    "all_stratified_fraction_denominator": (
                        "all species-stratified RPK for this pathway; participant "
                        "fractions summarized among participants with total RPK > 0"
                    ),
                    "within_cluster_fraction_denominator": (
                        "Cluster1-Cluster5 fixed-cluster RPK only; participant fractions "
                        "summarized among participants with fixed-cluster RPK > 0"
                    ),
                }
            )

        summary_row: dict[str, object] = {
            "pathway_id": pathway_id,
            "pathway_name": str(selected_names[pathway_id]),
            "age_analysis_samples": int(selected["n"].iloc[0]),
            "microbe_attribution_samples": n_samples,
            "samples_with_any_stratified_contribution": n_valid_total,
            "samples_with_any_fixed_cluster_contribution": n_valid_cluster,
            "cluster_coverage_mean": cluster_coverage_mean,
            "cluster_coverage_median": cluster_coverage_median,
            "cluster_coverage_q25": cluster_coverage_q25,
            "cluster_coverage_q75": cluster_coverage_q75,
            "cluster_coverage_pooled": cluster_coverage_pooled,
            "outside_cluster_species_fraction_mean": float(mean_total_fraction[5]),
            "unresolved_taxon_fraction_mean": float(mean_total_fraction[6]),
            "unassigned_fraction_mean": float(
                mean_total_fraction[5] + mean_total_fraction[6]
            ),
            "dominant_pattern": dominant_pattern,
            "dominant_pattern_label": classification,
            "dominant_pattern_fraction_within_cluster": maximum,
            "dominant_pattern_fraction_of_all_stratified_rpk": (
                float(mean_total_fraction[dominant_pattern - 1])
                if dominant_pattern
                else np.nan
            ),
            "dominant_pattern_sample_support_within_cluster": (
                sample_dominance_support
            ),
            "dominant_fraction_threshold": dominant_fraction_threshold,
            "classification_status": classification_status,
            "classification_scope": (
                "relative composition within frozen Cluster1-Cluster5 taxa; not dominance "
                "of the whole-community pathway"
            ),
            "stable_assigned_contributor_taxa_n": int(
                stable_pairs["species_key"].nunique()
            ),
            "representative_taxa": representative_taxa,
        }
        for pattern_id in PATTERN_IDS:
            summary_row[f"Cluster{pattern_id}_mean_fraction_of_all"] = float(
                mean_total_fraction[pattern_id - 1]
            )
            summary_row[f"Cluster{pattern_id}_mean_within_cluster_fraction"] = float(
                mean_within_cluster[pattern_id - 1]
            )
            summary_row[f"Cluster{pattern_id}_stable_contributor_taxa_n"] = int(
                stable_counts.get(pattern_id, 0)
            )
        summary_rows.append(summary_row)

    contribution_long = pd.DataFrame(long_rows)
    classification = pd.DataFrame(summary_rows)
    all_fraction_sums = contribution_long.groupby("pathway_id", sort=False)[
        "mean_fraction_of_all_stratified_rpk"
    ].sum(min_count=1)
    finite_all_sums = all_fraction_sums[np.isfinite(all_fraction_sums)]
    if not np.allclose(finite_all_sums.to_numpy(dtype=float), 1.0, atol=1e-10):
        raise RuntimeError(
            "Mean all-stratified contribution fractions do not sum to one"
        )
    fixed_long = contribution_long.loc[
        contribution_long["component_scope"].eq("fixed_cluster_pattern")
    ]
    within_fraction_sums = fixed_long.groupby("pathway_id", sort=False)[
        "mean_within_cluster_fraction"
    ].sum(min_count=1)
    finite_within_sums = within_fraction_sums[np.isfinite(within_fraction_sums)]
    if not np.allclose(finite_within_sums.to_numpy(dtype=float), 1.0, atol=1e-10):
        raise RuntimeError("Mean within-cluster Cluster1-Cluster5 fractions do not sum to one")
    coverage_from_components = fixed_long.groupby("pathway_id", sort=False)[
        "mean_fraction_of_all_stratified_rpk"
    ].sum(min_count=1)
    classification_coverage = classification.set_index("pathway_id")[
        "cluster_coverage_mean"
    ].reindex(coverage_from_components.index)
    comparable_coverage = np.isfinite(coverage_from_components.to_numpy(dtype=float))
    comparable_coverage &= np.isfinite(classification_coverage.to_numpy(dtype=float))
    if not np.allclose(
        coverage_from_components.to_numpy(dtype=float)[comparable_coverage],
        classification_coverage.to_numpy(dtype=float)[comparable_coverage],
        atol=1e-10,
    ):
        raise RuntimeError(
            "Cluster coverage does not equal the summed Cluster1-Cluster5 fraction of all RPK"
        )
    all_coverage = classification["cluster_coverage_mean"].to_numpy(dtype=float)
    finite_coverage = all_coverage[np.isfinite(all_coverage)]
    coverage_summary = {
        "pathways_with_coverage": int(finite_coverage.size),
        "pathways_without_coverage": int(all_coverage.size - finite_coverage.size),
        "minimum": float(finite_coverage.min()) if finite_coverage.size else None,
        "median": float(np.median(finite_coverage)) if finite_coverage.size else None,
        "maximum": float(finite_coverage.max()) if finite_coverage.size else None,
    }
    audit = {
        "analysis_samples": n_samples,
        "selected_pathways": n_pathways,
        "fixed_cluster_taxa": int(len(universe)),
        "stratified_columns_read": int(len(mapping)),
        "source_null_cells_treated_as_zero_contribution": source_missing_values,
        "null_policy": (
            "Within existing microbe-level rows, null stratified RPK is treated "
            "as no contributed reads; missing samples are not imputed."
        ),
        "cluster_coverage_summary": coverage_summary,
        "mass_balance_checks": {
            "all_stratified_component_means_sum_to_one": True,
            "within_cluster_C1_C5_means_sum_to_one": True,
            "cluster_coverage_equals_summed_C1_C5_all_fraction": True,
            "absolute_tolerance": 1e-10,
        },
        "all_stratified_fraction_summary": (
            "unweighted mean of participant-level component fractions among "
            "participants with any stratified RPK for the pathway"
        ),
        "all_stratified_fraction_denominator": (
            "all taxon-stratified RPK, including taxa outside the fixed oral "
            "pattern universe"
        ),
        "within_cluster_fraction_summary": (
            "unweighted mean of participant-level Cluster1-Cluster5 fractions among "
            "participants with any fixed-cluster RPK for the pathway"
        ),
        "within_cluster_fraction_denominator": (
            "sum of Cluster1-Cluster5 fixed-cluster RPK only; the five reported means sum "
            "to one because they use the same participant set"
        ),
        "min_contribution_samples_for_representative_taxa": int(
            min_contribution_samples
        ),
        "dominant_fraction_threshold": float(dominant_fraction_threshold),
        "classification_counts": {
            str(key): int(value)
            for key, value in classification["dominant_pattern_label"]
            .value_counts()
            .items()
        },
        "classification_scope": (
            "predominance within clustered taxa only; cluster coverage is "
            "reported separately and no whole-pathway dominance is claimed"
        ),
        "inference_role": "descriptive decomposition; no additional hypothesis test",
    }
    return contribution_long, classification, pair_table, audit


def p_text(value: float) -> str:
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.3f}"


def _wrap_figure_text(
    fig: mpl.figure.Figure,
    value: object,
    width_mm: float,
    fontsize: float,
) -> str:
    """Wrap complete labels using rendered font widths, without ellipses."""
    compact = " ".join(str(value).split())
    if compact.casefold() in {"nan", "none", "<na>"}:
        return ""
    renderer = fig.canvas.get_renderer()
    font = FontProperties(family=mpl.rcParams["font.family"], size=fontsize)
    max_pixels = width_mm / 25.4 * fig.dpi

    def fits(text: str) -> bool:
        return renderer.get_text_width_height_descent(text, font, False)[0] <= max_pixels

    lines: list[str] = []
    current = ""
    for word in compact.split():
        candidate = f"{current} {word}".strip()
        if fits(candidate):
            current = candidate
            continue
        if current:
            lines.append(current)
        current = ""
        for character in word:
            if current and not fits(current + character):
                lines.append(current)
                current = ""
            current += character
    if current:
        lines.append(current)
    return "\n".join(lines)


def _table_axes(
    fig: mpl.figure.Figure,
    widths_mm: list[float],
    row_heights_mm: np.ndarray,
    bottom_mm: float,
) -> tuple[list[mpl.axes.Axes], np.ndarray, np.ndarray]:
    """Use physical millimetres for label-aware row spacing and exact export size."""
    edges = np.concatenate(([0.0], np.cumsum(row_heights_mm)))
    centers = (edges[:-1] + edges[1:]) / 2.0
    plot_height_mm = float(edges[-1])
    height_mm = plot_height_mm + bottom_mm + 2.0
    fig.set_size_inches(FIGURE_WIDTH_MM / 25.4, height_mm / 25.4)
    if not np.isclose(sum(widths_mm) + 3.0 * (len(widths_mm) - 1) + 4.0, FIGURE_WIDTH_MM):
        raise ValueError("Physical panel widths do not fill the 100 mm figure contract")
    axes: list[mpl.axes.Axes] = []
    left_mm = 2.0
    for width_mm in widths_mm:
        axis = fig.add_axes(
            [left_mm / FIGURE_WIDTH_MM, bottom_mm / height_mm,
             width_mm / FIGURE_WIDTH_MM, plot_height_mm / height_mm]
        )
        axis.set_ylim(plot_height_mm, 0.0)
        axis.set_yticks([])
        axis.tick_params(axis="both", which="both", labelsize=5.7, length=2.0,
                         width=0.55, color=MUTED, pad=2.5)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axes.append(axis)
        left_mm += width_mm + 3.0
    return axes, centers, edges


def _export_table_figure(
    fig: mpl.figure.Figure,
    axes: dict[str, mpl.axes.Axes],
    stem: Path,
    preview_dpi: int,
    extra_outputs: dict[Path, int | None] | None = None,
) -> tuple[dict[str, str], dict[str, object]]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    bounds = fig.get_tightbbox(fig.canvas.get_renderer())
    tolerance_inches = 0.5 / 72.0
    if (bounds.x0 < -tolerance_inches or bounds.y0 < -tolerance_inches
            or bounds.x1 > fig.get_figwidth() + tolerance_inches
            or bounds.y1 > fig.get_figheight() + tolerance_inches):
        raise RuntimeError("Figure content exceeds its fixed physical canvas")
    alignment_path = stem.with_suffix(".alignment.json")
    alignment = require_matplotlib_panel_alignment(
        fig, axes=axes, json_out=alignment_path, tolerance_pt=1.5,
    )
    alignment.update({
        "width_mm": float(fig.get_figwidth() * 25.4),
        "height_mm": float(fig.get_figheight() * 25.4),
        "fixed_canvas_export": True,
        "content_within_canvas": True,
        "subtitles_shown": False,
    })
    write_json(alignment_path, alignment)
    files: dict[str, str] = {"alignment_json": str(alignment_path.resolve())}
    # Tight cropping changes the physical page width; keep the 100 mm canvas.
    with mpl.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0}):
        for extension, key in ((".pdf", "pdf"), (".svg", "svg"), (".png", "png_preview")):
            path = stem.with_suffix(extension)
            fig.savefig(path, dpi=preview_dpi, bbox_inches=None, pad_inches=0,
                        facecolor="white")
            files[key] = str(path.resolve())
        for path, dpi in (extra_outputs or {}).items():
            fig.savefig(path, dpi=dpi or preview_dpi, bbox_inches=None, pad_inches=0,
                        facecolor="white")
            files[path.name] = str(path.resolve())
    plt.close(fig)
    return files, alignment


# --------------------------------------------------------------------------- #
# Fig. 4d main panel (88 × 62 mm): single-line names | age effect | C1–C5 share
# --------------------------------------------------------------------------- #
MAIN_WIDTH_MM, MAIN_HEIGHT_MM = 88.0, 62.0
MAIN_EDGE_MM = 1.0
MAIN_GAP_MM = 2.0
MAIN_FOREST_SHARE = 0.4375        # forest : bars = 35 : 45
MAIN_TEXT = "#222222"
MAIN_POINT = "#333333"
MAIN_ZERO = "#999999"
MAIN_FS = {"name": 6.0, "tick": 6.0, "axis_title": 6.5, "legend": 6.0, "na": 6.0}
MAIN_PNG_DPI = 600
# Cluster palette shared with Fig. 4b (independent of the omics palette).
MAIN_PATTERN_COLORS = {
    1: "#2E3A59",
    2: "#7F93AE",
    3: "#9AAE8C",
    4: "#C49A6C",
    5: "#7E5A78",
}
# Short single-line names (British spelling). Everything else: sentence case.
MAIN_NAME_OVERRIDES = {
    "superpathway of coenzyme A biosynthesis I (bacteria)": "CoA biosynthesis superpathway I",
    "tetrapyrrole biosynthesis I (from glutamate)": "Tetrapyrrole biosynthesis I",
    "methylerythritol phosphate pathway I": "MEP pathway I",
    "partial TCA cycle (obligate autotrophs)": "Partial TCA cycle",
    "NAD de novo biosynthesis I (from aspartate)": "NAD de novo biosynthesis I",
    "stearate biosynthesis II (bacteria and plants)": "Stearate biosynthesis II",
    "heme b biosynthesis I (aerobic)": "Haem b biosynthesis I",
    "L-arginine biosynthesis I (via L-ornithine)": "L-arginine biosynthesis I",
    "L-arginine biosynthesis IV (archaebacteria)": "L-arginine biosynthesis IV",
}


def main_display_name(name: object) -> str:
    name = " ".join(str(name).split())
    if name in MAIN_NAME_OVERRIDES:
        return MAIN_NAME_OVERRIDES[name]
    words = []
    for i, word in enumerate(name.split()):
        keep = (any(c.isupper() for c in word[1:]) or re.match(r"^\(?[A-Z]-", word)
                or re.fullmatch(r"[IVX]+", word))
        if keep:
            words.append(word)
        elif i == 0:
            words.append(word[:1].upper() + word[1:])
        else:
            words.append(word.lower())
    return " ".join(words).replace("heme", "haem").replace("Heme", "Haem")


def _main_font_family() -> str:
    """Arial from ~/.fonts or the system; fall back to Helvetica / Liberation Sans."""
    from matplotlib import font_manager
    import glob
    import os
    for path in glob.glob(os.path.expanduser("~/.fonts/*.ttf")):
        font_manager.fontManager.addfont(path)
    for family in ("Arial", "Helvetica", "Liberation Sans"):
        try:
            font_manager.findfont(FontProperties(family=family), fallback_to_default=False)
            return family
        except ValueError:
            continue
    return "DejaVu Sans"


def _forest_ticks(lo: float, hi: float) -> list[float]:
    for step in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0):
        ticks = np.arange(math.ceil(lo / step) * step, hi + 1e-9, step)
        if 2 <= len(ticks) <= 3:
            return [round(float(t), 3) + 0.0 for t in ticks]
    return [round(lo, 2), round(hi, 2)]


def draw_main_figure(
    merged: pd.DataFrame,
    fractions: np.ndarray,
    present: np.ndarray,
    absent: np.ndarray,
    beta: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    stem: Path,
    preview_dpi: int,
) -> tuple[dict[str, str], dict[str, object]]:
    family = _main_font_family()
    rc = {
        "font.family": family, "font.size": MAIN_FS["tick"], "text.color": MAIN_TEXT,
        "axes.linewidth": 0.5, "axes.edgecolor": MAIN_TEXT, "axes.labelcolor": MAIN_TEXT,
        "xtick.color": MAIN_TEXT, "xtick.labelsize": MAIN_FS["tick"],
        "xtick.major.size": 2.0, "xtick.major.width": 0.5, "xtick.direction": "out",
        "xtick.major.pad": 1.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "pdf.fonttype": 42,
    }
    with mpl.rc_context(rc):
        fig = plt.figure(figsize=(MAIN_WIDTH_MM / 25.4, MAIN_HEIGHT_MM / 25.4),
                         facecolor="white")
        renderer = fig.canvas.get_renderer()

        def size_mm(text: str, fontsize: float) -> tuple[float, float]:
            artist = fig.text(0, 0, text, fontsize=fontsize)
            box = artist.get_window_extent(renderer)
            artist.remove()
            return box.width / fig.dpi * 25.4, box.height / fig.dpi * 25.4

        names = [main_display_name(name) for name in merged["pathway_name"]]
        label_w = max(size_mm(name, MAIN_FS["name"])[0] for name in names) + 0.3
        _, tick_h = size_mm("0.1", MAIN_FS["tick"])
        _, title_h = size_mm("Age effect\n(s.d. per 10 years)", MAIN_FS["axis_title"])
        top_mm = MAIN_EDGE_MM + 2.5 + 1.5                 # legend row + gap
        bottom_mm = MAIN_EDGE_MM + title_h + 1.0 + tick_h + 2.0 + 1.0
        plot_h = MAIN_HEIGHT_MM - top_mm - bottom_mm
        # The "100" tick label is centred on the right edge of the bars.
        right_mm = max(MAIN_EDGE_MM, size_mm("100", MAIN_FS["tick"])[0] / 2 + 0.4)
        plot_w = MAIN_WIDTH_MM - MAIN_EDGE_MM - right_mm - label_w - 2 * MAIN_GAP_MM
        forest_w = plot_w * MAIN_FOREST_SHARE
        bars_w = plot_w - forest_w
        grid = fig.add_gridspec(
            1, 5, width_ratios=[label_w, MAIN_GAP_MM, forest_w, MAIN_GAP_MM, bars_w],
            wspace=0, left=MAIN_EDGE_MM / MAIN_WIDTH_MM, right=1 - right_mm / MAIN_WIDTH_MM,
            bottom=bottom_mm / MAIN_HEIGHT_MM, top=1 - top_mm / MAIN_HEIGHT_MM,
        )
        ax_labels = fig.add_subplot(grid[0, 0])
        ax_effect = fig.add_subplot(grid[0, 2])
        ax_sources = fig.add_subplot(grid[0, 4])
        n_rows = len(names)
        row_h = plot_h / n_rows
        centers = (np.arange(n_rows) + 0.5) * row_h
        for axis in (ax_labels, ax_effect, ax_sources):
            axis.set_ylim(plot_h, 0.0)
            axis.set_yticks([])
            axis.spines["left"].set_visible(False)

        ax_labels.axis("off")
        name_artists = [
            ax_labels.text(1.0, y, name, transform=ax_labels.get_yaxis_transform(),
                           ha="right", va="center", fontsize=MAIN_FS["name"])
            for name, y in zip(names, centers)
        ]

        lo = min(0.0, float(np.min(low))) - 0.02
        hi = float(np.max(high)) + 0.02
        ax_effect.axvline(0, color=MAIN_ZERO, linewidth=0.5, linestyle=(0, (2.5, 1.5)), zorder=0)
        ax_effect.hlines(centers, low, high, color=MAIN_POINT, linewidth=0.6,
                         capstyle="butt", zorder=1)
        ax_effect.plot(beta, centers, "o", ms=3, color=MAIN_POINT, mew=0, zorder=2)
        ax_effect.set_xlim(lo, hi)
        ax_effect.set_xticks(_forest_ticks(lo, hi))
        ax_effect.xaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda v, _: "0" if abs(v) < 1e-12 else f"{v:g}"))
        ax_effect.set_xlabel("Age effect\n(s.d. per 10 years)",
                             fontsize=MAIN_FS["axis_title"], labelpad=1.5)

        left = np.zeros(n_rows)
        for index, pattern_id in enumerate(PATTERN_IDS):
            width = np.where(present, fractions[:, index] * 100.0, 0.0)
            ax_sources.barh(centers, width, left=left, height=0.7 * row_h,
                            color=MAIN_PATTERN_COLORS[pattern_id], edgecolor="white",
                            linewidth=0.3)
            left += width
        for y in centers[absent]:
            ax_sources.text(50, y, "NA", ha="center", va="center", fontsize=MAIN_FS["na"],
                            color=MUTED)
        ax_sources.set_xlim(0, 100)
        ax_sources.set_xticks([0, 50, 100])
        ax_sources.set_xlabel("Share of Cluster 1–Cluster 5\nabundance (%)",
                              fontsize=MAIN_FS["axis_title"], labelpad=1.5)

        # Legend: one row directly above the bars, 2.5 × 2.5 mm swatches, no frame.
        swatch = 2.5 / 25.4 * 72 / MAIN_FS["legend"]
        bars_box = ax_sources.get_position()
        fig.legend(
            handles=[Patch(facecolor=MAIN_PATTERN_COLORS[i], edgecolor="none", label=f"Cluster{i}")
                     for i in PATTERN_IDS],
            loc="lower right", ncol=len(PATTERN_IDS), frameon=False,
            fontsize=MAIN_FS["legend"], handlelength=swatch, handleheight=swatch,
            handletextpad=0.35, columnspacing=0.8, borderaxespad=0, borderpad=0,
            bbox_to_anchor=(bars_box.x1, bars_box.y1 + 1.2 / MAIN_HEIGHT_MM),
        )

        # Self-check: single-line names that fit their column; bars sum to 100 %.
        fig.canvas.draw()
        for artist in name_artists:
            width = artist.get_window_extent(renderer).width / fig.dpi * 25.4
            if "\n" in artist.get_text() or width > label_w + 0.05:
                raise RuntimeError(f"Pathway name is not single-line: {artist.get_text()!r}")
        if present.any() and not np.allclose(fractions[present].sum(axis=1), 1.0, atol=1e-8):
            raise RuntimeError("Stacked C1–C5 bars do not sum to 100%")

        files, alignment = _export_table_figure(
            fig, {"pathway_names": ax_labels, "age_effect": ax_effect,
                  "source_composition": ax_sources}, stem, preview_dpi,
            extra_outputs={stem.parent / "fig4d.pdf": None,
                           stem.parent / "fig4d.png": MAIN_PNG_DPI},
        )
    alignment["font_family"] = family
    alignment["display_names"] = dict(zip(merged["pathway_id"].astype(str), names))
    return files, alignment


def draw_combined_figure(
    selected: pd.DataFrame,
    age_heatmap: pd.DataFrame,
    age_bin_source: pd.DataFrame,
    contribution_long: pd.DataFrame,
    classification: pd.DataFrame,
    universe_audit: dict[str, object],
    age_manifest: dict[str, object],
    microbe_analysis_n: int,
    output_dir: Path,
    preview_dpi: int,
) -> tuple[dict[str, str], dict[str, object]]:
    """Export a 100 mm main figure and a separate supplementary figure."""
    merged = selected.merge(
        classification, on=["pathway_id", "pathway_name"], how="left",
        validate="one_to_one",
    ).sort_values(
        ["beta_pathway_sd_per_10y", "pathway_id"], ascending=[False, True],
        kind="mergesort",
    )
    if merged.empty:
        raise AnalysisSkip("No pathway remains for the main and supplementary figures")
    if merged["n"].nunique() != 1:
        raise ValueError("Displayed age-association pathways have inconsistent n")
    pathway_order = merged["pathway_id"].astype(str).tolist()
    contribution_heat = (
        contribution_long.loc[
            contribution_long["component_scope"].eq("fixed_cluster_pattern")
        ].pivot(index="pathway_id", columns="component", values="mean_within_cluster_fraction")
        .reindex(index=pathway_order, columns=[f"Cluster{i}" for i in PATTERN_IDS])
        .astype(float)
    )
    fractions = contribution_heat.to_numpy(dtype=float)
    present = np.isfinite(fractions).all(axis=1)
    absent = np.isnan(fractions).all(axis=1)
    if not np.all(present | absent):
        raise ValueError("Each source-composition row must be complete or entirely missing")
    if (np.any(fractions[present] < 0) or np.any(fractions[present] > 1)
            or not np.allclose(fractions[present].sum(axis=1), 1.0, atol=1e-8, rtol=0)):
        raise ValueError("Within-cluster source fractions must be in [0, 1] and sum to one")
    beta = merged["beta_pathway_sd_per_10y"].to_numpy(dtype=float)
    low = merged["ci95_low_pathway_sd_per_10y"].to_numpy(dtype=float)
    high = merged["ci95_high_pathway_sd_per_10y"].to_numpy(dtype=float)
    if (not np.isfinite(np.concatenate([beta, low, high])).all()
            or np.any(low > beta) or np.any(beta > high)):
        raise ValueError("Every displayed age effect needs a finite, enclosing confidence interval")

    # Main (Fig. 4d): single-line names, continuous-age effect, C1–C5 source share.
    stem = output_dir / "oral_metacyc_pathway_age_cluster_attribution"
    main_files, main_alignment = draw_main_figure(
        merged, fractions, present, absent, beta, low, high, stem, preview_dpi,
    )

    # Supplement: provenance and descriptive support, in the identical row order.
    supplement = plt.figure(figsize=(FIGURE_WIDTH_MM / 25.4, 2.0), facecolor="white")
    supp_names = [
        _wrap_figure_text(supplement, row.pathway_name, 41.5, 5.8)
        or _wrap_figure_text(supplement, row.pathway_id, 41.5, 5.8)
        for row in merged.itertuples(index=False)
    ]
    details = [
        _wrap_figure_text(supplement, f"{row.pathway_id} · {row.dominant_pattern_label}", 41.5, 5.2)
        + ("\n" + taxa if (taxa := _wrap_figure_text(
            supplement, row.representative_taxa, 41.5, 5.2)) else "")
        for row in merged.itertuples(index=False)
    ]
    supp_line_spacing = 1.22
    supp_row_heights = np.asarray([
        max(6.2, ((name.count("\n") + 1) * 5.8 * supp_line_spacing
                  + (detail.count("\n") + 1) * 5.2 * supp_line_spacing) / 72 * 25.4 + 1.3)
        for name, detail in zip(supp_names, details)
    ])
    display_mode = str(age_manifest.get("figure_selection", {}).get(
        "mode", age_manifest.get("arguments", {}).get("display_selection", "bonferroni")))
    correction_column = "q_bh_global" if display_mode == "bh" else "p_bonferroni_global"
    correction_label = "BH\nq" if display_mode == "bh" else "Bonferroni\nP"
    pattern_counts = universe_audit["pattern_taxa_counts"]
    notes = [
        f"Age model: n={int(merged['n'].iloc[0]):,}; attribution: n={microbe_analysis_n:,} matched participants.",
        "Fixed taxa: " + "; ".join(f"Cluster{i}, n={pattern_counts[str(i)]}" for i in PATTERN_IDS) + ".",
        "Coverage = fixed Cluster1–Cluster5 RPK / all stratified RPK. Main-figure source bars use Cluster1–Cluster5 RPK only; participant fractions are averaged equally.",
        "Age bins and sources are descriptive. Age effects use continuous age with sex adjustment; intervals are 95% CIs. Taxa listed above are stable contributors.",
        "BH q values are corrected across all retained pathways." if display_mode == "bh"
        else "Bonferroni P values are corrected across all retained pathways.",
    ]
    if display_mode != "bonferroni":
        notes.append(f"Exploratory display selection: {display_mode}.")
    footer = "\n".join(_wrap_figure_text(supplement, note, 96.0, 5.2) for note in notes)
    footer_height_mm = (footer.count("\n") + 1) * 5.2 * 1.25 / 72 * 25.4
    bottom_mm = footer_height_mm + 22.0
    supp_axes, supp_centers, edges = _table_axes(
        supplement, [42.0, 26.0, 9.0, 10.0], supp_row_heights, bottom_mm,
    )
    ax_detail, ax_age, ax_coverage, ax_q = supp_axes
    ax_detail.set_xlim(0, 1)
    ax_detail.axis("off")
    for index, (name, detail) in enumerate(zip(supp_names, details)):
        top = edges[index] + 0.5
        ax_detail.text(0, top, name, ha="left", va="top", fontsize=5.8,
                       linespacing=supp_line_spacing, color=TEXT)
        detail_top = top + (name.count("\n") + 1) * 5.8 * supp_line_spacing / 72 * 25.4 + 0.2
        ax_detail.text(0, detail_top, detail, ha="left", va="top", fontsize=5.2,
                       linespacing=supp_line_spacing, color=MUTED)
    age_heat = age_heatmap.loc[pathway_order]
    age_cmap = LinearSegmentedColormap.from_list("age_pattern", [AGE_LOW, AGE_MID, AGE_HIGH])
    age_cmap.set_bad(MISSING)
    age_image = ax_age.pcolormesh(
        np.arange(age_heat.shape[1] + 1) - 0.5, edges,
        np.ma.masked_invalid(age_heat.to_numpy(dtype=float)), shading="flat",
        cmap=age_cmap, norm=TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0),
        edgecolors="white", linewidth=0.4,
    )
    ax_age.set_xticks(np.arange(age_heat.shape[1]))
    ax_age.set_xticklabels(list(age_heat.columns), fontsize=5.4, rotation=55,
                           ha="right", rotation_mode="anchor")
    ax_age.tick_params(axis="x", length=0)
    ax_age.set_xlabel("Age (years)", fontsize=5.5, labelpad=2.0)
    coverage = merged["cluster_coverage_mean"].to_numpy(dtype=float)
    coverage_cmap = LinearSegmentedColormap.from_list("coverage", ["#F0F7F5", "#3F8B87"])
    coverage_cmap.set_bad(MISSING)
    ax_coverage.pcolormesh(
        [-0.5, 0.5], edges, np.ma.masked_invalid(coverage[:, None]), shading="flat",
        cmap=coverage_cmap, norm=Normalize(vmin=0, vmax=1), edgecolors="white", linewidth=0.4,
    )
    ax_coverage.set_xticks([])
    ax_coverage.set_xlabel("Coverage\n(%)", fontsize=5.2, labelpad=4.0)
    for y, value in zip(supp_centers, coverage):
        ax_coverage.text(0, y, f"{100 * value:.0f}" if np.isfinite(value) else "NA",
                         ha="center", va="center", fontsize=5.5,
                         color="white" if np.isfinite(value) and value >= 0.65 else TEXT)
    ax_q.set_xlim(0, 1)
    ax_q.set_xticks([])
    ax_q.set_xlabel(correction_label, fontsize=5.2, labelpad=4.0)
    for y, row in zip(supp_centers, merged.itertuples(index=False)):
        ax_q.text(0.5, y, p_text(float(getattr(row, correction_column))),
                  ha="center", va="center", fontsize=5.2, color=TEXT)
    height_mm = supplement.get_figheight() * 25.4
    age_position = ax_age.get_position()
    colorbar_axis = supplement.add_axes([
        age_position.x0, (footer_height_mm + 9.0) / height_mm,
        age_position.width, 1.6 / height_mm,
    ])
    colorbar = supplement.colorbar(age_image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_ticks([-2, 0, 2])
    colorbar.set_ticklabels(["−2", "0", "+2"])
    colorbar.ax.tick_params(labelsize=5.2, length=1.5, width=0.45, pad=1.5)
    colorbar.outline.set_visible(False)
    colorbar.set_label("Age-bin row Z-score", fontsize=5.2, labelpad=1.5)
    supplement.text(2.0 / FIGURE_WIDTH_MM, 2.0 / height_mm, footer,
                    ha="left", va="bottom", fontsize=5.2, linespacing=1.25, color=MUTED)
    supplementary_files, supplementary_alignment = _export_table_figure(
        supplement, {"pathway_details": ax_detail, "age_heatmap": ax_age,
                     "cluster_coverage": ax_coverage, "multiplicity": ax_q},
        output_dir / f"{stem.name}_supplementary", preview_dpi,
    )
    caption_path = output_dir / f"{stem.name}.captions.md"
    caption_path.write_text(
        "# Main figure\n\n"
        "Rows show the selected pathways (shortened names; full MetaCyc names and IDs are "
        "given in the supplementary table), ordered by decreasing age effect. "
        + ("All displayed pathways satisfy Bonferroni P < 0.05. "
           if bool((merged["p_bonferroni_global"] < 0.05).all()) else "")
        + "Dots and lines are the upstream sex-adjusted continuous-age effect and 95% CI "
        "in pathway s.d. per 10 years. Source bars show mean participant-level C1–C5 "
        "fractions conditional on any fixed-cluster contribution; they sum to 100%. "
        "NA denotes no defined within-cluster composition. These fractions do not "
        "decompose the age regression coefficient and do not imply whole-community dominance. "
        "The supplementary figure reports cluster coverage and other supporting information.\n\n"
        "# Supplementary figure\n\n"
        "The same pathways and ordering show original identifiers, within-cluster class, "
        "stable representative taxa, sex-adjusted age-bin abundance row Z-scores, "
        "fixed-cluster coverage of all stratified RPK, and the upstream multiplicity-adjusted "
        "P/q value. Age bins and source attribution are descriptive.\n\n"
        + "\n\n".join(notes) + "\n",
        encoding="utf-8",
    )
    files = dict(main_files)
    files.update({f"supplementary_{key}": value for key, value in supplementary_files.items()})
    files["captions_markdown"] = str(caption_path.resolve())
    return files, {
        "status": "PASS", "width_mm": FIGURE_WIDTH_MM,
        "pathway_order": pathway_order,
        "main": main_alignment, "supplementary": supplementary_alignment,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--age-association-dir",
        type=Path,
        default=script_dir / "outputs" / "oral_metacyc_age_association",
        help="Completed output directory from 09 age-association analysis.",
    )
    parser.add_argument(
        "--taxon-pattern-universe",
        type=Path,
        default=(
            script_dir
            / "outputs"
            / "oral_metacyc_pathway_analysis"
            / "oral_taxon_pattern_universe.csv"
        ),
        help=(
            "Complete frozen taxon-to-Cluster1-Cluster5 table produced by the revised 08; "
            "its size is read dynamically and clustering is not rerun."
        ),
    )
    parser.add_argument(
        "--microbe-pathway-table",
        type=Path,
        default=Path(
            "/home/ec2-user/studies/hpp_datasets/oral_microbiome/aggregated/"
            "humann_aggregated/pathway_abundance_microbe_level.arrow"
        ),
        help="HUMAnN species-stratified pathway contribution Arrow.",
    )
    parser.add_argument(
        "--sample-key-columns",
        default="",
        help=(
            "Optional comma-separated sample key. Empty reuses the exact key "
            "recorded by 09; an explicit value must match it."
        ),
    )
    parser.add_argument("--min-contribution-samples", type=int, default=500)
    parser.add_argument(
        "--dominant-contribution-threshold",
        type=float,
        default=0.50,
        help=(
            "Minimum mean within-cluster fraction required to label the covered "
            "part of a pathway Cluster1-Cluster5; otherwise label Mixed."
        ),
    )
    parser.add_argument("--representative-taxa", type=int, default=2)
    parser.add_argument("--batch-columns", type=int, default=128)
    parser.add_argument("--preview-dpi", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.output_dir is None:
        args.output_dir = (
            script_dir / "outputs" / "oral_metacyc_age_cluster_attribution"
        )
    if args.figure_dir is None:
        args.figure_dir = (
            script_dir / "figures" / "oral_metacyc_age_cluster_attribution"
        )
    args.run_id = args.run_id or uuid.uuid4().hex
    for name in (
        "min_contribution_samples",
        "representative_taxa",
        "batch_columns",
        "preview_dpi",
    ):
        if int(getattr(args, name)) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.preview_dpi < 150:
        parser.error("--preview-dpi must be at least 150")
    if not 0 < args.dominant_contribution_threshold <= 1:
        parser.error("--dominant-contribution-threshold must lie in (0, 1]")
    return args


def initial_manifest(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": 2,
        "analysis_version": ANALYSIS_VERSION,
        "run_id": args.run_id,
        "status": "running",
        "started_utc": utc_now(),
        "analysis_question": (
            "What fraction of each participant-level age-associated community "
            "MetaCyc pathway is covered by the complete frozen oral-taxon cluster "
            "universe, and how is that covered fraction composed across Cluster1-Cluster5?"
        ),
        "analysis_units": {
            "age_association": "independent participant",
            "pattern_assignment": "oral microbial taxon",
            "cluster_coverage": (
                "participant-level fixed-cluster RPK fraction of all stratified RPK"
            ),
            "within_cluster_composition": (
                "participant-level Cluster1-Cluster5 fraction of fixed-cluster RPK"
            ),
        },
        "figure_contract": {
            "core_claim": (
                "Age-associated community pathways have quantifiable coverage by "
                "the complete frozen oral-taxon cluster universe, and their "
                "covered contribution can be decomposed descriptively across Cluster1-Cluster5."
            ),
            "archetype": "quantitative grid",
            "width_mm": FIGURE_WIDTH_MM,
            "main_regions": [
                "complete pathway names",
                "continuous-age effect and existing 95% confidence interval",
                "mean participant-level composition within fixed Cluster1-Cluster5 taxa",
            ],
            "supplementary_regions": [
                "pathway IDs, within-cluster classes and stable representative taxa",
                "descriptive age-bin abundance",
                "fixed-cluster coverage of all stratified RPK",
                "upstream multiplicity-adjusted P/q values",
                "cohort sizes, fixed taxon counts and denominator notes",
            ],
            "region_subtitles": False,
            "full_pathway_names": True,
            "physical_export": "100 mm canvas; no tight cropping; compact label-aware row heights",
            "inference_boundary": (
                "Cluster1-Cluster5 are feature clusters, not participant groups; attribution "
                "is descriptive, within-cluster predominance is not whole-pathway "
                "dominance, and neither is an independent validation."
            ),
        },
        "arguments": {key: value for key, value in vars(args).items()},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "matplotlib": mpl.__version__,
        },
        "outputs": {},
    }


def run_analysis(args: argparse.Namespace, manifest: dict[str, object]) -> None:
    required_inputs = {
        "age_association_dir": args.age_association_dir,
        "taxon_pattern_universe": args.taxon_pattern_universe,
        "microbe_pathway_table": args.microbe_pathway_table,
    }
    missing = [str(path) for path in required_inputs.values() if not path.exists()]
    if missing:
        raise AnalysisSkip("Required input is missing: " + "; ".join(missing))
    manifest["inputs"] = {
        "taxon_pattern_universe": file_record(args.taxon_pattern_universe),
        "microbe_pathway_table": file_record(args.microbe_pathway_table),
    }

    (
        selected,
        age_heatmap,
        age_bin_source,
        sample_audit,
        age_manifest,
        age_paths,
    ) = load_age_analysis(args.age_association_dir)
    manifest["upstream_age_analysis"] = {
        "run_id": age_manifest.get("run_id"),
        "analysis_version": age_manifest.get("analysis_version"),
        "status": age_manifest.get("status"),
        "display_selection": age_manifest.get("figure_selection", {}).get("mode"),
        "files": {key: file_record(path) for key, path in age_paths.items()},
    }
    selected["pathway_id"] = selected["pathway_id"].astype(str)

    universe, universe_audit = load_pattern_universe(args.taxon_pattern_universe)
    manifest["fixed_taxon_pattern_universe"] = universe_audit
    sample_key_columns = resolve_sample_key_columns(
        args.sample_key_columns,
        age_manifest,
    )
    schema = read_arrow_schema(args.microbe_pathway_table)
    microbe_rows, alignment_table, alignment_audit = align_microbe_samples(
        args.microbe_pathway_table,
        schema,
        sample_key_columns,
        sample_audit,
    )
    alignment_path = args.output_dir / "sample_alignment_audit.csv"
    alignment_table.to_csv(alignment_path, index=False)
    manifest["sample_alignment"] = alignment_audit
    manifest["outputs"]["sample_alignment_audit"] = str(alignment_path.resolve())

    mapping, mapping_audit = parse_microbe_columns(
        schema,
        selected["pathway_id"].tolist(),
        universe,
    )
    mapping_path = args.output_dir / "microbe_column_pattern_mapping_audit.csv"
    mapping.to_csv(mapping_path, index=False)
    manifest["microbe_column_mapping"] = mapping_audit
    manifest["outputs"]["microbe_column_pattern_mapping_audit"] = str(
        mapping_path.resolve()
    )

    (
        contribution_long,
        classification,
        pair_table,
        contribution_audit,
    ) = summarize_pattern_contributions(
        args.microbe_pathway_table,
        microbe_rows,
        mapping,
        selected,
        universe,
        args.batch_columns,
        args.min_contribution_samples,
        args.dominant_contribution_threshold,
        args.representative_taxa,
    )
    contribution_path = args.output_dir / "pathway_pattern_contributions_long.csv"
    classification_path = args.output_dir / "pathway_pattern_classification.csv"
    pair_path = args.output_dir / "pathway_taxon_contributions.csv"
    outside_breakdown_path = (
        args.output_dir / "pathway_outside_cluster_contribution_breakdown.csv"
    )
    coverage_path = args.output_dir / "pathway_cluster_coverage.csv"
    within_cluster_path = args.output_dir / "pathway_within_cluster_composition.csv"
    contribution_long.to_csv(contribution_path, index=False)
    classification.to_csv(classification_path, index=False)
    pair_table.to_csv(pair_path, index=False)
    contribution_long.loc[
        ~contribution_long["component_scope"].eq("fixed_cluster_pattern")
    ].to_csv(outside_breakdown_path, index=False)
    classification[
        [
            "pathway_id",
            "pathway_name",
            "age_analysis_samples",
            "microbe_attribution_samples",
            "samples_with_any_stratified_contribution",
            "samples_with_any_fixed_cluster_contribution",
            "cluster_coverage_mean",
            "cluster_coverage_median",
            "cluster_coverage_q25",
            "cluster_coverage_q75",
            "cluster_coverage_pooled",
            "outside_cluster_species_fraction_mean",
            "unresolved_taxon_fraction_mean",
        ]
    ].to_csv(coverage_path, index=False)
    contribution_long.loc[
        contribution_long["component_scope"].eq("fixed_cluster_pattern")
    ].to_csv(within_cluster_path, index=False)
    manifest["pattern_attribution"] = contribution_audit
    manifest["outputs"].update(
        {
            "pathway_pattern_contributions_long": str(contribution_path.resolve()),
            "pathway_pattern_classification": str(classification_path.resolve()),
            "pathway_taxon_contributions": str(pair_path.resolve()),
            "pathway_outside_cluster_contribution_breakdown": str(
                outside_breakdown_path.resolve()
            ),
            "pathway_cluster_coverage": str(coverage_path.resolve()),
            "pathway_within_cluster_composition": str(within_cluster_path.resolve()),
        }
    )

    figure_source = selected.merge(
        classification,
        on=["pathway_id", "pathway_name"],
        how="left",
        validate="one_to_one",
    )
    figure_source_path = args.output_dir / "figure_pathway_source_data.csv"
    figure_source.to_csv(figure_source_path, index=False)
    manifest["outputs"]["figure_pathway_source_data"] = str(
        figure_source_path.resolve()
    )

    figure_files, figure_alignment = draw_combined_figure(
        selected,
        age_heatmap,
        age_bin_source,
        contribution_long,
        classification,
        universe_audit,
        age_manifest,
        int(len(microbe_rows)),
        args.figure_dir,
        args.preview_dpi,
    )
    manifest["figure_alignment"] = figure_alignment
    manifest["outputs"]["figure_files"] = figure_files
    manifest["status"] = "completed"
    manifest["finished_utc"] = utc_now()


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        args.output_dir / "oral_metacyc_pathway_age_cluster_attribution_manifest.json"
    )
    manifest = initial_manifest(args)
    write_json(manifest_path, manifest)
    try:
        run_analysis(args, manifest)
    except AnalysisSkip as exc:
        manifest["status"] = "skipped"
        manifest["skip_reason"] = str(exc)
        manifest["finished_utc"] = utc_now()
        write_json(manifest_path, manifest)
        print(f"Run ID: {args.run_id}")
        print("Status: skipped")
        print(f"Skip reason: {exc}")
        print(f"Manifest: {manifest_path.resolve()}")
        return 0
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        manifest["finished_utc"] = utc_now()
        write_json(manifest_path, manifest)
        raise
    write_json(manifest_path, manifest)
    print(f"Run ID: {args.run_id}")
    print("Status: completed")
    print(
        "Age pathways displayed: "
        f"{manifest['pattern_attribution']['selected_pathways']:,}"
    )
    print(
        "Microbe attribution samples: "
        f"{manifest['pattern_attribution']['analysis_samples']:,}"
    )
    print(
        "Classification counts: "
        f"{manifest['pattern_attribution']['classification_counts']}"
    )
    print(f"Main figure PDF: {manifest['outputs']['figure_files']['pdf']}")
    print(f"Supplementary figure PDF: {manifest['outputs']['figure_files']['supplementary_pdf']}")
    print(f"Manifest: {manifest_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
