#!/usr/bin/env python3
"""Descriptive MetaCyc landscape of oral taxa in five precomputed patterns.

The primary result is a quantitative, descriptive five-pattern landscape of
original HUMAnN/MetaCyc pathways. Pathways enter the main figure through
prespecified detection and taxon-support criteria, never through a P-value.
For every displayed pathway, species-stratified HUMAnN RPK is averaged across
the complete-case cohort, summed within each fixed oral-taxon pattern, divided
by every taxon in that frozen pattern, log1p transformed and row
standardized across P1-P5 for visualization. The untransformed values and
support counts are retained in Source Data.

Taxon-set pathway-carriage enrichment is a separate, exploratory supplementary
analysis. Each pattern is compared with the other four using a one-sided
Fisher exact test. Benjamini-Hochberg FDR is reported across the complete
eligible pathway-by-pattern family; Bonferroni is retained as a strict
sensitivity analysis. Neither result controls entry into the primary figure.
The script does not group pathways with a project-specific theme dictionary
and does not infer host pathway activation.

Three source matrices have deliberately separate roles and are aligned by an
explicit sample key before any prevalence or abundance summary is calculated:

1. the un-imputed MetaPhlAn species-abundance Parquet defines the reference
   cohort and audits current-cohort prevalence of the already clustered taxa;
   this audit never removes a member from the frozen P1-P5 universe;
2. ``pathway_abundance_pathway_level.arrow`` supplies community-level HUMAnN
   pathways for the global first-percentile positive detection threshold.
   Zero/null abundances are replaced with half that threshold, sum-normalized
   per sample, scaled to parts per million and log10 transformed; pathways must
   pass the prespecified sample prevalence;
3. ``pathway_abundance_microbe_level.arrow`` supplies species-stratified
   HUMAnN RPK for taxon contributions. A pathway-taxon edge requires raw RPK >
   0 in at least the prespecified number of samples. Null RPK is treated as no
   contributed reads only for this contribution rule.

The un-imputed oral Parquet defines the candidate reference cohort. The final
analytic cohort is the complete-case intersection of that Parquet and both
HUMAnN views. A sample missing from any view is excluded from all three; missing
HUMAnN rows are never imputed as zero. Positional matching and row-count
equivalence are never used as evidence of sample identity.

If required inputs, sample identity, taxonomy coverage, or prespecified
descriptive eligibility criteria are not met, the run writes a manifest with
status "skipped". Failure of exploratory enrichment criteria does not suppress
an otherwise valid descriptive landscape.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import textwrap
import traceback
import uuid
import warnings
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
import scipy
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from scipy.stats import fisher_exact


ANALYSIS_VERSION = "4.1.0-frozen-cluster-universe"
PATTERN_IDS = (1, 2, 3, 4, 5)
FIGURE_WIDTH_MM = 183.0
FIGURE_MAX_HEIGHT_MM = 170.0
DEFAULT_SCOPE_EXCLUSION_REGEX = ""
HUMANN_SPECIAL_FEATURES = {
    "UNMAPPED",
    "UNINTEGRATED",
    "UNGROUPED",
    "UNCLASSIFIED",
}
TEXT = "#222222"
MUTED = "#666666"
MISSING = "#D9D9D9"
HEAT_LOW = "#0072B2"
HEAT_MID = "#F7F7F7"
HEAT_HIGH = "#D55E00"
REQUIRED_CLUSTER_COLUMNS = {"feature", "omics", "pattern_id", "pattern"}


class SampleAlignmentError(ValueError):
    """Raised when the three matrices cannot be aligned without guessing."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path, max_bytes: int | None = None) -> str | None:
    if max_bytes is not None and max_bytes > 0 and path.stat().st_size > max_bytes:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, hash_limit: int | None = None) -> dict[str, object]:
    stat = path.stat()
    digest = sha256_file(path, max_bytes=hash_limit)
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "sha256": digest,
        "sha256_status": (
            "computed"
            if digest is not None
            else f"skipped: file exceeds {int(hash_limit or 0)} bytes"
        ),
    }


def normalize_sample_key_component(
    values: pd.Series,
    column_name: str,
    source_label: str,
) -> pd.Series:
    """Normalize an identifier component without changing its semantics."""

    normalized = values.astype("string").str.strip()
    missing_tokens = {"", "nan", "none", "null", "na", "n/a", "<na>"}
    normalized = normalized.mask(normalized.str.casefold().isin(missing_tokens))
    if normalized.isna().any():
        raise SampleAlignmentError(
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
    """Build a unique composite key for one row per biological sample."""

    columns = list(key_columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SampleAlignmentError(
            f"{source_label} lacks requested sample-key columns {missing}; "
            f"available columns begin with {list(map(str, frame.columns[:20]))}"
        )
    components = [
        normalize_sample_key_component(frame[column], column, source_label)
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
    duplicate_mask = keys.duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_examples = sorted(keys.loc[duplicate_mask].unique())[:5]
        raise SampleAlignmentError(
            f"{source_label} sample key is not unique: "
            f"{int(duplicate_mask.sum())} rows participate in duplicates; "
            f"examples: {duplicate_examples}. Use --sample-key-columns with "
            "a biospecimen-level composite key."
        )
    return keys.reset_index(drop=True)


def read_oral_sample_keys(
    path: Path,
    key_columns: Iterable[str],
) -> pd.Series:
    columns = list(key_columns)
    try:
        frame = pd.read_parquet(path, columns=columns)
    except Exception as exc:
        raise SampleAlignmentError(
            "The oral-abundance Parquet cannot supply the requested sample key "
            f"{columns}: {type(exc).__name__}: {exc}"
        ) from exc
    return build_sample_keys(frame, columns, "oral-abundance Parquet")


def read_arrow_sample_keys(
    path: Path,
    key_columns: Iterable[str],
    source_label: str,
) -> pd.Series:
    """Read only physical Arrow index/metadata fields used for alignment."""

    columns = list(key_columns)
    schema_names = read_arrow_schema(path)
    missing = [column for column in columns if column not in schema_names]
    if missing:
        non_pathway_columns = [
            name
            for name in schema_names
            if "|" not in str(name)
            and split_pathway_label(name)[0].upper() not in HUMANN_SPECIAL_FEATURES
        ]
        raise SampleAlignmentError(
            f"{source_label} lacks requested sample-key columns {missing}. "
            "The Arrow must retain its pandas sample index/metadata; positional "
            "row matching is prohibited. Non-stratified schema examples: "
            f"{list(map(str, non_pathway_columns[:20]))}"
        )
    table = feather.read_table(path, columns=columns, memory_map=True)
    frame = table.to_pandas(ignore_metadata=True)
    return build_sample_keys(frame, columns, source_label)


def align_analysis_samples(
    oral_abundance_file: Path,
    community_pathway_file: Path,
    microbe_pathway_file: Path,
    key_columns: Iterable[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, dict[str, object]]:
    """Select the sample-key intersection shared by all three matrices."""

    columns = list(key_columns)
    oral_keys = read_oral_sample_keys(oral_abundance_file, columns)
    community_keys = read_arrow_sample_keys(
        community_pathway_file,
        columns,
        "community-level HUMAnN Arrow",
    )
    microbe_keys = read_arrow_sample_keys(
        microbe_pathway_file,
        columns,
        "microbe-level HUMAnN Arrow",
    )

    oral_index = pd.Index(oral_keys)
    community_index = pd.Index(community_keys)
    microbe_index = pd.Index(microbe_keys)
    in_community = oral_index.isin(community_index)
    in_microbe = oral_index.isin(microbe_index)
    complete_case_mask = in_community & in_microbe
    matched = oral_index[complete_case_mask]
    missing_from_community = oral_index[~in_community]
    missing_from_microbe = oral_index[~in_microbe]
    missing_from_either = oral_index[~complete_case_mask]
    missing_from_community_only = oral_index[(~in_community) & in_microbe]
    missing_from_microbe_only = oral_index[in_community & (~in_microbe)]
    missing_from_both = oral_index[(~in_community) & (~in_microbe)]
    community_extra = community_index.difference(oral_index, sort=False)
    microbe_extra = microbe_index.difference(oral_index, sort=False)

    included_hash = hashlib.sha256(
        "\n".join(sorted(map(str, matched))).encode("utf-8")
    ).hexdigest()
    alignment_status = "completed" if len(matched) > 0 else "skipped"
    retention_fraction = (
        float(len(matched) / len(oral_index)) if len(oral_index) else 0.0
    )
    rows = pd.DataFrame(
        [
            {
                "source": "oral_abundance_reference",
                "input_samples": len(oral_index),
                "oral_reference_samples_present": len(oral_index),
                "oral_reference_samples_missing": 0,
                "source_only_samples_excluded": 0,
                "complete_case_analysis_samples": len(matched),
                "oral_reference_samples_excluded_complete_case": len(
                    missing_from_either
                ),
            },
            {
                "source": "community_pathway_arrow",
                "input_samples": len(community_index),
                "oral_reference_samples_present": len(oral_index)
                - len(missing_from_community),
                "oral_reference_samples_missing": len(missing_from_community),
                "source_only_samples_excluded": len(community_extra),
                "complete_case_analysis_samples": len(matched),
                "oral_reference_samples_excluded_complete_case": len(
                    missing_from_either
                ),
            },
            {
                "source": "microbe_pathway_arrow",
                "input_samples": len(microbe_index),
                "oral_reference_samples_present": len(oral_index)
                - len(missing_from_microbe),
                "oral_reference_samples_missing": len(missing_from_microbe),
                "source_only_samples_excluded": len(microbe_extra),
                "complete_case_analysis_samples": len(matched),
                "oral_reference_samples_excluded_complete_case": len(
                    missing_from_either
                ),
            },
        ]
    )
    audit: dict[str, object] = {
        "status": alignment_status,
        "sample_key_columns": columns,
        "reference_cohort": "oral_abundance_table",
        "policy": (
            "complete-case intersection across the oral-reference Parquet and both "
            "HUMAnN views; exclude any sample absent from any source; exclude "
            "source-only samples; select and order rows by sample key, never position"
        ),
        "oral_reference_samples": int(len(oral_index)),
        "community_input_samples": int(len(community_index)),
        "microbe_input_samples": int(len(microbe_index)),
        "analysis_samples": int(len(matched)),
        "oral_reference_samples_excluded_complete_case": int(len(missing_from_either)),
        "complete_case_retention_fraction": retention_fraction,
        "community_extra_samples_excluded": int(len(community_extra)),
        "microbe_extra_samples_excluded": int(len(microbe_extra)),
        "oral_samples_missing_from_community": int(len(missing_from_community)),
        "oral_samples_missing_from_microbe": int(len(missing_from_microbe)),
        "oral_samples_missing_from_community_only": int(
            len(missing_from_community_only)
        ),
        "oral_samples_missing_from_microbe_only": int(len(missing_from_microbe_only)),
        "oral_samples_missing_from_both_humann_views": int(len(missing_from_both)),
        "missing_sample_rows_imputed_as_zero": False,
        "sample_missingness_policy": (
            "complete-case exclusion before species prevalence, pathway threshold, "
            "normalization, and contribution summaries"
        ),
        "included_sample_key_set_sha256": included_hash,
        "identifiers_exported": False,
        "privacy_note": (
            "only aggregate membership counts and a set hash are exported; raw "
            "sample identifiers are not written to analysis outputs"
        ),
    }
    if alignment_status != "completed":
        audit["skip_reason"] = (
            "No sample key is shared by the oral-abundance Parquet, the "
            "community-level HUMAnN Arrow, and the microbe-level HUMAnN Arrow"
        )
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty, rows, audit

    oral_rows = oral_index.get_indexer(matched).astype(np.int64)
    community_rows = community_index.get_indexer(matched).astype(np.int64)
    microbe_rows = microbe_index.get_indexer(matched).astype(np.int64)
    if np.any(oral_rows < 0) or np.any(community_rows < 0) or np.any(microbe_rows < 0):
        raise RuntimeError(
            "Completed sample alignment contains an unresolved row index"
        )
    matched_sequence = np.arange(len(matched), dtype=np.int64)
    audit["oral_row_subsetting_required"] = bool(
        not np.array_equal(oral_rows, matched_sequence)
        or len(oral_index) != len(matched)
    )
    audit["community_row_reordering_required"] = bool(
        not np.array_equal(community_rows, matched_sequence)
        or len(community_index) != len(matched)
    )
    audit["microbe_row_reordering_required"] = bool(
        not np.array_equal(microbe_rows, matched_sequence)
        or len(microbe_index) != len(matched)
    )
    return oral_rows, community_rows, microbe_rows, rows, audit


def subset_arrow_rows(
    frame: pd.DataFrame,
    row_indices: np.ndarray | None,
    source_label: str,
) -> pd.DataFrame:
    if row_indices is None:
        return frame
    indices = np.asarray(row_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise RuntimeError(f"{source_label} received an invalid empty row selector")
    if np.any(indices < 0) or np.any(indices >= len(frame)):
        raise RuntimeError(f"{source_label} row selector lies outside the Arrow table")
    return frame.iloc[indices].reset_index(drop=True)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def benjamini_hochberg(values: Iterable[float]) -> np.ndarray:
    p_values = np.asarray(list(values), dtype=float)
    if p_values.ndim != 1:
        raise ValueError("p-values must be one-dimensional")
    if p_values.size == 0:
        return np.array([], dtype=float)
    if not np.isfinite(p_values).all() or np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("p-values must be finite and lie within [0, 1]")
    order = np.argsort(p_values, kind="mergesort")
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def canonical_species(value: object) -> str | None:
    """Convert full taxonomies and HUMAnN labels to one species-level key."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    tail = str(value).strip().split("|")[-1]
    if "s__" not in tail:
        return None
    species = tail.rsplit("s__", 1)[1]
    species = species.split(".t__", 1)[0]
    species = re.sub(r"[^A-Za-z0-9]+", "_", species).strip("_").casefold()
    return species or None


def species_display(value: object) -> str:
    tail = str(value).strip().split("|")[-1]
    if "s__" in tail:
        tail = tail.rsplit("s__", 1)[1]
    tail = tail.split(".t__", 1)[0]
    words = [part for part in re.split(r"[_\s]+", tail) if part]
    if not words:
        return "Unresolved species"
    return " ".join(words[:4])


def split_pathway_label(raw_label: object) -> tuple[str, str]:
    raw = str(raw_label).strip()
    if ":" in raw:
        pathway_id, pathway_name = raw.split(":", 1)
        return pathway_id.strip(), pathway_name.strip()
    return raw, ""


def choose_pathway_name(values: Iterable[object]) -> str:
    cleaned = [
        str(value).strip() for value in values if pd.notna(value) and str(value).strip()
    ]
    if not cleaned:
        return ""
    counts = pd.Series(cleaned).value_counts(sort=True)
    top_count = int(counts.iloc[0])
    return sorted(counts.loc[counts == top_count].index.astype(str))[0]


def load_oral_universe(
    cluster_file: Path,
    oral_abundance_file: Path,
    omics_label: str,
    min_species_detected_samples: int,
    sample_row_indices: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Load the exact fixed cluster universe and audit, but do not refilter, it."""
    clusters = pd.read_csv(cluster_file)
    missing = sorted(REQUIRED_CLUSTER_COLUMNS - set(clusters.columns))
    if missing:
        raise KeyError("Cluster table lacks required columns: " + ", ".join(missing))

    oral = clusters.loc[clusters["omics"].astype(str) == omics_label].copy()
    if oral.empty:
        raise ValueError(f"No rows have omics == {omics_label!r}")
    oral["pattern_id"] = pd.to_numeric(oral["pattern_id"], errors="raise").astype(int)
    unexpected = sorted(set(oral["pattern_id"]) - set(PATTERN_IDS))
    if unexpected:
        raise ValueError(f"Unexpected oral pattern IDs: {unexpected}")
    if oral["pattern"].isna().any():
        raise ValueError("Oral cluster rows contain missing pattern labels")

    pattern_label_counts = oral.groupby("pattern_id")["pattern"].nunique()
    inconsistent_labels = pattern_label_counts.loc[pattern_label_counts != 1]
    if not inconsistent_labels.empty:
        raise ValueError(
            "Each pattern ID must map to exactly one label: "
            + repr(inconsistent_labels.to_dict())
        )

    oral["species_key"] = oral["feature"].map(canonical_species)
    if oral["species_key"].isna().any():
        examples = oral.loc[oral["species_key"].isna(), "feature"].head(5).tolist()
        raise ValueError(
            "Some oral cluster features are not species-level taxonomies: "
            + "; ".join(map(str, examples))
        )
    collisions = oral.loc[
        oral["species_key"].duplicated(keep=False),
        ["feature", "pattern_id", "species_key"],
    ]
    if not collisions.empty:
        raise ValueError(
            "Canonical species collisions make cluster membership ambiguous:\n"
            + collisions.head(10).to_string(index=False)
        )

    oral["species_display"] = oral["feature"].map(species_display)
    oral = oral[
        ["feature", "species_key", "species_display", "pattern_id", "pattern"]
    ].sort_values(["pattern_id", "species_key"], kind="mergesort")
    counts_before = {
        str(pattern_id): int((oral["pattern_id"] == pattern_id).sum())
        for pattern_id in PATTERN_IDS
    }

    abundance = pd.read_parquet(oral_abundance_file)
    input_abundance_samples = int(len(abundance))
    if sample_row_indices is not None:
        indices = np.asarray(sample_row_indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0:
            raise RuntimeError("Oral abundance received an invalid empty row selector")
        if np.any(indices < 0) or np.any(indices >= len(abundance)):
            raise RuntimeError("Oral abundance row selector lies outside the Parquet")
        abundance = abundance.iloc[indices].reset_index(drop=True)
    abundance_feature_columns = [
        str(column)
        for column in abundance.columns
        if canonical_species(column) is not None
    ]
    if not abundance_feature_columns:
        raise ValueError(
            "The oral-abundance Parquet has no species-level taxonomy columns"
        )
    abundance_keys = [canonical_species(column) for column in abundance_feature_columns]
    duplicated_keys = sorted(
        {
            str(key)
            for key in abundance_keys
            if key is not None and abundance_keys.count(key) > 1
        }
    )
    if duplicated_keys:
        raise ValueError(
            "Canonical species collisions occur in the oral-abundance Parquet: "
            + "; ".join(duplicated_keys[:10])
        )

    prevalence_rows: list[dict[str, object]] = []
    for column, species_key in zip(abundance_feature_columns, abundance_keys):
        values = numeric_column(
            abundance[column],
            column,
            source_label="Oral MetaPhlAn",
        )
        missing = np.isnan(values)
        prevalence_rows.append(
            {
                "abundance_feature": column,
                "species_key": str(species_key),
                "abundance_samples_n": int(len(values)),
                "species_detected_samples": int(
                    np.count_nonzero((~missing) & (values > 0))
                ),
                "species_missing_samples": int(missing.sum()),
                "species_zero_samples": int(
                    np.count_nonzero((~missing) & (values == 0))
                ),
            }
        )
    prevalence = pd.DataFrame(prevalence_rows)
    prevalence["species_prevalence_status"] = np.where(
        prevalence["species_detected_samples"] >= min_species_detected_samples,
        "meets_reference_prevalence_audit",
        "below_reference_prevalence_audit",
    )

    universe_audit = oral.merge(
        prevalence,
        on="species_key",
        how="left",
        validate="one_to_one",
    )
    universe_audit["species_prevalence_status"] = universe_audit[
        "species_prevalence_status"
    ].fillna("absent_from_reference_abundance_table")
    fixed_universe = universe_audit[
        [
            "feature",
            "species_key",
            "species_display",
            "pattern_id",
            "pattern",
            "abundance_feature",
            "abundance_samples_n",
            "species_detected_samples",
            "species_missing_samples",
            "species_zero_samples",
            "species_prevalence_status",
        ]
    ].sort_values(["pattern_id", "species_key"], kind="mergesort")
    counts_meeting_audit = {
        str(pattern_id): int(
            (
                (universe_audit["pattern_id"] == pattern_id)
                & universe_audit["species_prevalence_status"].eq(
                    "meets_reference_prevalence_audit"
                )
            ).sum()
        )
        for pattern_id in PATTERN_IDS
    }
    labels = {
        str(pattern_id): str(
            oral.loc[oral["pattern_id"] == pattern_id, "pattern"].iloc[0]
        )
        for pattern_id in PATTERN_IDS
        if counts_before[str(pattern_id)] > 0
    }
    audit = {
        "omics_filter": omics_label,
        "universe_definition": (
            "all unique species-level oral taxa in the fixed cluster assignment "
            "table; current-cohort MetaPhlAn prevalence is audited but never "
            "changes frozen P1-P5 membership"
        ),
        "inference_unit": "oral microbial taxon",
        "oral_abundance_input_samples": input_abundance_samples,
        "oral_abundance_samples": int(len(abundance)),
        "sample_subset_policy": (
            "sample-ID-aligned oral reference cohort"
            if sample_row_indices is not None
            else "all input rows"
        ),
        "oral_abundance_species_features": int(len(prevalence)),
        "cluster_taxa_before_prevalence_filter": int(len(oral)),
        "cluster_taxa_in_fixed_universe": int(len(fixed_universe)),
        "universe_n": int(len(fixed_universe)),
        "pattern_counts_before_prevalence_filter": counts_before,
        "pattern_counts": counts_before,
        "reference_prevalence_threshold_samples": int(min_species_detected_samples),
        "taxa_meeting_reference_prevalence_audit": int(
            universe_audit["species_prevalence_status"]
            .eq("meets_reference_prevalence_audit")
            .sum()
        ),
        "taxa_below_reference_prevalence_audit": int(
            universe_audit["species_prevalence_status"]
            .eq("below_reference_prevalence_audit")
            .sum()
        ),
        "taxa_absent_from_reference_abundance_table": int(
            universe_audit["species_prevalence_status"]
            .eq("absent_from_reference_abundance_table")
            .sum()
        ),
        "pattern_counts_meeting_reference_prevalence_audit": (counts_meeting_audit),
        "pattern_labels": labels,
        "present_patterns": [
            pattern_id
            for pattern_id in PATTERN_IDS
            if counts_before[str(pattern_id)] > 0
        ],
        "missing_patterns": [
            pattern_id
            for pattern_id in PATTERN_IDS
            if counts_before[str(pattern_id)] == 0
        ],
        "cluster_taxa_not_meeting_reference_prevalence_audit": universe_audit.loc[
            universe_audit["species_prevalence_status"]
            != "meets_reference_prevalence_audit",
            [
                "feature",
                "species_key",
                "pattern_id",
                "species_detected_samples",
                "species_prevalence_status",
            ],
        ].to_dict("records"),
    }
    return (
        fixed_universe.reset_index(drop=True),
        universe_audit.sort_values(
            ["pattern_id", "species_key"], kind="mergesort"
        ).reset_index(drop=True),
        audit,
    )


def read_arrow_schema(path: Path) -> list[str]:
    """Read an Arrow IPC/Feather schema without loading the full matrix."""
    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_file(source)
        return list(reader.schema.names)


def parse_humann_columns(
    microbe_column_names: Iterable[str],
    community_column_names: Iterable[str],
    universe_species: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, set[str], dict[str, object]]:
    microbe_names = [str(column) for column in microbe_column_names]
    community_names = [str(column) for column in community_column_names]

    for source_label, names in (
        ("microbe-level", microbe_names),
        ("pathway-level", community_names),
    ):
        seen: set[str] = set()
        duplicates: set[str] = set()
        for name in names:
            if name in seen:
                duplicates.add(name)
            seen.add(name)
        if duplicates:
            raise ValueError(
                f"Duplicate {source_label} Arrow column names cannot be "
                "selected unambiguously: " + "; ".join(sorted(duplicates)[:5])
            )

    stratified_records: list[dict[str, str]] = []
    all_humann_species: set[str] = set()
    all_stratified_pathway_ids: set[str] = set()
    stratified_columns = 0
    for column_text in microbe_names:
        if "|" not in column_text:
            continue
        stratified_columns += 1
        pathway_raw, taxon_raw = column_text.rsplit("|", 1)
        pathway_id, pathway_name = split_pathway_label(pathway_raw)
        if not pathway_id:
            continue
        all_stratified_pathway_ids.add(pathway_id)
        species_key = canonical_species(taxon_raw)
        if species_key is None:
            continue
        all_humann_species.add(species_key)
        if species_key not in universe_species:
            continue
        stratified_records.append(
            {
                "column": column_text,
                "pathway_id": pathway_id,
                "pathway_name": pathway_name,
                "species_key": species_key,
            }
        )
    feature_info = pd.DataFrame(
        stratified_records,
        columns=["column", "pathway_id", "pathway_name", "species_key"],
    )
    matched_stratified_pathway_ids = set(feature_info["pathway_id"].astype(str))
    community_records: list[dict[str, str]] = []
    unrecognized_community_columns: list[str] = []
    for column_text in community_names:
        if "|" in column_text:
            continue
        pathway_id, pathway_name = split_pathway_label(column_text)
        normalized_id = pathway_id.strip().upper()
        is_special = normalized_id in HUMANN_SPECIAL_FEATURES or any(
            normalized_id.startswith(prefix + "-") for prefix in HUMANN_SPECIAL_FEATURES
        )
        if pathway_id in all_stratified_pathway_ids or is_special:
            community_records.append(
                {
                    "column": column_text,
                    "pathway_id": pathway_id,
                    "pathway_name": pathway_name,
                }
            )
        else:
            unrecognized_community_columns.append(column_text)
    community_info = pd.DataFrame(
        community_records,
        columns=["column", "pathway_id", "pathway_name"],
    )
    audit = {
        "microbe_level_schema_columns": int(len(microbe_names)),
        "pathway_level_schema_columns": int(len(community_names)),
        "microbe_level_stratified_columns": int(stratified_columns),
        "pathway_level_columns_containing_taxon_delimiter": int(
            sum("|" in name for name in community_names)
        ),
        "matched_stratified_columns_before_scope_filter": int(len(feature_info)),
        "recognized_community_pathway_columns": int(len(community_info)),
        "unrecognized_pathway_level_columns": int(len(unrecognized_community_columns)),
        "unrecognized_pathway_level_column_examples": (
            unrecognized_community_columns[:10]
        ),
        "all_stratified_pathway_ids": int(len(all_stratified_pathway_ids)),
        "matched_stratified_pathway_ids": int(len(matched_stratified_pathway_ids)),
        "stratified_pathway_ids_with_community_abundance": int(
            len(matched_stratified_pathway_ids & set(community_info["pathway_id"]))
        ),
        "humann_species_total": int(len(all_humann_species)),
        "matched_cluster_species": int(len(all_humann_species & universe_species)),
    }
    return feature_info, community_info, all_humann_species, audit


def pathway_scope_reason(
    pathway_id: str,
    pathway_name: str,
    exclusion_regex: re.Pattern[str] | None,
) -> str | None:
    normalized_id = pathway_id.strip().upper()
    if normalized_id in HUMANN_SPECIAL_FEATURES or any(
        normalized_id.startswith(prefix + "-") for prefix in HUMANN_SPECIAL_FEATURES
    ):
        return "excluded_humann_special_feature"
    searchable = f"{pathway_id} {pathway_name}".strip()
    if exclusion_regex is not None and exclusion_regex.search(searchable):
        return "excluded_explicit_out_of_scope_name"
    return None


def build_scope_audit(
    feature_info: pd.DataFrame,
    exclusion_pattern: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if feature_info.empty:
        empty = pd.DataFrame(
            columns=[
                "pathway_id",
                "pathway_name",
                "scope_status",
                "scope_exclusion_reason",
            ]
        )
        return (
            empty,
            feature_info.copy(),
            {
                "pathways_before_scope_filter": 0,
                "pathways_after_scope_filter": 0,
                "pathways_excluded": 0,
            },
        )

    compiled = re.compile(exclusion_pattern) if exclusion_pattern else None
    scoped = feature_info.copy()
    scoped["column_scope_reason"] = [
        pathway_scope_reason(pathway_id, pathway_name, compiled)
        for pathway_id, pathway_name in zip(
            scoped["pathway_id"], scoped["pathway_name"]
        )
    ]

    rows: list[dict[str, object]] = []
    for pathway_id, group in scoped.groupby("pathway_id", sort=True):
        names = sorted(
            {
                str(value).strip()
                for value in group["pathway_name"]
                if pd.notna(value) and str(value).strip()
            }
        )
        reasons = sorted(
            {
                str(value)
                for value in group["column_scope_reason"]
                if pd.notna(value) and str(value)
            }
        )
        status = "excluded" if reasons else "included"
        rows.append(
            {
                "pathway_id": str(pathway_id),
                "pathway_name": choose_pathway_name(group["pathway_name"]),
                "all_observed_names": "; ".join(names),
                "n_distinct_names": int(len(names)),
                "n_matched_columns": int(len(group)),
                "n_matched_taxa": int(group["species_key"].nunique()),
                "scope_status": status,
                "scope_exclusion_reason": "; ".join(reasons),
            }
        )
    scope_audit = pd.DataFrame(rows)
    included_ids = set(
        scope_audit.loc[scope_audit["scope_status"] == "included", "pathway_id"]
    )
    eligible_columns = scoped.loc[scoped["pathway_id"].isin(included_ids)].copy()
    exclusion_rule = "exact HUMAnN special-feature IDs only"
    if exclusion_pattern:
        exclusion_rule += (
            " plus a user-supplied prespecified scope regex; no "
            "biological-theme grouping"
        )
    summary = {
        "exclusion_rule": exclusion_rule,
        "out_of_scope_name_regex": exclusion_pattern,
        "pathways_before_scope_filter": int(len(scope_audit)),
        "pathways_after_scope_filter": int(len(included_ids)),
        "pathways_excluded": int((scope_audit["scope_status"] == "excluded").sum()),
        "pathways_with_name_conflicts": int(
            (scope_audit["n_distinct_names"] > 1).sum()
        ),
    }
    return scope_audit, eligible_columns, summary


def numeric_column(
    values: pd.Series,
    column_name: str,
    source_label: str = "HUMAnN",
) -> np.ndarray:
    """Validate one abundance column while preserving source missingness."""
    source_missing = values.isna().to_numpy(dtype=bool)
    converted = pd.to_numeric(values, errors="coerce")
    converted_missing = converted.isna().to_numpy(dtype=bool)
    conversion_failure = (~source_missing) & converted_missing
    if conversion_failure.any():
        raise ValueError(
            f"{source_label} column {column_name!r} contains "
            f"{int(conversion_failure.sum())} non-numeric, non-missing values"
        )
    numeric = converted.to_numpy(dtype=float, na_value=np.nan)
    infinite = np.isinf(numeric)
    if infinite.any():
        raise ValueError(
            f"{source_label} column {column_name!r} contains "
            f"{int(infinite.sum())} infinite values"
        )
    if np.any(numeric[np.isfinite(numeric)] < 0):
        raise ValueError(
            f"Negative {source_label} abundance occurs in column {column_name!r}"
        )
    return numeric


def summarize_humann_pairs(
    arrow_file: Path,
    feature_info: pd.DataFrame,
    batch_columns: int,
    sample_row_indices: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if feature_info.empty:
        return pd.DataFrame(), {
            "selected_columns": 0,
            "samples": 0,
            "collapsed_pathway_species_pairs": 0,
            "pathway_species_pairs": 0,
            "source_missing_values": 0,
            "pathway_species_sample_nulls_treated_as_zero": 0,
        }

    selected_columns = feature_info["column"].tolist()
    column_stats: list[dict[str, object]] = []
    input_samples: int | None = None
    n_samples: int | None = None
    for start in range(0, len(selected_columns), batch_columns):
        batch_names = selected_columns[start : start + batch_columns]
        table = feather.read_table(
            arrow_file,
            columns=batch_names,
            memory_map=True,
        )
        frame = table.to_pandas(ignore_metadata=True)
        if input_samples is None:
            input_samples = int(len(frame))
        elif len(frame) != input_samples:
            raise RuntimeError("Arrow batches have inconsistent sample counts")
        frame = subset_arrow_rows(
            frame,
            sample_row_indices,
            "microbe-level HUMAnN Arrow",
        )
        if n_samples is None:
            n_samples = int(len(frame))
        elif len(frame) != n_samples:
            raise RuntimeError("Aligned Arrow batches have inconsistent sample counts")
        for column_name in batch_names:
            values = numeric_column(frame[column_name], column_name)
            missing = np.isnan(values)
            zero_filled_values = values.copy()
            zero_filled_values[missing] = 0.0
            column_stats.append(
                {
                    "column": column_name,
                    "n_detected_samples": int(
                        np.count_nonzero((~missing) & (values > 0))
                    ),
                    "mean_rpk_all_samples": float(zero_filled_values.mean()),
                    "n_source_missing_values": int(missing.sum()),
                }
            )

    stats = feature_info.merge(
        pd.DataFrame(column_stats),
        on="column",
        how="left",
        validate="one_to_one",
    )
    pair_rows: list[dict[str, object]] = []
    collapsed_pairs = 0
    for (pathway_id, species_key), group in stats.groupby(
        ["pathway_id", "species_key"], sort=True, dropna=False
    ):
        columns = group["column"].tolist()
        if len(columns) == 1:
            detected = int(group["n_detected_samples"].iloc[0])
            mean_rpk = float(group["mean_rpk_all_samples"].iloc[0])
            pair_missing_samples = int(group["n_source_missing_values"].iloc[0])
        else:
            collapsed_pairs += 1
            table = feather.read_table(
                arrow_file,
                columns=columns,
                memory_map=True,
            )
            frame = subset_arrow_rows(
                table.to_pandas(ignore_metadata=True),
                sample_row_indices,
                "microbe-level HUMAnN Arrow",
            )
            arrays = [
                numeric_column(frame[column_name], column_name)
                for column_name in columns
            ]
            stacked = np.column_stack(arrays)
            pair_missing = np.all(np.isnan(stacked), axis=1)
            pair_values = np.nansum(stacked, axis=1)
            detected = int(np.count_nonzero((~pair_missing) & (pair_values > 0)))
            mean_rpk = float(pair_values.mean())
            pair_missing_samples = int(pair_missing.sum())
        pair_rows.append(
            {
                "pathway_id": str(pathway_id),
                "pathway_name": choose_pathway_name(group["pathway_name"]),
                "species_key": str(species_key),
                "n_detected_samples": detected,
                "mean_rpk_all_samples": mean_rpk,
                "n_missing_samples_treated_as_zero": pair_missing_samples,
                "n_source_columns": int(len(columns)),
            }
        )
    source_missing_values = int(
        pd.DataFrame(column_stats)["n_source_missing_values"].sum()
    )
    columns_with_missing = int(
        (pd.DataFrame(column_stats)["n_source_missing_values"] > 0).sum()
    )
    pair_missing_values = int(
        sum(int(row["n_missing_samples_treated_as_zero"]) for row in pair_rows)
    )
    audit = {
        "selected_columns": int(len(selected_columns)),
        "input_samples": int(input_samples or 0),
        "samples": int(n_samples or 0),
        "sample_id_subset_applied": sample_row_indices is not None,
        "pathway_species_collapse_rule": (
            "sum distinct HUMAnN stratified columns that collapse to the same "
            "MetaCyc pathway ID and canonical species"
        ),
        "collapsed_pathway_species_pairs": int(collapsed_pairs),
        "pathway_species_pairs": int(len(pair_rows)),
        "contribution_rule": (
            "raw species-stratified RPK > 0; genuine null means no "
            "contributed reads and never counts as detected"
        ),
        "source_missing_values": source_missing_values,
        "source_columns_with_missing_values": columns_with_missing,
        "pathway_species_sample_nulls_treated_as_zero": pair_missing_values,
        "missing_value_policy": (
            "genuine Arrow null/NaN means no contributed reads for raw RPK "
            "summaries; non-numeric, infinite, and negative values fail"
        ),
    }
    return pd.DataFrame(pair_rows), audit


def preprocess_community_pathways(
    arrow_file: Path,
    community_info: pd.DataFrame,
    detection_percentile: float,
    min_detected_samples: int,
    batch_columns: int,
    sample_row_indices: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply the prespecified HUMAnN pathway preprocessing sequence."""
    if community_info.empty:
        return pd.DataFrame(), {
            "status": "skipped",
            "skip_reason": "No community-level HUMAnN pathway columns were found",
        }
    duplicated_ids = community_info.loc[
        community_info["pathway_id"].duplicated(keep=False), "pathway_id"
    ].astype(str)
    if not duplicated_ids.empty:
        raise ValueError(
            "Community-level HUMAnN pathway IDs are not unique: "
            + "; ".join(sorted(set(duplicated_ids))[:10])
        )

    columns = community_info["column"].tolist()
    positive_blocks: list[np.ndarray] = []
    raw_audit: dict[str, dict[str, int]] = {}
    input_samples: int | None = None
    n_samples: int | None = None
    for start in range(0, len(columns), batch_columns):
        batch_names = columns[start : start + batch_columns]
        frame = feather.read_table(
            arrow_file,
            columns=batch_names,
            memory_map=True,
        ).to_pandas(ignore_metadata=True)
        if input_samples is None:
            input_samples = int(len(frame))
        elif len(frame) != input_samples:
            raise RuntimeError("Arrow batches have inconsistent sample counts")
        frame = subset_arrow_rows(
            frame,
            sample_row_indices,
            "community-level HUMAnN Arrow",
        )
        if n_samples is None:
            n_samples = int(len(frame))
        elif len(frame) != n_samples:
            raise RuntimeError("Aligned Arrow batches have inconsistent sample counts")
        for column_name in batch_names:
            values = numeric_column(frame[column_name], column_name)
            finite = np.isfinite(values)
            positive = values[finite & (values > 0)]
            if positive.size:
                positive_blocks.append(positive)
            raw_audit[column_name] = {
                "raw_positive_values": int(positive.size),
                "raw_zero_values": int(np.count_nonzero(finite & (values == 0))),
                "raw_missing_values": int(np.count_nonzero(~finite)),
            }

    if not positive_blocks:
        return pd.DataFrame(), {
            "status": "skipped",
            "skip_reason": (
                "No positive, non-zero community pathway abundance is available "
                "for detection-threshold estimation"
            ),
            "samples": int(n_samples or 0),
            "input_samples": int(input_samples or 0),
            "community_pathways": int(len(community_info)),
        }
    all_positive = np.concatenate(positive_blocks)
    detection_threshold = float(
        np.percentile(all_positive, detection_percentile, method="linear")
    )
    if not np.isfinite(detection_threshold) or detection_threshold <= 0:
        raise RuntimeError("The pathway detection threshold is not positive and finite")
    zero_replacement = detection_threshold / 2.0

    denominator = np.zeros(int(n_samples or 0), dtype=float)
    for start in range(0, len(columns), batch_columns):
        batch_names = columns[start : start + batch_columns]
        frame = feather.read_table(
            arrow_file,
            columns=batch_names,
            memory_map=True,
        ).to_pandas(ignore_metadata=True)
        frame = subset_arrow_rows(
            frame,
            sample_row_indices,
            "community-level HUMAnN Arrow",
        )
        for column_name in batch_names:
            values = numeric_column(frame[column_name], column_name)
            replace = np.isnan(values) | (values == 0)
            adjusted = values.copy()
            adjusted[replace] = zero_replacement
            denominator += adjusted
    if not np.isfinite(denominator).all() or np.any(denominator <= 0):
        raise RuntimeError(
            "Pathway sum-normalization denominator is non-positive or non-finite"
        )

    names = community_info.set_index("column")
    rows: list[dict[str, object]] = []
    for start in range(0, len(columns), batch_columns):
        batch_names = columns[start : start + batch_columns]
        frame = feather.read_table(
            arrow_file,
            columns=batch_names,
            memory_map=True,
        ).to_pandas(ignore_metadata=True)
        frame = subset_arrow_rows(
            frame,
            sample_row_indices,
            "community-level HUMAnN Arrow",
        )
        for column_name in batch_names:
            values = numeric_column(frame[column_name], column_name)
            finite = np.isfinite(values)
            detected = finite & (values >= detection_threshold)
            replace = (~finite) | (values == 0)
            adjusted = values.copy()
            adjusted[replace] = zero_replacement
            ppm = adjusted / denominator * 1_000_000.0
            if not np.isfinite(ppm).all() or np.any(ppm <= 0):
                raise RuntimeError(
                    f"Invalid normalized pathway abundance for {column_name!r}"
                )
            log10_ppm = np.log10(ppm)
            pathway_id = str(names.loc[column_name, "pathway_id"])
            pathway_name = str(names.loc[column_name, "pathway_name"])
            rows.append(
                {
                    "pathway_id": pathway_id,
                    "pathway_name": pathway_name,
                    "community_column": column_name,
                    "community_samples_n": int(len(values)),
                    "pathway_detection_percentile": float(detection_percentile),
                    "global_pathway_detection_threshold": detection_threshold,
                    "zero_and_null_replacement_value": zero_replacement,
                    "community_detected_samples": int(detected.sum()),
                    "community_raw_positive_samples": int(
                        np.count_nonzero(finite & (values > 0))
                    ),
                    "community_zero_values_replaced": raw_audit[column_name][
                        "raw_zero_values"
                    ],
                    "community_missing_values_replaced": raw_audit[column_name][
                        "raw_missing_values"
                    ],
                    "community_mean_raw_abundance": float(
                        np.nanmean(values) if finite.any() else np.nan
                    ),
                    "community_mean_ppm": float(ppm.mean()),
                    "community_mean_log10_ppm": float(log10_ppm.mean()),
                    "community_pathway_prevalence_status": (
                        "retained_community_pathway_prevalence"
                        if int(detected.sum()) >= min_detected_samples
                        else "excluded_community_pathway_detected_in_too_few_samples"
                    ),
                }
            )
    result = pd.DataFrame(rows).sort_values("pathway_id", kind="mergesort")
    audit = {
        "status": "completed",
        "input_samples": int(input_samples or 0),
        "samples": int(n_samples or 0),
        "sample_id_subset_applied": sample_row_indices is not None,
        "community_pathways": int(len(result)),
        "detection_threshold_definition": (
            f"{detection_percentile:g}th percentile of every positive, non-zero "
            "community pathway abundance after scope exclusions"
        ),
        "detection_threshold": detection_threshold,
        "zero_and_null_replacement": zero_replacement,
        "normalization": (
            "replace zero/null with half the detection threshold; sum-normalize "
            "within sample; multiply by 1,000,000; log10 transform"
        ),
        "minimum_detected_samples": int(min_detected_samples),
        "positive_values_used_for_threshold": int(all_positive.size),
        "raw_zero_values_replaced": int(
            sum(item["raw_zero_values"] for item in raw_audit.values())
        ),
        "raw_missing_values_replaced": int(
            sum(item["raw_missing_values"] for item in raw_audit.values())
        ),
        "retained_pathways": int(
            (
                result["community_pathway_prevalence_status"]
                == "retained_community_pathway_prevalence"
            ).sum()
        ),
        "sum_normalization_denominator_min": float(denominator.min()),
        "sum_normalization_denominator_max": float(denominator.max()),
    }
    return result.reset_index(drop=True), audit


def taxonomy_coverage(
    universe: pd.DataFrame,
    humann_species: set[str],
) -> dict[str, object]:
    matched = set(universe["species_key"]) & humann_species
    pattern_rates: dict[str, float] = {}
    pattern_matched: dict[str, int] = {}
    for pattern_id in PATTERN_IDS:
        pattern_set = set(
            universe.loc[universe["pattern_id"] == pattern_id, "species_key"]
        )
        matched_n = len(pattern_set & matched)
        pattern_matched[str(pattern_id)] = int(matched_n)
        pattern_rates[str(pattern_id)] = (
            float(matched_n / len(pattern_set)) if pattern_set else 0.0
        )
    unmatched = universe.loc[
        ~universe["species_key"].isin(matched),
        ["species_key", "species_display", "pattern_id"],
    ]
    return {
        "matched_taxa_n": int(len(matched)),
        "unmatched_taxa_n": int(len(universe) - len(matched)),
        "overall_rate": float(len(matched) / len(universe)) if len(universe) else 0.0,
        "pattern_matched_taxa": pattern_matched,
        "pattern_rates": pattern_rates,
        "unmatched_taxa": unmatched.to_dict("records"),
    }


def build_pathway_filter_audit(
    scope_audit: pd.DataFrame,
    community_summary: pd.DataFrame,
    pair_stats: pd.DataFrame,
    min_contribution_samples: int,
    universe_n: int,
    min_carrier_taxa: int,
    min_noncarrier_taxa: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separate descriptive eligibility from exploratory-test eligibility."""

    stable_edges = pair_stats.loc[
        pair_stats["n_detected_samples"] >= min_contribution_samples
    ].copy()
    pair_summary = pair_stats.groupby("pathway_id", as_index=False).agg(
        pathway_species_pairs_before_detection=("species_key", "nunique"),
        maximum_detected_samples=("n_detected_samples", "max"),
    )
    edge_summary = stable_edges.groupby("pathway_id", as_index=False).agg(
        carrier_taxa_n=("species_key", "nunique"),
        stable_pathway_species_edges=("species_key", "size"),
    )
    community_columns = [
        "pathway_id",
        "community_column",
        "community_samples_n",
        "pathway_detection_percentile",
        "global_pathway_detection_threshold",
        "zero_and_null_replacement_value",
        "community_detected_samples",
        "community_raw_positive_samples",
        "community_zero_values_replaced",
        "community_missing_values_replaced",
        "community_mean_raw_abundance",
        "community_mean_ppm",
        "community_mean_log10_ppm",
        "community_pathway_prevalence_status",
    ]
    audit = (
        scope_audit.merge(
            community_summary[community_columns],
            on="pathway_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            pair_summary,
            on="pathway_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            edge_summary,
            on="pathway_id",
            how="left",
            validate="one_to_one",
        )
    )
    integer_columns = [
        "pathway_species_pairs_before_detection",
        "maximum_detected_samples",
        "carrier_taxa_n",
        "stable_pathway_species_edges",
    ]
    for column in integer_columns:
        audit[column] = audit[column].fillna(0).astype(int)
    audit["noncarrier_taxa_n"] = universe_n - audit["carrier_taxa_n"]

    def descriptive_status(row: pd.Series) -> str:
        if row["scope_status"] != "included":
            return str(row["scope_exclusion_reason"])
        if pd.isna(row["community_pathway_prevalence_status"]):
            return "excluded_missing_community_pathway_abundance"
        if (
            row["community_pathway_prevalence_status"]
            != "retained_community_pathway_prevalence"
        ):
            return str(row["community_pathway_prevalence_status"])
        if row["stable_pathway_species_edges"] == 0:
            return "excluded_no_contribution_edge_meets_detection_threshold"
        return "included_in_descriptive_landscape"

    def exploratory_status(row: pd.Series) -> str:
        if row["descriptive_filter_status"] != "included_in_descriptive_landscape":
            return str(row["descriptive_filter_status"])
        if row["carrier_taxa_n"] < min_carrier_taxa:
            return "excluded_too_few_carrier_taxa"
        if row["noncarrier_taxa_n"] < min_noncarrier_taxa:
            return "excluded_too_few_noncarrier_taxa"
        return "included_in_exploratory_test_family"

    audit["descriptive_filter_status"] = audit.apply(
        descriptive_status,
        axis=1,
    )
    audit["exploratory_filter_status"] = audit.apply(
        exploratory_status,
        axis=1,
    )
    descriptive_ids = set(
        audit.loc[
            audit["descriptive_filter_status"] == "included_in_descriptive_landscape",
            "pathway_id",
        ]
    )
    exploratory_ids = set(
        audit.loc[
            audit["exploratory_filter_status"] == "included_in_exploratory_test_family",
            "pathway_id",
        ]
    )
    descriptive_edges = stable_edges.loc[
        stable_edges["pathway_id"].isin(descriptive_ids)
    ].copy()
    exploratory_edges = stable_edges.loc[
        stable_edges["pathway_id"].isin(exploratory_ids)
    ].copy()
    return (
        audit.sort_values("pathway_id", kind="mergesort"),
        descriptive_edges,
        exploratory_edges,
    )


def corrected_log2_odds_ratio(
    a: int,
    b: int,
    c: int,
    d: int,
) -> tuple[float, float, float]:
    aa, bb, cc, dd = [float(value) + 0.5 for value in (a, b, c, d)]
    log_or = math.log((aa * dd) / (bb * cc))
    se = math.sqrt(1.0 / aa + 1.0 / bb + 1.0 / cc + 1.0 / dd)
    scale = math.log(2.0)
    return (
        log_or / scale,
        (log_or - 1.96 * se) / scale,
        (log_or + 1.96 * se) / scale,
    )


def pathway_enrichment_table(
    eligible_edges: pd.DataFrame,
    universe: pd.DataFrame,
    pathway_names: dict[str, str],
    community_summary: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    universe_species = set(universe["species_key"])
    pattern_species = {
        pattern_id: set(
            universe.loc[universe["pattern_id"] == pattern_id, "species_key"]
        )
        for pattern_id in PATTERN_IDS
    }
    present_by_pathway = eligible_edges.groupby("pathway_id")["species_key"].agg(
        lambda values: set(values)
    )
    rows: list[dict[str, object]] = []
    for pathway_id in sorted(present_by_pathway.index.astype(str)):
        present_species = set(present_by_pathway.loc[pathway_id]) & universe_species
        for pattern_id in PATTERN_IDS:
            foreground = pattern_species[pattern_id]
            a = len(present_species & foreground)
            b = len(foreground) - a
            c = len(present_species - foreground)
            d = (len(universe_species) - len(foreground)) - c
            if min(a, b, c, d) < 0:
                raise RuntimeError(
                    f"Invalid contingency table for {pathway_id}, P{pattern_id}: "
                    f"{[a, b, c, d]}"
                )
            exact_or, p_value = fisher_exact(
                [[a, b], [c, d]],
                alternative="greater",
            )
            log2_or, ci_low, ci_high = corrected_log2_odds_ratio(a, b, c, d)
            rows.append(
                {
                    "pathway_id": pathway_id,
                    "pathway_name": pathway_names.get(pathway_id, ""),
                    "pattern_id": pattern_id,
                    "foreground_with": a,
                    "foreground_without": b,
                    "nonforeground_with": c,
                    "nonforeground_without": d,
                    "foreground_n": int(len(foreground)),
                    "nonforeground_n": int(len(universe_species) - len(foreground)),
                    "universe_n": int(len(universe_species)),
                    "exact_odds_ratio": float(exact_or),
                    "log2_odds_ratio_ha": float(log2_or),
                    "ci95_low_log2_ha": float(ci_low),
                    "ci95_high_log2_ha": float(ci_high),
                    "p_fisher_greater": float(p_value),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["p_bonferroni_global"] = np.minimum(
        result["p_fisher_greater"] * len(result),
        1.0,
    )
    result["q_bh_global"] = benjamini_hochberg(result["p_fisher_greater"])
    result["exploratory_positive_bh"] = (result["q_bh_global"] < alpha) & (
        result["log2_odds_ratio_ha"] > 0
    )
    result["strict_positive_bonferroni_sensitivity"] = (
        result["p_bonferroni_global"] < alpha
    ) & (result["log2_odds_ratio_ha"] > 0)
    result["test"] = "one-sided Fisher exact test (greater)"
    result["analysis_role"] = "exploratory supplementary taxon-set enrichment"
    result["multiple_testing"] = (
        "Benjamini-Hochberg FDR across every eligible pathway x five patterns; "
        "Bonferroni reported as a strict sensitivity analysis"
    )
    result[
        "effect_interval"
    ] = "Haldane-Anscombe corrected log2 odds ratio; approximate Wald 95% CI"
    community_columns = [
        "pathway_id",
        "pathway_detection_percentile",
        "global_pathway_detection_threshold",
        "zero_and_null_replacement_value",
        "community_detected_samples",
        "community_mean_raw_abundance",
        "community_mean_ppm",
        "community_mean_log10_ppm",
    ]
    result = result.merge(
        community_summary[community_columns],
        on="pathway_id",
        how="left",
        validate="many_to_one",
    )
    return result.sort_values(
        ["p_bonferroni_global", "p_fisher_greater", "log2_odds_ratio_ha"],
        ascending=[True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)


def pathway_abundance_table(
    descriptive_edges: pd.DataFrame,
    universe: pd.DataFrame,
    pathway_names: dict[str, str],
) -> pd.DataFrame:
    """Summarize quantitative pathway contribution across the five patterns."""

    joined = descriptive_edges.merge(
        universe[["species_key", "pattern_id"]],
        on="species_key",
        how="inner",
        validate="many_to_one",
    )
    summed = joined.groupby(["pathway_id", "pattern_id"], as_index=False).agg(
        total_mean_rpk=("mean_rpk_all_samples", "sum"),
        stable_contributor_taxa_n=("species_key", "nunique"),
        stable_detection_samples_median=("n_detected_samples", "median"),
        stable_detection_samples_min=("n_detected_samples", "min"),
        stable_detection_samples_max=("n_detected_samples", "max"),
    )
    pattern_counts = universe["pattern_id"].value_counts().to_dict()
    rows: list[dict[str, object]] = []
    for pathway_id in sorted(descriptive_edges["pathway_id"].unique()):
        raw_values: list[float] = []
        pattern_records: list[dict[str, float | int]] = []
        for pattern_id in PATTERN_IDS:
            matched = summed.loc[
                (summed["pathway_id"] == pathway_id)
                & (summed["pattern_id"] == pattern_id)
            ]
            total = (
                float(matched["total_mean_rpk"].iloc[0]) if not matched.empty else 0.0
            )
            denominator = int(pattern_counts.get(pattern_id, 0))
            raw_values.append(total / denominator if denominator else np.nan)
            pattern_records.append(
                {
                    "total_mean_rpk": total,
                    "stable_contributor_taxa_n": (
                        int(matched["stable_contributor_taxa_n"].iloc[0])
                        if not matched.empty
                        else 0
                    ),
                    "stable_detection_samples_median": (
                        float(matched["stable_detection_samples_median"].iloc[0])
                        if not matched.empty
                        else np.nan
                    ),
                    "stable_detection_samples_min": (
                        int(matched["stable_detection_samples_min"].iloc[0])
                        if not matched.empty
                        else 0
                    ),
                    "stable_detection_samples_max": (
                        int(matched["stable_detection_samples_max"].iloc[0])
                        if not matched.empty
                        else 0
                    ),
                }
            )
        transformed = np.log1p(np.asarray(raw_values, dtype=float))
        finite = np.isfinite(transformed)
        if finite.all() and float(np.std(transformed, ddof=0)) > 0:
            z_values = (transformed - float(np.mean(transformed))) / float(
                np.std(transformed, ddof=0)
            )
            z_status = "variable_across_patterns"
        elif finite.all():
            z_values = np.zeros(len(PATTERN_IDS), dtype=float)
            z_status = "constant_across_patterns_set_to_zero"
        else:
            z_values = np.full(len(PATTERN_IDS), np.nan)
            z_status = "invalid_nonfinite_values"
        for pattern_id, raw, log_value, z_value, support in zip(
            PATTERN_IDS,
            raw_values,
            transformed,
            z_values,
            pattern_records,
        ):
            assigned_taxa_n = int(pattern_counts.get(pattern_id, 0))
            rows.append(
                {
                    "pathway_id": pathway_id,
                    "pathway_name": pathway_names.get(pathway_id, ""),
                    "pattern_id": pattern_id,
                    "pattern_taxa_n": assigned_taxa_n,
                    "stable_contributor_taxa_n": int(
                        support["stable_contributor_taxa_n"]
                    ),
                    "stable_contributor_fraction": (
                        float(support["stable_contributor_taxa_n"]) / assigned_taxa_n
                        if assigned_taxa_n
                        else np.nan
                    ),
                    "total_mean_rpk": float(support["total_mean_rpk"]),
                    "mean_rpk_per_assigned_taxon": float(raw),
                    "log1p_mean_rpk_per_assigned_taxon": float(log_value),
                    "row_zscore": (float(z_value) if np.isfinite(z_value) else np.nan),
                    "row_zscore_status": z_status,
                    "stable_detection_samples_median": float(
                        support["stable_detection_samples_median"]
                    ),
                    "stable_detection_samples_min": int(
                        support["stable_detection_samples_min"]
                    ),
                    "stable_detection_samples_max": int(
                        support["stable_detection_samples_max"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_descriptive_figure_selection(
    abundance: pd.DataFrame,
    descriptive_edges: pd.DataFrame,
    universe: pd.DataFrame,
    community_summary: pd.DataFrame,
    max_pathways: int,
    min_figure_carriers: int,
    representative_taxa_n: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Choose main-figure pathways without using pattern labels or P-values."""

    support = descriptive_edges.groupby("pathway_id", as_index=False).agg(
        pathway_name=("pathway_name", choose_pathway_name),
        stable_contributor_taxa_n=("species_key", "nunique"),
        stable_pathway_species_edges=("species_key", "size"),
        stable_detection_samples_median=("n_detected_samples", "median"),
        stable_detection_samples_min=("n_detected_samples", "min"),
        stable_detection_samples_max=("n_detected_samples", "max"),
        total_mean_rpk_all_taxa=("mean_rpk_all_samples", "sum"),
    )
    support["mean_rpk_per_retained_taxon"] = support["total_mean_rpk_all_taxa"] / int(
        len(universe)
    )
    community_columns = [
        "pathway_id",
        "pathway_detection_percentile",
        "global_pathway_detection_threshold",
        "zero_and_null_replacement_value",
        "community_detected_samples",
        "community_mean_raw_abundance",
        "community_mean_ppm",
        "community_mean_log10_ppm",
    ]
    support = support.merge(
        community_summary[community_columns],
        on="pathway_id",
        how="left",
        validate="one_to_one",
    )
    before_display_filter = int(len(support))
    carriers_ok = support["stable_contributor_taxa_n"] >= min_figure_carriers
    dropped_carriers = int((~carriers_ok).sum())
    ranked = support.loc[carriers_ok].sort_values(
        [
            "stable_contributor_taxa_n",
            "stable_detection_samples_median",
            "total_mean_rpk_all_taxa",
            "pathway_id",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    ranked = ranked.head(max_pathways).copy()

    dominant = (
        abundance.sort_values(
            ["pathway_id", "mean_rpk_per_assigned_taxon", "pattern_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .drop_duplicates("pathway_id")[
            ["pathway_id", "pattern_id", "mean_rpk_per_assigned_taxon"]
        ]
        .rename(
            columns={
                "pattern_id": "dominant_pattern",
                "mean_rpk_per_assigned_taxon": (
                    "dominant_pattern_mean_rpk_per_assigned_taxon"
                ),
            }
        )
    )
    ranked = ranked.merge(
        dominant,
        on="pathway_id",
        how="left",
        validate="one_to_one",
    )

    display_map = universe.set_index("species_key")["species_display"].to_dict()
    pattern_map = universe.set_index("species_key")["pattern_id"].to_dict()
    rows: list[dict[str, object]] = []
    for rank, pathway in enumerate(ranked.itertuples(index=False), start=1):
        contributors = (
            descriptive_edges.loc[descriptive_edges["pathway_id"] == pathway.pathway_id]
            .sort_values(
                ["mean_rpk_all_samples", "n_detected_samples", "species_key"],
                ascending=[False, False, True],
                kind="mergesort",
            )
            .drop_duplicates("species_key")
            .head(representative_taxa_n)
        )
        representative_taxa = [
            (
                f"{display_map.get(species_key, species_key)} "
                f"(P{int(pattern_map[species_key])})"
            )
            for species_key in contributors["species_key"]
        ]
        rows.append(
            {
                "display_rank": rank,
                "pathway_id": str(pathway.pathway_id),
                "pathway_name": str(pathway.pathway_name),
                "dominant_pattern": int(pathway.dominant_pattern),
                "dominant_pattern_mean_rpk_per_assigned_taxon": float(
                    pathway.dominant_pattern_mean_rpk_per_assigned_taxon
                ),
                "overall_stable_contributor_taxa_n": int(
                    pathway.stable_contributor_taxa_n
                ),
                "retained_taxa_n": int(len(universe)),
                "overall_stable_contributor_fraction": float(
                    pathway.stable_contributor_taxa_n / len(universe)
                ),
                "stable_pathway_species_edges": int(
                    pathway.stable_pathway_species_edges
                ),
                "overall_stable_detection_samples_median": float(
                    pathway.stable_detection_samples_median
                ),
                "overall_stable_detection_samples_min": int(
                    pathway.stable_detection_samples_min
                ),
                "overall_stable_detection_samples_max": int(
                    pathway.stable_detection_samples_max
                ),
                "total_mean_rpk_all_taxa": float(pathway.total_mean_rpk_all_taxa),
                "mean_rpk_per_retained_taxon": float(
                    pathway.mean_rpk_per_retained_taxon
                ),
                "pathway_detection_percentile": float(
                    pathway.pathway_detection_percentile
                ),
                "global_pathway_detection_threshold": float(
                    pathway.global_pathway_detection_threshold
                ),
                "zero_and_null_replacement_value": float(
                    pathway.zero_and_null_replacement_value
                ),
                "community_detected_samples": int(pathway.community_detected_samples),
                "community_mean_raw_abundance": float(
                    pathway.community_mean_raw_abundance
                ),
                "community_mean_ppm": float(pathway.community_mean_ppm),
                "community_mean_log10_ppm": float(pathway.community_mean_log10_ppm),
                "representative_taxa": "; ".join(representative_taxa),
            }
        )
    selection = pd.DataFrame(rows)
    audit = {
        "analysis_role": "primary descriptive five-pattern landscape",
        "descriptive_pathways_after_scope_and_detection": before_display_filter,
        "pathways_dropped_for_too_few_global_stable_contributors": (dropped_carriers),
        "minimum_global_stable_contributor_taxa": int(min_figure_carriers),
        "selection_rule": (
            "rank all descriptively eligible original MetaCyc pathways by "
            "overall stable contributor-taxon count, median contributor-edge "
            "detection samples, total all-sample mean RPK, then pathway ID; "
            "pattern labels, Fisher P values, BH q values and Bonferroni values "
            "are not used"
        ),
        "p_value_gate_used": False,
        "forced_quota_by_pattern": False,
        "displayed_pathways": int(len(selection)),
        "maximum_displayed_pathways": int(max_pathways),
    }
    return selection, audit


def rendered_text_bbox_collisions(
    text_artists: list[mpl.text.Text],
    renderer: object,
) -> list[dict[str, str]]:
    collisions: list[dict[str, str]] = []
    for index, left in enumerate(text_artists):
        left_box = left.get_window_extent(renderer=renderer)
        for right in text_artists[index + 1 :]:
            right_box = right.get_window_extent(renderer=renderer)
            if left_box.overlaps(right_box):
                collisions.append({"left": left.get_text(), "right": right.get_text()})
    return collisions


def audit_pdf_fonts(pdf_path: Path) -> dict[str, object]:
    """Reject Type 3 or unembedded PDF fonts when pdffonts is available."""
    executable = shutil.which("pdffonts")
    if executable is None:
        return {
            "status": "not_run",
            "reason": "pdffonts executable is unavailable",
        }
    completed = subprocess.run(
        [executable, str(pdf_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "pdffonts failed: " + (completed.stderr.strip() or completed.stdout.strip())
        )
    records = [
        line.strip() for line in completed.stdout.splitlines()[2:] if line.strip()
    ]
    if not records:
        raise RuntimeError("pdffonts returned no font records")
    type3 = [line for line in records if "Type 3" in line or "Type3" in line]
    unembedded = [
        line
        for line in records
        if len(line.split()) >= 5 and line.split()[-5].casefold() != "yes"
    ]
    if type3 or unembedded:
        raise RuntimeError(
            "PDF font audit failed: "
            + json.dumps(
                {"type3": type3, "unembedded": unembedded},
                ensure_ascii=False,
            )
        )
    return {
        "status": "passed",
        "font_records": records,
        "type3_fonts": 0,
        "unembedded_fonts": 0,
    }


def draw_pathway_figure(
    selection: pd.DataFrame,
    abundance: pd.DataFrame,
    pattern_counts: dict[str, int],
    analysis_sample_n: int,
    pathway_detection_percentile: float,
    pathway_detection_threshold: float,
    min_pathway_detected_samples: int,
    min_contribution_samples: int,
    figure_base: Path,
    preview_dpi: int,
) -> tuple[dict[str, str], dict[str, object]]:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    n_rows = int(len(selection))
    width_in = FIGURE_WIDTH_MM / 25.4
    height_in = min(
        FIGURE_MAX_HEIGHT_MM / 25.4,
        max(3.7, 2.05 + 0.55 * n_rows),
    )
    fig = plt.figure(figsize=(width_in, height_in), facecolor="white")
    ax = fig.add_axes([0.015, 0.18, 0.97, 0.70])
    ax.set_xlim(0, 100)
    ax.set_ylim(n_rows + 1.65, -1.25)
    ax.axis("off")

    heat_left = 48.0
    heat_cell = 4.15
    heat_right = heat_left + heat_cell * len(PATTERN_IDS)
    support_left = 76.0
    support_right = 91.0
    row_offset = 0.25

    heat_cmap = LinearSegmentedColormap.from_list(
        "oral_metacyc_z",
        [HEAT_LOW, HEAT_MID, HEAT_HIGH],
    )
    heat_norm = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0)

    retained_taxa_values = selection["retained_taxa_n"].astype(int).unique()
    if len(retained_taxa_values) != 1 or retained_taxa_values[0] <= 0:
        raise RuntimeError(
            "Descriptive figure selection has inconsistent retained-taxon counts"
        )
    retained_taxa_n = int(retained_taxa_values[0])

    def support_x(value: float) -> float:
        return support_left + (value / retained_taxa_n) * (support_right - support_left)

    header_texts = [
        ax.text(
            0.2,
            -0.90,
            "MetaCyc pathway / representative oral taxa",
            fontsize=6.0,
            fontweight="bold",
            color=TEXT,
            va="bottom",
        ),
        ax.text(
            (heat_left + heat_right) / 2,
            -0.90,
            "Mean RPK contribution per assigned taxon",
            fontsize=6.0,
            fontweight="bold",
            color=TEXT,
            ha="center",
            va="bottom",
        ),
        ax.text(
            (support_left + support_right) / 2,
            -0.90,
            "Stable contributor support",
            fontsize=6.0,
            fontweight="bold",
            color=TEXT,
            ha="center",
            va="bottom",
        ),
    ]
    for offset, pattern_id in enumerate(PATTERN_IDS):
        x_center = heat_left + heat_cell * (offset + 0.5)
        header_texts.append(
            ax.text(
                x_center,
                -0.48,
                f"P{pattern_id}\nn={pattern_counts[str(pattern_id)]}",
                ha="center",
                va="bottom",
                multialignment="center",
                linespacing=1.05,
                fontsize=5.0,
                color=TEXT,
            )
        )

    support_annotation_texts: list[mpl.text.Text] = []
    for row_index, row in enumerate(selection.itertuples(index=False)):
        y = float(row_index) + row_offset
        ax.plot([0, 100], [y + 0.49, y + 0.49], color="#ECEEF0", linewidth=0.55)
        display_name = str(row.pathway_name).strip() or str(row.pathway_id)
        ax.text(
            0.2,
            y - 0.12,
            textwrap.shorten(display_name, width=76, placeholder="…"),
            fontsize=5.7,
            fontweight="bold",
            color=TEXT,
            va="center",
        )
        detail = f"{row.pathway_id}  ·  {row.representative_taxa}"
        ax.text(
            0.2,
            y + 0.21,
            textwrap.shorten(detail, width=78, placeholder="…"),
            fontsize=5.0,
            fontstyle="italic",
            color=MUTED,
            va="center",
        )

        pathway_abundance = abundance.loc[
            abundance["pathway_id"] == row.pathway_id
        ].set_index("pattern_id")
        for offset, pattern_id in enumerate(PATTERN_IDS):
            value = float(pathway_abundance.loc[pattern_id, "row_zscore"])
            facecolor = heat_cmap(heat_norm(value)) if np.isfinite(value) else MISSING
            ax.add_patch(
                Rectangle(
                    (heat_left + offset * heat_cell, y - 0.34),
                    heat_cell,
                    0.68,
                    facecolor=facecolor,
                    edgecolor="white",
                    linewidth=0.55,
                )
            )

        carrier_x = support_x(float(row.overall_stable_contributor_taxa_n))
        ax.plot(
            [support_left, carrier_x],
            [y, y],
            color="#B7BBC0",
            linewidth=1.15,
            zorder=2,
        )
        ax.scatter(
            [carrier_x],
            [y],
            s=27,
            facecolor="#4C535A",
            edgecolor="#30343A",
            linewidth=0.45,
            zorder=3,
        )
        support_annotation_texts.append(
            ax.text(
                99.5,
                y,
                (
                    f"{int(row.overall_stable_contributor_taxa_n)}/"
                    f"{int(row.retained_taxa_n)} taxa"
                ),
                fontsize=5.0,
                color=TEXT,
                ha="right",
                va="center",
            )
        )

    axis_y = n_rows + 0.38
    tick_locator = MaxNLocator(nbins=4, integer=True)
    support_ticks = [
        int(value)
        for value in tick_locator.tick_values(0, retained_taxa_n)
        if 0 <= value <= retained_taxa_n
    ]
    ax.plot(
        [support_left, support_right],
        [axis_y, axis_y],
        color="#74787D",
        linewidth=0.65,
    )
    for tick in support_ticks:
        x_position = support_x(tick)
        ax.plot(
            [x_position, x_position],
            [axis_y, axis_y + 0.10],
            color="#74787D",
            linewidth=0.6,
        )
        ax.text(
            x_position,
            axis_y + 0.20,
            f"{tick:g}",
            ha="center",
            va="top",
            fontsize=5.0,
            color=TEXT,
        )
    ax.text(
        (support_left + support_right) / 2,
        axis_y + 0.58,
        "Stable contributing taxa (n)",
        ha="center",
        va="top",
        fontsize=5.2,
        color=TEXT,
    )

    gradient = np.linspace(-2.0, 2.0, 256)[None, :]
    ax.imshow(
        gradient,
        cmap=heat_cmap,
        norm=heat_norm,
        extent=(heat_left, heat_right, axis_y + 0.12, axis_y + 0.42),
        aspect="auto",
        interpolation="nearest",
    )
    for value, label in [(-2.0, "−2"), (0.0, "0"), (2.0, "+2")]:
        x_position = heat_left + ((value + 2.0) / 4.0) * (heat_right - heat_left)
        ax.text(
            x_position,
            axis_y + 0.52,
            label,
            ha="center",
            va="top",
            fontsize=5.0,
            color=TEXT,
        )
    ax.text(
        (heat_left + heat_right) / 2,
        axis_y + 0.88,
        "log1p mean RPK (row Z-score)",
        ha="center",
        va="top",
        fontsize=5.2,
        color=TEXT,
    )

    fig.text(
        0.018,
        0.071,
        (
            f"Complete-case cohort: n={analysis_sample_n:,} samples. Community "
            "pathways: raw abundance ≥ global positive-value "
            f"percentile {pathway_detection_percentile:g} threshold "
            f"({pathway_detection_threshold:.3g}) in "
            f"≥{min_pathway_detected_samples:,} samples; zeros/nulls were replaced "
            "before PPM and log10 transformation."
        ),
        ha="left",
        va="bottom",
        fontsize=5.0,
        color=MUTED,
    )
    fig.text(
        0.018,
        0.043,
        (
            f"Stable contributors: species-stratified RPK > 0 in "
            f"≥{min_contribution_samples:,} samples. Pathways are ranked by "
            "overall stable contributor-taxon count, detection support and mean "
            "RPK; no P-value is used for selection."
        ),
        ha="left",
        va="bottom",
        fontsize=5.0,
        color=MUTED,
    )
    fig.text(
        0.018,
        0.015,
        (
            "Heatmap: log1p mean RPK per assigned oral taxon, row Z-scored across "
            "P1–P5; header n denotes taxa, not participants. Representative taxa "
            "are descriptive contributors. Functional potential does not imply "
            "host activation."
        ),
        ha="left",
        va="bottom",
        fontsize=5.0,
        color=MUTED,
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    visible_text = [
        item
        for item in fig.findobj(mpl.text.Text)
        if item.get_visible() and item.get_text().strip()
    ]
    minimum_font = min(item.get_fontsize() for item in visible_text)
    if minimum_font < 5.0:
        raise RuntimeError(
            f"Figure contains source text below 5 pt: {minimum_font:.2f} pt"
        )
    header_collisions = rendered_text_bbox_collisions(header_texts, renderer)
    if header_collisions:
        raise RuntimeError(
            "Figure header collision audit failed: "
            + json.dumps(header_collisions, ensure_ascii=False)
        )
    support_right_display = float(ax.transData.transform((support_right, 0.0))[0])
    support_annotation_gaps_pt = [
        float(
            (artist.get_window_extent(renderer=renderer).x0 - support_right_display)
            / renderer.points_to_pixels(1.0)
        )
        for artist in support_annotation_texts
    ]
    if min(support_annotation_gaps_pt) < 1.5:
        raise RuntimeError(
            "Support-bar/text spacing audit failed: minimum gap is "
            f"{min(support_annotation_gaps_pt):.2f} pt; required gap is 1.50 pt"
        )

    figure_base.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "svg": figure_base.with_suffix(".svg"),
        "pdf": figure_base.with_suffix(".pdf"),
        "png": figure_base.with_suffix(".png"),
    }
    preview_error: str | None = None
    try:
        fig.savefig(paths["svg"], facecolor="white")
        fig.savefig(paths["pdf"], facecolor="white")
        try:
            fig.savefig(
                paths["png"],
                dpi=preview_dpi,
                facecolor="white",
            )
        except OSError as exc:
            preview_error = f"{type(exc).__name__}: {exc}"
            warnings.warn(
                "Vector exports succeeded but the optional PNG preview failed: "
                + preview_error
            )
    finally:
        plt.close(fig)

    pdf_font_audit = audit_pdf_fonts(paths["pdf"])
    svg_text = paths["svg"].read_text(encoding="utf-8")
    if "<text" not in svg_text:
        raise RuntimeError("SVG text was converted to paths")
    existing_paths = {
        key: str(value.resolve()) for key, value in paths.items() if value.exists()
    }
    qa = {
        "archetype": "single-axis quantitative pathway landscape",
        "evidence_role": "primary descriptive five-pattern functional landscape",
        "core_conclusion_boundary": (
            "stable original MetaCyc pathway contributions vary descriptively "
            "across five fixed oral-taxon patterns; the figure makes no "
            "inferential enrichment claim"
        ),
        "alignment_gate": "not applicable: one scientific plotting axis",
        "minimum_source_font_pt": float(minimum_font),
        "header_collision_audit": {
            "status": "passed",
            "n_collisions": int(len(header_collisions)),
        },
        "support_annotation_spacing_audit": {
            "status": "passed",
            "minimum_required_gap_pt": 1.5,
            "minimum_observed_gap_pt": float(min(support_annotation_gaps_pt)),
        },
        "final_width_mm": FIGURE_WIDTH_MM,
        "final_height_mm": float(height_in * 25.4),
        "editable_svg_text": True,
        "pdf_fonttype": 42,
        "pdf_font_audit": pdf_font_audit,
        "vector_exports": ["svg", "pdf"],
        "preview_export": (
            f"png_{preview_dpi}_dpi" if "png" in existing_paths else None
        ),
        "preview_error": preview_error,
        "authoritative_main_figure_formats": ["pdf", "svg"],
        "tiff_export": (
            "not generated: Nature main figures require editable vector artwork; "
            "PNG is an optional preview only"
        ),
        "statistics": {
            "complete_case_sample_n": int(analysis_sample_n),
            "sample_summary_unit": (
                "complete-case HUMAnN sample used to estimate each "
                "pathway-taxon all-sample mean RPK"
            ),
            "pattern_header_n_unit": "taxon in the frozen oral cluster universe",
            "heatmap_aggregation": (
                "sum stable contributor-taxon all-sample mean RPK within each "
                "pattern; divide by every frozen cluster taxon assigned to the "
                "pattern; log1p; row Z-score across P1-P5"
            ),
            "test_in_main_figure": "none; descriptive display",
            "multiple_testing_in_main_figure": "not applicable",
            "p_value_selection_gate": False,
        },
        "review_risks": [
            "row Z-scores suppress absolute between-pathway abundance differences",
            "the heatmap displays point summaries rather than participant-level distributions",
            "representative taxa and support bars are descriptive",
            "exploratory taxon-set enrichment is reported outside the main figure",
            "pathway functional potential does not establish host activation",
        ],
    }
    return existing_paths, qa


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oral-abundance-table", type=Path, required=True)
    parser.add_argument(
        "--microbe-pathway-table",
        "--humann-table",
        "--stratified-table",
        dest="microbe_pathway_table",
        type=Path,
        required=True,
        help=(
            "pathway_abundance_microbe_level.arrow containing "
            "species-stratified HUMAnN pathway contributions"
        ),
    )
    parser.add_argument(
        "--community-pathway-table",
        type=Path,
        required=True,
        help=(
            "pathway_abundance_pathway_level.arrow containing community-level "
            "HUMAnN pathway abundance"
        ),
    )
    parser.add_argument("--molecule-clusters", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--sample-key-columns",
        default="participant_id,research_stage",
        help=(
            "Comma-separated sample-identity columns shared by the oral Parquet "
            "and both HUMAnN Arrow files. Arrow fields may be stored pandas-index "
            "levels. The default participant_id,research_stage distinguishes "
            "sampling visits; use another biospecimen-level composite key only "
            "when dictated by the upstream sample manifest."
        ),
    )
    parser.add_argument("--omics-label", default="Oral microbiome")
    parser.add_argument(
        "--min-species-detected-samples",
        type=int,
        default=500,
        help=(
            "Reference-cohort prevalence audit threshold for frozen cluster taxa; "
            "it does not remove taxa or change P1-P5 membership"
        ),
    )
    parser.add_argument("--min-pathway-detected-samples", type=int, default=500)
    parser.add_argument("--min-contribution-samples", type=int, default=500)
    parser.add_argument("--pathway-detection-percentile", type=float, default=1.0)
    parser.add_argument(
        "--min-carrier-taxa",
        type=int,
        default=3,
        help="Minimum carrier taxa for the exploratory Fisher test family",
    )
    parser.add_argument(
        "--min-noncarrier-taxa",
        type=int,
        default=3,
        help="Minimum noncarrier taxa for the exploratory Fisher test family",
    )
    parser.add_argument(
        "--min-figure-carriers",
        type=int,
        default=2,
        help=(
            "Minimum stable contributor taxa across all five patterns for "
            "descriptive main-figure eligibility"
        ),
    )
    parser.add_argument("--min-overall-mapping-rate", type=float, default=0.50)
    parser.add_argument("--min-pattern-mapping-rate", type=float, default=0.25)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Exploratory global BH-FDR and strict Bonferroni sensitivity threshold",
    )
    parser.add_argument("--max-pathways", type=int, default=8)
    parser.add_argument("--representative-taxa", type=int, default=2)
    parser.add_argument("--batch-columns", type=int, default=256)
    parser.add_argument("--preview-dpi", type=int, default=450)
    parser.add_argument(
        "--scope-exclusion-regex",
        default=DEFAULT_SCOPE_EXCLUSION_REGEX,
        help=(
            "Optional prespecified extra scope exclusion. Empty by default; "
            "HUMAnN technical special features are always excluded."
        ),
    )
    parser.add_argument(
        "--hash-input-max-bytes",
        type=int,
        default=512 * 1024 * 1024,
    )
    args = parser.parse_args(argv)
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must lie within (0, 1)")
    if not 1 <= args.max_pathways <= 8:
        raise ValueError("--max-pathways must lie within [1, 8]")
    positive_names = [
        "min_species_detected_samples",
        "min_pathway_detected_samples",
        "min_contribution_samples",
        "min_carrier_taxa",
        "min_noncarrier_taxa",
        "min_figure_carriers",
        "representative_taxa",
        "batch_columns",
        "preview_dpi",
    ]
    for name in positive_names:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.preview_dpi < 300:
        raise ValueError("--preview-dpi must be at least 300")
    if not 0 < args.pathway_detection_percentile <= 100:
        raise ValueError("--pathway-detection-percentile must lie within (0, 100]")
    for name in ("min_overall_mapping_rate", "min_pattern_mapping_rate"):
        value = float(getattr(args, name))
        if not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must lie within [0, 1]")
    if args.scope_exclusion_regex:
        re.compile(args.scope_exclusion_regex)
    args.sample_key_columns = tuple(
        dict.fromkeys(
            column.strip()
            for column in str(args.sample_key_columns).split(",")
            if column.strip()
        )
    )
    if not args.sample_key_columns:
        raise ValueError("--sample-key-columns must contain at least one column")
    for path in (
        args.oral_abundance_table,
        args.microbe_pathway_table,
        args.community_pathway_table,
        args.molecule_clusters,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.oral_abundance_table.name.casefold() == (
        "070-oral_microbiome_imputed.parquet"
    ):
        raise ValueError(
            "070-oral_microbiome_imputed.parquet is explicitly prohibited; "
            "provide raw oral_microbiome_with_metadata.parquet"
        )
    for role, path in (
        ("community pathway abundance", args.community_pathway_table),
        ("microbe-stratified pathway abundance", args.microbe_pathway_table),
    ):
        filename = path.name.casefold()
        if "gene_families" in filename or "pathway_coverage" in filename:
            raise ValueError(
                f"{path.name!r} is not a pathway-abundance input for {role}"
            )
    if "microbe_level" in args.community_pathway_table.name.casefold():
        raise ValueError(
            "--community-pathway-table received a microbe-level file; use "
            "pathway_abundance_pathway_level.arrow"
        )
    if "pathway_level" in args.microbe_pathway_table.name.casefold():
        raise ValueError(
            "--microbe-pathway-table received a pathway-level file; use "
            "pathway_abundance_microbe_level.arrow"
        )
    args.run_id = args.run_id or uuid.uuid4().hex
    return args


def initial_manifest(args: argparse.Namespace) -> dict[str, object]:
    script_path = Path(__file__).resolve()
    manifest: dict[str, object] = {
        "schema_version": 8,
        "analysis_version": ANALYSIS_VERSION,
        "run_id": str(args.run_id),
        "status": "running",
        "skip_reason": None,
        "started_utc": utc_now(),
        "completed_utc": None,
        "analysis_role": (
            "primary_descriptive_oral_taxon_pathway_landscape_with_"
            "exploratory_supplementary_enrichment"
        ),
        "core_conclusion_boundary": (
            "describe how stable original HUMAnN/MetaCyc pathway contributions "
            "are distributed across five precomputed oral-taxon patterns; no "
            "P-value controls main-figure inclusion and no host activation or "
            "causal interpretation is made"
        ),
        "analysis_method": {
            "upstream_profiler_provenance": {
                "status": "not inferred from abundance matrices",
                "report_separately": [
                    "HUMAnN version",
                    "MetaPhlAn version and database release",
                    "UniRef database release",
                ],
                "rationale": (
                    "software/database versions cannot be reconstructed "
                    "reliably from column names and are therefore never guessed"
                ),
            },
            "source_roles": {
                "oral_abundance_table": (
                    "un-imputed MetaPhlAn species abundance used as the candidate "
                    "reference cohort and to audit, but not refilter, the frozen "
                    "cluster taxa"
                ),
                "community_pathway_table": (
                    "pathway_abundance_pathway_level.arrow; community-level "
                    "abundance used for pathway preprocessing"
                ),
                "microbe_pathway_table": (
                    "pathway_abundance_microbe_level.arrow; species-stratified "
                    "RPK used for taxon contribution"
                ),
                "cluster_table": (
                    "fixed P1-P5 labels; clustering is never recomputed here"
                ),
            },
            "cohort_consistency_gate": (
                "strict sample-key matching across the oral MetaPhlAn Parquet and "
                "both HUMAnN views; the final cohort is their complete-case "
                "intersection, samples absent from any source and source-only rows "
                "are excluded without zero imputation, and positional or row-count "
                "matching is prohibited"
            ),
            "sample_key_columns": list(args.sample_key_columns),
            "pathway_unit": (
                "original HUMAnN/MetaCyc pathway ID parsed from a "
                "species-stratified Arrow column"
            ),
            "foregrounds": "oral taxa assigned to each of P1-P5",
            "background": (
                "the complete union of oral taxa in the fixed cluster table; "
                "no post-clustering prevalence filter changes membership"
            ),
            "species_prevalence": (
                "raw, un-imputed MetaPhlAn abundance > 0 in at least "
                f"{args.min_species_detected_samples} samples, reported as a "
                "current-cohort audit only"
            ),
            "community_pathway_preprocessing": {
                "threshold": (
                    f"global {args.pathway_detection_percentile:g}th percentile "
                    "of all positive, non-zero community pathway abundances "
                    "after HUMAnN special-feature and prespecified scope exclusions"
                ),
                "zero_and_null_policy": (
                    "replace with half the global detection threshold only for "
                    "normalization; neither value counts as detected"
                ),
                "normalization": (
                    "sum-normalize per sample, multiply by 1,000,000 (PPM), "
                    "then log10 transform"
                ),
                "pathway_prevalence": (
                    "raw community abundance at or above the global detection "
                    f"threshold in at least {args.min_pathway_detected_samples} "
                    "samples"
                ),
            },
            "pathway_species_occurrence": (
                f"species-stratified RPK > 0 in at least "
                f"{args.min_contribution_samples} samples"
            ),
            "independent_filtering": {
                "descriptive_landscape": (
                    "scope-qualified, community-prevalent original MetaCyc "
                    "pathways with at least one taxon contribution edge meeting "
                    "the sample-detection threshold"
                ),
                "main_figure_minimum_global_stable_contributor_taxa": (
                    args.min_figure_carriers
                ),
                "exploratory_enrichment": {
                    "minimum_carrier_taxa": args.min_carrier_taxa,
                    "minimum_noncarrier_taxa": args.min_noncarrier_taxa,
                },
                "rationale": (
                    "descriptive ranking uses pathway-wide support and abundance; "
                    "exploratory-test filters use pathway-wide carriage counts; "
                    "neither uses pattern-specific P values"
                ),
            },
            "test": (
                "exploratory supplementary one-sided Fisher exact test for each "
                "eligible pathway x pattern, pattern versus the other four"
            ),
            "multiple_testing": (
                "Benjamini-Hochberg FDR across the complete eligible pathway x "
                "five-pattern family; Bonferroni is a strict sensitivity analysis; "
                "neither gates the primary figure"
            ),
            "effect_size": (
                "Haldane-Anscombe corrected log2 odds ratio and approximate "
                "Wald 95% confidence interval; exact Fisher p-value"
            ),
            "heatmap": (
                "sum all threshold-qualified mean species-stratified RPK for a "
                "pathway within a pattern, divide by every assigned oral taxon "
                "in that pattern, log1p transform, then row Z-score across P1-P5"
            ),
            "representative_taxa": (
                "stable contributors with the largest all-sample mean RPK across "
                "all five patterns, labelled with their fixed pattern; descriptive "
                "only"
            ),
            "explicitly_not_used": [
                "070-oral_microbiome_imputed.parquet",
                "functional_theme_dictionary.tsv",
                "regex-defined biological modules",
                "outputs/oral_function_patterns",
                "age, sex, BMI, or other metadata columns",
                "OLS metabolic-outcome models because no metabolic outcome is "
                "part of this cluster-pathway question",
                "cluster-centre files",
                "host-pathway activation claims",
                "P-value or q-value gate for primary-figure inclusion",
                "forced pathway quota for any pattern",
            ],
        },
        "thresholds": {
            "min_species_detected_samples": args.min_species_detected_samples,
            "min_pathway_detected_samples": args.min_pathway_detected_samples,
            "min_contribution_samples": args.min_contribution_samples,
            "pathway_detection_percentile": args.pathway_detection_percentile,
            "min_carrier_taxa": args.min_carrier_taxa,
            "min_noncarrier_taxa": args.min_noncarrier_taxa,
            "min_figure_carriers": args.min_figure_carriers,
            "min_overall_mapping_rate": args.min_overall_mapping_rate,
            "min_pattern_mapping_rate": args.min_pattern_mapping_rate,
            "exploratory_alpha": args.alpha,
            "max_pathways": args.max_pathways,
            "representative_taxa": args.representative_taxa,
            "scope_exclusion_regex": args.scope_exclusion_regex,
        },
        "inputs": {
            "oral_abundance_table": file_record(
                args.oral_abundance_table,
                hash_limit=args.hash_input_max_bytes,
            ),
            "community_pathway_table": file_record(
                args.community_pathway_table,
                hash_limit=args.hash_input_max_bytes,
            ),
            "microbe_pathway_table": file_record(
                args.microbe_pathway_table,
                hash_limit=args.hash_input_max_bytes,
            ),
            "molecule_clusters": file_record(args.molecule_clusters),
            "analysis_script": file_record(script_path),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pyarrow": pa.__version__,
                "scipy": scipy.__version__,
                "matplotlib": mpl.__version__,
                "pillow": package_version("pillow"),
            },
        },
        "outputs": {},
    }
    fingerprint_payload = {
        "analysis_script_sha256": manifest["inputs"]["analysis_script"]["sha256"],
        "cluster_sha256": manifest["inputs"]["molecule_clusters"]["sha256"],
        "oral_abundance_sha256": manifest["inputs"]["oral_abundance_table"]["sha256"],
        "oral_abundance_size": manifest["inputs"]["oral_abundance_table"]["size_bytes"],
        "oral_abundance_mtime": manifest["inputs"]["oral_abundance_table"][
            "modified_utc"
        ],
        "community_pathway_size": manifest["inputs"]["community_pathway_table"][
            "size_bytes"
        ],
        "community_pathway_mtime": manifest["inputs"]["community_pathway_table"][
            "modified_utc"
        ],
        "microbe_pathway_size": manifest["inputs"]["microbe_pathway_table"][
            "size_bytes"
        ],
        "microbe_pathway_mtime": manifest["inputs"]["microbe_pathway_table"][
            "modified_utc"
        ],
        "thresholds": manifest["thresholds"],
        "sample_key_columns": list(args.sample_key_columns),
        "analysis_version": ANALYSIS_VERSION,
    }
    manifest["analysis_fingerprint"] = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    return manifest


def finish_manifest(
    manifest_path: Path,
    manifest: dict[str, object],
    status: str,
    reason: str | None = None,
) -> None:
    manifest["status"] = status
    manifest["skip_reason"] = reason
    manifest["completed_utc"] = utc_now()
    write_json(manifest_path, manifest)


def run_analysis(
    args: argparse.Namespace,
    manifest: dict[str, object],
    manifest_path: Path,
) -> None:
    try:
        (
            oral_sample_rows,
            community_sample_rows,
            microbe_sample_rows,
            sample_alignment_table,
            sample_alignment_audit,
        ) = align_analysis_samples(
            args.oral_abundance_table,
            args.community_pathway_table,
            args.microbe_pathway_table,
            args.sample_key_columns,
        )
    except SampleAlignmentError as exc:
        manifest["sample_alignment"] = {
            "status": "skipped",
            "sample_key_columns": list(args.sample_key_columns),
            "skip_reason": str(exc),
            "positional_matching_used": False,
        }
        finish_manifest(manifest_path, manifest, "skipped", str(exc))
        return

    sample_alignment_file = args.output_dir / "sample_alignment_audit.csv"
    sample_alignment_table.to_csv(sample_alignment_file, index=False)
    manifest["sample_alignment"] = sample_alignment_audit
    manifest["outputs"]["sample_alignment_audit"] = str(sample_alignment_file.resolve())
    if sample_alignment_audit["status"] != "completed":
        finish_manifest(
            manifest_path,
            manifest,
            "skipped",
            str(sample_alignment_audit["skip_reason"]),
        )
        return

    universe, species_prevalence_audit, universe_audit = load_oral_universe(
        args.molecule_clusters,
        args.oral_abundance_table,
        args.omics_label,
        args.min_species_detected_samples,
        oral_sample_rows,
    )
    universe_file = args.output_dir / "oral_taxon_pattern_universe.csv"
    species_audit_file = args.output_dir / "oral_species_preprocessing_audit.csv"
    universe.to_csv(universe_file, index=False)
    species_prevalence_audit.to_csv(species_audit_file, index=False)
    manifest["cluster_universe"] = universe_audit
    manifest["outputs"]["oral_taxon_pattern_universe"] = str(universe_file.resolve())
    manifest["outputs"]["oral_species_preprocessing_audit"] = str(
        species_audit_file.resolve()
    )
    if universe_audit["missing_patterns"]:
        finish_manifest(
            manifest_path,
            manifest,
            "skipped",
            "The current oral cluster table does not contain all five patterns: "
            + ", ".join(map(str, universe_audit["missing_patterns"])),
        )
        return

    microbe_schema_names = read_arrow_schema(args.microbe_pathway_table)
    community_schema_names = read_arrow_schema(args.community_pathway_table)
    feature_info, community_info, humann_species, schema_audit = parse_humann_columns(
        microbe_schema_names,
        community_schema_names,
        set(universe["species_key"]),
    )
    manifest["humann_schema"] = schema_audit
    coverage = taxonomy_coverage(universe, humann_species)
    manifest["species_mapping_coverage"] = coverage
    if coverage["overall_rate"] < args.min_overall_mapping_rate:
        finish_manifest(
            manifest_path,
            manifest,
            "skipped",
            "Overall oral-taxon mapping rate is below the prespecified threshold: "
            f"{coverage['overall_rate']:.3f} < "
            f"{args.min_overall_mapping_rate:.3f}",
        )
        return
    low_patterns = [
        pattern_id
        for pattern_id, rate in coverage["pattern_rates"].items()
        if rate < args.min_pattern_mapping_rate
    ]
    if low_patterns:
        finish_manifest(
            manifest_path,
            manifest,
            "skipped",
            "Pattern-specific oral-taxon mapping rate is below threshold for: "
            + ", ".join(f"P{pattern_id}" for pattern_id in low_patterns),
        )
        return
    if feature_info.empty:
        finish_manifest(
            manifest_path,
            manifest,
            "skipped",
            "No species-stratified HUMAnN columns map to the current oral taxa",
        )
        return

    scope_audit, scoped_columns, scope_summary = build_scope_audit(
        feature_info,
        args.scope_exclusion_regex,
    )
    scope_file = args.output_dir / "metacyc_pathway_scope_audit.csv"
    scope_audit.to_csv(scope_file, index=False)
    manifest["pathway_scope_filter"] = scope_summary
    manifest["outputs"]["metacyc_pathway_scope_audit"] = str(scope_file.resolve())
    if scoped_columns.empty:
        finish_manifest(
            manifest_path,
            manifest,
            "skipped",
            "All matched HUMAnN pathways were excluded by the prespecified "
            "scope filter",
        )
        return

    compiled_scope = (
        re.compile(args.scope_exclusion_regex) if args.scope_exclusion_regex else None
    )
    community_scope_audit = community_info.copy()
    community_scope_audit["scope_exclusion_reason"] = [
        pathway_scope_reason(pathway_id, pathway_name, compiled_scope)
        for pathway_id, pathway_name in zip(
            community_scope_audit["pathway_id"],
            community_scope_audit["pathway_name"],
        )
    ]
    community_scope_audit["scope_status"] = np.where(
        community_scope_audit["scope_exclusion_reason"].isna(),
        "included",
        "excluded",
    )
    community_scope_file = args.output_dir / "metacyc_community_pathway_scope_audit.csv"
    community_scope_audit.to_csv(community_scope_file, index=False)
    manifest["outputs"]["metacyc_community_pathway_scope_audit"] = str(
        community_scope_file.resolve()
    )
    scoped_community = community_scope_audit.loc[
        community_scope_audit["scope_status"] == "included",
        ["column", "pathway_id", "pathway_name"],
    ].copy()
    manifest["community_pathway_scope_filter"] = {
        "recognized_before_scope_filter": int(len(community_scope_audit)),
        "retained_for_threshold_and_normalization": int(len(scoped_community)),
        "excluded": int((community_scope_audit["scope_status"] == "excluded").sum()),
    }
    if scoped_community.empty:
        finish_manifest(
            manifest_path,
            manifest,
            "skipped",
            "No MetaCyc pathway ID in the pathway-level Arrow matches a "
            "pathway ID in the microbe-level Arrow after HUMAnN technical "
            "feature exclusions; verify that both files come from the same "
            "HUMAnN run and database release",
        )
        return

    community_summary, community_preprocessing = preprocess_community_pathways(
        args.community_pathway_table,
        scoped_community,
        args.pathway_detection_percentile,
        args.min_pathway_detected_samples,
        args.batch_columns,
        community_sample_rows,
    )
    manifest["community_pathway_preprocessing"] = community_preprocessing
    community_file = args.output_dir / "metacyc_community_pathway_preprocessing.csv"
    community_summary.to_csv(community_file, index=False)
    manifest["outputs"]["metacyc_community_pathway_preprocessing"] = str(
        community_file.resolve()
    )
    if community_preprocessing.get("status") != "completed":
        finish_manifest(
            manifest_path,
            manifest,
            "skipped",
            str(community_preprocessing.get("skip_reason")),
        )
        return
    community_samples = int(community_preprocessing["samples"])
    analysis_samples = int(sample_alignment_audit["analysis_samples"])
    oral_samples = int(universe_audit["oral_abundance_samples"])
    if community_samples != analysis_samples or oral_samples != analysis_samples:
        raise RuntimeError(
            "Sample-ID selectors produced inconsistent analysis-cohort sizes: "
            f"oral={oral_samples}, community={community_samples}, "
            f"alignment={analysis_samples}"
        )
    if community_samples < args.min_pathway_detected_samples:
        finish_manifest(
            manifest_path,
            manifest,
            "skipped",
            f"Only {community_samples} HUMAnN samples are available, below "
            "the community-pathway prevalence threshold of "
            f"{args.min_pathway_detected_samples}",
        )
        return

    pair_stats, pair_audit = summarize_humann_pairs(
        args.microbe_pathway_table,
        scoped_columns,
        args.batch_columns,
        microbe_sample_rows,
    )
    manifest["humann_pair_summary"] = pair_audit
    all_pairs_file = args.output_dir / "metacyc_pathway_species_all.csv"
    pair_stats.to_csv(all_pairs_file, index=False)
    manifest["outputs"]["metacyc_pathway_species_all"] = str(all_pairs_file.resolve())
    if int(pair_audit["samples"]) != analysis_samples:
        raise RuntimeError(
            "Sample-ID selector produced an inconsistent microbe-level cohort: "
            f"microbe={pair_audit['samples']}, alignment={analysis_samples}"
        )
    if pair_audit["samples"] < args.min_contribution_samples:
        finish_manifest(
            manifest_path,
            manifest,
            "skipped",
            f"Only {pair_audit['samples']} HUMAnN samples are available, below "
            "the species-contribution occurrence threshold of "
            f"{args.min_contribution_samples}",
        )
        return

    (
        filter_audit,
        descriptive_edges,
        exploratory_edges,
    ) = build_pathway_filter_audit(
        scope_audit,
        community_summary,
        pair_stats,
        args.min_contribution_samples,
        len(universe),
        args.min_carrier_taxa,
        args.min_noncarrier_taxa,
    )
    filter_file = args.output_dir / "metacyc_pathway_filter_audit.csv"
    filter_audit.to_csv(filter_file, index=False)
    descriptive_edge_file = (
        args.output_dir / "metacyc_pathway_species_descriptive_eligible.csv"
    )
    exploratory_edge_file = (
        args.output_dir / "metacyc_pathway_species_exploratory_eligible.csv"
    )
    descriptive_edges.to_csv(descriptive_edge_file, index=False)
    exploratory_edges.to_csv(exploratory_edge_file, index=False)
    descriptive_pathways = filter_audit.loc[
        filter_audit["descriptive_filter_status"] == "included_in_descriptive_landscape"
    ]
    exploratory_pathways = filter_audit.loc[
        filter_audit["exploratory_filter_status"]
        == "included_in_exploratory_test_family"
    ]
    manifest["independent_filtering"] = {
        "descriptive_pathways": int(len(descriptive_pathways)),
        "descriptive_stable_pathway_species_edges": int(len(descriptive_edges)),
        "exploratory_pathways_in_global_test_family": int(len(exploratory_pathways)),
        "exploratory_pathway_species_edges": int(len(exploratory_edges)),
        "descriptive_status_counts": {
            str(key): int(value)
            for key, value in filter_audit["descriptive_filter_status"]
            .value_counts()
            .items()
        },
        "exploratory_status_counts": {
            str(key): int(value)
            for key, value in filter_audit["exploratory_filter_status"]
            .value_counts()
            .items()
        },
    }
    manifest["outputs"]["metacyc_pathway_filter_audit"] = str(filter_file.resolve())
    manifest["outputs"]["metacyc_pathway_species_descriptive_eligible"] = str(
        descriptive_edge_file.resolve()
    )
    manifest["outputs"]["metacyc_pathway_species_exploratory_eligible"] = str(
        exploratory_edge_file.resolve()
    )
    if descriptive_edges.empty:
        finish_manifest(
            manifest_path,
            manifest,
            "skipped",
            "No original MetaCyc pathway satisfies the prespecified detection, "
            "community-prevalence, and stable taxon-contribution criteria for "
            "the descriptive landscape",
        )
        return

    pathway_names = (
        filter_audit.set_index("pathway_id")["pathway_name"].astype(str).to_dict()
    )
    abundance = pathway_abundance_table(
        descriptive_edges,
        universe,
        pathway_names,
    )
    abundance_file = (
        args.output_dir / "metacyc_pathway_descriptive_abundance_by_pattern.csv"
    )
    abundance.to_csv(abundance_file, index=False)
    manifest["outputs"]["metacyc_pathway_descriptive_abundance_by_pattern"] = str(
        abundance_file.resolve()
    )

    if exploratory_edges.empty:
        enrichment = pd.DataFrame(
            columns=[
                "pathway_id",
                "pathway_name",
                "pattern_id",
                "foreground_with",
                "foreground_without",
                "nonforeground_with",
                "nonforeground_without",
                "foreground_n",
                "nonforeground_n",
                "universe_n",
                "exact_odds_ratio",
                "log2_odds_ratio_ha",
                "ci95_low_log2_ha",
                "ci95_high_log2_ha",
                "p_fisher_greater",
                "p_bonferroni_global",
                "q_bh_global",
                "exploratory_positive_bh",
                "strict_positive_bonferroni_sensitivity",
                "test",
                "analysis_role",
                "multiple_testing",
                "effect_interval",
            ]
        )
    else:
        enrichment = pathway_enrichment_table(
            exploratory_edges,
            universe,
            pathway_names,
            community_summary,
            args.alpha,
        )
    enrichment_file = args.output_dir / "metacyc_pathway_enrichment_exploratory_all.csv"
    enrichment.to_csv(enrichment_file, index=False)
    manifest["exploratory_statistical_tests"] = {
        "analysis_role": "exploratory supplementary",
        "controls_primary_figure_selection": False,
        "eligible_pathways": int(enrichment["pathway_id"].nunique()),
        "patterns_per_pathway": int(len(PATTERN_IDS)),
        "global_test_count": int(len(enrichment)),
        "positive_pathways_global_bh_fdr": int(
            enrichment.loc[
                enrichment["exploratory_positive_bh"].astype(bool),
                "pathway_id",
            ].nunique()
        ),
        "positive_pathways_global_bonferroni_sensitivity": int(
            enrichment.loc[
                enrichment["strict_positive_bonferroni_sensitivity"].astype(bool),
                "pathway_id",
            ].nunique()
        ),
        "alpha": float(args.alpha),
        "test": "one-sided Fisher exact test (greater)",
        "correction": (
            "Benjamini-Hochberg FDR across all eligible pathway-pattern tests; "
            "Bonferroni retained as a strict sensitivity analysis"
        ),
        "effect": (
            "Haldane-Anscombe corrected log2 odds ratio with approximate Wald "
            "95% confidence interval"
        ),
    }
    manifest["outputs"]["metacyc_pathway_enrichment_exploratory_all"] = str(
        enrichment_file.resolve()
    )

    selection, selection_audit = build_descriptive_figure_selection(
        abundance,
        descriptive_edges,
        universe,
        community_summary,
        args.max_pathways,
        args.min_figure_carriers,
        args.representative_taxa,
    )
    selection_file = (
        args.output_dir / "metacyc_pathway_descriptive_figure_selection.csv"
    )
    selection.to_csv(selection_file, index=False)
    manifest["figure_selection"] = selection_audit
    manifest["outputs"]["metacyc_pathway_descriptive_figure_selection"] = str(
        selection_file.resolve()
    )
    if selection.empty:
        finish_manifest(
            manifest_path,
            manifest,
            "skipped",
            "No descriptively eligible original MetaCyc pathway has the "
            "prespecified minimum number of global stable contributor taxa",
        )
        return

    source_data = abundance.loc[
        abundance["pathway_id"].isin(selection["pathway_id"])
    ].merge(
        selection[
            [
                "pathway_id",
                "display_rank",
                "dominant_pattern",
                "dominant_pattern_mean_rpk_per_assigned_taxon",
                "overall_stable_contributor_taxa_n",
                "retained_taxa_n",
                "overall_stable_contributor_fraction",
                "stable_pathway_species_edges",
                "overall_stable_detection_samples_median",
                "overall_stable_detection_samples_min",
                "overall_stable_detection_samples_max",
                "total_mean_rpk_all_taxa",
                "mean_rpk_per_retained_taxon",
                "pathway_detection_percentile",
                "global_pathway_detection_threshold",
                "zero_and_null_replacement_value",
                "community_detected_samples",
                "community_mean_raw_abundance",
                "community_mean_ppm",
                "community_mean_log10_ppm",
                "representative_taxa",
            ]
        ],
        on="pathway_id",
        how="left",
        validate="many_to_one",
    )
    source_data_file = (
        args.output_dir / "metacyc_pathway_descriptive_figure_source_data.csv"
    )
    source_data.to_csv(source_data_file, index=False)
    manifest["outputs"]["metacyc_pathway_descriptive_figure_source_data"] = str(
        source_data_file.resolve()
    )

    figure_base = args.figure_dir / (
        "oral_metacyc_descriptive_five_pattern_landscape_"
        + str(manifest["analysis_fingerprint"])[:10]
    )
    figure_paths, figure_qa = draw_pathway_figure(
        selection,
        abundance,
        universe_audit["pattern_counts"],
        analysis_samples,
        args.pathway_detection_percentile,
        float(community_preprocessing["detection_threshold"]),
        args.min_pathway_detected_samples,
        args.min_contribution_samples,
        figure_base,
        args.preview_dpi,
    )
    qa_file = args.output_dir / "metacyc_pathway_descriptive_figure_qa.json"
    write_json(qa_file, figure_qa)
    manifest["figure"] = {
        "analysis_role": "primary descriptive five-pattern landscape",
        "files": figure_paths,
        "qa": str(qa_file.resolve()),
        "exploratory_enrichment_table": str(enrichment_file.resolve()),
    }
    manifest["outputs"]["metacyc_pathway_descriptive_figure_qa"] = str(
        qa_file.resolve()
    )
    finish_manifest(manifest_path, manifest, "completed")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "oral_metacyc_pathway_analysis_manifest.json"
    manifest = initial_manifest(args)
    write_json(manifest_path, manifest)
    try:
        run_analysis(args, manifest, manifest_path)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["skip_reason"] = None
        manifest["completed_utc"] = utc_now()
        manifest["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(manifest_path, manifest)
        raise

    refreshed = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"Run ID: {refreshed['run_id']}")
    print(f"Status: {refreshed['status']}")
    if refreshed.get("skip_reason"):
        print(f"Skip reason: {refreshed['skip_reason']}")
    print(f"Manifest: {manifest_path.resolve()}")
    for name, path in refreshed.get("outputs", {}).items():
        print(f"{name}: {path}")
    for name, path in refreshed.get("figure", {}).get("files", {}).items():
        print(f"figure_{name}: {path}")


if __name__ == "__main__":
    main()
