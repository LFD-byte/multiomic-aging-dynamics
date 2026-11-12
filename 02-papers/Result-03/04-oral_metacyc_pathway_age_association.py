#!/usr/bin/env python3
"""Participant-level oral MetaCyc pathway associations with chronological age.

This is an independent downstream analysis of community-level HUMAnN pathway
abundance. It does not use the five oral-taxon patterns and does not import or
rerun molecular clustering.

The preprocessing and association framework is adapted from Keshet et al.,
Nature Communications (2024), doi:10.1038/s41467-024-53832-y:

1. exclude HUMAnN special features (UNMAPPED, UNINTEGRATED, UNGROUPED and
   UNCLASSIFIED);
2. retain samples with at least a prespecified number of positive eligible
   pathways (10 by default);
3. interpret sparse Arrow null cells within retained profiles as pathway
   non-detection, recode them to zero and record every change in an audit;
4. define one global detection threshold as the first percentile of all
   positive, non-zero community-pathway abundances;
5. replace zeroes with half that threshold, sum-normalize each sample, scale to
   parts per million and apply log10 transformation;
6. retain pathways detected at or above the threshold in at least 500 samples;
7. fit one two-sided OLS model per retained pathway and correct the complete
   pathway test family using Bonferroni, with BH-FDR reported additionally.

The adaptation is deliberate rather than a text substitution. Chronological
age is the exposure of interest, so the primary model is

    standardized log10(PPM pathway) ~ age per 10 years + sex.

HC3 heteroskedasticity-robust confidence intervals and P values are used by
default. A paper-direction sensitivity model,

    age ~ standardized log10(PPM pathway) + sex,

is also exported. Age is bounded by prespecified cohort eligibility limits and
is not subjected to the paper's metabolic-outcome 8-SD deletion/5-SD clipping.

The primary figure combines (i) sex-adjusted, age-bin descriptive means shown
as within-pathway row Z-scores and (ii) continuous-age OLS effects with 95% CIs.
Age bins are visualization only and never define the inferential test. By
default, figure rows must pass the paper-compatible Bonferroni threshold. If no
pathway passes, the complete association table is retained and the figure is
skipped rather than populated with selected non-significant results.

Only the oral metadata Parquet and community-level pathway Arrow are required.
The microbe-level HUMAnN Arrow is intentionally not required because this
analysis does not estimate taxon contributions. Samples are aligned by an
explicit unique key; positional matching is prohibited.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import sys
import textwrap
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
import pyarrow.parquet as parquet
import scipy
import statsmodels.api as sm
import statsmodels
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator


ANALYSIS_VERSION = "1.1.0-sample-qc-and-sparse-null-recoding"
FIGURE_WIDTH_MM = 183.0
FIGURE_MAX_HEIGHT_MM = 170.0
TEXT = "#222222"
MUTED = "#666666"
HEAT_LOW = "#0072B2"
HEAT_MID = "#F7F7F7"
HEAT_HIGH = "#D55E00"
MISSING = "#D9D9D9"
HUMANN_SPECIAL_FEATURES = {
    "UNMAPPED",
    "UNINTEGRATED",
    "UNGROUPED",
    "UNCLASSIFIED",
}
DEFAULT_ARROW_METADATA_COLUMNS = {
    "participant_id",
    "research_stage",
    "cohort",
    "array_index",
    "collection_date",
    "age",
    "sex",
    "bmi",
    "index",
    "__index_level_0__",
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
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


class AnalysisSkip(RuntimeError):
    """A prespecified condition prevents a defensible main result."""


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


def require_matplotlib_panel_alignment(
    fig: mpl.figure.Figure,
    axes: dict[str, mpl.axes.Axes],
    json_out: Path,
    tolerance_pt: float = 1.5,
) -> dict[str, object]:
    """Block export when the three row-aligned plot areas do not align.

    This small, self-contained gate is used because the remote analysis server
    does not necessarily have the local manuscript-figure QA package. It checks
    the final rendered plot-area top and bottom edges in physical points and
    persists the measurements for review.
    """

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
    passed = top_spread_pt <= tolerance_pt and bottom_spread_pt <= tolerance_pt
    payload = {
        "status": "PASS" if passed else "FIX BEFORE DELIVERY",
        "tolerance_pt": float(tolerance_pt),
        "top_edge_spread_pt": float(top_spread_pt),
        "bottom_edge_spread_pt": float(bottom_spread_pt),
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
            f"Final rendered panel alignment exceeds {tolerance_pt:g} pt: "
            f"top={top_spread_pt:.3f}, bottom={bottom_spread_pt:.3f}"
        )
    return payload


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
            stat.st_mtime, tz=timezone.utc
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
    if exclusion_regex and exclusion_regex.search(
        f"{pathway_id} {pathway_name}".strip()
    ):
        return "excluded_prespecified_scope_regex"
    return None


def discover_community_pathways(
    schema: pa.Schema,
    sample_key_columns: Iterable[str],
    extra_metadata_columns: Iterable[str],
    exclusion_pattern: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = set(DEFAULT_ARROW_METADATA_COLUMNS)
    metadata.update(sample_key_columns)
    metadata.update(extra_metadata_columns)
    exclusion_regex = re.compile(exclusion_pattern) if exclusion_pattern else None
    records: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for field in schema:
        column = str(field.name)
        pathway_id, pathway_name = split_pathway_label(column)
        reason: str | None = None
        if column in metadata:
            reason = "excluded_metadata_column"
        elif "|" in column:
            reason = "excluded_taxon_stratified_column"
        elif not (
            pa.types.is_integer(field.type)
            or pa.types.is_floating(field.type)
            or pa.types.is_decimal(field.type)
        ):
            reason = "excluded_non_numeric_column"
        else:
            reason = pathway_scope_reason(
                pathway_id,
                pathway_name,
                exclusion_regex,
            )
        status = "included" if reason is None else "excluded"
        audit_rows.append(
            {
                "column": column,
                "arrow_type": str(field.type),
                "pathway_id": pathway_id,
                "pathway_name": pathway_name,
                "scope_status": status,
                "scope_reason": reason or "",
            }
        )
        if reason is not None:
            continue
        if pathway_id in seen_ids:
            raise ValueError(
                "Community-level Arrow contains more than one numeric column "
                f"for MetaCyc pathway ID {pathway_id!r}; aliases cannot be "
                "collapsed without an explicit biological rule"
            )
        seen_ids.add(pathway_id)
        records.append(
            {
                "column": column,
                "pathway_id": pathway_id,
                "pathway_name": pathway_name,
            }
        )
    pathways = pd.DataFrame(
        records,
        columns=["column", "pathway_id", "pathway_name"],
    )
    audit = pd.DataFrame(audit_rows)
    if pathways.empty:
        raise AnalysisSkip(
            "No numeric community-level MetaCyc pathway column remains after "
            "excluding sample metadata, stratified columns and HUMAnN special features"
        )
    return pathways, audit


def numeric_matrix(frame: pd.DataFrame, source_label: str) -> np.ndarray:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    conversion_failure = frame.notna() & numeric.isna()
    if conversion_failure.any().any():
        column = str(conversion_failure.any(axis=0).idxmax())
        raise ValueError(
            f"{source_label} column {column!r} contains non-numeric values"
        )
    values = numeric.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError(f"{source_label} contains infinite values")
    finite = values[np.isfinite(values)]
    if finite.size and np.any(finite < 0):
        raise ValueError(f"{source_label} contains negative abundance values")
    return values


def load_and_align_metadata(
    metadata_path: Path,
    pathway_arrow: Path,
    key_columns: list[str],
    age_column: str,
    sex_column: str,
    age_min: float,
    age_max: float,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, object]]:
    parquet_schema = parquet.ParquetFile(metadata_path).schema_arrow
    available = set(parquet_schema.names)
    required = list(dict.fromkeys(key_columns + [age_column, sex_column]))
    missing = [column for column in required if column not in available]
    if missing:
        raise ValueError(
            f"Metadata Parquet lacks required columns {missing}; "
            f"available metadata columns include {list(parquet_schema.names[:20])}"
        )
    metadata = pd.read_parquet(metadata_path, columns=required)
    metadata_keys = build_sample_keys(metadata, key_columns, "oral metadata Parquet")

    arrow_schema = read_arrow_schema(pathway_arrow)
    arrow_names = set(arrow_schema.names)
    arrow_missing = [column for column in key_columns if column not in arrow_names]
    if arrow_missing:
        raise ValueError(
            "Community-level HUMAnN Arrow lacks sample-key columns "
            f"{arrow_missing}; row-position matching is prohibited"
        )
    arrow_key_frame = feather.read_table(
        pathway_arrow,
        columns=key_columns,
        memory_map=True,
    ).to_pandas(ignore_metadata=True)
    arrow_keys = build_sample_keys(
        arrow_key_frame,
        key_columns,
        "community-level HUMAnN Arrow",
    )

    metadata_index = pd.Index(metadata_keys)
    arrow_index = pd.Index(arrow_keys)
    shared = metadata_index[metadata_index.isin(arrow_index)]
    if shared.empty:
        raise AnalysisSkip(
            "No sample key is shared by the oral metadata Parquet and "
            "community-level HUMAnN Arrow"
        )
    metadata_rows = metadata_index.get_indexer(shared).astype(np.int64)
    arrow_rows = arrow_index.get_indexer(shared).astype(np.int64)
    if np.any(metadata_rows < 0) or np.any(arrow_rows < 0):
        raise RuntimeError("Sample alignment produced an unresolved row index")

    aligned = metadata.iloc[metadata_rows].reset_index(drop=True).copy()
    aligned["_sample_key"] = shared.astype(str)
    aligned[age_column] = pd.to_numeric(aligned[age_column], errors="coerce")
    sex_text = aligned[sex_column].astype("string").str.strip()
    valid = (
        aligned[age_column].notna()
        & np.isfinite(aligned[age_column].to_numpy(dtype=float, na_value=np.nan))
        & aligned[age_column].between(age_min, age_max, inclusive="both")
        & sex_text.notna()
        & ~sex_text.str.casefold().isin({"", "nan", "none", "null", "<na>"})
    )
    excluded_outcome_covariate = int((~valid).sum())
    aligned = aligned.loc[valid].reset_index(drop=True)
    arrow_rows = arrow_rows[valid.to_numpy(dtype=bool)]
    if aligned.empty:
        raise AnalysisSkip(
            "No aligned sample has complete age and sex within the prespecified age range"
        )
    sex_levels = sorted(aligned[sex_column].astype(str).unique().tolist())
    if len(sex_levels) != 2:
        raise AnalysisSkip(
            "Sex adjustment requires exactly two observed levels; "
            f"observed {sex_levels}"
        )

    shared_hash = hashlib.sha256(
        "\n".join(sorted(aligned["_sample_key"].astype(str))).encode("utf-8")
    ).hexdigest()
    metadata_only = metadata_index.difference(arrow_index, sort=False)
    arrow_only = arrow_index.difference(metadata_index, sort=False)
    alignment_table = pd.DataFrame(
        [
            {
                "source": "oral_metadata_parquet",
                "input_samples": len(metadata_index),
                "shared_before_age_sex_qc": len(shared),
                "samples_after_age_sex_qc": len(aligned),
                "source_only_samples_excluded": len(metadata_only),
            },
            {
                "source": "community_pathway_arrow",
                "input_samples": len(arrow_index),
                "shared_before_age_sex_qc": len(shared),
                "samples_after_age_sex_qc": len(aligned),
                "source_only_samples_excluded": len(arrow_only),
            },
        ]
    )
    audit = {
        "sample_key_columns": key_columns,
        "metadata_input_samples": int(len(metadata_index)),
        "community_pathway_input_samples": int(len(arrow_index)),
        "shared_before_age_sex_qc": int(len(shared)),
        "excluded_for_missing_or_out_of_range_age_or_missing_sex": (
            excluded_outcome_covariate
        ),
        "aligned_samples_after_age_sex_qc": int(len(aligned)),
        "metadata_only_samples_excluded": int(len(metadata_only)),
        "community_only_samples_excluded": int(len(arrow_only)),
        "age_min": float(aligned[age_column].min()),
        "age_max": float(aligned[age_column].max()),
        "age_mean": float(aligned[age_column].mean()),
        "age_sd": float(aligned[age_column].std(ddof=1)),
        "sex_levels": sex_levels,
        "included_sample_key_set_sha256": shared_hash,
        "identifiers_exported": False,
        "alignment_policy": (
            "unique sample-key intersection; metadata order retained; no "
            "positional matching and no missing HUMAnN row imputation"
        ),
    }
    return aligned, arrow_rows, alignment_table, audit


def preprocess_pathways(
    pathway_arrow: Path,
    arrow_rows: np.ndarray,
    pathway_info: pd.DataFrame,
    sample_keys: pd.Series,
    detection_percentile: float,
    min_detected_samples: int,
    min_observed_pathways_per_sample: int,
    sparse_null_policy: str,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, dict[str, object], np.ndarray,]:
    columns = pathway_info["column"].tolist()
    frame = feather.read_table(
        pathway_arrow,
        columns=columns,
        memory_map=True,
    ).to_pandas(ignore_metadata=True)
    frame = frame.iloc[arrow_rows].reset_index(drop=True)
    raw = numeric_matrix(frame, "community-level HUMAnN pathway matrix")
    if len(sample_keys) != raw.shape[0]:
        raise RuntimeError(
            "Aligned sample keys and community pathway rows have different lengths"
        )

    finite_mask = np.isfinite(raw)
    positive_mask = finite_mask & (raw > 0)
    positive_pathways = np.count_nonzero(positive_mask, axis=1)
    finite_pathways = np.count_nonzero(finite_mask, axis=1)
    null_pathways = np.count_nonzero(~finite_mask, axis=1)
    literal_zero_pathways = np.count_nonzero(finite_mask & (raw == 0), axis=1)
    sample_keep = positive_pathways >= min_observed_pathways_per_sample
    retained_sample_count = int(np.count_nonzero(sample_keep))
    excluded_sample_count = int(sample_keep.size - retained_sample_count)

    sample_key_text = (
        pd.Series(sample_keys, copy=False).astype(str).reset_index(drop=True)
    )
    analysis_row_after_qc = np.full(sample_keep.size, -1, dtype=int)
    analysis_row_after_qc[sample_keep] = np.arange(retained_sample_count, dtype=int)
    sample_audit = pd.DataFrame(
        {
            "aligned_row_index": np.arange(sample_keep.size, dtype=int),
            "analysis_row_index_after_qc": analysis_row_after_qc,
            "sample_key_sha256": [
                hashlib.sha256(value.encode("utf-8")).hexdigest()
                for value in sample_key_text
            ],
            "eligible_pathways": int(raw.shape[1]),
            "finite_pathways_before_fill": finite_pathways.astype(int),
            "positive_pathways_before_fill": positive_pathways.astype(int),
            "literal_zero_pathways_before_fill": literal_zero_pathways.astype(int),
            "null_pathways_before_fill": null_pathways.astype(int),
            "minimum_required_positive_pathways": int(min_observed_pathways_per_sample),
            "sample_preprocessing_status": np.where(
                sample_keep,
                "retained_for_age_association",
                "excluded_fewer_than_minimum_positive_pathways",
            ),
            "null_cells_recoded_to_zero": np.where(
                sample_keep,
                null_pathways,
                0,
            ).astype(int),
        }
    )
    if retained_sample_count == 0:
        raise AnalysisSkip(
            "No aligned sample has at least "
            f"{min_observed_pathways_per_sample} positive eligible MetaCyc pathways"
        )

    raw = raw[sample_keep].copy()
    retained_null_mask = np.isnan(raw)
    retained_null_values = int(np.count_nonzero(retained_null_mask))
    if retained_null_values and sparse_null_policy == "error":
        raise AnalysisSkip(
            "Community-level HUMAnN pathway matrix contains "
            f"{retained_null_values:,} null/NaN cells after sample QC. Rerun with "
            "--sparse-null-policy zero only if the Arrow nulls encode pathway "
            "non-detection rather than unavailable measurements"
        )
    if retained_null_values:
        raw[retained_null_mask] = 0.0
    if np.isnan(raw).any():
        raise RuntimeError("Sparse-null preprocessing left unresolved NaN values")

    positive = raw[raw > 0]
    if positive.size == 0:
        raise AnalysisSkip(
            "No positive community pathway abundance remains for threshold estimation"
        )
    threshold = float(np.percentile(positive, detection_percentile, method="linear"))
    if not np.isfinite(threshold) or threshold <= 0:
        raise RuntimeError("Global pathway detection threshold is not positive")
    zero_replacement = threshold / 2.0
    adjusted = raw.copy()
    adjusted[adjusted == 0] = zero_replacement
    denominator = adjusted.sum(axis=1)
    if not np.isfinite(denominator).all() or np.any(denominator <= 0):
        raise RuntimeError("Pathway sum-normalization denominator is invalid")
    ppm = adjusted / denominator[:, None] * 1_000_000.0
    log10_ppm = np.log10(ppm)
    detected_samples = np.count_nonzero(raw >= threshold, axis=0)
    raw_positive_samples = np.count_nonzero(raw > 0, axis=0)
    transformed_sd = np.std(log10_ppm, axis=0, ddof=1)
    retained = (
        (detected_samples >= min_detected_samples)
        & np.isfinite(transformed_sd)
        & (transformed_sd > 0)
    )

    audit = pathway_info.copy()
    audit["raw_positive_samples"] = raw_positive_samples.astype(int)
    audit["detected_samples_at_global_threshold"] = detected_samples.astype(int)
    audit["mean_raw_abundance"] = raw.mean(axis=0)
    audit["mean_ppm"] = ppm.mean(axis=0)
    audit["mean_log10_ppm"] = log10_ppm.mean(axis=0)
    audit["sd_log10_ppm"] = transformed_sd
    audit["preprocessing_status"] = np.where(
        detected_samples < min_detected_samples,
        "excluded_detected_in_fewer_than_minimum_samples",
        np.where(
            ~(np.isfinite(transformed_sd) & (transformed_sd > 0)),
            "excluded_zero_or_invalid_transformed_variance",
            "retained_for_age_association",
        ),
    )
    retained_matrix = log10_ppm[:, retained]
    retained_info = audit.loc[retained].reset_index(drop=True)
    if retained_info.empty:
        raise AnalysisSkip(
            "No pathway passes the prespecified detection and variance criteria"
        )
    summary = {
        "aligned_samples_before_pathway_qc": int(sample_keep.size),
        "minimum_positive_pathways_per_sample": int(min_observed_pathways_per_sample),
        "samples_excluded_by_pathway_qc": excluded_sample_count,
        "samples_retained_after_pathway_qc": retained_sample_count,
        "retained_sample_fraction": float(retained_sample_count / sample_keep.size),
        "sample_qc_count_definition": (
            "strictly positive finite abundances among eligible unstratified "
            "community pathways after HUMAnN special-feature and scope exclusion"
        ),
        "positive_pathways_per_sample_before_qc_min": int(positive_pathways.min()),
        "positive_pathways_per_sample_before_qc_median": float(
            np.median(positive_pathways)
        ),
        "positive_pathways_per_sample_before_qc_max": int(positive_pathways.max()),
        "sparse_null_policy": sparse_null_policy,
        "null_cells_before_sample_qc": int(null_pathways.sum()),
        "null_cells_in_excluded_samples": int(null_pathways[~sample_keep].sum()),
        "null_cells_recoded_to_zero_in_retained_samples": retained_null_values,
        "literal_zero_cells_in_retained_samples_before_fill": int(
            literal_zero_pathways[sample_keep].sum()
        ),
        "sample_identifiers_in_audit": "SHA-256 hashes only",
        "input_pathways_after_scope_exclusion": int(len(pathway_info)),
        "positive_values_used_for_threshold": int(positive.size),
        "detection_percentile": float(detection_percentile),
        "global_detection_threshold": threshold,
        "zero_replacement": zero_replacement,
        "null_replacement": (
            "sparse Arrow nulls in retained samples were recoded to zero, then "
            "all zero values were replaced with half the global detection threshold"
            if sparse_null_policy == "zero"
            else None
        ),
        "minimum_detected_samples": int(min_detected_samples),
        "retained_pathways": int(len(retained_info)),
        "normalization": (
            "recode sparse Arrow nulls as non-detection zero; replace all zero "
            "values with half the global detection threshold; "
            "sum-normalize eligible pathways within each sample; multiply by "
            "1,000,000; log10 transform"
        ),
        "missing_data_boundary": (
            "Only within-profile sparse pathway null cells were recoded. Missing "
            "samples, sample identifiers, age and sex were not imputed."
        ),
        "abundance_interpretation": "relative community-level pathway abundance",
        "denominator_min": float(denominator.min()),
        "denominator_max": float(denominator.max()),
    }
    return retained_matrix, audit, sample_audit, summary, sample_keep


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        return np.array([], dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("P values must be finite and lie in [0, 1]")
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted = ranked * values.size / np.arange(1, values.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def sex_design(sex: pd.Series) -> tuple[np.ndarray, list[str]]:
    encoded = pd.get_dummies(
        sex.astype(str),
        prefix="sex",
        drop_first=True,
        dtype=float,
    )
    if encoded.shape[1] != 1:
        raise AnalysisSkip(
            f"Sex adjustment expected one indicator column; obtained {list(encoded.columns)}"
        )
    design = np.column_stack(
        [np.ones(len(sex), dtype=float), encoded.to_numpy(dtype=float)]
    )
    return design, ["intercept", *encoded.columns.astype(str).tolist()]


def residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    coefficients = np.linalg.pinv(design) @ values
    return values - design @ coefficients


def fit_age_associations(
    metadata: pd.DataFrame,
    log10_ppm: np.ndarray,
    retained_info: pd.DataFrame,
    age_column: str,
    sex_column: str,
    covariance: str,
    alpha: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    age = metadata[age_column].to_numpy(dtype=float)
    age_10_centered = (age - age.mean()) / 10.0
    base_design, base_names = sex_design(metadata[sex_column])
    age_residual = residualize(age[:, None], base_design).ravel()
    rows: list[dict[str, object]] = []
    for index, pathway in retained_info.reset_index(drop=True).iterrows():
        y = log10_ppm[:, index]
        y_mean = float(y.mean())
        y_sd = float(y.std(ddof=1))
        y_z = (y - y_mean) / y_sd

        primary_design = np.column_stack(
            [np.ones(len(y_z)), age_10_centered, base_design[:, 1:]]
        )
        primary = sm.OLS(y_z, primary_design).fit(cov_type=covariance)
        primary_classical = sm.OLS(y_z, primary_design).fit()
        primary_ci = np.asarray(primary.conf_int(alpha=alpha))[1]

        paper_design = np.column_stack([np.ones(len(y_z)), y_z, base_design[:, 1:]])
        paper_direction = sm.OLS(age, paper_design).fit(cov_type=covariance)
        paper_ci = np.asarray(paper_direction.conf_int(alpha=alpha))[1]

        y_residual = residualize(y_z[:, None], base_design).ravel()
        denominator = float(
            np.sqrt(np.dot(age_residual, age_residual) * np.dot(y_residual, y_residual))
        )
        partial_r = (
            float(np.dot(age_residual, y_residual) / denominator)
            if denominator > 0
            else np.nan
        )
        rows.append(
            {
                "pathway_id": str(pathway["pathway_id"]),
                "pathway_name": str(pathway["pathway_name"]),
                "community_column": str(pathway["column"]),
                "n": int(len(y_z)),
                "detected_samples": int(
                    pathway["detected_samples_at_global_threshold"]
                ),
                "mean_log10_ppm": y_mean,
                "sd_log10_ppm": y_sd,
                "beta_pathway_sd_per_10y": float(primary.params[1]),
                "se_hc_or_selected_covariance": float(primary.bse[1]),
                "ci95_low_pathway_sd_per_10y": float(primary_ci[0]),
                "ci95_high_pathway_sd_per_10y": float(primary_ci[1]),
                "t_age": float(primary.tvalues[1]),
                "p_age_two_sided": float(primary.pvalues[1]),
                "p_age_two_sided_classical_ols": float(primary_classical.pvalues[1]),
                "partial_r_age_pathway_adjusted_sex": partial_r,
                "model_r_squared": float(primary.rsquared),
                "model_df_resid": float(primary.df_resid),
                "paper_direction_beta_age_years_per_1sd_pathway": float(
                    paper_direction.params[1]
                ),
                "paper_direction_ci95_low": float(paper_ci[0]),
                "paper_direction_ci95_high": float(paper_ci[1]),
                "paper_direction_p_two_sided": float(paper_direction.pvalues[1]),
            }
        )
    result = pd.DataFrame(rows)
    result["q_bh_global"] = bh_adjust(result["p_age_two_sided"].to_numpy(dtype=float))
    result["p_bonferroni_global"] = np.minimum(
        result["p_age_two_sided"] * len(result), 1.0
    )
    result["positive_bh_fdr"] = result["q_bh_global"] < alpha
    result["positive_bonferroni"] = result["p_bonferroni_global"] < alpha
    result = result.sort_values(
        ["p_age_two_sided", "pathway_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    audit = {
        "primary_model": (
            "standardized log10(PPM pathway) ~ centered age per 10 years + sex"
        ),
        "sensitivity_model": (
            "chronological age in years ~ standardized log10(PPM pathway) + sex"
        ),
        "covariance_estimator": covariance,
        "two_sided_tests": True,
        "test_family": "all retained community-level MetaCyc pathways for age",
        "test_count": int(len(result)),
        "bonferroni_positive": int(result["positive_bonferroni"].sum()),
        "bh_fdr_positive": int(result["positive_bh_fdr"].sum()),
        "alpha": float(alpha),
        "age_outlier_processing": (
            "none; age was restricted to prespecified cohort eligibility bounds"
        ),
        "sex_design_columns": base_names,
    }
    return result, audit


def select_display_pathways(
    associations: pd.DataFrame,
    log10_ppm: np.ndarray,
    retained_info: pd.DataFrame,
    mode: str,
    alpha: float,
    max_pathways: int,
    correlation_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mode == "bonferroni":
        candidates = associations.loc[associations["p_bonferroni_global"] < alpha]
    elif mode == "bh":
        candidates = associations.loc[associations["q_bh_global"] < alpha]
    elif mode == "top":
        candidates = associations
    else:  # pragma: no cover - guarded by argparse
        raise ValueError(f"Unsupported display mode: {mode}")
    candidates = candidates.sort_values(
        ["p_age_two_sided", "pathway_id"],
        kind="mergesort",
    )
    if candidates.empty:
        return candidates.copy(), pd.DataFrame(
            columns=[
                "pathway_id",
                "display_status",
                "correlated_with",
                "abs_spearman_r",
            ]
        )

    info_index = {
        str(row.pathway_id): idx
        for idx, row in retained_info.reset_index(drop=True).iterrows()
    }
    selected_ids: list[str] = []
    audit_rows: list[dict[str, object]] = []
    for row in candidates.itertuples(index=False):
        pathway_id = str(row.pathway_id)
        if len(selected_ids) >= max_pathways:
            audit_rows.append(
                {
                    "pathway_id": pathway_id,
                    "display_status": "excluded_after_max_pathways_reached",
                    "correlated_with": "",
                    "abs_spearman_r": np.nan,
                }
            )
            continue
        candidate_values = log10_ppm[:, info_index[pathway_id]]
        redundant_with = ""
        redundant_r = np.nan
        if correlation_threshold < 1.0:
            candidate_rank = (
                pd.Series(candidate_values).rank(method="average").to_numpy()
            )
            for selected_id in selected_ids:
                selected_values = log10_ppm[:, info_index[selected_id]]
                selected_rank = (
                    pd.Series(selected_values).rank(method="average").to_numpy()
                )
                rho = float(np.corrcoef(candidate_rank, selected_rank)[0, 1])
                if abs(rho) >= correlation_threshold:
                    redundant_with = selected_id
                    redundant_r = abs(rho)
                    break
        if redundant_with:
            status = "excluded_from_figure_for_correlation_redundancy"
        else:
            selected_ids.append(pathway_id)
            status = "included_in_figure"
        audit_rows.append(
            {
                "pathway_id": pathway_id,
                "display_status": status,
                "correlated_with": redundant_with,
                "abs_spearman_r": redundant_r,
            }
        )
    selected = associations.loc[associations["pathway_id"].isin(selected_ids)].copy()
    selected["_order"] = pd.Categorical(
        selected["pathway_id"],
        categories=selected_ids,
        ordered=True,
    )
    selected = selected.sort_values("_order").drop(columns="_order")
    return selected.reset_index(drop=True), pd.DataFrame(audit_rows)


def make_age_bins(
    age: pd.Series,
    age_min: float,
    age_max: float,
    bin_width: float,
) -> tuple[pd.Categorical, np.ndarray, list[str]]:
    interval_count = (age_max - age_min) / bin_width
    rounded = int(round(interval_count))
    if rounded < 2 or not np.isclose(interval_count, rounded):
        raise ValueError(
            "The prespecified age range must divide exactly into at least two bins"
        )
    edges = np.linspace(age_min, age_max, rounded + 1)
    edges_for_cut = edges.copy()
    edges_for_cut[-1] = np.nextafter(edges_for_cut[-1], np.inf)
    labels = [f"{edges[i]:g}-{edges[i + 1]:g}" for i in range(len(edges) - 1)]
    binned = pd.cut(
        age,
        bins=edges_for_cut,
        labels=labels,
        include_lowest=True,
        right=False,
        ordered=True,
    )
    if binned.isna().any():
        raise RuntimeError("A retained age was not assigned to an age bin")
    return binned, edges, labels


def build_age_bin_source_data(
    metadata: pd.DataFrame,
    log10_ppm: np.ndarray,
    retained_info: pd.DataFrame,
    selected: pd.DataFrame,
    age_column: str,
    sex_column: str,
    age_min: float,
    age_max: float,
    bin_width: float,
    min_bin_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    binned, edges, labels = make_age_bins(
        metadata[age_column], age_min, age_max, bin_width
    )
    bin_counts = binned.value_counts(sort=False).reindex(labels, fill_value=0)
    if int(bin_counts.min()) < min_bin_n:
        raise AnalysisSkip(
            "At least one prespecified display age bin contains fewer than "
            f"{min_bin_n} samples: {bin_counts.to_dict()}"
        )
    base_design, _ = sex_design(metadata[sex_column])
    grand_means = log10_ppm.mean(axis=0, keepdims=True)
    adjusted = residualize(log10_ppm, base_design) + grand_means
    info_index = {
        str(row.pathway_id): idx
        for idx, row in retained_info.reset_index(drop=True).iterrows()
    }
    rows: list[dict[str, object]] = []
    heat_rows: list[np.ndarray] = []
    heat_index: list[str] = []
    for selected_row in selected.itertuples(index=False):
        pathway_id = str(selected_row.pathway_id)
        column_index = info_index[pathway_id]
        adjusted_means: list[float] = []
        for bin_index, label in enumerate(labels):
            mask = np.asarray(binned == label, dtype=bool)
            raw_values = log10_ppm[mask, column_index]
            adjusted_values = adjusted[mask, column_index]
            adjusted_mean = float(adjusted_values.mean())
            adjusted_means.append(adjusted_mean)
            rows.append(
                {
                    "pathway_id": pathway_id,
                    "pathway_name": str(selected_row.pathway_name),
                    "age_bin": label,
                    "age_bin_start": float(edges[bin_index]),
                    "age_bin_end": float(edges[bin_index + 1]),
                    "n": int(mask.sum()),
                    "raw_mean_log10_ppm": float(raw_values.mean()),
                    "raw_sd_log10_ppm": float(raw_values.std(ddof=1)),
                    "sex_adjusted_mean_log10_ppm": adjusted_mean,
                    "sex_adjusted_se_log10_ppm": float(
                        adjusted_values.std(ddof=1) / np.sqrt(mask.sum())
                    ),
                }
            )
        values = np.asarray(adjusted_means, dtype=float)
        sd = float(values.std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            z_values = np.full(values.shape, np.nan)
        else:
            z_values = (values - values.mean()) / sd
        heat_rows.append(z_values)
        heat_index.append(pathway_id)
    source = pd.DataFrame(rows)
    heatmap = pd.DataFrame(heat_rows, index=heat_index, columns=labels)
    audit = {
        "age_bins": labels,
        "age_bin_counts": {str(k): int(v) for k, v in bin_counts.items()},
        "minimum_age_bin_n": int(min_bin_n),
        "heatmap_measure": (
            "sex-adjusted mean log10(PPM), standardized within each pathway "
            "across age bins"
        ),
        "inference_uses_age_bins": False,
    }
    return source, heatmap, audit


def pathway_display_label(pathway_name: str, pathway_id: str) -> str:
    name = pathway_name.strip() or pathway_id
    compact = " ".join(name.split())
    shortened = textwrap.shorten(compact, width=52, placeholder="...")
    return f"{shortened}\n{pathway_id}" if name != pathway_id else shortened


def p_text(value: float) -> str:
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.3f}"


def draw_figure(
    selected: pd.DataFrame,
    heatmap: pd.DataFrame,
    age_bin_source: pd.DataFrame,
    output_dir: Path,
    preview_dpi: int,
    display_mode: str,
    covariance: str,
    analysis_n: int,
) -> tuple[dict[str, str], dict[str, object]]:
    if selected.empty:
        raise AnalysisSkip("No pathway is available for the age-association figure")
    ordered = selected.sort_values(
        ["beta_pathway_sd_per_10y", "pathway_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    heat = heatmap.loc[ordered["pathway_id"]]
    row_count = len(ordered)
    figure_height_mm = min(
        FIGURE_MAX_HEIGHT_MM,
        max(105.0, 45.0 + 6.5 * row_count),
    )
    fig = plt.figure(
        figsize=(FIGURE_WIDTH_MM / 25.4, figure_height_mm / 25.4),
        facecolor="white",
    )
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[2.35, 1.25, 0.72],
        wspace=0.23,
        left=0.37,
        right=0.985,
        top=0.82,
        bottom=0.22,
    )
    ax_heat = fig.add_subplot(grid[0, 0])
    ax_effect = fig.add_subplot(grid[0, 1], sharey=ax_heat)
    ax_p = fig.add_subplot(grid[0, 2], sharey=ax_heat)

    cmap = LinearSegmentedColormap.from_list(
        "age_association_heatmap",
        [HEAT_LOW, HEAT_MID, HEAT_HIGH],
    )
    cmap.set_bad(MISSING)
    image = ax_heat.imshow(
        np.ma.masked_invalid(heat.to_numpy(dtype=float)),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0),
    )
    ax_heat.set_xticks(np.arange(heat.shape[1]))
    bin_counts = (
        age_bin_source[["age_bin", "n"]]
        .drop_duplicates()
        .set_index("age_bin")["n"]
        .reindex(heat.columns)
    )
    ax_heat.set_xticklabels(
        list(heat.columns),
        ha="center",
        fontsize=5.5,
    )
    for column_index, label in enumerate(heat.columns):
        ax_heat.text(
            column_index,
            -0.075,
            f"n={int(bin_counts[label])}",
            transform=ax_heat.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=5.3,
            color=MUTED,
            clip_on=False,
        )
    ax_heat.set_yticks(np.arange(row_count))
    ax_heat.set_yticklabels(
        [
            pathway_display_label(str(row.pathway_name), str(row.pathway_id))
            for row in ordered.itertuples(index=False)
        ],
        fontsize=5.5,
        ha="right",
    )
    ax_heat.tick_params(axis="y", length=0, pad=5)
    ax_heat.tick_params(axis="x", length=0)
    ax_heat.set_xticks(np.arange(-0.5, heat.shape[1], 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, row_count, 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=0.75)
    ax_heat.tick_params(which="minor", bottom=False, left=False)
    ax_heat.set_title(
        "Sex-adjusted pathway abundance across age",
        fontsize=7.5,
        fontweight="bold",
        pad=8,
    )

    y = np.arange(row_count)
    beta = ordered["beta_pathway_sd_per_10y"].to_numpy(dtype=float)
    low = ordered["ci95_low_pathway_sd_per_10y"].to_numpy(dtype=float)
    high = ordered["ci95_high_pathway_sd_per_10y"].to_numpy(dtype=float)
    colors = np.where(beta >= 0, HEAT_HIGH, HEAT_LOW)
    ax_effect.axvline(0, color="#999999", linewidth=0.8, linestyle="--", zorder=0)
    ax_effect.hlines(y, low, high, color="#4D4D4D", linewidth=1.0, zorder=1)
    ax_effect.scatter(
        beta,
        y,
        c=colors,
        s=22,
        edgecolors="#333333",
        linewidths=0.45,
        zorder=2,
    )
    bound = float(np.nanmax(np.abs(np.concatenate([low, high]))))
    bound = max(0.05, bound * 1.12)
    ax_effect.set_xlim(-bound, bound)
    ax_effect.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_effect.set_yticks(y)
    ax_effect.tick_params(axis="y", left=False, labelleft=False)
    ax_effect.tick_params(axis="x", labelsize=6)
    ax_effect.set_title(
        "Continuous-age association",
        fontsize=7.5,
        fontweight="bold",
        pad=8,
    )
    ax_effect.set_xlabel("Pathway SD per 10 years (95% CI)", fontsize=6.5)

    ax_p.set_xlim(0, 1)
    ax_p.set_ylim(row_count - 0.5, -0.5)
    ax_p.axis("off")
    correction_column = (
        "p_bonferroni_global" if display_mode == "bonferroni" else "q_bh_global"
    )
    correction_label = "PBonf" if display_mode == "bonferroni" else "qBH"
    if display_mode == "top":
        correction_column = "p_bonferroni_global"
        correction_label = "PBonf"
    ax_p.set_title(
        correction_label,
        fontsize=7.5,
        fontweight="bold",
        pad=8,
    )
    for row_index, row in enumerate(ordered.itertuples(index=False)):
        ax_p.text(
            0.0,
            row_index,
            p_text(float(getattr(row, correction_column))),
            va="center",
            ha="left",
            fontsize=6,
            color=TEXT,
        )

    fig.text(
        0.015,
        0.955,
        "Oral MetaCyc pathways associated with chronological age",
        fontsize=9.2,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.015,
        0.905,
        (
            f"n={analysis_n:,} independent participants; OLS adjusted for sex; "
            f"{covariance} covariance; two-sided tests"
        ),
        fontsize=6.5,
        color=MUTED,
    )
    colorbar_axis = fig.add_axes([0.405, 0.105, 0.205, 0.016])
    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_ticks([-2, 0, 2])
    colorbar.set_ticklabels(["-2", "0", "+2"])
    colorbar.ax.tick_params(labelsize=6, length=2)
    colorbar.set_label(
        "Within-pathway Z-score of sex-adjusted mean log10(PPM)",
        fontsize=6,
        labelpad=2,
    )
    fig.text(
        0.015,
        0.015,
        (
            "Heatmap bins are descriptive; inference uses continuous age. "
            "Relative functional abundance does not establish pathway activity, "
            "causality or host-pathway activation."
        ),
        fontsize=5.6,
        color=MUTED,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "oral_metacyc_pathway_age_association"
    pdf_path = stem.with_suffix(".pdf")
    svg_path = stem.with_suffix(".svg")
    png_path = stem.with_suffix(".png")
    alignment_path = stem.with_suffix(".alignment.json")
    alignment = require_matplotlib_panel_alignment(
        fig,
        axes={
            "heatmap": ax_heat,
            "effect": ax_effect,
            "multiplicity": ax_p,
        },
        json_out=alignment_path,
        tolerance_pt=1.5,
    )
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=preview_dpi, bbox_inches="tight")
    plt.close(fig)
    return (
        {
            "pdf": str(pdf_path.resolve()),
            "svg": str(svg_path.resolve()),
            "png_preview": str(png_path.resolve()),
            "alignment_json": str(alignment_path.resolve()),
        },
        alignment,
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_metadata = (
        script_dir.parent
        / "01-interim_data"
        / "omics_data"
        / "oral_microbiome_with_metadata.parquet"
    )
    default_arrow = Path(
        "/home/ec2-user/studies/hpp_datasets/oral_microbiome/aggregated/"
        "humann_aggregated/pathway_abundance_pathway_level.arrow"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-table",
        type=Path,
        default=default_metadata,
        help="Oral microbiome metadata Parquet containing age and sex.",
    )
    parser.add_argument(
        "--community-pathway-table",
        type=Path,
        default=default_arrow,
        help="Community-level HUMAnN pathway_abundance_pathway_level.arrow.",
    )
    parser.add_argument(
        "--sample-key-columns",
        default="participant_id",
        help="Comma-separated unique sample key shared by both inputs.",
    )
    parser.add_argument("--age-column", default="age")
    parser.add_argument("--sex-column", default="sex")
    parser.add_argument(
        "--arrow-metadata-columns",
        default="",
        help="Additional comma-separated Arrow metadata columns to exclude.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--age-min", type=float, default=40.0)
    parser.add_argument("--age-max", type=float, default=70.0)
    parser.add_argument("--age-bin-width", type=float, default=5.0)
    parser.add_argument("--min-age-bin-n", type=int, default=100)
    parser.add_argument("--detection-percentile", type=float, default=1.0)
    parser.add_argument("--min-detected-samples", type=int, default=500)
    parser.add_argument(
        "--min-observed-pathways-per-sample",
        type=int,
        default=10,
        help=(
            "Sample-level QC threshold applied after pathway-scope exclusion and "
            "before null filling. Retain samples with at least this many strictly "
            "positive eligible community pathways (default: 10; use 1 and 20 for "
            "prespecified sensitivity analyses)."
        ),
    )
    parser.add_argument(
        "--sparse-null-policy",
        choices=["zero", "error"],
        default="zero",
        help=(
            "How to handle null cells in retained HUMAnN pathway profiles. "
            "'zero' treats sparse Arrow nulls as pathway non-detection and then "
            "applies the half-detection-threshold replacement; 'error' stops."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--covariance",
        choices=["HC3", "nonrobust"],
        default="HC3",
        help="OLS covariance estimator; HC3 is the robust default.",
    )
    parser.add_argument(
        "--display-selection",
        choices=["bonferroni", "bh", "top"],
        default="bonferroni",
        help=(
            "Figure-row rule. 'bonferroni' mirrors the paper; 'bh' and 'top' "
            "must be described as exploratory."
        ),
    )
    parser.add_argument("--max-pathways", type=int, default=12)
    parser.add_argument(
        "--display-correlation-threshold",
        type=float,
        default=0.90,
        help=(
            "Greedy absolute Spearman threshold for figure-only redundancy "
            "pruning; set 1 to disable. Full statistics are never pruned."
        ),
    )
    parser.add_argument("--scope-exclusion-regex", default="")
    parser.add_argument("--preview-dpi", type=int, default=300)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.output_dir is None:
        args.output_dir = script_dir / "outputs" / "oral_metacyc_age_association"
    if args.figure_dir is None:
        args.figure_dir = script_dir / "figures" / "oral_metacyc_age_association"
    args.run_id = args.run_id or uuid.uuid4().hex
    if not np.isfinite(args.age_min) or not np.isfinite(args.age_max):
        parser.error("age bounds must be finite")
    if args.age_max <= args.age_min:
        parser.error("--age-max must be greater than --age-min")
    if not np.isfinite(args.age_bin_width) or args.age_bin_width <= 0:
        parser.error("--age-bin-width must be positive")
    if not 0 < args.detection_percentile < 100:
        parser.error("--detection-percentile must lie strictly between 0 and 100")
    if (
        args.min_detected_samples < 1
        or args.min_age_bin_n < 1
        or args.min_observed_pathways_per_sample < 1
    ):
        parser.error("minimum sample counts must be positive")
    if not 0 < args.alpha < 1:
        parser.error("--alpha must lie strictly between 0 and 1")
    if args.max_pathways < 1:
        parser.error("--max-pathways must be positive")
    if not 0 < args.display_correlation_threshold <= 1:
        parser.error("--display-correlation-threshold must lie in (0, 1]")
    if args.preview_dpi < 150:
        parser.error("--preview-dpi must be at least 150")
    return args


def initial_manifest(args: argparse.Namespace) -> dict[str, object]:
    return {
        "analysis_version": ANALYSIS_VERSION,
        "run_id": args.run_id,
        "status": "running",
        "started_utc": utc_now(),
        "analysis_question": (
            "Which community-level oral MetaCyc pathways are associated with "
            "chronological age across independent participants?"
        ),
        "analysis_unit": "one unique participant-level oral microbiome sample",
        "primary_model": ("standardized log10(PPM pathway) ~ age per 10 years + sex"),
        "sample_quality_rule": (
            "retain samples meeting --min-observed-pathways-per-sample, counted "
            "as strictly positive eligible community pathways before filling"
        ),
        "sparse_null_interpretation": (
            "under --sparse-null-policy zero, within-profile Arrow null cells are "
            "treated as pathway non-detection; missing samples and covariates are "
            "never imputed"
        ),
        "explicitly_not_used": [
            "five oral-taxon pattern assignments",
            "microbe-level HUMAnN contribution Arrow",
            "070-oral_microbiome_imputed.parquet",
            "project-generated functional theme dictionary",
            "age categories for inferential testing",
        ],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "matplotlib": mpl.__version__,
        },
        "arguments": {key: value for key, value in vars(args).items()},
    }


def run_analysis(args: argparse.Namespace, manifest: dict[str, object]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        args.output_dir / "oral_metacyc_pathway_age_association_manifest.json"
    )
    key_columns = split_csv(args.sample_key_columns)
    extra_metadata = (
        split_csv(args.arrow_metadata_columns)
        if args.arrow_metadata_columns.strip()
        else []
    )
    for label, path in (
        ("metadata table", args.metadata_table),
        ("community pathway table", args.community_pathway_table),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required {label} is missing: {path}")
    manifest["inputs"] = {
        "metadata_table": file_record(args.metadata_table),
        "community_pathway_table": file_record(args.community_pathway_table),
    }

    metadata, arrow_rows, alignment_table, alignment_audit = load_and_align_metadata(
        args.metadata_table,
        args.community_pathway_table,
        key_columns,
        args.age_column,
        args.sex_column,
        args.age_min,
        args.age_max,
    )
    alignment_path = args.output_dir / "sample_alignment_audit.csv"
    alignment_table.to_csv(alignment_path, index=False)
    manifest["sample_alignment"] = alignment_audit
    arrow_schema = read_arrow_schema(args.community_pathway_table)
    pathway_info, scope_audit = discover_community_pathways(
        arrow_schema,
        key_columns,
        extra_metadata,
        args.scope_exclusion_regex,
    )
    scope_path = args.output_dir / "metacyc_pathway_scope_audit.csv"
    scope_audit.to_csv(scope_path, index=False)

    (
        log10_ppm,
        preprocessing_audit,
        sample_preprocessing_audit,
        preprocessing_summary,
        sample_keep,
    ) = preprocess_pathways(
        args.community_pathway_table,
        arrow_rows,
        pathway_info,
        metadata["_sample_key"],
        args.detection_percentile,
        args.min_detected_samples,
        args.min_observed_pathways_per_sample,
        args.sparse_null_policy,
    )
    sample_preprocessing_path = (
        args.output_dir / "metacyc_sample_preprocessing_audit.csv"
    )
    sample_preprocessing_audit.to_csv(sample_preprocessing_path, index=False)
    samples_before_pathway_qc = int(len(metadata))
    metadata = metadata.loc[sample_keep].reset_index(drop=True)
    if len(metadata) != log10_ppm.shape[0]:
        raise RuntimeError(
            "Sample QC produced different metadata and pathway-matrix row counts"
        )
    final_sample_hash = hashlib.sha256(
        "\n".join(sorted(metadata["_sample_key"].astype(str))).encode("utf-8")
    ).hexdigest()
    alignment_audit.update(
        {
            "samples_before_pathway_sample_qc": samples_before_pathway_qc,
            "samples_excluded_by_pathway_sample_qc": int((~sample_keep).sum()),
            "final_analysis_samples": int(len(metadata)),
            "final_included_sample_key_set_sha256": final_sample_hash,
        }
    )
    manifest["sample_alignment"] = alignment_audit
    preprocessing_path = args.output_dir / "metacyc_pathway_preprocessing_audit.csv"
    preprocessing_audit.to_csv(preprocessing_path, index=False)
    retained_info = preprocessing_audit.loc[
        preprocessing_audit["preprocessing_status"] == "retained_for_age_association"
    ].reset_index(drop=True)
    manifest["pathway_preprocessing"] = preprocessing_summary

    associations, association_audit = fit_age_associations(
        metadata,
        log10_ppm,
        retained_info,
        args.age_column,
        args.sex_column,
        args.covariance,
        args.alpha,
    )
    associations_path = args.output_dir / "metacyc_pathway_age_associations_all.csv"
    associations.to_csv(associations_path, index=False)
    manifest["statistical_analysis"] = association_audit

    selected, selection_audit = select_display_pathways(
        associations,
        log10_ppm,
        retained_info,
        args.display_selection,
        args.alpha,
        args.max_pathways,
        args.display_correlation_threshold,
    )
    selection_audit_path = args.output_dir / "figure_pathway_selection_audit.csv"
    selection_audit.to_csv(selection_audit_path, index=False)
    selected_path = args.output_dir / "figure_pathways.csv"
    selected.to_csv(selected_path, index=False)
    manifest["figure_selection"] = {
        "mode": args.display_selection,
        "alpha": float(args.alpha),
        "maximum_pathways": int(args.max_pathways),
        "figure_only_absolute_spearman_redundancy_threshold": float(
            args.display_correlation_threshold
        ),
        "displayed_pathways": int(len(selected)),
        "full_association_table_pruned": False,
        "exploratory_display": args.display_selection != "bonferroni",
    }
    output_map: dict[str, object] = {
        "sample_alignment_audit": str(alignment_path.resolve()),
        "metacyc_sample_preprocessing_audit": str(sample_preprocessing_path.resolve()),
        "metacyc_pathway_scope_audit": str(scope_path.resolve()),
        "metacyc_pathway_preprocessing_audit": str(preprocessing_path.resolve()),
        "metacyc_pathway_age_associations_all": str(associations_path.resolve()),
        "figure_pathway_selection_audit": str(selection_audit_path.resolve()),
        "figure_pathways": str(selected_path.resolve()),
    }

    if selected.empty:
        manifest["status"] = "completed_no_main_figure"
        manifest["skip_reason"] = (
            f"No retained pathway satisfies display-selection={args.display_selection!r} "
            f"at alpha={args.alpha:g}; the complete association table is available"
        )
        manifest["outputs"] = output_map
        manifest["finished_utc"] = utc_now()
        write_json(manifest_path, manifest)
        print(f"Run ID: {args.run_id}")
        print("Status: completed_no_main_figure")
        print(f"Reason: {manifest['skip_reason']}")
        print(f"Analysis samples after pathway QC: {len(metadata):,}")
        print(f"Sample preprocessing audit: {sample_preprocessing_path.resolve()}")
        print(f"Full associations: {associations_path.resolve()}")
        print(f"Manifest: {manifest_path.resolve()}")
        return

    age_bin_source, heatmap, age_bin_audit = build_age_bin_source_data(
        metadata,
        log10_ppm,
        retained_info,
        selected,
        args.age_column,
        args.sex_column,
        args.age_min,
        args.age_max,
        args.age_bin_width,
        args.min_age_bin_n,
    )
    age_bin_source_path = args.output_dir / "figure_age_bin_source_data.csv"
    heatmap_path = args.output_dir / "figure_heatmap_row_zscores.csv"
    age_bin_source.to_csv(age_bin_source_path, index=False)
    heatmap.rename_axis("pathway_id").reset_index().to_csv(heatmap_path, index=False)
    figure_files, figure_alignment = draw_figure(
        selected,
        heatmap,
        age_bin_source,
        args.figure_dir,
        args.preview_dpi,
        args.display_selection,
        args.covariance,
        len(metadata),
    )
    output_map.update(
        {
            "figure_age_bin_source_data": str(age_bin_source_path.resolve()),
            "figure_heatmap_row_zscores": str(heatmap_path.resolve()),
            "figure_files": figure_files,
        }
    )
    manifest["age_bin_display"] = age_bin_audit
    manifest["figure_alignment"] = figure_alignment
    manifest["outputs"] = output_map
    manifest["status"] = "completed"
    manifest["finished_utc"] = utc_now()
    write_json(manifest_path, manifest)
    print(f"Run ID: {args.run_id}")
    print("Status: completed")
    print(f"Analysis samples: {len(metadata):,}")
    print(
        "Samples excluded by pathway QC: "
        f"{preprocessing_summary['samples_excluded_by_pathway_qc']:,}"
    )
    print(f"Retained pathways: {len(retained_info):,}")
    print(
        "Bonferroni-positive pathways: "
        f"{int(associations['positive_bonferroni'].sum()):,}"
    )
    print(f"Displayed pathways: {len(selected):,}")
    print(f"Full associations: {associations_path.resolve()}")
    print(f"Figure PDF: {figure_files['pdf']}")
    print(f"Manifest: {manifest_path.resolve()}")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        args.output_dir / "oral_metacyc_pathway_age_association_manifest.json"
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
