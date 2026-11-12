#!/usr/bin/env python
"""Leakage-free cross-validated PLS1 analysis of age across five omics layers.

This is a standalone, auditable implementation of the age model.

Statistical contract
--------------------
* The independent unit is the participant. Repeated rows are averaged within
  participant before any cross-validation split.
* Feature filtering, mean imputation, scaling, PLS1 fitting, component sign,
  and component-score normalization are learned from each training fold only.
* Every participant receives an out-of-fold (OOF) component score and an OOF
  age prediction from a model that did not see that participant.
* The permutation test permutes participant-level ages and refits PLS1 inside
  every fold for every permutation. Label-independent X preprocessing is
  cached within a fold because X and the fixed folds do not change.
* The empirical two-sided P value is
  ``(1 + sum(abs(rho_null) >= abs(rho_observed))) / (B + 1)``.
* The displayed 95% confidence interval is a participant bootstrap of the
  fixed OOF score/age pairs. It is conditional on these fitted folds and does
  not represent full training-set instability.

One component is fixed a priori, so no component-selection loop is needed. If
component count or any other hyperparameter is tuned later, that tuning must be
nested inside each outer training fold.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import rankdata, spearmanr
from sklearn.model_selection import KFold


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "01-interim_data" / "omics_data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_FIGURE_DIR = SCRIPT_DIR / "figures"
# Figure geometry (page width 127 mm). Height can be adjusted freely.
FIG_WIDTH_MM = 127.0
FIG_HEIGHT_MM = 88.0
MM_TO_INCH = 1.0 / 25.4

DEFAULT_DELIVERY_PDF = (
    REPO_ROOT / "output" / "pdf" / "multi_omics_pls_age_spearman_leakage_free_cv.pdf"
)


@dataclass(frozen=True)
class OmicsSpec:
    key: str
    title: str
    color: str
    filename_candidates: tuple[str, ...]


OMICS_SPECS: tuple[OmicsSpec, ...] = (
    OmicsSpec(
        "transcriptomics",
        "Transcriptomics",
        "#CB6E70",
        ("transcriptomics_with_metadata.parquet",),
    ),
    OmicsSpec(
        "metabolomics",
        "Metabolomics",
        "#68AADD",
        ("metabolomics_with_metadata.parquet",),
    ),
    OmicsSpec(
        "lipidomics",
        "Lipidomics",
        "#54AD94",
        ("lipidomics_with_metadata.parquet",),
    ),
    OmicsSpec(
        "gut_microbiome",
        "Gut Microbiome",
        "#DDA062",
        ("gut_microbiome_with_metadata.parquet", "gut_with_metadata.parquet"),
    ),
    OmicsSpec(
        "oral_microbiome",
        "Oral Microbiome",
        "#896EAE",
        ("oral_microbiome_with_metadata.parquet", "oral_with_metadata.parquet"),
    ),
)


PARTICIPANT_COLUMN_CANDIDATES = (
    "participant_id",
    "participant",
    "eid",
    "subject_id",
    "subject",
    "individual_id",
    "individual",
    "id",
)
AGE_COLUMN_CANDIDATES = (
    "age",
    "age_years",
    "chronological_age",
    "age_at_visit",
    "age_at_recruitment",
)

# These are metadata, not molecular measurements. Matching is case-insensitive
# after punctuation is normalized. Additional columns can be excluded by CLI.
KNOWN_METADATA_COLUMNS = {
    "age",
    "age_years",
    "chronological_age",
    "age_at_visit",
    "age_at_recruitment",
    "participant_id",
    "participant",
    "eid",
    "subject_id",
    "subject",
    "individual_id",
    "individual",
    "id",
    "sex",
    "gender",
    "cohort",
    "research_stage",
    "array_index",
    "batch",
    "visit",
    "visit_id",
    "sample_id",
    "sample_type",
    "collection_date",
    "collection_timestamp",
    "collection_datetime",
    "timezone",
    "timepoint",
    "year_of_birth",
    "month_of_birth",
    "bmi",
}


@dataclass
class OmicsData:
    participant_id: np.ndarray
    age: np.ndarray
    x: np.ndarray
    feature_names: np.ndarray
    block: np.ndarray | None
    participant_column: str
    age_column: str
    source_file: str


@dataclass
class FoldData:
    x_train: np.ndarray
    x_test: np.ndarray
    feature_indices: np.ndarray
    means: np.ndarray
    scales: np.ndarray


@dataclass
class CrossFitResult:
    key: str
    title: str
    color: str
    source_file: str
    participant_column: str
    age_column: str
    participant_id: np.ndarray
    age: np.ndarray
    oof_score: np.ndarray
    oof_age_prediction: np.ndarray
    fold_id: np.ndarray
    null_rho: np.ndarray
    n_raw_features: int
    fold_feature_counts: list[int]
    fold_x_variance_pct: list[float]
    rho: float
    ci_low: float
    ci_high: float
    p_permutation: float
    q_fdr: float = math.nan
    q2_cv: float = math.nan


def _normalized_name(value: object) -> str:
    return "_".join(
        str(value).strip().lower().replace("-", "_").replace(" ", "_").split("_")
    )


def _resolve_column(
    columns: Iterable[object], explicit: str | None, candidates: Sequence[str], label: str
) -> str:
    columns = [str(column) for column in columns]
    if explicit:
        if explicit not in columns:
            raise KeyError(f"Requested {label} column {explicit!r} is absent.")
        return explicit
    normalized = {_normalized_name(column): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    raise KeyError(
        f"Could not infer the {label} column. Available columns begin with: {columns[:20]}"
    )


def _resolve_data_file(data_dir: Path, spec: OmicsSpec) -> Path:
    for filename in spec.filename_candidates:
        candidate = data_dir / filename
        if candidate.exists():
            return candidate
    tried = ", ".join(spec.filename_candidates)
    raise FileNotFoundError(f"No input file found for {spec.title}; tried: {tried}")


def load_and_aggregate(
    data_path: Path,
    participant_column: str | None,
    age_column: str | None,
    permutation_block_column: str | None,
    extra_excluded_columns: Sequence[str],
) -> OmicsData:
    """Load one omics table and aggregate repeated rows within participant."""
    frame = pd.read_parquet(data_path)
    participant_column = _resolve_column(
        frame.columns, participant_column, PARTICIPANT_COLUMN_CANDIDATES, "participant"
    )
    age_column = _resolve_column(frame.columns, age_column, AGE_COLUMN_CANDIDATES, "age")
    if permutation_block_column and permutation_block_column not in frame.columns:
        raise KeyError(
            f"Permutation block column {permutation_block_column!r} is absent from {data_path.name}."
        )

    excluded_normalized = set(KNOWN_METADATA_COLUMNS)
    excluded_normalized.update(_normalized_name(column) for column in extra_excluded_columns)
    excluded = {participant_column, age_column}
    if permutation_block_column:
        excluded.add(permutation_block_column)

    feature_columns: list[str] = []
    for column in frame.columns:
        column_name = str(column)
        if column_name in excluded or _normalized_name(column_name) in excluded_normalized:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            feature_columns.append(column_name)
    if not feature_columns:
        raise ValueError(f"No numeric molecular features remained in {data_path.name}.")

    working_columns = [participant_column, age_column, *feature_columns]
    if permutation_block_column:
        working_columns.append(permutation_block_column)
    working = frame.loc[:, working_columns].copy()
    working[age_column] = pd.to_numeric(working[age_column], errors="coerce")

    aggregation: dict[str, str] = {age_column: "mean"}
    aggregation.update({column: "mean" for column in feature_columns})
    if permutation_block_column:
        aggregation[permutation_block_column] = "first"
    grouped = working.groupby(participant_column, sort=True, observed=True).agg(aggregation)
    grouped = grouped.loc[np.isfinite(grouped[age_column].to_numpy(dtype=float))]

    participant_ids = grouped.index.to_numpy(copy=True)
    age = grouped[age_column].to_numpy(dtype=np.float64, copy=True)
    x = grouped[feature_columns].to_numpy(dtype=np.float32, copy=True)
    block = None
    if permutation_block_column:
        block = grouped[permutation_block_column].astype("string").fillna("<missing>").to_numpy()

    del frame, working, grouped
    gc.collect()
    return OmicsData(
        participant_id=participant_ids,
        age=age,
        x=x,
        feature_names=np.asarray(feature_columns, dtype=object),
        block=block,
        participant_column=participant_column,
        age_column=age_column,
        source_file=str(data_path.resolve()),
    )


def prepare_fold(
    x: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
    minimum_nonmissing_fraction: float,
) -> FoldData:
    """Learn feature selection, imputation, and scaling from training rows only."""
    raw_train = np.asarray(x[train_index, :], dtype=np.float32)
    required = max(1, int(math.ceil(minimum_nonmissing_fraction * len(train_index))))
    nonmissing = np.isfinite(raw_train).sum(axis=0)
    keep_missingness = nonmissing >= required
    if not np.any(keep_missingness):
        raise ValueError("All features failed the training-fold missingness threshold.")

    feature_indices = np.flatnonzero(keep_missingness)
    x_train = np.asarray(raw_train[:, keep_missingness], dtype=np.float32)
    del raw_train
    means = np.nanmean(x_train, axis=0, dtype=np.float64).astype(np.float32)
    missing_train = ~np.isfinite(x_train)
    if missing_train.any():
        row_index, column_index = np.nonzero(missing_train)
        x_train[row_index, column_index] = means[column_index]
    del missing_train

    variances = np.var(x_train, axis=0, dtype=np.float64)
    keep_variance = np.isfinite(variances) & (variances > np.finfo(np.float32).eps)
    if not np.any(keep_variance):
        raise ValueError("All retained features had zero training-fold variance.")
    feature_indices = feature_indices[keep_variance]
    x_train = np.asarray(x_train[:, keep_variance], dtype=np.float32)
    means = means[keep_variance]
    scales = np.sqrt(variances[keep_variance]).astype(np.float32)

    x_test = np.asarray(x[np.ix_(test_index, feature_indices)], dtype=np.float32)
    missing_test = ~np.isfinite(x_test)
    if missing_test.any():
        row_index, column_index = np.nonzero(missing_test)
        x_test[row_index, column_index] = means[column_index]
    del missing_test

    x_train -= means
    x_train /= scales
    x_test -= means
    x_test /= scales
    return FoldData(x_train, x_test, feature_indices, means, scales)


def _fit_pls1(
    x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit exact one-response, one-component PLS and score held-out rows."""
    y_train = np.asarray(y_train, dtype=np.float64)
    y_mean = float(np.mean(y_train))
    y_centered = y_train - y_mean
    weights = np.asarray(x_train.T @ y_centered, dtype=np.float64)
    weight_norm = float(np.linalg.norm(weights))
    if not np.isfinite(weight_norm) or weight_norm <= np.finfo(float).eps:
        raise ValueError("Degenerate PLS1 weight vector in a training fold.")
    weights /= weight_norm

    train_score = np.asarray(x_train @ weights, dtype=np.float64)
    test_score = np.asarray(x_test @ weights, dtype=np.float64)
    train_score_mean = float(np.mean(train_score))
    train_score_sd = float(np.std(train_score, ddof=0))
    if not np.isfinite(train_score_sd) or train_score_sd <= np.finfo(float).eps:
        raise ValueError("Degenerate PLS1 training score in a training fold.")
    standardized_test_score = (test_score - train_score_mean) / train_score_sd

    denominator = float(train_score @ train_score)
    y_loading = float(train_score @ y_centered) / denominator
    test_prediction = y_mean + test_score * y_loading

    # Preserve the original notebook's descriptive convention for the title:
    # variance of component scores divided by summed standardized-X variance.
    total_x_variance = float(np.sum(np.var(x_train, axis=0, dtype=np.float64)))
    x_variance_pct = (
        100.0 * float(np.var(train_score, ddof=0)) / total_x_variance
        if total_x_variance > 0
        else math.nan
    )
    return standardized_test_score, test_prediction, weights, x_variance_pct


def _fit_pls1_score_batch(
    x_train: np.ndarray, x_test: np.ndarray, y_train_batch: np.ndarray
) -> np.ndarray:
    """Refit PLS1 for a batch of permuted training-label columns."""
    y_train_batch = np.asarray(y_train_batch, dtype=np.float32)
    y_centered = y_train_batch - np.mean(y_train_batch, axis=0, keepdims=True)
    weights = np.asarray(x_train.T @ y_centered, dtype=np.float32)
    norms = np.linalg.norm(weights, axis=0)
    valid = np.isfinite(norms) & (norms > np.finfo(np.float32).eps)
    safe_norms = np.where(valid, norms, 1.0)
    weights /= safe_norms

    train_scores = np.asarray(x_train @ weights, dtype=np.float32)
    test_scores = np.asarray(x_test @ weights, dtype=np.float32)
    train_means = np.mean(train_scores, axis=0)
    train_sds = np.std(train_scores, axis=0, ddof=0)
    valid &= np.isfinite(train_sds) & (train_sds > np.finfo(np.float32).eps)
    safe_sds = np.where(valid, train_sds, 1.0)
    scores = (test_scores - train_means) / safe_sds
    if not np.all(valid):
        scores[:, ~valid] = np.nan
    return scores


def make_permutation_indices(
    n: int, n_permutations: int, rng: np.random.Generator, block: np.ndarray | None
) -> np.ndarray:
    """Create participant-level unrestricted or within-block label permutations."""
    permutations = np.empty((n_permutations, n), dtype=np.int32)
    base = np.arange(n, dtype=np.int32)
    if block is None:
        for row in range(n_permutations):
            permutations[row] = rng.permutation(base)
        return permutations

    block_codes, _ = pd.factorize(block, sort=True)
    block_members = [np.flatnonzero(block_codes == code) for code in np.unique(block_codes)]
    for row in range(n_permutations):
        permutation = base.copy()
        for members in block_members:
            permutation[members] = rng.permutation(members)
        permutations[row] = permutation
    return permutations


def _spearman_columns(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Column-wise Spearman correlations with bounded temporary memory."""
    if left.shape != right.shape:
        raise ValueError(f"Spearman arrays have different shapes: {left.shape} vs {right.shape}")
    left_rank = rankdata(left, axis=0)
    right_rank = rankdata(right, axis=0)
    left_rank -= np.mean(left_rank, axis=0, keepdims=True)
    right_rank -= np.mean(right_rank, axis=0, keepdims=True)
    numerator = np.sum(left_rank * right_rank, axis=0)
    denominator = np.sqrt(
        np.sum(left_rank * left_rank, axis=0) * np.sum(right_rank * right_rank, axis=0)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(left.shape[1], np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def bootstrap_spearman_ci(
    score: np.ndarray,
    age: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    estimates = np.empty(n_bootstrap, dtype=np.float64)
    n = len(age)
    for index in range(n_bootstrap):
        sample = rng.integers(0, n, size=n)
        estimates[index] = spearmanr(score[sample], age[sample]).statistic
    finite = estimates[np.isfinite(estimates)]
    if len(finite) < max(20, n_bootstrap // 2):
        return math.nan, math.nan
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def crossfit_pls1_refit_permutation(
    spec: OmicsSpec,
    data: OmicsData,
    n_splits: int,
    n_permutations: int,
    n_bootstrap: int,
    permutation_batch_size: int,
    minimum_nonmissing_fraction: float,
    seed: int,
) -> CrossFitResult:
    """Run OOF PLS1 and a participant-level, fully refitted permutation test."""
    n = len(data.age)
    if n < n_splits:
        raise ValueError(f"{spec.title} has {n} participants, fewer than {n_splits} folds.")
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.arange(n)))

    rng = np.random.default_rng(seed)
    permutation_indices = make_permutation_indices(n, n_permutations, rng, data.block)
    oof_score = np.full(n, np.nan, dtype=np.float64)
    oof_prediction = np.full(n, np.nan, dtype=np.float64)
    fold_id = np.full(n, -1, dtype=np.int16)
    permuted_oof_score = np.full((n, n_permutations), np.nan, dtype=np.float32)
    fold_feature_counts: list[int] = []
    fold_x_variance_pct: list[float] = []

    for fold_number, (train_index, test_index) in enumerate(folds, start=1):
        prepared = prepare_fold(
            data.x, train_index, test_index, minimum_nonmissing_fraction
        )
        score, prediction, _, x_variance_pct = _fit_pls1(
            prepared.x_train, prepared.x_test, data.age[train_index]
        )
        oof_score[test_index] = score
        oof_prediction[test_index] = prediction
        fold_id[test_index] = fold_number
        fold_feature_counts.append(int(len(prepared.feature_indices)))
        fold_x_variance_pct.append(float(x_variance_pct))

        for start in range(0, n_permutations, permutation_batch_size):
            stop = min(start + permutation_batch_size, n_permutations)
            batch_indices = permutation_indices[start:stop, :][:, train_index]
            permuted_y_train = data.age[batch_indices].T
            batch_scores = _fit_pls1_score_batch(
                prepared.x_train, prepared.x_test, permuted_y_train
            )
            permuted_oof_score[test_index, start:stop] = batch_scores
        print(
            f"  fold {fold_number}/{n_splits}: train={len(train_index)}, "
            f"test={len(test_index)}, retained_features={len(prepared.feature_indices):,}",
            flush=True,
        )
        del prepared
        gc.collect()

    if not np.all(np.isfinite(oof_score)) or not np.all(np.isfinite(oof_prediction)):
        raise RuntimeError(f"Non-finite observed OOF results for {spec.title}.")

    rho = float(spearmanr(oof_score, data.age).statistic)
    null_rho = np.empty(n_permutations, dtype=np.float64)
    for start in range(0, n_permutations, permutation_batch_size):
        stop = min(start + permutation_batch_size, n_permutations)
        permuted_age = data.age[permutation_indices[start:stop, :]].T
        null_rho[start:stop] = _spearman_columns(
            permuted_oof_score[:, start:stop], permuted_age
        )
    finite_null = null_rho[np.isfinite(null_rho)]
    if len(finite_null) != n_permutations:
        raise RuntimeError(
            f"Only {len(finite_null)}/{n_permutations} finite permutation statistics for {spec.title}."
        )
    extreme = int(np.sum(np.abs(finite_null) >= abs(rho)))
    p_permutation = float((1 + extreme) / (n_permutations + 1))
    ci_low, ci_high = bootstrap_spearman_ci(
        oof_score, data.age, n_bootstrap, np.random.default_rng(seed + 1)
    )
    denominator = float(np.sum((data.age - np.mean(data.age)) ** 2))
    q2_cv = 1.0 - float(np.sum((data.age - oof_prediction) ** 2)) / denominator

    del permuted_oof_score, permutation_indices
    gc.collect()
    return CrossFitResult(
        key=spec.key,
        title=spec.title,
        color=spec.color,
        source_file=data.source_file,
        participant_column=data.participant_column,
        age_column=data.age_column,
        participant_id=data.participant_id,
        age=data.age,
        oof_score=oof_score,
        oof_age_prediction=oof_prediction,
        fold_id=fold_id,
        null_rho=null_rho,
        n_raw_features=int(data.x.shape[1]),
        fold_feature_counts=fold_feature_counts,
        fold_x_variance_pct=fold_x_variance_pct,
        rho=rho,
        ci_low=ci_low,
        ci_high=ci_high,
        p_permutation=p_permutation,
        q2_cv=q2_cv,
    )


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def _format_probability(value: float, label: str) -> str:
    if value < 0.001:
        return rf"{label} < 0.001"
    return rf"{label} = {value:.3f}"


def _significance_stars(q_value: float) -> str:
    if q_value <= 0.001:
        return "***"
    if q_value <= 0.01:
        return "**"
    if q_value <= 0.05:
        return "*"
    return ""


def add_vector_hexbin(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    gridsize: int = 35,
    extent: Sequence[float] | None = None,
) -> PolyCollection:
    """Draw explicit vector hexagons, avoiding backend offset-rendering defects."""
    cmap = LinearSegmentedColormap.from_list("hex_fade", ["#FFFFFF", color])
    original = ax.hexbin(
        x,
        y,
        gridsize=gridsize,
        mincnt=1,
        cmap=cmap,
        extent=extent,
        linewidths=0,
        edgecolors="none",
        rasterized=False,
    )
    offsets = np.asarray(original.get_offsets())
    paths = original.get_paths()
    if not paths or len(offsets) == 0:
        return original
    values = np.asarray(original.get_array())
    facecolors = original.cmap(original.norm(values))
    base_vertices = np.asarray(paths[0].vertices)
    polygons = [base_vertices + offset for offset in offsets]
    explicit = PolyCollection(
        polygons,
        facecolors=facecolors,
        edgecolors="none",
        linewidths=0,
        rasterized=False,
    )
    original.remove()
    ax.add_collection(explicit)
    return explicit


def _linear_trend_with_bootstrap_band(
    age: np.ndarray,
    score: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = np.linspace(float(np.min(age)), float(np.max(age)), 160)
    slope, intercept = np.polyfit(age, score, 1)
    line = intercept + slope * grid
    bootstrap_lines = np.empty((n_bootstrap, len(grid)), dtype=np.float64)
    n = len(age)
    for index in range(n_bootstrap):
        sample = rng.integers(0, n, size=n)
        boot_slope, boot_intercept = np.polyfit(age[sample], score[sample], 1)
        bootstrap_lines[index] = boot_intercept + boot_slope * grid
    low, high = np.percentile(bootstrap_lines, [2.5, 97.5], axis=0)
    return grid, line, low, high


FONT = "Arial"


def _register_arial() -> None:
    """Register Arial (same approach as 01-cohort_feature.ipynb) and fail loudly if absent."""
    for font_file in (Path.home() / ".fonts").glob("*.ttf"):
        font_manager.fontManager.addfont(str(font_file))
    font_manager.findfont(font_manager.FontProperties(family=FONT), fallback_to_default=False)
    font_manager.findfont(
        font_manager.FontProperties(family=FONT, weight="bold"), fallback_to_default=False
    )


def _configure_matplotlib() -> None:
    _register_arial()
    mpl.rcParams.update(
        {
            "font.family": FONT,
            # Subscripts such as ρ_CV are typeset with \mathrm, i.e. upright Arial.
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT,
            "mathtext.it": FONT,
            "mathtext.bf": f"{FONT}:bold",
            "mathtext.cal": FONT,
            "mathtext.sf": FONT,
            "font.size": 6.5,
            "axes.titlesize": 7.0,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "xtick.major.pad": 1.5,
            "ytick.major.pad": 1.5,
            "axes.labelpad": 2.0,
            "axes.titlepad": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            # No "tight" bbox: the saved figure must stay exactly FIG_WIDTH_MM wide.
        }
    )


def create_original_style_figure(
    results: Sequence[CrossFitResult],
    pdf_path: Path,
    svg_path: Path,
    png_path: Path,
    seed: int,
) -> None:
    """Recreate the notebook's 2 x 3 layout using leakage-free OOF results.

    Panel titles carry only the omics name; the compact in-panel box shows
    rho_CV (stars = BH-FDR q) and the mean fold X variance. Exact p_perm,
    q_FDR and Q2_CV are written to the summary CSV.
    """
    _configure_matplotlib()
    fig, axes = plt.subplots(
        2, 3, figsize=(FIG_WIDTH_MM * MM_TO_INCH, FIG_HEIGHT_MM * MM_TO_INCH), dpi=300
    )
    # Fixed margins keep the three columns aligned and the width exactly 127 mm.
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.085, top=0.955,
                        wspace=0.30, hspace=0.42)
    flat_axes = axes.ravel()

    for index, (axis, result) in enumerate(zip(flat_axes[:5], results)):
        y_low, y_high = np.percentile(result.oof_score, [5, 99])
        y_padding = max((y_high - y_low) * 0.08, np.finfo(float).eps)
        y_min, y_max = y_low - y_padding, y_high + y_padding
        x_padding = (float(np.max(result.age)) - float(np.min(result.age))) * 0.05
        x_min = float(np.min(result.age)) - x_padding
        x_max = float(np.max(result.age)) + x_padding
        add_vector_hexbin(
            axis,
            result.age,
            result.oof_score,
            result.color,
            extent=[x_min, x_max, y_min, y_max],
        )
        grid, line, low, high = _linear_trend_with_bootstrap_band(
            result.age,
            result.oof_score,
            np.random.default_rng(seed + 10_000 + index),
        )
        axis.fill_between(grid, low, high, color=result.color, alpha=0.14, linewidth=0)
        axis.plot(grid, line, color="black", linestyle="--", linewidth=1.0)
        mean_x_variance = float(np.mean(result.fold_x_variance_pct))
        axis.set_title(result.title)
        axis.set_xlabel("Age (years)")
        if index in (0, 3):
            axis.set_ylabel("OOF PLS Comp. 1 score")
        annotation = (
            f"ρ$_{{\\mathrm{{CV}}}}$ = {result.rho:.2f}{_significance_stars(result.q_fdr)}" + "\n"
            + f"X var {mean_x_variance:.1f}%"
        )
        axis.text(
            0.04,
            0.97,
            annotation,
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=5.5,
            linespacing=1.15,
            zorder=7,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9,
                  "edgecolor": "#999999", "linewidth": 0.4},
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4, integer=True))
        axis.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)

    summary_axis = flat_axes[5]
    summary_order = [3, 0, 1, 2, 4]
    ordered = [results[index] for index in summary_order]
    positions = np.arange(len(ordered))
    values = np.asarray([result.rho for result in ordered])
    lower = values - np.asarray([result.ci_low for result in ordered])
    upper = np.asarray([result.ci_high for result in ordered]) - values
    summary_axis.barh(
        positions,
        values,
        color=[result.color for result in ordered],
        alpha=0.72,
        edgecolor=[result.color for result in ordered],
        linewidth=0.5,
        height=0.42,
    )
    summary_axis.errorbar(
        values,
        positions,
        xerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="#444444",
        elinewidth=0.5,
        capsize=1.5,
        capthick=0.5,
    )
    summary_axis.axvline(0, color="#777777", linestyle="--", linewidth=0.4)
    # Omics names sit inside the panel above each bar, so the summary axes keep
    # the same left edge as the column above.
    summary_axis.set_yticks([])
    summary_axis.spines["left"].set_visible(False)
    summary_axis.invert_yaxis()
    summary_axis.set_title("Summary")
    summary_axis.set_xlabel(r"Spearman ρ$_{\mathrm{CV}}$ (95% CI)")
    max_upper = float(np.nanmax([result.ci_high for result in ordered]))
    x_left = min(-0.02, float(np.nanmin([result.ci_low for result in ordered])) - 0.02)
    x_right = max_upper * 1.30
    for position, result in zip(positions, ordered):
        summary_axis.text(
            0.015 * max_upper,
            position - 0.27,
            result.title,
            va="bottom",
            ha="left",
            fontsize=5.5,
            color="#222222",
        )
        summary_axis.text(
            result.ci_high + 0.03 * max_upper,
            position,
            f"{result.rho:.2f}{_significance_stars(result.q_fdr)}",
            va="center",
            ha="left",
            fontsize=5.5,
            color="#222222",
        )
    summary_axis.set_xlim(x_left, x_right)
    summary_axis.set_ylim(len(ordered) - 0.5, -0.85)
    summary_axis.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    summary_axis.grid(axis="x", color="#DDDDDD", linewidth=0.3)
    summary_axis.set_axisbelow(True)
    summary_axis.spines["top"].set_visible(False)
    summary_axis.spines["right"].set_visible(False)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(pdf_path, format="pdf", facecolor="white")
    fig.savefig(svg_path, format="svg", facecolor="white")
    fig.savefig(png_path, format="png", dpi=600, facecolor="white")
    plt.close(fig)


def _summary_frame(results: Sequence[CrossFitResult], n_permutations: int) -> pd.DataFrame:
    records = []
    for result in results:
        extreme = int(np.sum(np.abs(result.null_rho) >= abs(result.rho)))
        records.append(
            {
                "omics": result.title,
                "n_participants": len(result.age),
                "n_raw_features": result.n_raw_features,
                "fold_features_mean": float(np.mean(result.fold_feature_counts)),
                "fold_features_min": int(np.min(result.fold_feature_counts)),
                "fold_features_max": int(np.max(result.fold_feature_counts)),
                "fold_x_variance_pct_mean": float(np.mean(result.fold_x_variance_pct)),
                "rho_cv": result.rho,
                "rho_cv_ci_low": result.ci_low,
                "rho_cv_ci_high": result.ci_high,
                "p_permutation_two_sided": result.p_permutation,
                "q_fdr_bh": result.q_fdr,
                "permutation_extreme_count": extreme,
                "n_permutations": n_permutations,
                "q2_cv": result.q2_cv,
                "source_file": result.source_file,
            }
        )
    return pd.DataFrame.from_records(records)


def save_tabular_outputs(
    results: Sequence[CrossFitResult], output_dir: Path, suffix: str, n_permutations: int
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"multi_omics_pls_age_{suffix}_summary.csv"
    scores_path = output_dir / f"multi_omics_pls_age_{suffix}_oof_scores.parquet"
    null_path = output_dir / f"multi_omics_pls_age_{suffix}_permutation_null.parquet"

    _summary_frame(results, n_permutations).to_csv(summary_path, index=False)
    score_frames = []
    null_frames = []
    for result in results:
        score_frames.append(
            pd.DataFrame(
                {
                    "omics": result.title,
                    "participant_id": result.participant_id,
                    "age": result.age,
                    "oof_pls1_score": result.oof_score,
                    "oof_age_prediction": result.oof_age_prediction,
                    "outer_fold": result.fold_id,
                }
            )
        )
        null_frames.append(
            pd.DataFrame(
                {
                    "omics": result.title,
                    "permutation": np.arange(1, len(result.null_rho) + 1),
                    "rho_null": result.null_rho,
                }
            )
        )
    pd.concat(score_frames, ignore_index=True).to_parquet(scores_path, index=False)
    pd.concat(null_frames, ignore_index=True).to_parquet(null_path, index=False)
    return {"summary": summary_path, "scores": scores_path, "null": null_path}


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    results: Sequence[CrossFitResult],
    outputs: dict[str, Path],
) -> None:
    metadata = {
        "analysis": "Leakage-free participant-level cross-fitted one-component PLS",
        "independent_unit": "participant",
        "outer_cv": {
            "splitter": "KFold(shuffle=True)",
            "n_splits": args.n_splits,
            "seed": args.seed,
            "fixed_components": 1,
        },
        "training_fold_only_operations": [
            "feature missingness filter",
            "mean imputation",
            "zero-variance filter",
            "feature scaling",
            "PLS1 supervised weight fit",
            "component orientation",
            "component-score normalization",
        ],
        "permutation_test": {
            "n_permutations": args.n_permutations,
            "level": "participant",
            "scheme": (
                f"within {args.permutation_block_column}"
                if args.permutation_block_column
                else "unrestricted"
            ),
            "refit": "PLS1 refit in every outer fold for every permutation",
            "cached_operation": "label-independent training-fold X preprocessing only",
            "statistic": "two-sided absolute OOF Spearman rho",
            "empirical_p_formula": "(1 + extreme_count) / (B + 1)",
        },
        "confidence_interval": {
            "method": "percentile participant bootstrap of fixed OOF score-age pairs",
            "n_bootstrap": args.n_bootstrap,
            "limitation": "conditional on fitted folds; does not include full training-set instability",
        },
        "multiplicity": "Benjamini-Hochberg FDR across the five omics tests",
        "exchangeability_assumption": (
            "Ages are exchangeable within the selected permutation blocks."
            if args.permutation_block_column
            else "Participant ages are exchangeable under the null; use --permutation-block-column "
            "when cohort/batch restrictions are required."
        ),
        "parameters": {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
        "datasets": [
            {
                "key": result.key,
                "title": result.title,
                "source_file": result.source_file,
                "participant_column": result.participant_column,
                "age_column": result.age_column,
                "n_participants": len(result.age),
                "n_raw_features": result.n_raw_features,
                "fold_feature_counts": result.fold_feature_counts,
                "fold_x_variance_pct": result.fold_x_variance_pct,
            }
            for result in results
        ],
        "outputs": {name: str(path.resolve()) for name, path in outputs.items()},
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def run_self_test() -> None:
    """Fast numerical and leakage-boundary checks for the core implementation."""
    from sklearn.cross_decomposition import PLSRegression

    rng = np.random.default_rng(20260901)
    x = rng.normal(size=(72, 11)).astype(np.float32)
    x[rng.random(x.shape) < 0.08] = np.nan
    y = 50.0 + np.nan_to_num(x[:, 0]) * 2.0 + rng.normal(size=len(x))
    train = np.arange(54)
    test = np.arange(54, 72)

    prepared = prepare_fold(x, train, test, 0.5)
    score, prediction, weights, _ = _fit_pls1(prepared.x_train, prepared.x_test, y[train])
    reference = PLSRegression(n_components=1, scale=False, max_iter=1000)
    reference.fit(prepared.x_train, y[train])
    reference_train_score = reference.transform(prepared.x_train)[:, 0]
    reference_test_score = reference.transform(prepared.x_test)[:, 0]
    orientation = np.sign(np.corrcoef(reference_train_score, y[train])[0, 1])
    reference_test_score *= orientation
    reference_train_score *= orientation
    reference_standardized = (
        reference_test_score - np.mean(reference_train_score)
    ) / np.std(reference_train_score)
    if not np.allclose(score, reference_standardized, atol=2e-5, rtol=2e-5):
        raise AssertionError("Closed-form PLS1 scores disagree with sklearn PLSRegression.")
    if not np.allclose(prediction, reference.predict(prepared.x_test).ravel(), atol=2e-5):
        raise AssertionError("Closed-form PLS1 predictions disagree with sklearn PLSRegression.")
    if not np.isfinite(weights).all():
        raise AssertionError("PLS1 weights contain non-finite values.")

    changed = x.copy()
    changed[test] = np.nan_to_num(changed[test], nan=0.0) + 1_000_000.0
    prepared_changed = prepare_fold(changed, train, test, 0.5)
    if not np.array_equal(prepared.feature_indices, prepared_changed.feature_indices):
        raise AssertionError("Held-out X changed training-only feature selection.")
    if not np.array_equal(prepared.means, prepared_changed.means):
        raise AssertionError("Held-out X changed training-only imputation means.")
    if not np.array_equal(prepared.scales, prepared_changed.scales):
        raise AssertionError("Held-out X changed training-only scaling.")
    if not np.array_equal(prepared.x_train, prepared_changed.x_train):
        raise AssertionError("Held-out X changed standardized training data.")

    y_batch = np.column_stack([rng.permutation(y[train]), rng.permutation(y[train])])
    refitted_scores = _fit_pls1_score_batch(
        prepared.x_train, prepared.x_test, y_batch
    )
    if np.allclose(refitted_scores[:, 0], refitted_scores[:, 1]):
        raise AssertionError("Different permutations did not refit to different PLS scores.")
    q_values = benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
    if not np.all((q_values >= 0) & (q_values <= 1)):
        raise AssertionError("Benjamini-Hochberg output is invalid.")
    print("Self-test passed: PLS1 equivalence, leakage boundary, refit, and FDR checks.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--delivery-pdf", type=Path, default=DEFAULT_DELIVERY_PDF)
    parser.add_argument(
        "--omics",
        nargs="+",
        choices=[spec.key for spec in OMICS_SPECS],
        default=[spec.key for spec in OMICS_SPECS],
    )
    parser.add_argument("--participant-column", default=None)
    parser.add_argument("--age-column", default=None)
    parser.add_argument("--permutation-block-column", default=None)
    parser.add_argument("--exclude-column", action="append", default=[])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-permutations", type=int, default=999)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--permutation-batch-size", type=int, default=32)
    parser.add_argument("--minimum-nonmissing-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suffix", default="leakage_free_cv")
    parser.add_argument("--self-test", action="store_true")
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least 2.")
    if args.n_permutations < 1:
        raise ValueError("--n-permutations must be positive.")
    if args.n_bootstrap < 20:
        raise ValueError("--n-bootstrap must be at least 20.")
    if args.permutation_batch_size < 1:
        raise ValueError("--permutation-batch-size must be positive.")
    if not 0 < args.minimum_nonmissing_fraction <= 1:
        raise ValueError("--minimum-nonmissing-fraction must lie in (0, 1].")
    if len(args.omics) != 5:
        raise ValueError(
            "The original-style 2 x 3 figure requires all five omics layers; "
            "omit --omics for the complete analysis."
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    validate_arguments(args)

    selected = {key for key in args.omics}
    specs = [spec for spec in OMICS_SPECS if spec.key in selected]
    results: list[CrossFitResult] = []
    for index, spec in enumerate(specs):
        path = _resolve_data_file(args.data_dir, spec)
        print(f"[{index + 1}/{len(specs)}] {spec.title}: loading {path}", flush=True)
        data = load_and_aggregate(
            path,
            args.participant_column,
            args.age_column,
            args.permutation_block_column,
            args.exclude_column,
        )
        print(
            f"  participants={len(data.age):,}; candidate_features={data.x.shape[1]:,}; "
            f"participant={data.participant_column!r}; age={data.age_column!r}",
            flush=True,
        )
        result = crossfit_pls1_refit_permutation(
            spec,
            data,
            n_splits=args.n_splits,
            n_permutations=args.n_permutations,
            n_bootstrap=args.n_bootstrap,
            permutation_batch_size=args.permutation_batch_size,
            minimum_nonmissing_fraction=args.minimum_nonmissing_fraction,
            seed=args.seed + 1000 * index,
        )
        results.append(result)
        print(
            f"  rho_CV={result.rho:.4f}; p_perm={result.p_permutation:.4g}; "
            f"Q2_CV={result.q2_cv:.4f}",
            flush=True,
        )
        del data
        gc.collect()

    q_values = benjamini_hochberg([result.p_permutation for result in results])
    for result, q_value in zip(results, q_values):
        result.q_fdr = float(q_value)

    stem = f"multi_omics_pls_age_spearman_{args.suffix}"
    pdf_path = args.figure_dir / f"{stem}.pdf"
    svg_path = args.figure_dir / f"{stem}.svg"
    png_path = args.figure_dir / f"{stem}.png"
    create_original_style_figure(
        results,
        pdf_path,
        svg_path,
        png_path,
        seed=args.seed,
    )
    outputs = save_tabular_outputs(results, args.output_dir, args.suffix, args.n_permutations)
    outputs.update({"figure_pdf": pdf_path, "figure_svg": svg_path, "figure_png": png_path})
    metadata_path = args.output_dir / f"multi_omics_pls_age_{args.suffix}_metadata.json"
    outputs["metadata"] = metadata_path
    write_metadata(metadata_path, args, results, outputs)

    args.delivery_pdf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf_path, args.delivery_pdf)
    print("\nFinal summary", flush=True)
    print(_summary_frame(results, args.n_permutations).to_string(index=False), flush=True)
    print(f"\nVector PDF: {args.delivery_pdf.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
