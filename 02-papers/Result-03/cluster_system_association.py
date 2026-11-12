"""Associate cross-omics nonlinear molecular-cluster scores with 12 systems.

The pipeline is deliberately participant based:

1. aggregate repeated omics rows per participant;
2. z-score every molecule within its own omics layer;
3. average molecules belonging to the same cluster, then z-score that omics
   cluster score so that large omics layers cannot dominate;
4. average all omics scores available for a participant and correct the mean
   for its correlation-derived variance inflation;
5. fit ``system feature ~ cluster score + age + sex`` and report the
   standardized coefficient, partial r, two-sided P value and BH-FDR.

System parquet files are not required until ``main`` is run, so all score and
association functions can also be imported and used from a notebook.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
META_COLS = [
    "participant_id", "cohort", "research_stage", "array_index",
    "collection_date", "collection_timestamp", "timezone", "age",
    "year_of_birth", "month_of_birth", "sex", "collection_datetime",
    "batch", "bmi",
]

OMICS_FILES = {
    "Transcriptomics": "transcriptomics_with_metadata.parquet",
    "Metabolomics": "metabolomics_with_metadata.parquet",
    "Lipidomics": "lipidomics_with_metadata.parquet",
    "Gut microbiome": "gut_microbiome_with_metadata.parquet",
    "Oral microbiome": "oral_microbiome_with_metadata.parquet",
}

SYSTEM_FILES = {
    "cardio": "020-cardiovascular_system_with_metadata.parquet",
    "body_comp": "body_composition_with_metadata.parquet",
    "bone_density": "bone_density_with_metadata.parquet",
    "diet": "diet_with_metadata.parquet",
    "sleep": "sleep_with_metadata.parquet",
    "liver_health": "liver_health_with_metadata.parquet",
    "cgm": "cgm_with_metadata.parquet",
    "frailty": "frailty_with_metadata.parquet",
    "life_style": "life_style_with_metadata.parquet",
    "immune": "immune_system_with_metadata.parquet",
    "hema": "hematology_system_with_metadata.parquet",
    "renal_function": "renal_function_with_metadata.parquet",
}


@dataclass
class ClusterScoreResult:
    """All reusable products of cross-omics cluster-score construction."""

    scores: pd.DataFrame
    omics_scores: pd.DataFrame
    variance_diagnostics: pd.DataFrame
    participant_covariates: pd.DataFrame


def _available_parquet_columns(path: str | Path) -> list[str]:
    """Read only parquet metadata; this avoids loading a wide omics table."""
    try:
        import pyarrow.parquet as pq
    except ImportError:  # pragma: no cover - normal project environment has it
        return list(pd.read_parquet(path).columns)
    return pq.ParquetFile(path).schema.names


def load_omics(path: str | Path, wanted_features: Sequence[str] | None = None) -> pd.DataFrame:
    """Load metadata plus requested features, silently ignoring absent columns."""
    available = _available_parquet_columns(path)
    requested = META_COLS if wanted_features is None else META_COLS + list(wanted_features)
    columns = list(dict.fromkeys(c for c in requested if c in available))
    if "participant_id" not in columns:
        raise ValueError(f"{path} has no participant_id column")
    df = pd.read_parquet(path, columns=columns)
    df["participant_id"] = df["participant_id"].astype("string")
    return df


def load_system(
    path: str | Path,
    extra_drop: Sequence[str] | None = None,
    age_range: tuple[float, float] | None = (40.0, 70.0),
) -> tuple[pd.DataFrame, list[str]]:
    """Load one system; columns not identified as metadata are features."""
    df = pd.read_parquet(path)
    if "participant_id" not in df:
        raise ValueError(f"{path} has no participant_id column")
    df["participant_id"] = df["participant_id"].astype("string")
    drop = [c for c in (extra_drop or ()) if c in df]
    if drop:
        df = df.drop(columns=drop)
    if age_range is not None and "age" in df:
        age = pd.to_numeric(df["age"], errors="coerce")
        df = df.loc[age.between(*age_range)].copy()
    feature_cols = [c for c in df if c not in META_COLS]
    return df, feature_cols


def _zscore_series(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").astype(float)
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd <= np.finfo(float).eps:
        return pd.Series(np.nan, index=x.index, dtype=float)
    return (x - x.mean()) / sd


def _mode_or_nan(values: pd.Series):
    mode = values.dropna().mode()
    return mode.iloc[0] if not mode.empty else np.nan


def build_participant_covariates(
    omics_tables: Mapping[str, pd.DataFrame],
    id_col: str = "participant_id",
) -> pd.DataFrame:
    """Pool age/sex across omics, useful when a system lacks those columns."""
    pieces = []
    for df in omics_tables.values():
        cols = [c for c in (id_col, "age", "sex") if c in df]
        if id_col in cols and len(cols) > 1:
            pieces.append(df[cols].copy())
    if not pieces:
        return pd.DataFrame(columns=[id_col, "age", "sex"])
    pooled = pd.concat(pieces, ignore_index=True, sort=False)
    pooled[id_col] = pooled[id_col].astype("string")
    result = pd.DataFrame({id_col: pooled[id_col].dropna().unique()})
    if "age" in pooled:
        age = pd.to_numeric(pooled["age"], errors="coerce")
        age_by_id = age.groupby(pooled[id_col]).median()
        result["age"] = result[id_col].map(age_by_id)
    if "sex" in pooled:
        sex_by_id = pooled.groupby(id_col, dropna=True)["sex"].agg(_mode_or_nan)
        result["sex"] = result[id_col].map(sex_by_id)
    return result


def _omics_cluster_score(
    df: pd.DataFrame,
    features: Sequence[str],
    id_col: str,
    min_features: int,
) -> pd.Series | None:
    """Feature-z-score -> row mean -> participant mean -> omics-score z-score."""
    present = [f for f in features if f in df]
    if not present:
        return None
    numeric = df[present].apply(pd.to_numeric, errors="coerce")
    # Equal participant weighting: collapse repeated measurements before scaling.
    participant_features = numeric.groupby(df[id_col], dropna=True).mean()
    z_features = participant_features.apply(_zscore_series, axis=0)
    observed = z_features.notna().sum(axis=1)
    raw_score = z_features.mean(axis=1, skipna=True).where(observed >= min_features)
    score = _zscore_series(raw_score)
    score.index = score.index.astype("string")
    return score


def _pairwise_score_correlation(wide: pd.DataFrame) -> pd.DataFrame:
    corr = wide.corr(min_periods=3).fillna(0.0)
    for col in corr.columns:
        corr.loc[col, col] = 1.0
    return corr


def build_cluster_scores(
    omics_tables: Mapping[str, pd.DataFrame],
    cluster_table: pd.DataFrame,
    *,
    id_col: str = "participant_id",
    omics_col: str = "omics",
    feature_col: str = "feature",
    cluster_col: str = "cluster_id",
    min_features_per_score: int = 1,
) -> ClusterScoreResult:
    """Build variance-corrected participant scores for every molecular cluster.

    If a participant has a subset A of k omics scores with correlation matrix R,
    ``Var(mean) = 1' R 1 / k^2``.  The returned score is mean/sqrt(Var(mean)).
    ``variance_inflation_factor`` is relative to independent omics (variance 1/k).
    """
    required = {feature_col, omics_col, cluster_col}
    missing = required.difference(cluster_table.columns)
    if missing:
        raise ValueError(f"cluster_table is missing columns: {sorted(missing)}")
    if min_features_per_score < 1:
        raise ValueError("min_features_per_score must be >= 1")

    table = cluster_table.dropna(subset=[feature_col, omics_col, cluster_col]).copy()
    table[feature_col] = table[feature_col].astype(str)
    long_parts: list[pd.DataFrame] = []
    for (cluster_id, omics), group in table.groupby([cluster_col, omics_col], sort=True):
        if omics not in omics_tables:
            warnings.warn(f"Cluster assignments contain unavailable omics {omics!r}; skipped")
            continue
        score = _omics_cluster_score(
            omics_tables[omics], group[feature_col].unique(), id_col,
            min_features_per_score,
        )
        if score is None:
            warnings.warn(f"No assigned features found for cluster={cluster_id}, omics={omics}")
            continue
        long_parts.append(pd.DataFrame({
            id_col: score.index,
            cluster_col: cluster_id,
            omics_col: omics,
            "omics_cluster_score": score.to_numpy(),
            "n_cluster_features_present": sum(
                f in omics_tables[omics] for f in group[feature_col].unique()
            ),
        }))
    if not long_parts:
        raise ValueError("No cluster scores could be constructed from the supplied data")
    omics_scores = pd.concat(long_parts, ignore_index=True)

    score_parts: list[pd.DataFrame] = []
    diagnostic_parts: list[pd.DataFrame] = []
    for cluster_id, cluster_long in omics_scores.groupby(cluster_col, sort=True):
        wide = cluster_long.pivot(index=id_col, columns=omics_col, values="omics_cluster_score")
        corr = _pairwise_score_correlation(wide)
        corr_long = corr.rename_axis("omics_1").reset_index().melt(
            id_vars="omics_1", var_name="omics_2", value_name="score_correlation"
        )
        corr_long[cluster_col] = cluster_id
        diagnostic_parts.append(corr_long)

        rows = []
        for participant_id, values in wide.iterrows():
            available = values.dropna()
            k = len(available)
            if k == 0:
                continue
            names = available.index.tolist()
            r_sub = corr.loc[names, names].to_numpy(dtype=float)
            variance_of_mean = float(r_sub.sum() / (k * k))
            if not np.isfinite(variance_of_mean) or variance_of_mean <= 1e-12:
                variance_of_mean = 1.0 / k
            naive_mean = float(available.mean())
            rows.append({
                id_col: participant_id,
                cluster_col: cluster_id,
                "cluster_score": naive_mean / np.sqrt(variance_of_mean),
                "uncorrected_omics_mean": naive_mean,
                "variance_of_omics_mean": variance_of_mean,
                "variance_inflation_factor": k * variance_of_mean,
                "n_available_omics": k,
                "available_omics": "|".join(map(str, names)),
            })
        score_parts.append(pd.DataFrame(rows))

    scores = pd.concat(score_parts, ignore_index=True)
    pattern_cols = [c for c in (cluster_col, "pattern_id", "pattern") if c in table]
    if len(pattern_cols) > 1:
        annotations = table[pattern_cols].drop_duplicates(subset=[cluster_col])
        scores = scores.merge(annotations, on=cluster_col, how="left")
    diagnostics = pd.concat(diagnostic_parts, ignore_index=True)
    covariates = build_participant_covariates(omics_tables, id_col=id_col)
    return ClusterScoreResult(scores, omics_scores, diagnostics, covariates)


def _aggregate_system_participants(
    system_df: pd.DataFrame,
    feature_cols: Sequence[str],
    id_col: str,
) -> pd.DataFrame:
    """Collapse repeated system rows without treating visits as independent."""
    data = system_df.copy()
    data[id_col] = data[id_col].astype("string")
    numeric_cols = [c for c in list(feature_cols) + ["age"] if c in data]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    aggregations = {c: "mean" for c in numeric_cols}
    if "sex" in data:
        aggregations["sex"] = _mode_or_nan
    return data.groupby(id_col, as_index=False, dropna=True).agg(aggregations)


def _coalesce_covariates(
    system: pd.DataFrame,
    participant_covariates: pd.DataFrame | None,
    id_col: str,
) -> pd.DataFrame:
    if participant_covariates is None or participant_covariates.empty:
        return system
    external = participant_covariates.copy()
    external[id_col] = external[id_col].astype("string")
    merged = system.merge(external, on=id_col, how="left", suffixes=("", "_omics"))
    for col in ("age", "sex"):
        fallback = f"{col}_omics"
        if fallback not in merged:
            continue
        if col in merged:
            merged[col] = merged[col].where(merged[col].notna(), merged[fallback])
        else:
            merged[col] = merged[fallback]
        merged = merged.drop(columns=fallback)
    return merged


def _sex_design(sex: pd.Series) -> pd.DataFrame:
    """Return k-1 numeric indicators and retain missingness for complete cases."""
    values = sex.astype("string")
    dummies = pd.get_dummies(values, prefix="sex", drop_first=True, dtype=float)
    dummies.loc[values.isna(), :] = np.nan
    return dummies


def _fit_adjusted_ols(
    data: pd.DataFrame,
    outcome: str,
    exposure: str,
    covariates: Sequence[str],
    min_samples: int,
) -> dict[str, float] | None:
    columns = [outcome, exposure] + [c for c in covariates if c in data]
    model = data[columns].copy()
    model[outcome] = pd.to_numeric(model[outcome], errors="coerce")
    model[exposure] = pd.to_numeric(model[exposure], errors="coerce")
    design_parts = []
    for covariate in covariates:
        if covariate not in model:
            continue
        if covariate == "sex":
            design_parts.append(_sex_design(model[covariate]))
        else:
            design_parts.append(pd.DataFrame({covariate: pd.to_numeric(model[covariate], errors="coerce")}))
    cov_design = pd.concat(design_parts, axis=1) if design_parts else pd.DataFrame(index=model.index)
    complete = model[[outcome, exposure]].notna().all(axis=1)
    if not cov_design.empty:
        complete &= cov_design.notna().all(axis=1)
    model = model.loc[complete]
    cov_design = cov_design.loc[complete]
    n = len(model)
    if n < min_samples:
        return None
    y = _zscore_series(model[outcome]).to_numpy()
    x = _zscore_series(model[exposure]).to_numpy()
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        return None
    cov_arrays = []
    for col in cov_design:
        values = cov_design[col].to_numpy(dtype=float)
        if np.nanstd(values) > np.finfo(float).eps:
            cov_arrays.append(values)
    X = np.column_stack([np.ones(n), x] + cov_arrays)
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    df_resid = n - rank
    if df_resid <= 0:
        return None
    residual = y - X @ beta
    mse = float(residual @ residual / df_resid)
    covariance = mse * np.linalg.pinv(X.T @ X)
    se = float(np.sqrt(max(covariance[1, 1], 0.0)))
    if se <= 0 or not np.isfinite(se):
        return None
    t_value = float(beta[1] / se)
    p_value = float(2 * stats.t.sf(abs(t_value), df_resid))
    partial_r = float(np.sign(t_value) * np.sqrt(t_value * t_value / (t_value * t_value + df_resid)))
    return {
        "n": n,
        "standardized_beta": float(beta[1]),
        "standard_error": se,
        "t_value": t_value,
        "df_resid": int(df_resid),
        "partial_r": partial_r,
        "p_value": p_value,
    }


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """NaN-preserving Benjamini-Hochberg adjusted P values."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(p[valid])]
    ranked = p[order] * len(valid) / np.arange(1, len(valid) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def associate_one_system(
    cluster_scores: pd.DataFrame,
    system_df: pd.DataFrame,
    feature_cols: Sequence[str],
    system_name: str,
    *,
    participant_covariates: pd.DataFrame | None = None,
    id_col: str = "participant_id",
    cluster_col: str = "cluster_id",
    covariates: Sequence[str] = ("age", "sex"),
    min_samples: int = 50,
) -> pd.DataFrame:
    """Fit adjusted linear models for all cluster-by-feature pairs."""
    features = [c for c in feature_cols if c in system_df]
    participants = _aggregate_system_participants(system_df, features, id_col)
    participants = _coalesce_covariates(participants, participant_covariates, id_col)
    rows = []
    for cluster_id, score_group in cluster_scores.groupby(cluster_col, sort=True):
        score_cols = [id_col, "cluster_score"]
        for col in ("n_available_omics", "pattern_id", "pattern"):
            if col in score_group:
                score_cols.append(col)
        merged = participants.merge(score_group[score_cols], on=id_col, how="inner")
        used_covariates = [c for c in covariates if c in merged]
        missing_covariates = [c for c in covariates if c not in merged]
        if missing_covariates:
            raise ValueError(
                f"{system_name}: required covariates are absent from both the "
                f"system and pooled omics metadata: {missing_covariates}. "
                "Adjusted associations were not fitted."
            )
        for feature in features:
            fit = _fit_adjusted_ols(
                merged, feature, "cluster_score", used_covariates, min_samples
            )
            if fit is None:
                continue
            row = {
                "system": system_name,
                "feature": feature,
                cluster_col: cluster_id,
                "covariates": "+".join(used_covariates) or "none",
                "mean_available_omics": float(
                    merged.loc[
                        merged[[feature, "cluster_score"]].notna().all(axis=1),
                        "n_available_omics",
                    ].mean()
                ) if "n_available_omics" in merged else np.nan,
                **fit,
            }
            for col in ("pattern_id", "pattern"):
                if col in merged and merged[col].notna().any():
                    row[col] = merged[col].dropna().iloc[0]
            rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["fdr_within_system"] = benjamini_hochberg(result["p_value"])
    return result


def associate_all_systems(
    cluster_scores: pd.DataFrame,
    systems: Mapping[str, tuple[pd.DataFrame, Sequence[str]]],
    *,
    participant_covariates: pd.DataFrame | None = None,
    covariates: Sequence[str] = ("age", "sex"),
    min_samples: int = 50,
) -> pd.DataFrame:
    """Run all 12 systems and add one global BH-FDR column."""
    parts = [
        associate_one_system(
            cluster_scores, df, features, name,
            participant_covariates=participant_covariates,
            covariates=covariates,
            min_samples=min_samples,
        )
        for name, (df, features) in systems.items()
    ]
    parts = [part for part in parts if not part.empty]
    if not parts:
        return pd.DataFrame()
    result = pd.concat(parts, ignore_index=True)
    result["fdr_global"] = benjamini_hochberg(result["p_value"])
    return result.sort_values(["fdr_global", "p_value"], ignore_index=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omics-dir", type=Path, default=HERE.parent / "01-interim_data" / "omics_data")
    parser.add_argument("--systems-dir", type=Path, default=HERE.parent / "01-interim_data" / "systems_data")
    parser.add_argument(
        "--cluster-file", type=Path,
        default=HERE / "outputs" / "molecule_clustering" / "molecule_cluster_assignments_agglomerative.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs" / "cluster_system_association")
    parser.add_argument("--age-min", type=float, default=40.0)
    parser.add_argument("--age-max", type=float, default=70.0)
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--min-features-per-score", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    clusters = pd.read_csv(args.cluster_file)
    omics_tables = {}
    for omics, filename in OMICS_FILES.items():
        wanted = clusters.loc[clusters["omics"].eq(omics), "feature"].astype(str).unique()
        omics_tables[omics] = load_omics(args.omics_dir / filename, wanted)
    score_result = build_cluster_scores(
        omics_tables, clusters,
        min_features_per_score=args.min_features_per_score,
    )

    systems = {}
    missing_files = []
    for name, filename in SYSTEM_FILES.items():
        path = args.systems_dir / filename
        if not path.exists():
            missing_files.append(str(path))
            continue
        extra_drop = ["bmi"] if name == "body_comp" else None
        systems[name] = load_system(path, extra_drop, (args.age_min, args.age_max))
    if missing_files:
        warnings.warn(f"Unavailable system files were skipped ({len(missing_files)}): " + "; ".join(missing_files))
    if not systems:
        raise FileNotFoundError(f"No system parquet files found in {args.systems_dir}")

    associations = associate_all_systems(
        score_result.scores,
        systems,
        participant_covariates=score_result.participant_covariates,
        min_samples=args.min_samples,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_result.scores.to_csv(args.output_dir / "participant_cluster_scores.csv", index=False)
    score_result.omics_scores.to_csv(args.output_dir / "participant_omics_cluster_scores.csv", index=False)
    score_result.variance_diagnostics.to_csv(args.output_dir / "omics_score_correlations.csv", index=False)
    score_result.participant_covariates.to_csv(args.output_dir / "participant_covariates.csv", index=False)
    associations.to_csv(args.output_dir / "cluster_system_associations.csv", index=False)
    print(f"Saved {len(associations):,} fitted associations to {args.output_dir}")


if __name__ == "__main__":
    main()
