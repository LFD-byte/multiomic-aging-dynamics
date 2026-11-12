"""Panel c: dose-response associations of integrated DeltaAge with 12 systems.

For every eligible phenotype, this participant-level analysis fits either

* a linear model for a continuous outcome (standardized beta and partial r), or
* an unpenalized logistic model for a binary outcome (OR per 1-SD DeltaAge).

All models adjust for chronological age and sex. Integrated DeltaAge is
the age-adjusted residual of the four-omics stacked out-of-fold predicted age,
standardized once in the full clock cohort. Repeated phenotype measurements are
collapsed before modelling. BH-FDR is controlled across all included phenotypes
(primary), with within-system FDR also reported.

Phenotype inventory is fully automatic: every column of a system table except
known metadata (``META_COLS``) is treated as a computable outcome. Instrument
fields such as DXA scan dose are removed upstream by
``Result-01/00-filter_system_features.py``. Columns with exactly two levels are modelled as binary;
all other columns are coerced to numeric and modelled as continuous. Columns
that cannot support a model (e.g. too few numeric values, zero variance) are
recorded in the audit table at the model stage rather than silently dropped.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import optimize, special, stats


HERE = Path(__file__).resolve().parent
RESULT03 = HERE.parent / "Result-04" / "outputs"
DEFAULT_CLOCK = RESULT03 / "four_omics_meta_dataset.parquet"
DEFAULT_SYSTEM_DIRS = (
    HERE.parent / "01-interim_data" / "systems_data",
    HERE.parent / "01-data" / "systems_data",
)
FEATURE_LIST_DIR = HERE.parent / "Result-01" / "outputs" / "feature_names"
DEFAULT_OUTPUT = HERE / "outputs"
DEFAULT_FIGURES = HERE / "figures"

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
SYSTEM_LABELS = {
    "cardio": "Cardiovascular",
    "body_comp": "Body composition",
    "bone_density": "Bone density",
    "diet": "Diet",
    "sleep": "Sleep",
    "liver_health": "Liver health",
    "cgm": "Glucose regulation",
    "frailty": "Frailty",
    "life_style": "Lifestyle",
    "immune": "Immune",
    "hema": "Hematopoietic",
    "renal_function": "Renal function",
}
SYSTEM_COLORS = {
    "cardio": "#C44E52",
    "body_comp": "#DD8452",
    "bone_density": "#CCB974",
    "diet": "#8C8C3A",
    "sleep": "#8172B2",
    "liver_health": "#937860",
    "cgm": "#DA8BC3",
    "frailty": "#4C72B0",
    "life_style": "#64B5CD",
    "immune": "#55A868",
    "hema": "#2A9D8F",
    "renal_function": "#577590",
}
META_COLS = {
    "participant_id", "participantid", "cohort", "research_stage",
    "array_index", "collection_date", "collection_timestamp", "timezone",
    "batch", "age", "true_age", "sex", "gender", "bmi", "body_mass_index",
    "year_of_birth", "month_of_birth", "collection_datetime", "data_source",
    "hmo", "visit", "visit_id", "sample_id", "index",
    # Also metadata in Result-01/00-filter_system_features.py; keep the two lists aligned.
    "pred_age", "omics", "collection_time",
}
MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "null", "unknown", "missing"}
MIN_N = 50
MIN_EVENTS = 20
FDR_ALPHA = 0.05
TOP_N_PER_SYSTEM = 10


@dataclass(frozen=True)
class PhenotypeSpec:
    system: str
    feature: str
    label: str
    outcome_type: str
    direction: float = 1.0


def _normalise(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _humanise(value: str) -> str:
    text = re.sub(r"^(bt|cgm|body_comp)__?", "", value, flags=re.IGNORECASE)
    text = re.sub(r"_float_value$", "", text, flags=re.IGNORECASE)
    return re.sub(r"[_\-]+", " ", text).strip().capitalize()


def bh_fdr(values: Iterable[float]) -> np.ndarray:
    """NaN-preserving Benjamini-Hochberg adjusted P values."""
    p = np.asarray(list(values), dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return out
    order = valid[np.argsort(p[valid])]
    ranked = p[order] * len(valid) / np.arange(1, len(valid) + 1)
    out[order] = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    return out


def find_system_dir(requested: Path | str | None) -> Path:
    if requested is not None:
        requested = Path(requested).expanduser()
        if not requested.is_absolute():
            requested = (HERE / requested).resolve()
    candidates = (requested,) if requested is not None else DEFAULT_SYSTEM_DIRS
    for path in candidates:
        if path is not None and path.exists():
            return path
    checked = "; ".join(str(p) for p in candidates if p is not None)
    raise FileNotFoundError(f"No systems_data directory found; checked {checked}")


def load_integrated_delta_age(clock_path: Path) -> pd.DataFrame:
    """Return one leakage-controlled, globally standardized DeltaAge per person."""
    clock = pd.read_parquet(clock_path)
    required = {"participant_id", "true_age", "pred_age"}
    missing = required.difference(clock.columns)
    if missing:
        raise ValueError(f"Clock table lacks columns: {sorted(missing)}")
    x = clock[list(required)].copy()
    x["participant_id"] = x["participant_id"].astype(str).str.strip()
    for col in ("true_age", "pred_age"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna().groupby("participant_id", as_index=False).agg(
        age=("true_age", "mean"), pred_age=("pred_age", "mean")
    )
    if len(x) < 3:
        raise ValueError("Fewer than three participants have valid clock predictions")
    design = np.column_stack([np.ones(len(x)), x["age"].to_numpy(float)])
    raw_delta = x["pred_age"].to_numpy(float) - x["age"].to_numpy(float)
    residual = raw_delta - design @ (np.linalg.pinv(design) @ raw_delta)
    sd = residual.std(ddof=1)
    if not np.isfinite(sd) or sd <= np.finfo(float).eps:
        raise ValueError("Integrated DeltaAge has zero or invalid variance")
    x["delta_age"] = residual
    x["delta_age_sd"] = residual / sd
    return x[["participant_id", "age", "delta_age", "delta_age_sd"]]


def _coerce_binary(series: pd.Series) -> pd.Series:
    """Coerce conventional numeric/boolean binary encodings to 0/1."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    numeric = pd.to_numeric(series, errors="coerce")
    observed = set(numeric.dropna().unique())
    if observed and len(observed) <= 2:
        ordered = sorted(observed)
        if len(ordered) == 1:
            return pd.Series(np.nan, index=series.index, dtype=float)
        return numeric.map({ordered[0]: 0.0, ordered[1]: 1.0})
    text = series.astype("string").str.strip().str.lower()
    text = text.mask(text.isin(MISSING_TOKENS))
    mapping = {
        "false": 0.0, "no": 0.0, "negative": 0.0, "control": 0.0,
        "true": 1.0, "yes": 1.0, "positive": 1.0, "case": 1.0,
        "female": 0.0, "male": 1.0, "f": 0.0, "m": 1.0,
    }
    converted = text.map(mapping)
    return converted if converted.notna().sum() == text.notna().sum() else pd.Series(
        np.nan, index=series.index, dtype=float
    )


def infer_outcome_type(series: pd.Series) -> str:
    """Classify a non-metadata column: two levels -> binary, otherwise continuous.

    Every non-metadata column is treated as computable. Non-numeric values in
    a continuous column become missing and are dropped by complete-case
    analysis; whether enough data remain is decided at the model stage.
    """
    observed = series.dropna()
    if not observed.empty:
        binary = _coerce_binary(series)
        n_valid = observed.astype("string").str.lower().map(
            lambda v: v not in MISSING_TOKENS
        ).sum()
        if binary.notna().sum() == n_valid and binary.nunique(dropna=True) == 2:
            return "binary"
    return "continuous"


def _mode(values: pd.Series):
    mode = values.dropna().mode()
    return mode.iloc[0] if len(mode) else np.nan


def _covariate_columns(df: pd.DataFrame) -> dict[str, str]:
    lookup = {_normalise(c): c for c in df.columns}
    result = {}
    for target, candidates in {"sex": ("sex", "gender")}.items():
        for candidate in candidates:
            if candidate in lookup:
                result[target] = lookup[candidate]
                break
    return result


def load_external_covariates(path: Path | None) -> pd.DataFrame:
    """Load an optional participant_id/sex table (parquet or CSV)."""
    if path is None:
        return pd.DataFrame(columns=["participant_id", "sex"])
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    lookup = {_normalise(c): c for c in df.columns}
    pid = lookup.get("participant_id") or lookup.get("participantid")
    if pid is None:
        raise ValueError(f"Covariate table {path} has no participant_id")
    covs = _covariate_columns(df)
    if "sex" not in covs:
        raise ValueError(f"Covariate table must contain sex; resolved {covs}")
    out = df[[pid, covs["sex"]]].rename(
        columns={pid: "participant_id", covs["sex"]: "sex"}
    )
    out["participant_id"] = out["participant_id"].astype(str).str.strip()
    return out.groupby("participant_id", as_index=False).agg(sex=("sex", _mode))


def _inventory_for_system(
    system: str, df: pd.DataFrame,
) -> tuple[list[PhenotypeSpec], list[dict]]:
    """Every non-metadata column is a phenotype; nothing is skipped here."""
    audit: list[dict] = []
    phenotypes = []
    for feature in df.columns:
        if _normalise(feature) in META_COLS:
            continue
        feature = str(feature)
        label = _humanise(feature)
        outcome_type = infer_outcome_type(df[feature])
        detail = ""
        if outcome_type == "continuous":
            raw = df[feature]
            numeric = pd.to_numeric(raw, errors="coerce")
            text = raw.astype("string").str.strip().str.lower()
            non_numeric = int((raw.notna() & ~text.isin(MISSING_TOKENS) & numeric.isna()).sum())
            if non_numeric:
                detail = f"{non_numeric} non-numeric values set to missing"
        phenotypes.append(PhenotypeSpec(system, feature, label, outcome_type))
        audit.append({"system": system, "feature": feature, "label": label,
                      "outcome_type": outcome_type, "status": "included", "detail": detail})
    return phenotypes, audit


def _aggregate_system(df: pd.DataFrame, phenotypes: list[PhenotypeSpec]) -> pd.DataFrame:
    lookup = {_normalise(c): c for c in df.columns}
    pid = lookup.get("participant_id") or lookup.get("participantid")
    if pid is None:
        raise ValueError("system table lacks participant_id")
    covs = _covariate_columns(df)
    wanted = [p.feature for p in phenotypes]
    selected = [pid] + wanted + list(covs.values())
    x = df[list(dict.fromkeys(selected))].copy()
    x[pid] = x[pid].astype(str).str.strip()
    aggregations = {}
    for phenotype in phenotypes:
        if phenotype.outcome_type == "binary":
            x[phenotype.feature] = _coerce_binary(x[phenotype.feature])
            aggregations[phenotype.feature] = _mode
        else:
            x[phenotype.feature] = pd.to_numeric(x[phenotype.feature], errors="coerce")
            aggregations[phenotype.feature] = "median"
    if "sex" in covs:
        aggregations[covs["sex"]] = _mode
    out = x.groupby(pid, as_index=False).agg(aggregations).rename(columns={pid: "participant_id"})
    for target, source in covs.items():
        out = out.rename(columns={source: target})
    return out


def load_system_tables(
    system_dir: Path,
) -> tuple[dict[str, tuple[pd.DataFrame, list[PhenotypeSpec]]], pd.DataFrame]:
    tables = {}
    audits: list[dict] = []
    for system, filename in SYSTEM_FILES.items():
        path = system_dir / filename
        if not path.exists():
            audits.append({"system": system, "feature": "", "label": "",
                           "outcome_type": "", "status": "skipped",
                           "detail": f"system file absent: {path}"})
            continue
        df = pd.read_parquet(path)
        phenotypes, audit = _inventory_for_system(system, df)
        audits.extend(audit)
        if phenotypes:
            tables[system] = (_aggregate_system(df, phenotypes), phenotypes)
    return tables, pd.DataFrame(audits)


def compare_with_feature_lists(
    inventory_audit: pd.DataFrame, feature_dir: Path | str = FEATURE_LIST_DIR,
) -> pd.DataFrame:
    """Compare included phenotypes with Result-01 curated lists (system_<key>.csv).

    Returns one row per system with expected/included counts and the feature
    names that are extra (in the table but not curated) or missing. Emits a
    warning when any system disagrees. Systems without a list are reported as such.
    """
    import warnings
    feature_dir = Path(feature_dir)
    included = inventory_audit.loc[inventory_audit["status"].eq("included")]
    rows = []
    for system in SYSTEM_FILES:
        observed = set(included.loc[included["system"].eq(system), "feature"].astype(str))
        list_path = feature_dir / f"system_{system}.csv"
        if not list_path.exists():
            rows.append({"system": system, "n_expected": np.nan, "n_included": len(observed),
                         "n_extra": np.nan, "n_missing": np.nan, "extra": "", "missing": "",
                         "note": f"no list: {list_path}"})
            continue
        expected = set(pd.read_csv(list_path, encoding="utf-8-sig")["feature"].astype(str))
        extra, missing = sorted(observed - expected), sorted(expected - observed)
        rows.append({"system": system, "n_expected": len(expected), "n_included": len(observed),
                     "n_extra": len(extra), "n_missing": len(missing),
                     "extra": ";".join(extra), "missing": ";".join(missing),
                     "note": "ok" if not extra and not missing else "MISMATCH"})
    table = pd.DataFrame(rows)
    bad = table.loc[table["note"].eq("MISMATCH"), "system"].tolist()
    if bad:
        warnings.warn("Included phenotypes differ from Result-01 feature lists for: "
                      + ", ".join(bad) + "; see the 'extra'/'missing' columns.", stacklevel=2)
    return table


def _sex_design(values: pd.Series) -> pd.DataFrame:
    text = values.astype("string")
    dummies = pd.get_dummies(text, prefix="sex", drop_first=True, dtype=float)
    dummies.loc[text.isna(), :] = np.nan
    return dummies


def _model_frame(data: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, np.ndarray]:
    required = [outcome, "delta_age_sd", "age", "sex"]
    d = data[required].copy()
    for col in (outcome, "delta_age_sd", "age"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    sex = _sex_design(d["sex"])
    complete = d[[outcome, "delta_age_sd", "age"]].notna().all(axis=1)
    if not sex.empty:
        complete &= sex.notna().all(axis=1)
    d, sex = d.loc[complete], sex.loc[complete]
    numeric_cov = []
    for col in ("age",):
        value = d[col].to_numpy(float)
        sd = value.std(ddof=0)
        if np.isfinite(sd) and sd > np.finfo(float).eps:
            numeric_cov.append((value - value.mean()) / sd)
    blocks = [np.ones(len(d)), d["delta_age_sd"].to_numpy(float), *numeric_cov]
    blocks.extend(sex[col].to_numpy(float) for col in sex if sex[col].nunique() > 1)
    return d, np.column_stack(blocks)


def fit_continuous(data: pd.DataFrame, outcome: str, min_n: int) -> dict | None:
    d, design = _model_frame(data, outcome)
    if len(d) < min_n:
        return None
    y_raw = d[outcome].to_numpy(float)
    y_sd = y_raw.std(ddof=1)
    if not np.isfinite(y_sd) or y_sd <= np.finfo(float).eps:
        return None
    y = (y_raw - y_raw.mean()) / y_sd
    beta = np.linalg.pinv(design) @ y
    residual = y - design @ beta
    rank = np.linalg.matrix_rank(design)
    dof = len(d) - rank
    if dof <= 1:
        return None
    covariance = (residual @ residual / dof) * np.linalg.pinv(design.T @ design)
    se = float(np.sqrt(max(covariance[1, 1], 0.0)))
    if not np.isfinite(se) or se <= 0:
        return None
    t_value = float(beta[1] / se)
    p_value = float(2 * stats.t.sf(abs(t_value), dof))
    partial_r = float(np.sign(t_value) * np.sqrt(t_value**2 / (t_value**2 + dof)))
    return {"n": len(d), "n_cases": np.nan, "n_controls": np.nan,
            "effect": float(beta[1]), "ci_low": float(beta[1] - 1.96 * se),
            "ci_high": float(beta[1] + 1.96 * se), "standard_error": se,
            "standardized_beta": float(beta[1]), "partial_r": partial_r,
            "odds_ratio": np.nan, "or_ci_low": np.nan, "or_ci_high": np.nan,
            "statistic": t_value, "p_value": p_value, "df_resid": dof}


def _logistic_nll(beta: np.ndarray, design: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    eta = design @ beta
    value = float(np.logaddexp(0.0, eta).sum() - y @ eta)
    gradient = design.T @ (special.expit(eta) - y)
    return value, gradient


def fit_binary(data: pd.DataFrame, outcome: str, min_n: int, min_events: int) -> dict | None:
    d, design = _model_frame(data, outcome)
    y = d[outcome].to_numpy(float)
    if len(d) < min_n or not set(np.unique(y)).issubset({0.0, 1.0}):
        return None
    cases, controls = int(y.sum()), int((y == 0).sum())
    if cases < min_events or controls < min_events or np.linalg.matrix_rank(design) < design.shape[1]:
        return None
    fit = optimize.minimize(
        lambda b: _logistic_nll(b, design, y), np.zeros(design.shape[1]),
        jac=True, method="BFGS", options={"gtol": 1e-8, "maxiter": 1000},
    )
    # BFGS may report precision loss after reaching a valid optimum; assess the
    # score and finite estimates directly rather than trusting only success.
    beta = fit.x
    _, gradient = _logistic_nll(beta, design, y)
    probability = special.expit(design @ beta)
    hessian = design.T @ (design * (probability * (1 - probability))[:, None])
    if (not np.isfinite(beta).all() or np.max(np.abs(gradient)) > 1e-4 or
            np.linalg.cond(hessian) > 1e12 or abs(beta[1]) > 10):
        return None
    covariance = np.linalg.inv(hessian)
    se = float(np.sqrt(covariance[1, 1]))
    z_value = float(beta[1] / se)
    p_value = float(2 * stats.norm.sf(abs(z_value)))
    lo, hi = beta[1] + np.array([-1.0, 1.0]) * 1.96 * se
    return {"n": len(d), "n_cases": cases, "n_controls": controls,
            "effect": float(np.exp(beta[1])), "ci_low": float(np.exp(lo)),
            "ci_high": float(np.exp(hi)), "standard_error": se,
            "standardized_beta": np.nan, "partial_r": np.nan,
            "odds_ratio": float(np.exp(beta[1])), "or_ci_low": float(np.exp(lo)),
            "or_ci_high": float(np.exp(hi)), "statistic": z_value,
            "p_value": p_value, "df_resid": np.nan}


def analyse(
    clocks: pd.DataFrame,
    system_tables: dict[str, tuple[pd.DataFrame, list[PhenotypeSpec]]],
    external_covariates: pd.DataFrame | None = None,
    min_n: int = MIN_N,
    min_events: int = MIN_EVENTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit all eligible phenotypes and return results plus model-level audit."""
    rows, audit = [], []
    external = external_covariates if external_covariates is not None else pd.DataFrame()
    for system in SYSTEM_FILES:
        if system not in system_tables:
            continue
        phenodata, phenotypes = system_tables[system]
        merged = clocks.merge(phenodata, on="participant_id", how="inner", suffixes=("", "_system"))
        if not external.empty:
            merged = merged.merge(external, on="participant_id", how="left", suffixes=("", "_external"))
        for covariate in ("sex",):
            system_col, external_col = f"{covariate}_system", f"{covariate}_external"
            candidates = [c for c in (covariate, system_col, external_col) if c in merged]
            if candidates:
                value = merged[candidates[0]]
                for candidate in candidates[1:]:
                    value = value.where(value.notna(), merged[candidate])
                merged[covariate] = value
        missing_covariates = [c for c in ("age", "sex") if c not in merged]
        for phenotype in phenotypes:
            detail, fit = "", None
            if missing_covariates:
                detail = f"missing covariates: {','.join(missing_covariates)}"
            else:
                fit = (fit_binary(merged, phenotype.feature, min_n, min_events)
                       if phenotype.outcome_type == "binary"
                       else fit_continuous(merged, phenotype.feature, min_n))
                if fit is None:
                    detail = "insufficient complete cases/events, zero variance, or unstable fit"
            status = "included" if fit is not None else "skipped"
            audit.append({"system": system, "feature": phenotype.feature,
                          "label": phenotype.label, "outcome_type": phenotype.outcome_type,
                          "status": status, "detail": detail,
                          "n_overlap_before_complete_case": len(merged)})
            if fit is None:
                continue
            # direction=-1 makes larger effects consistently mean higher risk.
            if phenotype.direction == -1:
                if phenotype.outcome_type == "continuous":
                    fit["effect"], fit["standardized_beta"], fit["partial_r"] = (
                        -fit["effect"], -fit["standardized_beta"], -fit["partial_r"])
                    fit["ci_low"], fit["ci_high"] = -fit["ci_high"], -fit["ci_low"]
                    fit["statistic"] = -fit["statistic"]
                else:
                    fit["effect"] = fit["odds_ratio"] = 1 / fit["odds_ratio"]
                    fit["ci_low"], fit["ci_high"] = 1 / fit["or_ci_high"], 1 / fit["or_ci_low"]
                    fit["or_ci_low"], fit["or_ci_high"] = fit["ci_low"], fit["ci_high"]
                    fit["statistic"] = -fit["statistic"]
            rows.append({"system": system, "system_label": SYSTEM_LABELS[system],
                         "feature": phenotype.feature, "label": phenotype.label,
                         "outcome_type": phenotype.outcome_type,
                         "direction": phenotype.direction,
                         "covariates": "age+sex", **fit})
    results = pd.DataFrame(rows)
    model_audit = pd.DataFrame(audit)
    if results.empty:
        return results, model_audit
    results["q_value_global"] = bh_fdr(results["p_value"])
    results["q_value_within_system"] = results.groupby("system", group_keys=False)[
        "p_value"
    ].transform(bh_fdr)
    results["fdr_significant"] = results["q_value_global"] < FDR_ALPHA
    order = {system: i for i, system in enumerate(SYSTEM_FILES)}
    results["_system_order"] = results["system"].map(order)
    results = results.sort_values(
        ["_system_order", "outcome_type", "q_value_global", "feature"]
    ).drop(columns="_system_order").reset_index(drop=True)
    return results, model_audit


def select_top_phenotypes(results: pd.DataFrame, n_per_system: int = TOP_N_PER_SYSTEM) -> pd.DataFrame:
    """Select the smallest-P phenotypes per system for display only.

    Inferential columns, including global FDR, must be computed on the complete
    result table before this function is called. Selection never changes the
    multiplicity family.
    """
    if n_per_system < 1:
        raise ValueError("n_per_system must be at least 1")
    if results.empty:
        return results.copy()
    required = {"system", "p_value"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Results lack ranking columns: {sorted(missing)}")
    order = {system: i for i, system in enumerate(SYSTEM_FILES)}
    selected = (
        results.sort_values(["system", "p_value", "feature"], na_position="last")
        .groupby("system", sort=False, group_keys=False)
        .head(n_per_system)
        .copy()
    )
    selected["display_rank_within_system"] = selected.groupby("system").cumcount() + 1
    selected["_system_order"] = selected["system"].map(order)
    return selected.sort_values(
        ["_system_order", "p_value", "feature"]
    ).drop(columns="_system_order").reset_index(drop=True)


def _marker_sizes(n: pd.Series, low: float = 26, high: float = 100) -> np.ndarray:
    values = np.sqrt(pd.to_numeric(n, errors="coerce").to_numpy(float))
    if len(values) == 0 or np.nanmax(values) <= np.nanmin(values):
        return np.full(len(values), (low + high) / 2)
    return low + (values - np.nanmin(values)) / (np.nanmax(values) - np.nanmin(values)) * (high - low)


def _effect_limits(values: pd.Series, lows: pd.Series, highs: pd.Series, null: float) -> tuple[float, float]:
    all_values = pd.concat([values, lows, highs]).replace([np.inf, -np.inf], np.nan).dropna()
    if all_values.empty:
        return (null - 1, null + 1)
    lo, hi = float(all_values.min()), float(all_values.max())
    span = max(hi - lo, 0.15 if null == 0 else 0.2)
    return min(lo, null) - 0.08 * span, max(hi, null) + 0.08 * span


def _draw_forest_page(
    plot: pd.DataFrame,
    *,
    beta_limits: tuple[float, float],
    or_limits: tuple[float, float],
    page_number: int,
    total_pages: int,
    selection_note: str | None,
) -> plt.Figure:
    """Draw one bounded-height page of the complete forest."""
    n_rows = len(plot)
    y = np.arange(n_rows)[::-1]
    height = max(8.0, 0.31 * n_rows + 2.6)
    fig = plt.figure(figsize=(15.8, height))
    grid = fig.add_gridspec(1, 4, width_ratios=[5.7, 3.0, 3.0, 1.5], wspace=0.06)
    label_ax = fig.add_subplot(grid[0, 0])
    beta_ax = fig.add_subplot(grid[0, 1], sharey=label_ax)
    or_ax = fig.add_subplot(grid[0, 2], sharey=label_ax)
    info_ax = fig.add_subplot(grid[0, 3], sharey=label_ax)

    for ax in (label_ax, beta_ax, or_ax, info_ax):
        ax.set_ylim(-0.8, n_rows - 0.2)
    label_ax.set_xlim(0, 1)
    info_ax.set_xlim(0, 1)
    label_ax.axis("off")
    info_ax.axis("off")

    previous = None
    boundaries = []
    for i, row in plot.iterrows():
        yi = y[i]
        if previous != row["system"]:
            if previous is not None:
                boundaries.append(yi + 0.55)
            label_ax.text(0.0, yi, row["system_label"], va="center", ha="left",
                          fontsize=8.6, weight="bold", color=SYSTEM_COLORS[row["system"]])
            label_x = 0.30
            previous = row["system"]
        else:
            label_x = 0.30
        label_ax.text(label_x, yi, row["label"], va="center", ha="left", fontsize=7.2,
                      color="#252525")
        info_ax.text(0.02, yi, f"{int(row['n']):,}", va="center", ha="left", fontsize=7.0)
        info_ax.text(0.48, yi, f"{row['q_value_global']:.2g}", va="center", ha="left",
                     fontsize=7.0, weight="bold" if row["fdr_significant"] else "normal")
        target = beta_ax if row["outcome_type"] == "continuous" else or_ax
        color = SYSTEM_COLORS[row["system"]]
        target.plot([row["ci_low"], row["ci_high"]], [yi, yi], color=color, lw=1.15, zorder=2)
        target.scatter([row["effect"]], [yi], s=row["_marker_size"],
                       facecolor=color if row["fdr_significant"] else "white",
                       edgecolor=color, linewidth=1.25, zorder=3)

    for boundary in boundaries:
        for ax in (label_ax, beta_ax, or_ax, info_ax):
            ax.axhline(boundary, color="#D9D9D9", lw=0.7, zorder=0)

    binary = plot.loc[plot["outcome_type"].eq("binary")]
    beta_ax.axvline(0, color="#555555", ls="--", lw=0.85)
    or_ax.axvline(1, color="#555555", ls="--", lw=0.85)
    beta_ax.set_xlim(beta_limits)
    if len(binary):
        or_ax.set_xscale("log")
    or_ax.set_xlim(max(or_limits[0], 0.01), or_limits[1])
    beta_ax.set_xlabel("Standardized β per 1-SD ΔAge (95% CI)", fontsize=8.5)
    or_ax.set_xlabel("Odds ratio per 1-SD ΔAge (95% CI)", fontsize=8.5)
    for ax in (beta_ax, or_ax):
        ax.grid(axis="x", color="#ECECEC", lw=0.65)
        ax.tick_params(axis="x", labelsize=7.2)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.spines[["top", "right", "left"]].set_visible(False)

    label_ax.text(0, n_rows - 0.05, "System / phenotype", fontsize=8, weight="bold", va="bottom")
    info_ax.text(0.02, n_rows - 0.05, "N", fontsize=8, weight="bold", va="bottom")
    info_ax.text(0.48, n_rows - 0.05, "FDR", fontsize=8, weight="bold", va="bottom")
    legend = [
        Line2D([0], [0], marker="o", color="#555", markerfacecolor="#555", lw=0,
               label="Global FDR < 0.05"),
        Line2D([0], [0], marker="o", color="#555", markerfacecolor="white", lw=0,
               label="Global FDR ≥ 0.05"),
    ]
    fig.legend(handles=legend, frameon=False, loc="lower center", ncol=2,
               bbox_to_anchor=(0.57, 0.012), fontsize=8)
    page_suffix = f" | page {page_number}/{total_pages}" if total_pages > 1 else ""
    fig.suptitle("Panel c | Dose–response associations of integrated ΔAge across 12 systems" +
                 page_suffix,
                 fontsize=13, weight="bold", y=0.995)
    note = (
        "Adjusted for chronological age and sex. " +
        ((selection_note.rstrip(". ") + ". ") if selection_note else "") +
        "Point area encodes complete-case N; colour encodes system. "
        "Filled points pass BH-FDR across all fitted phenotypes."
    )
    fig.text(0.57, 0.035, note,
             ha="center", fontsize=7.8, color="#444444")
    fig.subplots_adjust(top=0.975, bottom=max(0.07, 0.55 / height), left=0.02, right=0.99)
    return fig


def plot_forest(
    results: pd.DataFrame, output_base: Path, max_rows_per_page: int = 70,
    selection_note: str | None = None,
) -> list[Path]:
    """Save a paginated system-grouped forest and return its PNG page paths.

    A single 871-row raster exceeds Matplotlib's 2^16-pixel dimension limit and
    is not readable. All pages therefore share identical effect axes and point-
    size scaling, while the PDF stores the complete forest as one multipage file.
    """
    if results.empty:
        raise ValueError("No fitted phenotype is available for plotting")
    if max_rows_per_page < 10:
        raise ValueError("max_rows_per_page must be at least 10")
    plot = results.copy().reset_index(drop=True)
    plot["_marker_size"] = _marker_sizes(plot["n"])
    continuous = plot.loc[plot["outcome_type"].eq("continuous")]
    binary = plot.loc[plot["outcome_type"].eq("binary")]
    beta_limits = _effect_limits(continuous["effect"], continuous["ci_low"],
                                 continuous["ci_high"], 0)
    or_limits = _effect_limits(binary["effect"], binary["ci_low"], binary["ci_high"], 1)
    total_pages = int(np.ceil(len(plot) / max_rows_per_page))
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_paths: list[Path] = []
    with PdfPages(output_base.with_suffix(".pdf")) as pdf:
        for page_index, start in enumerate(range(0, len(plot), max_rows_per_page), start=1):
            page = plot.iloc[start:start + max_rows_per_page].reset_index(drop=True)
            fig = _draw_forest_page(
                page, beta_limits=beta_limits, or_limits=or_limits,
                page_number=page_index, total_pages=total_pages,
                selection_note=selection_note,
            )
            png_path = (output_base.with_suffix(".png") if total_pages == 1 else
                        output_base.parent / f"{output_base.name}_page_{page_index:02d}.png")
            fig.savefig(png_path, dpi=350, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            png_paths.append(png_path)
            plt.close(fig)
    return png_paths


def write_caption(
    results: pd.DataFrame, path: Path, *, total_tested: int | None = None,
) -> None:
    significant = int(results["fdr_significant"].sum())
    systems = int(results["system"].nunique())
    total = len(results) if total_tested is None else int(total_tested)
    selection = (
        f"The forest displays up to {TOP_N_PER_SYSTEM} phenotypes with the smallest two-sided "
        f"P values per system ({len(results)} displayed); all {total} fitted phenotypes were "
        "retained in the global multiplicity correction. "
        if total > len(results) else ""
    )
    text = (
        "Panel c | Dose-response associations of integrated biological age acceleration with "
        f"{len(results)} phenotypes spanning {systems} physiological systems. Integrated DeltaAge "
        "was defined as the age-adjusted residual of the four-omics stacked out-of-fold age "
        "prediction and standardized in the full clock cohort. Continuous outcomes were analysed "
        "using linear regression and are shown as standardized beta coefficients; binary outcomes "
        "were analysed using logistic regression and are shown as odds ratios. Every model adjusted "
        "for chronological age and sex. Horizontal lines denote Wald 95% confidence intervals; "
        "point area represents complete-case sample size and colour denotes physiological system. "
        + selection +
        f"Filled points passed global Benjamini-Hochberg FDR < 0.05 ({significant} displayed)."
    )
    path.write_text(text, encoding="utf-8")


def run(
    clock_path: Path = DEFAULT_CLOCK,
    system_dir: Path | None = None,
    covariate_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT,
    figure_dir: Path = DEFAULT_FIGURES,
    min_n: int = MIN_N,
    min_events: int = MIN_EVENTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    clocks = load_integrated_delta_age(clock_path)
    systems, inventory_audit = load_system_tables(find_system_dir(system_dir))
    feature_check = compare_with_feature_lists(inventory_audit)
    feature_check.to_csv(output_dir / "panel_c_feature_list_check.csv", index=False)
    covariates = load_external_covariates(covariate_path)
    results, model_audit = analyse(clocks, systems, covariates, min_n, min_events)
    audit = pd.concat([inventory_audit.assign(stage="inventory"),
                       model_audit.assign(stage="model")], ignore_index=True, sort=False)
    audit.to_csv(output_dir / "panel_c_dose_response_audit.csv", index=False)
    clocks.to_parquet(output_dir / "panel_c_integrated_delta_age.parquet", index=False)
    if results.empty:
        raise RuntimeError("No phenotype model was fitted; inspect panel_c_dose_response_audit.csv")
    results.to_csv(output_dir / "panel_c_dose_response_effects.csv", index=False)
    displayed = select_top_phenotypes(results, TOP_N_PER_SYSTEM)
    displayed.to_csv(output_dir / "panel_c_dose_response_displayed_top10.csv", index=False)
    plot_forest(displayed, figure_dir / "panel_c_dose_response_forest",
                max_rows_per_page=max(10, len(displayed)),
                selection_note="Up to 10 smallest-P phenotypes per system are displayed")
    write_caption(displayed, output_dir / "panel_c_dose_response_caption.txt",
                  total_tested=len(results))
    diagnostics = {
        "clock_path": str(clock_path), "system_dir": str(find_system_dir(system_dir)),
        "phenotype_inventory": "all non-metadata columns",
        "covariate_path": str(covariate_path) if covariate_path else "system tables",
        "n_clock_participants": len(clocks), "n_models": len(results),
        "n_systems": int(results["system"].nunique()),
        "n_continuous": int(results["outcome_type"].eq("continuous").sum()),
        "n_binary": int(results["outcome_type"].eq("binary").sum()),
        "n_global_fdr": int(results["fdr_significant"].sum()),
        "display_rule": "10 smallest two-sided P values per system; all if fewer than 10",
        "n_displayed": len(displayed),
        "fdr_family": "all fitted phenotypes", "covariates": ["age", "sex"],
    }
    (output_dir / "panel_c_dose_response_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    print(f"Panel c: {len(results)} phenotypes across {results.system.nunique()} systems; "
          f"{results.fdr_significant.sum()} pass global FDR < {FDR_ALPHA}")
    return results, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock", type=Path, default=DEFAULT_CLOCK)
    parser.add_argument("--systems-dir", type=Path)
    parser.add_argument("--covariates", type=Path,
                        help="Optional parquet/CSV with participant_id and sex")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--min-n", type=int, default=MIN_N)
    parser.add_argument("--min-events", type=int, default=MIN_EVENTS)
    args = parser.parse_args()
    run(args.clock, args.systems_dir, args.covariates,
        args.output_dir, args.figure_dir, args.min_n, args.min_events)


if __name__ == "__main__":
    main()
