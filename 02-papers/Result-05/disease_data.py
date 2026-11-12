"""Data loaders for the baseline-diagnosis odds-ratio analysis (disease_odds_ratio.py).

Clock scores are read from the Result-04 outputs written by 02-aging_clock.py;
diagnoses and sex are read from the HPP datasets (set HPP_DATA_ROOT if needed).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

HERE = Path(__file__).resolve().parent
RESULT04 = HERE.parent / "Result-04" / "outputs"
HPP_DATA_ROOT = Path(os.environ.get("HPP_DATA_ROOT", "/home/ec2-user/studies/hpp_datasets"))
DEFAULT_POPULATION = HPP_DATA_ROOT / "population" / "population.parquet"
DEFAULT_MEDICAL_CONDITIONS = HPP_DATA_ROOT / "medical_conditions" / "medical_conditions.parquet"
META_PATH = RESULT04 / "four_omics_meta_dataset.parquet"
METABOLOMICS_PATH = RESULT04 / "metabolomics_parallel_predictions.parquet"

AGE_LABEL = "Chronological age"
STACK_LABEL = "4-omics stack"
MET_LABEL = "Metabolomics†"
PAIRED_MODELS = (
    AGE_LABEL, "Transcriptomics", "Lipidomics",
    "Gut Microbiome", "Oral Microbiome", STACK_LABEL,
)
BASELINE_SOURCES = ("initial medical survey", "in system drop down list")


def _residualised_acceleration(age: pd.Series, prediction: pd.Series) -> np.ndarray:
    delta = prediction.to_numpy(float) - age.to_numpy(float)
    model = LinearRegression().fit(age.to_numpy(float).reshape(-1, 1), delta)
    return delta - model.predict(age.to_numpy(float).reshape(-1, 1))


def load_sex(path: Path) -> pd.DataFrame:
    pop = pd.read_parquet(path).reset_index()
    if not {"participant_id", "sex"}.issubset(pop.columns):
        raise ValueError("population table must contain participant_id and sex")
    pop["participant_id"] = pop["participant_id"].astype(str)
    pop["sex"] = pd.to_numeric(pop["sex"].astype(str), errors="coerce")
    pop = pop.dropna(subset=["sex"]).drop_duplicates("participant_id")
    unique = set(pop["sex"].unique())
    if not unique.issubset({0, 1}):
        raise ValueError(f"sex must be coded 0/1; observed {sorted(unique)}")
    return pop[["participant_id", "sex"]]


class ConditionSpec:
    outcome: str
    system: str
    icd11_codes: tuple[str, ...]
    ascertainment: str


# Selected before looking at AUC: common baseline diagnoses that are generally
# established by clinical measurement, laboratory testing, imaging, or standard
# diagnostic work-up. The parquet contains diagnosis records, not the raw values;
# therefore these are not re-derived threshold phenotypes.
CONDITIONS = (
    ConditionSpec("Hypercholesterolaemia", "Metabolic", ("5C80", "5C80.0"),
                  "Recorded ICD-11 diagnosis; ordinarily lipid-test based"),
    ConditionSpec("Hypertension", "Cardiovascular", ("BA00",),
                  "Recorded ICD-11 diagnosis; ordinarily blood-pressure based"),
    ConditionSpec("Hyperglycaemia", "Glycaemic", ("5A40",),
                  "Recorded ICD-11 diagnosis; ordinarily glucose-test based"),
    ConditionSpec("Obesity", "Body composition", ("5B81",),
                  "Recorded ICD-11 diagnosis; ordinarily anthropometry based"),
    ConditionSpec("Fatty liver", "Liver", ("DB92",),
                  "Recorded ICD-11 diagnosis; ordinarily imaging/clinical based"),
    ConditionSpec("Hypothyroidism", "Endocrine", ("5A00",),
                  "Recorded ICD-11 diagnosis; ordinarily thyroid-test based"),
    ConditionSpec("Cholelithiasis", "Hepatobiliary", ("DC11",),
                  "Recorded ICD-11 diagnosis; ordinarily imaging based"),
    ConditionSpec("Asthma", "Respiratory", ("CA23",),
                  "Recorded ICD-11 diagnosis; clinical/functional work-up"),
)


def load_baseline_diagnoses(path: Path):
    """Create one binary endpoint per diagnosis among baseline-survey respondents.

    Controls are not the entire clock cohort: they are participants with at least
    one baseline medical-survey record and without the relevant code. Follow-up
    survey rows are excluded to preserve the cross-sectional baseline estimand.
    """
    raw = pd.read_parquet(path).reset_index()
    required = {"participant_id", "icd11_code", "data_source"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"medical conditions table missing columns: {sorted(missing)}")
    raw["participant_id"] = raw["participant_id"].astype(str)
    raw["icd11_code"] = raw["icd11_code"].astype("string").str.strip()
    baseline = raw[raw["data_source"].isin(BASELINE_SOURCES)].copy()
    surveyed = pd.Index(baseline["participant_id"].dropna().unique(), name="participant_id")
    if surveyed.empty:
        raise ValueError(f"no records found for baseline sources {BASELINE_SOURCES}")

    endpoints, rows = {}, []
    for spec in CONDITIONS:
        cases = set(baseline.loc[baseline["icd11_code"].isin(spec.icd11_codes), "participant_id"])
        endpoint = pd.DataFrame({"participant_id": surveyed.astype(str)})
        endpoint["outcome"] = endpoint["participant_id"].isin(cases).astype(int)
        endpoints[spec.outcome] = endpoint
        rows.append({
            "outcome": spec.outcome,
            "system": spec.system,
            "definition": " or ".join(spec.icd11_codes),
            "ascertainment": spec.ascertainment,
            "source": str(path),
            "baseline_sources": "; ".join(BASELINE_SOURCES),
            "status": "included",
            "n_surveyed": len(endpoint),
            "n_cases_all_surveyed": int(endpoint["outcome"].sum()),
        })
    return endpoints, pd.DataFrame(rows), baseline


def load_primary_scores() -> pd.DataFrame:
    """Six scores on the complete Result-04 four-omics stacking cohort."""
    meta = pd.read_parquet(META_PATH)
    required = {"participant_id", "true_age", "pred_age"}
    for omics in PAIRED_MODELS[1:-1]:
        required.update((f"age__{omics}", f"pred__{omics}"))
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"four-omics meta table missing: {sorted(missing)}")
    aggregate = {c: "mean" for c in required if c != "participant_id"}
    p = meta[list(required)].groupby("participant_id", as_index=False).agg(aggregate)
    out = pd.DataFrame({
        "participant_id": p["participant_id"].astype(str),
        AGE_LABEL: p["true_age"],
    })
    for omics in PAIRED_MODELS[1:-1]:
        out[omics] = _residualised_acceleration(p[f"age__{omics}"], p[f"pred__{omics}"])
    out[STACK_LABEL] = _residualised_acceleration(p["true_age"], p["pred_age"])
    if out[list(PAIRED_MODELS)].isna().any().any():
        raise ValueError("missing score in four-omics common cohort")
    return out


def load_metabolomics_scores() -> pd.DataFrame:
    """Metabolomics clock on its own maximum available Result-04 cohort."""
    met = pd.read_parquet(METABOLOMICS_PATH)
    required = {"participant_id", "true_age", "pred_age"}
    missing = required.difference(met.columns)
    if missing:
        raise ValueError(f"metabolomics prediction table missing: {sorted(missing)}")
    p = (met.dropna(subset=list(required)).groupby("participant_id", as_index=False)
         .agg(true_age=("true_age", "mean"), pred_age=("pred_age", "mean")))
    p["participant_id"] = p["participant_id"].astype(str)
    p[MET_LABEL] = _residualised_acceleration(p["true_age"], p["pred_age"])
    p[AGE_LABEL] = p["true_age"]
    return p[["participant_id", AGE_LABEL, MET_LABEL]]
