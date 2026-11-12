#!/usr/bin/env python3
"""Individual-feature associations and two-crest proportions across 12 systems.

Run with --help. Scientific computation and graphics use Python exclusively.
Figures are exported directly (PDF, 300 dpi PNG, SVG); no external figure-QA helper scripts are used.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, stats

HERE = Path(__file__).resolve().parent
OMICS = ["transcriptomics", "metabolomics", "lipidomics", "gut_microbiome", "oral_microbiome"]
SYSTEM_FILES = {
    "cardio": "020-cardiovascular_system_with_metadata.parquet",
    "body_comp": "body_composition_with_metadata.parquet",
    "bone_density": "bone_density_with_metadata.parquet",
    "diet": "diet_with_metadata.parquet", "sleep": "sleep_with_metadata.parquet",
    "liver_health": "liver_health_with_metadata.parquet", "cgm": "cgm_with_metadata.parquet",
    "frailty": "frailty_with_metadata.parquet", "life_style": "life_style_with_metadata.parquet",
    "immune": "immune_system_with_metadata.parquet", "hema": "hematology_system_with_metadata.parquet",
    "renal_function": "renal_function_with_metadata.parquet",
}
SYSTEM_LABELS = dict(zip(SYSTEM_FILES, [
    "Cardiovascular", "Body composition", "Bone density", "Diet", "Sleep", "Liver health",
    "Insulin resistance", "Frailty", "Lifestyle", "Immune system", "Hematopoietic system", "Renal function",
]))
FIGURE_GROUPS = {
    "group1": ("cardio", "body_comp", "bone_density", "cgm", "immune", "hema"),
    "group2": ("diet", "sleep", "liver_health", "frailty", "life_style", "renal_function"),
}
# Groups whose panels are reordered by association strength (strongest first) and drawn without a-f letters.
STRENGTH_ORDERED_GROUPS = {"group1"}
UNLABELED_GROUPS = {"group1"}
META_COLS = {
    "participant_id", "cohort", "research_stage", "array_index", "collection_date",
    "collection_timestamp", "timezone", "age", "year_of_birth", "month_of_birth", "sex",
    "collection_datetime", "batch", "bmi", "pred_age", "omics", "collection_time",
}
CREST_RULES = [("Crest1", om, age) for om, age in zip(OMICS, [52, 48, 48, 50, 48])] + [
    ("Crest2", "transcriptomics", 64)]
CREST_COLORS = {"Crest1": "#488CC7", "Crest2": "#F39A4B"}
TEST_KEYS = ["omics", "feature_id", "system", "indicator_id"]
SCOPE_COLUMNS = {
    "global": [], "within_omics": ["omics"], "within_system": ["system"],
    "within_omics_system": ["omics", "system"], "within_indicator": ["system", "indicator_id"],
    "within_omics_indicator": ["omics", "system", "indicator_id"],
}
MISSING_TEXT = ["", "na", "nan", "unknown", "missing", "null"]


def parse_config(argv=None, here=HERE):
    here = Path(here).resolve()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag, default in {
        "systems-dir": here / "../01-interim_data/systems_data",
        "omics-dir": here / "../01-interim_data/omics_data", "peak-file": None,
        "deswan-dir": here / "outputs/deswan-window_7.5",
        "output-dir": here / "outputs/05-crest_system_association",
        "figure-dir": here / "figures/crest_system_association",
        "association-file": None, "indicator-file": None, "column-map": None,
    }.items():
        help_text = {"peak-file": "Optional explicit peak list; omit to generate from Result-02 DE-SWAN results",
                     "deswan-dir": "Upstream DE-SWAN result directory (default: outputs/deswan-window_7.5 beside this script)",
                     "output-dir": "Tables, provenance and QA logs (default: outputs/05-crest_system_association)",
                     "figure-dir": "PDF, 300 dpi PNG and SVG (default: figures/crest_system_association beside this script)"}
        p.add_argument("--" + flag, type=Path, default=default, help=help_text.get(flag))
    p.add_argument("--peak-mode", choices=["significant_list", "q", "flag"], default="significant_list")
    p.add_argument("--peak-threshold", type=float, default=0.05)
    p.add_argument("--peak-adjust-method", choices=["BH", "fdr", "BY", "bonferroni", "holm", "none"], default="BH",
                   help="Peak p-value correction within each omics/window; defaults to upstream settings, otherwise BH")
    p.add_argument("--q-threshold", type=float, default=0.05)
    p.add_argument("--p-adjust-method", choices=["BH", "fdr", "BY", "bonferroni", "holm", "none"], default="BH")
    p.add_argument("--p-adjust-scope", choices=list(SCOPE_COLUMNS), default="global")
    p.add_argument("--min-samples", type=int, default=50)
    p.add_argument("--age-min", type=float, default=-math.inf)
    p.add_argument("--age-max", type=float, default=math.inf)
    p.add_argument("--clr-pseudocount", type=float, default=1.0)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--width-mm", type=float, choices=[210.0], default=210.0,
                   help="Fixed figure width: 210 mm")
    p.add_argument("--height-mm", type=float, default=130,
                   help="Minimum figure height in mm (default: 130); expanded to fit x-axis labels")
    p.add_argument("--font", default="Arial")
    p.add_argument("--prepare-only", action="store_true", help="Extract significant peak features from DE-SWAN results, write members/audits, then stop")
    args = list(sys.argv[1:] if argv is None else argv)
    cfg = p.parse_args(args)
    cfg.explicit = {arg.split("=")[0][2:].replace("-", "_") for arg in args if arg.startswith("--")}
    cfg.here = here
    if cfg.peak_file is None and "peak_mode" in cfg.explicit:
        p.error("--peak-mode applies to an explicit --peak-file; default DE-SWAN extraction uses peak q values")
    if cfg.peak_file is not None and "peak_adjust_method" in cfg.explicit:
        p.error("--peak-adjust-method applies to DE-SWAN extraction, not a supplied --peak-file")
    for key in ["q_threshold", "peak_threshold"]:
        if not math.isfinite(getattr(cfg, key)) or not 0 < getattr(cfg, key) <= 1:
            p.error(f"{key} must be in (0, 1]")
    for key in ["min_samples", "top_n", "width_mm", "height_mm", "clr_pseudocount"]:
        if not math.isfinite(getattr(cfg, key)) or getattr(cfg, key) <= 0:
            p.error(f"{key} must be positive and finite")
    if math.isnan(cfg.age_min) or math.isnan(cfg.age_max) or cfg.age_min > cfg.age_max:
        p.error("Invalid age limits")
    return cfg


@dataclass
class Audit:
    mapping: list = field(default_factory=list)
    participants: list = field(default_factory=list)
    features: list = field(default_factory=list)
    events: dict = field(default_factory=dict)


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in (sorted(value) if isinstance(value, set) else value)]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(value, path):
    Path(path).write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(table, path):
    table.to_csv(path, index=False, na_rep="NA", encoding="utf-8", compression="infer", lineterminator="\n")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def table_columns(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input not found: {path}")
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        columns = pq.read_schema(path).names
    elif any(str(path).lower().endswith(s) for s in [".csv", ".csv.gz", ".tsv", ".tsv.gz"]):
        opener = gzip.open if path.suffix.lower() == ".gz" else open
        with opener(path, "rt", encoding="utf-8-sig", newline="") as f:
            columns = next(csv.reader(f, delimiter="\t" if ".tsv" in path.name.lower() else ","))
    else:
        raise ValueError("Supported table formats: Parquet, CSV, TSV, CSV.gz, TSV.gz")
    if len(columns) != len(set(columns)):
        raise ValueError(f"Duplicate column names in {path}")
    return columns


def read_table(path, columns=None, character_only=False):
    names = table_columns(path)
    if Path(path).suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=columns, engine="pyarrow")
    dtype = "string" if character_only else ({"participant_id": "string"} if "participant_id" in names else None)
    return pd.read_csv(path, usecols=columns, dtype=dtype,
                       sep="\t" if ".tsv" in str(path).lower() else ",", encoding="utf-8-sig",
                       keep_default_na=False, na_values=["", "NA", "NaN", "nan", "null"])


def resolve_input(folder, filename):
    candidates = [Path(folder) / filename] + [Path(folder) / (Path(filename).stem + s)
                                             for s in [".csv", ".csv.gz", ".tsv", ".tsv.gz"]]
    return next((p for p in candidates if p.is_file()), candidates[0])


def get_column(raw, field_name, aliases, section, maps, audit, required=True):
    requested = maps.get(section, {}).get(field_name)
    if requested:
        if requested not in raw:
            raise ValueError(f"Mapped column absent: {section}.{field_name} -> {requested}")
        chosen = requested
    elif field_name in raw:
        chosen = field_name
    else:
        candidates = [name for name in aliases if name in raw]
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous aliases for {section}.{field_name}: {candidates}; use --column-map")
        chosen = candidates[0] if candidates else None
    if chosen is None and required:
        raise ValueError(f"Missing field: {section}.{field_name}")
    audit.mapping.append({"input_role": section, "canonical_field": field_name, "source_column": chosen})
    return None if chosen is None else raw[chosen].reset_index(drop=True)


def canonical_omics(values):
    aliases = {
        "transcriptomic": "transcriptomics", "transcriptome": "transcriptomics", "rna": "transcriptomics",
        "metabolomic": "metabolomics", "metabolome": "metabolomics", "lipidomic": "lipidomics",
        "lipidome": "lipidomics", "gut_microbiota": "gut_microbiome", "oral_microbiota": "oral_microbiome",
        "转录组": "transcriptomics", "代谢组": "metabolomics", "脂质组": "lipidomics",
        "肠道微生物组": "gut_microbiome", "口腔微生物组": "oral_microbiome",
    }
    v = values.astype("string").str.strip().str.lower().str.replace(r"[- ]+", "_", regex=True).replace(aliases)
    if not v.isin(OMICS).all():
        raise ValueError(f"Missing or unknown omics label: {v[~v.isin(OMICS)].unique().tolist()}")
    return v


def check_ids(table, columns):
    for col in columns:
        table[col] = table[col].astype("string").str.strip()
        if table[col].isna().any() or table[col].eq("").any():
            raise ValueError(f"Missing identifier: {col}")
    return table


def probability(values, name):
    text = values.astype("string").str.strip()
    out = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    missing = text.isna() | text.str.lower().isin(["", "na", "nan", "null"])
    if (np.isnan(out) & ~missing.to_numpy()).any():
        raise ValueError(f"Nonnumeric probability in {name}")
    if (np.isfinite(out) & ((out < 0) | (out > 1))).any():
        raise ValueError(f"Probability outside [0,1] in {name}")
    out[~np.isfinite(out)] = np.nan
    return out


def parse_bool(values, name):
    text = values.astype("string").str.strip().str.lower()
    good = ["true", "1", "yes", "ok", "success", "valid", "completed"]
    bad = ["false", "0", "no", "failed", "invalid", "missing", "not_tested"]
    if (~(text.isna() | text.isin(good + bad))).any():
        raise ValueError(f"Unrecognized boolean in {name}")
    return text.isin(good).to_numpy(dtype=bool)


def discover_deswan_inputs(directory):
    """Find one upstream result per omics within this Result-02 analysis only."""
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(
            f"DE-SWAN results directory not found: {directory}. Run the earlier DE-SWAN analysis in Result-02, "
            "or set --deswan-dir to its existing output directory. --peak-file is only for explicit list reuse.")

    def discover(omics, kind, required=False):
        if kind == "deswan_breakpoint":
            endings = (".rds", ".parquet", ".csv", ".csv.gz", ".tsv", ".tsv.gz")
        else:
            endings = (".csv",)
        for folder in [directory, directory / "endogenous"]:
            if not folder.is_dir():
                continue
            found = [path for path in folder.iterdir() if path.is_file() and
                     (path.name.startswith(f"{omics}_{kind}_window_") or
                      any(path.name == f"{omics}_{kind}{suffix}" for suffix in endings) or
                      path.name == f"panel_c_{kind}_{omics}.csv") and path.name.lower().endswith(endings)]
            if len(found) > 1:
                raise ValueError(f"Multiple {kind} inputs for {omics} in {folder}; use --deswan-dir with one result/window per omics: {[p.name for p in found]}")
            if found:
                return found[0]
        if required:
            raise FileNotFoundError(f"Missing {omics}_deswan_breakpoint_window_*.rds (or a complete long-table export) in {directory}. Run the earlier DE-SWAN analysis for this omics first.")
        return None

    return {omics: {"result": discover(omics, "deswan_breakpoint", True),
                    "settings": discover(omics, "peak_settings"), "crest": discover(omics, "crest")}
            for omics in OMICS}


def read_deswan_tests(path, audit):
    """Deserialize existing DE-SWAN tests in Python; no model refit or R subprocess."""
    if Path(path).suffix.lower() == ".rds":
        try:
            import rdata
        except ImportError as error:
            raise ImportError("Reading DE-SWAN RDS requires the Python package rdata. Install with: python -m pip install 'rdata>=0.11.2,<2'") from error
        result = rdata.read_rds(path)
        if not isinstance(result, dict) or not isinstance(result.get("p"), pd.DataFrame):
            raise ValueError(f"DE-SWAN RDS must contain a data frame named p: {path}")
        raw = result["p"]
    else:
        raw = read_table(path, character_only=True)
    if raw.columns.duplicated().any():
        raise ValueError(f"Duplicate DE-SWAN result columns in {path}")
    def col(name, aliases=(), required=True):
        return get_column(raw, name, aliases, "deswan_peak", {}, audit, required)
    factor = col("factor").astype("string").str.strip()
    table = pd.DataFrame({"feature": col("feature", ["variable", "feature_id"]),
                          "peak_age": pd.to_numeric(col("peak_age", ["window.center", "window_center", "crest_age"]), errors="coerce").astype(float),
                          "factor": factor})
    p = col("p_value", ["pvalue", "p_peak"], False)
    q = col("q_value", ["qvalue", "q_peak"], False)
    if p is None and q is None:
        raise ValueError(f"DE-SWAN results lack age-effect p or q values: {path}")
    keep = table.factor.eq("qt").fillna(False)
    table = table.loc[keep].copy()
    if table.empty:
        raise ValueError(f"No DE-SWAN age-effect tests (factor == 'qt') in {path}")
    check_ids(table, ["feature"])
    if not np.isfinite(table.peak_age).all():
        raise ValueError(f"Invalid DE-SWAN window ages in {path}")
    if table.duplicated(["feature", "peak_age"]).any():
        raise ValueError(f"Duplicate DE-SWAN feature/age qt tests in {path}")
    table["p_value"] = np.nan if p is None else probability(p.loc[keep], "DE-SWAN peak p")
    table["q_value"] = np.nan if q is None else probability(q.loc[keep], "DE-SWAN peak q")
    audit.events.setdefault("deswan_input_rows", []).append({"source": str(path.resolve()),
        "n_rows_total": len(raw), "n_age_effect_rows": len(table), "n_other_factor_rows": len(raw) - len(table),
        "supplied_q_column": q is not None})
    return table.reset_index(drop=True), q is not None


def extract_deswan_peak_tests(table, omics, cfg, *, has_q=False, cutoff=None, method=None):
    """Extract the six user-defined exact-age tests using the full feature family."""
    cutoff = cfg.peak_threshold if cutoff is None else cutoff
    method = cfg.peak_adjust_method if method is None else method
    all_features = pd.Index(table.feature.drop_duplicates().sort_values(), name="feature")
    parts = []
    for crest, wanted_omics, age in CREST_RULES:
        if omics != wanted_omics:
            continue
        matching_ages = table.loc[np.isclose(table.peak_age, age, rtol=0, atol=1e-8), "peak_age"].unique()
        if len(matching_ages) != 1:
            raise ValueError(f"Exact peak age {age} must occur once in the window grid for {omics}; nearest-age substitution is not permitted")
        selected = table.loc[table.peak_age.eq(matching_ages[0])].set_index("feature")
        complete = selected.reindex(all_features).reset_index()
        complete["input_test_present"] = complete.feature.isin(selected.index)
        valid_p = np.isfinite(complete.p_value)
        if has_q:
            q_source, family_n = "supplied_peak_q_used_unchanged", np.nan
        else:
            complete["q_value"] = np.nan
            complete.loc[valid_p, "q_value"] = adjust_pvalues(complete.loc[valid_p, "p_value"].to_numpy(), method)
            q_source, family_n = "computed_from_all_valid_age_effect_p_in_omics_window", int(valid_p.sum())
        complete["crest"], complete["set_name"] = crest, "crest_1" if crest == "Crest1" else "crest_2"
        complete["omics"], complete["peak_age"] = omics, age
        complete["main_peak_rank"] = 1 if crest == "Crest1" else 2
        complete["q_cutoff"], complete["fdr_method"] = cutoff, "precomputed" if has_q else method
        complete["peak_q_source"], complete["peak_correction_family_n"] = q_source, family_n
        complete["peak_correction_scope"] = "upstream" if has_q else "within_omics_window_all_features"
        complete["peak_significant"] = complete.q_value.lt(cutoff)
        parts.append(complete.drop(columns="factor"))
    return pd.concat(parts, ignore_index=True)


def generate_peak_significant_molecules(cfg, sources, audit):
    """Create the significant peak list directly from Result-02 DE-SWAN outputs."""
    pieces, selected_peaks = [], []
    for omics in OMICS:
        files = sources[omics]
        path = files["result"]
        print(f"Extracting exact-age DE-SWAN features: {omics} ({path.name})", flush=True)
        settings = None
        if files["settings"] is not None:
            frame = read_table(files["settings"], character_only=True)
            if len(frame) != 1 or not {"q_cutoff", "fdr_method"}.issubset(frame.columns):
                raise ValueError(f"Peak settings need one row with q_cutoff and fdr_method: {files['settings']}")
            settings = frame.iloc[0]
            source_cutoff = float(settings.q_cutoff)
            if not math.isfinite(source_cutoff) or not 0 < source_cutoff <= 1:
                raise ValueError(f"Invalid upstream peak q cutoff for {omics}")
            if "omics" in settings and canonical_omics(pd.Series([settings.omics])).iloc[0] != omics:
                raise ValueError(f"Peak settings omics mismatch for {omics}")
        source_md5 = hashlib.md5(path.read_bytes()).hexdigest()
        if settings is not None and "source_md5" in settings and pd.notna(settings.source_md5):
            if source_md5 != str(settings.source_md5).lower():
                raise ValueError(f"DE-SWAN source hash differs from the peak settings for {omics}; regenerate the matching Result-02 peak tables")
        cutoff = source_cutoff if settings is not None and "peak_threshold" not in cfg.explicit else cfg.peak_threshold
        method = str(settings.fdr_method) if settings is not None and "peak_adjust_method" not in cfg.explicit else cfg.peak_adjust_method
        raw, has_q = read_deswan_tests(path, audit)
        tests = extract_deswan_peak_tests(raw, omics, cfg, has_q=has_q, cutoff=cutoff, method=method)
        tests["source_file"] = str(path.resolve())
        tests["source_rds"] = str(path.resolve()) if path.suffix.lower() == ".rds" else pd.NA
        tests["source_md5"], tests["source_sha256"] = source_md5, sha256(path)
        tests["molecule_id"] = tests.omics + "::" + tests.feature
        pieces.append(tests)
        crest_table = read_table(files["crest"]) if files["crest"] else None
        comparable = settings is not None and cutoff == source_cutoff and method == str(settings.fdr_method)
        for (crest, age), rows in tests.groupby(["crest", "peak_age"], sort=False):
            expected = np.nan
            if crest_table is not None and {"peak_age", "raw_n_at_peak"}.issubset(crest_table.columns):
                ages = pd.to_numeric(crest_table.peak_age, errors="raise")
                found = crest_table.loc[np.isclose(ages, age, rtol=0, atol=1e-8)]
                if len(found) > 1:
                    raise ValueError(f"Duplicate peak-count rows for {omics} at age {age}")
                if len(found) == 1:
                    expected = float(found.raw_n_at_peak.iloc[0])
            n_sig = int(rows.peak_significant.sum())
            count_status = "not_available" if not np.isfinite(expected) else "different_threshold_or_method" if not comparable else "matched" if expected == n_sig else "mismatch"
            selected_peaks.append({"crest": crest, "set_name": "crest_1" if crest == "Crest1" else "crest_2",
                "omics": omics, "peak_age": age, "n_features_total": len(rows), "n_tested": int(rows.q_value.notna().sum()),
                "n_significant": n_sig, "q_cutoff": cutoff, "fdr_method": rows.fdr_method.iloc[0],
                "peak_q_source": rows.peak_q_source.iloc[0], "upstream_raw_n_at_peak": expected,
                "count_check_status": count_status, "source_file": str(path.resolve()), "source_sha256": tests.source_sha256.iloc[0]})
            if count_status == "mismatch":
                write_csv(pd.DataFrame(selected_peaks), cfg.output_dir / "crest_selected_peaks.csv")
                raise ValueError(f"Peak significant count differs from Result-02 for {omics}, age {age}: {n_sig} vs {expected:g}")
            print(f"  {crest}, age {age:g}: {n_sig}/{rows.q_value.notna().sum()} peak-significant features", flush=True)
    all_tests = pd.concat(pieces, ignore_index=True)
    significant = all_tests.loc[all_tests.peak_significant].copy()
    write_csv(all_tests, cfg.output_dir / "crest_peak_all_tests.csv.gz")
    write_csv(significant, cfg.output_dir / "crest_peak_significant_molecules.csv")
    write_csv(pd.DataFrame(selected_peaks), cfg.output_dir / "crest_selected_peaks.csv")
    audit.events["peak_input_source"] = "generated_from_Result_01_DE_SWAN_results"
    audit.events["peak_correction_family"] = "Each omics x exact age window: all valid age-effect tests, before selecting significant features"
    audit.events["generated_peak_file"] = str((cfg.output_dir / "crest_peak_significant_molecules.csv").resolve())
    return significant


def build_membership(raw, cfg, maps=None, audit=None):
    maps, audit = maps or {}, audit if audit is not None else Audit()
    def col(name, aliases=(), required=True):
        return get_column(raw, name, aliases, "peak", maps, audit, required)
    peaks = pd.DataFrame({"omics": canonical_omics(col("omics")),
                          "feature_id": col("feature_id", ["feature", "molecule"]),
                          "crest_age": pd.to_numeric(col("crest_age", ["peak_age"]), errors="coerce")})
    check_ids(peaks, ["omics", "feature_id"])
    if not np.isfinite(peaks.crest_age).all():
        raise ValueError("Missing or invalid peak age")
    q = col("q_peak", ["q_value", "q"], False)
    peaks["q_peak"] = np.nan if q is None else probability(q, "q_peak")
    if cfg.peak_mode == "q":
        if q is None:
            raise ValueError("Peak q mode requires adjusted peak q values; raw p values are not silently corrected")
        peaks = peaks.loc[peaks.q_peak < cfg.peak_threshold]
    elif cfg.peak_mode == "flag":
        peaks = peaks.loc[parse_bool(col("peak_significant", ["is_significant", "significant"]), "peak_significant")]
    parts = []
    for crest, omics, age in CREST_RULES:
        subset = peaks.loc[peaks.omics.eq(omics) & peaks.crest_age.eq(age)].copy()
        if subset.empty:
            raise ValueError(f"No peak-significant members found for {crest}, {omics}, age {age}")
        subset["crest"] = crest
        parts.append(subset)
    members = pd.concat(parts, ignore_index=True)
    before = len(members)
    members = members.groupby(["crest", "omics", "feature_id", "crest_age"], as_index=False, sort=True).q_peak.min()
    members["peak_membership_source"] = cfg.peak_mode
    audit.events.update(peak_input_rows=len(raw), selected_membership_rows_before_dedup=before,
                        duplicate_memberships_collapsed=before - len(members), peak_mode=cfg.peak_mode)
    return members[["crest", "omics", "feature_id", "crest_age", "q_peak", "peak_membership_source"]]


def prepare_membership_outputs(members, output_dir):
    counts = members.groupby(["crest", "omics", "crest_age"], as_index=False).size().rename(columns={"size": "n_features_total"})
    totals = members.groupby("crest", as_index=False).size().rename(columns={"size": "n_features_total"})
    totals["omics"], totals["crest_age"] = "ALL", np.nan
    counts["crest_age"] = counts.crest_age.astype(float)
    overlap = members.loc[members.crest.eq("Crest1"), ["omics", "feature_id"]].merge(
        members.loc[members.crest.eq("Crest2"), ["omics", "feature_id"]], validate="one_to_one")
    write_csv(members, output_dir / "crest_membership.csv")
    write_csv(pd.concat([counts, totals], ignore_index=True), output_dir / "crest_feature_counts.csv")
    write_csv(overlap, output_dir / "crest_overlap.csv")
    print(f"Crest members: {totals.set_index('crest').n_features_total.to_dict()}; overlap retained: {len(overlap)}", flush=True)


def read_catalog(raw, maps=None, audit=None):
    maps, audit = maps or {}, audit if audit is not None else Audit()
    def col(name, aliases=(), required=True):
        return get_column(raw, name, aliases, "indicator", maps, audit, required)
    result = pd.DataFrame({"system": col("system"), "indicator_id": col("indicator_id", ["indicator", "feature"])})
    check_ids(result, ["system", "indicator_id"])
    labels = col("indicator_label", ["label"], False)
    result["indicator_label"] = result.indicator_id.str.replace("_", " ", regex=False) if labels is None else labels.astype("string").str.strip()
    missing = result.indicator_label.isna() | result.indicator_label.eq("")
    result.loc[missing, "indicator_label"] = result.loc[missing, "indicator_id"]
    result = result.drop_duplicates()
    if result.duplicated(["system", "indicator_id"]).any():
        raise ValueError("Conflicting indicator labels")
    if set(result.system) != set(SYSTEM_FILES):
        raise ValueError("Indicator catalogue must include exactly the 12 configured systems")
    if result.indicator_id.isin(META_COLS).any():
        raise ValueError("Indicator catalogue includes a metadata/covariate column")
    result["system_order"] = result.system.map({s: i + 1 for i, s in enumerate(SYSTEM_FILES)})
    return result.sort_values(["system_order", "indicator_id"]).reset_index(drop=True)


def numeric_vector(values):
    if pd.api.types.is_datetime64_any_dtype(values.dtype) or pd.api.types.is_timedelta64_dtype(values.dtype):
        return np.full(len(values), np.nan)
    out = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    out[~np.isfinite(out)] = np.nan
    return out


def clean_metadata(raw):
    if "participant_id" not in raw:
        raise ValueError("Participant data require participant_id")
    out = raw.copy()
    out["participant_id"] = out.participant_id.astype("string").str.strip()
    out["age"] = numeric_vector(out.age) if "age" in out else np.nan
    sex = out.sex if "sex" in out else pd.Series(pd.NA, index=out.index, dtype="string")
    out["sex"] = sex.astype("string").str.strip()
    out.loc[out.sex.str.lower().isin(MISSING_TEXT), "sex"] = pd.NA
    return out.loc[out.participant_id.notna() & out.participant_id.ne("")]


def mode_or_na(values):
    modes = values.mode(dropna=True)
    return modes.iloc[0] if len(modes) else pd.NA


def aggregate_participants(raw, features, role, audit):
    data = clean_metadata(raw[[c for c in dict.fromkeys(["participant_id", "age", "sex"] + features) if c in raw]])
    # Construct one numeric block: thousands of per-column insertions fragment wide omics tables.
    numeric = pd.DataFrame({name: numeric_vector(data[name]) if name in data else np.nan
                            for name in dict.fromkeys(features + ["age"])}, index=data.index)
    groups = data.groupby("participant_id", sort=True)
    out = numeric.groupby(data.participant_id, sort=True).mean().join(groups.sex.agg(mode_or_na).rename("sex"))
    audit.participants.append({
        "role": role, "input_rows": len(raw), "rows_missing_id": len(raw) - len(data),
        "participants": len(out), "repeated_rows_collapsed": len(data) - len(out),
        "participants_with_conflicting_sex": int((groups.sex.nunique() > 1).sum()),
    })
    return out.reset_index()


def clr_full_composition(raw, features, pseudocount=1.0):
    if not math.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError("CLR pseudocount must be positive and finite")
    matrix = np.column_stack([numeric_vector(raw[f]) for f in features])
    if (matrix < 0).any():
        raise ValueError("Negative microbiome abundance; input must be untransformed abundance")
    valid = np.isfinite(matrix).all(axis=1)
    out = np.full(matrix.shape, np.nan)
    if valid.any():
        logx = np.log(matrix[valid] + pseudocount)
        out[valid] = logx - logx.mean(axis=1, keepdims=True)
    return pd.DataFrame(out, columns=features, index=raw.index)


def load_omics_tables(members, cfg, paths, audit):
    tables, metadata = {}, []
    for om in OMICS:
        wanted = members.loc[members.omics.eq(om), "feature_id"].drop_duplicates().tolist()
        if set(wanted) & META_COLS:
            raise ValueError(f"Peak features overlap metadata columns in {om}")
        columns = table_columns(paths[om])
        microbe = om in ["gut_microbiome", "oral_microbiome"]
        selected = columns if microbe else [c for c in columns if c in META_COLS or c in wanted]
        raw = read_table(paths[om], columns=selected)
        metadata.append(clean_metadata(raw[[c for c in ["participant_id", "age", "sex"] if c in raw]]))
        present = {f: f in raw for f in wanted}
        if microbe:
            composition = [c for c in columns if c not in META_COLS]
            if not composition:
                raise ValueError(f"No microbiome composition columns in {om}")
            nonnumeric = [c for c in composition if not pd.api.types.is_numeric_dtype(raw[c].dtype)]
            if nonnumeric:
                raise ValueError(f"Nonnumeric microbiome composition columns in {om}: {nonnumeric}")
            clr = clr_full_composition(raw, composition, cfg.clr_pseudocount)
            audit.events[om + "_clr"] = {
                "n_composition_features": len(composition), "n_rows": len(clr),
                "n_incomplete_composition_rows": int(clr.isna().all(axis=1).sum()), "pseudocount": cfg.clr_pseudocount,
            }
            for name in wanted:
                if name in clr:
                    raw[name] = clr[name]
        tables[om] = aggregate_participants(raw, wanted, om, audit)
        for name in wanted:
            audit.features.append({"omics": om, "feature_id": name, "input_present": present[name],
                                   "n_participants_observed": int(tables[om][name].notna().sum()),
                                   "transformation": "CLR_full_composition" if microbe else "supplied_numeric_scale"})
        print(f"Loaded {om}: {len(tables[om])} participants, {len(wanted)} union features", flush=True)
    pooled = pd.concat(metadata, ignore_index=True)
    fallback = pooled.groupby("participant_id", sort=True).agg(age=("age", "median"), sex=("sex", mode_or_na)).reset_index()
    return tables, fallback


def coalesce_covariates(system, fallback):
    result = system.merge(fallback, on="participant_id", how="left", suffixes=("", "_omics"), validate="one_to_one")
    result["age"] = result.age.fillna(result.age_omics)
    result["sex"] = result.sex.fillna(result.sex_omics)
    return result.drop(columns=["age_omics", "sex_omics"])


def mask_groups(matrix):
    groups = {}
    for j in range(matrix.shape[1]):
        key = np.packbits(np.isfinite(matrix[:, j])).tobytes()
        groups.setdefault(key, []).append(j)
    return list(groups.values())


def fit_matrix_block(X, Y, C, min_samples=50):
    """Exact pairwise OLS via residualization; columns in a block share a mask."""
    n, nx, ny = X.shape[0], X.shape[1], Y.shape[1]
    ix, iy = np.repeat(np.arange(nx), ny), np.tile(np.arange(ny), nx)
    out = pd.DataFrame({"ix": ix, "iy": iy, "n": n, "df_resid": np.nan,
                        "standardized_beta": np.nan, "partial_r": np.nan, "t_value": np.nan,
                        "p_association": np.nan, "test_status": "insufficient_samples"})
    if n < min_samples:
        return out
    q, r, _ = linalg.qr(C, mode="economic", pivoting=True, check_finite=False)
    diagonal = np.abs(np.diag(r))
    rank = int(np.sum(diagonal > 1e-10 * diagonal.max()))
    df = n - rank - 1
    out["df_resid"] = df
    if df <= 0:
        out["test_status"] = "no_residual_df"
        return out
    xc, yc = X - X.mean(axis=0), Y - Y.mean(axis=0)
    sx0, sy0 = np.sum(xc ** 2, axis=0), np.sum(yc ** 2, axis=0)
    xs = xc / np.sqrt(np.where(sx0 > 0, sx0, 1))
    ys = yc / np.sqrt(np.where(sy0 > 0, sy0, 1))
    basis = q[:, :rank]
    xr, yr = xs - basis @ (basis.T @ xs), ys - basis @ (basis.T @ ys)
    sx, sy, cross = np.sum(xr ** 2, axis=0), np.sum(yr ** 2, axis=0), xr.T @ yr
    valid_x = np.isfinite(sx0) & (sx0 > 0) & (sx > 1e-12)
    valid_y = np.isfinite(sy0) & (sy0 > 0) & (sy > 1e-12)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        pr = np.clip(cross / np.sqrt(np.outer(sx, sy)), -1, 1)
        t = pr * np.sqrt(df / np.maximum(0, 1 - pr ** 2))
        beta = cross / sx[:, None]
        p = 2 * stats.t.sf(np.abs(t), df)
    out["standardized_beta"], out["partial_r"] = beta[ix, iy], pr[ix, iy]
    out["t_value"], out["p_association"], out["test_status"] = t[ix, iy], p[ix, iy], "ok"
    out.loc[~valid_x[ix], "test_status"] = "constant_or_collinear_feature"
    out.loc[~valid_y[iy], "test_status"] = "constant_or_collinear_indicator"
    out.loc[out.test_status.eq("ok") & ~(np.isfinite(out.p_association) & np.isfinite(out.standardized_beta)), "test_status"] = "numerical_failure"
    out.loc[~out.test_status.eq("ok"), ["standardized_beta", "partial_r", "t_value", "p_association"]] = np.nan
    return out


def fit_one_omics_system(omics_dt, system_dt, features, indicators, omics_name, system_name, cfg):
    xdata, ydata = omics_dt.set_index("participant_id"), system_dt.set_index("participant_id")
    ids = xdata.index.intersection(ydata.index, sort=False)
    xdata, ydata = xdata.loc[ids], ydata.loc[ids]
    eligible = np.isfinite(ydata.age) & ydata.sex.notna() & ydata.age.between(cfg.age_min, cfg.age_max)
    X = xdata.loc[eligible, features].to_numpy(dtype=float)
    Y = ydata.loc[eligible, indicators].to_numpy(dtype=float)
    age = ydata.loc[eligible, "age"].to_numpy(dtype=float)
    sex = pd.get_dummies(ydata.loc[eligible, "sex"], drop_first=True, dtype=float).to_numpy()
    C = np.column_stack([np.ones(len(age)), age, sex])
    chunks = []
    for gx in mask_groups(X):
        for gy in mask_groups(Y):
            complete = np.isfinite(X[:, gx[0]]) & np.isfinite(Y[:, gy[0]])
            result = fit_matrix_block(X[complete][:, gx], Y[complete][:, gy], C[complete], cfg.min_samples)
            result["feature_id"] = np.asarray(features)[np.asarray(gx)[result.ix.to_numpy()]]
            result["indicator_id"] = np.asarray(indicators)[np.asarray(gy)[result.iy.to_numpy()]]
            chunks.append(result.drop(columns=["ix", "iy"]))
    out = pd.concat(chunks, ignore_index=True)
    out["omics"], out["system"] = omics_name, system_name
    out["test_valid"] = out.test_status.eq("ok")
    return out


def adjust_pvalues(p, method):
    p = np.asarray(p, dtype=float)
    n = len(p)
    if not n:
        return p.copy()
    order = np.argsort(p, kind="stable")
    ranked, ranks = p[order], np.arange(1, n + 1)
    if method in ["BH", "fdr", "BY"]:
        factor = np.sum(1 / ranks) if method == "BY" else 1.0
        q = np.minimum.accumulate((ranked * n * factor / ranks)[::-1])[::-1]
    elif method == "holm":
        q = np.maximum.accumulate(ranked * (n - ranks + 1))
    elif method == "bonferroni":
        q = ranked * n
    elif method == "none":
        q = ranked
    else:
        raise ValueError(f"Unknown correction method: {method}")
    result = np.empty(n)
    result[order] = np.clip(q, 0, 1)
    return result


def adjust_associations(associations, method="BH", scope="global"):
    if scope not in SCOPE_COLUMNS:
        raise ValueError(f"Unknown correction scope: {scope}")
    out = associations.reset_index(drop=True).copy()
    out["q_association"], out["correction_family_n"] = np.nan, np.nan
    valid = out.test_valid & np.isfinite(out.p_association)
    groups = out.loc[valid].groupby(SCOPE_COLUMNS[scope], sort=False).groups.values() if SCOPE_COLUMNS[scope] else [out.index[valid]]
    for idx in groups:
        out.loc[idx, "q_association"] = adjust_pvalues(out.loc[idx, "p_association"].to_numpy(), method)
        out.loc[idx, "correction_family_n"] = len(idx)
    out["correction_method"], out["correction_scope"] = method, scope
    return out


def run_new_associations(members, catalog, cfg, paths, audit):
    omics_tables, fallback = load_omics_tables(members, cfg, paths["omics"], audit)
    parts, inventory, generated = [], [], []
    for system in SYSTEM_FILES:
        print(f"System: {system}", flush=True)
        raw = read_table(paths["systems"][system])
        indicators = [c for c in raw if c not in META_COLS] if catalog is None else catalog.loc[catalog.system.eq(system), "indicator_id"].tolist()
        if not indicators:
            raise ValueError(f"No indicators in {system}")
        absent = set(indicators) - set(raw)
        for name in indicators:
            values = raw[name] if name in raw else pd.Series(np.nan, index=raw.index)
            numeric = numeric_vector(values)
            inventory.append({"system": system, "indicator_id": name, "input_present": name not in absent,
                              "source_class": str(values.dtype), "n_source_rows": len(raw),
                              "n_numeric_observed": int(np.isfinite(numeric).sum()),
                              "n_nonnumeric_observed": int((values.notna() & np.isnan(numeric)).sum())})
        data = coalesce_covariates(aggregate_participants(raw, indicators, system, audit), fallback)
        audit.events[system + "_participant_covariates"] = {
            "n_participants": len(data), "n_age_observed": int(data.age.notna().sum()),
            "n_sex_observed": int(data.sex.notna().sum()),
            "n_covariates_and_age_limits_eligible": int((data.age.between(cfg.age_min, cfg.age_max) & data.sex.notna()).sum()),
        }
        generated.extend({"system": system, "indicator_id": name, "indicator_label": name.replace("_", " ")} for name in indicators)
        for om in OMICS:
            features = members.loc[members.omics.eq(om), "feature_id"].drop_duplicates().tolist()
            result = fit_one_omics_system(omics_tables[om], data, features, indicators, om, system, cfg)
            missing_features = [f["feature_id"] for f in audit.features if f["omics"] == om and not f["input_present"]]
            result.loc[result.feature_id.isin(missing_features), "test_status"] = "feature_absent_from_input"
            result.loc[result.indicator_id.isin(absent), "test_status"] = "indicator_absent_from_input"
            result["test_valid"] = result.test_status.eq("ok")
            parts.append(result)
            print(f"  {om}: {result.test_valid.sum()}/{len(result)} valid tests", flush=True)
    catalog = read_catalog(pd.DataFrame(generated), audit=audit) if catalog is None else catalog
    audit.events["new_tests_family"] = "All valid unique tests in the two-crest feature union x all catalogued indicators, before crest splitting and ranking"
    associations = adjust_associations(pd.concat(parts, ignore_index=True), cfg.p_adjust_method, cfg.p_adjust_scope)
    return associations, catalog, pd.DataFrame(inventory)


def load_associations(raw, catalog, cfg, maps=None, audit=None):
    maps, audit = maps or {}, audit if audit is not None else Audit()
    def col(name, aliases=(), required=True):
        return get_column(raw, name, aliases, "association", maps, audit, required)
    a = pd.DataFrame({"omics": canonical_omics(col("omics")), "feature_id": col("feature_id", ["feature", "molecule"]),
                      "indicator_id": col("indicator_id", ["indicator", "phenotype"])})
    system = col("system", required=False)
    if system is None:
        if catalog.indicator_id.duplicated().any():
            raise ValueError("Associations require system because indicator IDs repeat across systems")
        a["system"] = a.indicator_id.astype("string").str.strip().map(catalog.set_index("indicator_id").system)
    else:
        a["system"] = system
    check_ids(a, TEST_KEYS)
    known = pd.MultiIndex.from_frame(catalog[["system", "indicator_id"]])
    if not pd.MultiIndex.from_frame(a[["system", "indicator_id"]]).isin(known).all():
        raise ValueError("Association indicators absent from the catalogue")
    q = col("q_association", ["q_value", "fdr_global", "q"], False)
    p = col("p_association", ["p_value", "p"], False)
    valid = col("test_valid", ["tested", "test_completed"], False)
    status = col("test_status", ["status"], False)
    if q is None and p is None:
        raise ValueError("Association input needs adjusted q or raw p values")
    a["p_association"] = np.nan if p is None else probability(p, "p_association")
    a["q_association"] = np.nan if q is None else probability(q, "q_association")
    valid_flag = np.ones(len(a), dtype=bool) if valid is None else parse_bool(valid, "test_valid")
    if valid is None and status is not None:
        valid_flag = status.astype("string").str.strip().str.lower().isin(["ok", "success", "valid", "completed"]).to_numpy()
    a["test_valid"] = valid_flag & np.isfinite(a.p_association if q is None else a.q_association)
    a["test_status"] = status if status is not None else np.where(a.test_valid, "ok", "invalid_or_missing_probability")
    # Inferential duplicates must agree; never choose one conflicting test silently.
    before = len(a)
    a = a.drop_duplicates().reset_index(drop=True)
    conflicts = a.loc[a.duplicated(TEST_KEYS, keep=False)]
    if len(conflicts):
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(conflicts, cfg.output_dir / "conflicting_association_duplicates.csv")
        raise ValueError("Conflicting duplicate associations; inspect conflicting_association_duplicates.csv")
    audit.events.update(association_input_rows=before, association_exact_duplicates_collapsed=before - len(a))
    if q is None:
        if not {"p_adjust_method", "p_adjust_scope"}.issubset(cfg.explicit):
            raise ValueError("Imported raw p values require explicit --p-adjust-method and --p-adjust-scope before significance or ranking")
        a = adjust_associations(a, cfg.p_adjust_method, cfg.p_adjust_scope)
        audit.events["association_q_source"] = "Computed from the complete imported p-value family, before crest selection"
    else:
        a["correction_method"], a["correction_scope"], a["correction_family_n"] = "precomputed_q_used_unchanged", "upstream", np.nan
        audit.events["association_q_source"] = "Supplied adjusted q values, no repeated adjustment and no p fallback"
    return a


def summarize_associations(members, a, catalog, q_threshold=0.05, top_n=10):
    if a.duplicated(TEST_KEYS).any() or members.duplicated(["crest", "omics", "feature_id"]).any():
        raise ValueError("Associations and within-crest memberships must be unique before summarization")
    a = a.copy()
    a["valid_for_proportion"] = a.test_valid & np.isfinite(a.q_association) & a.q_association.between(0, 1)
    a["significant"] = a.valid_for_proportion & a.q_association.lt(q_threshold)
    matched = a.merge(members[["crest", "omics", "feature_id"]], on=["omics", "feature_id"], how="inner")
    group = ["crest", "omics", "system", "indicator_id"]
    counts = matched.groupby(group, as_index=False).agg(n_records=("test_valid", "size"),
                                                       n_tested=("valid_for_proportion", "sum"), n_significant=("significant", "sum"))
    totals = members.groupby(["crest", "omics"], as_index=False).size().rename(columns={"size": "n_features_total"})
    strata = totals.merge(catalog, how="cross").merge(counts, on=group, how="left", validate="one_to_one")
    for col in ["n_records", "n_tested", "n_significant"]:
        strata[col] = strata[col].fillna(0).astype(int)
    strata["n_no_record"] = strata.n_features_total - strata.n_records
    strata["n_invalid_or_missing_q"] = strata.n_records - strata.n_tested
    strata["n_untested"] = strata.n_features_total - strata.n_tested
    if not ((strata.n_no_record >= 0) & (strata.n_invalid_or_missing_q >= 0) & (strata.n_tested >= strata.n_significant)).all():
        raise ValueError("Invalid feature/test counts")
    count_cols = ["n_features_total", "n_records", "n_tested", "n_significant", "n_no_record", "n_invalid_or_missing_q", "n_untested"]
    id_cols = ["system_order", "system", "indicator_id", "indicator_label"]
    pooled = strata.groupby(["crest"] + id_cols, as_index=False)[count_cols].sum()
    for table in [strata, pooled]:
        table["proportion"] = table.n_significant / table.n_tested.replace(0, np.nan)
        table["test_coverage"] = table.n_tested / table.n_features_total
    measures = count_cols + ["proportion", "test_coverage"]
    wide = pooled.pivot(index=id_cols, columns="crest", values=measures)
    wide.columns = [f"{metric}_{crest}" for metric, crest in wide.columns]
    wide = wide.reset_index()
    props = wide[["proportion_Crest1", "proportion_Crest2"]]
    wide["ranking_score"], wide["mean_proportion"] = props.max(axis=1), props.mean(axis=1)
    wide = wide.sort_values(["system_order", "ranking_score", "mean_proportion", "indicator_label", "indicator_id"],
                            ascending=[True, False, False, True, True], na_position="last", kind="stable").reset_index(drop=True)
    wide["rank_within_system"] = pd.Series(pd.NA, index=wide.index, dtype="Int64")
    eligible = wide.ranking_score.notna()
    wide.loc[eligible, "rank_within_system"] = wide.loc[eligible].groupby("system", sort=False).cumcount().to_numpy() + 1
    wide["selected_top10"] = wide.rank_within_system.le(top_n).fillna(False).astype(bool)
    ranking = wide[["system", "indicator_id", "ranking_score", "mean_proportion", "rank_within_system", "selected_top10"]]
    pooled = pooled.merge(ranking, on=["system", "indicator_id"], validate="many_to_one")
    strata = strata.merge(ranking, on=["system", "indicator_id"], validate="many_to_one")
    c1 = strata.loc[strata.crest.eq("Crest1")].merge(
        pooled.loc[pooled.crest.eq("Crest1"), ["system", "indicator_id", "n_tested"]].rename(columns={"n_tested": "pooled_n_tested"}),
        on=["system", "indicator_id"], validate="many_to_one")
    c1["denominator_weight"] = c1.n_tested / c1.pooled_n_tested.replace(0, np.nan)
    c1["pooled_proportion_contribution"] = c1.n_significant / c1.pooled_n_tested.replace(0, np.nan)
    pooled = pooled.sort_values(["system_order", "rank_within_system", "indicator_id", "crest"], na_position="last").reset_index(drop=True)
    c1 = c1.sort_values(["system_order", "rank_within_system", "indicator_id", "omics"], na_position="last").reset_index(drop=True)
    return {"wide": wide, "long": pooled, "top": pooled.loc[pooled.selected_top10].copy(), "crest1_omics": c1}


def figure_caption(cfg, imported=False):
    model = ("Supplied association q values are used unchanged; imported raw p values require an explicitly specified correction. " if imported else
             f"Two-sided participant-level OLS models adjust for age and sex, using at least {cfg.min_samples} complete participants per pair. "
             f"New p values are adjusted by {cfg.p_adjust_method}, scope {cfg.p_adjust_scope}, before crest splitting or ranking. "
             "The global family includes all valid unique tests of the two-crest feature union against every catalogued indicator. ")
    return (
        "Significant molecular feature associations across 12 systems, split into two prespecified six-system figures. "
        "In group1, panels are ordered by decreasing mean proportion of significantly associated features across the displayed "
        "indicators and both crests (ties broken by the maximum proportion); group2 panels follow the listed system order. Crest1 is the union of peak-significant features from "
        "transcriptomics at 52, metabolomics at 48, lipidomics at 48, gut microbiome at 50 and oral microbiome at 48 years. "
        "Crest2 comprises peak-significant transcriptomic features at 64 years. Shared features remain in both crests. "
        f"For each indicator, the numerator counts distinct crest features with association q < {cfg.q_threshold}; "
        "the denominator counts valid completed tests with a finite association q. Positive and negative associations both count. "
        "Crest1 pools the five numerators and denominators, with equal weight per feature. Zero denominators are NA. "
        f"Both crests display the same top {cfg.top_n} eligible indicators per system, ranked by their maximum proportion, "
        "then available-case mean, indicator label and ID. " + model +
        "Repeated numeric observations are averaged within participant. Peak ages define membership and do not select nearby-aged participants. "
        "Point height indicates the proportion of significantly associated features, not effect magnitude or association direction. "
        "The y-axis spans 0-50%; upward triangles at 50% mark proportions above the displayed range. "
        "Crest omics compositions differ, so proportion differences cannot be attributed to age stage alone. "
        "No between-crest difference test or significance marker is shown."
    )


def fit_x_label_space(fig, axes, padding_mm=3):
    """Fit rendered tick labels while preserving figure width and plot-area sizes."""
    initial_height_mm = fig.get_figheight() * 25.4
    for _ in range(3):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        padding_px = padding_mm * fig.dpi / 25.4
        row_label_bottoms = []
        for row in axes:
            boxes = [label.get_window_extent(renderer) for ax in row for label in ax.get_xticklabels()
                     if label.get_visible() and label.get_text()]
            row_label_bottoms.append(min((box.y0 for box in boxes), default=min(ax.bbox.y0 for ax in row)))
        lower_row_top = max(ax.get_tightbbox(renderer).y1 for ax in axes[1])
        row_clearance_px = row_label_bottoms[0] - lower_row_top
        extra_bottom_px = max(0, padding_px - row_label_bottoms[-1])
        extra_gap_px = max(0, padding_px - row_clearance_px)
        if max(extra_bottom_px, extra_gap_px) <= 0.25:
            return {"initial_height_mm": initial_height_mm,
                    "final_height_mm": fig.get_figheight() * 25.4,
                    "minimum_padding_mm": padding_mm,
                    "bottom_label_margin_mm": row_label_bottoms[-1] / fig.dpi * 25.4,
                    "between_rows_clearance_mm": row_clearance_px / fig.dpi * 25.4}
        height_px = fig.bbox.height
        bottom_px = fig.subplotpars.bottom * height_px
        top_px = fig.subplotpars.top * height_px
        axes_height_px = axes[0, 0].bbox.height
        gap_px = axes[0, 0].bbox.y0 - axes[1, 0].bbox.y1
        new_height_px = height_px + extra_bottom_px + extra_gap_px
        fig.set_size_inches(fig.get_figwidth(), new_height_px / fig.dpi, forward=True)
        fig.subplots_adjust(bottom=(bottom_px + extra_bottom_px) / new_height_px,
                            top=(top_px + extra_bottom_px + extra_gap_px) / new_height_px,
                            hspace=(gap_px + extra_gap_px) / axes_height_px)
    raise RuntimeError("Could not fit x-axis labels with the required page and row clearance")


def system_strength(plot_data, systems):
    """Mean displayed proportion (top indicators x both crests) per system; max proportion breaks ties."""
    rows = []
    for order, system in enumerate(systems):
        y = pd.to_numeric(plot_data.loc[plot_data.system.eq(system), "proportion"], errors="coerce")
        y = y[np.isfinite(y)]
        rows.append({"system": system, "listed_order": order + 1,
                     "mean_proportion": y.mean() if len(y) else np.nan,
                     "max_proportion": y.max() if len(y) else np.nan, "n_points": int(len(y))})
    table = pd.DataFrame(rows).sort_values(["mean_proportion", "max_proportion", "listed_order"],
                                           ascending=[False, False, True], na_position="last", kind="stable")
    table["panel_position"] = np.arange(1, len(table) + 1)
    return table.reset_index(drop=True)


def ordered_systems(plot_data, group):
    systems = FIGURE_GROUPS[group]
    if group not in STRENGTH_ORDERED_GROUPS:
        return tuple(systems)
    return tuple(system_strength(plot_data, systems).system)


def build_figure(plot_data, catalog, cfg, fixture=False, group="group1"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [cfg.font, "Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 6.5, "pdf.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": 0.5,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "text.usetex": False, "figure.autolayout": False, "figure.constrained_layout.use": False})
    systems = ordered_systems(plot_data, group)
    fig, axes = plt.subplots(2, 3, figsize=(210 / 25.4, cfg.height_mm / 25.4), squeeze=False)
    # Equal plot areas leave physical space for the rotated indicator labels.
    fig.subplots_adjust(left=0.085, right=0.985, top=0.905, bottom=0.27, hspace=2.15, wspace=0.23)
    y_max = 0.5
    for index, (system, ax) in enumerate(zip(systems, axes.flat)):
        ax.set_gid(system)
        data = plot_data.loc[plot_data.system.eq(system)].copy()
        labels = data[["indicator_id", "indicator_label", "rank_within_system"]].drop_duplicates().sort_values("rank_within_system")
        ids = labels.indicator_id.tolist()
        x = np.arange(len(ids), dtype=float)
        for crest, delta in [("Crest1", -0.14), ("Crest2", 0.14)]:
            y = data.loc[data.crest.eq(crest)].set_index("indicator_id").reindex(ids).proportion.to_numpy(dtype=float)
            valid = np.isfinite(y)
            ax.vlines(x[valid] + delta, 0, np.minimum(y[valid], y_max), colors="#BBBBBB", linewidth=0.55, zorder=2)
            within = valid & (y <= y_max)
            ax.scatter(x[within] + delta, y[within], color=CREST_COLORS[crest], s=10, linewidths=0, zorder=3, clip_on=False)
            # Indicate values beyond the requested view limit without changing the data.
            above = valid & (y > y_max)
            if above.any():
                ax.scatter(x[above] + delta, np.full(above.sum(), y_max), color=CREST_COLORS[crest],
                           marker="^", s=12, linewidths=0, zorder=3, clip_on=False)
            for missing_x in x[~valid]:
                ax.text(missing_x + delta, -0.025, "NA", transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=5.5, color="#444444", clip_on=False)
        ax.set_xlim(-0.6, max(len(ids) - 0.4, 0.6))
        ax.set_ylim(0, y_max)
        ax.set_yticks(np.linspace(0, y_max, 6))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.tick_params(axis="y", labelsize=6.5, length=2.5, width=0.5, pad=2)
        ax.tick_params(axis="x", length=0, pad=8, labelsize=6.5)
        wrapped = []
        for label in labels.indicator_label:
            # At 210 mm, each indicator slot accommodates at most two text lines.
            wrap_width = 30
            lines = textwrap.wrap(str(label), width=wrap_width, break_long_words=True, break_on_hyphens=False)
            while len(lines) > 2:
                wrap_width += 1
                lines = textwrap.wrap(str(label), width=wrap_width, break_long_words=True, break_on_hyphens=False)
            wrapped.append("\n".join(lines))
        ax.set_xticks(x, wrapped, rotation=90, ha="right", va="center", rotation_mode="anchor")
        ax.grid(axis="y", color="#EEEEEE", linewidth=0.4, zorder=0)
        ax.set_title(SYSTEM_LABELS[system], loc="left", pad=5, fontsize=7)
        if group not in UNLABELED_GROUPS:
            ax.annotate(chr(97 + index), xy=(0, 1), xycoords="axes fraction", xytext=(-20, 5),
                        textcoords="offset points", fontsize=7, fontweight="bold", va="baseline")
        if not ids:
            ax.grid(False)
            ax.text(0.5, 0.5, "No indicators with valid tests", transform=ax.transAxes,
                    ha="center", va="center", fontsize=6.5, color="#444444")
    fig._crest_x_label_layout = fit_x_label_space(fig, axes)
    fig._crest_systems = list(systems)
    fig._crest_system_strength = (system_strength(plot_data, FIGURE_GROUPS[group])
                                  if group in STRENGTH_ORDERED_GROUPS else None)
    handles = [Line2D([], [], color=CREST_COLORS[c], marker="o", linestyle="none", markersize=3.5) for c in CREST_COLORS]
    fig.legend(handles, ["Crest1: earlier peaks across five omics", "Crest2: transcriptomic peak at 64 years"],
               loc="upper center", bbox_to_anchor=(0.535, 0.989), ncol=2, fontsize=7,
               columnspacing=2.5, handletextpad=0.5)
    fig.text(0.016, (fig.subplotpars.top + fig.subplotpars.bottom) / 2, "Proportion of significantly associated features", rotation=90,
             ha="center", va="center", rotation_mode="anchor", fontsize=7)
    if fixture:
        fig.text(0.995, 0.5, "TEST FIXTURE - not empirical results", rotation=90,
                 ha="right", va="center", rotation_mode="anchor", fontsize=5.5, color="#666666")
    return fig


def figure_paths(cfg, group):
    stem = cfg.figure_dir.resolve() / f"crest_system_association_{group}"
    return {"pdf": stem.with_suffix(".pdf"), "png": stem.with_suffix(".png"), "svg": stem.with_suffix(".svg")}


def export_figure(fig, cfg, group="group1"):
    import matplotlib.pyplot as plt
    qa = cfg.output_dir / "qa" / group
    qa.mkdir(parents=True, exist_ok=True)
    paths = figure_paths(cfg, group)
    status_file = qa / "figure_export_status.json"
    status = {"status": "running", "stage": "create_figure_directory", "group": group,
              "systems": list(getattr(fig, "_crest_systems", FIGURE_GROUPS[group])), "expected_files": paths, "saved_files": {}}
    write_json(status, status_file)
    try:
        cfg.figure_dir.mkdir(parents=True, exist_ok=True)
        (qa / "README_figures.txt").write_text(
            "The final association figures are saved here:\n" +
            "\n".join(str(path) for path in paths.values()) +
            "\nCheck figure_export_status.json if any format is missing.\n", encoding="utf-8")
        print(f"Figure group: {group}; panel order: {', '.join(SYSTEM_LABELS[s] for s in status['systems'])}", flush=True)
        if getattr(fig, "_crest_system_strength", None) is not None:
            write_csv(fig._crest_system_strength, qa / "system_strength_order.csv")
        print(f"Figure directory: {cfg.figure_dir.resolve()}", flush=True)
        width_mm, height_mm = fig.get_size_inches() * 25.4
        status["figure_size_mm"] = [width_mm, height_mm]
        write_json(fig._crest_x_label_layout, qa / "x_label_layout.json")
        print(f"Figure size: {width_mm:g} x {height_mm:g} mm; y-axis: 0-50%; no overall title or footer", flush=True)
        fig.canvas.draw()
        for ext, path in paths.items():
            status["stage"] = "save_" + ext
            write_json(status, status_file)
            print(f"Saving {ext.upper()}: {path}", flush=True)
            fig.savefig(path, format=ext, **({"dpi": 300} if ext == "png" else {}))
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"No nonempty {ext.upper()} was produced: {path}")
            status["saved_files"][ext] = {"path": path, "bytes": path.stat().st_size, "sha256": sha256(path)}
            write_json(status, status_file)
        pdf_status = "not_run"
        status.update(status="all_formats_exported", stage="finished", pdf_audit_status=pdf_status)
        write_json(status, status_file)
        print(f"Saved PDF, PNG (300 dpi) and SVG to: {cfg.figure_dir.resolve()}", flush=True)
        return pdf_status
    except Exception as error:
        status.update(status="failed", error=str(error))
        write_json(status, status_file)
        raise RuntimeError(
            f"Figure export failed at {status['stage']}: {error}. "
            f"See {status_file.resolve()}. Expected association figures are in {cfg.figure_dir.resolve()}.") from error
    finally:
        plt.close(fig)


def export_grouped_figures(plot_data, catalog, cfg, fixture=False):
    """Render the two fixed system groups without changing the pooled analysis."""
    statuses = {}
    for group in FIGURE_GROUPS:
        fig = build_figure(plot_data, catalog, cfg, fixture=fixture, group=group)
        statuses[group] = export_figure(fig, cfg, group=group)
    return statuses


def file_inventory(cfg):
    paths = {"omics": {om: resolve_input(cfg.omics_dir, om + "_with_metadata.parquet") for om in OMICS},
             "systems": {s: resolve_input(cfg.systems_dir, filename) for s, filename in SYSTEM_FILES.items()}}
    entries = [("omics:" + k, v) for k, v in paths["omics"].items()] + [("system:" + k, v) for k, v in paths["systems"].items()]
    if cfg.peak_file is not None:
        entries.insert(0, ("peak", cfg.peak_file))
    else:
        paths["deswan"] = discover_deswan_inputs(cfg.deswan_dir)
        entries = [(f"deswan_{kind}:{omics}", path) for omics, items in paths["deswan"].items()
                   for kind, path in items.items() if path is not None] + entries
    entries += [(k, getattr(cfg, k + "_file")) for k in ["association", "indicator"] if getattr(cfg, k + "_file")]
    if cfg.column_map:
        entries.append(("column_map", cfg.column_map))
    rows = []
    for role, path in entries:
        used = not role.startswith(("omics:", "system:")) or (not cfg.prepare_only and cfg.association_file is None)
        rows.append({"role": role, "path": str(path.resolve()), "exists": path.is_file(),
                     "bytes": path.stat().st_size if path.is_file() else np.nan, "used_this_run": used,
                     "sha256": sha256(path) if used and path.is_file() else None})
    paths["inventory"] = pd.DataFrame(rows)
    return paths


def save_audits(audit, cfg):
    for entries, filename in [(audit.mapping, "column_mapping.csv"), (audit.participants, "participant_audit.csv"),
                              (audit.features, "feature_input_audit.csv")]:
        if entries:
            write_csv(pd.DataFrame(entries).drop_duplicates(), cfg.output_dir / filename)
    versions = {"python": sys.version, "executable": sys.executable, "platform": platform.platform()}
    for name in ["numpy", "pandas", "scipy", "matplotlib", "pyarrow", "pymupdf", "rdata"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed_in_analysis_interpreter"
    write_json(versions, cfg.output_dir / "python_environment.json")


def main(argv=None, here=HERE):
    cfg = parse_config(argv, here)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    audit = Audit()
    manifest = {"status": "started", "started_utc": datetime.now(timezone.utc).isoformat(), "config": vars(cfg),
                "script_sha256": sha256(Path(__file__)), "implementation": "python",
                "participant_age_policy": "all_available_ages" if cfg.age_min == -math.inf and cfg.age_max == math.inf else "explicit_overall_age_limits"}
    def checkpoint(status):
        manifest.update(status=status, audit=audit.events, updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(manifest, cfg.output_dir / "analysis_manifest.json")
    try:
        maps = json.loads(cfg.column_map.read_text(encoding="utf-8-sig")) if cfg.column_map else {}
        paths = file_inventory(cfg)
        write_csv(paths["inventory"], cfg.output_dir / "input_file_inventory.csv")
        if cfg.peak_file is None:
            peak_data = generate_peak_significant_molecules(cfg, paths["deswan"], audit)
            # Generated rows are already significant under their recorded per-omics settings.
            members = build_membership(peak_data, cfg, {}, audit)
        else:
            audit.events["peak_input_source"] = "explicit_peak_file"
            members = build_membership(read_table(cfg.peak_file, character_only=True), cfg, maps, audit)
        prepare_membership_outputs(members, cfg.output_dir)
        manifest.update(n_crest_memberships=len(members), n_unique_features=len(members[["omics", "feature_id"]].drop_duplicates()))
        if cfg.prepare_only:
            checkpoint("prepared_membership_only_no_association_results")
            print("Prepared verified crest members. No empirical association figure was generated.", flush=True)
            return {"members": members, "cfg": cfg}
        catalog = read_catalog(read_table(cfg.indicator_file, character_only=True), maps, audit) if cfg.indicator_file else None
        if cfg.association_file:
            if catalog is None:
                raise ValueError("--association-file requires a complete --indicator-file")
            a = load_associations(read_table(cfg.association_file, character_only=True), catalog, cfg, maps, audit)
        else:
            missing = paths["inventory"].loc[~paths["inventory"].exists & paths["inventory"].role.str.startswith(("omics:", "system:"))]
            if len(missing):
                checkpoint("blocked_missing_participant_data")
                raise FileNotFoundError(f"Missing {len(missing)} participant input files. See input_file_inventory.csv; run on the server with --systems-dir and --omics-dir.")
            a, catalog, inventory = run_new_associations(members, catalog, cfg, paths, audit)
            write_csv(inventory, cfg.output_dir / "indicator_inventory.csv")
        # Imported input is not overwritten; a normalized copy keeps supplied q values unchanged.
        assoc_output = cfg.output_dir / ("imported_feature_indicator_associations.csv.gz" if cfg.association_file else "feature_indicator_associations.csv.gz")
        if not cfg.association_file or assoc_output.resolve() != cfg.association_file.resolve():
            write_csv(a, assoc_output)
        write_csv(catalog, cfg.output_dir / "indicator_catalog.csv")
        write_csv(a.groupby(["omics", "system", "test_status"], dropna=False).size().reset_index(name="n_records"), cfg.output_dir / "association_status_counts.csv")
        summary = summarize_associations(members, a, catalog, cfg.q_threshold, cfg.top_n)
        for key, filename in {"wide": "all_indicators_two_crests.csv", "long": "all_indicators_by_crest.csv",
                              "top": "top10_plot_data.csv", "crest1_omics": "crest1_by_omics.csv"}.items():
            write_csv(summary[key], cfg.output_dir / filename)
        (cfg.output_dir / "figure_caption.txt").write_text(figure_caption(cfg, bool(cfg.association_file)) + "\n", encoding="utf-8")
        manifest.update(n_indicators=len(catalog), n_tests_attempted=len(a), n_tests_valid=int((a.test_valid & np.isfinite(a.q_association)).sum()),
                        n_plot_indicators=len(summary["top"][["system", "indicator_id"]].drop_duplicates()))
        manifest["figure_groups"] = {group: list(systems) for group, systems in FIGURE_GROUPS.items()}
        manifest["figure_files"] = {group: figure_paths(cfg, group) for group in FIGURE_GROUPS}
        checkpoint("analysis_complete_render_pending")
        manifest["pdf_audit_status"] = export_grouped_figures(summary["top"], catalog, cfg)
        checkpoint("complete")
        print(f"Completed. Tables and QA: {cfg.output_dir.resolve()}; figures: {cfg.figure_dir.resolve()}", flush=True)
        return {"members": members, "associations": a, "catalog": catalog, "summary": summary, "cfg": cfg}
    except Exception as error:
        manifest["error"] = str(error)
        checkpoint("blocked_missing_participant_data" if manifest["status"] == "blocked_missing_participant_data" else "failed")
        raise
    finally:
        save_audits(audit, cfg)


if __name__ == "__main__":
    main()
