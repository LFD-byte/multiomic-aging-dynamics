#!/usr/bin/env python
"""Manuscript numbers for Results section 1 (System-wide characterization).

Re-derives every number quoted in

    \\subsection{System-wide characterization in a large cohort}

from the data files and completed analysis outputs, and compares each one with
the value written in ``03-manuscript/sn-article.tex``. Nothing is re-fitted;
the script only reads files.

Sources
-------
Samples / features measured (raw HPP tables, before any feature filtering):
    <hpp-datasets>/serum_lipidomics/serum_lipidomics.parquet
    <hpp-datasets>/nightingale_metabolomics/nightingale_metabolomics.parquet
    <hpp-datasets>/gut_microbiome/gut_microbiome.parquet               (samples)
    <hpp-datasets>/gut_microbiome/metaphlan4/aggregated/metaphlan_abundance_species.parquet
    <hpp-datasets>/oral_microbiome/oral_microbiome.parquet             (samples)
    <hpp-datasets>/oral_microbiome/aggregated/metaphlan_abundances_aggregated/species.parquet
    <preprocessing>/interim_data/rna_seq/024-rna_seq.parquet           (RNA counts merged with sample table)
  When a raw HPP table is not available, the row count of the matching
  ``01-preprocessing/processed_data`` file is used for samples (no rows are
  dropped there); feature counts are never taken from filtered files.
Participants / sex (age 40-70 analysis tables):
    ../01-interim_data/omics_data/<omics>_with_metadata.parquet (or .csv)
PLS age association:
    outputs/multi_omics_pls_age_leakage_free_cv_summary.csv (02-omics_pca_leakage_free_cv.py)
Age-trend heatmap (informational):
    outputs/[panel_e_]age_trend_row_info_dys.csv (03-age_trend.ipynb)

Usage
-----
    python 00-manuscript_stats.py
    python 00-manuscript_stats.py --hpp-datasets-dir /path/to/hpp_datasets --outputs-dir /path/to/outputs

Writes ``outputs/manuscript_stats/manuscript_stats.{txt,csv,json}`` and exits
with code 1 if any check FAILs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PAPERS_DIR = SCRIPT_DIR.parent
PROJECT_DIR = PAPERS_DIR.parent
SECTION_TITLE = "System-wide characterization in a large cohort"


# =============================================================================
# Generic claim-checking helpers (identical block in every Result script)
# =============================================================================
@dataclass
class Claim:
    key: str
    description: str
    reported: str                 # value exactly as written in the .tex
    computed: float | None = None
    op: str = "eq"                # eq | <= | < | >= | > | info (shown, never FAILs)
    unit: str = ""
    source: str = ""
    note: str = ""
    status: str = ""
    computed_fmt: str = ""
    in_tex: bool | None = None


def parse_reported(text: str) -> tuple[float, str, int]:
    """Parse a LaTeX-formatted number -> (value, style, decimals)."""
    s = (text.replace("$", "").replace("{", "").replace("}", "").replace(",", "")
         .replace("\\,", "").replace("−", "-").replace("~", "").replace(" ", "")
         .replace("\\%", "").replace("%", ""))
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)(?:\\times|×|x)10\^(-?\d+)", s) or \
        re.fullmatch(r"(-?\d+(?:\.\d+)?)[eE](-?\d+)", s)
    if m:
        mant = m.group(1)
        dec = len(mant.split(".")[1]) if "." in mant else 0
        return float(mant) * 10 ** int(m.group(2)), "sci", dec
    m = re.fullmatch(r"-?\d+(?:\.\d+)?", s)
    if not m:
        raise ValueError(f"cannot parse reported value {text!r}")
    dec = len(s.split(".")[1]) if "." in s else 0
    return float(s), "fix", dec


def fmt_like(x: float, style: str, dec: int) -> str:
    if style == "sci":
        return f"{x:.{dec}e}"
    return f"{x:,.{dec}f}"


def evaluate(c: Claim) -> Claim:
    value, style, dec = parse_reported(c.reported)
    if c.computed is None or (isinstance(c.computed, float) and math.isnan(c.computed)):
        c.status, c.computed_fmt = "MISSING", "NA"
        return c
    x = float(c.computed)
    c.computed_fmt = fmt_like(x, style, dec)
    if c.op == "info":
        c.status = "INFO"
        return c
    if c.op == "eq":
        ok = fmt_like(x, style, dec) == fmt_like(value, style, dec)
    elif c.op in ("<=", ">="):
        # inclusive bounds are judged at the precision written in the text
        xr = float(fmt_like(x, style, dec).replace(",", ""))
        ok = xr <= value if c.op == "<=" else xr >= value
    elif c.op == "<":
        ok = x < value
    elif c.op == ">":
        ok = x > value
    else:
        raise ValueError(c.op)
    c.status = "PASS" if ok else "FAIL"
    return c


def _norm_tex(s: str) -> str:
    return re.sub(r"[\s$~{}]|\\,", "", s)


def section_text(tex_path: Path, title: str) -> str | None:
    if not tex_path.exists():
        return None
    tex = tex_path.read_text(encoding="utf-8", errors="ignore")
    start = tex.find(title)
    if start < 0:
        return None
    nxt = re.search(r"\\(sub)?section\{", tex[start + len(title):])
    end = start + len(title) + nxt.start() if nxt else len(tex)
    body = tex[start:end]
    return "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("%"))


class Report:
    def __init__(self, title: str, tex_path: Path, section: str):
        self.title = title
        self.claims: list[Claim] = []
        self.info: list[str] = []
        self.sec = section_text(tex_path, section)
        self.tex_path = tex_path

    def add(self, key, description, reported, computed, op="eq", unit="", source="", note="",
            tex=True):
        if isinstance(computed, (np.generic,)):
            computed = computed.item()
        c = evaluate(Claim(key, description, reported, computed, op, unit,
                           str(source), note))
        if self.sec is not None and tex:
            c.in_tex = _norm_tex(c.reported) in _norm_tex(self.sec)
        self.claims.append(c)
        return c

    def missing(self, key, description, reported, why, source=""):
        return self.add(key, description, reported, None, source=source, note=why)

    def log(self, line: str = ""):
        self.info.append(line)

    def write(self, out_dir: Path) -> int:
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = [asdict(c) for c in self.claims]
        pd.DataFrame(rows).to_csv(out_dir / "manuscript_stats.csv", index=False)
        counts = pd.Series([c.status for c in self.claims]).value_counts().to_dict()
        (out_dir / "manuscript_stats.json").write_text(json.dumps(
            {"title": self.title, "tex": str(self.tex_path), "status_counts": counts,
             "claims": rows, "info": self.info}, indent=2, default=str), encoding="utf-8")
        w = max([len(c.key) for c in self.claims] + [10])
        lines = [f"== {self.title} ==",
                 f"manuscript: {self.tex_path} "
                 f"({'section found' if self.sec else 'SECTION NOT FOUND'})",
                 f"status: {counts}", "",
                 f"{'status':<8}{'key':<{w + 2}}{'reported':>16} {'op':>3} {'computed':>16}  tex  description"]
        for c in self.claims:
            tex_flag = {True: "ok", False: "--", None: "?"}[c.in_tex]
            lines.append(f"{c.status:<8}{c.key:<{w + 2}}{c.reported:>16} {c.op:>3} "
                         f"{c.computed_fmt:>16}  {tex_flag:<4} {c.description}"
                         + (f"  [{c.note}]" if c.note else ""))
        bad = [c for c in self.claims if c.status not in ("PASS", "INFO")]
        if bad:
            lines += ["", "Needs attention:"]
            for c in bad:
                lines.append(f"  [{c.status}] {c.key}: manuscript {c.reported}, data "
                             f"{c.computed_fmt} ({c.source}) {c.note}")
        not_in_tex = [c for c in self.claims if c.in_tex is False]
        if not_in_tex:
            lines += ["", "Reported string not found in the manuscript section "
                          "(text edited? update the script or the manuscript):"]
            lines += [f"  {c.key}: {c.reported}" for c in not_in_tex]
        if self.info:
            lines += ["", "Additional information:"] + ["  " + l for l in self.info]
        txt = "\n".join(lines)
        (out_dir / "manuscript_stats.txt").write_text(txt + "\n", encoding="utf-8")
        print(txt)
        print(f"\nWritten to {out_dir}")
        return 1 if counts.get("FAIL", 0) else 0


def first_existing(*paths: Path) -> Path | None:
    for p in paths:
        if p is not None and Path(p).exists():
            return Path(p)
    return None
# =============================================================================


META_COLS = {"participant_id", "cohort", "research_stage", "array_index", "batch",
             "collection_date", "collection_timestamp", "timezone", "sample_name",
             "age", "sex", "bmi", "year_of_birth", "month_of_birth",
             "collection_datetime", "data_source", "hmo", "__index_level_0__"}
OMICS = ["transcriptomics", "metabolomics", "lipidomics", "gut_microbiome", "oral_microbiome"]


def normalize_sex(s: pd.Series) -> pd.Series:
    """Return 1 = male, 0 = female, whatever the stored type ("1", 1.0, "M", "male")."""
    num = pd.to_numeric(s, errors="coerce")
    txt = s.astype(str).str.strip().str.lower()
    num = num.where(num.notna(), txt.map({"m": 1, "male": 1, "f": 0, "female": 0}))
    return num


def parquet_shape(path: Path) -> tuple[int, list[str]]:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(path))
    return int(pf.metadata.num_rows), list(pf.schema_arrow.names)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--outputs-dir", type=Path, default=SCRIPT_DIR / "outputs")
    p.add_argument("--interim-dir", type=Path, default=PAPERS_DIR / "01-interim_data")
    p.add_argument("--preprocessing-dir", type=Path, default=PROJECT_DIR / "01-preprocessing")
    p.add_argument("--hpp-datasets-dir", type=Path, default="/home/ec2-user/studies/hpp_datasets")
    p.add_argument("--tex", type=Path, default=PROJECT_DIR / "03-manuscript" / "sn-article.tex")
    p.add_argument("--report-dir", type=Path, default=None,
                   help="Default: <outputs-dir>/manuscript_stats")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    R = Report("Result 1: system-wide characterization", a.tex, SECTION_TITLE)
    hpp, pre = a.hpp_datasets_dir, a.preprocessing_dir
    if not hpp.exists():
        # local layout: <project_phenoai>/hpp_datasets; server layout: ~/studies/hpp_datasets
        for cand in [PROJECT_DIR.parent / "hpp_datasets", PROJECT_DIR.parents[2] / "hpp_datasets"]:
            if cand.exists():
                hpp = cand
                break
    R.log(f"hpp_datasets dir: {hpp} ({'found' if hpp.exists() else 'NOT FOUND - pass --hpp-datasets-dir'})")
    R.log(f"01-preprocessing dir: {pre} ({'found' if pre.exists() else 'NOT FOUND - pass --preprocessing-dir'})")
    proc = pre / "processed_data"

    # ---------------------------------------------------------------- samples
    sample_src = {
        "transcriptomics": [pre / "interim_data" / "rna_seq" / "024-rna_seq.parquet"],
        "metabolomics": [hpp / "nightingale_metabolomics" / "nightingale_metabolomics.parquet",
                         proc / "029-nightingale_metabolomics_imputed.parquet"],
        "lipidomics": [hpp / "serum_lipidomics" / "serum_lipidomics.parquet",
                       proc / "008-serum_lipidomics_imputed.parquet"],
        "gut_microbiome": [hpp / "gut_microbiome" / "gut_microbiome.parquet",
                           proc / "013-gut_microbiome_imputed.parquet"],
        "oral_microbiome": [hpp / "oral_microbiome" / "oral_microbiome.parquet"],
    }
    feature_src = {
        "transcriptomics": pre / "interim_data" / "rna_seq" / "024-rna_seq.parquet",
        "metabolomics": first_existing(
            hpp / "nightingale_metabolomics" / "nightingale_metabolomics.parquet",
            proc / "029-nightingale_metabolomics_imputed.parquet"),  # no feature filter in 029
        "lipidomics": hpp / "serum_lipidomics" / "serum_lipidomics.parquet",
        "gut_microbiome": hpp / "gut_microbiome" / "metaphlan4" / "aggregated"
                          / "metaphlan_abundance_species.parquet",
        "oral_microbiome": hpp / "oral_microbiome" / "aggregated"
                           / "metaphlan_abundances_aggregated" / "species.parquet",
    }
    n_samp: dict[str, int | None] = {}
    for om, cands in sample_src.items():
        path = first_existing(*cands)
        if path is None:
            n_samp[om] = None
            R.log(f"samples[{om}]: no source found ({', '.join(map(str, cands))})")
            continue
        n_samp[om] = parquet_shape(path)[0]
        R.log(f"samples[{om}] = {n_samp[om]:,}  <- {path}")
    n_feat: dict[str, int | None] = {}
    for om, path in feature_src.items():
        if path is None or not Path(path).exists():
            n_feat[om] = None
            R.log(f"features[{om}]: raw table not found ({path})")
            continue
        _, cols = parquet_shape(path)
        if "microbiome" in om:
            feats = [c for c in cols if c.startswith("k__")]
        else:
            feats = [c for c in cols if c not in META_COLS]
        n_feat[om] = len(feats)
        R.log(f"features[{om}] = {len(feats):,} of {len(cols):,} columns  <- {path}")

    # oral species table has more rows than the oral sample table
    sp = feature_src["oral_microbiome"]
    if sp is not None and Path(sp).exists():
        R.log(f"note: oral species table rows = {parquet_shape(sp)[0]:,} "
              f"(oral sample table = {n_samp.get('oral_microbiome')})")

    def tot(d, keys):
        vals = [d.get(k) for k in keys]
        return None if any(v is None for v in vals) else int(sum(vals))

    blood = tot(n_samp, ["transcriptomics", "metabolomics", "lipidomics"])
    R.add("samples_total", "biological samples obtained", "31,433", tot(n_samp, OMICS),
          source="raw sample tables")
    R.add("samples_blood", "blood samples (RNA + metabolomics + lipidomics)", "11,923", blood,
          source="raw sample tables")
    R.add("samples_stool", "stool samples", "10,694", n_samp["gut_microbiome"],
          source=str(first_existing(*sample_src["gut_microbiome"])))
    R.add("samples_oral", "oral swab samples", "8,816", n_samp["oral_microbiome"],
          source=str(first_existing(*sample_src["oral_microbiome"])),
          note="" if n_samp["oral_microbiome"] else "needs hpp_datasets/oral_microbiome")
    R.add("features_total", "molecular and microbial features measured", "52,874",
          tot(n_feat, OMICS), source="raw feature tables")
    for om, rep, lab in [("transcriptomics", "49,614", "transcripts"),
                         ("metabolomics", "250", "metabolites"),
                         ("lipidomics", "187", "lipids"),
                         ("gut_microbiome", "2,087", "gut microbial taxa"),
                         ("oral_microbiome", "736", "oral microbial taxa")]:
        R.add(f"features_{om}", lab, rep, n_feat[om], source=str(feature_src[om]))
    meas = None
    if all(n_samp.get(o) is not None and n_feat.get(o) is not None for o in OMICS):
        meas = int(sum(n_samp[o] * n_feat[o] for o in OMICS))
    R.add("measurements_total", "individual measurements (sum samples x features)",
          "226,772,193", meas, source="samples x features")

    # ------------------------------------------------- participants and sex
    ids, visits = [], []
    om_dir = a.interim_dir / "omics_data"
    for om in OMICS:
        pq_path, csv_path = om_dir / f"{om}_with_metadata.parquet", om_dir / f"{om}_with_metadata.csv"
        cols = ["participant_id", "research_stage", "sex", "age"]
        try:
            if pq_path.exists():
                d = pd.read_parquet(pq_path, columns=cols)
            elif csv_path.exists():
                d = pd.read_csv(csv_path, usecols=cols)
            else:
                R.log(f"participants: missing {pq_path}")
                continue
        except ImportError:
            d = pd.read_csv(csv_path, usecols=cols)
        d["participant_id"] = d["participant_id"].astype(str)
        d["sex"] = normalize_sex(d["sex"])
        d["omics"] = om
        visits.append(d)
    if visits:
        v = pd.concat(visits, ignore_index=True)
        vis = v.drop_duplicates(["participant_id", "research_stage"])
        n_part = v["participant_id"].nunique()
        # The sentence describes the whole HPP cohort (>10,000 aged 40-70), not
        # only participants with omics, so this is shown for context only.
        R.add("participants_gt_10000", "unique omics participants aged 40-70 "
              "(manuscript: >10,000 in the whole cohort)", "10,000", n_part, op="info",
              source=str(om_dir))
        # 01-cohort_feature.ipynb: one row per (participant, visit) across omics
        if vis["sex"].isna().all():
            R.log(f"sex column could not be interpreted; raw values: {visits[0]['sex'].unique()[:5]}")
        male_pct = 100 * (vis["sex"] == 1).mean() if vis["sex"].notna().any() else None
        R.add("male_pct_visits", "% male (participant-visits, as in Fig. 1b)", "47",
              male_pct, unit="%", source=str(om_dir))
        pp = v.drop_duplicates("participant_id")
        R.log(f"participants = {n_part:,}; participant-visits = {len(vis):,}; "
              f"% male by participant = {100 * (pp['sex'] == 1).mean():.1f}%; "
              f"age range {v['age'].min():.2f}-{v['age'].max():.2f}")
        for om, g in v.groupby("omics"):
            R.log(f"  {om}: rows {len(g):,}, participants {g['participant_id'].nunique():,}")
    else:
        R.missing("participants_gt_10000", "unique participants aged 40-70", "10,000",
                  "no *_with_metadata tables found", source=str(om_dir))

    # ------------------------------------------------------------------ PLS
    pls = first_existing(a.outputs_dir / "multi_omics_pls_age_leakage_free_cv_summary.csv")
    if pls is None:
        R.missing("pls_max_p", "PLS permutation P (all layers)", "0.001",
                  "missing multi_omics_pls_age_leakage_free_cv_summary.csv")
    else:
        s = pd.read_csv(pls)
        R.add("pls_max_p", "max permutation P across 5 layers (all P <= 0.001)", "0.001",
              s["p_permutation_two_sided"].max(), op="<=", source=pls.name)
        R.add("pls_max_q", "max BH q across 5 layers (significance retained)", "0.05",
              s["q_fdr_bh"].max(), op="<", source=pls.name,
              note="manuscript states significance retained after BH", tex=False)
        top = s.loc[s["rho_cv"].idxmax()]
        R.add("pls_gut_rho", "gut microbiome PLS Spearman rho (largest)", "0.30",
              float(s.loc[s["omics"].str.lower().str.startswith("gut"), "rho_cv"].iloc[0]),
              source=pls.name,
              note="" if str(top["omics"]).lower().startswith("gut")
              else f"largest is {top['omics']}!")
        R.add("pls_n_layers", "omics layers with significant age association", "5",
              int((s["q_fdr_bh"] < 0.05).sum()), source=pls.name, tex=False)
        for r in s.itertuples():
            R.log(f"PLS {r.omics:<16} n={r.n_participants:>5} rho={r.rho_cv:.3f} "
                  f"[{r.rho_cv_ci_low:.3f}, {r.rho_cv_ci_high:.3f}] "
                  f"P={r.p_permutation_two_sided:.3g} q={r.q_fdr_bh:.3g}")

    # ---------------------------------------------------- heatmap (info only)
    hm = first_existing(a.outputs_dir / "age_trend_row_info_dys.csv",
                        a.outputs_dir / "panel_e_age_trend_row_info_dys.csv")
    if hm is not None:
        h = pd.read_csv(hm)
        col = "omics" if "omics" in h.columns else h.columns[-1]
        R.log(f"Fig. 1e heatmap features: {len(h):,} "
              f"({h[col].value_counts().to_dict()})  <- {hm.name}")

    return R.write(a.report_dir or a.outputs_dir / "manuscript_stats")


if __name__ == "__main__":
    raise SystemExit(main())
