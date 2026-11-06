#!/usr/bin/env python
"""Manuscript numbers for Results section 5 (AgeAccel, disease and phenotypes).

Re-derives every number quoted in

    \\subsection{Aging-clock acceleration shows disease-specific discriminative
                 value beyond age and sex}

from the completed outputs of the Result-05 scripts and compares each value
with ``03-manuscript/sn-article.tex``. Nothing is re-fitted; the script only
reads tables.

Sources (relative to --outputs-dir, default ``outputs/`` beside this script)
--------------------------------------------------------------------------
disease_or_results.csv                  disease_odds_ratio.py
panel_c_dose_response_effects.csv       dose_response_forest.py
  (falls back to the older name dose_response_effects.csv)

Usage
-----
    python manuscript_stats.py
    python manuscript_stats.py --outputs-dir /path/to/Result-05/outputs

Writes ``<outputs-dir>/manuscript_stats/manuscript_stats.{txt,csv,json}``;
exit code 1 if any check FAILs.
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
SECTION_TITLE = "Aging-clock acceleration shows disease-specific discriminative value beyond age and sex"
STACK = "4-omics stack"
CARDIOMETABOLIC = {"Hypercholesterolaemia", "Hypertension", "Hyperglycaemia", "Obesity", "Fatty liver"}


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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--outputs-dir", type=Path, default=SCRIPT_DIR / "outputs")
    p.add_argument("--tex", type=Path, default=PROJECT_DIR / "03-manuscript" / "sn-article.tex")
    p.add_argument("--report-dir", type=Path, default=None)
    return p.parse_args(argv)


def disease_checks(R, f: Path):
    if not f.exists():
        R.missing("or_n_tests", "condition-score tests", "40", f"missing {f}")
        return
    d = pd.read_csv(f)
    p = d[d["cohort"] == "four_omics_common"] if "cohort" in d else d
    sig = p[p["q_value"] < 0.05]
    R.add("or_n_participants", "participants (same for all scores)", "1,848", int(p["n"].max()), source=f.name,
          note="" if p["n"].nunique() == 1 else f"n varies: {sorted(p['n'].unique())}")
    R.add("or_cases_min", "cases per condition (min)", "80", int(p["n_cases"].min()), source=f.name)
    R.add("or_cases_max", "cases per condition (max)", "405", int(p["n_cases"].max()), source=f.name)
    R.add("or_n_tests", "condition-score tests (BH family)", "40", len(p), source=f.name)
    R.add("or_n_conditions", "conditions", "8", p["outcome"].nunique(), source=f.name, tex=False)
    R.add("or_n_sig", "tests passing BH correction", "12", len(sig), source=f.name)
    R.add("or_all_sig_above1", "all significant ORs > 1", "1", int((sig["odds_ratio"] > 1).all()), tex=False)
    R.add("or_all_sig_cardiometabolic", "all significant tests cardiometabolic", "1",
          int(sig["outcome"].isin(CARDIOMETABOLIC).all()), tex=False,
          note=f"conditions: {sorted(sig['outcome'].unique())}")

    def row(outcome, score):
        x = p[(p["outcome"] == outcome) & (p["score"] == score)]
        return x.iloc[0] if len(x) else None

    def or_block(key, outcome, score, orr, lo, hi, q):
        r = row(outcome, score)
        g = (lambda c: None if r is None else r[c])
        R.add(f"{key}_or", f"{score} / {outcome}: OR per 1 s.d.", orr, g("odds_ratio"), source=f.name)
        if lo:
            R.add(f"{key}_lo", f"{score} / {outcome}: 95% CI low", lo, g("or_ci_low"), source=f.name)
            R.add(f"{key}_hi", f"{score} / {outcome}: 95% CI high", hi, g("or_ci_high"), source=f.name)
        if q:
            R.add(f"{key}_q", f"{score} / {outcome}: q", q, g("q_value"), source=f.name)

    or_block("stack_obesity", "Obesity", STACK, "1.68", "1.38", "2.06", "5.68\\times10^{-6}")
    r = row("Obesity", STACK)
    R.add("stack_sd_years", "1 s.d. of four-omics AgeAccel in years", "3.6",
          None if r is None else r["score_sd_years"], source=f.name)
    or_block("stack_fattyliver", "Fatty liver", STACK, "1.46", "1.19", "1.78", "0.00129")
    or_block("stack_hyperglyc", "Hyperglycaemia", STACK, "1.42", "1.19", "1.69", "4.72\\times10^{-4}")
    or_block("stack_hypertension", "Hypertension", STACK, "1.25", "1.07", "1.47", "0.0238")
    or_block("stack_hyperchol", "Hypercholesterolaemia", STACK, "1.33", "1.18", "1.49", "1.47\\times10^{-5}")
    stack_sig = sig[sig["score"] == STACK]["outcome"].nunique()
    R.add("stack_n_sig", "conditions associated with four-omics AgeAccel", "5", stack_sig, tex=False,
          source=f.name)

    lip = sig[sig["score"] == "Lipidomics"]
    R.add("lipid_n_sig", "conditions associated with lipidomic AgeAccel (text: same five)", "5", len(lip),
          tex=False, source=f.name, note="" if set(lip["outcome"]) == set(sig[sig["score"] == STACK]["outcome"])
          else f"lipid {sorted(lip['outcome'])} vs stack {sorted(sig[sig['score'] == STACK]['outcome'])}")
    R.add("lipid_or_min", "lipidomic ORs (min)", "1.23", lip["odds_ratio"].min() if len(lip) else None, source=f.name)
    R.add("lipid_or_max", "lipidomic ORs (max)", "1.56", lip["odds_ratio"].max() if len(lip) else None, source=f.name)
    hc = p[p["outcome"] == "Hypercholesterolaemia"]
    R.add("lipid_strongest_hyperchol", "lipidomics is strongest correlate of hypercholesterolaemia", "1",
          int(hc.loc[hc["odds_ratio"].idxmax(), "score"] == "Lipidomics") if len(hc) else None, tex=False)
    or_block("lipid_hyperchol", "Hypercholesterolaemia", "Lipidomics", "1.51", "1.34", "1.70", "4.01\\times10^{-10}")

    rna = sig[sig["score"] == "Transcriptomics"]
    R.add("rna_n_sig", "conditions associated with transcriptomic AgeAccel", "1", len(rna), tex=False,
          note=str(sorted(rna["outcome"])))
    or_block("rna_obesity", "Obesity", "Transcriptomics", "1.46", "1.19", "1.78", "0.00129")
    oral = sig[sig["score"] == "Oral Microbiome"]
    R.add("oral_n_sig", "conditions associated with oral AgeAccel", "1", len(oral), tex=False,
          note=str(sorted(oral["outcome"])))
    or_block("oral_hyperglyc", "Hyperglycaemia", "Oral Microbiome", "1.23", "1.05", "1.44", "0.0384")
    gut = p[p["score"] == "Gut Microbiome"]
    R.add("gut_min_q", "gut AgeAccel, smallest q (text: all q >= 0.182)", "0.182", gut["q_value"].min(),
          op=">=", source=f.name)
    other = p[p["outcome"].isin(["Hypothyroidism", "Cholelithiasis", "Asthma"])]
    R.add("noncardio_min_q", "hypothyroidism / cholelithiasis / asthma, smallest q (text: >= 0.056)", "0.056",
          other["q_value"].min(), op=">=", source=f.name,
          note=f"exact min q = {other['q_value'].min():.4f}" if len(other) else "")
    R.add("max_or", "largest OR among significant tests (text: OR <= 1.68)", "1.68", sig["odds_ratio"].max(),
          op="<=", source=f.name, note=f"exact {sig['odds_ratio'].max():.4f}")
    for r in sig.sort_values(["outcome", "score"]).itertuples():
        R.log(f"OR sig: {r.outcome:<22} {r.score:<16} {r.odds_ratio:.2f} [{r.or_ci_low:.2f}, {r.or_ci_high:.2f}] "
              f"q {r.q_value:.3g}")


def phenotype_checks(R, out: Path):
    f = next((p for p in [out / "panel_c_dose_response_effects.csv", out / "dose_response_effects.csv"]
              if p.exists()), None)
    if f is None:
        R.missing("pheno_n_sig", "AgeAccel-phenotype associations", "107", "missing panel_c_dose_response_effects.csv")
        return
    e = pd.read_csv(f)
    qcol = "q_value_global"
    s = e[e[qcol] < 0.05]
    R.add("pheno_n_tested", "phenotypes tested", "871", len(e), source=f.name)
    R.add("pheno_n_systems", "systems", "12", e["system"].nunique(), source=f.name)
    R.add("pheno_n_min", "complete-case n (min)", "1,162", int(e["n"].min()), source=f.name)
    R.add("pheno_n_max", "complete-case n (max)", "1,863", int(e["n"].max()), source=f.name)
    R.add("pheno_n_sig", "associations passing global BH", "107", len(s), source=f.name)
    R.add("pheno_pct_sig", "% of phenotypes", "12.3", 100 * len(s) / len(e), source=f.name)
    R.add("pheno_n_sys_sig", "systems with associations", "8", s["system"].nunique(), source=f.name, tex=False,
          note=str(s.groupby("system").size().to_dict()))
    bc = s[s["system"] == "body_comp"]
    R.add("pheno_bodycomp_n", "body-composition associations", "74", len(bc), source=f.name)
    R.add("pheno_bodycomp_all_pos", "all body-composition associations positive", "1",
          int((bc["effect"] > 0).all()), tex=False)

    def feat(key, pattern, beta, lo, hi, q, label, system=None):
        x = s if system is None else s[s["system"] == system]
        x = x[x["feature"].str.fullmatch(pattern, case=False)]
        r = x.iloc[0] if len(x) == 1 else None
        note = "" if len(x) == 1 else f"{len(x)} features matched '{pattern}'"
        g = (lambda c: None if r is None else r[c])
        R.add(f"{key}_beta", f"{label}: standardized beta", beta, g("effect"), source=f.name, note=note)
        R.add(f"{key}_lo", f"{label}: 95% CI low", lo, g("ci_low"), source=f.name)
        R.add(f"{key}_hi", f"{label}: 95% CI high", hi, g("ci_high"), source=f.name)
        R.add(f"{key}_q", f"{label}: q", q, g(qcol), source=f.name)

    feat("arm_fat", r"body_comp_arm_right_fat_mass", "0.134", "0.090", "0.178", "6.48\\times10^{-7}",
         "right-arm fat mass")
    feat("vat_area", r"total_scan_vat_area", "0.117", "0.076", "0.158", "3.99\\times10^{-6}", "VAT area")
    feat("waist", r"waist_circumference", "0.104", "0.062", "0.146", "4.81\\times10^{-5}", "waist circumference")
    feat("r_wave_I", r"r_mv_I", "0.151", "0.098", "0.205", "3.99\\times10^{-6}", "ECG R-wave amplitude lead I")
    feat("heart_rate", r"hr_bpm", "0.105", "0.050", "0.160", "0.00227", "heart rate")
    feat("standing_dbp", r"standing_one_min_blood_pressure_diastolic", "0.102", "0.048", "0.155", "0.00258",
         "standing diastolic BP")
    feat("sat70", r"saturation_below_70", "0.110", "0.052", "0.167", "0.00227", "saturation below 70")
    feat("min_sat", r"saturation_min_value", "-0.083", "-0.138", "-0.028", "0.0291", "minimum O2 saturation")
    feat("lymph", r".*lymphocytes_abs.*", "0.100", "0.048", "0.152", "0.00209", "absolute lymphocytes")
    feat("mono", r".*monocytes_abs.*", "0.073", "0.022", "0.123", "0.0438", "absolute monocytes")
    feat("fibre_carb", r"fiber_carb_ratio", "0.172", "0.127", "0.217", "1.47\\times10^{-10}", "fibre-to-carb ratio")
    top = s.sort_values(qcol).iloc[0] if len(s) else None
    R.add("fibre_is_strongest", "strongest single association is fibre-to-carbohydrate ratio", "1",
          None if top is None else int(top["feature"] == "fiber_carb_ratio"), tex=False,
          note="" if top is None else f"top = {top['feature']}")
    neg = e[e["system"].isin(["cgm", "liver_health", "frailty", "renal_function"])]
    R.add("glyc_liver_frail_renal_min_q", "glycaemic/liver/frailty/renal, smallest q", "0.260",
          neg[qcol].min() if len(neg) else None, source=f.name)
    R.log("phenotype associations by system: " + str(s.groupby("system").size().to_dict()))


def main(argv=None) -> int:
    a = parse_args(argv)
    R = Report("Result 5: AgeAccel, disease and phenotypes", a.tex, SECTION_TITLE)
    disease_checks(R, a.outputs_dir / "disease_or_results.csv")
    phenotype_checks(R, a.outputs_dir)
    return R.write(a.report_dir or a.outputs_dir / "manuscript_stats")


if __name__ == "__main__":
    raise SystemExit(main())
