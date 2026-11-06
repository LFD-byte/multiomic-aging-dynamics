#!/usr/bin/env python
"""Manuscript numbers for Results section 4 (cross-omics aging clock).

Re-derives every number quoted in

    \\subsection{A cross-omics clock links molecular transition programmes to
                 accelerated aging}

from the completed outputs of the Result-04 scripts and compares each value
with ``03-manuscript/sn-article.tex``. Nothing is re-fitted; participant-level
clock statistics are recomputed from the saved repeat-averaged predictions with
the same formulas as ``02-plot_fig6b_four_omics_clock.py``.

Sources (relative to --outputs-dir, default ``outputs/`` beside this script)
--------------------------------------------------------------------------
nonlinearity_model_metrics.csv, nonlinearity_bootstrap_improvement.csv
                                        01-nonlinearity_model_comparison.py
panel_a_clock_predictions.parquet (or sensitivity_clock_performance_source.csv),
meta_repeat_metrics.csv, diagnostics.json
                                        02-aging_clock.py
sensitivity_inference.json              01-aging_clock_sensitivity.py

Usage
-----
    python manuscript_stats.py
    python manuscript_stats.py --outputs-dir /path/to/Result-04/outputs

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
SECTION_TITLE = "A cross-omics clock links molecular transition programmes to accelerated aging"
CLOCK = "Multi-Omics (4-omics stacking)"


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


def benchmark_checks(R, out: Path):
    f = out / "nonlinearity_model_metrics.csv"
    if not f.exists():
        R.missing("rna_en_mae", "transcriptomics Elastic Net MAE", "5.16", f"missing {f}")
        return
    m = pd.read_csv(f).set_index(["omics", "model"])

    def g(om, model, col):
        try:
            return float(m.loc[(om, model), col])
        except KeyError:
            return None

    for key, om, model, col, rep, lab in [
        ("rna_en_mae", "Transcriptomics", "Elastic Net", "MAE", "5.16", "transcriptomics Elastic Net MAE (y)"),
        ("rna_en_r2", "Transcriptomics", "Elastic Net", "R2", "0.306", "transcriptomics Elastic Net R2"),
        ("rna_lgbm_mae", "Transcriptomics", "LightGBM", "MAE", "5.36", "transcriptomics LightGBM MAE"),
        ("rna_lgbm_r2", "Transcriptomics", "LightGBM", "R2", "0.264", "transcriptomics LightGBM R2"),
        ("rna_gam_mae", "Transcriptomics", "Spline GAM", "MAE", "7.35", "transcriptomics spline GAM MAE"),
        ("rna_gam_r2", "Transcriptomics", "Spline GAM", "R2", "-0.647", "transcriptomics spline GAM R2"),
        ("met_en_mae", "Metabolomics", "Elastic Net", "MAE", "5.52", "metabolomics Elastic Net MAE"),
        ("met_en_r2", "Metabolomics", "Elastic Net", "R2", "0.238", "metabolomics Elastic Net R2"),
        ("gut_en_r2", "Gut Microbiome", "Elastic Net", "R2", "0.099", "gut Elastic Net R2"),
        ("gut_nl_r2", "Gut Microbiome", "Nonlinear representative", "R2", "0.153", "gut nonlinear R2"),
        ("oral_en_r2", "Oral Microbiome", "Elastic Net", "R2", "0.042", "oral Elastic Net R2"),
        ("oral_nl_r2", "Oral Microbiome", "Nonlinear representative", "R2", "0.076", "oral nonlinear R2"),
    ]:
        R.add(key, lab, rep, g(om, model, col), source=f.name)
    best = m.reset_index().loc[lambda d: d["model"] != "Nonlinear representative"].sort_values("MAE").iloc[0]
    R.add("best_single_is_rna_en", "most accurate single-omics clock is transcriptomics EN", "1",
          int(best["omics"] == "Transcriptomics" and best["model"] == "Elastic Net"), tex=False,
          note=f"best = {best['omics']} {best['model']} (MAE {best['MAE']:.2f})")
    for om in ["Transcriptomics", "Metabolomics", "Lipidomics", "Gut Microbiome", "Oral Microbiome"]:
        R.log(f"{om:<16}: " + ", ".join(
            f"{mod} MAE {m.loc[(om, mod), 'MAE']:.2f} R2 {m.loc[(om, mod), 'R2']:.3f}"
            for mod in ["Elastic Net", "LightGBM", "Spline GAM", "Nonlinear representative"]
            if (om, mod) in m.index))

    b = out / "nonlinearity_bootstrap_improvement.csv"
    if not b.exists():
        R.missing("lipid_mae_change", "lipidomics MAE change (%)", "0.1", f"missing {b}")
        return
    bi = pd.read_csv(b).set_index(["omics", "contrast"])

    def imp(om, col):
        try:
            return float(bi.loc[(om, "MAE improvement (%)"), col])
        except KeyError:
            return None

    for om, key, v, lo, hi in [("Lipidomics", "lipid", "0.1", "-0.7", "0.9"),
                               ("Gut Microbiome", "gut", "4.8", "4.2", "5.4"),
                               ("Oral Microbiome", "oral", "3.9", "3.3", "4.5")]:
        R.add(f"{key}_mae_change", f"{om}: MAE reduction by nonlinear model (%)", v, imp(om, "value"), source=b.name)
        R.add(f"{key}_mae_change_lo", f"{om}: 95% CI low (%)", lo, imp(om, "ci_low"), source=b.name)
        R.add(f"{key}_mae_change_hi", f"{om}: 95% CI high (%)", hi, imp(om, "ci_high"), source=b.name)


def load_participant_predictions(out: Path):
    pq = out / "panel_a_clock_predictions.parquet"
    if pq.exists():
        try:
            d = pd.read_parquet(pq)
            if "clock_name" in d:
                d = d[d["clock_name"] == CLOCK]
            d = d.groupby(d["participant_id"].astype(str))[["true_age", "pred_age"]].mean().reset_index()
            return d, pq.name
        except ImportError:
            pass
    csv = out / "sensitivity_clock_performance_source.csv"
    if csv.exists():
        return pd.read_csv(csv), csv.name
    return None, ""


def clock_checks(R, out: Path):
    d, src = load_participant_predictions(out)
    if d is None:
        R.missing("clock4_mae", "four-omics clock MAE", "4.41", "no participant-level predictions found")
    else:
        x, y = d["true_age"].to_numpy(float), d["pred_age"].to_numpy(float)
        R.add("clock4_n", "participants with all four layers", "1,865", len(d), source=src)
        R.add("clock4_mae", "four-omics clock MAE (participant means)", "4.41", np.mean(np.abs(y - x)), source=src)
        r2 = 1 - np.sum((y - x) ** 2) / np.sum((x - x.mean()) ** 2)
        R.add("clock4_r2_pct", "variance explained (%)", "45.6", 100 * r2, source=src)
        R.add("clock4_pearson", "Pearson r", "0.676", np.corrcoef(x, y)[0, 1], source=src)
        R.add("clock4_spearman", "Spearman rho", "0.686",
              pd.Series(x).rank().corr(pd.Series(y).rank()), source=src)
        R.add("clock4_slope", "slope of predicted on chronological age", "0.44", np.polyfit(x, y, 1)[0],
              source=src)
    rm = out / "meta_repeat_metrics.csv"
    if rm.exists():
        r = pd.read_csv(rm)
        R.add("clock4_repeat_mae_mean", "MAE across repeats, mean", "4.49", r["mae"].mean(), source=rm.name)
        R.add("clock4_repeat_mae_sd", "MAE across repeats, s.d.", "0.02", r["mae"].std(ddof=1), source=rm.name,
              note="s.d. with ddof=1")
        R.log(f"repeat MAE sd (ddof=0) = {r['mae'].std(ddof=0):.4f}; n repeats = {len(r)}")
    m = out / "nonlinearity_model_metrics.csv"
    if m.exists() and d is not None:
        mm = pd.read_csv(m)
        best_single = mm.loc[mm["model"] != "Nonlinear representative", "MAE"].min()
        R.add("clock4_beats_single", "four-omics MAE lower than any single-omics MAE", "1",
              int(np.mean(np.abs(d["pred_age"] - d["true_age"])) < best_single), tex=False,
              note=f"best single-omics MAE {best_single:.2f}")


def sensitivity_checks(R, out: Path):
    dg = out / "diagnostics.json"
    if dg.exists():
        j = json.loads(dg.read_text(encoding="utf-8"))
        n5, n4 = j.get("old_five_omics_intersection"), j.get("observed_four_omics_intersection")
        R.add("n_five_layers", "participants with all five layers", "271", n5, source=dg.name)
        if n5 and n4:
            R.add("cohort_fold_gain", "four- vs five-layer cohort ratio (text: almost sevenfold)", "7",
                  n4 / n5, op="<", source=dg.name, tex=False, note=f"{n4}/{n5} = {n4 / n5:.2f}")
    si = out / "sensitivity_inference.json"
    if not si.exists():
        R.missing("dmae_5v4", "Delta MAE five vs four layers", "-0.11", f"missing {si}")
        return
    j = json.loads(si.read_text(encoding="utf-8"))
    mae, r2 = j["mae_delta_5_minus_4"], j["r2_delta_5_minus_4"]
    R.add("sens_n", "participants in five-layer comparison", "271", j.get("n"), source=si.name)
    R.add("dmae_5v4", "Delta MAE (5 - 4 layers, y)", "-0.11", mae["estimate"], source=si.name)
    R.add("dmae_5v4_lo", "Delta MAE 95% CI low", "-0.28", mae["bootstrap_95_ci"][0], source=si.name)
    R.add("dmae_5v4_hi", "Delta MAE 95% CI high", "0.08", mae["bootstrap_95_ci"][1], source=si.name)
    R.add("dmae_5v4_p", "Delta MAE paired permutation P", "0.26", mae["paired_swap_permutation_p"], source=si.name)
    R.add("dr2_5v4", "Delta R2 (5 - 4 layers)", "0.039", r2["estimate"], source=si.name)
    R.add("dr2_5v4_lo", "Delta R2 95% CI low", "-0.002", r2["bootstrap_95_ci"][0], source=si.name)
    R.add("dr2_5v4_hi", "Delta R2 95% CI high", "0.075", r2["bootstrap_95_ci"][1], source=si.name)
    R.add("dr2_5v4_p", "Delta R2 paired permutation P", "0.043", r2["paired_swap_permutation_p"], source=si.name)


def main(argv=None) -> int:
    a = parse_args(argv)
    R = Report("Result 4: cross-omics aging clock", a.tex, SECTION_TITLE)
    benchmark_checks(R, a.outputs_dir)
    clock_checks(R, a.outputs_dir)
    sensitivity_checks(R, a.outputs_dir)
    return R.write(a.report_dir or a.outputs_dir / "manuscript_stats")


if __name__ == "__main__":
    raise SystemExit(main())
