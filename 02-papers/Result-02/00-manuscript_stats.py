#!/usr/bin/env python
"""Manuscript numbers for Results section 2 (DE-SWAN multi-omics waves).

Re-derives every number quoted in

    \\subsection{Mapping age-associated multi-omics waves reveals distinct
                 functional signatures}

from the completed outputs of the Result-02 scripts and compares each value
with ``03-manuscript/sn-article.tex``. Nothing is re-fitted; the only
computation beyond table look-ups is the panel-d kernel smoothing, repeated
exactly as in ``04-plot_oral_pathway_panel_d.py`` to check the trajectory
statements.

Sources (relative to --outputs-dir, default ``outputs/`` beside this script)
--------------------------------------------------------------------------
deswan-window_7.5/<omics>_{figure_statistics,main_peaks,breakpoints}_window_7p5.csv
                                                  01-de_swan_omics_one.R
transcriptomics_pathway/kegg_enrichment/all_pathway_enrichment_results.csv
transcriptomics_pathway/similarity_network/network_summary_by_crest.csv
                                                  02-transcriptomics_pathway_*.R
lipid_pathway/{crest_lipid_selection_summary,lion_custom_statistic_summary,
               lion_age_48_two_sided_result}.csv  03_* scripts / LION web
microbiome_pathway_enrichment/age48_oral_microbiome_window_7p5_*.csv
                                                  04-oral_microbiome_pathway.ipynb
05-crest_system_association/{crest_feature_counts,indicator_catalog,
               top10_plot_data,all_indicators_by_crest}.csv
                                                  05-crest_system_association.py

Usage
-----
    python manuscript_stats.py
    python manuscript_stats.py --outputs-dir /path/to/Result-02/outputs

Writes ``<outputs-dir>/manuscript_stats/manuscript_stats.{txt,csv,json}``;
exit code 1 if any check FAILs. Phenotype groups (e.g. visceral adipose
tissue) are matched by regular expressions on indicator ids/labels; the
matched indicators are listed in the report so the mapping can be audited.
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
SECTION_TITLE = "Mapping age-associated multi-omics waves reveals distinct functional signatures"


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


OMICS = ["transcriptomics", "metabolomics", "lipidomics", "gut_microbiome", "oral_microbiome"]
TAG = "window_7p5"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--outputs-dir", type=Path, default=SCRIPT_DIR / "outputs")
    p.add_argument("--deswan-dir", type=Path, default=None,
                   help="Default: <outputs-dir>/deswan-window_7.5")
    p.add_argument("--crest-system-dir", type=Path, default=None,
                   help="Default: <outputs-dir>/05-crest_system_association")
    p.add_argument("--oral-prefix", default="age48_oral_microbiome_window_7p5")
    p.add_argument("--tex", type=Path, default=PROJECT_DIR / "03-manuscript" / "sn-article.tex")
    p.add_argument("--report-dir", type=Path, default=None)
    return p.parse_args(argv)


def pct(x):
    return None if x is None or not np.isfinite(x) else 100.0 * float(x)


def match(df: pd.DataFrame, pattern: str, exclude: str | None = None) -> pd.DataFrame:
    text = (df["indicator_id"].astype(str) + " " + df["indicator_label"].astype(str))
    keep = text.str.contains(pattern, case=False, regex=True)
    if exclude:
        keep &= ~text.str.contains(exclude, case=False, regex=True)
    return df[keep]


def deswan_checks(R: "Report", ddir: Path):
    fs, peaks, bps = {}, {}, {}
    for om in OMICS:
        f1 = ddir / f"{om}_figure_statistics_{TAG}.csv"
        f2 = ddir / f"{om}_main_peaks_{TAG}.csv"
        f3 = ddir / f"{om}_breakpoints_{TAG}.csv"
        if f1.exists():
            fs[om] = pd.read_csv(f1).iloc[0]
        if f2.exists():
            peaks[om] = pd.read_csv(f2).sort_values("peak_age")
        if f3.exists():
            bps[om] = pd.read_csv(f3)
    if len(fs) == len(OMICS):
        n_any = int(sum(int(fs[o]["n_features_significant_any_window"]) for o in OMICS))
        R.add("deswan_features_any_transition",
              "features with >=1 significant local transition (sum over layers)",
              "3,632", n_any, source=f"{ddir.name}/*_figure_statistics_{TAG}.csv")
        for o in OMICS:
            R.log(f"DE-SWAN {o:<16} significant in any window: "
                  f"{int(fs[o]['n_features_significant_any_window']):>5} of "
                  f"{int(fs[o]['n_features_tested_any_window']):>6} tested")
        if bps:
            b = pd.concat(bps.values(), ignore_index=True)
            g = b.groupby(["omics", "feature"])["n_sig_wins"].max()
            R.log("  alternative definitions (features with >=k significant windows): "
                  + ", ".join(f"k={k}: {(g >= k).sum():,}" for k in (1, 2, 3)))
    else:
        R.missing("deswan_features_any_transition", "features with >=1 transition", "3,632",
                  f"missing figure_statistics for {sorted(set(OMICS) - set(fs))}")

    expected = {"metabolomics": [48], "lipidomics": [48], "oral_microbiome": [48],
                "gut_microbiome": [50], "transcriptomics": [52, 64]}
    for om, ages in expected.items():
        if om not in peaks:
            for a in ages:
                R.missing(f"peak_age_{om}_{a}", f"{om} main peak", str(a), "main_peaks missing")
            continue
        got = peaks[om]["peak_age"].tolist()
        R.log(f"main peaks {om:<16}: " + ", ".join(
            f"{int(r.peak_age)} y (n={int(r.peak_n)})" for r in peaks[om].itertuples()))
        for i, a in enumerate(ages):
            val = got[i] if i < len(got) else None
            R.add(f"peak_age_{om}_{a}", f"{om} main crest age (#{i + 1})", str(a), val,
                  source=f"{om}_main_peaks_{TAG}.csv",
                  note="" if len(got) == len(ages) else f"{len(got)} main peaks found: {got}")
    if "transcriptomics" in peaks:
        t = peaks["transcriptomics"].set_index("peak_age")["peak_n"]
        R.add("rna_n_52", "transcripts significant at 52 y", "508", t.get(52), source="main_peaks")
        R.add("rna_n_64", "transcripts significant at 64 y (crest 2)", "2,802", t.get(64),
              source="main_peaks")
    if all(o in peaks for o in OMICS):
        first = {o: peaks[o].iloc[0] for o in OMICS}
        n1 = int(sum(int(first[o]["peak_n"]) for o in OMICS))
        R.add("crest1_n_from_peaks", "crest 1 = features at first main crest of each layer",
              "853", n1, source="main_peaks (first crest per layer)")
    return peaks, bps


def enrichment_checks(R, base: Path):
    f = base / "kegg_enrichment" / "all_pathway_enrichment_results.csv"
    if not f.exists():
        for k, rep in [("terms_crest52", "232"), ("terms_crest64", "625")]:
            R.missing(k, "enriched terms (GO BP + KEGG + Reactome, FDR<0.05)", rep, f"missing {f}")
        return
    e = pd.read_csv(f)
    sig = e[e["p.adjust"] < 0.05]
    by = sig.groupby("peak_age").size()
    by_db = sig.groupby(["peak_age", "database"]).size()
    R.log("enriched terms by crest x database: " + "; ".join(
        f"{int(a)} y {db}: {n}" for (a, db), n in by_db.items()))
    R.add("terms_crest52", "enriched GO BP/KEGG/Reactome terms at 52 y (FDR<0.05)", "232",
          by.get(52, 0), source=f.name)
    R.add("terms_crest64", "enriched GO BP/KEGG/Reactome terms at 64 y (FDR<0.05)", "625",
          by.get(64, 0), source=f.name)

    def term(age, desc, db=None):
        d = e[(e["peak_age"] == age) & (e["Description"].str.lower() == desc.lower())]
        if db:
            d = d[d["database"].str.contains(db, case=False)]
        return (float(d["p.adjust"].min()) if len(d) else None), len(d)

    for key, age, desc, db, rep, lab in [
        ("fdr_immune_activation_52", 52, "activation of immune response", "GO",
         "3.08\\times10^{-3}", "immune-response activation, 52 y"),
        ("fdr_tnf_production_52", 52, "tumor necrosis factor production", "GO",
         "8.74\\times10^{-6}", "TNF production, 52 y"),
        ("fdr_neutrophil_degranulation_64", 64, "Neutrophil degranulation", "Reactome",
         "6.96\\times10^{-18}", "neutrophil degranulation, 64 y"),
        ("fdr_response_to_virus_64", 64, "response to virus", "GO",
         "4.76\\times10^{-10}", "response to virus, 64 y"),
    ]:
        val, n = term(age, desc, db)
        R.add(key, f"FDR {lab}", rep, val, source=f.name,
              note="" if n else f"term '{desc}' not found")
    # show which related terms carry the quoted TNF FDR, if any
    tnf = e[e["Description"].str.contains("tumor necrosis factor", case=False)
            & (e["p.adjust"] < 0.05)]
    for _, r in tnf.iterrows():
        R.log(f"  TNF-related term at {int(r['peak_age'])} y: {r['Description']} "
              f"({r['database']}) FDR {r['p.adjust']:.3g}")

    ns = base / "similarity_network" / "network_summary_by_crest.csv"
    if ns.exists():
        s = pd.read_csv(ns).set_index("peak_age")
        for age in (52, 64):
            col = "top15_display_communities"
            R.add(f"modules_{age}", f"displayed network modules at {age} y", "3",
                  s.loc[age, col] if age in s.index and col in s else None, source=ns.name,
                  tex=False)
            if age in s.index and "significant_terms" in s:
                R.log(f"network {age} y: significant_terms={s.loc[age, 'significant_terms']}, "
                      f"final_modules={s.loc[age, 'final_modules']}")
    else:
        R.log(f"network summary not found: {ns}")


def lipid_checks(R, base: Path):
    sel = base / "crest_lipid_selection_summary.csv"
    if sel.exists():
        s = pd.read_csv(sel)
        s = s[s["peak_age"] == 48].iloc[0]
        R.add("lipid48_n_sig", "significant lipids at 48 y", "75", s["n_selected_lipids"], source=sel.name)
        R.add("lipid48_n_older_high", "of which higher in older parcel", "63", s["n_older_high"],
              source=sel.name)
    else:
        R.missing("lipid48_n_older_high", "lipids higher in older parcel", "63", f"missing {sel}")
    cs = base / "lion_custom_statistic_summary.csv"
    if cs.exists():
        c = pd.read_csv(cs)
        c = c[c["requested_peak_age"] == 48]
        R.add("lion48_n_mapped", "LION-mapped lipids ranked at 48 y", "169",
              c["exported_lipids"].iloc[0] if len(c) else None, source=cs.name)
    lr = base / "lion_age_48_two_sided_result.csv"
    if not lr.exists():
        R.missing("lion48_n_terms", "LION terms FDR<0.05", "18", f"missing {lr}")
        return
    d = pd.read_csv(lr, encoding="utf-8-sig")
    qcol = [c for c in d.columns if "FDR" in c][0]
    dcol = [c for c in d.columns if c.lower().startswith(("discription", "description"))][0]
    R.add("lion48_n_terms", "LION terms at 48 y (FDR<0.05)", "18", int((d[qcol] < 0.05).sum()),
          source=lr.name)

    def q(name):
        x = d[d[dcol].str.strip().str.lower() == name.lower()]
        return (float(x[qcol].iloc[0]), str(x["Regulated"].iloc[0])) if len(x) else (None, "")

    for key, name, rep, direction in [
        ("lion_glycerolipids", "glycerolipids [GL]", "2.63\\times10^{-3}", "UP"),
        ("lion_lipid_droplet", "lipid droplet", "5.76\\times10^{-3}", "UP"),
        ("lion_lipid_storage", "lipid storage", "5.76\\times10^{-3}", "UP"),
        ("lion_tag", "triacylglycerols [GL0301]", "6.18\\times10^{-3}", "UP"),
        ("lion_lysoGPL", "lysoglycerophospholipids", "6.53\\times10^{-4}", "DOWN"),
        ("lion_sfa", "saturated fatty acid", "6.53\\times10^{-4}", "DOWN"),
        ("lion_gt18C", "fatty acid with more than 18 carbons", "6.53\\times10^{-4}", "DOWN"),
    ]:
        val, reg = q(name)
        R.add(key, f"LION FDR: {name} ({direction})", rep, val, source=lr.name,
              note="" if reg in ("", direction) else f"direction is {reg}, text says {direction}")


def oral_checks(R, base: Path, prefix: str, peaks, bps):
    if "oral_microbiome" in peaks and "oral_microbiome" in bps:
        n_sig = int(peaks["oral_microbiome"].query("peak_age == 48")["peak_n"].sum())
        b = bps["oral_microbiome"]
        fg = b[b["break_age"] == 48]["feature"].drop_duplicates()
        R.add("oral48_n_sig", "significant oral taxa at 48-y crest", "23", n_sig, source="main_peaks")
        R.add("oral48_n_breakpoint48", "oral taxa with consolidated breakpoint at 48 y (text: six)", "6",
              len(fg), source="oral_microbiome_breakpoints", tex=False)
        species = fg.str.split("s__").str[-1].tolist()
        R.log(f"oral 48-y breakpoint taxa: {species}")
        for sp in ["Streptococcus_gordonii", "Lautropia_mirabilis", "Haemophilus_sputorum"]:
            R.add(f"oral48_has_{sp}", f"{sp.replace('_', ' ')} among 48-y breakpoint taxa", "1",
                  int(sp in species), tex=False)
    f = base / f"{prefix}_eligible_pathway_age_heatmap_sample_level_rpk.csv"
    if not f.exists():
        R.missing("oral_n_pathways", "displayed MetaCyc pathways (panel d)", "14", f"missing {f}")
        return
    d = pd.read_csv(f)
    meta = {"participant_id", "age", "age_bin_index", "_age_bin", "research_stage"}
    paths = [c for c in d.columns if c not in meta]
    x = d[paths].apply(pd.to_numeric, errors="raise")
    age = pd.to_numeric(d["age"], errors="coerce")
    keep = age.between(40, 70)
    a, lx = age[keep].to_numpy(float), np.log1p(x[keep].to_numpy(float))
    grid = np.arange(41, 70, dtype=float)
    w = np.exp(-0.5 * ((a[None, :] - grid[:, None]) / 2.0) ** 2)
    sm = pd.DataFrame(((w @ lx) / w.sum(axis=1, keepdims=True)).T, index=paths, columns=grid)
    z = sm.sub(sm.mean(axis=1), axis=0).div(sm.std(axis=1, ddof=0).replace(0, np.nan), axis=0)
    z = z.drop(index=[p for p in z.index if p.upper() == "UNINTEGRATED"], errors="ignore")
    n = len(z)
    below = int((z.loc[:, 41.0:47.0] < 0).all(axis=1).sum())
    peak_age = z.idxmax(axis=1)
    late = int(peak_age.between(63, 69).sum())
    R.add("oral_n_pathways", "displayed MetaCyc pathways (panel d)", "14", n, source=f.name)
    R.add("oral_below_mean_41_47", "pathways below grid mean at every age 41-47", "14", below,
          source=f.name, note="recomputed: log1p RPK, Gaussian bw 2 y, row z over 41-69")
    R.add("oral_peak_63_69_majority", "pathways peaking at 63-69 y ('mostly' = >half)",
          str(n // 2), late, op=">", source=f.name, tex=False)
    R.log(f"oral pathways peak ages: {peak_age.astype(int).value_counts().sort_index().to_dict()}; "
          f"samples {int(keep.sum()):,}, participants {d.loc[keep, 'participant_id'].nunique():,}")


def crest_system_checks(R, base: Path):
    cnt = base / "crest_feature_counts.csv"
    if cnt.exists():
        c = pd.read_csv(cnt)
        tot = c[c["omics"] == "ALL"].set_index("crest")["n_features_total"]
        R.add("crest1_n", "crest 1 features", "853", tot.get("Crest1"), source=cnt.name)
        R.add("crest2_n", "crest 2 transcripts", "2,802", tot.get("Crest2"), source=cnt.name)
    top = base / "top10_plot_data.csv"
    allc = base / "all_indicators_by_crest.csv"
    cat = base / "indicator_catalog.csv"
    if not (top.exists() and allc.exists()):
        R.missing("phenotypes_n", "phenotypes tested", "871",
                  f"missing association results in {base} (run 05 on the server)")
        return
    t = pd.read_csv(top)
    t = t[t["selected_top10"].astype(str).str.lower().isin(["true", "1"])]
    al = pd.read_csv(allc)
    if cat.exists():
        k = pd.read_csv(cat)
        R.add("phenotypes_n", "phenotypes in indicator catalogue", "871", len(k), source=cat.name)
        R.add("systems_n", "physiological systems", "12", k["system"].nunique(), source=cat.name)
    tested = al[al["n_tested"] > 0]
    R.log(f"phenotypes with >=1 valid test: {tested[['system', 'indicator_id']].drop_duplicates().shape[0]}")
    sys_sig = al.groupby("system")["n_significant"].sum()
    R.add("systems_with_assoc", "systems with >=1 significant association", "12",
          int((sys_sig > 0).sum()), tex=False, source=allc.name)

    w = t.pivot_table(index=["system", "indicator_id", "indicator_label"], columns="crest",
                      values="proportion").reset_index()
    means = w.groupby("system")[["Crest1", "Crest2"]].mean()
    rev = sorted(means.index[means["Crest2"] >= means["Crest1"]])
    R.log("mean displayed proportion by system (crest1 / crest2): " + "; ".join(
        f"{s} {r.Crest1:.1%}/{r.Crest2:.1%}" for s, r in means.iterrows()))
    R.add("systems_crest2_ge_crest1", "systems where crest 2 >= crest 1 (text: immune only)", "1",
          len(rev), source=top.name, note=f"systems: {rev}", tex=False)
    R.add("immune_is_exception", "the exception is the immune system", "1",
          int(rev == ["immune"]), tex=False)

    def rng(df, crest, key, lo_rep, hi_rep, label):
        v = df[crest].dropna() * 100
        R.add(f"{key}_min", f"{label} (min %)", lo_rep, v.min() if len(v) else None, source=top.name)
        R.add(f"{key}_max", f"{label} (max %)", hi_rep, v.max() if len(v) else None, source=top.name)
        R.log(f"{label}: " + "; ".join(f"{r.indicator_label} {r[crest]:.1%}" for _, r in df.iterrows()))

    bc = w[w["system"] == "body_comp"]
    rng(bc, "Crest1", "bodycomp_crest1_top10", "35", "38", "body composition top-10, crest 1")
    vat = match(bc, r"visceral|\bvat\b")
    rng(vat, "Crest2", "bodycomp_crest2_vat", "27", "28", "visceral adipose (area/mass/volume), crest 2")
    trunk = match(bc, r"trunk|android", exclude=r"visceral|\bvat\b")
    rng(trunk, "Crest2", "bodycomp_crest2_trunk_android", "12", "14",
        "regional trunk / android fat, crest 2")

    cm = w[w["system"].isin(["cgm", "cardio", "bone_density"])]
    rng(cm, "Crest1", "cardiometabolic_crest1", "13", "21",
        "insulin resistance + cardiovascular + bone density, crest 1")
    R.add("cardiometabolic_crest2_max", "same systems, crest 2 max (text: fewer than 8%)", "8",
          pct(cm["Crest2"].max()), op="<", source=top.name)
    R.add("diet_crest1_max", "diet, crest 1 max (%)", "31",
          pct(w.loc[w.system == "diet", "Crest1"].max()), source=top.name)
    R.add("sleep_crest1_max", "sleep, crest 1 max (%)", "22",
          pct(w.loc[w.system == "sleep", "Crest1"].max()), source=top.name)

    im = w[w["system"].isin(["immune", "hema"])]
    mono_n = match(im, r"mono", exclude=r"%|pct|percent|perc|ratio")
    mono_p = match(im, r"mono.*(%|pct|percent|perc)|(%|pct|percent|perc).*mono")
    for df, key, rep2, rep1, lab in [(mono_n, "monocyte_count", "43", "29", "absolute monocyte count"),
                                     (mono_p, "monocyte_pct", "27", "23", "monocyte percentage")]:
        if len(df) == 1:
            R.add(f"{key}_crest2", f"{lab}, crest 2 (%)", rep2, pct(df["Crest2"].iloc[0]), source=top.name)
            R.add(f"{key}_crest1", f"{lab}, crest 1 (%)", rep1, pct(df["Crest1"].iloc[0]), source=top.name)
        else:
            R.missing(f"{key}_crest2", f"{lab}, crest 2", rep2,
                      f"{len(df)} indicators matched: {df['indicator_label'].tolist()}")
    nl = match(im, r"neutro|lympho")
    v = pd.concat([nl["Crest1"], nl["Crest2"]]).dropna() * 100
    R.add("neut_lymph_min", "neutrophil/lymphocyte indices, both crests (min %)", "23",
          v.min() if len(v) else None, source=top.name)
    R.add("neut_lymph_max", "neutrophil/lymphocyte indices, both crests (max %)", "28",
          v.max() if len(v) else None, source=top.name)
    R.log("immune displayed: " + "; ".join(
        f"{r.indicator_label} c1 {r.Crest1:.1%} c2 {r.Crest2:.1%}" for _, r in im.iterrows()))


def main(argv=None) -> int:
    a = parse_args(argv)
    R = Report("Result 2: DE-SWAN multi-omics waves", a.tex, SECTION_TITLE)
    out = a.outputs_dir
    peaks, bps = deswan_checks(R, a.deswan_dir or out / "deswan-window_7.5")
    enrichment_checks(R, out / "transcriptomics_pathway")
    lipid_checks(R, out / "lipid_pathway")
    oral_checks(R, out / "microbiome_pathway_enrichment", a.oral_prefix, peaks, bps)
    crest_system_checks(R, a.crest_system_dir or out / "05-crest_system_association")
    return R.write(a.report_dir or out / "manuscript_stats")


if __name__ == "__main__":
    raise SystemExit(main())
