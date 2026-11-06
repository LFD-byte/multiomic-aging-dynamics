#!/usr/bin/env python
"""Manuscript numbers for Results section 3 (multi-omics trajectories).

Re-derives every number quoted in

    \\subsection{Multi-omics trajectories track divergent physiological
                 variation across adulthood}

from the completed outputs of the Result-03 scripts and compares each value
with ``03-manuscript/sn-article.tex``. Nothing is re-fitted; the script only
reads tables.

Sources (relative to --outputs-dir, default ``outputs/`` beside this script)
--------------------------------------------------------------------------
molecule_clustering/{silhouette_scores,cluster_omics_composition,
                     molecule_cluster_assignments}_agglomerative.csv   01-molecule_clustering.py
transcriptome_cluster_pathway_landscape/{cluster_analysis_summary,
                     all_databases_cluster_vs_rest_ora}.csv            02-transcriptome_cluster_pathway_landscape.R
lion_web_five_patterns/{lion_input_summary.csv, results/lion_p0*_ranked_result.csv}
                                                                       03-* scripts / LION web
oral_metacyc_age_association/*  and
oral_metacyc_age_cluster_attribution/pathway_pattern_classification.csv
                                                                       04-oral_metacyc_* scripts
crest_cluster_intersections/{molecule_matching_diagnostics,
                     cluster_crest_overlap_heatmap}.csv                05-crest_cluster_intersections.py
cluster_system_association/cluster_system_associations.csv             cluster_system_association.py

Cluster numbers in the manuscript are the 1-based ``pattern_id`` (1 late
accelerating rise, 2 gradual rise, 3 accelerating decline, 4 midlife trough,
5 midlife peak), not the 0-based ``cluster_id`` of the clustering output.

Usage
-----
    python manuscript_stats.py
    python manuscript_stats.py --outputs-dir /path/to/Result-03/outputs

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
SECTION_TITLE = "Multi-omics trajectories track divergent physiological variation across adulthood"


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


def clustering_checks(R, d: Path):
    sil = d / "silhouette_scores_agglomerative.csv"
    if sil.exists():
        s = pd.read_csv(sil)
        R.add("best_k", "k maximizing mean silhouette", "5", int(s.loc[s["silhouette_score"].idxmax(), "k"]),
              source=sil.name, tex=False)
    comp = d / "cluster_omics_composition_agglomerative.csv"
    if not comp.exists():
        R.missing("c1_pct_transcripts", "% transcripts in cluster 1", "99", f"missing {comp}")
        return
    c = pd.read_csv(comp)
    c["omics_l"] = c["omics"].str.lower()

    def prop(pid, om):
        x = c[(c["pattern_id"] == pid) & c["omics_l"].str.startswith(om)]
        return 100 * float(x["proportion"].iloc[0]) if len(x) else 0.0

    def n(pid, om):
        x = c[(c["pattern_id"] == pid) & c["omics_l"].str.startswith(om)]
        return int(x["n"].iloc[0]) if len(x) else 0

    R.add("c1_pct_transcripts", "% transcripts in cluster 1", "99", prop(1, "transcript"), source=comp.name)
    R.add("c5_pct_transcripts", "% transcripts in cluster 5", "57", prop(5, "transcript"), source=comp.name)
    R.add("c5_pct_gut", "% gut taxa in cluster 5", "34", prop(5, "gut"), source=comp.name)
    R.add("c2_n_metabolites", "metabolites in cluster 2", "130", n(2, "metabol"), source=comp.name)
    R.add("n_metabolites_clustered", "clustered metabolites", "141",
          int(c.loc[c["omics_l"].str.startswith("metabol"), "n"].sum()), source=comp.name)
    R.add("c2_n_lipids", "lipids in cluster 2", "66", n(2, "lipid"), source=comp.name)
    R.add("n_lipids_clustered", "clustered lipids", "93",
          int(c.loc[c["omics_l"].str.startswith("lipid"), "n"].sum()), source=comp.name)
    sizes = c.groupby("pattern_id")["n"].sum()
    R.add("n_clustered", "clustered features", "5,241", int(sizes.sum()), source=comp.name)
    R.add("c45_pct_features", "clusters 4+5 share of clustered features (%)", "17.8",
          100 * sizes.loc[[4, 5]].sum() / sizes.sum(), source=comp.name)
    R.log("cluster sizes: " + ", ".join(f"C{k} {v:,}" for k, v in sizes.items()))
    return sizes


def transcriptome_checks(R, d: Path):
    cs = d / "cluster_analysis_summary.csv"
    if cs.exists():
        s = pd.read_csv(cs)
        tot = s.groupby("group_id")["fdr_significant_enriched_terms"].sum()
        R.log("enriched terms per cluster (all databases): " + ", ".join(f"C{k} {v}" for k, v in tot.items()))
        R.add("ora_terms_c1_4_min", "enriched terms, clusters 1-4 (min)", "93", tot.loc[[1, 2, 3, 4]].min(),
              source=cs.name)
        R.add("ora_terms_c1_4_max", "enriched terms, clusters 1-4 (max)", "174", tot.loc[[1, 2, 3, 4]].max(),
              source=cs.name)
        R.add("ora_terms_c5", "enriched terms, cluster 5 (text: none)", "0", tot.get(5, 0), source=cs.name,
              tex=False)
    else:
        R.missing("ora_terms_c1_4_min", "enriched terms clusters 1-4", "93", f"missing {cs}")
    ora = d / "all_databases_cluster_vs_rest_ora.csv"
    if not ora.exists():
        return
    o = pd.read_csv(ora)

    def term(gid, name):
        x = o[(o["group_id"] == gid) & (o["term"].str.lower() == name.lower())]
        return x.sort_values("fdr").iloc[0] if len(x) else None

    t = term(3, "tRNA processing in the mitochondrion")
    R.add("c3_mt_trna_fold", "C3 mitochondrial tRNA processing, fold enrichment", "8.55",
          None if t is None else t["fold_enrichment"], source=ora.name)
    R.add("c3_mt_trna_fdr", "C3 mitochondrial tRNA processing, FDR", "7.68\\times10^{-23}",
          None if t is None else t["fdr"], source=ora.name)
    t = term(1, "Neutrophil degranulation")
    R.add("c1_neutrophil_fdr", "C1 neutrophil degranulation, FDR", "3.11\\times10^{-9}",
          None if t is None else t["fdr"], source=ora.name)
    t = term(4, "chromosome organization")
    R.add("c4_chromosome_org_fdr", "C4 chromosome organization, FDR", "1.33\\times10^{-4}",
          None if t is None else t["fdr"], source=ora.name)
    for gid, pat, lab in [(3, r"rRNA processing", "C3 rRNA processing"),
                          (3, r"^translation$", "C3 translation"),
                          (1, r"Class I MHC|MHC class I", "C1 class I MHC antigen presentation"),
                          (2, r"cell adhesion|mononuclear cell migration", "C2 adhesion / migration")]:
        x = o[(o["group_id"] == gid) & o["term"].str.contains(pat, case=False) & o["significant"].astype(bool)]
        R.log(f"{lab}: {len(x)} significant terms; best FDR "
              f"{x['fdr'].min():.3g}" if len(x) else f"{lab}: no significant term")


def lion_checks(R, d: Path):
    s = d / "lion_input_summary.csv"
    if s.exists():
        x = pd.read_csv(s)
        R.add("lion_n_lipids", "lipids in LION ranking (all patterns)", "90",
              int(x["ranked_background_lipids"].iloc[0]), source=s.name)
    res = {}
    for f in sorted((d / "results").glob("lion_p0*_ranked_result.csv")):
        pid = int(re.search(r"lion_p0(\d)", f.name).group(1))
        t = pd.read_csv(f, encoding="utf-8-sig")
        t = t.rename(columns={[c for c in t.columns if "FDR" in c][0]: "q",
                              [c for c in t.columns if c.lower().startswith(("discription", "description"))][0]: "term"})
        res[pid] = t
    if not res:
        R.missing("lion_c2_storage_fdr", "C2 storage-lipid terms FDR", "4.42\\times10^{-4}",
                  f"no ranked results in {d / 'results'}")
        return
    for pid, t in res.items():
        R.log(f"LION C{pid}: {int((t['q'] < 0.05).sum())} terms FDR<0.05")
    c2 = res.get(2)
    if c2 is not None:
        st = c2[c2["term"].str.contains(r"lipid storage|lipid droplet|triacylglycerol|glycerolipids \[GL\]",
                                        case=False, regex=True)]
        R.add("lion_c2_storage_fdr_max", "C2 storage-lipid terms, largest FDR (text: all 4.42e-4)",
              "4.42\\times10^{-4}", st["q"].max() if len(st) else None, source="lion_p02")
        R.add("lion_c2_storage_fdr_min", "C2 storage-lipid terms, smallest FDR", "4.42\\times10^{-4}",
              st["q"].min() if len(st) else None, source="lion_p02")
        R.log("C2 storage terms: " + "; ".join(f"{r.term} {r.q:.3g}" for r in st.itertuples()))
    mem = []
    for pid in (3, 5):
        if pid in res:
            t = res[pid]
            m = t[t["term"].str.contains(r"headgroup|curvature", case=False) & (t["q"] < 0.05)]
            mem.append(m.assign(cluster=pid))
    if mem:
        m = pd.concat(mem)
        R.add("lion_c35_charge_curv_fdr_max", "C3/C5 headgroup-charge & curvature terms, max FDR (text: <=1.34e-2)",
              "1.34\\times10^{-2}", m["q"].max() if len(m) else None, op="<=", source="lion_p03/p05")
        R.log("C3/C5 charge/curvature: " + "; ".join(f"C{r.cluster} {r.term} {r.q:.3g}" for r in m.itertuples()))
        gp = pd.concat([res[p].assign(cluster=p) for p in (3, 5) if p in res], ignore_index=True)
        gp = gp[gp["term"].str.contains(r"glycerophospho|membrane component|headgroup|curvature|endoplasmic",
                                        case=False) & (gp["q"] < 0.05)]
        R.log(f"  max FDR over C3/C5 membrane-glycerophospholipid terms: {gp['q'].max():.4g} "
              f"({gp.loc[gp['q'].idxmax(), 'term']}, C{gp.loc[gp['q'].idxmax(), 'cluster']})")


def oral_checks(R, age_dir: Path, attr_dir: Path):
    allp = age_dir / "metacyc_pathway_age_associations_all.csv"
    fig = age_dir / "figure_pathways.csv"
    if allp.exists() and fig.exists():
        a = pd.read_csv(allp)
        f = pd.read_csv(fig)
        R.add("metacyc_M", "MetaCyc pathways tested", "284", len(a), source=allp.name)
        R.add("metacyc_N", "participants in pathway age models", "8,105", int(a["n"].max()), source=allp.name,
              note="" if a["n"].nunique() == 1 else f"n varies: {sorted(a['n'].unique())[:5]}")
        R.add("metacyc_K", "pathways Bonferroni P<0.05", "80", int((a["p_bonferroni_global"] < 0.05).sum()),
              source=allp.name)
        R.add("metacyc_displayed", "displayed non-redundant pathways", "12", len(f), source=fig.name)
        R.add("metacyc_displayed_up", "displayed pathways increasing with age", "12",
              int((f["beta_pathway_sd_per_10y"] > 0).sum()), source=fig.name)
    else:
        R.missing("metacyc_M", "MetaCyc pathways tested", "284", f"missing outputs in {age_dir}")
    cl = attr_dir / "pathway_pattern_classification.csv"
    if not cl.exists():
        R.missing("metacyc_c2_top", "pathways where cluster 2 contributes most (text: six)", "6",
                  f"missing {cl}")
        return
    c = pd.read_csv(cl)
    wc = [f"Cluster{k}_mean_within_cluster_fraction" for k in range(1, 6)]
    argmax = c[wc].to_numpy().argmax(axis=1) + 1
    R.add("metacyc_c2_top", "pathways with largest contribution from cluster 2 (text: six)", "6",
          int((argmax == 2).sum()), source=cl.name, tex=False)
    R.log(f"dominant_pattern labels (>=50% rule): {c['dominant_pattern_label'].value_counts().to_dict()}")
    for key, pat, k, rep in [("tca_c2", r"TCA cycle", 2, "87"),
                             ("ppgpp_c5", r"ppGpp", 5, "76"),
                             ("stearate_c5", r"stearate", 5, "81")]:
        x = c[c["pathway_name"].str.contains(pat, case=False, regex=True)]
        if "TCA" in pat:
            px = x[x["pathway_name"].str.contains("partial", case=False)]
            x = px if len(px) else x
        val = 100 * x[f"Cluster{k}_mean_within_cluster_fraction"].iloc[0] if len(x) == 1 else None
        R.add(f"metacyc_{key}", f"cluster {k} share of '{pat}' pathway (%)", rep, val, source=cl.name,
              note="" if len(x) == 1 else f"{len(x)} pathways matched: {x['pathway_name'].tolist()}")


def crest_checks(R, d: Path):
    diag = d / "molecule_matching_diagnostics.csv"
    hm = d / "cluster_crest_overlap_heatmap.csv"
    if diag.exists():
        g = pd.read_csv(diag).set_index("metric")["value"]
        n_cl, n_in = int(g["unique_clustered_molecules"]), int(g["peak_molecules_with_cluster_assignment"])
        n_crest, n_out = int(g["unique_peak_significant_molecules"]), int(g["peak_molecules_without_cluster_assignment"])
        R.add("overlap_n_clustered", "clustered features", "5,241", n_cl, source=diag.name)
        R.add("overlap_n_in_crest", "clustered features in >=1 crest", "2,228", n_in, source=diag.name)
        R.add("overlap_pct_in_crest", "% of clustered features in >=1 crest", "42.5", 100 * n_in / n_cl,
              source=diag.name)
        R.add("crest_n_features", "crest features (union of crests)", "3,531", n_crest, source=diag.name)
        R.add("crest_not_clustered", "crest features not passing age-association screen", "1,303", n_out,
              source=diag.name)
        R.add("crest_not_clustered_pct", "% crest features not clustered", "36.9", 100 * n_out / n_crest,
              source=diag.name)
    else:
        R.missing("overlap_n_in_crest", "clustered features in a crest", "2,228", f"missing {diag}")
    if hm.exists():
        h = pd.read_csv(hm)
        p = h.pivot_table(index="cluster", columns="category", values="percent_of_cluster")
        R.add("c1_pct_crest2_only", "% of cluster 1 that is crest-2-only", "62.5", p.loc[1, "crest_2_only"],
              source=hm.name)
        top = int(p["crest_1_only"].idxmax())
        R.add("c2_pct_crest1_only", "% of cluster 2 that is crest-1-only", "24.6", p.loc[2, "crest_1_only"],
              source=hm.name, note="" if top == 2 else f"crest-1-only most frequent in C{top}, not C2")
        R.add("c35_max_single_crest", "max % of C3 or C5 in either crest alone (text: no more than)",
              "7.4", p.loc[[3, 5], ["crest_1_only", "crest_2_only"]].max().max(), source=hm.name)
        R.log("crest-only shares by cluster (%): " + "; ".join(
            f"C{i} c1-only {r.crest_1_only:.1f}, c2-only {r.crest_2_only:.1f}, both {r.both_crests:.1f}"
            for i, r in p.iterrows()))


def system_checks(R, f: Path, sizes):
    if not f.exists():
        R.missing("sys_n_assoc", "cluster-phenotype associations FDR<0.05", "444", f"missing {f}")
        return
    d = pd.read_csv(f)
    pid = "pattern_id" if "pattern_id" in d else "cluster_id"
    R.add("sys_n_phenotypes", "phenotypes tested", "871", d[["system", "feature"]].drop_duplicates().shape[0],
          source=f.name)
    R.add("sys_n_systems_tested", "systems tested", "12", d["system"].nunique(), source=f.name)
    R.add("sys_n_min", "participants per test (min)", "4,626", int(d["n"].min()), source=f.name)
    R.add("sys_n_max", "participants per test (max)", "9,761", int(d["n"].max()), source=f.name)
    s = d[d["fdr_global"] < 0.05]
    R.add("sys_n_assoc", "associations FDR<0.05", "444", len(s), source=f.name)
    R.add("sys_n_pheno_assoc", "phenotypes with >=1 association", "312",
          s[["system", "feature"]].drop_duplicates().shape[0], source=f.name)
    missing_sys = sorted(set(d["system"]) - set(s["system"]))
    R.add("sys_n_systems_assoc", "systems with >=1 association (text: all 12)", "12", s["system"].nunique(),
          source=f.name, note=f"no association in: {missing_sys}" if missing_sys else "")
    by = s.groupby(pid)
    n = by.size()
    R.add("sys_c45_n", "associations from clusters 4+5", "311", int(n.get(4, 0) + n.get(5, 0)), source=f.name)
    R.add("sys_c45_pct", "% of associations from clusters 4+5", "70.0", 100 * (n.get(4, 0) + n.get(5, 0)) / len(s),
          source=f.name)
    c5, c4, c3, c2, c1 = (s[s[pid] == k] for k in (5, 4, 3, 2, 1))
    R.add("sys_c5_n", "cluster 5 associations", "182", len(c5), source=f.name)
    R.add("sys_c5_pct_inverse", "cluster 5 % inverse", "94", 100 * (c5["partial_r"] < 0).mean(), source=f.name)
    R.add("sys_c5_bone", "cluster 5 bone-density associations", "64", int((c5["system"] == "bone_density").sum()),
          source=f.name)
    lt = c5[c5["feature"].str.contains("trunk_left_tissue_percent_fat", case=False)]
    R.add("sys_c5_trunk_r", "C5 left-trunk tissue fat %, partial r", "-0.079",
          lt["partial_r"].iloc[0] if len(lt) else None, source=f.name)
    R.add("sys_c5_trunk_fdr", "C5 left-trunk tissue fat %, FDR", "1.30\\times10^{-9}",
          lt["fdr_global"].iloc[0] if len(lt) else None, source=f.name)
    R.add("sys_c4_n", "cluster 4 associations", "129", len(c4), source=f.name)
    R.add("sys_c4_pct_positive", "cluster 4 % positive", "63", 100 * (c4["partial_r"] > 0).mean(), source=f.name)
    hr = c4[(c4["system"] == "sleep") & c4["feature"].str.contains("heart_rate", case=False)]
    R.add("sys_c4_sleep_hr", "cluster 4 sleep heart-rate measures", "12", len(hr), source=f.name)
    fib = c3[c3["feature"].str.fullmatch("dietary_fiber_g", case=False)]
    R.add("sys_c3_fibre_r", "C3 dietary fibre, partial r", "-0.111", fib["partial_r"].iloc[0] if len(fib) else None,
          source=f.name)
    R.add("sys_c3_fibre_fdr", "C3 dietary fibre, FDR", "1.62\\times10^{-22}",
          fib["fdr_global"].iloc[0] if len(fib) else None, source=f.name)
    hy = c2[c2["feature"].str.contains("grade_hypo", case=False)]
    R.add("sys_c2_hypo_r", "C2 hypoglycaemia risk grade, partial r", "-0.077",
          hy["partial_r"].iloc[0] if len(hy) else None, source=f.name)
    R.add("sys_c2_hypo_fdr", "C2 hypoglycaemia risk grade, FDR", "7.93\\times10^{-10}",
          hy["fdr_global"].iloc[0] if len(hy) else None, source=f.name)
    top2 = c2.sort_values("fdr_global").head(1)
    R.log(f"C2 strongest: {top2['system'].iloc[0]}:{top2['feature'].iloc[0]} "
          f"(r {top2['partial_r'].iloc[0]:.3f}, FDR {top2['fdr_global'].iloc[0]:.3g})" if len(top2) else "C2: none")
    liv = c2[c2["system"] == "liver_health"]
    R.add("sys_c2_liver_inverse", "C2 liver-elastography associations inverse (fraction)", "1",
          (liv["partial_r"] < 0).mean() if len(liv) else None, tex=False, source=f.name,
          note=f"{len(liv)} liver associations")
    cs = c1[c1["system"].isin(["cardio", "sleep"])]
    R.add("sys_c1_cardio_sleep_positive", "C1 cardiovascular+sleep associations positive (fraction)", "1",
          (cs["partial_r"] > 0).mean() if len(cs) else None, tex=False, source=f.name,
          note=f"{len(cs)} associations")
    R.log("associations by cluster x system:\n    " +
          s.groupby([pid, "system"]).size().unstack(fill_value=0).to_string().replace("\n", "\n    "))


def main(argv=None) -> int:
    a = parse_args(argv)
    R = Report("Result 3: multi-omics trajectories", a.tex, SECTION_TITLE)
    out = a.outputs_dir
    sizes = clustering_checks(R, out / "molecule_clustering")
    transcriptome_checks(R, out / "transcriptome_cluster_pathway_landscape")
    lion_checks(R, out / "lion_web_five_patterns")
    oral_checks(R, out / "oral_metacyc_age_association", out / "oral_metacyc_age_cluster_attribution")
    crest_checks(R, out / "crest_cluster_intersections")
    system_checks(R, out / "cluster_system_association" / "cluster_system_associations.csv", sizes)
    return R.write(a.report_dir or out / "manuscript_stats")


if __name__ == "__main__":
    raise SystemExit(main())
