#!/usr/bin/env python
"""Manuscript numbers for the oral MetaCyc age-association sentence.

Fills the placeholders in:

    Of [M] pathways retained in [N] participants, [K] were associated with age
    after Bonferroni correction, and the 12 non-redundant pathways displayed all
    increased with age, by approximately 0.1 s.d. per decade
    (Fig. trajectory-system-coupling d).

Read-only summary of the completed outputs of
``04-oral_metacyc_pathway_age_association.py`` (all statistics) and, when
available, ``04-oral_metacyc_pathway_cluster_attribution.py`` (consistency
check of the displayed pathways and the attribution sample size). Nothing is
re-fitted.

Definitions
-----------
M  rows of ``metacyc_pathway_age_associations_all.csv`` (= Bonferroni test
   family; equals manifest ``pathway_preprocessing.retained_pathways``).
N  participants in the age models (column ``n``; equals manifest
   ``pathway_preprocessing.samples_retained_after_pathway_qc``).
K  pathways with ``p_bonferroni_global < alpha``.
The displayed pathways are ``figure_pathways.csv`` (Bonferroni-significant,
greedy |Spearman| redundancy pruning, capped at ``--max-pathways``).
Effects are ``beta_pathway_sd_per_10y``: s.d. of log10(PPM) per 10 years of
age, adjusted for sex, HC3 95% CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_AGE_DIR = SCRIPT_DIR / "outputs" / "oral_metacyc_age_association"
DEFAULT_ATTR_DIR = SCRIPT_DIR / "outputs" / "oral_metacyc_age_cluster_attribution"
BETA = "beta_pathway_sd_per_10y"
LOW = "ci95_low_pathway_sd_per_10y"
HIGH = "ci95_high_pathway_sd_per_10y"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--age-association-dir", type=Path, default=DEFAULT_AGE_DIR)
    parser.add_argument("--attribution-dir", type=Path, default=DEFAULT_ATTR_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <attribution-dir>/manuscript_stats",
    )
    return parser.parse_args(argv)


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"


def describe(values: pd.Series) -> dict[str, float]:
    v = values.astype(float)
    return {
        "n": int(v.size),
        "mean": float(v.mean()) if v.size else np.nan,
        "median": float(v.median()) if v.size else np.nan,
        "min": float(v.min()) if v.size else np.nan,
        "max": float(v.max()) if v.size else np.nan,
        "q25": float(v.quantile(0.25)) if v.size else np.nan,
        "q75": float(v.quantile(0.75)) if v.size else np.nan,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    age_dir = args.age_association_dir
    attr_dir = args.attribution_dir
    out_dir = args.output_dir or (attr_dir / "manuscript_stats")

    manifest_path = age_dir / "oral_metacyc_pathway_age_association_manifest.json"
    all_path = age_dir / "metacyc_pathway_age_associations_all.csv"
    fig_path = age_dir / "figure_pathways.csv"
    sel_audit_path = age_dir / "figure_pathway_selection_audit.csv"
    for path in (manifest_path, all_path, fig_path):
        if not path.exists():
            sys.exit(f"Missing 09 output: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = manifest.get("status")
    if status not in {"completed", "completed_no_main_figure"}:
        sys.exit(f"09 manifest status is {status!r}; run 09 to completion first")
    stat_audit = manifest.get("statistical_analysis", {})
    prep = manifest.get("pathway_preprocessing", {})
    selection = manifest.get("figure_selection", {})
    alpha = float(stat_audit.get("alpha", selection.get("alpha", 0.05)))

    assoc = pd.read_csv(all_path)
    assoc["pathway_id"] = assoc["pathway_id"].astype(str)
    fig = pd.read_csv(fig_path)
    fig["pathway_id"] = fig["pathway_id"].astype(str)

    checks: list[tuple[str, bool, str]] = []

    # ---- M and N --------------------------------------------------------
    M = int(len(assoc))
    n_values = sorted(assoc["n"].astype(int).unique().tolist())
    N = int(n_values[0]) if len(n_values) == 1 else int(max(n_values))
    checks.append(("single N across all pathway models", len(n_values) == 1,
                   f"unique n = {n_values}"))
    if "retained_pathways" in prep:
        checks.append(("M == manifest retained_pathways",
                       M == int(prep["retained_pathways"]),
                       f"table {M} vs manifest {prep['retained_pathways']}"))
    if "samples_retained_after_pathway_qc" in prep:
        checks.append(("N == manifest samples_retained_after_pathway_qc",
                       N == int(prep["samples_retained_after_pathway_qc"]),
                       f"table {N} vs manifest "
                       f"{prep['samples_retained_after_pathway_qc']}"))

    # ---- K --------------------------------------------------------------
    sig = assoc.loc[assoc["p_bonferroni_global"] < alpha].copy()
    K = int(len(sig))
    K_up = int((sig[BETA] > 0).sum())
    K_down = int((sig[BETA] < 0).sum())
    K_bh = int((assoc["q_bh_global"] < alpha).sum())
    if "bonferroni_positive" in stat_audit:
        checks.append(("K == manifest bonferroni_positive",
                       K == int(stat_audit["bonferroni_positive"]),
                       f"table {K} vs manifest {stat_audit['bonferroni_positive']}"))

    # ---- displayed (non-redundant) pathways ------------------------------
    D = int(len(fig))
    D_up = int((fig[BETA] > 0).sum())
    all_up = D > 0 and D_up == D
    all_ci_above_zero = bool((fig[LOW] > 0).all()) if D else False
    checks.append(("all displayed pathways Bonferroni-significant",
                   bool((fig["p_bonferroni_global"] < alpha).all()), ""))

    sel_counts: dict[str, int] = {}
    if sel_audit_path.exists():
        sel_audit = pd.read_csv(sel_audit_path)
        sel_counts = sel_audit["display_status"].value_counts().to_dict()
    capped = sel_counts.get("excluded_after_max_pathways_reached", 0) > 0
    redundant_n = sel_counts.get("excluded_from_figure_for_correlation_redundancy", 0)

    beta_fig = describe(fig[BETA])
    beta_sig = describe(sig[BETA])
    beta_sig_up = describe(sig.loc[sig[BETA] > 0, BETA])

    # ---- optional 10 consistency check ----------------------------------
    attr_info: dict[str, object] = {}
    attr_manifest_path = (
        attr_dir / "oral_metacyc_pathway_age_cluster_attribution_manifest.json"
    )
    attr_src_path = attr_dir / "figure_pathway_source_data.csv"
    if attr_manifest_path.exists():
        am = json.loads(attr_manifest_path.read_text(encoding="utf-8"))
        align = am.get("sample_alignment", {})
        attr_info = {
            "status": am.get("status"),
            "upstream_age_run_id": am.get("upstream_age_analysis", {}).get("run_id"),
            "age_run_id": manifest.get("run_id"),
            "age_analysis_samples": align.get("age_analysis_samples"),
            "shared_samples_for_pattern_attribution": align.get(
                "shared_samples_for_pattern_attribution"),
        }
        checks.append(("10 built on this 09 run",
                       attr_info["upstream_age_run_id"] == attr_info["age_run_id"],
                       f"{attr_info['upstream_age_run_id']} vs {attr_info['age_run_id']}"))
    if attr_src_path.exists():
        src = pd.read_csv(attr_src_path)
        same = set(src["pathway_id"].astype(str)) == set(fig["pathway_id"])
        checks.append(("10 figure pathways == 09 figure_pathways", same,
                       f"{src['pathway_id'].nunique()} vs {D}"))

    # ---- outputs --------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = ["pathway_id", "pathway_name", "n", BETA, LOW, HIGH,
            "p_age_two_sided", "p_bonferroni_global", "q_bh_global"]
    fig_table = fig[[c for c in keep if c in fig.columns]].copy()
    fig_table.insert(0, "display_rank", np.arange(1, D + 1))
    fig_table.to_csv(out_dir / "displayed_pathway_effects.csv", index=False)
    sig[[c for c in keep if c in sig.columns]].to_csv(
        out_dir / "bonferroni_significant_pathways.csv", index=False)

    summary = {
        "M_retained_pathways": M,
        "N_participants": N,
        "K_bonferroni_significant": K,
        "K_increasing": K_up,
        "K_decreasing": K_down,
        "BH_FDR_significant": K_bh,
        "alpha": alpha,
        "bonferroni_threshold_raw_p": alpha / M if M else np.nan,
        "displayed_pathways": D,
        "displayed_increasing": D_up,
        "displayed_all_increasing": bool(all_up),
        "displayed_all_ci_low_above_zero": all_ci_above_zero,
        "display_capped_at_max_pathways": bool(capped),
        "selection_audit_counts": sel_counts,
        "max_pathways": selection.get("maximum_pathways"),
        "redundancy_threshold_abs_spearman": selection.get(
            "figure_only_absolute_spearman_redundancy_threshold"),
        "beta_sd_per_decade_displayed": beta_fig,
        "beta_sd_per_decade_bonferroni_all": beta_sig,
        "beta_sd_per_decade_bonferroni_increasing": beta_sig_up,
        "age_range": [manifest.get("arguments", {}).get("age_min"),
                      manifest.get("arguments", {}).get("age_max")],
        "age_run_id": manifest.get("run_id"),
        "attribution": attr_info,
        "checks": [{"check": c, "pass": bool(ok), "detail": d} for c, ok, d in checks],
    }
    (out_dir / "manuscript_stats.json").write_text(
        json.dumps(summary, indent=2, default=lambda o: o.item()
                   if hasattr(o, "item") else str(o)),
        encoding="utf-8")

    b = beta_fig
    sentence = (
        f"Of {M:,} pathways retained in {N:,} participants, {K:,} were associated "
        f"with age after Bonferroni correction, and the {D} non-redundant pathways "
        f"displayed {'all increased' if all_up else f'({D_up}/{D} increasing)'} "
        f"with age, by approximately {b['median']:.2f} s.d. per decade "
        f"(median; range {b['min']:.2f}-{b['max']:.2f})."
    )
    lines = [
        "== Oral MetaCyc age association: manuscript numbers ==",
        f"09 run_id: {manifest.get('run_id')}  status: {status}",
        f"[M] retained pathways (test family): {M:,}",
        f"[N] participants: {N:,}",
        f"[K] Bonferroni < {alpha:g}: {K:,}  (increase {K_up}, decrease {K_down}); "
        f"raw P threshold {alpha / M:.2e}; BH-FDR < {alpha:g}: {K_bh:,}",
        f"Displayed pathways: {D} (increasing {D_up}/{D}; all CI low > 0: "
        f"{all_ci_above_zero})",
        f"  selection audit: {sel_counts}",
        f"  capped at max_pathways ({selection.get('maximum_pathways')}): {capped}; "
        f"redundant excluded before cap: {redundant_n}",
        f"Displayed beta (s.d./decade): mean {b['mean']:.3f}, median {b['median']:.3f}, "
        f"IQR {b['q25']:.3f}-{b['q75']:.3f}, range {b['min']:.3f}-{b['max']:.3f}",
        f"All K beta (s.d./decade): median {beta_sig['median']:.3f}, "
        f"range {beta_sig['min']:.3f}-{beta_sig['max']:.3f}",
    ]
    if attr_info:
        lines.append(
            f"10 attribution: status {attr_info['status']}, shared samples "
            f"{attr_info['shared_samples_for_pattern_attribution']} of "
            f"{attr_info['age_analysis_samples']}")
    lines.append("")
    lines.append("Displayed pathways:")
    for r in fig_table.itertuples(index=False):
        lines.append(
            f"  {r.display_rank:>2}. {r.pathway_id}  beta {getattr(r, BETA):+.3f} "
            f"[{getattr(r, LOW):+.3f}, {getattr(r, HIGH):+.3f}]  "
            f"P_bonf {fmt_p(r.p_bonferroni_global)}  {r.pathway_name}")
    lines.append("")
    lines.append("Checks:")
    for c, ok, d in checks:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {c} {('- ' + d) if d else ''}")
    lines.append("")
    lines.append("Draft sentence:")
    lines.append("  " + sentence)
    if capped:
        lines.append(
            "  NOTE: the figure hit the max-pathways cap, so these are the top "
            f"{D} non-redundant pathways by P, not all non-redundant pathways.")
    if not all_up:
        lines.append("  NOTE: not all displayed pathways increase with age; "
                     "revise 'all increased'.")
    report = "\n".join(lines)
    (out_dir / "manuscript_stats.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nWritten to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
