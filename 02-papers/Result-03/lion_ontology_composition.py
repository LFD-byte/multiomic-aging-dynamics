"""Reconstruct LION annotations and audit cluster-wise annotation coverage.

Uses the official LION database and LION/web name mappings. The naming rules
below cover the standardized names in this analysis, with direct matching,
sum-composition matching and the published headgroup/explicit-chain fallback.
No fatty-acid chains are predicted from a sum composition. Before plotting,
all reconstructed term totals must match every supplied LION result export.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import re
import sqlite3

import numpy as np
import pandas as pd


def normalize_lion_name(name: str) -> str:
    """Apply LION/web aliases for the standardized input naming conventions."""
    name = re.sub(r"^FA\(", "FFA(", str(name))
    name = name.replace("GlcCer", "HexCer").replace("LacCer", "Hex2Cer")
    match = re.fullmatch(r"L([PC][A-Z]*)\(([^()]*)\)", name)
    if match:
        chains = match[2]
        if "0:0" not in chains.split("/"):
            chains += "/0:0"
        name = f"{match[1]}({chains})"
    return name


def generalize_lion_name(name: str) -> str:
    def sum_chains(match: re.Match) -> str:
        chains = [tuple(map(int, pair)) for pair in re.findall(r"(\d+):(\d+)", match[0])]
        if (0, 0) in chains:
            return match[0]
        return f"{sum(c for c, _ in chains)}:{sum(d for _, d in chains)}"

    return re.sub(r"(?:iso-)?[dt]?\d+:\d+(?:[/_]\d+:\d+)+", sum_chains, name)


def reconstruct_annotations(mapping: pd.DataFrame, reference_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    files = {
        "database": reference_dir / "LION_DB.sqlite",
        "associations": reference_dir / "lion_association.txt",
        "rules": reference_dir / "lion_rules.csv",
        "fa_rules": reference_dir / "lion_fa_rules.csv",
        "provenance": reference_dir / "database_provenance.json",
        "web_commit": reference_dir / "lion_web_commit.txt",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Official LION annotation references are missing: " + ", ".join(missing))

    with sqlite3.connect(files["database"].resolve().as_uri() + "?mode=ro", uri=True) as db:
        ids = dict(db.execute("SELECT _id, id FROM term"))
        parents: dict[str, set[str]] = defaultdict(set)
        for child, parent in db.execute("SELECT _id, _parent_id FROM parents"):
            parents[ids[child]].add(ids[parent])
    known_ids = set(ids.values())

    @lru_cache(maxsize=None)
    def ancestors(term: str) -> frozenset[str]:
        if term not in known_ids:
            raise ValueError(f"Annotation seed {term} is absent from the LION database.")
        result = {term}
        for parent in parents[term]:
            result.update(ancestors(parent))
        return frozenset(result)

    associations: dict[str, set[str]] = defaultdict(set)
    for line in files["associations"].read_text(encoding="utf-8").splitlines():
        name, term_text = line.split("\t", 1)
        associations[name.replace(" ", "")].update(re.findall(r"LION:\d+", term_text))
    rules = pd.read_csv(files["rules"]).fillna("")
    fa_rules = pd.read_csv(files["fa_rules"]).fillna("")
    fa_lookup = dict(zip(fa_rules["name"], fa_rules["LION"]))
    rows = []
    for lipid in mapping.itertuples(index=False):
        name = normalize_lion_name(lipid.standard_name)
        seeds = set(associations.get(name, set()))
        status = "direct"
        if not seeds:
            status = "smart_matching"
            seeds.update(associations.get(generalize_lion_name(name), set()))
            chains = re.findall(r"(?:iso-)?[dt]?\d+:\d+", name)
            headgroup = name.split("(", 1)[0] if "(" in name else ""
            if headgroup in {"PC", "PE", "PA", "PG", "PI", "PS", "CL"} and "0:0" in chains:
                headgroup = "L" + headgroup
            linkage = "P" if re.search(r"P-\d", name) else "O" if re.search(r"O-\d", name) else "D"
            if not seeds:
                matches = rules.loc[rules["RULE1"].eq(headgroup) & rules["RULE1"].ne("")]
                if not matches["RULE2"].eq("").all():
                    matches = matches.loc[matches["RULE2"].eq(linkage)]
                seeds.update(matches["LION"])
            for chain in chains:
                fa_name = {
                    "d18:0": "d18:0 (dihydrosphingosine)",
                    "d18:1": "d18:1 (sphingosine)",
                }.get(chain, "C" + chain if chain[0].isdigit() else chain)
                if fa_name in fa_lookup:
                    seeds.add(fa_lookup[fa_name])
        term_ids: set[str] = set()
        for seed in seeds:
            term_ids.update(ancestors(seed))
        rows.append({
            "standard_name": lipid.standard_name,
            "pattern_id": int(lipid.pattern_id),
            "pattern": lipid.pattern,
            "normalized_name": name,
            "annotation_status": status if seeds else "unmatched",
            "seed_term_ids": ";".join(sorted(seeds)),
            "all_term_ids": frozenset(term_ids),
        })
    return pd.DataFrame(rows), list(files.values())


def summarize_ontology_coverage(
    annotations: pd.DataFrame,
    all_results: pd.DataFrame,
    selected_terms: pd.DataFrame,
    pattern_ids: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return percentages, lipid membership and a complete count cross-check.

    Percentages use all submitted, hard-assigned lipid species as denominator.
    Unmatched lipids remain in this denominator; their term membership stays
    missing in the audit, rather than asserting biological absence.
    """
    if annotations["standard_name"].duplicated().any():
        raise ValueError("Duplicate lipid names would inflate ontology coverage.")
    if set(annotations["pattern_id"]) != set(pattern_ids):
        raise ValueError("Annotation coverage must include all five clusters.")

    memberships = []
    checks = []
    for term_id, term_results in all_results.groupby("term_id", sort=True):
        total = int(annotations["all_term_ids"].map(lambda ids: term_id in ids).sum())
        expected = term_results["annotated"].unique()
        matched = len(expected) == 1 and np.isfinite(expected[0]) and total == expected[0]
        checks.append({
            "term_id": term_id,
            "description": term_results["description"].iloc[0],
            "web_annotated": expected[0] if len(expected) == 1 else np.nan,
            "reconstructed_annotated": total,
            "result_rows_checked": len(term_results),
            "matches_web_export": bool(matched),
        })
        for lipid in annotations.itertuples(index=False):
            memberships.append({
                "term_id": term_id,
                "standard_name": lipid.standard_name,
                "pattern_id": lipid.pattern_id,
                "normalized_name": lipid.normalized_name,
                "annotation_status": lipid.annotation_status,
                "seed_term_ids": lipid.seed_term_ids,
                "is_annotated": pd.NA if lipid.annotation_status == "unmatched" else term_id in lipid.all_term_ids,
            })
    count_audit = pd.DataFrame(checks)
    failures = count_audit.loc[~count_audit["matches_web_export"]]
    if not failures.empty:
        raise ValueError(
            "Reconstructed annotations disagree with the LION exports; do not plot "
            "inferred cluster percentages. Mismatches: "
            + failures.to_json(orient="records")
        )
    membership_audit = pd.DataFrame(memberships)
    membership_audit["is_annotated"] = membership_audit["is_annotated"].astype("boolean")
    selected_ids = set(selected_terms["term_id"])
    if not selected_ids.issubset(set(count_audit["term_id"])):
        raise ValueError("Every selected display term requires an audited LION result.")

    rows = []
    for term in selected_terms.sort_values("display_order").itertuples(index=False):
        for pattern_id in pattern_ids:
            group = annotations.loc[annotations["pattern_id"].eq(pattern_id)]
            labels = group["pattern"].dropna().unique()
            if len(labels) != 1:
                raise ValueError(f"Ambiguous pattern label for cluster {pattern_id}.")
            numerator = int(group["all_term_ids"].map(lambda ids: term.term_id in ids).sum())
            denominator = len(group)
            rows.append({
                "term_id": term.term_id,
                "term_label": term.term_label,
                "display_order": term.display_order,
                "pattern_id": pattern_id,
                "pattern": labels[0],
                "assigned_lipids": denominator,
                "ontology_matched_lipids": int(group["annotation_status"].ne("unmatched").sum()),
                "unmatched_lipids": int(group["annotation_status"].eq("unmatched").sum()),
                "annotated_lipids": numerator,
                "denominator": denominator,
                "annotated_percentage": 100.0 * numerator / denominator,
            })
    summary = pd.DataFrame(rows)
    summary["row_z_score"] = summary.groupby("term_id")["annotated_percentage"].transform(
        lambda values: (values - values.mean()) / values.std(ddof=0)
        if values.std(ddof=0) > 0 else np.zeros(len(values))
    )
    summary["display_value"] = summary["annotated_percentage"].map(lambda value: f"{value:.0f}%")
    return summary, membership_audit, count_audit
