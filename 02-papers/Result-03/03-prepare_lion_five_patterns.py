#!/usr/bin/env python
"""Prepare LION/web inputs for the five molecular-age trajectory patterns.

Two complementary analyses are exported:

1. Ranked enrichment (primary): every LION-compatible lipid receives a
   continuous, pattern-specific trajectory-resemblance statistic.  The score
   is the distance to the closest non-target morphology centroid minus the
   distance to the target centroid.  Larger values therefore mean stronger
   relative resemblance to the requested pattern, and every LION/web ranked
   run must use ``high to low``.
2. Target-list ORA (sensitivity): lipids assigned to one hard trajectory
   cluster are tested against the common set of all compatible lipids.

The morphology calculation and scaling reproduce the 14-feature Ward
clustering in ``molecule_clustering_ncomms.py``.  The exported files contain
lipid names and derived scores only; no participant-level data are included.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
RESULT02 = HERE.parent / "Result-02"
DEFAULT_CLUSTER_DIR = HERE / "outputs" / "molecule_clustering"
DEFAULT_MAPPING = RESULT02 / "outputs" / "lipid_pathway" / "lipid_name_mapping.csv"
DEFAULT_OUTPUT = HERE / "outputs" / "lion_web_five_patterns"

EXPECTED_PATTERN_IDS = (1, 2, 3, 4, 5)
EXCLUDED_MAPPING_STATUSES = {"unsupported", "ambiguous_subclass"}
PATTERN_SLUGS = {
    1: "late_accelerating_rise",
    2: "gradual_rise",
    3: "accelerating_decline",
    4: "midlife_trough",
    5: "midlife_peak",
}
LION_RESULT_COLUMNS = (
    "Term ID",
    "Discription",  # spelling used by the current LION/web export
    "Annotated",
    "p-value",
    "FDR q-value",
)


def extract_morphology(curve: np.ndarray) -> np.ndarray:
    """Return the same 14 morphology features used for the five clusters."""

    curve = np.asarray(curve, dtype=float)
    n = len(curve)
    seg = n // 3
    d1 = np.gradient(curve)
    d2 = np.gradient(d1)
    d1_early, d1_mid, d1_late = (
        d1[:seg].mean(),
        d1[seg : 2 * seg].mean(),
        d1[2 * seg :].mean(),
    )
    signs = np.sign(d1)
    sign_changes = np.sum(np.abs(np.diff(signs)) > 0) / n
    mu_early, mu_mid, mu_late = (
        curve[:seg].mean(),
        curve[seg : 2 * seg].mean(),
        curve[2 * seg :].mean(),
    )
    return np.array(
        [
            d1_early,
            d1_mid,
            d1_late,
            sign_changes,
            mu_early,
            mu_mid,
            mu_late,
            np.abs(d2).mean(),
            np.argmax(np.abs(d2)) / n,
            curve[-1] - curve[0],
            d1_late - d1_early,
            mu_mid - (mu_early + mu_late) / 2,
            np.abs(d1[:seg]).mean(),
            np.abs(d1[2 * seg :]).mean(),
        ],
        dtype=float,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def write_lines(values: Iterable[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = [str(value).strip() for value in values if str(value).strip()]
    path.write_text("\n".join(clean) + "\n", encoding="utf-8", newline="\n")


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def load_aligned_data(cluster_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    curve_path = cluster_dir / "gam_curves.npz"
    metadata_path = cluster_dir / "gam_curve_metadata.csv"
    assignment_path = cluster_dir / "molecule_cluster_assignments_agglomerative.csv"
    for path in (curve_path, metadata_path, assignment_path):
        if not path.exists():
            raise FileNotFoundError(f"Required clustering input not found: {path}")

    with np.load(curve_path) as archive:
        if "curves" not in archive:
            raise ValueError(f"{curve_path} does not contain an array named 'curves'.")
        curves = np.asarray(archive["curves"], dtype=float)

    metadata = pd.read_csv(metadata_path)
    assignments = pd.read_csv(assignment_path)
    require_columns(metadata, {"feature", "omics"}, "GAM metadata")
    require_columns(
        assignments,
        {"feature", "omics", "cluster_id", "pattern_id", "pattern"},
        "cluster assignments",
    )
    if len(curves) != len(metadata):
        raise ValueError(
            f"Curve/metadata length mismatch: {len(curves)} curves and "
            f"{len(metadata)} metadata rows."
        )
    if metadata.duplicated(["feature", "omics"]).any():
        raise ValueError("GAM metadata contains duplicated feature/omics keys.")
    if assignments.duplicated(["feature", "omics"]).any():
        raise ValueError("Cluster assignments contain duplicated feature/omics keys.")

    metadata = metadata.copy()
    metadata["curve_index"] = np.arange(len(metadata), dtype=int)
    aligned = metadata.merge(
        assignments,
        on=["feature", "omics"],
        how="left",
        validate="one_to_one",
    ).sort_values("curve_index")
    if aligned["pattern_id"].isna().any():
        missing = aligned.loc[aligned["pattern_id"].isna(), ["feature", "omics"]]
        raise ValueError(f"Missing cluster assignments for:\n{missing.head(10)}")
    aligned["pattern_id"] = aligned["pattern_id"].astype(int)
    observed = tuple(sorted(aligned["pattern_id"].unique()))
    if observed != EXPECTED_PATTERN_IDS:
        raise ValueError(f"Expected patterns {EXPECTED_PATTERN_IDS}, observed {observed}.")
    return curves, aligned


def compute_pattern_scores(
    curves: np.ndarray, aligned: pd.DataFrame
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Compute continuous resemblance scores in the original feature space."""

    morphology = np.vstack([extract_morphology(curve) for curve in curves])
    if not np.all(np.isfinite(morphology)):
        raise ValueError("Non-finite trajectory morphology values were detected.")
    morphology_z = StandardScaler().fit_transform(morphology)

    labels = aligned["pattern_id"].to_numpy(dtype=int)
    centroids = np.vstack(
        [morphology_z[labels == pattern_id].mean(axis=0) for pattern_id in EXPECTED_PATTERN_IDS]
    )
    # Root-mean-square distance is Euclidean distance divided by sqrt(14).  It
    # preserves ranks while making the scale independent of feature count.
    distances = np.sqrt(
        np.mean((morphology_z[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
    )

    curve_centers = {
        pattern_id: curves[labels == pattern_id].mean(axis=0)
        for pattern_id in EXPECTED_PATTERN_IDS
    }
    score_table = aligned[["curve_index", "feature", "omics", "cluster_id", "pattern_id", "pattern"]].copy()
    for column_index, pattern_id in enumerate(EXPECTED_PATTERN_IDS):
        target_distance = distances[:, column_index]
        other_distance = np.delete(distances, column_index, axis=1).min(axis=1)
        score_table[f"p{pattern_id:02d}_target_morphology_distance"] = target_distance
        score_table[f"p{pattern_id:02d}_nearest_other_distance"] = other_distance
        score_table[f"p{pattern_id:02d}_trajectory_resemblance_score"] = (
            other_distance - target_distance
        )
        score_table[f"p{pattern_id:02d}_curve_correlation"] = [
            safe_correlation(curve, curve_centers[pattern_id]) for curve in curves
        ]
    return score_table, curve_centers


def add_lion_mapping(lipid_scores: pd.DataFrame, mapping_path: Path) -> pd.DataFrame:
    if not mapping_path.exists():
        raise FileNotFoundError(f"Lipid-name mapping not found: {mapping_path}")
    mapping = pd.read_csv(mapping_path)
    require_columns(
        mapping,
        {"feature", "standard_name", "mapping_status"},
        "Lipid-name mapping",
    )
    if mapping.duplicated("feature").any():
        raise ValueError("Lipid-name mapping contains duplicated original feature names.")
    mapped = lipid_scores.merge(
        mapping,
        on="feature",
        how="left",
        validate="one_to_one",
    )
    standard = mapped["standard_name"].fillna("").astype(str).str.strip()
    status = mapped["mapping_status"].fillna("missing_mapping").astype(str).str.strip()
    mapped["standard_name"] = standard
    mapped["mapping_status"] = status
    mapped["lion_eligible"] = standard.ne("") & ~status.isin(EXCLUDED_MAPPING_STATUSES)

    # A composition-level name cannot be assigned unambiguously to one target
    # list if multiple measured features with that same name occupy different
    # hard clusters.  Such names remain usable in ranking mode after averaging,
    # but are omitted from ORA target/background files.
    eligible = mapped[mapped["lion_eligible"]]
    pattern_counts = eligible.groupby("standard_name")["pattern_id"].nunique()
    cross_pattern_names = set(pattern_counts[pattern_counts > 1].index)
    mapped["cross_pattern_name_conflict"] = mapped["standard_name"].isin(cross_pattern_names)
    mapped["ora_eligible"] = mapped["lion_eligible"] & ~mapped["cross_pattern_name_conflict"]
    return mapped


def expected_names(pattern_id: int, mode: str) -> dict[str, str]:
    slug = PATTERN_SLUGS[pattern_id]
    stem = f"lion_p{pattern_id:02d}_{slug}_{mode}"
    return {
        "result_csv": f"{stem}_result.csv",
        "plot_svg": f"{stem}_plot.svg",
        "network_svg": f"{stem}_network.svg",
    }


def build_readme(summary: pd.DataFrame, manifest: pd.DataFrame) -> str:
    counts = {
        int(row.pattern_id): int(row.ora_target_lipids)
        for row in summary.itertuples(index=False)
    }
    ranking_rows = manifest[manifest["mode"].eq("ranked")]
    ora_rows = manifest[manifest["mode"].eq("ora")]

    ranking_table = [
        "| Trajectory group | Input file to paste | Direction | Required name of the downloaded file |",
        "|---|---|---|---|",
    ]
    for row in ranking_rows.itertuples(index=False):
        ranking_table.append(
            f"| P{row.pattern_id}: {row.pattern} | `{row.primary_input}` | "
            f"**high to low** | `{row.expected_result_csv}` |"
        )

    ora_table = [
        "| Trajectory group | Target file | Targets | Required name of the downloaded file |",
        "|---|---|---:|---|",
    ]
    for row in ora_rows.itertuples(index=False):
        ora_table.append(
            f"| P{row.pattern_id}: {row.pattern} | `{row.primary_input}` | "
            f"{counts[row.pattern_id]} | `{row.expected_result_csv}` |"
        )

    required_files = "\n".join(
        f"- `{name}`" for name in ranking_rows["expected_result_csv"].tolist()
    )
    optional_files = "\n".join(
        f"- `{name}`" for name in ora_rows["expected_result_csv"].tolist()
    )

    return f"""# LION/web run instructions for the five lipid trajectory groups

The inputs in this directory follow the LION/web input conventions of the Result-02 lipid analysis. Uploaded files contain lipid names and derived scores only, with no participant-level data.

`lion_web_upload_bundle.zip` is for copying and archiving only. Unpack it and use the TXT files as described below; do not upload the ZIP itself to the website.

## A. Primary analysis: ranked enrichment (required, 5 runs)

1. Open the corresponding `*_custom_statistic.txt` and copy its entire content.
2. In the ranking/custom-statistic mode of LION/web, skip **(i) process input** and paste the two columns into **(ii) analysis**.
3. Select **high to low** for all five groups. The first column is the LION standard lipid name; a higher score in the second column means closer to the target trajectory. The score is not an up- or down-regulation effect.
4. Download the complete, unfiltered result table as CSV, rename it immediately and place it in `results/` in this directory.
5. In every run, analyse all LION terms (no term preselection), enable **exclude parent terms with same associations as child** and disable **automatically send unmatched lipids to LION-team**.

{chr(10).join(ranking_table)}

The 5 required primary results:

{required_files}

## B. Sensitivity analysis: target-list ORA (recommended, 5 runs)

Use the corresponding `*_target.txt` as the target and `inputs/target_list/lion_all_patterns_background.txt` as the background in every run. Download the complete result table and name it as listed below.

{chr(10).join(ora_table)}

The 5 recommended ORA results:

{optional_files}

P4 has only {counts[4]} uploadable lipids and P1 only {counts[1]}, so their ORA power is limited. Target numbers differ greatly between groups, so the five groups cannot be compared directly by the number of significant terms. ORA validates the hard assignment; ranked enrichment is the primary analysis.

## C. What to keep from the website

- **Required**: the complete result table as CSV, not only rows with FDR < 0.05. The current LION/web table usually contains `Term ID`, `Discription`, `Annotated`, `p-value` and `FDR q-value`.
- **Optional but recommended**: if the website provides enrichment and network plots, name them according to `expected_plot_svg` and `expected_network_svg` in the manifest, preferably as SVG.
- Record the run date, the number of lipids recognized by the website and the number unmatched in `lion_web_run_log.csv`. If unmatched names occur, save the names shown by the website as `lion_unmatched_lipids.txt`.
- Functional-theme summarization can proceed once the 5 primary CSV files are in place; ORA files can be added later.
- Do not use screenshots as result tables, and do not overwrite the downloaded files by re-saving them in Excel.

After placing the 5 primary results, run the check from the project root:

```sh
python 02-papers/Result-03/03-validate-lion-five-pattern-results.py
```

If all 10 ranked + ORA results are complete, append `--require-ora`.

## D. Score definition and interpretive limits

The 14 shape features of the original analysis are recomputed from the z-standardized GAM curves of all 5,241 clustered molecules and standardized as in the original clustering. For lipid *i* and target trajectory *k*:

`score(i,k) = distance to the nearest non-target trajectory centre - distance to the target trajectory centre`

A high score means that the age trajectory of the lipid lies closer to the target group in shape space. It is used for ranking only and does not indicate increased or decreased concentration or pathway activation. Hard cluster assignment and continuous similarity are complementary evidence, and negative scores for a few boundary lipids are expected.

`ranking_auc_for_hard_assignment` in `lion_input_summary.csv` uses the hard assignment as an independent check of ranking direction; it is not used for filtering or for computing LION p values.

## E. Directory contents

- `inputs/ranking/`: 5 headerless, tab-separated two-column files ready to paste, and audit CSV files with headers.
- `inputs/target_list/`: 5 target files, 5 rest files and 1 shared background, each with one lipid name per line.
- `lion_web_run_manifest.csv`: inputs, direction and mandatory result file names of the 10 runs.
- `lion_web_run_log.csv`: template run log for the website, to be filled with recognition coverage and notes.
- `lion_input_summary.csv`: number of targets and background lipids per group and ranking-direction QC.
- `audit/lipid_mapping_and_pattern_scores.csv`: mapping status and continuous scores for the five groups for the 93 original lipids.
- `audit/lion_mapping_exclusions.csv`: entries not uploaded and the reasons.
- `lion_web_upload_bundle.zip`: complete upload bundle for copying to the computer running the browser.
"""


def validate_outputs(output_dir: Path, manifest: pd.DataFrame, background: list[str]) -> None:
    ranking = manifest[manifest["mode"].eq("ranked")]
    ora = manifest[manifest["mode"].eq("ora")]
    if len(ranking) != 5 or len(ora) != 5 or len(manifest) != 10:
        raise AssertionError("The run manifest must contain five ranked and five ORA runs.")
    background_set = set(background)
    if len(background_set) != len(background):
        raise AssertionError("The ORA background contains duplicated lipid names.")

    targets: list[set[str]] = []
    for row in ranking.itertuples(index=False):
        path = output_dir / row.primary_input
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        if len(lines) != int(row.input_lipids):
            raise AssertionError(f"Unexpected ranked-input length: {path}")
        fields = [line.split("\t") for line in lines]
        if any(len(parts) != 2 for parts in fields):
            raise AssertionError(f"Ranked input is not exactly two columns: {path}")
        names = [parts[0] for parts in fields]
        scores = [float(parts[1]) for parts in fields]
        if len(names) != len(set(names)):
            raise AssertionError(f"Duplicated names in ranked input: {path}")
        if not all(math.isfinite(score) for score in scores):
            raise AssertionError(f"Non-finite score in ranked input: {path}")
        if any(scores[i] < scores[i + 1] for i in range(len(scores) - 1)):
            raise AssertionError(f"Ranked input is not sorted high-to-low: {path}")

    for row in ora.itertuples(index=False):
        path = output_dir / row.primary_input
        target = set(path.read_text(encoding="utf-8").splitlines())
        if "" in target:
            target.remove("")
        if len(target) != int(row.target_lipids):
            raise AssertionError(f"Unexpected ORA target length: {path}")
        if not target.issubset(background_set):
            raise AssertionError(f"ORA target is not a background subset: {path}")
        targets.append(target)
    if set.union(*targets) != background_set or sum(map(len, targets)) != len(background_set):
        raise AssertionError("The five ORA targets do not partition the common background.")


def make_zip(output_dir: Path, paths: list[Path]) -> Path:
    archive_path = output_dir / "lion_web_upload_bundle.zip"
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths, key=lambda value: value.as_posix()):
            archive.write(path, arcname=path.relative_to(output_dir).as_posix())
    return archive_path


def run(cluster_dir: Path, mapping_path: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_dir = output_dir / "inputs" / "ranking"
    target_dir = output_dir / "inputs" / "target_list"
    audit_dir = output_dir / "audit"
    results_dir = output_dir / "results"
    for path in (ranking_dir, target_dir, audit_dir, results_dir):
        path.mkdir(parents=True, exist_ok=True)

    curves, aligned = load_aligned_data(cluster_dir)
    scores, _ = compute_pattern_scores(curves, aligned)
    lipid_scores = scores[scores["omics"].eq("Lipidomics")].copy()
    mapped = add_lion_mapping(lipid_scores, mapping_path)
    mapped = mapped.sort_values(["pattern_id", "feature"]).reset_index(drop=True)
    mapped_audit_path = audit_dir / "lipid_mapping_and_pattern_scores.csv"
    exclusions_path = audit_dir / "lion_mapping_exclusions.csv"
    mapped.to_csv(mapped_audit_path, index=False)
    mapped.loc[~mapped["lion_eligible"]].to_csv(exclusions_path, index=False)

    rank_source = mapped[mapped["lion_eligible"]].copy()
    ora_source = mapped[mapped["ora_eligible"]].copy()
    # Average only if multiple measured features share one LION composition
    # name.  No such duplicates occur in the current 93-lipid data set.
    ranking_background = sorted(rank_source["standard_name"].unique())
    ora_background = sorted(ora_source["standard_name"].unique())
    background_path = target_dir / "lion_all_patterns_background.txt"
    write_lines(ora_background, background_path)

    manifest_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    generated_paths: list[Path] = [background_path, mapped_audit_path, exclusions_path]
    pattern_labels = (
        aligned[["pattern_id", "pattern"]]
        .drop_duplicates()
        .set_index("pattern_id")["pattern"]
        .to_dict()
    )

    for pattern_id in EXPECTED_PATTERN_IDS:
        pattern = str(pattern_labels[pattern_id])
        slug = PATTERN_SLUGS[pattern_id]
        input_stem = f"lion_p{pattern_id:02d}_{slug}"
        score_column = f"p{pattern_id:02d}_trajectory_resemblance_score"

        ranked = (
            rank_source.groupby("standard_name", as_index=False)
            .agg(
                custom_statistic=(score_column, "mean"),
                source_feature_count=("feature", "nunique"),
                assigned_patterns=(
                    "pattern_id",
                    lambda values: ";".join(map(str, sorted(set(values)))),
                ),
            )
            .sort_values(["custom_statistic", "standard_name"], ascending=[False, True])
            .reset_index(drop=True)
        )
        ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
        ranking_txt = ranking_dir / f"{input_stem}_custom_statistic.txt"
        ranking_csv = ranking_dir / f"{input_stem}_custom_statistic_audit.csv"
        ranked.to_csv(ranking_csv, index=False)
        ranked[["standard_name", "custom_statistic"]].to_csv(
            ranking_txt,
            sep="\t",
            index=False,
            header=False,
            float_format="%.10g",
            lineterminator="\n",
        )

        target = sorted(
            ora_source.loc[ora_source["pattern_id"].eq(pattern_id), "standard_name"].unique()
        )
        rest = sorted(set(ora_background).difference(target))
        target_path = target_dir / f"{input_stem}_target.txt"
        rest_path = target_dir / f"{input_stem}_rest.txt"
        write_lines(target, target_path)
        write_lines(rest, rest_path)
        generated_paths.extend([ranking_txt, ranking_csv, target_path, rest_path])

        score_benchmark = (
            ora_source.groupby("standard_name", as_index=False)
            .agg(pattern_id=("pattern_id", "first"), score=(score_column, "mean"))
        )
        benchmark_target = score_benchmark["pattern_id"].eq(pattern_id)
        benchmark_rank = score_benchmark["score"].rank(method="min", ascending=False)
        ranking_auc = float(roc_auc_score(benchmark_target, score_benchmark["score"]))
        median_target_rank = float(benchmark_rank[benchmark_target].median())
        top_n_target_recall = float(
            (benchmark_rank[benchmark_target] <= int(benchmark_target.sum())).mean()
        )

        ranked_names = expected_names(pattern_id, "ranked")
        ora_names = expected_names(pattern_id, "ora")
        manifest_rows.append(
            {
                "run_order": pattern_id,
                "priority": "primary",
                "run_id": f"P{pattern_id:02d}-ranked",
                "pattern_id": pattern_id,
                "pattern": pattern,
                "mode": "ranked",
                "ranking_direction": "high_to_low",
                "analyze_all_lion_terms": True,
                "exclude_parent_with_same_child_associations": True,
                "send_unmatched_lipids_to_lion_team": False,
                "primary_input": ranking_txt.relative_to(output_dir).as_posix(),
                "background_input": "",
                "input_lipids": len(ranked),
                "target_lipids": "",
                "expected_result_csv": ranked_names["result_csv"],
                "expected_plot_svg": ranked_names["plot_svg"],
                "expected_network_svg": ranked_names["network_svg"],
            }
        )
        manifest_rows.append(
            {
                "run_order": 5 + pattern_id,
                "priority": "sensitivity",
                "run_id": f"P{pattern_id:02d}-ora",
                "pattern_id": pattern_id,
                "pattern": pattern,
                "mode": "ora",
                "ranking_direction": "not_applicable",
                "analyze_all_lion_terms": True,
                "exclude_parent_with_same_child_associations": True,
                "send_unmatched_lipids_to_lion_team": False,
                "primary_input": target_path.relative_to(output_dir).as_posix(),
                "background_input": background_path.relative_to(output_dir).as_posix(),
                "input_lipids": len(target),
                "target_lipids": len(target),
                "expected_result_csv": ora_names["result_csv"],
                "expected_plot_svg": ora_names["plot_svg"],
                "expected_network_svg": ora_names["network_svg"],
            }
        )
        summary_rows.append(
            {
                "pattern_id": pattern_id,
                "pattern": pattern,
                "raw_assigned_lipids": int(mapped["pattern_id"].eq(pattern_id).sum()),
                "mapped_assigned_lipids": int(
                    (mapped["pattern_id"].eq(pattern_id) & mapped["lion_eligible"]).sum()
                ),
                "ranked_background_lipids": len(ranking_background),
                "ora_target_lipids": len(target),
                "ora_background_lipids": len(ora_background),
                "ora_rest_lipids": len(rest),
                "ranking_auc_for_hard_assignment": ranking_auc,
                "median_rank_of_hard_assigned_lipids": median_target_rank,
                "top_n_recall_of_hard_assigned_lipids": top_n_target_recall,
            }
        )

    manifest = pd.DataFrame(manifest_rows).sort_values("run_order").reset_index(drop=True)
    summary = pd.DataFrame(summary_rows)
    manifest_path = output_dir / "lion_web_run_manifest.csv"
    summary_path = output_dir / "lion_input_summary.csv"
    manifest.to_csv(manifest_path, index=False)
    summary.to_csv(summary_path, index=False)
    generated_paths.extend([manifest_path, summary_path])

    run_log = manifest[
        ["run_order", "priority", "run_id", "pattern_id", "pattern", "mode", "expected_result_csv"]
    ].copy()
    run_log["run_date"] = ""
    run_log["lion_version_or_access_date"] = ""
    run_log["submitted_lipids"] = manifest["input_lipids"]
    run_log["recognized_lipids"] = ""
    run_log["unmatched_lipids"] = ""
    run_log["result_saved"] = False
    run_log["notes"] = ""
    run_log_path = output_dir / "lion_web_run_log.csv"
    run_log.to_csv(run_log_path, index=False)
    generated_paths.append(run_log_path)

    readme_path = output_dir / "README_LION_WEB_FIVE_PATTERNS.md"
    readme_path.write_text(build_readme(summary, manifest), encoding="utf-8", newline="\n")
    generated_paths.append(readme_path)

    expected_result_list = manifest[
        [
            "priority",
            "run_id",
            "expected_result_csv",
            "expected_plot_svg",
            "expected_network_svg",
        ]
    ]
    expected_result_path = results_dir / "EXPECTED_RESULT_FILENAMES.csv"
    expected_result_list.to_csv(expected_result_path, index=False)
    generated_paths.append(expected_result_path)

    validate_outputs(output_dir, manifest, ora_background)

    input_paths = [cluster_dir / "gam_curves.npz", cluster_dir / "gam_curve_metadata.csv",
                   cluster_dir / "molecule_cluster_assignments_agglomerative.csv", mapping_path]
    analysis_manifest = {
        "analysis": "LION/web inputs for five molecular-age patterns",
        "primary_mode": "ranked enrichment",
        "sensitivity_mode": "target-list ORA",
        "ranking_direction_for_all_patterns": "high_to_low",
        "ranking_score": (
            "nearest non-target morphology-centroid RMS distance minus "
            "target morphology-centroid RMS distance"
        ),
        "morphology_features": 14,
        "all_clustered_molecules_used_for_scaling": int(len(aligned)),
        "raw_lipids": int(len(mapped)),
        "lion_ranked_lipids": int(len(ranking_background)),
        "lion_ora_background_lipids": int(len(ora_background)),
        "excluded_mapping_statuses": sorted(EXCLUDED_MAPPING_STATUSES),
        "expected_lion_result_columns": list(LION_RESULT_COLUMNS),
        "input_sha256": {path.as_posix(): sha256(path) for path in input_paths},
    }
    analysis_manifest_path = output_dir / "analysis_manifest.json"
    analysis_manifest_path.write_text(
        json.dumps(analysis_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    generated_paths.append(analysis_manifest_path)

    checksum_path = output_dir / "input_checksums.sha256"
    checksum_lines = [
        f"{sha256(path)}  {path.relative_to(output_dir).as_posix()}"
        for path in generated_paths
        if path.exists()
    ]
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8", newline="\n")
    generated_paths.append(checksum_path)

    archive_path = make_zip(
        output_dir,
        [
            path
            for path in generated_paths
            if path.is_file() and path != output_dir / "lion_web_upload_bundle.zip"
        ],
    )
    return {
        "output_dir": str(output_dir),
        "archive": str(archive_path),
        "raw_lipids": int(len(mapped)),
        "ranked_lipids": int(len(ranking_background)),
        "ora_background_lipids": int(len(ora_background)),
        "excluded_lipids": int((~mapped["lion_eligible"]).sum()),
        "pattern_target_counts": {
            str(row.pattern_id): int(row.ora_target_lipids)
            for row in summary.itertuples(index=False)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-dir", type=Path, default=DEFAULT_CLUSTER_DIR)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.cluster_dir.resolve(), args.mapping.resolve(), args.output_dir.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
