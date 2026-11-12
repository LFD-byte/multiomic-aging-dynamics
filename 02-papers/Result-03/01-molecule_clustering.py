"""Step 1 · Cluster nonlinear molecular age trajectories (computation only).

Per-molecule GAM fits (40–70 y, z-scaled) are summarised by 14 morphology
features and clustered with Ward linkage; k = 5…9 is selected by silhouette
score. Results are written to ``outputs/molecule_clustering/`` and are drawn by
``01-plot_molecule_clustering.py``. GAM curves are cached because fitting all
transcriptomic features is the expensive step.

Usage::

    python 01-molecule_clustering.py                    # agglomerative (Ward)
    python 01-molecule_clustering.py --method kmeans
    python 01-molecule_clustering.py --force-refit      # refit GAM curves

Outputs (``outputs/molecule_clustering/``):
    gam_curves.npz, gam_curve_metadata.csv
    silhouette_scores_{method}.csv
    molecule_cluster_assignments_{method}.csv
    cluster_center_curves_{method}.csv
    cluster_omics_composition_{method}.csv
    analysis_manifest.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pygam import LinearGAM, s
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "01-interim_data" / "omics_data"
OUTPUT_DIR = HERE / "outputs" / "molecule_clustering"

AGE_MIN, AGE_MAX = 40.0, 70.0
N_GRID = 100
MIN_SAMPLES = 50
N_SPLINES = 10
N_CLUSTERS_RANGE = range(5, 10)
RANDOM_STATE = 42

META_COLS = {
    "participant_id", "cohort", "research_stage", "array_index",
    "collection_date", "collection_timestamp", "collection_datetime",
    "timezone", "year_of_birth", "month_of_birth", "batch", "data_source",
    "hmo", "bmi", "sex", "age",
}

OMICS_FILES = {
    "Transcriptomics": "transcriptomics_dysregulated.parquet",
    "Metabolomics": "metabolomics_dysregulated.parquet",
    "Lipidomics": "lipidomics_dysregulated.parquet",
    "Gut microbiome": "gut_microbiome_dysregulated.parquet",
    "Oral microbiome": "oral_microbiome_dysregulated.parquet",
}
MICROBIOME_OMICS = {"Gut microbiome", "Oral microbiome"}

OMICS_ORDER = [
    "Transcriptomics", "Lipidomics", "Metabolomics",
    "Gut microbiome", "Oral microbiome",
]
AGE_GRID = np.linspace(AGE_MIN, AGE_MAX, N_GRID)


def zscore(x: np.ndarray) -> np.ndarray:
    """NaN-aware z-score; constant arrays become zero."""
    mean = np.nanmean(x)
    std = np.nanstd(x)
    return (x - mean) / std if np.isfinite(std) and std > 1e-10 else np.zeros_like(x)


def clr_transform(x: np.ndarray) -> np.ndarray:
    """Centered log-ratio transform with the original +1 pseudocount."""
    x = np.nan_to_num(x.astype(float), nan=0.0) + 1.0
    proportions = x / x.sum(axis=1, keepdims=True)
    logged = np.log(proportions)
    return logged - logged.mean(axis=1, keepdims=True)


def fit_gam(age: np.ndarray, values: np.ndarray) -> np.ndarray | None:
    mask = np.isfinite(age) & np.isfinite(values)
    if mask.sum() < MIN_SAMPLES:
        return None
    try:
        gam = LinearGAM(s(0, n_splines=N_SPLINES)).fit(
            age[mask, None], values[mask]
        )
        return gam.predict(AGE_GRID[:, None])
    except Exception as exc:  # one failed molecule should not abort the run
        warnings.warn(f"GAM fit skipped: {exc}", RuntimeWarning)
        return None


def extract_morphology(curve: np.ndarray) -> np.ndarray:
    """Return the 14 morphology features used in the original notebook."""
    n = len(curve)
    seg = n // 3
    d1 = np.gradient(curve)
    d2 = np.gradient(d1)
    d1_early, d1_mid, d1_late = (
        d1[:seg].mean(), d1[seg : 2 * seg].mean(), d1[2 * seg :].mean()
    )
    signs = np.sign(d1)
    sign_changes = np.sum(np.abs(np.diff(signs)) > 0) / n
    mu_early, mu_mid, mu_late = (
        curve[:seg].mean(), curve[seg : 2 * seg].mean(), curve[2 * seg :].mean()
    )
    return np.array([
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
    ])


def infer_pattern_label(center: np.ndarray) -> str:
    """Assign a concise descriptive label to a cluster center."""
    n = len(center)
    seg = n // 3
    d1 = np.gradient(center)
    overall = center[-1] - center[0]
    u_score = center[seg : 2 * seg].mean() - (
        center[:seg].mean() + center[2 * seg :].mean()
    ) / 2
    acceleration = d1[2 * seg :].mean() - d1[:seg].mean()
    early_flat = np.abs(d1[:seg]).mean() < 0.015
    late_steep = np.abs(d1[2 * seg :]).mean() > 0.025
    n_changes = np.sum(np.abs(np.diff(np.sign(d1))) > 0)

    # Strong endpoint-to-endpoint trends take precedence over local waviness.
    if overall > 1.5:
        return "Late accelerating rise" if acceleration > 0.05 else "Gradual rise"
    if overall < -1.5:
        return "Accelerating decline" if acceleration < -0.02 else "Gradual decline"
    if u_score < -0.18:
        return "Midlife trough"
    if u_score > 0.18:
        return "Midlife peak"
    if n_changes >= 4:
        return "Multiphase"
    if early_flat and late_steep and overall > 0.05:
        return "Late accelerating rise"
    if early_flat and late_steep and overall < -0.05:
        return "Late accelerating decline"
    if acceleration > 0.02 and overall > 0:
        return "Accelerating rise"
    if acceleration < -0.02 and overall < 0:
        return "Accelerating decline"
    if overall > 0.05:
        return "Gradual rise"
    if overall < -0.05:
        return "Gradual decline"
    return "Stable"


def fit_all_curves(force: bool = False) -> tuple[np.ndarray, pd.DataFrame]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    curve_file = OUTPUT_DIR / "gam_curves.npz"
    meta_file = OUTPUT_DIR / "gam_curve_metadata.csv"
    if curve_file.exists() and meta_file.exists() and not force:
        curves = np.load(curve_file)["curves"]
        metadata = pd.read_csv(meta_file)
        print(f"Loaded {len(curves):,} cached GAM curves")
        return curves, metadata

    curves: list[np.ndarray] = []
    records: list[dict[str, str]] = []
    for omics, filename in OMICS_FILES.items():
        path = DATA_DIR / filename
        df = pd.read_parquet(path)
        feature_cols = [c for c in df.columns if c not in META_COLS]
        keep = df["age"].between(AGE_MIN, AGE_MAX) & df["age"].notna()
        age = df.loc[keep, "age"].to_numpy(dtype=float)
        matrix = df.loc[keep, feature_cols].to_numpy(dtype=float)
        if omics in MICROBIOME_OMICS:
            matrix = clr_transform(matrix)

        print(f"Fitting {omics}: {len(feature_cols):,} molecules, {len(age):,} samples")
        successes = 0
        for j, feature in enumerate(feature_cols):
            curve = fit_gam(age, zscore(matrix[:, j]))
            if curve is not None and np.all(np.isfinite(curve)):
                curves.append(zscore(curve))
                records.append({"feature": feature, "omics": omics})
                successes += 1
            if (j + 1) % 250 == 0 or j + 1 == len(feature_cols):
                print(f"  {j + 1:>5,}/{len(feature_cols):,} processed; {successes:,} retained")
        del df, matrix

    curve_array = np.asarray(curves, dtype=np.float32)
    metadata = pd.DataFrame(records)
    np.savez_compressed(curve_file, curves=curve_array)
    metadata.to_csv(meta_file, index=False)
    print(f"Cached {len(curve_array):,} GAM curves")
    return curve_array, metadata


def cluster_curves(
    curves: np.ndarray, method: str = "agglomerative"
) -> tuple[np.ndarray, pd.DataFrame]:
    morphology = np.vstack([extract_morphology(c) for c in curves])
    morphology_scaled = StandardScaler().fit_transform(morphology)
    rows: list[dict[str, float]] = []
    best_labels: np.ndarray | None = None
    best_score = -np.inf
    best_k = -1

    for k in N_CLUSTERS_RANGE:
        if method == "kmeans":
            model = KMeans(
                n_clusters=k, random_state=RANDOM_STATE, n_init=50, max_iter=1000
            )
        else:
            model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(morphology_scaled)
        score = silhouette_score(morphology_scaled, labels)
        rows.append({"k": k, "silhouette_score": score})
        print(f"k={k}: silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    assert best_labels is not None
    scores = pd.DataFrame(rows)
    scores.to_csv(OUTPUT_DIR / f"silhouette_scores_{method}.csv", index=False)
    print(f"Selected k={best_k} (silhouette={best_score:.4f})")
    return best_labels, scores


def build_cluster_table(
    curves: np.ndarray, metadata: pd.DataFrame, labels: np.ndarray
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    # Largest cluster first gives stable, immediately interpretable panel order.
    cluster_ids = sorted(np.unique(labels), key=lambda cid: -np.sum(labels == cid))
    panel_lookup = {cid: i + 1 for i, cid in enumerate(cluster_ids)}
    out = metadata.copy()
    out["cluster_id"] = labels
    out["pattern_id"] = [panel_lookup[cid] for cid in labels]

    cluster_info: list[dict[str, object]] = []
    for cid in cluster_ids:
        mask = labels == cid
        members = curves[mask]
        center = members.mean(axis=0)
        cluster_info.append({
            "cluster_id": int(cid),
            "pattern_id": panel_lookup[cid],
            "members": members,
            "center": center,
            "omics": metadata.loc[mask, "omics"].to_numpy(),
            "n": int(mask.sum()),
            "label": infer_pattern_label(center),
        })
    label_lookup = {x["cluster_id"]: x["label"] for x in cluster_info}
    out["pattern"] = out["cluster_id"].map(label_lookup)
    return out, cluster_info


def save_results(
    curves: np.ndarray,
    cluster_table: pd.DataFrame,
    cluster_info: list[dict[str, object]],
    method: str,
) -> None:
    cluster_table.to_csv(OUTPUT_DIR / f"molecule_cluster_assignments_{method}.csv", index=False)
    centers = pd.DataFrame({"age": AGE_GRID})
    for info in cluster_info:
        centers[f'pattern_{info["pattern_id"]}'] = info["center"]
    centers.to_csv(OUTPUT_DIR / f"cluster_center_curves_{method}.csv", index=False)

    composition = (
        cluster_table.groupby(["pattern_id", "pattern", "omics"], observed=True)
        .size().rename("n").reset_index()
    )
    composition["proportion"] = composition["n"] / composition.groupby("pattern_id")["n"].transform("sum")
    composition.to_csv(OUTPUT_DIR / f"cluster_omics_composition_{method}.csv", index=False)

    manifest = {
        "age_range": [AGE_MIN, AGE_MAX],
        "n_age_grid": N_GRID,
        "n_splines": N_SPLINES,
        "cluster_method": method,
        "n_molecules": int(len(curves)),
        "n_clusters": int(len(cluster_info)),
        "random_state": RANDOM_STATE,
    }
    (OUTPUT_DIR / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=["agglomerative", "kmeans"], default="agglomerative")
    parser.add_argument("--force-refit", action="store_true")
    args = parser.parse_args()

    curves, metadata = fit_all_curves(force=args.force_refit)
    labels, _ = cluster_curves(curves, method=args.method)
    cluster_table, cluster_info = build_cluster_table(curves, metadata, labels)
    save_results(curves, cluster_table, cluster_info, args.method)
    print(cluster_table.groupby(["pattern_id", "pattern"]).size().rename("n").to_string())
    print(f"Saved clustering results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
