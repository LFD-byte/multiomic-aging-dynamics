"""Step 1b · Exploratory clustering of molecular age trajectories, k starting at 2.

Same pipeline as ``01-molecule_clustering.py`` (GAM curves 40–70 y, 14
morphology features, Ward / k-means), but scans k = 2…K_MAX instead of 5…9.
All functions are imported from the original script, so the computation is
identical; only the k range and the output location differ.

Isolation from the original results:
    * the original script and ``outputs/molecule_clustering/`` are never written;
    * cached GAM curves there are only *read* if present; otherwise curves are
      fitted and cached inside this script's own output directory.

Usage::

    python 01b-molecule_clustering_k_from2.py
    python 01b-molecule_clustering_k_from2.py --method kmeans
    python 01b-molecule_clustering_k_from2.py --k-max 12
    python 01b-molecule_clustering_k_from2.py --k-select 3   # export a chosen k

Outputs (``outputs/molecule_clustering_k_from2/``):
    cluster_validity_scores_{method}.csv        silhouette / CH / DB for each k
    per_k/molecule_cluster_assignments_{method}_k{k}.csv
    per_k/cluster_center_curves_{method}_k{k}.csv
    per_k/cluster_omics_composition_{method}_k{k}.csv
    molecule_cluster_assignments_{method}.csv   selected k (best silhouette or --k-select)
    cluster_center_curves_{method}.csv
    cluster_omics_composition_{method}.csv
    silhouette_scores_{method}.csv              same column format as the original
    analysis_manifest_{method}.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent

# Import the original script as a module (file name starts with a digit).
_spec = importlib.util.spec_from_file_location(
    "molecule_clustering_base", HERE / "01-molecule_clustering.py"
)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

ORIGINAL_DIR = base.OUTPUT_DIR                      # read-only here
OUTPUT_DIR = HERE / "outputs" / "molecule_clustering_k_from2"
PER_K_DIR = OUTPUT_DIR / "per_k"

K_MIN = 2
K_MAX_DEFAULT = 9
RANDOM_STATE = base.RANDOM_STATE
AGE_GRID = base.AGE_GRID


def load_or_fit_curves(force: bool = False) -> tuple[np.ndarray, pd.DataFrame, str]:
    """Reuse cached GAM curves without ever writing to the original folder."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for folder in (OUTPUT_DIR, ORIGINAL_DIR):
        curve_file = folder / "gam_curves.npz"
        meta_file = folder / "gam_curve_metadata.csv"
        if not force and curve_file.exists() and meta_file.exists():
            curves = np.load(curve_file)["curves"]
            metadata = pd.read_csv(meta_file)
            print(f"Loaded {len(curves):,} cached GAM curves from {folder}")
            return curves, metadata, str(folder)

    # Redirect the base module's cache location to our own folder before fitting.
    base.OUTPUT_DIR = OUTPUT_DIR
    curves, metadata = base.fit_all_curves(force=True)
    return curves, metadata, str(OUTPUT_DIR)


def scan_k(
    curves: np.ndarray, method: str, k_max: int
) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    morphology = np.vstack([base.extract_morphology(c) for c in curves])
    X = StandardScaler().fit_transform(morphology)

    labels_by_k: dict[int, np.ndarray] = {}
    rows: list[dict[str, float]] = []
    for k in range(K_MIN, k_max + 1):
        if method == "kmeans":
            model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50, max_iter=1000)
        else:
            model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(X)
        labels_by_k[k] = labels
        sizes = np.bincount(labels)
        row = {
            "k": k,
            "silhouette_score": silhouette_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
            "min_cluster_size": int(sizes.min()),
            "max_cluster_size": int(sizes.max()),
        }
        rows.append(row)
        print(
            f"k={k:>2}: silhouette={row['silhouette_score']:.4f}  "
            f"CH={row['calinski_harabasz']:.1f}  DB={row['davies_bouldin']:.4f}  "
            f"sizes {row['min_cluster_size']}–{row['max_cluster_size']}"
        )
    return labels_by_k, pd.DataFrame(rows)


def export_solution(
    curves: np.ndarray,
    metadata: pd.DataFrame,
    labels: np.ndarray,
    method: str,
    folder: Path,
    suffix: str = "",
) -> pd.DataFrame:
    table, info = base.build_cluster_table(curves, metadata, labels)
    table.to_csv(folder / f"molecule_cluster_assignments_{method}{suffix}.csv", index=False)

    centers = pd.DataFrame({"age": AGE_GRID})
    for x in info:
        centers[f'pattern_{x["pattern_id"]}'] = x["center"]
    centers.to_csv(folder / f"cluster_center_curves_{method}{suffix}.csv", index=False)

    comp = (
        table.groupby(["pattern_id", "pattern", "omics"], observed=True)
        .size().rename("n").reset_index()
    )
    comp["proportion"] = comp["n"] / comp.groupby("pattern_id")["n"].transform("sum")
    comp.to_csv(folder / f"cluster_omics_composition_{method}{suffix}.csv", index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=["agglomerative", "kmeans"], default="agglomerative")
    parser.add_argument("--k-max", type=int, default=K_MAX_DEFAULT)
    parser.add_argument("--k-select", type=int, default=None,
                        help="k to export as the main solution (default: best silhouette)")
    parser.add_argument("--force-refit", action="store_true",
                        help="refit GAM curves (cached in this script's own output folder)")
    args = parser.parse_args()
    if args.k_max < K_MIN:
        parser.error(f"--k-max must be >= {K_MIN}")

    curves, metadata, curve_source = load_or_fit_curves(force=args.force_refit)
    labels_by_k, scores = scan_k(curves, args.method, args.k_max)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PER_K_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUTPUT_DIR / f"cluster_validity_scores_{args.method}.csv", index=False)
    scores[["k", "silhouette_score"]].to_csv(
        OUTPUT_DIR / f"silhouette_scores_{args.method}.csv", index=False
    )

    for k, labels in labels_by_k.items():
        export_solution(curves, metadata, labels, args.method, PER_K_DIR, suffix=f"_k{k}")

    best_k = int(scores.loc[scores["silhouette_score"].idxmax(), "k"])
    selected_k = args.k_select if args.k_select is not None else best_k
    if selected_k not in labels_by_k:
        parser.error(f"--k-select {selected_k} is outside {K_MIN}…{args.k_max}")
    table = export_solution(curves, metadata, labels_by_k[selected_k], args.method, OUTPUT_DIR)

    manifest = {
        "based_on": "01-molecule_clustering.py",
        "age_range": [base.AGE_MIN, base.AGE_MAX],
        "n_age_grid": base.N_GRID,
        "n_splines": base.N_SPLINES,
        "cluster_method": args.method,
        "k_range": [K_MIN, args.k_max],
        "best_k_silhouette": best_k,
        "selected_k": selected_k,
        "n_molecules": int(len(curves)),
        "random_state": RANDOM_STATE,
        "gam_curve_source": curve_source,
    }
    (OUTPUT_DIR / f"analysis_manifest_{args.method}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"Best silhouette k={best_k}; exported k={selected_k}")
    print(table.groupby(["pattern_id", "pattern"]).size().rename("n").to_string())
    print(f"Saved exploratory results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
