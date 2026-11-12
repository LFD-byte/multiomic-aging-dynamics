#!/usr/bin/env python
"""Local side: redraw the cluster x system circle heatmap from the exported minimum data.

Input (produced on the server by 06a-export_circle_heatmap_min_data.py)
    outputs/circle_heatmap_min_data.csv
    outputs/circle_heatmap_min_data.json   optional; if present its SHA-256 is checked first
Output (same files and names as 06-cluster_system_circle_heatmap.ipynb)
    figures/cluster_system_circle_heatmap.png/.pdf/.svg and _feature_order.csv

Plotting is done by plot_circle_heatmap() of plot_cluster_system_circle_heatmap.py (imported,
not copied) with the parameters of the notebook below; keep them in step with the notebook.

Usage
    python 06b-plot_circle_heatmap_from_min_data.py [--data CSV] [--output-prefix PATH]
                                                    [--top-features-per-system 10]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from plot_cluster_system_circle_heatmap import plot_circle_heatmap  # noqa: E402

# Parameters of 06-cluster_system_circle_heatmap.ipynb
FIGURE_WIDTH_MM = 210
TOP_FEATURES_PER_SYSTEM = 10
EFFECT_COLUMN = "partial_r"
FDR_COLUMN = "fdr_global"
FDR_THRESHOLD = 0.05
VLIM = None
OUTPUT_PREFIX = HERE / "figures" / "cluster_system_circle_heatmap"


def check_manifest(data: Path) -> None:
    man = data.with_suffix(".json")
    if not man.exists():
        print(f"note: {man.name} not found; checksum not verified")
        return
    info = json.loads(man.read_text(encoding="utf-8"))
    if hashlib.sha256(data.read_bytes()).hexdigest() != info.get("sha256"):
        sys.exit(f"{data.name} does not match {man.name} (SHA-256 differs); "
                 "copy both files again from the server")
    print(f"data verified: {info['n_rows']} rows, {info['n_phenotypes']} phenotypes, "
          f"{info['n_clusters']} clusters; exported {info['created']} from {info['source']} "
          f"(mtime {info['source_mtime']})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=HERE / "outputs" / "circle_heatmap_min_data.csv")
    p.add_argument("--output-prefix", type=Path, default=OUTPUT_PREFIX)
    p.add_argument("--top-features-per-system", type=int, default=TOP_FEATURES_PER_SYSTEM,
                   help="0 shows every phenotype")
    a = p.parse_args()
    if not a.data.exists():
        sys.exit(f"{a.data} not found - run 06a-export_circle_heatmap_min_data.py on the server "
                 "and copy circle_heatmap_min_data.csv/.json here")
    check_manifest(a.data)
    associations = pd.read_csv(a.data)
    fig, order = plot_circle_heatmap(
        associations,
        effect_col=EFFECT_COLUMN,
        fdr_col=FDR_COLUMN,
        fdr_threshold=FDR_THRESHOLD,
        top_features_per_system=a.top_features_per_system,
        vlim=VLIM,
        figsize=(FIGURE_WIDTH_MM / 25.4, FIGURE_WIDTH_MM / 25.4),
        output_prefix=a.output_prefix,
    )
    plt.close(fig)
    print(f"Plotted {order['feature'].nunique():,} phenotypes from {order['system'].nunique()} "
          f"systems; saved {a.output_prefix}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
