#!/usr/bin/env python
"""Server side: export the minimum data needed to redraw the cluster x system circle heatmap
(06-cluster_system_circle_heatmap.ipynb) locally.

Run on the server after cluster_system_association.py / 06-cluster_system_association.ipynb.

Input
    outputs/cluster_system_association/cluster_system_associations.csv
Output (copy these two small files to the local Result-03/outputs/)
    outputs/circle_heatmap_min_data.csv   every association row, in the original row order
                                          (it fixes the angular order of phenotypes), with only
                                          the columns the heatmap uses: system, feature,
                                          cluster_id, partial_r, fdr_global, pattern_id; every
                                          value is copied as the original text
    outputs/circle_heatmap_min_data.json  provenance and SHA-256 (checked by 06b)

All rows are kept (not only the top 10 per system) because the top-10 choice is made by the
plotting function from every phenotype of a system, so TOP_FEATURES_PER_SYSTEM can still be
changed locally. Before writing, the script checks that prepare_circle_data() of
plot_cluster_system_circle_heatmap.py returns exactly the same table from the export as from
the full source, both with the top-10 rule and with all phenotypes.
Only summary statistics are exported; no individual-level data.

Usage
    python 06a-export_circle_heatmap_min_data.py [--input CSV] [--out CSV]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "outputs" / "cluster_system_association" / "cluster_system_associations.csv"
EFFECT_COL, FDR_COL = "partial_r", "fdr_global"            # as in the notebook
REQUIRED = ["system", "feature", "cluster_id", EFFECT_COL, FDR_COL]
OPTIONAL = ["pattern_id"]                                    # orders and names the rings
TEXT_KW = dict(dtype=str, keep_default_na=False, na_filter=False)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=SOURCE)
    p.add_argument("--out", type=Path, default=HERE / "outputs" / "circle_heatmap_min_data.csv")
    a = p.parse_args()

    if not a.input.exists():
        sys.exit(f"{a.input} not found - run the cluster-system association first")
    d = pd.read_csv(a.input, **TEXT_KW)
    if miss := [c for c in REQUIRED if c not in d.columns]:
        sys.exit(f"{a.input} lacks columns {miss}")
    columns = REQUIRED + [c for c in OPTIONAL if c in d.columns]
    out = d[columns]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)

    # Check 1: every exported field is the identical source text, in the same row order.
    pd.testing.assert_frame_equal(pd.read_csv(a.out, **TEXT_KW), out, check_exact=True)
    # Check 2: the plotting function selects and orders exactly the same data from both files.
    sys.path.insert(0, str(HERE))
    from plot_cluster_system_circle_heatmap import prepare_circle_data
    full, slim = pd.read_csv(a.input), pd.read_csv(a.out)
    for top in (10, 0):
        kw = dict(effect_col=EFFECT_COL, fdr_col=FDR_COL, top_features_per_system=top)
        from_slim = prepare_circle_data(slim, **kw)[0]
        from_full = prepare_circle_data(full, **kw)[0]
        # the full table may carry extra columns ("pattern", "n") that the drawing never uses
        pd.testing.assert_frame_equal(from_slim, from_full[from_slim.columns], check_exact=True)

    manifest = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "source": str(a.input.resolve()),
        "source_mtime": dt.datetime.fromtimestamp(a.input.stat().st_mtime).isoformat(timespec="seconds"),
        "source_sha256": sha256(a.input),
        "file": a.out.name,
        "sha256": sha256(a.out),
        "n_rows": len(out),
        "n_systems": int(out["system"].nunique()),
        "n_phenotypes": int(out[["system", "feature"]].drop_duplicates().shape[0]),
        "n_clusters": int(out["cluster_id"].nunique()),
        "columns": columns,
    }
    man = a.out.with_suffix(".json")
    man.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {a.out} ({a.out.stat().st_size / 1024:.1f} KB, {len(out)} rows, "
          f"{manifest['n_phenotypes']} phenotypes, {manifest['n_clusters']} clusters) and {man.name}")
    print("copy both files to the local Result-03/outputs/, then run "
          "python 06b-plot_circle_heatmap_from_min_data.py")


if __name__ == "__main__":
    main()
