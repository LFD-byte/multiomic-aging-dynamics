#!/usr/bin/env python
"""Server side: export the minimum data needed to redraw Fig. 6d locally.

Run on the server after 02-dose_response_forest.ipynb (or dose_response_forest.py).

Input
    outputs/panel_c_dose_response_effects.csv   (all fitted phenotypes, the BH-FDR family)
Output (copy these two small files to the local Result-05/outputs/)
    outputs/fig6d_min_data.csv    the 10 columns 02-plot_system_accel.py reads,
                                  for EVERY fitted phenotype; each value is copied as the
                                  original text (never parsed and re-printed), so 04 parses
                                  exactly the same numbers locally as on the server
    outputs/fig6d_min_data.json   provenance: source path/mtime/SHA-256, row counts, SHA-256 of
                                  the CSV (checked by 02b-plot_fig6d_from_min_data.py)

Why every row and not only the top 10 per system: the figure prints "N phenotypes tested"
per system and draws the supplementary systems, and the QC report uses the overall FDR
count and the n range across all tests. Only participant-free summary statistics are
exported; no individual-level data leave the server.

Usage
    python 02a-export_fig6d_min_data.py [--data-dir outputs] [--out outputs/fig6d_min_data.csv]
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
EFFECTS_CSV = "panel_c_dose_response_effects.csv"
# Exactly the columns load() in 02-plot_system_accel.py requires.
COLUMNS = ["system", "feature", "label", "outcome_type", "n",
           "effect", "ci_low", "ci_high", "p_value", "q_value_global"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=HERE / "outputs",
                   help=f"folder with {EFFECTS_CSV}")
    p.add_argument("--out", type=Path, default=HERE / "outputs" / "fig6d_min_data.csv")
    a = p.parse_args()

    src = a.data_dir / EFFECTS_CSV
    if not src.exists():
        sys.exit(f"{src} not found - run 02-dose_response_forest.ipynb first")
    # Read every field as text: a float -> text -> float round trip with pandas' default
    # parser can move a value by one ulp, so numbers are copied verbatim instead.
    text_kw = dict(dtype=str, keep_default_na=False, na_filter=False)
    d = pd.read_csv(src, **text_kw)
    if miss := [c for c in COLUMNS if c not in d.columns]:
        sys.exit(f"{src} lacks columns {miss}")
    if d.duplicated(["system", "feature"]).any():
        sys.exit(f"{src} has duplicated system/feature rows")

    out = d[COLUMNS]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)

    # Check 1: every exported field is the identical source text.
    back = pd.read_csv(a.out, **text_kw)
    pd.testing.assert_frame_equal(back, out.reset_index(drop=True), check_exact=True)
    # Check 2: parsed the way 04 parses it (pd.read_csv defaults), the export gives exactly
    # the same values as the source file.
    pd.testing.assert_frame_equal(pd.read_csv(a.out), pd.read_csv(src)[COLUMNS],
                                  check_exact=True)

    manifest = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "source": str(src.resolve()),
        "source_mtime": dt.datetime.fromtimestamp(src.stat().st_mtime).isoformat(timespec="seconds"),
        "source_sha256": sha256(src),
        "file": a.out.name,
        "sha256": sha256(a.out),
        "n_rows": len(out),
        "n_systems": int(out["system"].nunique()),
        "rows_per_system": {k: int(v) for k, v in out["system"].value_counts().sort_index().items()},
        "columns": COLUMNS,
    }
    man_path = a.out.with_suffix(".json")
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {a.out} ({a.out.stat().st_size / 1024:.1f} KB, {len(out)} rows, "
          f"{manifest['n_systems']} systems) and {man_path.name}")
    print("copy both files to the local Result-05/outputs/, then run "
          "python 02b-plot_fig6d_from_min_data.py")


if __name__ == "__main__":
    main()
