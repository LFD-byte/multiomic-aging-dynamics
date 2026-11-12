#!/usr/bin/env python
"""Server side: export the minimum data needed to redraw the crest-system lollipop figures locally.

Run on the server after 05-crest_system_association.py.

Input
    outputs/05-crest_system_association/top10_plot_data.csv
Output (copy these two small files to the local Result-02/outputs/)
    outputs/crest_lollipop_min_data.csv   only the rows drawn (selected_top10) and the six
                                          columns 05-plot_crest_system_lollipop.py reads; every
                                          value is copied as the original text (never parsed
                                          and re-printed), so the plot parses identical numbers
    outputs/crest_lollipop_min_data.json  provenance and SHA-256 (checked by 05b)

Before writing, the script checks that load_table() of 05-plot_crest_system_lollipop.py returns
exactly the same plotting table from the export as from the full source file.
Only proportions per indicator and crest are exported; no individual-level data.

Usage
    python 05a-export_crest_lollipop_min_data.py [--input CSV] [--out CSV]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PLOT_SCRIPT = HERE / "05-plot_crest_system_lollipop.py"
SOURCE = HERE / "outputs" / "05-crest_system_association" / "top10_plot_data.csv"
COLUMNS = ["system", "indicator_id", "indicator_label", "rank_within_system", "crest", "proportion"]
TEXT_KW = dict(dtype=str, keep_default_na=False, na_filter=False)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plot_module():
    spec = importlib.util.spec_from_file_location("crest_lollipop_plot", PLOT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=SOURCE)
    p.add_argument("--out", type=Path, default=HERE / "outputs" / "crest_lollipop_min_data.csv")
    a = p.parse_args()

    if not a.input.exists():
        sys.exit(f"{a.input} not found - run 05-crest_system_association.py first")
    d = pd.read_csv(a.input, **TEXT_KW)
    if miss := [c for c in COLUMNS if c not in d.columns]:
        sys.exit(f"{a.input} lacks columns {miss}")
    if "selected_top10" in d.columns:          # same rule as load_table()
        d = d[d["selected_top10"].str.lower().isin(["true", "1"])]
    out = d[COLUMNS].reset_index(drop=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)

    # Check 1: every exported field is the identical source text.
    pd.testing.assert_frame_equal(pd.read_csv(a.out, **TEXT_KW), out, check_exact=True)
    # Check 2: the plotting script builds exactly the same table from both files.
    plot = load_plot_module()
    pd.testing.assert_frame_equal(plot.load_table(a.out), plot.load_table(a.input), check_exact=True)

    manifest = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "source": str(a.input.resolve()),
        "source_mtime": dt.datetime.fromtimestamp(a.input.stat().st_mtime).isoformat(timespec="seconds"),
        "source_sha256": sha256(a.input),
        "file": a.out.name,
        "sha256": sha256(a.out),
        "n_rows": len(out),
        "n_systems": int(out["system"].nunique()),
        "rows_per_system": {k: int(v) for k, v in out["system"].value_counts().sort_index().items()},
        "columns": COLUMNS,
    }
    man = a.out.with_suffix(".json")
    man.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {a.out} ({a.out.stat().st_size / 1024:.1f} KB, {len(out)} rows, "
          f"{manifest['n_systems']} systems) and {man.name}")
    print("copy both files to the local Result-02/outputs/, then run "
          "python 05b-plot_crest_lollipop_from_min_data.py")


if __name__ == "__main__":
    main()
