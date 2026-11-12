#!/usr/bin/env python
"""Local side: redraw the crest-system lollipop figures from the exported minimum data.

Input (produced on the server by 05a-export_crest_lollipop_min_data.py)
    outputs/crest_lollipop_min_data.csv
    outputs/crest_lollipop_min_data.json   optional; if present its SHA-256 is checked first
Output (same files and names as 05-plot_crest_system_lollipop.py)
    figures/crest_system_association/crest_system_lollipop_<group>.pdf/.svg/.png
    and _source_data.csv

All plotting is done by make_figures() of 05-plot_crest_system_lollipop.py itself (imported,
not copied). Font: Arial (else Liberation Sans/Helvetica, as in 05-plot).

Usage
    python 05b-plot_crest_lollipop_from_min_data.py [--data CSV] [--figure-dir DIR]
                                                    [--groups group1 group2]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLOT_SCRIPT = HERE / "05-plot_crest_system_lollipop.py"


def load_plot_module():
    spec = importlib.util.spec_from_file_location("crest_lollipop_plot", PLOT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "make_figures"):
        sys.exit(f"{PLOT_SCRIPT} is an older version without make_figures(); copy the current "
                 "05-plot_crest_system_lollipop.py next to this script")
    return module


def check_manifest(data: Path) -> None:
    man = data.with_suffix(".json")
    if not man.exists():
        print(f"note: {man.name} not found; checksum not verified")
        return
    info = json.loads(man.read_text(encoding="utf-8"))
    if hashlib.sha256(data.read_bytes()).hexdigest() != info.get("sha256"):
        sys.exit(f"{data.name} does not match {man.name} (SHA-256 differs); "
                 "copy both files again from the server")
    print(f"data verified: {info['n_rows']} rows, {info['n_systems']} systems; "
          f"exported {info['created']} from {info['source']} (mtime {info['source_mtime']})")


def main() -> int:
    plot = load_plot_module()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=HERE / "outputs" / "crest_lollipop_min_data.csv")
    p.add_argument("--figure-dir", type=Path, default=HERE / "figures" / "crest_system_association")
    p.add_argument("--groups", nargs="+", choices=list(plot.GROUPS), default=list(plot.GROUPS))
    p.add_argument("--stem", default="crest_system_lollipop")
    a = p.parse_args()
    if not a.data.exists():
        sys.exit(f"{a.data} not found - run 05a-export_crest_lollipop_min_data.py on the server "
                 "and copy crest_lollipop_min_data.csv/.json here")
    check_manifest(a.data)
    return plot.make_figures(a.data, a.figure_dir, a.groups, a.stem)


if __name__ == "__main__":
    sys.exit(main())
