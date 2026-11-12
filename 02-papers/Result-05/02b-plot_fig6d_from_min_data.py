#!/usr/bin/env python
"""Local side: redraw Fig. 6d (and the supplementary figure) from the exported minimum data.

Input (produced on the server by 02a-export_fig6d_min_data.py)
    outputs/fig6d_min_data.csv
    outputs/fig6d_min_data.json   optional; if present its SHA-256 is checked first
Output (same files and names as 02-plot_system_accel.py; default ./figures/system_accel)
    Fig6d_system_accel.pdf / .png, Fig6d_qc_report.txt,
    SuppFig_system_accel_other_systems.pdf / .png / _qc_report.txt

All plotting is done by make_figures() of 02-plot_system_accel.py itself (imported, not
copied), so the figure is the same as running 04 on the server; only the source path printed
in the QC report differs. Font: Arial must be installed (default on Windows); set FIG_FONT to
preview with another font.

Usage
    python 02b-plot_fig6d_from_min_data.py [--data outputs/fig6d_min_data.csv]
                                           [--out-dir DIR] [--systems S1 ... S6] [--top-n 10]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLOT_SCRIPT = HERE / "02-plot_system_accel.py"


def load_plot_module():
    spec = importlib.util.spec_from_file_location("fig6d_plot", PLOT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "make_figures"):
        sys.exit(f"{PLOT_SCRIPT} is an older version without make_figures(); copy the current "
                 "02-plot_system_accel.py next to this script")
    return module


def check_manifest(data: Path) -> None:
    man = data.with_suffix(".json")
    if not man.exists():
        print(f"note: {man.name} not found; checksum not verified")
        return
    info = json.loads(man.read_text(encoding="utf-8"))
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    if digest != info.get("sha256"):
        sys.exit(f"{data.name} does not match {man.name} (SHA-256 differs); "
                 "copy both files again from the server")
    print(f"data verified: {info['n_rows']} rows, {info['n_systems']} systems; "
          f"exported {info['created']} from {info['source']} (mtime {info['source_mtime']})")


def main() -> None:
    fig = load_plot_module()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=HERE / "outputs" / "fig6d_min_data.csv")
    p.add_argument("--out-dir", type=Path, default=HERE / "figures" / "system_accel")
    p.add_argument("--systems", nargs=6, default=fig.MAIN_SYSTEMS, metavar="SYSTEM")
    p.add_argument("--top-n", type=int, default=fig.TOP_N)
    a = p.parse_args()
    if not a.data.exists():
        sys.exit(f"{a.data} not found - run 02a-export_fig6d_min_data.py on the server and copy "
                 "fig6d_min_data.csv/.json here")
    check_manifest(a.data)
    fig.make_figures(a.data, a.systems, a.top_n, a.out_dir)


if __name__ == "__main__":
    main()
