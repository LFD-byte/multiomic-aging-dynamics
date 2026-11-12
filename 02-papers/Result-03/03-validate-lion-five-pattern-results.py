#!/usr/bin/env python
"""Validate LION/web result files returned for the five lipid trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "outputs" / "lion_web_five_patterns"


def validate_one(path: Path) -> dict[str, object]:
    row: dict[str, object] = {
        "file": path.name,
        "exists": path.exists(),
        "valid": False,
        "rows": 0,
        "significant_fdr_0_05": 0,
        "message": "",
    }
    if not path.exists():
        row["message"] = "missing"
        return row

    try:
        table = pd.read_csv(path)
    except Exception as exc:
        row["message"] = f"could not read CSV: {exc}"
        return row

    description_column = next(
        (column for column in ("Discription", "Description") if column in table.columns),
        None,
    )
    required = {"Term ID", "Annotated", "p-value", "FDR q-value"}
    missing = required.difference(table.columns)
    if description_column is None:
        missing.add("Discription/Description")
    if missing:
        row["message"] = f"missing columns: {sorted(missing)}"
        return row
    if table.empty:
        row["message"] = "empty result table"
        return row

    p_value = pd.to_numeric(table["p-value"], errors="coerce")
    q_value = pd.to_numeric(table["FDR q-value"], errors="coerce")
    annotated = pd.to_numeric(table["Annotated"], errors="coerce")
    invalid_numeric = p_value.isna() | q_value.isna() | annotated.isna()
    invalid_probability = ~p_value.between(0, 1) | ~q_value.between(0, 1)
    blank_term = table["Term ID"].fillna("").astype(str).str.strip().eq("")
    if invalid_numeric.any() or invalid_probability.any() or blank_term.any():
        row["message"] = (
            f"invalid rows: numeric={int(invalid_numeric.sum())}, "
            f"probability={int(invalid_probability.sum())}, blank_term={int(blank_term.sum())}"
        )
        return row

    row["valid"] = True
    row["rows"] = int(len(table))
    row["significant_fdr_0_05"] = int((q_value < 0.05).sum())
    row["message"] = "ok"
    return row


def run(root: Path, require_ora: bool = False) -> dict[str, object]:
    manifest_path = root / "lion_web_run_manifest.csv"
    results_dir = root / "results"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    required_manifest = {"priority", "run_id", "mode", "expected_result_csv"}
    missing = required_manifest.difference(manifest.columns)
    if missing:
        raise ValueError(f"Run manifest is missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for expected in manifest.itertuples(index=False):
        result = validate_one(results_dir / expected.expected_result_csv)
        result.update(
            {
                "priority": expected.priority,
                "run_id": expected.run_id,
                "mode": expected.mode,
                "required": expected.priority == "primary" or require_ora,
            }
        )
        rows.append(result)

    validation = pd.DataFrame(rows)[
        [
            "priority",
            "run_id",
            "mode",
            "required",
            "file",
            "exists",
            "valid",
            "rows",
            "significant_fdr_0_05",
            "message",
        ]
    ]
    results_dir.mkdir(parents=True, exist_ok=True)
    validation_path = results_dir / "lion_result_validation.csv"
    validation.to_csv(validation_path, index=False)

    required = validation["required"].astype(bool)
    failed = required & ~validation["valid"].astype(bool)
    report = {
        "validation_file": str(validation_path),
        "required_runs": int(required.sum()),
        "valid_required_runs": int((required & validation["valid"]).sum()),
        "missing_or_invalid_required_runs": validation.loc[failed, "run_id"].tolist(),
        "optional_ora_files_present": int(
            (validation["priority"].eq("sensitivity") & validation["exists"]).sum()
        ),
        "status": "complete" if not failed.any() else "incomplete",
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--require-ora",
        action="store_true",
        help="Require the five sensitivity-analysis ORA result files as well.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args.root.resolve(), require_ora=args.require_ora)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
