"""Generate a small synthetic multi-omics dataset for the demo.

The data are simulated and contain no participant information. Each omics
layer is written as ``<layer>_with_metadata.parquet`` with the columns
``participant_id``, ``age``, ``sex`` and one column per feature, which is the
layout expected by the analysis scripts in ``02-papers``.

Usage:
    python demo/make_demo_data.py --out-dir demo/data/omics_data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# layer name -> (number of participants, number of features, microbiome?)
LAYERS = {
    "transcriptomics": (400, 200, False),
    "metabolomics": (300, 50, False),
    "lipidomics": (400, 40, False),
    "gut_microbiome": (500, 60, True),
    "oral_microbiome": (500, 30, True),
}


def _age_effect(age: np.ndarray, shape: str) -> np.ndarray:
    """Standardized age effect with a given trajectory shape (age 40-70)."""
    x = (age - 40.0) / 30.0
    if shape == "linear":
        return x
    if shape == "accelerating":
        return x**3
    if shape == "midlife_peak":
        return np.exp(-((age - 52.0) ** 2) / (2 * 4.0**2))
    return np.zeros_like(age)


def simulate_layer(n_participants: int, n_features: int, microbiome: bool,
                   rng: np.random.Generator) -> pd.DataFrame:
    participant_id = np.array([f"P{i:05d}" for i in range(n_participants)])
    age = rng.uniform(40.0, 70.0, n_participants)
    sex = rng.integers(0, 2, n_participants)

    # About 10% of participants have a second visit two years later.
    repeat = rng.random(n_participants) < 0.10
    participant_id = np.concatenate([participant_id, participant_id[repeat]])
    age = np.concatenate([age, np.minimum(age[repeat] + 2.0, 70.0)])
    sex = np.concatenate([sex, sex[repeat]])
    n_rows = len(age)

    # 30% of features change with age, with one of three trajectory shapes.
    shapes = rng.choice(["linear", "accelerating", "midlife_peak", "none"],
                        size=n_features, p=[0.12, 0.09, 0.09, 0.70])
    signs = rng.choice([-1.0, 1.0], size=n_features)
    values = np.empty((n_rows, n_features))
    for j in range(n_features):
        signal = 1.5 * signs[j] * _age_effect(age, shapes[j])
        values[:, j] = signal + 0.3 * sex + rng.normal(0.0, 1.0, n_rows)

    if microbiome:
        # Relative abundances with sparsity, as in MetaPhlAn output.
        values = np.exp(values)
        values[rng.random(values.shape) < 0.3] = 0.0
        values = 100.0 * values / values.sum(axis=1, keepdims=True)

    prefix = "taxon" if microbiome else "feature"
    frame = pd.DataFrame(values, columns=[f"{prefix}_{j:04d}" for j in range(n_features)])
    frame.insert(0, "sex", sex)
    frame.insert(0, "age", np.round(age, 2))
    frame.insert(0, "participant_id", participant_id)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("demo/data/omics_data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for layer, (n_participants, n_features, microbiome) in LAYERS.items():
        frame = simulate_layer(n_participants, n_features, microbiome, rng)
        path = args.out_dir / f"{layer}_with_metadata.parquet"
        frame.to_parquet(path, index=False)
        print(f"{layer:16s} rows={len(frame):4d} features={n_features:3d} -> {path}")


if __name__ == "__main__":
    main()
