"""Relate significant molecules at DE-SWAN main peaks to molecular trajectories.

Main peaks (is_main == TRUE) are ordered by age separately within each omics.
Crest 1 is the union of significant molecules at each omic's first main peak;
Crest 2 is the corresponding union at second main peaks, where present.
The is_stable_main flag does not determine inclusion.

For the default Result-02/outputs/deswan-window_7.5 input, the first main peaks
are lipidomics 48, metabolomics 48, oral microbiome 48, gut microbiome 50 and
transcriptomics 52 years. Only transcriptomics has a second main peak, at 64.
Membership uses the exact peak-age DE-SWAN test for the corresponding omics,
with q < q_cutoff after the upstream per-window FDR correction (BH, q < 0.05
for the default input). Peak widths and median breakpoint ages are not used.

Rscript with DEswan reads the existing RDS results without refitting models.
Use --rscript to select an R environment; otherwise the project's VS Code R
configuration, R_HOME and PATH are searched. Source hashes and per-peak
significant counts are checked against the Result-02 peak tables/settings.

The script writes two publication outputs: a main figure containing the
seven-set UpSet plot and a compact 5 x 3 mutually exclusive overlap heatmap,
and a supplementary figure containing five complete three-set Venn diagrams.
Machine-readable audit tables accompany both figures.
Molecule identity is always the composite (omics, feature), preventing names
shared by different omics from being merged accidentally.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DEFAULT_DESWAN_DIR = (
    PROJECT / "02-papers" / "Result-02" / "outputs" / "deswan-window_7.5"
)
DEFAULT_CLUSTER_FILE = (
    HERE / "outputs" / "molecule_clustering"
    / "molecule_cluster_assignments_agglomerative.csv"
)
DEFAULT_OUTPUT_DIR = HERE / "outputs" / "crest_cluster_intersections"
DEFAULT_FIGURE_DIR = HERE / "figures"

EARLY_COLOR = "#169BC5"
LATE_COLOR = "#DD8A3C"
CLUSTER_COLOR = "#344A50"
INACTIVE_COLOR = "#C4C8CB"
ROW_COLOR = "#F1F0FA"
TEXT_COLOR = "#20262B"
GRID_COLOR = "#D9DDE0"
VENN_COLORS = ("#CF684E", "#E9937E", "#F0B29F")

CLUSTERS = tuple(range(1, 6))
SET_COLUMNS = ["crest_1", "crest_2"] + [f"cluster_{cluster}" for cluster in CLUSTERS]
SET_LABELS = {
    "crest_1": "Crest 1",
    "crest_2": "Crest 2",
    "cluster_1": "Cluster 1",
    "cluster_2": "Cluster 2",
    "cluster_3": "Cluster 3",
    "cluster_4": "Cluster 4",
    "cluster_5": "Cluster 5",
}
CLUSTER_NAMES = {
    1: "Late accelerating rise",
    2: "Gradual rise",
    3: "Accelerating decline",
    4: "Midlife trough",
    5: "Midlife peak",
}
HEATMAP_CATEGORIES = (
    ("crest_1_only", "Crest 1 only"),
    ("crest_2_only", "Crest 2 only"),
    ("both_crests", "Crest 1 & Crest 2"),
)


def canonical_omics(value: object) -> str:
    """Normalize an omics label while preserving distinct molecular layers."""
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    token = "_".join(piece for piece in token.split("_") if piece)
    aliases = {
        "transcriptomic": "transcriptomics",
        "rna": "transcriptomics",
        "metabolomic": "metabolomics",
        "lipidomic": "lipidomics",
        "gut_microbiota": "gut_microbiome",
        "oral_microbiota": "oral_microbiome",
    }
    return aliases.get(token, token)


def molecule_key(omics: object, feature: object) -> str:
    return f"{canonical_omics(omics)}::{str(feature).strip()}"


def discover_deswan_tables(deswan_dir: Path, table: str) -> list[tuple[Path, str]]:
    """Find current and legacy tables, rejecting ambiguous per-omics inputs."""
    window_marker = f"_{table}_window_"
    legacy_prefix = f"panel_c_{table}_"
    current_pattern = f"*{window_marker}*.csv"
    legacy_pattern = f"{legacy_prefix}*.csv"
    paths = sorted(set(deswan_dir.glob(current_pattern)) | set(deswan_dir.glob(legacy_pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No {current_pattern} or {legacy_pattern} files in {deswan_dir}"
        )
    tables = []
    seen: dict[str, Path] = {}
    for path in paths:
        if path.stem.startswith(legacy_prefix):
            source_omics = path.stem.removeprefix(legacy_prefix)
        else:
            source_omics = path.stem.rsplit(window_marker, 1)[0]
        omics = canonical_omics(source_omics)
        if omics in seen:
            raise ValueError(
                f"Multiple {table} tables for {omics}: {seen[omics].name}, {path.name}. "
                "Use a directory containing one table per omics and window size."
            )
        seen[omics] = path
        tables.append((path, omics))
    return tables


def crest_peak_labels(crests: pd.DataFrame) -> dict[str, str]:
    """List the discrete peak ages; these labels never describe an interval."""
    return {
        row.set_name: ", ".join(
            f"{age:g}" for age in sorted({float(age) for age in row.peak_ages.split(";")})
        ) + " y"
        for row in crests.itertuples(index=False)
    }


def heatmap_categories(crests: pd.DataFrame) -> tuple[tuple[str, str], ...]:
    peak_labels = crest_peak_labels(crests)
    return tuple(
        (category, f"{label}\n{peak_labels[category.removesuffix('_only')]}"
         if category.endswith("_only") else label)
        for category, label in HEATMAP_CATEGORIES
    )


def select_crest_peaks(deswan_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank main peaks within each omics and retain its first and second peaks."""
    pieces = []
    for path, source_omics in discover_deswan_tables(deswan_dir, "crest"):
        frame = pd.read_csv(path)
        required = {"peak_age", "is_main", "raw_n_at_peak"}
        if missing := required - set(frame.columns):
            raise ValueError(f"{path.name} lacks main-peak columns: {sorted(missing)}")
        frame = frame.copy()
        main_flags = frame["is_main"].astype(str).str.strip().str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
        if main_flags.isna().any():
            raise ValueError(f"{path.name} contains missing or invalid is_main flags")
        frame["is_main"] = main_flags.astype(bool)
        frame["peak_age"] = pd.to_numeric(frame["peak_age"], errors="raise")
        if not np.isfinite(frame["peak_age"]).all():
            raise ValueError(f"{path.name} contains non-finite peak ages")
        frame["omics"] = source_omics
        frame["source_file"] = path.name
        pieces.append(frame)

    audit = (
        pd.concat(pieces, ignore_index=True, sort=False)
        .sort_values(["omics", "peak_age"], kind="stable")
        .reset_index(drop=True)
    )
    main_peaks = audit.loc[audit["is_main"]]
    if main_peaks.duplicated(["omics", "peak_age"]).any():
        raise ValueError("Duplicate main-peak ages within an omics")
    audit["main_peak_rank"] = pd.Series(pd.NA, index=audit.index, dtype="Int64")
    audit.loc[main_peaks.index, "main_peak_rank"] = (
        main_peaks.groupby("omics", sort=False).cumcount() + 1
    )
    audit["retained_set"] = audit["main_peak_rank"].map({1: "crest_1", 2: "crest_2"})
    audit["selection_reason"] = np.where(
        ~audit["is_main"], "not_main",
        np.where(audit["retained_set"].notna(), "selected", "after_second_main_peak"),
    )
    selected = audit.loc[audit["retained_set"].notna()]
    if set(selected["retained_set"]) != {"crest_1", "crest_2"}:
        raise ValueError("Need first main peaks and at least one second main peak for the two crest sets")

    definitions = []
    for rank, part in selected.groupby("main_peak_rank", sort=True):
        ordinal = "first" if rank == 1 else "second"
        definitions.append(
            {
                "crest_id": int(rank),
                "set_name": f"crest_{rank}",
                "definition": f"Union of significant molecules at each omic's {ordinal} main peak",
                "membership_rule": "q < source q_cutoff at the exact omics-specific peak age",
                "n_omics": int(part["omics"].nunique()),
                "n_peaks": int(len(part)),
                "supporting_omics": ";".join(sorted(part["omics"].unique())),
                "peak_ages": ";".join(f"{age:g}" for age in sorted(part["peak_age"])),
                "omics_peak_ages": ";".join(
                    f"{row.omics}:{row.peak_age:g}" for row in part.itertuples(index=False)
                ),
            }
        )
    return pd.DataFrame(definitions), audit


def resolve_rscript(requested: Path | None = None) -> Path:
    if requested is not None:
        resolved = shutil.which(str(requested)) or requested
        if not Path(resolved).is_file():
            raise FileNotFoundError(f"Rscript not found: {requested}")
        return Path(resolved).resolve()
    candidates = []
    settings_path = PROJECT / ".vscode" / "settings.json"
    if settings_path.is_file():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8-sig"))
            configured_r = settings.get("r.rterm.windows")
            if configured_r:
                candidates.append(Path(configured_r).with_name("Rscript.exe"))
        except (ValueError, OSError):
            pass
    if r_home := os.environ.get("R_HOME"):
        candidates.extend(Path(r_home) / tail for tail in (
            "bin/Rscript", "bin/Rscript.exe", "bin/x64/Rscript.exe",
        ))
    if on_path := shutil.which("Rscript"):
        candidates.append(Path(on_path))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Rscript is required to read the DE-SWAN RDS results. "
        "Use --rscript with an R environment containing DEswan."
    )


def rscript_environment(rscript: Path) -> dict[str, str]:
    """Supply the selected R/Conda runtime's DLL paths for the child process."""
    env = os.environ.copy()
    for parent in rscript.parents:
        if (parent / "etc").is_dir() and (parent / "library").is_dir():
            env["R_HOME"] = str(parent)
        if os.name == "nt" and (parent / "conda-meta").is_dir():
            runtime_dirs = [parent, *(parent / tail for tail in (
                "Library/mingw-w64/bin", "Library/usr/bin", "Library/bin", "Scripts",
            ))]
            env["PATH"] = os.pathsep.join(map(str, runtime_dirs)) + os.pathsep + env.get("PATH", "")
            break
    if os.name == "nt":
        env["LC_ALL"] = "C"
    return env


R_PEAK_EXTRACTOR = r'''
args <- commandArgs(trailingOnly = TRUE)
if (!requireNamespace("DEswan", quietly = TRUE)) {
  stop("The selected R environment needs the DEswan package.")
}
requests <- read.csv(args[1L], stringsAsFactors = FALSE, check.names = FALSE)
rows <- list()
for (source in unique(requests$source_rds)) {
  result <- readRDS(source)
  required <- c("variable", "window.center", "factor", "pvalue")
  if (!is.list(result) || !is.data.frame(result$p) || !all(required %in% names(result$p))) {
    stop("Invalid DE-SWAN p-value table: ", source)
  }
  p <- result$p[result$p$factor == "qt", , drop = FALSE]
  if (!nrow(p) || anyDuplicated(p[, c("variable", "window.center")]) ||
      any(!is.na(p$pvalue) & (!is.finite(p$pvalue) | p$pvalue < 0 | p$pvalue > 1))) {
    stop("Invalid or duplicate feature/age qt p-values: ", source)
  }
  p_wide <- DEswan::reshape.DEswan(result, parameter = 1, factor = "qt")
  ages <- as.numeric(sub("^X", "", names(p_wide)[-1L]))
  if (any(!is.finite(ages)) || anyDuplicated(ages)) stop("Invalid RDS window ages")
  for (i in which(requests$source_rds == source)) {
    request <- requests[i, , drop = FALSE]
    column <- which(abs(ages - request$peak_age) < 1e-8)
    if (length(column) != 1L) stop("Exact peak age missing in RDS: ", request$omics, " ", request$peak_age)
    # Use the same per-window correction and full feature family as Result-02.
    q_wide <- DEswan::q.DEswan(p_wide, method = request$fdr_method)
    rows[[length(rows) + 1L]] <- data.frame(
      set_name = request$set_name, omics = request$omics, peak_age = request$peak_age,
      feature = as.character(p_wide$variable), p_value = p_wide[[column + 1L]],
      q_value = q_wide[[column + 1L]], stringsAsFactors = FALSE
    )
  }
}
write.csv(do.call(rbind, rows), args[2L], row.names = FALSE, na = "", fileEncoding = "UTF-8")
'''


def load_peak_significance(
    deswan_dir: Path, crest_audit: pd.DataFrame, rscript: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read exact-age q-values, checking provenance and upstream peak counts."""
    selected = crest_audit.loc[crest_audit["retained_set"].notna()].copy()
    selected = selected.rename(columns={"retained_set": "set_name"}).reset_index(drop=True)
    settings_files = {omics: path for path, omics in discover_deswan_tables(deswan_dir, "peak_settings")}
    for omics, indices in selected.groupby("omics").groups.items():
        if omics not in settings_files:
            raise FileNotFoundError(f"No peak_settings table for {omics}")
        settings = pd.read_csv(settings_files[omics])
        required = {"q_cutoff", "fdr_method", "source_rds", "source_md5"}
        if len(settings) != 1 or not required <= set(settings.columns):
            raise ValueError(f"{settings_files[omics].name} requires one row with {sorted(required)}")
        source = settings.iloc[0]
        source_name = Path(str(source.source_rds).replace("\\", "/")).name
        source_path = (deswan_dir / source_name).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"DE-SWAN RDS missing: {source_path}")
        digest = hashlib.md5()
        with source_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != str(source.source_md5).lower():
            raise ValueError(f"RDS hash differs from the peak settings for {omics}; regenerate Result-02 peak tables")
        cutoff = float(source.q_cutoff)
        if not np.isfinite(cutoff) or not 0 < cutoff <= 1:
            raise ValueError(f"Invalid q_cutoff for {omics}: {cutoff}")
        for column, value in {
            "q_cutoff": cutoff, "fdr_method": source.fdr_method,
            "source_rds": str(source_path), "source_md5": digest.hexdigest(),
            "source_settings_file": settings_files[omics].name,
        }.items():
            selected.loc[indices, column] = value

    executable = resolve_rscript(rscript)
    with tempfile.TemporaryDirectory(prefix="crest_peak_tests_") as temporary:
        folder = Path(temporary)
        request_file, result_file = folder / "requests.csv", folder / "peak_tests.csv"
        helper_file = folder / "extract_peak_tests.R"
        selected[["set_name", "omics", "peak_age", "source_rds", "fdr_method"]].to_csv(request_file, index=False)
        helper_file.write_text(R_PEAK_EXTRACTOR, encoding="utf-8")
        result = subprocess.run(
            [str(executable), "--vanilla", str(helper_file), str(request_file), str(result_file)],
            env=rscript_environment(executable), capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False,
        )
        if result.returncode:
            raise RuntimeError(f"RDS peak extraction failed (exit {result.returncode}):\n{result.stderr}\n{result.stdout}")
        tests = pd.read_csv(result_file, dtype={"feature": "string"}, keep_default_na=False)

    tests["feature"] = tests["feature"].str.strip()
    if tests["feature"].eq("").any():
        raise ValueError("Empty feature identifiers in DE-SWAN peak tests")
    for column in ("p_value", "q_value"):
        tests[column] = pd.to_numeric(tests[column].replace("", np.nan), errors="raise")
    keys = ["set_name", "omics", "peak_age"]
    tests = tests.merge(
        selected[keys + ["main_peak_rank", "q_cutoff", "fdr_method", "source_rds", "source_md5"]],
        on=keys, how="left", validate="many_to_one",
    )
    if tests["q_cutoff"].isna().any() or tests.duplicated(keys + ["feature"]).any():
        raise ValueError("Unexpected or duplicate feature/peak tests from RDS")
    tests["significant"] = tests["q_value"].lt(tests["q_cutoff"])
    counts = tests.groupby(keys, as_index=False).agg(
        n_tested=("q_value", "count"), n_significant=("significant", "sum"),
    )
    selected = selected.merge(counts, on=keys, how="left", validate="one_to_one")
    expected = pd.to_numeric(selected["raw_n_at_peak"], errors="raise")
    matches = selected["n_significant"].eq(expected)
    if not matches.all():
        mismatch = selected.loc[~matches, keys + ["raw_n_at_peak", "n_significant"]]
        raise ValueError("Exact-age significant counts differ from Result-02 peaks:\n" + mismatch.to_string(index=False))
    selected["count_matches_peak"] = matches
    significant = tests.loc[tests["significant"]].drop(columns="significant").copy()
    significant["molecule_id"] = [
        molecule_key(o, f) for o, f in zip(significant["omics"], significant["feature"])
    ]
    return significant, selected


def load_cluster_assignments(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"feature", "omics", "pattern_id"}
    if missing := required - set(frame.columns):
        raise ValueError(f"{path.name} lacks columns: {sorted(missing)}")
    frame = frame.copy()
    frame["omics"] = frame["omics"].map(canonical_omics)
    frame["feature"] = frame["feature"].astype(str).str.strip()
    frame["pattern_id"] = pd.to_numeric(frame["pattern_id"], errors="raise").astype(int)
    frame["molecule_id"] = [
        molecule_key(o, f) for o, f in zip(frame["omics"], frame["feature"])
    ]
    conflicts = frame.groupby("molecule_id")["pattern_id"].nunique()
    if (conflicts > 1).any():
        examples = conflicts[conflicts > 1].index[:5].tolist()
        raise ValueError(f"Molecules assigned to multiple clusters: {examples}")
    return frame.drop_duplicates("molecule_id", keep="first")


def build_membership(
    peak_molecules: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Union the significant molecules at the selected omics-specific peaks."""
    crest_sets = {
        name: set(
            peak_molecules.loc[peak_molecules["set_name"].eq(name), "molecule_id"]
        )
        for name in ("crest_1", "crest_2")
    }
    cluster_sets = {
        f"cluster_{cluster}": set(
            assignments.loc[assignments["pattern_id"].eq(cluster), "molecule_id"]
        )
        for cluster in CLUSTERS
    }
    sets = {**crest_sets, **cluster_sets}
    universe = sorted(set().union(*sets.values()))

    assignment_lookup = assignments.set_index("molecule_id")
    peak_lookup = (
        peak_molecules[["molecule_id", "omics", "feature"]]
        .drop_duplicates("molecule_id")
        .set_index("molecule_id")
    )
    identity = assignment_lookup[["omics", "feature"]].combine_first(peak_lookup)
    membership = identity.reindex(universe).reset_index(names="molecule_id")
    membership["pattern_id"] = membership["molecule_id"].map(
        assignment_lookup["pattern_id"]
    ).astype("Int64")
    if "pattern" in assignment_lookup:
        membership["pattern"] = membership["molecule_id"].map(
            assignment_lookup["pattern"]
        )
    for name in SET_COLUMNS:
        membership[name] = membership["molecule_id"].isin(sets[name])

    peak_ids = set(peak_molecules["molecule_id"])
    cluster_ids = set(assignments["molecule_id"])
    diagnostics = pd.DataFrame(
        [
            {
                "metric": "unique_peak_significant_molecules",
                "value": len(peak_ids),
            },
            {
                "metric": "unique_clustered_molecules",
                "value": len(cluster_ids),
            },
            {
                "metric": "peak_molecules_with_cluster_assignment",
                "value": len(peak_ids & cluster_ids),
            },
            {
                "metric": "peak_molecules_without_cluster_assignment",
                "value": len(peak_ids - cluster_ids),
            },
            {
                "metric": "clustered_molecules_without_selected_peak_significance",
                "value": len(cluster_ids - peak_ids),
            },
        ]
    )
    return membership, diagnostics


def exact_intersections(membership: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        membership.groupby(SET_COLUMNS, observed=True, dropna=False)
        .size()
        .rename("intersection_size")
        .reset_index()
    )
    grouped = grouped.loc[grouped["intersection_size"].gt(0)].copy()
    grouped["degree"] = grouped[SET_COLUMNS].sum(axis=1).astype(int)
    grouped["combination"] = grouped.apply(
        lambda row: ";".join(name for name in SET_COLUMNS if bool(row[name])), axis=1
    )
    return grouped.sort_values(
        ["intersection_size", "degree"], ascending=[False, False], ignore_index=True
    )


def display_intersections(table: pd.DataFrame, max_columns: int = 20) -> pd.DataFrame:
    """Put the five crest1/crest2/cluster triple intersections first."""
    priority_combinations = [
        f"crest_1;crest_2;cluster_{cluster}" for cluster in CLUSTERS
    ]
    priority = table.loc[table["combination"].isin(priority_combinations)].copy()
    priority["priority"] = priority["combination"].map(
        {name: index for index, name in enumerate(priority_combinations)}
    )
    priority = priority.sort_values("priority").drop(columns="priority")
    remainder = table.loc[~table.index.isin(priority.index)].copy()
    selected = pd.concat([priority, remainder], ignore_index=True)
    return selected.head(max_columns).copy()


def venn_regions(a: set[str], b: set[str], c: set[str]) -> dict[str, int]:
    return {
        "100": len(a - b - c),
        "010": len(b - a - c),
        "001": len(c - a - b),
        "110": len((a & b) - c),
        "101": len((a & c) - b),
        "011": len((b & c) - a),
        "111": len(a & b & c),
    }


def _format_int(value: float, _position: float | None = None) -> str:
    return f"{int(round(value)):,}"


def draw_venn(
    ax: plt.Axes,
    sets: tuple[set[str], set[str], set[str]],
    labels: tuple[str, str, str],
    title: str,
) -> dict[str, int]:
    a, b, c = sets
    regions = venn_regions(a, b, c)
    union_n = max(len(a | b | c), 1)
    circles = [
        ((-0.38, -0.02), VENN_COLORS[0]),
        ((0.00, 0.37), VENN_COLORS[1]),
        ((0.38, -0.02), VENN_COLORS[2]),
    ]
    for (x, y), color in circles:
        ax.add_patch(Circle((x, y), 0.58, facecolor=color, edgecolor="white", lw=1.0, alpha=0.72))

    positions = {
        "100": (-0.62, -0.12),
        "010": (0.00, 0.68),
        "001": (0.62, -0.12),
        "110": (-0.27, 0.30),
        "101": (0.00, -0.26),
        "011": (0.27, 0.30),
        "111": (0.00, 0.08),
    }
    for key, (x, y) in positions.items():
        count = regions[key]
        percentage = 100 * count / union_n
        percentage_label = (
            "<1%" if count > 0 and percentage < 1 else f"{percentage:.0f}%"
        )
        ax.text(
            x,
            y,
            f"{count:,}\n({percentage_label})",
            ha="center",
            va="center",
            fontsize=8.1,
            color=TEXT_COLOR,
            bbox={"boxstyle": "square,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.72},
        )
    ax.text(-0.78, -0.58, labels[0], ha="left", va="center", fontsize=8.2)
    ax.text(0.00, 0.93, labels[1], ha="center", va="center", fontsize=8.2)
    ax.text(0.78, -0.58, labels[2], ha="right", va="center", fontsize=8.2)
    ax.set_title(title, fontsize=10, pad=4)
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-0.78, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")
    return regions


def cluster_crest_heatmap(
    membership: pd.DataFrame, crests: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize mutually exclusive crest membership within each cluster."""
    categories = heatmap_categories(crests)
    rows = []
    for cluster in CLUSTERS:
        cluster_mask = membership[f"cluster_{cluster}"].astype(bool)
        cluster_size = int(cluster_mask.sum())
        category_masks = {
            "crest_1_only": cluster_mask & membership["crest_1"] & ~membership["crest_2"],
            "crest_2_only": cluster_mask & ~membership["crest_1"] & membership["crest_2"],
            "both_crests": cluster_mask & membership["crest_1"] & membership["crest_2"],
        }
        if sum(int(mask.sum()) for mask in category_masks.values()) > cluster_size:
            raise RuntimeError(f"Exclusive crest categories exceed Cluster {cluster} size")
        for category, label in categories:
            count = int(category_masks[category].sum())
            rows.append(
                {
                    "cluster": cluster,
                    "cluster_pattern": CLUSTER_NAMES[cluster],
                    "cluster_size": cluster_size,
                    "category": category,
                    "category_label": label.replace("\n", " "),
                    "count": count,
                    "percent_of_cluster": 100 * count / cluster_size if cluster_size else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _save_figure(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        (".png", {"dpi": 320}),
        (".pdf", {}),
    ):
        fig.savefig(
            output_stem.with_suffix(suffix),
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig. 4e: UpSet plot (88 mm wide, fixed canvas)
# --------------------------------------------------------------------------- #
FIG4E_W_MM, FIG4E_H_MM = 88.0, 55.0
FIG4E_EDGE_MM = 0.8
FIG4E_TEXT = "#222222"
FIG4E_INACTIVE = "#D4D4D4"
FIG4E_STRIPE = "#F1F1F1"           # same grey as the Fig. 4c block
FIG4E_PNG_DPI = 600
FIG4E_FS = {"tick": 6.0, "label": 6.0, "axis_title": 6.5, "legend": 6.0}
# Crest colours of Fig. 3 (Result-02/05-plot_crest_system_lollipop.py).
FIG3_CREST_COLORS = {"crest_1": "#4E89C7", "crest_2": "#F5A05A"}
BOTH_CRESTS_COLOR = "#555555"
# Cluster palette shared with Fig. 4b and 4d.
FIG4_CLUSTER_COLORS = {1: "#2E3A59", 2: "#7F93AE", 3: "#9AAE8C", 4: "#C49A6C", 5: "#7E5A78"}


def _fig4e_font_family() -> str:
    from matplotlib import font_manager
    import glob as _glob
    for path in _glob.glob(os.path.expanduser("~/.fonts/*.ttf")):
        font_manager.fontManager.addfont(path)
    for family in ("Arial", "Helvetica", "Liberation Sans"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=family),
                                  fallback_to_default=False)
            return family
        except ValueError:
            continue
    return "DejaVu Sans"


def _intersection_color(row) -> str:
    """Crest 1 / Crest 2 / both in Fig. 3 colours; cluster-only in its cluster colour."""
    if row.crest_1 and row.crest_2:
        return BOTH_CRESTS_COLOR
    if row.crest_1:
        return FIG3_CREST_COLORS["crest_1"]
    if row.crest_2:
        return FIG3_CREST_COLORS["crest_2"]
    for cluster in CLUSTERS:
        if getattr(row, f"cluster_{cluster}"):
            return FIG4_CLUSTER_COLORS[cluster]
    return FIG4E_TEXT


def _set_color(set_name: str) -> str:
    if set_name in FIG3_CREST_COLORS:
        return FIG3_CREST_COLORS[set_name]
    return FIG4_CLUSTER_COLORS[int(set_name.rsplit("_", 1)[1])]


def make_upset_figure(
    membership: pd.DataFrame,
    intersections: pd.DataFrame,
    crests: pd.DataFrame,
    output_stem: Path,
    max_intersections: int = 20,
) -> None:
    """Fig. 4e: the original UpSet layout redrawn at 88 × 55 mm."""
    display = display_intersections(intersections, max_intersections).reset_index(drop=True)
    peak_labels = crest_peak_labels(crests)
    set_labels = {
        name: f"{label} ({peak_labels[name]})" if name in peak_labels else label
        for name, label in SET_LABELS.items()
    }
    family = _fig4e_font_family()
    rc = {
        "font.family": family, "font.size": FIG4E_FS["tick"], "text.color": FIG4E_TEXT,
        "axes.edgecolor": FIG4E_TEXT, "axes.labelcolor": FIG4E_TEXT, "axes.linewidth": 0.5,
        "xtick.color": FIG4E_TEXT, "ytick.color": FIG4E_TEXT,
        "xtick.labelsize": FIG4E_FS["tick"], "ytick.labelsize": FIG4E_FS["tick"],
        "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
        "axes.grid": False, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    }
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(FIG4E_W_MM / 25.4, FIG4E_H_MM / 25.4))
        renderer = fig.canvas.get_renderer()

        def size_mm(text, size, weight="normal"):
            artist = fig.text(0, 0, text, fontsize=size, fontweight=weight)
            box = artist.get_window_extent(renderer)
            artist.remove()
            return box.width / fig.dpi * 25.4, box.height / fig.dpi * 25.4

        # ---- physical layout (mm) --------------------------------------- #
        edge = FIG4E_EDGE_MM
        label_w = max(size_mm(set_labels[n], FIG4E_FS["label"])[0] for n in SET_COLUMNS) + 0.6
        _, ytl_h = size_mm("1,000", FIG4E_FS["tick"])
        ytick_w = size_mm("1,000", FIG4E_FS["tick"])[0]
        ylab_h = size_mm("Intersection size", FIG4E_FS["axis_title"])[1]
        sets_w = 13.0                                       # set-size bars
        gap = 1.0
        # The largest set-size tick sits on the left edge of the inverted axis.
        biggest = _format_int(float(membership[SET_COLUMNS].sum().max()))
        left_mm = edge + size_mm(biggest, FIG4E_FS["tick"])[0] / 2 + 0.3
        matrix_w = FIG4E_W_MM - left_mm - edge - sets_w - gap - label_w
        legend_h = 2.5
        top_mm = edge + legend_h + 2.6          # room for the top y tick label
        _, xt_h = size_mm("2,000", FIG4E_FS["tick"])
        _, xl_h = size_mm("Set size", FIG4E_FS["axis_title"])
        bottom_mm = edge + xl_h + 1.0 + xt_h + 1.5 + 0.6
        body_h = FIG4E_H_MM - top_mm - bottom_mm
        matrix_h = body_h * 0.52
        bars_h = body_h - matrix_h - 1.0

        grid = fig.add_gridspec(
            3, 3, width_ratios=[sets_w, gap + label_w, matrix_w],
            height_ratios=[bars_h, 1.0, matrix_h], wspace=0, hspace=0,
            left=left_mm / FIG4E_W_MM, right=1 - edge / FIG4E_W_MM,
            bottom=bottom_mm / FIG4E_H_MM, top=1 - top_mm / FIG4E_H_MM,
        )
        ax_top = fig.add_subplot(grid[0, 2])
        ax_matrix = fig.add_subplot(grid[2, 2], sharex=ax_top)
        ax_sets = fig.add_subplot(grid[2, 0], sharey=ax_matrix)
        ax_labels = fig.add_subplot(grid[2, 1], sharey=ax_matrix)

        x = np.arange(len(display))
        colors = [_intersection_color(row) for row in display.itertuples(index=False)]

        # intersection sizes
        ax_top.bar(x, display["intersection_size"], color=colors, width=0.64, linewidth=0)
        ax_top.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        ax_top.yaxis.set_major_formatter(FuncFormatter(_format_int))
        ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
        ax_top.spines[["top", "right", "bottom"]].set_visible(False)
        ymax = max(float(display["intersection_size"].max()), 1.0)
        ax_top.set_ylim(0, ymax * 1.05)
        ax_top.set_ylabel("Intersection size", fontsize=FIG4E_FS["axis_title"], labelpad=1.5)

        # membership matrix
        n_sets = len(SET_COLUMNS)
        for row_index in range(0, n_sets, 2):
            for axis, x0, width in ((ax_matrix, -0.6, len(display) + 0.2),
                                    (ax_labels, 0.0, 1.0)):
                axis.add_patch(Rectangle((x0, row_index - 0.5), width, 1.0,
                                         facecolor=FIG4E_STRIPE, edgecolor="none", zorder=-3))
        for column_index, row in display.iterrows():
            active = [i for i, name in enumerate(SET_COLUMNS) if bool(row[name])]
            ax_matrix.scatter([column_index] * n_sets, range(n_sets), s=6,
                              color=FIG4E_INACTIVE, edgecolor="none", zorder=2)
            ax_matrix.scatter([column_index] * len(active), active, s=11,
                              color=colors[column_index], edgecolor="none", zorder=3)
            if len(active) > 1:
                ax_matrix.plot([column_index, column_index], [min(active), max(active)],
                               color=colors[column_index], lw=0.75, zorder=2.5)
        ax_matrix.set_xlim(-0.6, len(display) - 0.4)
        ax_matrix.set_ylim(n_sets - 0.5, -0.5)
        ax_matrix.set_xticks([])
        ax_matrix.set_yticks([])
        ax_matrix.spines[:].set_visible(False)

        # set labels (between set-size bars and matrix), right-aligned
        ax_labels.set_xlim(0, 1)
        ax_labels.axis("off")
        for i, name in enumerate(SET_COLUMNS):
            ax_labels.text(1.0 - 0.3 / (gap + label_w), i, set_labels[name], ha="right",
                           va="center", fontsize=FIG4E_FS["label"])

        # set sizes (bars grow to the left)
        set_sizes = membership[SET_COLUMNS].sum().to_numpy(int)
        ax_sets.barh(range(n_sets), set_sizes, height=0.56,
                     color=[_set_color(name) for name in SET_COLUMNS], linewidth=0)
        ax_sets.invert_xaxis()
        ax_sets.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True))
        ax_sets.xaxis.set_major_formatter(FuncFormatter(_format_int))
        ax_sets.tick_params(axis="y", left=False, labelleft=False)
        ax_sets.spines[["top", "right", "left"]].set_visible(False)
        ax_sets.set_xlabel("Set size", fontsize=FIG4E_FS["axis_title"], labelpad=1.5)

        # legend: one row above the intersection bars; "Cluster only" shows the
        # five cluster colours side by side because those bars take their cluster colour.
        from matplotlib.legend_handler import HandlerTuple
        from matplotlib.patches import Patch
        swatch = 2.5 / 25.4 * 72 / FIG4E_FS["legend"]
        cluster_strip = tuple(Patch(facecolor=FIG4_CLUSTER_COLORS[c], edgecolor="none")
                              for c in CLUSTERS)
        fig.legend(
            handles=[
                Patch(facecolor=FIG3_CREST_COLORS["crest_1"], edgecolor="none"),
                Patch(facecolor=FIG3_CREST_COLORS["crest_2"], edgecolor="none"),
                Patch(facecolor=BOTH_CRESTS_COLOR, edgecolor="none"),
                cluster_strip,
            ],
            labels=["Crest 1", "Crest 2", "Both crests", "Cluster only"],
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
            loc="lower right", ncol=4, frameon=False, fontsize=FIG4E_FS["legend"],
            handlelength=swatch, handleheight=swatch, handletextpad=0.4,
            columnspacing=1.0, borderaxespad=0, borderpad=0,
            bbox_to_anchor=(1 - edge / FIG4E_W_MM, 1 - (edge + legend_h) / FIG4E_H_MM),
        )

        # ---- checks: nothing outside the 88 × 55 mm canvas, no overlaps --------- #
        fig.canvas.draw()
        texts = [t for t in fig.findobj(plt.Text) if t.get_visible() and t.get_text().strip()]
        W, H = fig.bbox.width, fig.bbox.height
        for t in texts:
            b = t.get_window_extent(renderer)
            if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > W + 0.5 or b.y1 > H + 0.5:
                raise RuntimeError(f"Fig. 4e text is clipped: {t.get_text()!r}")
        boxes = [(t.get_text(), t.get_window_extent(renderer)) for t in texts]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if boxes[i][1].overlaps(boxes[j][1]):
                    raise RuntimeError(f"Fig. 4e text overlaps: {boxes[i][0]!r} × {boxes[j][0]!r}")

        output_stem.parent.mkdir(parents=True, exist_ok=True)
        for suffix, kwargs in ((".pdf", {}), (".svg", {}), (".png", {"dpi": FIG4E_PNG_DPI})):
            fig.savefig(output_stem.with_suffix(suffix), facecolor="white", **kwargs)
        fig.savefig(output_stem.parent / "fig4e.pdf", facecolor="white")
        fig.savefig(output_stem.parent / "fig4e.png", dpi=FIG4E_PNG_DPI, facecolor="white")
        plt.close(fig)


def make_heatmap_figure(
    heatmap: pd.DataFrame,
    crests: pd.DataFrame,
    output_stem: Path,
) -> None:
    """The 5 × 3 cluster-by-crest overlap heatmap, now exported on its own."""
    categories = heatmap_categories(crests)
    _set_plot_style()
    fig, ax_heat = plt.subplots(figsize=(6.2, 5.65))

    category_order = [category for category, _label in categories]
    matrix = (
        heatmap.pivot(index="cluster", columns="category", values="percent_of_cluster")
        .reindex(index=CLUSTERS, columns=category_order)
        .to_numpy(float)
    )
    counts = (
        heatmap.pivot(index="cluster", columns="category", values="count")
        .reindex(index=CLUSTERS, columns=category_order)
        .to_numpy(int)
    )
    cluster_sizes = (
        heatmap.drop_duplicates("cluster").set_index("cluster")["cluster_size"].reindex(CLUSTERS)
    )
    vmax = max(float(np.nanmax(matrix)), 1.0)
    cmap = LinearSegmentedColormap.from_list(
        "crest_overlap", ["#FFF7F1", "#F2B39D", "#B63A2B"]
    )
    image = ax_heat.imshow(matrix, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax_heat.set_xticks(
        np.arange(len(categories)),
        [label for _category, label in categories],
    )
    ax_heat.set_yticks(
        np.arange(len(CLUSTERS)),
        [
            f"Cluster {cluster}  {CLUSTER_NAMES[cluster]}\n(n={int(cluster_sizes.loc[cluster]):,})"
            for cluster in CLUSTERS
        ],
    )
    ax_heat.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    ax_heat.tick_params(axis="y", length=0, pad=6)
    ax_heat.set_title("Cluster overlap with main-peak sets", fontsize=10.5, pad=12)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            color = "white" if value >= vmax * 0.56 else TEXT_COLOR
            percentage_label = "<1%" if 0 < value < 1 else f"{value:.0f}%"
            ax_heat.text(
                column_index,
                row_index,
                f"{counts[row_index, column_index]:,}\n{percentage_label}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )
    ax_heat.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=1.8)
    ax_heat.tick_params(which="minor", bottom=False, left=False)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.048, pad=0.04)
    colorbar.set_label("Molecules in cluster (%)", rotation=90, labelpad=8)
    colorbar.outline.set_visible(False)

    fig.subplots_adjust(left=0.30, right=0.92, top=0.80, bottom=0.06)
    _save_figure(fig, output_stem)


def make_venn_supplement(
    membership: pd.DataFrame,
    crests: pd.DataFrame,
    output_stem: Path,
) -> pd.DataFrame:
    """Draw five complete Crest 1 / Crest 2 / cluster Venn diagrams."""
    _set_plot_style()
    set_values = {
        name: set(membership.loc[membership[name], "molecule_id"])
        for name in SET_COLUMNS
    }
    peak_labels = crest_peak_labels(crests)
    fig = plt.figure(figsize=(14.6, 8.2), constrained_layout=False)
    grid = fig.add_gridspec(2, 6, wspace=0.12, hspace=0.20)
    slots = [(0, 0, 2), (0, 2, 4), (0, 4, 6), (1, 1, 3), (1, 3, 5)]
    axes = [fig.add_subplot(grid[row, start:end]) for row, start, end in slots]
    venn_rows = []
    for ax, cluster in zip(axes, CLUSTERS):
        cluster_set = set_values[f"cluster_{cluster}"]
        regions = draw_venn(
            ax,
            (set_values["crest_1"], set_values["crest_2"], cluster_set),
            (
                f"Crest 1\n{peak_labels['crest_1']}",
                f"Crest 2\n{peak_labels['crest_2']}",
                f"Cluster {cluster}\n{CLUSTER_NAMES[cluster]}",
            ),
            f"Crest 1 / Crest 2 / Cluster {cluster}",
        )
        union_size = len(set_values["crest_1"] | set_values["crest_2"] | cluster_set)
        for region, count in regions.items():
            venn_rows.append(
                {
                    "cluster": cluster,
                    "cluster_pattern": CLUSTER_NAMES[cluster],
                    "region_code": region,
                    "count": count,
                    "union_size": union_size,
                    "percent_of_union": 100 * count / union_size if union_size else np.nan,
                }
            )
    fig.subplots_adjust(left=0.035, right=0.985, top=0.95, bottom=0.06)
    _save_figure(fig, output_stem)
    return pd.DataFrame(venn_rows)


def write_outputs(
    crests: pd.DataFrame,
    crest_audit: pd.DataFrame,
    selected_peaks: pd.DataFrame,
    peak_molecules: pd.DataFrame,
    membership: pd.DataFrame,
    diagnostics: pd.DataFrame,
    intersections: pd.DataFrame,
    heatmap: pd.DataFrame,
    venn: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    crests.to_csv(output_dir / "crest_definitions.csv", index=False)
    crest_audit.to_csv(output_dir / "crest_source_audit.csv", index=False)
    selected_peaks.to_csv(output_dir / "crest_selected_peaks.csv", index=False)
    peak_molecules.to_csv(output_dir / "crest_peak_significant_molecules.csv", index=False)
    membership.to_csv(output_dir / "molecule_set_membership.csv", index=False)
    diagnostics.to_csv(output_dir / "molecule_matching_diagnostics.csv", index=False)
    intersections.to_csv(output_dir / "upset_exact_intersections.csv", index=False)
    heatmap.to_csv(output_dir / "cluster_crest_overlap_heatmap.csv", index=False)
    venn.to_csv(output_dir / "venn_region_counts.csv", index=False)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deswan-dir", type=Path, default=DEFAULT_DESWAN_DIR)
    parser.add_argument("--cluster-file", type=Path, default=DEFAULT_CLUSTER_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--rscript", type=Path, help="Rscript executable in an R environment with DEswan")
    parser.add_argument("--max-intersections", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    crests, crest_audit = select_crest_peaks(args.deswan_dir)
    peak_molecules, selected_peaks = load_peak_significance(args.deswan_dir, crest_audit, args.rscript)
    assignments = load_cluster_assignments(args.cluster_file)
    membership, diagnostics = build_membership(peak_molecules, assignments)
    crests["n_molecules"] = crests["set_name"].map(membership[SET_COLUMNS].sum())
    intersections = exact_intersections(membership)
    heatmap = cluster_crest_heatmap(membership, crests)
    main_figure_stem = args.figure_dir / "crest_cluster_intersections"
    supplement_stem = args.figure_dir / "crest_cluster_venn_supplement"
    make_upset_figure(
        membership,
        intersections,
        crests,
        main_figure_stem,
        max_intersections=args.max_intersections,
    )
    make_heatmap_figure(heatmap, crests, args.figure_dir / "crest_cluster_overlap_heatmap")
    venn = make_venn_supplement(membership, crests, supplement_stem)
    write_outputs(
        crests,
        crest_audit,
        selected_peaks,
        peak_molecules,
        membership,
        diagnostics,
        intersections,
        heatmap,
        venn,
        args.output_dir,
    )
    print("Selected main peaks (exact-age significance):")
    print(selected_peaks[["set_name", "omics", "main_peak_rank", "peak_age", "q_cutoff", "fdr_method", "n_significant"]].to_string(index=False))
    print("\nSet sizes:")
    print(membership[SET_COLUMNS].sum().to_string())
    print(f"\nMain figure: {main_figure_stem.with_suffix('.png')}")
    print(f"Supplement: {supplement_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
