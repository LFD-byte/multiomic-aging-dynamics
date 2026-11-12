"""Association of omics AgeAccel with baseline recorded diagnoses (odds ratios).

For each of the eight prespecified diagnoses used in
incremental_disease_discrimination.py, an unpenalized logistic model

    diagnosis ~ AgeAccel (per 1 SD) + chronological age + sex

is compared with the nested model ``diagnosis ~ age + sex``. The effect of
interest is the odds ratio (OR) per 1-SD higher AgeAccel with a Wald 95%
confidence interval; significance is assessed with the likelihood-ratio test
(LRT, 1 d.f.) of the nested models. This replaces the DeLong comparison of
nested models, which is not a valid test of an added predictor.

Cohorts and scores are identical to the incremental-AUC analysis:

* Primary: the Result-04 four-omics common cohort with five AgeAccel scores
  (Transcriptomics, Lipidomics, Gut Microbiome, Oral Microbiome and the
  4-omics stack). AgeAccel is the age-residualized predicted-minus-chronological
  age; each score is standardized (mean 0, SD 1) within the analysis cohort.
  BH-FDR is controlled across all 40 primary tests (8 diagnoses x 5 scores).
* Metabolomics: its own maximum available cohort, a different set of
  participants, with a separate 8-test BH-FDR family. Descriptive only.

Analyses are cross-sectional associations with recorded baseline diagnoses.
BMI is deliberately not adjusted for in this primary specification.

Usage (from this directory):
    python disease_odds_ratio.py               # fit models, save CSVs, draw figures
    python disease_odds_ratio.py --plot-only   # redraw figures from saved CSV

Plotting reads only outputs/disease_or_results.csv; it does not load
participant-level data.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401  (used via matplotlib.ticker)
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "outputs"
DEFAULT_FIGURES = HERE / "figures"
RESULTS_NAME = "disease_or_results.csv"
EXPORT_DPI = 300
MAIN_WIDTH_MM = 89
EXTENDED_WIDTH_MM = 210
EXTENDED_HEIGHT_MM = 72
# Extended Data is split into two 1 x 4 figures; diagnoses in the order shown.
EXTENDED_GROUPS = {
    "group1": ("Hypercholesterolaemia", "Hypothyroidism", "Cholelithiasis", "Asthma"),
    "group2": ("Hypertension", "Hyperglycaemia", "Obesity", "Fatty liver"),
}

# Labels must match seven_clock_disease_panels / the saved CSV schema.
STACK_LABEL = "4-omics stack"
MET_LABEL = "Metabolomics†"
PRIMARY_SCORES = (
    "Transcriptomics", "Lipidomics", "Gut Microbiome",
    "Oral Microbiome", STACK_LABEL,
)
DISPLAY_ORDER = PRIMARY_SCORES + (MET_LABEL,)
SHORT = {
    "Transcriptomics": "Transcriptomics", "Lipidomics": "Lipidomics",
    "Gut Microbiome": "Gut microbiome", "Oral Microbiome": "Oral microbiome",
    STACK_LABEL: "Four-omics stack", MET_LABEL: "Metabolomics†",
}
COLORS = {
    "Transcriptomics": "#D95F59", "Lipidomics": "#2F9E73",
    "Gut Microbiome": "#E28C34", "Oral Microbiome": "#8E63B7",
    STACK_LABEL: "#202020", MET_LABEL: "#8FB6DA",
}
PRIMARY_COHORT = "four_omics_common"
MET_COHORT = "metabolomics_available"


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def bh_fdr(p: pd.Series) -> pd.Series:
    """NaN-preserving Benjamini-Hochberg adjusted P values."""
    out = pd.Series(np.nan, index=p.index, dtype=float)
    ok = p.notna()
    values = p[ok].to_numpy(float)
    if not len(values):
        return out
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    out.loc[ok] = adjusted
    return out


def fit_logistic(X: np.ndarray, y: np.ndarray, max_iter: int = 100, tol: float = 1e-10):
    """Unpenalized logistic regression by Newton-Raphson (IRLS).

    ``X`` must already contain an intercept column. Returns coefficients,
    their covariance (inverse observed information) and the log-likelihood.
    """
    beta = np.zeros(X.shape[1])
    loglik_old = -np.inf
    for _ in range(max_iter):
        eta = X @ beta
        prob = 1.0 / (1.0 + np.exp(-eta))
        weight = prob * (1.0 - prob)
        hessian = X.T @ (X * weight[:, None])
        step = np.linalg.solve(hessian, X.T @ (y - prob))
        beta = beta + step
        eta = X @ beta
        loglik = float(y @ eta - np.logaddexp(0.0, eta).sum())
        if abs(loglik - loglik_old) < tol and np.max(np.abs(step)) < 1e-8:
            break
        loglik_old = loglik
    else:
        raise RuntimeError("logistic regression did not converge")
    prob = 1.0 / (1.0 + np.exp(-(X @ beta)))
    hessian = X.T @ (X * (prob * (1.0 - prob))[:, None])
    if np.linalg.cond(hessian) > 1e12:
        raise RuntimeError("near-singular information matrix (possible separation)")
    return beta, np.linalg.inv(hessian), loglik


def _zscore(values: pd.Series) -> np.ndarray:
    x = values.to_numpy(float)
    return (x - x.mean()) / x.std(ddof=1)


def odds_ratio_row(data: pd.DataFrame, score: str, age_col: str) -> dict:
    """OR per 1-SD AgeAccel adjusted for age and sex, with LRT against age + sex."""
    from scipy import stats

    y = data["outcome"].to_numpy(float)
    ones = np.ones(len(data))
    age_z = _zscore(data[age_col])          # scaling age does not change the AgeAccel OR
    sex = data["sex"].to_numpy(float)
    score_z = _zscore(data[score])
    X0 = np.column_stack([ones, age_z, sex])
    X1 = np.column_stack([ones, score_z, age_z, sex])
    _, _, ll0 = fit_logistic(X0, y)
    beta, cov, ll1 = fit_logistic(X1, y)
    log_or, se = float(beta[1]), float(np.sqrt(cov[1, 1]))
    lr_stat = max(2.0 * (ll1 - ll0), 0.0)
    return {
        "n": len(y), "n_cases": int(y.sum()), "n_controls": int((y == 0).sum()),
        "score_sd_years": float(data[score].std(ddof=1)),
        "log_or": log_or, "se_log_or": se,
        "odds_ratio": float(np.exp(log_or)),
        "or_ci_low": float(np.exp(log_or - 1.959964 * se)),
        "or_ci_high": float(np.exp(log_or + 1.959964 * se)),
        "wald_z": log_or / se,
        "wald_p": float(2 * stats.norm.sf(abs(log_or / se))),
        "lr_stat": lr_stat,
        "lr_p": float(stats.chi2.sf(lr_stat, df=1)),
    }


def analyse(endpoints: dict[str, pd.DataFrame], primary: pd.DataFrame,
            metabolomics: pd.DataFrame, sex: pd.DataFrame, age_col: str) -> pd.DataFrame:
    """Fit every diagnosis x score model and add FDR within each cohort family."""
    rows = []
    for outcome, endpoint in endpoints.items():
        cohorts = (
            (PRIMARY_COHORT, primary, PRIMARY_SCORES),
            (MET_COHORT, metabolomics, (MET_LABEL,)),
        )
        for cohort, scores, names in cohorts:
            data = (scores.merge(sex, on="participant_id")
                    .merge(endpoint, on="participant_id").dropna())
            for score in names:
                row = {"outcome": outcome, "score": score, "cohort": cohort}
                try:
                    row.update(odds_ratio_row(data, score, age_col))
                    row["fit_status"] = "ok"
                except (RuntimeError, np.linalg.LinAlgError) as error:
                    row.update(n=len(data), n_cases=int(data["outcome"].sum()),
                               fit_status=f"failed: {error}")
                rows.append(row)
    results = pd.DataFrame(rows)
    # Primary family: all 40 four-omics-cohort tests; metabolomics: its own 8 tests.
    results["q_value"] = np.nan
    for cohort in (PRIMARY_COHORT, MET_COHORT):
        mask = results["cohort"].eq(cohort)
        results.loc[mask, "q_value"] = bh_fdr(results.loc[mask, "lr_p"])
    results["fdr_significant"] = results["q_value"] < 0.05
    results["covariates"] = "age+sex"
    results["fdr_family"] = np.where(results["cohort"].eq(PRIMARY_COHORT),
                                     "40 primary tests (8 diagnoses x 5 scores)",
                                     "8 metabolomics tests")
    return results


def run_analysis(medical_path: Path | None = None, population_path: Path | None = None,
                 output_dir: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    """Load inputs (disease_data.py), fit models and save CSVs."""
    # Imported lazily so that --plot-only needs no participant-level modules.
    from disease_data import (
        AGE_LABEL, DEFAULT_MEDICAL_CONDITIONS, DEFAULT_POPULATION, load_baseline_diagnoses,
        load_metabolomics_scores, load_primary_scores, load_sex,
    )

    medical_path = Path(medical_path or DEFAULT_MEDICAL_CONDITIONS)
    population_path = Path(population_path or DEFAULT_POPULATION)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    endpoints, audit, _ = load_baseline_diagnoses(medical_path)
    results = analyse(endpoints, load_primary_scores(), load_metabolomics_scores(),
                      load_sex(population_path), AGE_LABEL)
    results.to_csv(output_dir / RESULTS_NAME, index=False)
    audit.to_csv(output_dir / "disease_or_outcome_audit.csv", index=False)

    failed = results[results["fit_status"].ne("ok")]
    if len(failed):
        print(f"WARNING: {len(failed)} model(s) failed; see fit_status in {RESULTS_NAME}")
    summary = results.assign(
        cell=lambda d: d["odds_ratio"].map("{:.2f}".format)
        + d["q_value"].map(lambda q: _stars(q) if pd.notna(q) else "")
    ).pivot(index="outcome", columns="score", values="cell")
    print("OR per 1-SD AgeAccel (age + sex adjusted; * BH-FDR q < 0.05):")
    print(summary.reindex(columns=list(DISPLAY_ORDER)).to_string())
    return results


# --------------------------------------------------------------------------
# Figures (read the saved CSV only)
# --------------------------------------------------------------------------

def _stars(q: float) -> str:
    if q < .001: return "***"
    if q < .01: return "**"
    if q < .05: return "*"
    return ""


def _save_figure(fig, path: Path):
    with plt.rc_context({"savefig.bbox": None, "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig.savefig(path.with_suffix(".png"), dpi=EXPORT_DPI, bbox_inches=None)
        fig.savefig(path.with_suffix(".pdf"), bbox_inches=None)
        fig.savefig(path.with_suffix(".svg"), bbox_inches=None)


def _log_limits(results: pd.DataFrame) -> tuple[float, float]:
    lo = float(np.nanmin(results["or_ci_low"]))
    hi = float(np.nanmax(results["or_ci_high"]))
    return min(0.7, lo / 1.08), max(1.5, hi * 1.08)


def _format_log_axis(ax, xlim):
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ticks = [t for t in (0.5, 0.7, 1.0, 1.4, 2.0, 3.0) if xlim[0] <= t <= xlim[1]]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.axvline(1.0, color="#333333", lw=0.7, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def plot_main(results: pd.DataFrame, path: Path, outcome_order: list[str],
              width_mm: float = MAIN_WIDTH_MM):
    """Main-figure panel: four-omics stack AgeAccel, one row per diagnosis."""
    sub = (results[results["score"].eq(STACK_LABEL)]
           .set_index("outcome").reindex(outcome_order))
    n_rows = len(sub)
    fig, ax = plt.subplots(figsize=(width_mm / 25.4, (18 + 7.5 * n_rows) / 25.4))
    y = np.arange(n_rows)[::-1]
    xlim = _log_limits(sub)
    for yi, (outcome, row) in zip(y, sub.iterrows()):
        sig = bool(row["q_value"] < .05)
        ax.plot([row["or_ci_low"], row["or_ci_high"]], [yi, yi], color="#202020", lw=0.9)
        ax.plot(row["odds_ratio"], yi, "o", ms=4.2, mec="#202020", mew=0.9,
                mfc="#202020" if sig else "white", zorder=3)
        ax.text(1.02, yi, f"{row['odds_ratio']:.2f} ({row['or_ci_low']:.2f}–"
                f"{row['or_ci_high']:.2f}){_stars(row['q_value'])}",
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6)
    labels = [f"{o} ({int(sub.loc[o, 'n_cases'])})" for o in sub.index]
    ax.set_yticks(y, labels, fontsize=6.5)
    ax.set_ylim(-0.7, n_rows - 0.3)
    _format_log_axis(ax, xlim)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_xlabel("OR per 1-SD four-omics AgeAccel (95% CI)", fontsize=6.5)
    ax.text(1.02, n_rows - 0.35, "OR (95% CI)", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=6, weight="bold")
    fig.subplots_adjust(left=0.42, right=0.68, bottom=0.42 / (n_rows ** 0.5), top=0.93)
    _save_figure(fig, path)
    plt.close(fig)


def plot_extended(results: pd.DataFrame, path: Path, outcome_order: list[str],
                  width_mm: float = EXTENDED_WIDTH_MM, height_mm: float = EXTENDED_HEIGHT_MM,
                  xlim: tuple[float, float] | None = None):
    """Extended Data: all scores, one panel per diagnosis in a 1 x 4 row.

    ``xlim`` is shared across the split figures so their axes are comparable.
    """
    xlim = xlim or _log_limits(results)
    n_scores = len(DISPLAY_ORDER)
    fig, axes = plt.subplots(1, 4, figsize=(width_mm / 25.4, height_mm / 25.4),
                             sharex=True, sharey=True, squeeze=False)
    y = np.arange(n_scores)[::-1]
    for ax, outcome in zip(axes.flat, outcome_order):
        sub = results[results["outcome"].eq(outcome)].set_index("score").reindex(DISPLAY_ORDER)
        for yi, (score, row) in zip(y, sub.iterrows()):
            if not np.isfinite(row["odds_ratio"]):
                continue
            colour = COLORS[score]
            met = score == MET_LABEL
            ax.plot([row["or_ci_low"], row["or_ci_high"]], [yi, yi], color=colour,
                    lw=0.8, alpha=0.75 if met else 1.0)
            filled = (not met) and bool(row["q_value"] < .05)
            ax.plot(row["odds_ratio"], yi, "o", ms=3.4, mec=colour, mew=0.8,
                    mfc=colour if filled else "white", zorder=3)
        primary = sub.loc[STACK_LABEL]
        met_row = sub.loc[MET_LABEL]
        ax.set_title(f"{outcome}\n{int(primary['n_cases'])}/{int(primary['n'])} "
                     f"(Met†: {int(met_row['n_cases'])}/{int(met_row['n'])})",
                     fontsize=6.5, weight="bold", pad=4, linespacing=1.2)
        _format_log_axis(ax, xlim)
        ax.tick_params(axis="x", labelsize=6)
    for ax in axes.flat[len(outcome_order):]:
        ax.set_visible(False)
    axes[0, 0].set_yticks(y, [SHORT[s] for s in DISPLAY_ORDER], fontsize=6.5)
    axes[0, 0].set_ylim(-0.6, n_scores - 0.4)
    fig.supxlabel("OR per 1-SD AgeAccel, adjusted for age and sex (95% CI)", fontsize=7, y=0.03)
    handles = [
        Line2D([], [], marker="o", ls="none", color="#202020", mfc="#202020", ms=3.4,
               label="BH-FDR q < 0.05"),
        Line2D([], [], marker="o", ls="none", color="#202020", mfc="white", ms=3.4,
               label="q ≥ 0.05"),
        Line2D([], [], marker="o", ls="none", color=COLORS[MET_LABEL], mfc="white", ms=3.4,
               label="† Metabolomics, separate cohort (descriptive)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=6,
               bbox_to_anchor=(0.55, 0.995))
    fig.subplots_adjust(left=0.135, right=0.99, bottom=0.19, top=0.76, wspace=0.12)
    _save_figure(fig, path)
    plt.close(fig)


def run_plots(results_path: Path = DEFAULT_OUTPUT / RESULTS_NAME,
              figure_dir: Path = DEFAULT_FIGURES, sort_by_or: bool = False) -> pd.DataFrame:
    """Draw the main and Extended Data figures from the saved results CSV."""
    results_path, figure_dir = Path(results_path), Path(figure_dir)
    if not results_path.is_file():
        raise FileNotFoundError(f"{results_path} not found; run the analysis first.")
    results = pd.read_csv(results_path)
    required = {"outcome", "score", "n", "n_cases", "odds_ratio",
                "or_ci_low", "or_ci_high", "q_value"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"results CSV missing columns: {sorted(missing)}")
    outcome_order = results["outcome"].drop_duplicates().tolist()   # prespecified order
    if sort_by_or:
        stack = results[results["score"].eq(STACK_LABEL)].set_index("outcome")["odds_ratio"]
        outcome_order = stack.sort_values(ascending=False).index.tolist()
    figure_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "sans-serif", "font.size": 7,
                         "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "axes.linewidth": 0.6}):
        plot_main(results, figure_dir / "disease_or_main_stack", outcome_order)
        shared_xlim = _log_limits(results)
        for name, group in EXTENDED_GROUPS.items():
            missing_outcomes = set(group).difference(outcome_order)
            if missing_outcomes:
                raise ValueError(f"{name} diagnoses not in results: {sorted(missing_outcomes)}")
            order = [o for o in outcome_order if o in group] if sort_by_or else list(group)
            plot_extended(results, figure_dir / f"disease_or_extended_{name}", order,
                          xlim=shared_xlim)
    print(f"Figures saved to {figure_dir.resolve()}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--medical-conditions", type=Path, default=None)
    parser.add_argument("--population", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip model fitting and redraw figures from the saved CSV")
    parser.add_argument("--sort-by-or", action="store_true",
                        help="Order diagnoses by four-omics stack OR instead of the prespecified order")
    args = parser.parse_args()
    if not args.plot_only:
        run_analysis(args.medical_conditions, args.population, args.output_dir)
    run_plots(args.output_dir / RESULTS_NAME, args.figure_dir, sort_by_or=args.sort_by_or)


if __name__ == "__main__":
    main()
