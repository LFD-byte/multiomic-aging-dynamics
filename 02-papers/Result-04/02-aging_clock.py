"""Figure : four-omics stacking plus a parallel metabolomics clock.

Layer 1 is not refitted here. This script consumes the leakage-controlled,
10 x 5 outer-fold predictions created by 01-nonlinearity_model_comparison.
The input manifest must confirm 1,000 transcriptomic features and all supplied
features for the other omics, without repeated filtering or imputation.
The model family is pre-specified per omic: Elastic Net for transcriptomics,
metabolomics and lipidomics; the nonlinear representative for both microbiomes.
Layer 2 aligns four omics by participant_id within repeat and evaluates an
Elastic Net. Metabolomics is exported separately and never enters Layer 2.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
OUT, FIG = HERE / "outputs", HERE / "figures"
MODEL_DIR = OUT / "models"
LAYER1_FILE = OUT / "nonlinearity_oof_predictions.parquet"
PRED_COL_BY_OMIC = {
    "Transcriptomics": "Elastic Net",
    "Metabolomics": "Elastic Net",
    "Lipidomics": "Elastic Net",
    "Gut Microbiome": "Nonlinear representative",
    "Oral Microbiome": "Nonlinear representative",
}
SEED, OUTER_FOLDS, INNER_FOLDS = 202604, 5, 5
OLD_FIVE_OMICS_N = 271
MIN_FOUR_OMICS_N = int(np.ceil(1.5 * OLD_FIVE_OMICS_N))

OMICS = ["Transcriptomics", "Metabolomics", "Lipidomics",
         "Gut Microbiome", "Oral Microbiome"]
LAYER1_FEATURE_POLICY = {o: 1000 if o == "Transcriptomics" else "all" for o in OMICS}
FUSION = ["Transcriptomics", "Lipidomics", "Gut Microbiome", "Oral Microbiome"]
PARALLEL = "Metabolomics"
META_COLS = [f"pred__{x}" for x in FUSION]
COLORS = {"Transcriptomics": "#D95F5F", "Metabolomics": "#377EB8",
          "Lipidomics": "#2CA25F", "Gut Microbiome": "#E68A2E",
          "Oral Microbiome": "#8E63B6"}


def metrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return {"mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)),
            "r": float(np.corrcoef(y, pred)[0, 1])}


def load_layer1():
    rerun = "Rerun 01-nonlinearity_model_comparison.ipynb to rebuild Layer-1 predictions."
    if not LAYER1_FILE.exists():
        raise FileNotFoundError(f"Missing {LAYER1_FILE}. {rerun}")
    config_file = LAYER1_FILE.with_name("nonlinearity_run_config.json")
    if not config_file.exists():
        raise FileNotFoundError(f"Missing Layer-1 feature-policy manifest. {rerun}")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if (config.get("feature_policy") != LAYER1_FEATURE_POLICY or
            config.get("input_preprocessing") != "already_filtered_and_imputed"):
        raise ValueError(
            "Cached Layer-1 predictions do not use transcriptomics top 1,000 / "
            f"all other preprocessed features. {rerun}")
    df = pd.read_parquet(LAYER1_FILE)
    required = ({"omics", "participant_id", "age", "repeat", "fold", "n_features"} |
                set(PRED_COL_BY_OMIC.values()))
    if missing := required - set(df):
        raise ValueError(f"Layer-1 OOF file missing columns: {sorted(missing)}. {rerun}")
    df = df.loc[df.omics.isin(OMICS)].copy()
    if missing_omics := set(OMICS) - set(df.omics):
        raise ValueError(f"Layer-1 OOF file missing omics: {sorted(missing_omics)}. {rerun}")
    input_counts = config.get("input_feature_counts", {})
    for omic in OMICS:
        n_input = input_counts.get(omic)
        if not isinstance(n_input, int) or n_input < 1:
            raise ValueError(f"Missing input feature count for {omic}. {rerun}")
        expected = min(n_input, 1000) if omic == "Transcriptomics" else n_input
        if not df.loc[df.omics.eq(omic), "n_features"].eq(expected).all():
            raise ValueError(f"{omic}: cached OOF feature counts violate the feature policy. {rerun}")
    df["participant_id"] = df.participant_id.astype(str).str.strip()
    df["repeat"] = pd.to_numeric(df.repeat, errors="raise").astype(int)
    df["fold"] = pd.to_numeric(df.fold, errors="raise").astype(int)
    df["age"] = pd.to_numeric(df.age, errors="raise")
    for col in set(PRED_COL_BY_OMIC.values()):
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df


def participant_tables(df):
    """One cross-fitted clock value per participant, omic, and repeat."""
    tables = {}
    for omic in OMICS:
        pred_col = PRED_COL_BY_OMIC[omic]
        sub = df.loc[df.omics.eq(omic)]
        out = sub.groupby(["participant_id", "repeat"], as_index=False).agg(
            true_age=("age", "mean"), pred_age=(pred_col, "mean"),
            n_observations=("age", "size"), layer1_fold=("fold", "first"))
        if out.duplicated(["participant_id", "repeat"]).any():
            raise RuntimeError(f"{omic}: duplicate participant/repeat rows")
        print(f"  {omic:<18} participants={out.participant_id.nunique():>6,}  "
              f"repeats={out.repeat.nunique():>2}  model={pred_col}")
        tables[omic] = out
    return tables


def four_omics_table(tables):
    merged = None
    for omic in FUSION:
        part = tables[omic][["participant_id", "repeat", "true_age", "pred_age"]]
        part = part.rename(columns={"true_age": f"age__{omic}",
                                    "pred_age": f"pred__{omic}"})
        merged = part if merged is None else merged.merge(
            part, on=["participant_id", "repeat"], how="inner",
            validate="one_to_one")
    age_cols = [f"age__{x}" for x in FUSION]
    merged["true_age"] = merged[age_cols].median(axis=1)
    merged["cross_omic_age_spread"] = (merged[age_cols].max(axis=1) -
                                        merged[age_cols].min(axis=1))
    return merged.sort_values(["repeat", "participant_id"]).reset_index(drop=True)


def meta_pipeline(seed):
    return Pipeline([("scale", StandardScaler()),
                     ("model", ElasticNet(max_iter=100_000, random_state=seed))])


META_GRID = {"model__alpha": np.logspace(-4, 1, 40),
             "model__l1_ratio": [0.05, 0.2, 0.5, 0.8, 0.95, 1.0]}


def fit_meta_repeat(frame, repeat):
    """Nested Layer-2 CV for one Layer-1 repeat."""
    X, y = frame[META_COLS], frame.true_age.to_numpy(float)
    outer = KFold(OUTER_FOLDS, shuffle=True, random_state=SEED + repeat)
    oof, details = np.full(len(frame), np.nan), []
    for fold, (tr, va) in enumerate(outer.split(X), 1):
        inner = KFold(INNER_FOLDS, shuffle=True,
                      random_state=SEED + repeat * 100 + fold)
        search = GridSearchCV(meta_pipeline(SEED + repeat * 100 + fold),
                              META_GRID, scoring="neg_mean_absolute_error",
                              cv=inner, refit=True, n_jobs=-1)
        search.fit(X.iloc[tr], y[tr])
        oof[va] = search.predict(X.iloc[va])
        fm = metrics(y[va], oof[va])
        coef = search.best_estimator_.named_steps["model"].coef_
        details.append({"repeat": repeat, "fold": fold,
                        "n_train": len(tr), "n_valid": len(va), **fm,
                        "best_alpha": float(search.best_params_["model__alpha"]),
                        "best_l1_ratio": float(search.best_params_["model__l1_ratio"]),
                        "inner_best_mae": float(-search.best_score_),
                        "scaled_coefficients": dict(zip(META_COLS, map(float, coef)))})
    if np.isnan(oof).any():
        raise RuntimeError(f"Repeat {repeat}: incomplete meta OOF predictions")
    result = frame[["participant_id", "repeat", "true_age",
                    "cross_omic_age_spread"]].copy()
    result["pred_age"] = oof
    return result, details


def repeat_summary(pred):
    rows = [{"repeat": int(rep), "n": len(x), **metrics(x.true_age, x.pred_age)}
            for rep, x in pred.groupby("repeat")]
    by_rep = pd.DataFrame(rows)
    sm = {m: float(by_rep[m].mean()) for m in ["mae", "r2", "r"]}
    sm.update({f"{m}_sd": float(by_rep[m].std(ddof=1)) for m in ["mae", "r2", "r"]})
    return by_rep, sm


def fit_final(meta):
    avg = meta.groupby("participant_id", as_index=False).agg(
        {**{c: "mean" for c in META_COLS}, "true_age": "median"})
    cv = KFold(INNER_FOLDS, shuffle=True, random_state=SEED + 9000)
    search = GridSearchCV(meta_pipeline(SEED + 9000), META_GRID,
                          scoring="neg_mean_absolute_error", cv=cv,
                          refit=True, n_jobs=-1)
    search.fit(avg[META_COLS], avg.true_age)
    return search.best_estimator_, len(avg)


def downstream_predictions(tables, meta_pred):
    """One row per participant and clock, compatible with downstream notebooks."""
    rows = []
    for omic in OMICS:
        x = tables[omic].groupby("participant_id", as_index=False).agg(
            pred_age=("pred_age", "mean"), true_age=("true_age", "median"),
            n_repeats=("repeat", "nunique"))
        x["clock_name"], x["role"] = omic, (
            "parallel_validation" if omic == PARALLEL else "fusion_input")
        rows.append(x)
    x = meta_pred.groupby("participant_id", as_index=False).agg(
        pred_age=("pred_age", "mean"), true_age=("true_age", "median"),
        n_repeats=("repeat", "nunique"))
    x["clock_name"], x["role"] = "Multi-Omics (4-omics stacking)", "fusion_output"
    rows.append(x)
    out = pd.concat(rows, ignore_index=True)
    out["split"] = "repeated_nested_outer_oof_mean"
    return out[["participant_id", "clock_name", "pred_age", "true_age",
                "split", "role", "n_repeats"]]


def draw_architecture(n):
    fig, ax = plt.subplots(figsize=(11.5, 6.8)); ax.set(xlim=(0, 12), ylim=(0, 8)); ax.axis("off")
    def box(x, y, w, h, label, face, edge="#333", ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.08",
                     facecolor=face, edgecolor=edge, linewidth=1.2, linestyle=ls))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=9)
    def arrow(x1, y1, x2, y2, color="#555", ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=12, linewidth=1.2, color=color, linestyle=ls))
    for omic, y in zip(OMICS, [6.7, 5.35, 4, 2.65, 1.3]):
        box(.3, y, 2, .75, f"{omic}\nfull omic cohort", "#F7F7F7")
        arrow(2.3, y+.375, 2.75, y+.375)
        model_label = ("Elastic Net" if PRED_COL_BY_OMIC[omic] == "Elastic Net"
                       else "Nonlinear model")
        box(2.75, y, 2.15, .75, f"{model_label} clock\n10 x 5 outer OOF",
            COLORS[omic]+"33", COLORS[omic])
        if omic in FUSION: arrow(4.9, y+.375, 7.15, 4.5, COLORS[omic])
        else: arrow(4.9, y+.375, 9.3, y+.375, COLORS[omic], "--")
    box(7.15, 3.75, 2.15, 1.5, f"4-clock OOF matrix\nparticipant intersection\nn = {n:,}",
        "#FFF5D6", "#B78300"); arrow(9.3, 4.5, 9.75, 4.5, "#B78300")
    box(9.75, 3.95, 1.9, 1.1, "Elastic Net\nmeta-learner", "#E9E1F5", "#6F4A9E")
    box(9.3, 1.3, 2.35, .75, "Metabolomics clock\nparallel validation output",
        "#E6F1FA", COLORS[PARALLEL], "--")
    ax.text(.3, 7.65, "Layer 1 — five omic-specific clocks", fontsize=12, weight="bold")
    ax.text(7.15, 7.15, "Layer 2 — four-omics stacking", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG/"stacking_architecture.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG/"stacking_architecture.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True); MODEL_DIR.mkdir(exist_ok=True)
    print("Layer 1 — reuse verified 01-nonlinearity repeated outer-OOF predictions")
    raw = load_layer1(); tables = participant_tables(raw); meta = four_omics_table(tables)
    counts = meta.groupby("repeat").participant_id.nunique()
    if counts.nunique() != 1:
        raise RuntimeError(f"Intersection varies by repeat: {counts.to_dict()}")
    n = int(counts.iloc[0]); print(f"\nLayer 2 — four-omics intersection n={n:,} per repeat")
    if n < MIN_FOUR_OMICS_N:
        raise RuntimeError(f"n={n} < prespecified {MIN_FOUR_OMICS_N} (1.5 x 271)")
    age_spread = meta.groupby("participant_id").cross_omic_age_spread.max()
    print(f"  cross-omic age spread >1 year: {(age_spread > 1).sum():,}")

    pred_parts, details = [], []
    for rep, frame in meta.groupby("repeat", sort=True):
        pred, fold_info = fit_meta_repeat(frame.reset_index(drop=True), int(rep))
        pred_parts.append(pred); details.extend(fold_info)
        mm = metrics(pred.true_age, pred.pred_age)
        print(f"  repeat {int(rep):02d}: MAE={mm['mae']:.3f}, R2={mm['r2']:.3f}")
    meta_pred = pd.concat(pred_parts, ignore_index=True)
    rep_metrics, meta_sm = repeat_summary(meta_pred)
    final_model, final_n = fit_final(meta)
    joblib.dump({"model": final_model, "features": META_COLS},
                MODEL_DIR/"four_omics_elastic_net.joblib")

    perf = []
    for omic in OMICS:
        _, sm = repeat_summary(tables[omic])
        perf.append({"clock": omic, "stage": "Layer 1",
            "role": "parallel_validation" if omic == PARALLEL else "fusion_input",
            "layer1_model": PRED_COL_BY_OMIC[omic],
            "evaluation": "10x5_grouped_outer_oof_from_01b",
            "n_total": int((raw.omics.eq(omic) & raw.repeat.eq(1)).sum()),
            "n_participants": tables[omic].participant_id.nunique(),
            "mae_outer": sm["mae"], "mae_outer_sd": sm["mae_sd"],
            "r2_outer": sm["r2"], "r2_outer_sd": sm["r2_sd"],
            "r_outer": sm["r"], "r_outer_sd": sm["r_sd"],
            "mae_cv": sm["mae"], "r2_cv": sm["r2"], "r_cv": sm["r"],
            "mae_held": sm["mae"], "r2_held": sm["r2"]})
    perf.append({"clock": "Multi-Omics (4-omics stacking)", "stage": "Layer 2",
        "role": "fusion_output", "layer1_model": "Elastic Net meta-learner",
        "evaluation": "10x_nested_cv_on_layer1_oof",
        "n_total": n, "n_participants": n,
        "mae_outer": meta_sm["mae"], "mae_outer_sd": meta_sm["mae_sd"],
        "r2_outer": meta_sm["r2"], "r2_outer_sd": meta_sm["r2_sd"],
        "r_outer": meta_sm["r"], "r_outer_sd": meta_sm["r_sd"],
        "mae_cv": meta_sm["mae"], "r2_cv": meta_sm["r2"], "r_cv": meta_sm["r"],
        "mae_held": meta_sm["mae"], "r2_held": meta_sm["r2"]})

    downstream_predictions(tables, meta_pred).to_parquet(
        OUT/"panel_a_clock_predictions.parquet", index=False)
    pd.DataFrame(perf).to_csv(OUT/"panel_a_clock_performance.csv", index=False)
    meta.merge(meta_pred, on=["participant_id", "repeat", "true_age",
               "cross_omic_age_spread"], validate="one_to_one").to_parquet(
        OUT/"four_omics_meta_dataset.parquet", index=False)
    tables[PARALLEL].to_parquet(OUT/"metabolomics_parallel_predictions.parquet", index=False)
    rep_metrics.to_csv(OUT/"meta_repeat_metrics.csv", index=False)
    pd.DataFrame(details).to_json(OUT/"meta_cv_details.json",
                                  orient="records", indent=2)
    scaler = final_model.named_steps["scale"]
    enet = final_model.named_steps["model"]
    scaled_coef = dict(zip(META_COLS, map(float, enet.coef_)))
    raw_coef_values = enet.coef_ / scaler.scale_
    raw_intercept = float(enet.intercept_ - np.sum(enet.coef_ * scaler.mean_ / scaler.scale_))
    diag = {"layer1_source": str(LAYER1_FILE),
            "layer1_feature_policy": LAYER1_FEATURE_POLICY,
            "layer1_prediction_column_by_omic": PRED_COL_BY_OMIC,
            "fusion_omics": FUSION, "metabolomics_excluded_from_layer2": True,
            "old_five_omics_intersection": OLD_FIVE_OMICS_N,
            "minimum_four_omics_intersection": MIN_FOUR_OMICS_N,
            "observed_four_omics_intersection": n,
            "intersection_by_repeat": {str(k): int(v) for k, v in counts.items()},
            "participants_age_spread_gt_1y": int((age_spread > 1).sum()),
            "final_meta_parameters": final_model.get_params(), "final_meta_training_n": final_n,
            "final_scaled_coefficients": scaled_coef,
            "final_raw_age_scale_coefficients": dict(zip(META_COLS, map(float, raw_coef_values))),
            "final_raw_age_scale_intercept": raw_intercept}
    (OUT/"diagnostics.json").write_text(
        json.dumps(diag, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    draw_architecture(n)
    print("\nDone. Metabolomics never entered Layer 2.")


if __name__ == "__main__":
    main()
