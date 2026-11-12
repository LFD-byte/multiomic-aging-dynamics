"""Leakage-controlled linear/nonlinear comparison of five single-omics clocks.

Inputs are already filtered and imputed. Each omic uses all supplied molecular
features except transcriptomics, which uses the top 1,000 per outer training fold
and is trained last. Set AGING_CLOCK_PROFILE=quick for fewer CV folds/repeats;
the feature policy is unchanged. The default runs 5-fold CV x 10 repeats.
"""
from __future__ import annotations

import gc, json, os, warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from scipy.stats import rankdata, t as t_dist, wilcoxon
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import ElasticNetCV, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

try:
    from lightgbm import LGBMRegressor
except ImportError as exc:
    raise ImportError("Run `%pip install lightgbm` in the notebook kernel.") from exc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")
ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ANALYSIS_DIR.parents[1]
OMICS_DIR = PROJECT_DIR / "02-papers" / "01-interim_data" / "omics_data"
OUTPUT_DIR, FIGURE_DIR = ANALYSIS_DIR / "outputs", ANALYSIS_DIR / "figures"

OMICS_FILES = {
    "Metabolomics": "metabolomics_with_metadata.parquet",
    "Lipidomics": "lipidomics_with_metadata.parquet",
    "Gut Microbiome": "gut_microbiome_with_metadata.parquet",
    "Oral Microbiome": "oral_microbiome_with_metadata.parquet",
    "Transcriptomics": "transcriptomics_with_metadata.parquet",
}
# Dict insertion order is the training order; the slowest omic stays last.
OMICS_ORDER = list(OMICS_FILES)
OMICS_LABELS = {"Metabolomics": "Metabolomics",
                "Lipidomics": "Lipidomics",
                "Gut Microbiome": "Gut microbiome",
                "Oral Microbiome": "Oral microbiome",
                "Transcriptomics": "RNA"}
BASE_MODELS = ["Elastic Net", "Spline GAM", "LightGBM"]
ALL_MODELS = BASE_MODELS + ["Nonlinear representative"]
COLORS = {"Elastic Net": "#7A7F87", "Spline GAM": "#377EB8",
          "LightGBM": "#E66C2C", "Nonlinear representative": "#7B3294"}
META_COLS = {"participant_id", "cohort", "research_stage", "array_index",
             "collection_date", "collection_timestamp", "timezone", "age",
             "year_of_birth", "month_of_birth", "sex", "collection_datetime",
             "batch", "bmi", "data_source", "hmo", "pred_age", "omics",
             "collection_time"}


@dataclass(frozen=True)
class AnalysisConfig:
    age_min: int = 40
    age_max: int = 70
    outer_folds: int = 5
    outer_repeats: int = 10
    inner_folds: int = 5
    transcriptomics_features: int = 1000
    spearman_p: float = .10
    gam_p: float = .10
    top_monotonic: int = 1000
    top_nonlinear: int = 1000
    preselect_max: int = 2000
    chunk_size: int = 512
    n_bootstrap: int = 2000
    random_state: int = 5664


PROFILE = os.getenv("AGING_CLOCK_PROFILE", "publication").lower()
CONFIG = (AnalysisConfig() if PROFILE == "publication" else
          AnalysisConfig(outer_folds=3, outer_repeats=1, inner_folds=3,
                         n_bootstrap=200)
          if PROFILE == "quick" else None)
if CONFIG is None:
    raise ValueError("AGING_CLOCK_PROFILE must be 'publication' or 'quick'")


@dataclass
class OmicsMatrix:
    name: str
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: np.ndarray


def load_omics(name):
    """Load preprocessed features as supplied, retaining the 40–70 age cohort."""
    path = OMICS_DIR / OMICS_FILES[name]
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    missing = {"participant_id", "age"} - set(df)
    if missing:
        raise ValueError(f"{name} missing {sorted(missing)}")
    df = df.loc[df.age.between(CONFIG.age_min, CONFIG.age_max)].copy()
    features = [c for c in df.select_dtypes(include=np.number) if c not in META_COLS]
    if not features:
        raise ValueError(f"No numeric molecular features in {name}")
    out = OmicsMatrix(name, df[features].to_numpy(np.float32, copy=True),
                      df.age.to_numpy(float), df.participant_id.astype(str).to_numpy(),
                      np.asarray(features, object))
    for start in range(0, out.X.shape[1], CONFIG.chunk_size):
        block = out.X[:, start:start + CONFIG.chunk_size]
        invalid = np.flatnonzero(~np.isfinite(block).all(axis=0))
        if len(invalid):
            bad_names = out.feature_names[start + invalid[:5]].tolist()
            raise ValueError(
                f"{name}: preprocessed input contains missing/non-finite values in "
                f"{bad_names}. Fix the upstream data; this analysis does not "
                "filter or impute molecular features.")
    print(f"{name}: n={len(out.y):,}, participants={pd.Series(out.groups).nunique():,}, "
          f"features={out.X.shape[1]:,}")
    del df; gc.collect()
    return out


def grouped_splits(y, groups, n_splits, seed):
    """Participant-disjoint splits, stratified by participant median age when possible."""
    gf = pd.DataFrame({"group": groups, "age": y}).groupby("group", sort=False,
                                                                  as_index=False).age.median()
    ug, ga = gf.group.to_numpy(), gf.age.to_numpy()
    if len(ug) < n_splits:
        raise ValueError(f"Only {len(ug)} participants for {n_splits} folds")
    bins = pd.qcut(ga, q=min(5, max(2, len(ug)//n_splits)), labels=False,
                   duplicates="drop")
    counts = pd.Series(bins).value_counts()
    if len(counts) >= 2 and counts.min() >= n_splits:
        splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(ug, bins)
    else:
        splitter = KFold(n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(ug)
    for tr, te in split_iter:
        yield np.flatnonzero(np.isin(groups, ug[tr])), np.flatnonzero(np.isin(groups, ug[te]))


def spearman_screen(X, y):
    """Vectorized Spearman rho without constructing a feature-by-feature matrix."""
    yr = rankdata(y); yc = yr - yr.mean(); yss = np.sum(yc**2)
    rho = np.zeros(X.shape[1])
    for start in range(0, X.shape[1], CONFIG.chunk_size):
        stop = min(start + CONFIG.chunk_size, X.shape[1])
        xr = rankdata(X[:, start:stop], axis=0); xr -= xr.mean(axis=0)
        den = np.sqrt(np.sum(xr**2, axis=0) * yss)
        rho[start:stop] = np.divide(xr.T @ yc, den, out=np.zeros(stop-start), where=den > 0)
    df = max(len(y)-2, 1)
    tv = np.abs(rho) * np.sqrt(df / np.maximum(1-rho**2, 1e-12))
    return rho, 2*t_dist.sf(tv, df=df)


def univariate_gam_pvalue(x, y):
    """Partial F test comparing a linear term with a compact cubic-spline basis."""
    if np.unique(x).size < 6:
        return 1.0
    linear = np.column_stack([np.ones(len(x)), x])
    spl = SplineTransformer(n_knots=4, degree=3, knots="quantile",
                            include_bias=False).fit_transform(x[:, None])
    nonlinear = np.column_stack([np.ones(len(x)), spl])
    rss_l = np.sum((y-linear @ np.linalg.lstsq(linear, y, rcond=None)[0])**2)
    rss_n = np.sum((y-nonlinear @ np.linalg.lstsq(nonlinear, y, rcond=None)[0])**2)
    dfn, dfd = max(nonlinear.shape[1]-linear.shape[1], 1), len(y)-nonlinear.shape[1]
    if dfd <= 0 or rss_n <= 0 or rss_l <= rss_n:
        return 1.0
    return float(f_dist.sf(((rss_l-rss_n)/dfn)/(rss_n/dfd), dfn, dfd))


def select_features(X, y, names, seed, omics):
    """Keep all preprocessed features; rank only transcriptomics in training data."""
    names = np.asarray(names, object)
    n_features = X.shape[1]
    limit = min(CONFIG.transcriptomics_features, n_features)
    if omics != "Transcriptomics" or n_features <= limit:
        return pd.DataFrame({"feature_idx": np.arange(n_features),
                             "feature": names,
                             "selection_reason": "All preprocessed features"})

    # Preserve the existing monotonic/nonlinear ranking, without a second QC or
    # imputation step. P-value thresholds annotate the ranking, not feature counts.
    Xi = np.asarray(X, dtype=float)
    rho, sp = spearman_screen(Xi, y)
    mi = mutual_info_regression(Xi, y, random_state=seed)
    mono = np.argsort(np.abs(rho))[::-1][:max(CONFIG.top_monotonic, limit)]
    nonlin = np.argsort(mi)[::-1][:max(CONFIG.top_nonlinear, limit)]
    candidates = np.unique(np.r_[mono, nonlin, np.flatnonzero(sp < CONFIG.spearman_p)])
    preselect_max = max(CONFIG.preselect_max, limit)
    if len(candidates) > preselect_max:
        r1 = pd.Series(-np.abs(rho[candidates])).rank(method="min").to_numpy()
        r2 = pd.Series(-mi[candidates]).rank(method="min").to_numpy()
        candidates = candidates[np.argsort(np.minimum(r1, r2))[:preselect_max]]
    gp = np.array([univariate_gam_pvalue(Xi[:, j], y) for j in candidates])
    out = pd.DataFrame({"feature_idx": candidates,
                        "feature": names[candidates],
                        "spearman_rho": rho[candidates], "spearman_p": sp[candidates],
                        "mutual_information": mi[candidates], "gam_p": gp})
    out["passes_spearman"] = out.spearman_p < CONFIG.spearman_p
    out["passes_gam"] = out.gam_p < CONFIG.gam_p
    out["selection_score"] = np.minimum(out.spearman_p.clip(1e-300),
                                         out.gam_p.clip(1e-300))
    chosen = out.nsmallest(limit, "selection_score").copy()
    chosen["selection_reason"] = np.select(
        [chosen.passes_spearman & chosen.passes_gam,
         chosen.passes_spearman, chosen.passes_gam],
        ["Spearman+GAM", "Spearman", "GAM"], default="Top-ranked transcriptomics")
    return chosen.sort_values("selection_score").reset_index(drop=True)


def fit_predict_models(Xtr, ytr, gtr, Xte, seed):
    """Tune all hyperparameters only inside the current outer training set."""
    cv = list(grouped_splits(ytr, gtr, CONFIG.inner_folds, seed))
    en = Pipeline([("scaler", StandardScaler()),
                   ("model", ElasticNetCV(  l1_ratio=[0.05, 0.2, 0.5, 0.8, 0.95, 1.0],
                                            alphas=np.logspace(-3, 1, 50),  # alphas below 1e-3 converge poorly
                                            cv=cv,
                                            max_iter=100_000,
                                            precompute=False,       # avoids Gram-matrix validation errors
                                            tol=1e-4,
                                            selection="random",             # usually converges faster with highly correlated features
                                            n_jobs=-1,
                                            random_state=seed,
                                        ))])
    en.fit(Xtr, ytr)
    gam = GridSearchCV(
        Pipeline([("splines", SplineTransformer(degree=3, knots="quantile",
                                                 include_bias=False)),
                  ("scaler", StandardScaler()), ("model", Ridge())]),
        {"splines__n_knots": [4,5], "model__alpha": [.1,1.,10.,100.]},
        scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1)
    gam.fit(Xtr, ytr)
    lgb_grid = [{"model__max_depth":[d], "model__num_leaves":[leaves],
                 "model__learning_rate":[lr], "model__n_estimators":[trees],
                 "model__subsample":[sub], "model__colsample_bytree":[col]}
                for d,leaves,lr,trees,sub,col in
                [(2,7,.03,300,.8,.8),(3,15,.03,400,.8,.8),(4,15,.03,400,.8,.7),
                 (4,31,.02,600,.9,.8),(5,31,.03,400,.8,.8),(5,31,.05,250,.9,.7),
                 (6,31,.03,500,.8,.7),(-1,31,.02,600,.8,.8)]]
    lgb = GridSearchCV(
        Pipeline([("model", LGBMRegressor(objective="regression_l1", verbosity=-1,
                                           n_jobs=1, random_state=seed,
                                           deterministic=True, force_col_wise=True,
                                           subsample_freq=1))]),
        lgb_grid, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1)
    lgb.fit(Xtr, ytr)
    pred = {"Elastic Net": en.predict(Xte), "Spline GAM": gam.predict(Xte),
            "LightGBM": lgb.predict(Xte)}
    representative = "Spline GAM" if gam.best_score_ >= lgb.best_score_ else "LightGBM"
    pred["Nonlinear representative"] = pred[representative]
    tuning = {"elastic_l1_ratio": float(en.named_steps.model.l1_ratio_),
              "elastic_alpha": float(en.named_steps.model.alpha_),
              "gam_inner_mae": float(-gam.best_score_), "gam_params": gam.best_params_,
              "lightgbm_inner_mae": float(-lgb.best_score_),
              "lightgbm_params": lgb.best_params_,
              "nonlinear_representative": representative}
    return pred, tuning


def cross_validate_omic(data):
    preds, selected_rows, tuning_rows = [], [], []
    for repeat in range(1, CONFIG.outer_repeats+1):
        seed = CONFIG.random_state + 1000*repeat
        for fold, (tr, te) in enumerate(grouped_splits(data.y, data.groups,
                                                       CONFIG.outer_folds, seed), 1):
            fold_seed = seed + fold
            selected = select_features(data.X[tr], data.y[tr], data.feature_names,
                                       fold_seed, data.name)
            idx = selected.feature_idx.to_numpy(int)
            print(f"  repeat {repeat:02d}/{CONFIG.outer_repeats}, fold {fold}/"
                  f"{CONFIG.outer_folds}: train={len(tr):,}, test={len(te):,}, "
                  f"selected={len(idx)}")
            pred, tuning = fit_predict_models(data.X[np.ix_(tr,idx)], data.y[tr],
                                               data.groups[tr], data.X[np.ix_(te,idx)],
                                               fold_seed)
            frame = pd.DataFrame({"omics":data.name, "row_id":te,
                                  "participant_id":data.groups[te], "age":data.y[te],
                                  "repeat":repeat, "fold":fold,
                                  "n_features":len(idx),
                                  "nonlinear_representative":tuning["nonlinear_representative"]})
            for model, values in pred.items(): frame[model] = values
            preds.append(frame)
            selected.insert(0,"fold",fold); selected.insert(0,"repeat",repeat)
            selected.insert(0,"omics",data.name); selected_rows.append(selected)
            tuning_rows.append({"omics":data.name,"repeat":repeat,"fold":fold,
                                "n_features":len(idx),
                                **{k:json.dumps(v) if isinstance(v,dict) else v
                                   for k,v in tuning.items()}})
            gc.collect()
    return pd.concat(preds,ignore_index=True), pd.concat(selected_rows,ignore_index=True), pd.DataFrame(tuning_rows)


def metrics_pair(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    y_centered = y - y.mean()
    pred_centered = pred - pred.mean()
    denominator = np.sqrt(np.dot(y_centered, y_centered) *
                          np.dot(pred_centered, pred_centered))
    if len(y) > 2 and denominator > 0:
        pearson_r = float(np.dot(y_centered, pred_centered) / denominator)
    else:
        pearson_r = np.nan
    return mean_absolute_error(y,pred), r2_score(y,pred), pearson_r


def calculate_metrics(predictions):
    fold_rows, repeat_rows = [], []
    for (omics,repeat,fold), part in predictions.groupby(["omics","repeat","fold"]):
        for model in ALL_MODELS:
            mae,r2,pearson_r = metrics_pair(part.age,part[model])
            fold_rows.append({"omics":omics,"repeat":repeat,"fold":fold,"model":model,
                              "MAE":mae,"R2":r2,"PearsonR":pearson_r,
                              "n_test":len(part)})
    for (omics,repeat), part in predictions.groupby(["omics","repeat"]):
        for model in ALL_MODELS:
            mae,r2,pearson_r = metrics_pair(part.age,part[model])
            repeat_rows.append({"omics":omics,"repeat":repeat,"model":model,"MAE":mae,
                                "R2":r2,"PearsonR":pearson_r,"n":len(part),
                                "n_participants":part.participant_id.nunique()})
    return pd.DataFrame(fold_rows),pd.DataFrame(repeat_rows)


def fdr_bh(p):
    p=np.asarray(p,float); order=np.argsort(p); ranked=p[order]
    adj=np.minimum.accumulate((ranked*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    out=np.empty_like(adj); out[order]=np.clip(adj,0,1); return out


def paired_tests(fold_metrics):
    rows=[]
    for omics in OMICS_ORDER:
        part=fold_metrics.loc[fold_metrics.omics==omics]
        for metric in ["MAE","R2","PearsonR"]:
            wide=part.pivot(index=["repeat","fold"],columns="model",values=metric)
            if metric=="MAE":
                diff=wide["Elastic Net"]-wide["Nonlinear representative"]
                contrast="ElasticNet - nonlinear (positive favors nonlinear)"
            else:
                diff=wide["Nonlinear representative"]-wide["Elastic Net"]
                contrast="nonlinear - ElasticNet (positive favors nonlinear)"
            diff = diff.dropna()
            if np.allclose(diff,0): stat,p=0.,1.
            else:
                test=wilcoxon(diff,alternative="greater",zero_method="wilcox")
                stat,p=float(test.statistic),float(test.pvalue)
            sd=diff.std(ddof=1)
            rows.append({"omics":omics,"metric":metric,"contrast":contrast,
                         "mean_paired_advantage":diff.mean(),
                         "median_paired_advantage":diff.median(),
                         "cohen_dz":diff.mean()/sd if sd>0 else np.nan,
                         "wilcoxon_statistic":stat,"p_value":p,
                         "n_paired_folds":len(diff)})
    out=pd.DataFrame(rows); out["q_value_fdr_bh"]=np.nan
    for metric in ["MAE","R2","PearsonR"]:
        mask=out.metric==metric; out.loc[mask,"q_value_fdr_bh"]=fdr_bh(out.loc[mask,"p_value"])
    return out


def cluster_rows(groups,rng):
    unique,inverse=np.unique(groups,return_inverse=True)
    rows=[np.flatnonzero(inverse==i) for i in range(len(unique))]
    return np.concatenate([rows[i] for i in rng.integers(0,len(unique),len(unique))])


def bootstrap_improvement(predictions):
    rng=np.random.default_rng(CONFIG.random_state); rows=[]
    for omics in OMICS_ORDER:
        part=predictions.loc[predictions.omics==omics]
        avg=part.groupby("row_id",as_index=False).agg({"participant_id":"first","age":"first",
             "Elastic Net":"mean","Nonlinear representative":"mean"})
        y=avg.age.to_numpy(); en=avg["Elastic Net"].to_numpy()
        nl=avg["Nonlinear representative"].to_numpy()
        em,er,epr=metrics_pair(y,en); nm,nr,npr=metrics_pair(y,nl)
        ba,bp,br,bpr=[],[],[],[]
        for _ in range(CONFIG.n_bootstrap):
            idx=cluster_rows(avg.participant_id.to_numpy(),rng)
            emb,erb,eprb=metrics_pair(y[idx],en[idx])
            nmb,nrb,nprb=metrics_pair(y[idx],nl[idx])
            ba.append(emb-nmb); bp.append(100*(emb-nmb)/emb); br.append(nrb-erb)
            bpr.append(nprb-eprb)
        for contrast,value,vals in [("MAE improvement (years)",em-nm,ba),
                                    ("MAE improvement (%)",100*(em-nm)/em,bp),
                                    ("R2 improvement",nr-er,br),
                                    ("PearsonR improvement",npr-epr,bpr)]:
            lo,hi=np.quantile(vals,[.025,.975]); rows.append({"omics":omics,"contrast":contrast,
                 "value":value,"ci_low":lo,"ci_high":hi,"n":len(avg),
                 "n_participants":avg.participant_id.nunique()})
    return pd.DataFrame(rows)


def repeat_summary(repeat_metrics):
    return repeat_metrics.groupby(["omics","model"],as_index=False).agg(
        MAE=("MAE","mean"),MAE_sd=("MAE","std"),R2=("R2","mean"),R2_sd=("R2","std"),
        PearsonR=("PearsonR","mean"),PearsonR_sd=("PearsonR","std"),
        n=("n","max"),n_participants=("n_participants","max"))


def star(q): return "***" if q<.001 else "**" if q<.01 else "*" if q<.05 else "ns"


def plot_panel(summary,tests,stem):
    fig,axes=plt.subplots(1,3,figsize=(18.0,4.8)); x=np.arange(len(OMICS_ORDER))
    lookup=summary.set_index(["omics","model"])
    for ax,metric,ylabel in zip(
        axes,["MAE","R2","PearsonR"],
        ["MAE (years; lower is better)","$R^2$ (higher is better)",
         "Pearson $r$ (higher is better)"]):
        values=[]
        for i,omics in enumerate(OMICS_ORDER):
            en=lookup.loc[(omics,"Elastic Net"),metric]; nl=lookup.loc[(omics,"Nonlinear representative"),metric]
            values += [en,nl]; ax.plot([i-.12,i+.12],[en,nl],color="#B8B8B8",lw=2)
            ax.scatter(i-.12,en,s=62,color=COLORS["Elastic Net"],zorder=2,
                       label="Elastic Net" if i==0 else None)
            ax.scatter(i+.12,nl,s=62,color=COLORS["Nonlinear representative"],zorder=2,
                       label="Nonlinear (inner-CV winner)" if i==0 else None)
        offset=max((max(values)-min(values))*.1,.03)
        for i,omics in enumerate(OMICS_ORDER):
            q=tests.query("omics==@omics and metric==@metric").q_value_fdr_bh.iloc[0]
            ax.text(i,max(lookup.loc[(omics,"Elastic Net"),metric],lookup.loc[(omics,"Nonlinear representative"),metric])+offset,
                    star(q),ha="center")
        ax.set_xticks(x,[OMICS_LABELS[o] for o in OMICS_ORDER],rotation=18,ha="right")
        ax.set_ylabel(ylabel); ax.grid(axis="y",color="#E0E0E0",lw=.7)
        ax.spines[["top","right"]].set_visible(False)
        if metric in {"R2","PearsonR"}: ax.axhline(0,color="#555",lw=.8)
    axes[0].legend(frameon=False); fig.suptitle("Linear versus nonlinear aging clocks across five omics",y=1.02)
    fig.text(.5,-.01,"Mean pooled outer-CV performance; stars: paired fold-wise Wilcoxon, BH-FDR across five omics.",
             ha="center",fontsize=8.5,color="#555")
    fig.tight_layout(); fig.savefig(stem.with_suffix(".png"),dpi=300,bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"),bbox_inches="tight"); plt.close(fig)


def plot_forest(bootstrap,stem):
    part=bootstrap.query("contrast=='MAE improvement (%)'").set_index("omics").loc[OMICS_ORDER]
    y=np.arange(len(part))[::-1]; fig,ax=plt.subplots(figsize=(7.2,4.2))
    ax.errorbar(part.value,y,xerr=np.vstack([part.value-part.ci_low,part.ci_high-part.value]),
                fmt="o",color=COLORS["Nonlinear representative"],ecolor="#555",capsize=3)
    ax.axvline(0,color="#777",ls="--"); ax.set_yticks(y,[OMICS_LABELS[o] for o in OMICS_ORDER])
    ax.set_xlabel("MAE improvement of nonlinear model over Elastic Net (%)")
    ax.set_title("Participant-cluster bootstrap 95% confidence intervals")
    ax.grid(axis="x",color="#E5E5E5"); ax.spines[["top","right","left"]].set_visible(False)
    fig.tight_layout(); fig.savefig(stem.with_suffix(".png"),dpi=300,bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"),bbox_inches="tight"); plt.close(fig)


def plot_extended(fold_metrics,stem):
    fig,axes=plt.subplots(3,1,figsize=(13.5,11.5),sharex=True); width=.22
    for ax,metric in zip(axes,["MAE","R2","PearsonR"]):
        for i,omics in enumerate(OMICS_ORDER):
            for j,model in enumerate(BASE_MODELS):
                vals=fold_metrics.query("omics==@omics and model==@model")[metric]
                bp=ax.boxplot(vals,positions=[i+(j-1)*width],widths=width*.78,patch_artist=True,showfliers=False)
                bp["boxes"][0].set(facecolor=COLORS[model],edgecolor="white")
                bp["medians"][0].set(color="white",linewidth=1.3)
        ylabel={"MAE":"MAE (years)","R2":"$R^2$","PearsonR":"Pearson $r$"}[metric]
        ax.set_ylabel(ylabel)
        ax.grid(axis="y",color="#E5E5E5"); ax.spines[["top","right"]].set_visible(False)
    axes[1].set_xticks(np.arange(len(OMICS_ORDER)),[OMICS_LABELS[o] for o in OMICS_ORDER],rotation=18,ha="right")
    handles=[plt.Line2D([0],[0],marker="s",linestyle="",color=COLORS[m],label=m) for m in BASE_MODELS]
    axes[0].legend(handles=handles,frameon=False,ncol=3)
    fig.suptitle("Extended Data: complete outer-fold performance distributions")
    fig.tight_layout(); fig.savefig(stem.with_suffix(".png"),dpi=300,bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"),bbox_inches="tight"); plt.close(fig)


def run_analysis():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True); FIGURE_DIR.mkdir(parents=True,exist_ok=True)
    all_pred=[]; all_selected=[]; all_tuning=[]; cohorts=[]
    print(f"Profile={PROFILE}; {asdict(CONFIG)}")
    print("Each omic uses its own full cohort; no cross-omic intersection.")
    print(f"Preprocessed inputs: {OMICS_DIR}; no additional filtering or imputation.")
    print(f"Features: transcriptomics top {CONFIG.transcriptomics_features:,}; other omics all.")
    print("Training order: " + " -> ".join(OMICS_ORDER))
    for omics in OMICS_ORDER:
        print(f"\n--- {omics} ---"); data=load_omics(omics)
        cohorts.append({"omics":omics,"n_observations":len(data.y),
                        "n_participants":pd.Series(data.groups).nunique(),
                        "n_candidate_features":data.X.shape[1]})
        pred,selected,tuning=cross_validate_omic(data)
        all_pred.append(pred); all_selected.append(selected); all_tuning.append(tuning)
        del data; gc.collect()
    predictions=pd.concat(all_pred,ignore_index=True); selected=pd.concat(all_selected,ignore_index=True)
    tuning=pd.concat(all_tuning,ignore_index=True); cohorts=pd.DataFrame(cohorts)
    fold_metrics,repeat_metrics=calculate_metrics(predictions); summary=repeat_summary(repeat_metrics)
    tests=paired_tests(fold_metrics); bootstrap=bootstrap_improvement(predictions)
    predictions.to_parquet(OUTPUT_DIR/"nonlinearity_oof_predictions.parquet",index=False)
    for obj,name in [(selected,"nonlinearity_selected_features.csv"),(tuning,"nonlinearity_inner_tuning.csv"),
                     (cohorts,"nonlinearity_cohort_sizes.csv"),(fold_metrics,"nonlinearity_fold_metrics.csv"),
                     (repeat_metrics,"nonlinearity_repeat_metrics.csv"),(summary,"nonlinearity_model_metrics.csv"),
                     (tests,"nonlinearity_paired_tests.csv"),(bootstrap,"nonlinearity_bootstrap_improvement.csv")]:
        obj.to_csv(OUTPUT_DIR/name,index=False)
    (OUTPUT_DIR/"nonlinearity_run_config.json").write_text(
        json.dumps({"profile":PROFILE, **asdict(CONFIG),
                    "omics_dir":str(OMICS_DIR), "training_order":OMICS_ORDER,
                    "input_preprocessing":"already_filtered_and_imputed",
                    "feature_policy":{o:CONFIG.transcriptomics_features
                                      if o == "Transcriptomics" else "all"
                                      for o in OMICS_ORDER},
                    "input_feature_counts":dict(zip(cohorts.omics,
                                                     cohorts.n_candidate_features))},
                   indent=2),encoding="utf-8")
    plot_panel(summary,tests,FIGURE_DIR/"panel_b_linear_vs_nonlinear")
    plot_forest(bootstrap,FIGURE_DIR/"panel_b_mae_improvement_forest")
    plot_extended(fold_metrics,FIGURE_DIR/"extended_data_gam_lightgbm")
    print("\nIndependent cohort sizes:\n",cohorts.to_string(index=False))
    print("\nPaired tests:\n",tests.to_string(index=False,float_format=lambda x:f"{x:.4g}"))
    return predictions,summary,tests,bootstrap


if __name__ == "__main__":
    predictions,metrics,tests,bootstrap=run_analysis()
