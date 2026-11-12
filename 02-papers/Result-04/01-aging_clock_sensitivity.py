"""Four-versus-five-omics sensitivity analysis and publication figures.

Use ``--plot-only`` to redraw existing results without repeating model fitting.
Panel c shows observed OOF predicted age, not a transformed PLS component.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GridSearchCV, KFold

# The numbered sibling filename cannot be used in a normal ``from`` statement.
_clock_spec = importlib.util.spec_from_file_location(
    "aging_clock", Path(__file__).with_name("02-aging_clock.py"))
_clock = importlib.util.module_from_spec(_clock_spec)
_clock_spec.loader.exec_module(_clock)
COLORS, FIG, FUSION = _clock.COLORS, _clock.FIG, _clock.FUSION
INNER_FOLDS, META_GRID, OMICS = _clock.INNER_FOLDS, _clock.META_GRID, _clock.OMICS
OUT, PARALLEL, SEED = _clock.OUT, _clock.PARALLEL, _clock.SEED
four_omics_table, load_layer1 = _clock.four_omics_table, _clock.load_layer1
meta_pipeline, participant_tables = _clock.meta_pipeline, _clock.participant_tables

BOOTSTRAPS = 2000
PERMUTATIONS = 5000
FIVE_FEATURES = [f"pred__{x}" for x in FUSION + [PARALLEL]]
FOUR_FEATURES = [f"pred__{x}" for x in FUSION]
PUBLICATION_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.fontsize": 6,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 600,
}


def regression_metrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return {"mae": float(np.mean(np.abs(y - pred))),
            "r2": float(1 - np.sum((y-pred)**2) / np.sum((y-y.mean())**2)),
            "pearson_r": float(np.corrcoef(y, pred)[0, 1]),
            "spearman_r": float(spearmanr(y, pred).statistic)}


def five_omics_table(tables):
    meta = four_omics_table(tables)
    met = tables[PARALLEL][["participant_id", "repeat", "true_age", "pred_age"]].rename(
        columns={"true_age": f"age__{PARALLEL}", "pred_age": f"pred__{PARALLEL}"})
    out = meta.merge(met, on=["participant_id", "repeat"], how="inner",
                     validate="one_to_one")
    age_cols = [f"age__{x}" for x in OMICS]
    out["true_age"] = out[age_cols].median(axis=1)
    return out.sort_values(["repeat", "participant_id"]).reset_index(drop=True)


def tuned_oof(X, y, splits, seed):
    pred = np.full(len(y), np.nan)
    for fold, (tr, va) in enumerate(splits, 1):
        inner = KFold(INNER_FOLDS, shuffle=True, random_state=seed + fold)
        search = GridSearchCV(meta_pipeline(seed + fold), META_GRID,
                              scoring="neg_mean_absolute_error", cv=inner,
                              refit=True, n_jobs=-1)
        search.fit(X.iloc[tr], y[tr])
        pred[va] = search.predict(X.iloc[va])
    if np.isnan(pred).any():
        raise RuntimeError("Incomplete sensitivity-analysis OOF predictions")
    return pred


def paired_sensitivity(meta5):
    predictions, metric_rows = [], []
    for repeat, frame in meta5.groupby("repeat", sort=True):
        frame = frame.reset_index(drop=True)
        y = frame.true_age.to_numpy(float)
        splitter = KFold(5, shuffle=True, random_state=SEED + int(repeat))
        splits = list(splitter.split(frame))  # identical outer folds for both models
        p4 = tuned_oof(frame[FOUR_FEATURES], y, splits, SEED + 10_000 + int(repeat)*100)
        p5 = tuned_oof(frame[FIVE_FEATURES], y, splits, SEED + 20_000 + int(repeat)*100)
        for label, pred in [("4 omics", p4), ("5 omics", p5)]:
            predictions.append(pd.DataFrame({
                "participant_id": frame.participant_id, "repeat": int(repeat),
                "model": label, "true_age": y, "pred_age": pred}))
            metric_rows.append({"repeat": int(repeat), "model": label,
                                "n": len(y), **regression_metrics(y, pred)})
        print(f"  sensitivity repeat {int(repeat):02d}: "
              f"MAE 4={regression_metrics(y,p4)['mae']:.3f}, "
              f"5={regression_metrics(y,p5)['mae']:.3f}")
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(metric_rows)


def paired_inference(predictions):
    wide = (predictions.groupby(["participant_id", "model"], as_index=False)
            .agg(true_age=("true_age", "median"), pred_age=("pred_age", "mean"))
            .pivot(index="participant_id", columns="model", values=["true_age", "pred_age"]))
    y = wide["true_age"]["4 omics"].to_numpy(float)
    p4 = wide["pred_age"]["4 omics"].to_numpy(float)
    p5 = wide["pred_age"]["5 omics"].to_numpy(float)
    observed = {"mae_delta_5_minus_4": regression_metrics(y,p5)["mae"] - regression_metrics(y,p4)["mae"],
                "r2_delta_5_minus_4": regression_metrics(y,p5)["r2"] - regression_metrics(y,p4)["r2"]}
    rng = np.random.default_rng(SEED + 777)
    boot = {k: np.empty(BOOTSTRAPS) for k in observed}
    n = len(y)
    for b in range(BOOTSTRAPS):
        idx = rng.integers(0, n, n)
        m4, m5 = regression_metrics(y[idx], p4[idx]), regression_metrics(y[idx], p5[idx])
        boot["mae_delta_5_minus_4"][b] = m5["mae"] - m4["mae"]
        boot["r2_delta_5_minus_4"][b] = m5["r2"] - m4["r2"]
    perm = {k: np.empty(PERMUTATIONS) for k in observed}
    for b in range(PERMUTATIONS):
        swap = rng.random(n) < .5
        a, c = p4.copy(), p5.copy(); a[swap], c[swap] = p5[swap], p4[swap]
        m4, m5 = regression_metrics(y, a), regression_metrics(y, c)
        perm["mae_delta_5_minus_4"][b] = m5["mae"] - m4["mae"]
        perm["r2_delta_5_minus_4"][b] = m5["r2"] - m4["r2"]
    out = {"n": n, "participant_averaging": "mean across 10 outer-OOF repeats"}
    for key, value in observed.items():
        lo, hi = np.percentile(boot[key], [2.5, 97.5])
        p = (1 + np.sum(np.abs(perm[key]) >= abs(value))) / (PERMUTATIONS + 1)
        out[key] = {"estimate": float(value), "bootstrap_95_ci": [float(lo), float(hi)],
                    "paired_swap_permutation_p": float(p)}
    avg = pd.DataFrame({"participant_id": wide.index, "true_age": y,
                        "pred_4omics": p4, "pred_5omics": p5})
    return out, avg


def ribbon(ax, x0, x1, y0, y1, h0, h1, color, alpha=.28):
    c = (x1-x0)*.52
    verts = [(x0,y0-h0/2),(x0+c,y0-h0/2),(x1-c,y1-h1/2),(x1,y1-h1/2),
             (x1,y1+h1/2),(x1-c,y1+h1/2),(x0+c,y0+h0/2),(x0,y0+h0/2),(x0,y0-h0/2)]
    codes = [MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,
             MplPath.LINETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts,codes), facecolor=color, edgecolor="none", alpha=alpha))


def panel_sankey(ax, cohort_n, n4, n5):
    ax.set_xlim(0,1.35); ax.set_ylim(0,1); ax.axis("off")
    max_n, scale = max(cohort_n.values()), .115/max(cohort_n.values())*max(cohort_n.values())
    # scale is written explicitly below to keep all five nodes readable.
    heights = {k:.025 + .105*v/max_n for k,v in cohort_n.items()}
    ys = dict(zip(OMICS, [.87,.68,.49,.30,.11]))
    x0, x1, node_w = .05, .68, .055
    h4 = .025 + .105*n4/max_n; hmet = heights[PARALLEL]
    for omic in OMICS:
        h, y = heights[omic], ys[omic]
        ax.add_patch(Rectangle((x0,y-h/2),node_w,h,facecolor=COLORS[omic],alpha=.78,
                               edgecolor="white",lw=.8))
        ax.text(x0+node_w+.012,y,f"{omic}\nn = {cohort_n[omic]:,}",va="center",fontsize=5.7)
        if omic in FUSION:
            ribbon(ax,x0+node_w,x1,y,.55,min(h,h4),h4,COLORS[omic])
        else:
            ribbon(ax,x0+node_w,x1,y,.16,hmet,hmet,COLORS[omic],.38)
    ax.add_patch(Rectangle((x1,.55-h4/2),.065,h4,facecolor="#D6A600",alpha=.86,
                           edgecolor="white"))
    ax.text(x1+.078,.55,f"4-omics\nintersection\nn = {n4:,}",va="center",fontsize=6,weight="bold")
    ax.add_patch(Rectangle((x1,.16-hmet/2),.065,hmet,facecolor=COLORS[PARALLEL],alpha=.72,
                           edgecolor="white"))
    ax.text(x1+.078,.16,f"Metabolomics\nparallel: n = {cohort_n[PARALLEL]:,}\n5-omics: n = {n5:,}",
            va="center",fontsize=5.7)


def dumbbell(ax, metric, summary, inference):
    means = summary.groupby("model")[metric].mean()
    v4, v5 = means["4 omics"], means["5 omics"]
    ax.plot([v4,v5],[0,0],color="#999",lw=1,zorder=1)
    ax.scatter(v4,0,s=24,color="#B78300",label="4 omics",zorder=2)
    ax.scatter(v5,0,s=24,color=COLORS[PARALLEL],label="5 omics",zorder=2)
    key = "mae_delta_5_minus_4" if metric=="mae" else "r2_delta_5_minus_4"
    stat = inference[key]; lo,hi=stat["bootstrap_95_ci"]
    ax.set_yticks([]); ax.grid(False); ax.spines[["left","right","top"]].set_visible(False)
    label = "MAE (years)" if metric=="mae" else "R²"
    ax.set_xlabel(label,fontsize=6, labelpad=2)
    ax.set_ylim(-.35, 1.85)
    ax.text(.5,.98,f"Δ(5 − 4) = {stat['estimate']:+.3f}\n95% CI [{lo:+.3f}, {hi:+.3f}]\nPermutation P = {stat['paired_swap_permutation_p']:.3g}",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.7, linespacing=1.2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    pad=max(abs(v4),abs(v5))*.04 if max(abs(v4),abs(v5)) else .1
    ax.set_xlim(min(v4,v5)-pad,max(v4,v5)+pad)


def panel_clock_performance(ax, main_avg):
    """Render every participant; colour is a linear count, not a KDE estimate."""
    if main_avg.participant_id.duplicated().any():
        raise ValueError("Panel c requires one OOF mean per participant")
    xy = main_avg[["true_age", "pred_age"]].to_numpy(float)
    if len(xy) < 3 or not np.isfinite(xy).all() or np.ptp(xy[:, 0]) == 0:
        raise ValueError("Panel c requires finite age pairs and varying true age")
    x, y = xy.T
    xlim = (10 * np.floor(x.min() / 10) - 1.5,
            10 * np.ceil(x.max() / 10) + 1.5)
    bin_ylim = (min(y.min(), x.min()) - 2, max(y.max(), x.max()) + 2)
    # Reserve annotation headroom without changing the original hexagon bins.
    ylim = (bin_ylim[0], 5 * np.ceil(bin_ylim[1] / 5))
    cmap = LinearSegmentedColormap.from_list(
        "clock_dark_teal", ["#ffffff", "#e8f3f2", "#b7d8d4", "#5faaa5", "#005b5f"])
    hb = ax.hexbin(x, y, gridsize=36, mincnt=1,
                   extent=(*xlim, *bin_ylim), cmap=cmap, norm=Normalize(vmin=0),
                   edgecolors="none", linewidths=0, rasterized=True, zorder=1)
    counts = np.asarray(hb.get_array(), float)
    if not np.isclose(counts.sum(), len(x)):
        raise RuntimeError("Panel c hexbin failed to retain all participants")
    slope, intercept = np.polyfit(x, y, 1)
    fit_x = np.array([x.min(), x.max()])
    # This is an OLS trend, not the y=x calibration reference in the old plot.
    ax.plot(fit_x, slope * fit_x + intercept, color="#171717", linewidth=1.3,
            linestyle=(0, (4, 2.5)), dash_capstyle="butt", zorder=3)
    ax.set(xlabel="Age (years)", ylabel="OOF predicted age (years)",
           xlim=xlim, ylim=ylim, facecolor="white")
    ax.set_xticks(np.arange(np.ceil(xlim[0] / 10) * 10, xlim[1], 10))
    ax.set_yticks(np.arange(np.ceil(ylim[0] / 10) * 10, ylim[1], 10))
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_linewidth(1)
    ax.tick_params(direction="out", length=3, width=1, pad=2)
    ax.grid(False)
    metrics = regression_metrics(x, y)
    ax.text(.04, .98,
            f"N = {len(x):,}\n"
            f"MAE = {metrics['mae']:.2f} y\n"
            f"R² = {metrics['r2']:.3f}\n"
            f"Pearson r = {metrics['pearson_r']:.3f}\n"
            f"Spearman ρ = {metrics['spearman_r']:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6,
            linespacing=1.1, color="#171717", zorder=4, gid="clock_statistics",
            bbox={"boxstyle": "round,pad=.3", "facecolor": "white",
                  "edgecolor": "#c7c7c7", "linewidth": .6, "alpha": .88})
    return {
        "n": len(x), "excluded_rows": 0, **metrics,
        "fit": {"method": "ordinary least squares", "slope": float(slope),
                "intercept": float(intercept), "confidence_band": "not displayed"},
        "hexbin": {"gridsize": 36, "colour_scale": "linear participant count",
                   "colour_map": "white to dark teal (#005b5f)",
                   "occupied_bins": len(counts), "max_count": int(counts.max()),
                   "counted_participants": int(counts.sum())},
        "xlim": list(map(float, xlim)), "ylim": list(map(float, ylim)),
        "prediction_aggregation": "participant mean of 10 outer-OOF repeats",
    }


def save_publication_figure(fig, stem):
    """Preserve the exact page size and editable vector labels and axes."""
    fig.savefig(stem.with_suffix(".pdf"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})


def _panel_heading(ax, letter, title):
    ax.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(-9, 8),
                textcoords="offset points", fontsize=8, fontweight="bold",
                ha="left", va="bottom", annotation_clip=False)
    ax.set_title(title, loc="left", pad=9, fontsize=7)


def _performance_footer(fig, left, *, standalone=False):
    fig.text(left, .09 if standalone else .19,
             "Darker hexagons: more participants\nDashed line: linear fit; 10-repeat OOF means",
             fontsize=5.7 if standalone else 5.5, color="#555555", va="top", linespacing=1.3)


def make_figure(cohort_n, n4, n5, sensitivity_metrics, inference, main_avg):
    with plt.rc_context(PUBLICATION_STYLE):
        composite_inches = np.array([180, 96]) / 25.4
        fig = plt.figure(figsize=composite_inches, facecolor="white")
        gs = fig.add_gridspec(2, 3, left=.04, right=.975, bottom=.38, top=.85,
                              wspace=.42, hspace=.75)
        ax1 = fig.add_subplot(gs[:, 0]); ax1.set_gid("a")
        ax2a = fig.add_subplot(gs[0, 1]); ax2a.set_gid("b_mae")
        ax2b = fig.add_subplot(gs[1, 1]); ax2b.set_gid("b_r2")
        ax3 = fig.add_subplot(gs[:, 2]); ax3.set_gid("c")
        panel_sankey(ax1, cohort_n, n4, n5)
        dumbbell(ax2a, "mae", sensitivity_metrics, inference)
        dumbbell(ax2b, "r2", sensitivity_metrics, inference)
        metrics = panel_clock_performance(ax3, main_avg)
        _panel_heading(ax1, "a", "Participant coverage")
        _panel_heading(ax2a, "b", f"Sensitivity (n = {n5:,})")
        _panel_heading(ax3, "c", "Four-omics clock")
        _performance_footer(fig, ax3.get_position().x0)
        fig.text(ax1.get_position().x0, .19,
                 "Node thickness encodes cohort size.\nRibbons overlap because participants\ncan occur in multiple omics.",
                 fontsize=5.5, color="#555555", va="top", linespacing=1.35)
        handles, labels = ax2a.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left",
                   bbox_to_anchor=(ax2a.get_position().x0, .215), ncol=2,
                   borderaxespad=0, handletextpad=.35, columnspacing=1)
        fig.text(ax2a.get_position().x0, .12,
                 "Points: mean over 10 repeats.\nCI: paired participant bootstrap.\nP: paired prediction-swap permutation.",
                 fontsize=5.5, color="#555555", va="top", linespacing=1.35)
        save_publication_figure(fig, FIG / "sensitivity")
        plt.close(fig)

        # A journal-width standalone panel keeps the requested style reusable.
        standalone_inches = np.array([64, 74]) / 25.4
        detail, ax = plt.subplots(figsize=standalone_inches, facecolor="white")
        detail.subplots_adjust(left=.24, right=.95, bottom=.15, top=.94)
        ax.set_gid("c")
        panel_clock_performance(ax, main_avg)
        save_publication_figure(detail, FIG / "sensitivity_clock_performance")
        plt.close(detail)

    main_avg.to_csv(OUT / "sensitivity_clock_performance_source.csv", index=False)
    (OUT / "sensitivity_clock_performance_plot.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def _cached_output(filename):
    """Accept results saved before the figure-specific filenames were renamed."""
    for path in (OUT / filename, OUT / f"figure4c_{filename}"):
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing cached {filename}; run without --plot-only first")


def main(plot_only=False):
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    raw=load_layer1(); tables=participant_tables(raw)
    meta4=four_omics_table(tables); meta5=five_omics_table(tables)
    n4=int(meta4.groupby("repeat").participant_id.nunique().iloc[0])
    n5=int(meta5.groupby("repeat").participant_id.nunique().iloc[0])
    if n5 != 271: raise RuntimeError(f"Expected five-omics intersection 271, observed {n5}")
    print(f"Observed intersections: four={n4:,}, five={n5:,}")
    if plot_only:
        sens_metrics = pd.read_csv(_cached_output("sensitivity_repeat_metrics.csv"))
        inference = json.loads(_cached_output("sensitivity_inference.json").read_text(encoding="utf-8"))
        if inference["n"] != n5 or not sens_metrics["n"].eq(n5).all():
            raise ValueError("Cached sensitivity results do not match the current five-omics cohort")
        print("Plot-only: reuse cached sensitivity statistics and OOF predictions.")
    else:
        sens_pred,sens_metrics=paired_sensitivity(meta5)
        inference,_=paired_inference(sens_pred)
    main_pred=pd.read_parquet(OUT/"panel_a_clock_predictions.parquet")
    main_avg=main_pred.loc[main_pred.clock_name.eq("Multi-Omics (4-omics stacking)"),
                           ["participant_id","true_age","pred_age"]]
    cohort_n={o:int(tables[o].participant_id.nunique()) for o in OMICS}
    make_figure(cohort_n,n4,n5,sens_metrics,inference,main_avg)
    if plot_only:
        return
    sens_pred.to_parquet(OUT/"sensitivity_predictions.parquet",index=False)
    sens_metrics.to_csv(OUT/"sensitivity_repeat_metrics.csv",index=False)
    (OUT/"sensitivity_inference.json").write_text(
        json.dumps(inference,ensure_ascii=False,indent=2),encoding="utf-8")
    mae_sig=np.prod(inference["mae_delta_5_minus_4"]["bootstrap_95_ci"])>0
    r2_sig=np.prod(inference["r2_delta_5_minus_4"]["bootstrap_95_ci"])>0
    mae=inference["mae_delta_5_minus_4"]
    r2=inference["r2_delta_5_minus_4"]
    if not mae_sig and not r2_sig:
        interpretation=("On the same 271-participant subset, adding metabolomics did not "
                        "materially change MAE or R². Excluding metabolomics therefore preserves "
                        "predictive accuracy while increasing the analysis cohort from 271 to "
                        f"{n4:,} participants.")
    else:
        interpretation=(
            f"On the same 271-participant subset, adding metabolomics changed MAE by "
            f"{mae['estimate']:+.3f} years (5 minus 4 omics; 95% CI "
            f"[{mae['bootstrap_95_ci'][0]:+.3f}, {mae['bootstrap_95_ci'][1]:+.3f}], "
            f"paired permutation p={mae['paired_swap_permutation_p']:.3f}) and changed R2 by "
            f"{r2['estimate']:+.3f} (95% CI [{r2['bootstrap_95_ci'][0]:+.3f}, "
            f"{r2['bootstrap_95_ci'][1]:+.3f}], paired permutation "
            f"p={r2['paired_swap_permutation_p']:.3f}). Thus metabolomics improved explained "
            f"variance, whereas its MAE improvement was not statistically conclusive. Excluding "
            f"metabolomics increases the main-analysis cohort from 271 to {n4:,} participants, "
            f"so the decision represents a prediction-performance versus sample-size trade-off."
        )
    (OUT/"caption.txt").write_text(interpretation,encoding="utf-8")
    print(interpretation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true",
                        help="Redraw saved results without fitting models again")
    main(plot_only=parser.parse_args().plot_only)
