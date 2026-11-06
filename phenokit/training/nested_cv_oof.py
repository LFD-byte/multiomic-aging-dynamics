import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr, randint, uniform, loguniform

def compute_metrics(y_true, y_pred):
    """在同一组 y_true / y_pred 上计算所有指标（点估计）。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # 相关系数与p值
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    spearman_r, spearman_p = spearmanr(y_true, y_pred)

    return {
        "r2": float(r2),
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
    }

def bootstrap_metric_ci(y_true, y_pred, metric_fn, n_boot=1000, seed=42):
    """
    通用bootstrap：对任意 metric_fn(y_true, y_pred)->float 计算95%CI
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = metric_fn(y_true[idx], y_pred[idx])

    hat = metric_fn(y_true, y_pred)
    ci_low, ci_high = np.percentile(samples, [2.5, 97.5])
    return float(hat), (float(ci_low), float(ci_high)), samples

def nested_cv_oof_predictions_lasso(
    X, y, outer_splits=3, inner_splits=5,
    n_iter=10, random_state=42, n_jobs=-1
):
    X = np.asarray(X)
    y = np.asarray(y)

    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=random_state)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(max_iter=5000, random_state=random_state))
    ])

    # LASSO 只有 alpha
    param_dist = {
        "model__alpha": loguniform(1e-4, 1e2),
    }

    oof_pred = np.full_like(y, fill_value=np.nan, dtype=float)
    best_params_per_fold = []
    best_inner_score_per_fold = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=n_iter,              # 10 次随机搜索
            scoring="r2",
            cv=inner_cv,                # 5-fold
            random_state=random_state + fold,
            n_jobs=n_jobs,
            refit=True
        )
        search.fit(X_tr, y_tr)

        # out-of-sample 预测
        oof_pred[test_idx] = search.predict(X_te)

        best_params_per_fold.append(search.best_params_)
        best_inner_score_per_fold.append(search.best_score_)

        print(
            f"[Outer Fold {fold}] "
            f"best_inner_r2={search.best_score_:.4f}, "
            f"best_alpha={search.best_params_['model__alpha']:.6g}"
        )

    assert np.all(np.isfinite(oof_pred)), \
        "有样本没有得到 out-of-sample 预测，请检查CV拆分。"

    return oof_pred, best_params_per_fold, best_inner_score_per_fold


def nested_cv_oof_predictions_elasticnet(
    X, y, outer_splits=3, inner_splits=5,
    n_iter=10, random_state=42, n_jobs=-1
):
    X = np.asarray(X)
    y = np.asarray(y)

    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=random_state)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(max_iter=5000, random_state=random_state))
    ])

    param_dist = {
        "model__alpha": loguniform(1e-4, 1e2),
        "model__l1_ratio": uniform(0.05, 0.95),
    }

    oof_pred = np.full_like(y, fill_value=np.nan, dtype=float)
    best_params_per_fold = []
    best_inner_score_per_fold = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="r2",
            cv=inner_cv,
            random_state=random_state + fold,
            n_jobs=n_jobs,
            refit=True
        )
        search.fit(X_tr, y_tr)

        oof_pred[test_idx] = search.predict(X_te)

        best_params_per_fold.append(search.best_params_)
        best_inner_score_per_fold.append(search.best_score_)

        print(f"[Outer Fold {fold}] best_inner_r2={search.best_score_:.4f}, "
              f"best_alpha={search.best_params_['model__alpha']:.6g}, "
              f"best_l1_ratio={search.best_params_['model__l1_ratio']:.3f}")

    assert np.all(np.isfinite(oof_pred)), "有样本没有得到 out-of-sample 预测，请检查CV拆分。"
    return oof_pred, best_params_per_fold, best_inner_score_per_fold

def nested_cv_oof_predictions_lightgbm(
    X, y,
    outer_splits=3, inner_splits=5,
    n_iter=10, random_state=42, n_jobs=-1
):
    """
    外层3折：产生OOF预测
    内层5折 + RandomizedSearchCV(10次)：每个外层训练折内调参
    """
    X = np.asarray(X)
    y = np.asarray(y)

    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=random_state)

    base_model = lgb.LGBMRegressor(
        boosting_type="gbdt",
        n_estimators=1000,        # 可保留你原来的
        random_state=random_state,
        verbose=-1,
        n_jobs=n_jobs
    )

    # 随机搜索空间（你可以按论文/经验调整范围）
    # param_dist = {
    #     "num_leaves": randint(8, 64),
    #     "max_depth": randint(2, 10),
    #     "learning_rate": loguniform(1e-3, 0.2),
    #     "reg_alpha": loguniform(1e-4, 5.0),
    #     "reg_lambda": loguniform(1e-4, 5.0),
    #     "min_child_samples": randint(5, 150),
    #     "subsample": uniform(0.6, 0.4),         # 0.6~1.0
    #     "colsample_bytree": uniform(0.5, 0.5),  # 0.5~1.0
    # }
    param_dist = {
        "num_leaves": randint(8, 64),
        "max_depth": randint(2, 10),
        "learning_rate": loguniform(1e-3, 0.2),
        "reg_alpha": loguniform(1e-2, 20.0),
        "reg_lambda": loguniform(1e-4, 20.0),
        "min_child_samples": randint(5, 200),
        "subsample": uniform(0.6, 1.0),         # 0.6~1.0
        "colsample_bytree": uniform(0.5, 0.95),  # 0.5~1.0
    }

    oof_pred = np.full_like(y, fill_value=np.nan, dtype=float)
    best_params_per_fold = []
    best_inner_score_per_fold = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=n_iter,          # 10次
            scoring="r2",
            cv=inner_cv,            # 5折
            random_state=random_state + fold,
            n_jobs=n_jobs,
            refit=True
        )
        search.fit(X_tr, y_tr)

        oof_pred[test_idx] = search.predict(X_te)

        best_params_per_fold.append(search.best_params_)
        best_inner_score_per_fold.append(search.best_score_)

        bp = search.best_params_
        print(f"[Outer Fold {fold}] best_inner_r2={search.best_score_:.4f}, "
              f"num_leaves={bp['num_leaves']}, max_depth={bp['max_depth']}, "
              f"lr={bp['learning_rate']:.4g}")

    assert np.all(np.isfinite(oof_pred)), "有样本没有得到 out-of-sample 预测，请检查CV拆分。"
    return oof_pred, best_params_per_fold, best_inner_score_per_fold

def run_nested_cv_oof(
    X, y,
    model='lasso',
    outer_splits=3, inner_splits=5, n_iter=10,
    n_boot=1000, random_state=42, n_jobs=-1,
    bootstrap_all_metrics=False  # True则所有指标都bootstrap出CI
):
    if model == 'lasso':
        oof_pred, best_params, best_scores = nested_cv_oof_predictions_lasso(
                X, y,
                outer_splits=outer_splits,
                inner_splits=inner_splits,
                n_iter=n_iter,
                random_state=random_state,
                n_jobs=n_jobs
            )
    elif model == 'elastic_net':
        oof_pred, best_params, best_scores = nested_cv_oof_predictions_elasticnet(
            X, y,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            n_iter=n_iter,
            random_state=random_state,
            n_jobs=n_jobs
        )
    elif model == 'lightgbm':
        oof_pred, best_params, best_scores = nested_cv_oof_predictions_lightgbm(
            X, y,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            n_iter=n_iter,
            random_state=random_state,
            n_jobs=n_jobs
        )

    # 点估计：全部指标
    metrics = compute_metrics(y, oof_pred)

    # CI：默认只按原文给 r2 的bootstrap CI
    r2_hat, r2_ci95, _ = bootstrap_metric_ci(
        y, oof_pred,
        metric_fn=lambda a, b: r2_score(a, b),
        n_boot=n_boot,
        seed=random_state
    )
    metrics["r2"] = r2_hat
    metrics["r2_ci95"] = r2_ci95

    # 可选：所有指标也做CI（会更耗时）
    if bootstrap_all_metrics:
        for k, fn in {
            "mse": lambda a, b: mean_squared_error(a, b),
            "rmse": lambda a, b: np.sqrt(mean_squared_error(a, b)),
            "mae": lambda a, b: mean_absolute_error(a, b),
            "pearson_r": lambda a, b: pearsonr(a, b)[0],
            "spearman_r": lambda a, b: spearmanr(a, b)[0],
        }.items():
            hat, ci, _ = bootstrap_metric_ci(y, oof_pred, fn, n_boot=n_boot, seed=random_state)
            metrics[k] = hat
            metrics[f"{k}_ci95"] = ci
        # p值一般不建议bootstrap（统计意义更复杂），这里保留原始点估计

    print(f"OOF metrics:")
    for key in ["r2", "mse", "rmse", "mae", "pearson_r", "pearson_p", "spearman_r", "spearman_p"]:
        print(f"  {key}: {metrics[key]}")
    print(f"  r2_ci95: {metrics['r2_ci95']}")

    results = {
        "n": int(len(y)),
        "metrics": metrics,
        "oof_pred": oof_pred,
        "best_params_per_fold": best_params,
        "best_inner_r2_per_fold": best_scores
    }

    return results