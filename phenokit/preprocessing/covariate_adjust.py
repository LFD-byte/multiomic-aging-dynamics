# phenokit/preprocessing/covariate_adjust.py
from __future__ import annotations
from typing import Iterable, List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

def _build_design_matrix(df: pd.DataFrame, covariates: Iterable[str]) -> pd.DataFrame:
    """
    Build design matrix X (with intercept). Categorical columns are one-hot encoded with drop_first=True.
    Returns DataFrame with index aligned to df.
    """
    parts = []
    for cov in covariates:
        if cov not in df.columns:
            raise KeyError(f"Covariate '{cov}' not in dataframe columns")
        ser = df[cov]
        if pd.api.types.is_categorical_dtype(ser) or ser.dtype == object:
            # convert to dummies, drop_first to avoid multicollinearity
            d = pd.get_dummies(ser.astype("category"), prefix=cov, drop_first=True)
            parts.append(d)
        else:
            parts.append(ser.astype(float).to_frame(cov))
    if parts:
        X = pd.concat(parts, axis=1)
    else:
        X = pd.DataFrame(index=df.index)  # empty
    # add intercept
    X.insert(0, "Intercept", 1.0)
    return X


def adjust_for_covariates(
    df: pd.DataFrame,
    cols: Iterable[str],
    covariates: Iterable[str] = ("bmi", "sex"),
    method: str = "robust",
    return_models: bool = False,
    preserve_mean: bool = True,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Dict[str, float]]]]:
    """
    Adjust measurements for given covariates.
    Parameters
    ----------
    df : pd.DataFrame
        输入数据框，必须包含 covariates 列与 cols 列。每一行是一个样本/participant。
    cols : iterable of str
        需要校正的列名列表。
    covariates : iterable of str
        协变量列名（例如 "bmi", "sex"）。类别变量会自动 one-hot 编码。
    method : {"ols", "robust"}
        拟合方法。"ols" 使用 numpy.linalg.lstsq。
        "robust" 在可用 statsmodels 时使用 RLM，否则等价于 "ols"。
    return_models : bool
        是否返回每个列的回归系数字典，格式 {col: {"Intercept":..., "bmi":..., "sex_M":...}}
    preserve_mean : bool
        是否在返回的校正值上保留原始列的均值（常见做法：residual + mean），
        若 False 则直接返回 residual（均值为 0）。
    Returns
    -------
    df_out : pd.DataFrame
        原 df 的拷贝，新增列 "<col>_adj"（校正后值）
    models : dict or None
        若 return_models 为 True，则返回回归系数字典
    """
    df = df.copy()
    X_full = _build_design_matrix(df, covariates)  # includes intercept
    # keep column order
    cols = list(cols)

    models = {} if return_models else None

    for col in cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' is not in dataframe")
        y = df[col]

        # mask rows with missing y or missing any covariate used to fit
        mask_valid = y.notna() & X_full.notna().all(axis=1)
        if mask_valid.sum() == 0:
            # no usable rows -> fill adj as NaN
            df[f"{col}_adj"] = pd.NA
            if return_models:
                models[col] = {}
            continue

        X = X_full.loc[mask_valid].values  # numpy array
        yvec = y.loc[mask_valid].astype(float).values  # vector

        coef = None
        if method == "ols":
            # solve least squares
            # use rcond=None for numpy 1.14+ compatibility
            beta, *_ = np.linalg.lstsq(X, yvec, rcond=None)
            coef = beta
        elif method == "robust":
            try:
                import statsmodels.api as sm  # type: ignore
                model = sm.RLM(yvec, X, M=sm.robust.norms.HuberT())
                res = model.fit()
                coef = res.params
            except Exception:
                # fallback to OLS if statsmodels not available or fit fails
                beta, *_ = np.linalg.lstsq(X, yvec, rcond=None)
                coef = beta
        else:
            raise ValueError("method must be 'ols' or 'robust'")

        # predicted and residuals for valid rows
        y_pred = X.dot(coef)
        resid = yvec - y_pred

        # build adjusted vector: residual + mean(y) if preserve_mean else residual
        if preserve_mean:
            mean_y = np.nanmean(yvec)
            adj_valid = resid + mean_y
        else:
            adj_valid = resid

        # create full series filled with NaN, then assign adjusted values for valid positions
        adj_series = pd.Series(data=np.nan, index=df.index, dtype="float64")
        adj_series.loc[mask_valid] = adj_valid

        df[f"{col}_adj"] = adj_series

        if return_models:
            # map coef back to column names
            cols = X_full.columns.tolist()
            coef_map = {col: float(coef[i]) for i, col in enumerate(cols)}
            models[col] = coef_map

    return df, models
