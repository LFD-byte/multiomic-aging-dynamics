from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List, Dict, Any
import pandas as pd
from ...preprocessing.age import compute_precise_age
from ...preprocessing.bmi import attach_bmi_from_anthropometrics

@dataclass
class OmicsPrepConfig:
    # 清洗/筛选配置
    required_nonnull: Tuple[str, ...] = ("bmi", "sex", "age")
    age_range: Tuple[int, int] = (40, 70)  # inclusive
    # 元数据列（可按组学覆盖）
    metadata_cols: Tuple[str, ...] = (
        "participant_id", "cohort", "research_stage", "array_index",
        "collection_date", "collection_timestamp", "timezone",
        "age", "year_of_birth", "month_of_birth", "sex",
        "collection_datetime", "batch", "bmi"
    )
    # 额外可选：保留/丢弃其他列策略以后也能扩展


def load_and_prepare_omics(
    parquet_path: str,
    popu_parquet_path: str,
    anthropometrics_path: str,
    config: Optional[OmicsPrepConfig] = None,
    *,
    return_full: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    通用组学预处理：读 parquet -> 补充 age/bmi -> dropna -> 年龄筛选 -> 切分 meta/feature

    依赖你已有的函数：
      - compute_precise_age(df, popu_parquet_path=...)
      - attach_bmi_from_anthropometrics(df, anthropometrics_path=...)
    """
    cfg = config or OmicsPrepConfig()

    df = pd.read_parquet(parquet_path)
    if verbose:
        print(f"[raw] {parquet_path}: {df.shape}")

    # 你的项目函数：补充 age / bmi
    df = compute_precise_age(df, popu_parquet_path=popu_parquet_path)
    df = attach_bmi_from_anthropometrics(df, anthropometrics_path=anthropometrics_path)

    # dropna（只对实际存在的 required 列生效）
    req_cols = [c for c in cfg.required_nonnull if c in df.columns]
    if req_cols:
        df = df.dropna(subset=req_cols)

    # 年龄筛选（只有 age 存在才做）
    if "age" in df.columns and cfg.age_range is not None:
        lo, hi = cfg.age_range
        df = df[(df["age"] >= lo) & (df["age"] <= hi)]

    if verbose:
        print(f"[cleaned] {df.shape}")

    # 元数据列：只取存在的
    meta_cols = [c for c in cfg.metadata_cols if c in df.columns]
    df_meta = df[meta_cols].copy()
    df_feature = df.drop(columns=meta_cols, errors="ignore").copy()

    out: Dict[str, Any] = {
        "meta": df_meta,
        "feature": df_feature,
        "meta_cols": meta_cols,
        "shape_raw": None,      # 需要的话也可以记录
        "shape_cleaned": df.shape,
    }
    if return_full:
        out["full"] = df
    return out
