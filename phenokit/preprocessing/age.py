from __future__ import annotations
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
from .parse_collection_datatime import _parse_collection_datetime

def _build_birth_datetime(df: pd.DataFrame,
                          yob_col: str = "year_of_birth",
                          mob_col: str = "month_of_birth") -> pd.Series:
    """
    根据 year_of_birth, month_of_birth 列构建出生日期（以当月第一天为生日）。
    返回 timezone-aware (UTC) pd.Series
    """
    # 先确保为数值（可能是字符串），coerce errors -> NaN
    yob = pd.to_numeric(df[yob_col], errors="coerce")
    mob = pd.to_numeric(df[mob_col], errors="coerce")

    # 构建日期字符串或直接用 to_datetime on dataframe
    birth_df = pd.DataFrame({
        "year": yob.astype("Int64"),
        "month": mob.astype("Int64"),
        "day": 1
    }, index=df.index)

    # Where year or month is missing, to_datetime 会产生 NaT
    birth_ser = pd.to_datetime(birth_df.assign(day=1), errors="coerce")

    # localize to UTC to match collection datetime tz-aware series
    try:
        if birth_ser.dt.tz is None:
            birth_ser = birth_ser.dt.tz_localize("UTC")
        else:
            birth_ser = birth_ser.dt.tz_convert("UTC")
    except (AttributeError, TypeError):
        birth_ser = pd.to_datetime(birth_ser, errors="coerce").dt.tz_localize("UTC")

    return birth_ser


def compute_precise_age(df_system: pd.DataFrame,
                        popu_parquet_path: str,
                        participant_id_col: str = "participant_id",
                        yob_col: str = "year_of_birth",
                        mob_col: str = "month_of_birth",
                        ts_col: str = "collection_timestamp",
                        date_col: str = "collection_date",
                        sex_col: str = "sex",
                        dropna_age: bool = True) -> pd.DataFrame:
    """
    主函数：计算精确年龄（保留两位小数），并把结果写入新列 'age'（float，NaN 表示无法计算）
    参数:
      - df_system: 来自系统的 dataframe
      - popu_parquet_path: 人口表路径（与你原本相同）
      - dropna_age: 是否在返回前删除 age 缺失的行（保持原行为）
    返回合并并带有 'age' 列的 dataframe
    """
    df = df_system.copy()
    df[participant_id_col] = df[participant_id_col].astype(str)

    # 读人口数据（保持你原来的方式）
    df_popu = pd.read_parquet(popu_parquet_path).reset_index()
    df_popu["participant_id"] = df_popu["participant_id"].astype(str)

    # 只保留有出生年月的
    df_popu = df_popu.dropna(subset=[yob_col, mob_col])

    # 合并
    df_merged = pd.merge(
        df,
        df_popu[[participant_id_col, yob_col, mob_col, sex_col]],
        on=participant_id_col,
        how="left"
    )

    # 再次丢弃没有出生信息的（和你原来逻辑一致）
    df_merged = df_merged.dropna(subset=[yob_col, mob_col]).copy()

    # 解析 collection datetime（向量化）
    df_merged["collection_datetime"] = _parse_collection_datetime(df_merged, ts_col=ts_col, date_col=date_col)

    # 解析出生日期（向量化）
    birth_ser = _build_birth_datetime(df_merged, yob_col=yob_col, mob_col=mob_col)
    df_merged["birth_datetime"] = birth_ser

    # 计算差（以天为单位），先保证两个都是 tz-aware（上面我们 localize/convert 为 UTC）
    coll = df_merged["collection_datetime"]
    birth = df_merged["birth_datetime"]

    # 若 collection_datetime 为 NaT 或 birth 为 NaT，差值为 NaT -> 产生 NaN
    delta_days = (coll - birth).dt.total_seconds() / (3600 * 24)  # float days (可能为 NaN)

    # 年龄（考虑闰年平均 365.25），保留两位小数
    age_years = (delta_days / 365.25).round(2)

    # 对未来出生（delta_days < 0）标为 NaN（与你原实现把负年龄视为 None 一致）
    age_years = age_years.where(delta_days >= 0, other=np.nan)

    df_merged["age"] = age_years

    if dropna_age:
        df_merged = df_merged.dropna(subset=["age"])

    # 整理列（可选）：如果你不想保留中间列，可以删除 birth_datetime
    df_merged = df_merged.drop(columns=["birth_datetime", "collection_datetime", "year_of_birth", "month_of_birth"])

    return df_merged
