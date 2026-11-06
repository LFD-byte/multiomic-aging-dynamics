import pandas as pd
from .parse_collection_datatime import _parse_collection_datetime


def attach_bmi_from_anthropometrics(
    df_system: pd.DataFrame,
    anthropometrics_path: str = "/home/ec2-user/studies/hpp_datasets/anthropometrics/anthropometrics.parquet",
    ts_col: str = "collection_timestamp",
    date_col: str = "collection_date",
    max_diff_days: int = 180,
    id_col: str = "participant_id",
    group_keys: tuple[str, ...] = ("participant_id", "cohort", "research_stage", "array_index"),
    bmi_col: str = "bmi"
) -> pd.DataFrame:
    """
    从 anthropometrics 文件中，为 df_system 匹配并聚合 BMI（采样时间差 <= max_diff_days）。

    返回：原 df_system 的行级数据 + 聚合后的 bmi 列（按 group_keys）
    """
    df_system = df_system.copy()

    df_anthro = pd.read_parquet(anthropometrics_path).reset_index()

    # 统一 ID 类型，避免 merge 漏匹配
    df_system[id_col] = df_system[id_col].astype(str)
    df_anthro[id_col] = df_anthro[id_col].astype(str)

    # 解析采集时间
    target_dt = _parse_collection_datetime(df_system, ts_col, date_col)
    anthro_dt = _parse_collection_datetime(df_anthro, ts_col, date_col)

    df_system["_collection_dt_system"] = target_dt
    df_anthro["_collection_dt_anthro"] = anthro_dt

    # 只保留需要的列再 merge（更清晰也更省内存）
    df_merged = df_system.merge(
        df_anthro[[id_col, "_collection_dt_anthro", bmi_col]],
        on=id_col,
        how="left",
    )

    # 向量化计算时间差（NaT 会自然传播成 NaT）
    diff = (df_merged["_collection_dt_system"] - df_merged["_collection_dt_anthro"]).abs()
    df_merged = df_merged.loc[diff <= pd.Timedelta(days=max_diff_days)].copy()

    # 按 group_keys 聚合 BMI（mean）
    bmi_group = (
        df_merged.groupby(list(group_keys), as_index=False)[bmi_col]
        .mean()
        .dropna(subset=[bmi_col])
    )

    # 回填聚合后的 bmi：避免重复列，先去掉明细 bmi 再 merge
    df_merged = df_merged.drop(columns=[bmi_col], errors="ignore")
    df_out = df_merged.merge(bmi_group, on=list(group_keys), how="left")

    # 清理临时列
    df_out = df_out.drop(columns=["_collection_dt_system", "_collection_dt_anthro"], errors="ignore")
    return df_out
