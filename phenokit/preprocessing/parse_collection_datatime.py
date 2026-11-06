import pandas as pd

def _parse_collection_datetime(df: pd.DataFrame,
                               ts_col: str = "collection_timestamp",
                               date_col: str = "collection_date") -> pd.Series:
    """
    根据优先列解析 collection datetime：
      - 若有 collection_timestamp 列，认为它是 tz-aware 或 tz-naive 的 Timestamp，直接转换为 datetime
      - 否则尝试按格式 '%d-%m-%Y' 解析 collection_date
    返回 pd.Series（dtype datetime64[ns, UTC] 或 datetime64[ns] 然后本函数会本地化/转为 UTC）
    """
    if ts_col in df.columns:
        ser = pd.to_datetime(df[ts_col], errors="coerce")
    elif date_col in df.columns:
        ser = pd.to_datetime(df[date_col], format="%d-%m-%Y", errors="coerce")
    else:
        # 全缺
        ser = pd.Series([pd.NaT] * len(df), index=df.index, dtype="datetime64[ns]")

    # 如果是 pandas 的 DatetimeIndex/Series，用 dt 属性处理 tz
    # 如果 tz-naive -> localize to UTC; 如果 tz-aware -> convert to UTC
    try:
        if ser.dt.tz is None:
            ser = ser.dt.tz_localize("UTC")
        else:
            ser = ser.dt.tz_convert("UTC")
    except (AttributeError, TypeError):
        # 保守处理：转为 UTC-naive datetime64 then localize
        ser = pd.to_datetime(ser, errors="coerce").dt.tz_localize("UTC")
    return ser