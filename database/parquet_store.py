import logging
import os

import pandas as pd

from database.schema import ID_COLUMN, TARGET_COLUMNS

logger = logging.getLogger(__name__)


def table_to_parquet_filename(table: str) -> str:
    """HVDBLAB_h3730059.NEWS_CO_MPPG → NEWS_CO_MPPG.parquet"""
    return f"{table.split('.')[-1]}.parquet"


def resolve_target_parquet_path(cfg, output_dir: str = None) -> str:
    from pipelines.common.paths import path_from_cfg

    dir_ = output_dir or path_from_cfg(cfg, "target_parquet_dir")
    filename = str(cfg.paths.target_parquet_file)
    return os.path.join(dir_, filename)


def format_basc_dt(series: pd.Series) -> pd.Series:
    """parquet 원본과 동일하게 YYYY.MM.DD 문자열로 통일."""
    parsed = pd.to_datetime(series, errors="coerce", format="%Y.%m.%d")
    mask = parsed.isna()
    if mask.any():
        parsed2 = pd.to_datetime(series.loc[mask], errors="coerce", format="%Y%m%d")
        parsed.loc[mask] = parsed2
    return parsed.dt.strftime("%Y.%m.%d").fillna(series.astype(str))


def prepare_target_df(df: pd.DataFrame) -> pd.DataFrame:
    """append/INSERT 전 Hive DDL 컬럼 순서로 정렬."""
    out = df.reset_index(drop=True).copy()
    if "basc_dt" in out.columns:
        out["basc_dt"] = format_basc_dt(out["basc_dt"])
    for col in TARGET_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[TARGET_COLUMNS]


def _read_existing(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path, engine="pyarrow")


def append_to_target_parquet(path: str, df: pd.DataFrame, id_column: str = ID_COLUMN) -> int:
    """
    타깃 parquet(원본)에 신규 행을 append.
    news_mppg_id는 기존 MAX + 1부터 순차 부여.
    """
    if df.empty:
        return 0

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    existing = _read_existing(path)
    max_id = int(existing[id_column].max()) if not existing.empty and id_column in existing.columns else 0

    out = prepare_target_df(df)
    out[id_column] = range(max_id + 1, max_id + 1 + len(out))
    out = out[TARGET_COLUMNS]

    if not existing.empty:
        for col in TARGET_COLUMNS:
            if col not in existing.columns:
                existing[col] = None
        existing = existing[TARGET_COLUMNS]
        combined = pd.concat([existing, out], ignore_index=True)
    else:
        combined = out

    tmp_path = path + ".tmp"
    combined.to_parquet(tmp_path, index=False, engine="pyarrow")
    os.replace(tmp_path, path)

    logger.info(
        "parquet append 완료: +%s건 (MAX(%s)=%s → %s) → %s",
        f"{len(out):,}",
        id_column,
        f"{max_id:,}",
        f"{max_id + len(out):,}",
        path,
    )
    return len(out)
