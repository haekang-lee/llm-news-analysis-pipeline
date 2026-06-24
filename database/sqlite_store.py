import logging
import os
import sqlite3

import pandas as pd

from database.schema import ID_COLUMN, TARGET_COLUMNS

logger = logging.getLogger(__name__)

DEFAULT_INDEX_COLUMNS = [ID_COLUMN, "seq", "etct_cust_no"]


def export_parquet_to_sqlite(
    parquet_path: str,
    db_path: str,
    table_name: str = "news_co_mppg",
    index_columns: list = None,
) -> int:
    """
    output/target parquet 전체를 SQLite로 덮어쓰기 저장하고 인덱스 생성.
    bulk 적재 속도를 위해 journal_mode/synchronous를 일시적으로 끔.
    """
    if not os.path.exists(parquet_path):
        logger.warning("SQLite export 생략 — parquet 없음: %s", parquet_path)
        return 0

    index_columns = index_columns or DEFAULT_INDEX_COLUMNS

    logger.info("SQLite export 시작: %s → %s", parquet_path, db_path)
    df = pd.read_parquet(parquet_path, engine="pyarrow")

    for col in TARGET_COLUMNS:
        if col not in df.columns:
            df[col] = None

    before_count = len(df)
    news_class = df["news_clas"].astype("string").str.strip()
    doc_sentiment = df["doc_sentiment"].astype("string").str.strip()
    df = df[
        news_class.notna()
        & (news_class != "")
        & (~news_class.str.upper().isin(["NOISE", "NONE", "NAN", "NULL"]))
        & doc_sentiment.notna()
        & (doc_sentiment != "")
        & (~doc_sentiment.str.upper().isin(["NONE", "NAN", "NULL"]))
    ].copy()
    logger.info(
        "SQLite export 필터링: news_clas NOISE/None 및 doc_sentiment None 제외 %s → %s건 (-%s)",
        f"{before_count:,}",
        f"{len(df):,}",
        f"{before_count - len(df):,}",
    )

    df = df[TARGET_COLUMNS].astype(str)

    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")
        df.to_sql(table_name, conn, if_exists="replace", index=False)

        for col in index_columns:
            if col in df.columns:
                conn.execute(
                    f'CREATE INDEX IF NOT EXISTS idx_{col} ON {table_name} ("{col}")'
                )
        conn.commit()
    finally:
        conn.close()

    logger.info("SQLite export 완료: %s건 → %s", f"{len(df):,}", db_path)
    return len(df)
