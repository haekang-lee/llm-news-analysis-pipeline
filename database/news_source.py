import logging

import pandas as pd

from database.wf_csv import read_wf_source

logger = logging.getLogger(__name__)

# 다운스트림(추출/매핑)에서 사용하는 뉴스 원천 컬럼
NEWS_COLUMNS = ["seq", "site_name", "basc_dt", "title", "content", "doc_sentiment", "part_basc_dt"]


def load_recent_news(path: str, last_seq: int, last_part_basc_dt: str) -> pd.DataFrame:
    """
    WF가 내려놓은 최근 N일치 뉴스(CSV 또는 parquet)를 읽어 증분 필터를 적용.

    원천 테이블(get_daily_news_data)과 동일한 기준:
      - channel == 'NEWS'        (컬럼 있을 때만)
      - seq == p_seq             (컬럼 있을 때만)
      - title 이 비어있지 않음
      - seq > last_seq
      - part_basc_dt >= last_part_basc_dt
    """
    df = read_wf_source(path)
    if df.empty:
        logger.warning("뉴스 원천 파일이 비어있거나 존재하지 않음: %s", path)
        return df

    raw_count = len(df)

    if "channel" in df.columns:
        df = df[df["channel"].astype(str).str.upper() == "NEWS"]
    if "p_seq" in df.columns:
        df = df[pd.to_numeric(df["seq"], errors="coerce") == pd.to_numeric(df["p_seq"], errors="coerce")]

    for col in NEWS_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[NEWS_COLUMNS].copy()

    df = df[df["title"].notna() & (df["title"].astype(str).str.strip() != "")]

    df["seq"] = pd.to_numeric(df["seq"], errors="coerce")
    df = df[df["seq"].notna()]
    df["seq"] = df["seq"].astype("int64")
    df = df[df["seq"] > int(last_seq)]

    df = df[df["part_basc_dt"].astype(str) >= str(last_part_basc_dt)]

    df = df.reset_index(drop=True)
    logger.info(
        "뉴스 원천 로드: 원본 %s건 → 증분 필터(seq>%s, part_basc_dt>=%s) 후 %s건",
        f"{raw_count:,}",
        f"{int(last_seq):,}",
        last_part_basc_dt,
        f"{len(df):,}",
    )
    return df
