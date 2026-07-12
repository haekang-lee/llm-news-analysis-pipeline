import logging

import pandas as pd

from database.wf_csv import read_wf_source

logger = logging.getLogger(__name__)


def load_company_info(path: str) -> pd.DataFrame:
    """
    WF가 내려놓은 기업 마스터(CSV 또는 parquet)를 읽는다.
    이후 keep_latest_per_cust_no / rename_companies_df_columns 전처리는 daily_batch_main에서 수행.
    """
    df = read_wf_source(path)
    if df.empty:
        logger.warning("기업 마스터 파일이 비어있거나 존재하지 않음: %s", path)
        return df

    if "cust_no" in df.columns:
        # CSV에서 숫자로 읽히면 앞자리 0이 사라지므로 문자열로 통일
        df["cust_no"] = df["cust_no"].astype("string").str.strip()

    logger.info("기업 마스터 로드: %s건, 컬럼 %d개 → %s", f"{len(df):,}", len(df.columns), path)
    return df
