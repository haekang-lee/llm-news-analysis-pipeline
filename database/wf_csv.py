"""워크플로우(Hive→FILE) export CSV 공통 리더."""

import glob
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

# WF export 구분자 (리터럴 '\x01' 네 글자 + 레코드 '||')
WF_FIELD_SEP = "\\x01"
WF_ROW_SEP = "||"


def read_wf_csv(
    path: str,
    field_sep: str = WF_FIELD_SEP,
    row_sep: str = WF_ROW_SEP,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    WF가 떨군 CSV를 읽는다. pandas.read_csv는 멀티문자 행 구분자(||)를 못 다루므로 직접 파싱.
    레코드(||)로 먼저 나눠 본문 내 줄바꿈을 보존한 뒤, 각 레코드를 필드(\\x01)로 분리.
    """
    with open(path, encoding=encoding, errors="replace") as f:
        raw = f.read()

    raw = raw.rstrip("\n")
    if not raw:
        return pd.DataFrame()

    records = [r for r in raw.split(row_sep) if r != ""]
    rows = [r.split(field_sep) for r in records]
    if not rows:
        return pd.DataFrame()

    header, data = rows[0], rows[1:]
    ncol = len(header)
    good = [r for r in data if len(r) == ncol]
    bad = len(data) - len(good)
    if bad:
        logger.warning("WF CSV 컬럼수(%d) 불일치로 제외된 레코드: %d건 (%s)", ncol, bad, path)

    return pd.DataFrame(good, columns=header)


def read_wf_source(path: str) -> pd.DataFrame:
    """확장자에 따라 CSV(WF 포맷) 또는 parquet을 읽는다. 디렉터리면 내부 파일 전체를 concat."""
    if os.path.isdir(path):
        csv_files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if csv_files:
            return pd.concat([read_wf_csv(f) for f in csv_files], ignore_index=True)
        pq_files = sorted(glob.glob(os.path.join(path, "*.parquet")))
        if pq_files:
            return pd.concat([pd.read_parquet(f, engine="pyarrow") for f in pq_files], ignore_index=True)
        return pd.DataFrame()

    if not os.path.exists(path):
        return pd.DataFrame()

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return read_wf_csv(path)
    return pd.read_parquet(path, engine="pyarrow")
