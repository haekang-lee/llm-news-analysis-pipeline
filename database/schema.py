"""NEWS_CO_MPPG 타깃 테이블 스키마 (Hive DDL / parquet 공통)."""

ID_COLUMN = "news_mppg_id"

VALID_NEWS_CLAS = {"FINANCE", "BUSINESS", "MANAGEMENT", "INDUSTRY", "RISK", "NOISE"}

# Hive DDL 순서 + 파티션(basc_dt) 맨 끝 — SELECT * / positional INSERT 와 동일
TARGET_DATA_COLUMNS = [
    "seq",
    "site_nm",
    "news_tite",
    "news_ctt",
    "news_clas",
    "doc_sentiment",
    "news_summ",
    "news_ref",
    "etct_cust_no",
    "etct_cust_nm",
    "sglr_ases_modl_clas_cd",
    "std_inds_clas_nm",
    "primy_prod_nm",
    "basc_dt",
]

TARGET_COLUMNS = [ID_COLUMN] + TARGET_DATA_COLUMNS
