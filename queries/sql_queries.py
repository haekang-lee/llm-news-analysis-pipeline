def get_daily_news_data(last_seq: int, last_part_basc_dt: str, source_table: str = "", limit: int = None):
    limit_clause = f"LIMIT {limit}" if limit else ""
    return f"""
    SELECT seq, site_name, basc_dt, title, content, doc_sentiment, part_basc_dt
    FROM {source_table}
    WHERE part_basc_dt >= '{last_part_basc_dt}'
      AND seq > {last_seq}
      AND channel = 'NEWS'
      AND seq = p_seq
      AND title IS NOT NULL
    {limit_clause}
    """

def get_companies_data(table: str = ""):
    return f"""
    SELECT *
    FROM {table}
    """

def get_create_news_result_table(table: str = "") -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {table} (
        news_mppg_id            BIGINT      COMMENT '매핑 고유 ID (1부터 순차)',
        seq                     BIGINT      COMMENT '뉴스 일련번호',
        site_nm                 STRING      COMMENT '뉴스 사이트명',
        news_tite               STRING      COMMENT '뉴스 제목',
        news_ctt                STRING      COMMENT '뉴스 본문',
        news_clas               STRING      COMMENT '뉴스 분류 (FINANCE, BUSINESS, MANAGEMENT, INDUSTRY, RISK, NOISE)',
        doc_sentiment           STRING      COMMENT '뉴스 감성 분석',
        news_summ               STRING      COMMENT '뉴스 요약',
        news_ref                STRING      COMMENT '참고 정보',
        etct_cust_no            STRING      COMMENT '매핑된 고객번호',
        etct_cust_nm            STRING      COMMENT '매핑된 기업명',
        sglr_ases_modl_clas_cd  STRING      COMMENT '단독심사모델분류코드',
        std_inds_clas_nm        STRING      COMMENT '표준산업분류명',
        primy_prod_nm           STRING      COMMENT '주제품명'
    )
    PARTITIONED BY (basc_dt STRING COMMENT '기사 발행일자 (YYYYMMDD)')
    STORED AS PARQUET
    TBLPROPERTIES ('parquet.compression'='SNAPPY')
    """

def get_max_news_mppg_id(table: str = "") -> str:
    return f"SELECT COALESCE(MAX(news_mppg_id), 0) AS max_id FROM {table}"