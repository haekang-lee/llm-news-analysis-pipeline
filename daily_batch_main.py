import asyncio
import logging
import os
import time
import warnings
import pandas as pd
import pyarrow.parquet as pq
from hydra import compose, initialize_config_dir
from datetime import datetime

# 파이썬 기본 경고 무시
warnings.filterwarnings("ignore")

# 외부 라이브러리 로거 레벨 조정 (API 호출 로그 등 불필요한 INFO 로그 숨김)
logging.getLogger("torchvision").setLevel(logging.ERROR)
logging.getLogger("jaydebeapi").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

from log import setup_logging, cleanup_old_output_dirs

logger = logging.getLogger("news_data")

from database.hive_client import fetch_data, insert_dataframe_with_id
from queries.sql_queries import get_daily_news_data, get_companies_data
from pipelines.common.checkpoint import get_last_processed_info, save_last_processed_info, ensure_dirs, load_checkpoint, save_checkpoint
from pipelines.news.extraction import process_chunk
from pipelines.company.mapping_service import build_or_load_company_vector_db, process_one_chunk, normalize_company_name

# ── 설정 ──────────────────────────────────────────────────────────────────────
DAILY_OUTPUT_DIR  = "output/daily_batch"
COMPANY_INFO_PATH = "database/company_info.parquet"
CONCURRENT_LIMIT  = 50
SIMILARITY_THRESHOLD = 0.9
CHUNK_SIZE        = 1000

# 디버깅 모드 (True: DB 적재 직전에 결과를 화면에 출력하고 종료)
DRY_RUN = False

# parquet 컬럼명 → Hive DDL 컬럼명 매핑
COLUMN_RENAME = {
    "site_name"     : "site_nm",
    "title"         : "news_tite",
    "content"       : "news_ctt",
    "category"      : "news_clas",  # category(6종)를 DB의 news_clas 컬럼에 저장
    "doc_sentiment" : "doc_sentiment",
    "summary"       : "news_summ",
    "reference"     : "news_ref",
    "cust_no"       : "etct_cust_no",
    "cust_nm"       : "etct_cust_nm",
}

COLUMNS = [
    "seq", "site_name", "basc_dt", "title", "content",
    "category", "doc_sentiment", "summary", "reference",
    "cust_no", "cust_nm",
    "sglr_ases_modl_clas_cd", "std_inds_clas_nm", "primy_prod_nm",
]


def run_dry_run(final_df: pd.DataFrame, chunk_idx: int) -> None:
    """DRY_RUN 모드: DB 적재 없이 결과를 로그로 출력하고 종료."""
    import sys

    logger.warning("=" * 80)
    logger.warning("[DRY RUN] DB 적재 생략 — 청크 %d 결과 미리보기", chunk_idx)
    logger.warning("=" * 80)

    if final_df.empty:
        logger.warning("필터링 후 남은 데이터가 없습니다 (모두 NOISE 이거나 매핑 실패).")
    else:
        logger.warning("적재 대상: %s건", f"{len(final_df):,}")
        sample_df = final_df[COLUMNS].rename(columns=COLUMN_RENAME).head(5)
        preview_cols = ["seq", "site_nm", "news_tite", "news_clas", "doc_sentiment", "etct_cust_nm", "etct_cust_no"]
        preview_cols = [c for c in preview_cols if c in sample_df.columns]
        with pd.option_context("display.max_columns", None, "display.width", 1000, "display.max_colwidth", 50):
            logger.warning("\n%s", sample_df[preview_cols].to_string(index=False))
        logger.warning("전체 컬럼: %s", list(sample_df.columns))

    logger.warning("DRY_RUN=False 로 변경 후 재실행하세요.")
    sys.exit(0)


async def run_daily_batch():
    start_time = time.time()
    today_str = datetime.now().strftime("%Y%m%d")
    setup_logging(today_str=today_str)
    logger.info("========== 일배치 시작 (%s) ==========", today_str)
    cleanup_old_output_dirs(DAILY_OUTPUT_DIR)

    # 1. 설정 로드 및 디렉토리 준비
    config_dir = os.path.abspath("conf")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config")

    target_table      = cfg.tables.target
    news_source_table = cfg.tables.news_source
    companies_table   = cfg.tables.companies

    today_dir = os.path.join(DAILY_OUTPUT_DIR, today_str)
    raw_parquet_path = os.path.join(today_dir, f"raw_news_{today_str}.parquet")
    extracted_parts_dir = os.path.join(today_dir, "extracted_parts")
    mapped_parts_dir = os.path.join(today_dir, "mapped_parts")
    checkpoint_file = os.path.join(today_dir, "batch_checkpoint.json")

    ensure_dirs(today_dir, extracted_parts_dir, mapped_parts_dir)

    # 2. 신규 데이터 덤프 (DB -> 로컬 파케이)
    last_seq, last_part_basc_dt = get_last_processed_info()
    logger.info("마지막 처리 seq: %s (파티션: %s)", f"{last_seq:,}", last_part_basc_dt)

    dump_size = 0
    if not os.path.exists(raw_parquet_path):
        logger.info("신규 뉴스 데이터 DB에서 조회 중...")
        # 테스트용으로 3000건만 가져오려면 아래 쿼리에 limit=3000 을 추가
        # query = get_daily_news_data(last_seq, last_part_basc_dt, source_table=news_source_table, limit=3000)
        query = get_daily_news_data(last_seq, last_part_basc_dt, source_table=news_source_table)
        news_df = fetch_data(cfg, query)

        if news_df.empty:
            logger.info("신규 뉴스가 없습니다. 배치를 종료합니다.")
            return

        dump_size = len(news_df)
        news_df.to_parquet(raw_parquet_path, index=False)
        logger.info("[덤프완료] 신규 뉴스 %s건 → %s", f"{dump_size:,}", raw_parquet_path)
    else:
        logger.info("기존 덤프 파일 발견, 이어서 진행: %s", raw_parquet_path)

    # 최대 seq와 그에 해당하는 part_basc_dt 확인 (나중에 업데이트용)
    parquet_file = pq.ParquetFile(raw_parquet_path)
    df_meta = parquet_file.read(["seq", "part_basc_dt"]).to_pandas()
    
    # seq가 가장 큰 행의 seq와 part_basc_dt를 찾음
    max_idx = df_meta["seq"].idxmax()
    new_max_seq = int(df_meta.loc[max_idx, "seq"])
    new_max_part_dt = str(df_meta.loc[max_idx, "part_basc_dt"])

    # 3. 기업명 추출 (청크 단위)
    logger.info("[1단계] 기업명 추출 시작 (청크 단위)...")
    checkpoint = load_checkpoint(checkpoint_file, {"last_extracted_chunk": -1, "last_mapped_chunk": -1})
    start_chunk_idx = checkpoint["last_extracted_chunk"] + 1

    available_cols = ["seq", "site_name", "basc_dt", "title", "content", "doc_sentiment"]

    total_extracted_news = 0
    for chunk_idx, record_batch in enumerate(parquet_file.iter_batches(batch_size=CHUNK_SIZE, columns=available_cols)):
        if chunk_idx < start_chunk_idx:
            continue

        logger.info("[추출] 청크 %d 시작", chunk_idx)
        chunk_df = record_batch.to_pandas()

        extracted_df = await process_chunk(
            cfg=cfg,
            chunk_df=chunk_df,
            chunk_idx=chunk_idx,
            concurrent_limit=CONCURRENT_LIMIT,
            output_dir=extracted_parts_dir,
            save_parquet=True
        )

        # 기업명이 1개 이상 추출된 뉴스 건수
        extracted_count = int(
            extracted_df["company_names"]
            .apply(lambda v: isinstance(v, list) and len(v) > 0 or (isinstance(v, str) and v not in ("[]", "", "None")))
            .sum()
        ) if extracted_df is not None and "company_names" in extracted_df.columns else 0
        total_extracted_news += extracted_count
        logger.info("[추출] 청크 %d 완료 — 기업명 추출 성공: %s건 / %s건", chunk_idx, f"{extracted_count:,}", f"{len(chunk_df):,}")

        checkpoint["last_extracted_chunk"] = chunk_idx
        save_checkpoint(checkpoint_file, checkpoint)

    logger.info("[1단계 완료] 누적 기업명 추출 성공 뉴스: %s건", f"{total_extracted_news:,}")

    # 4. 기업 매핑 및 DB 적재 (청크 단위)
    logger.info("[2단계] 기업 매핑 및 DB 적재 시작...")
    logger.info("최신 기업 데이터 로드 중...")
    # companies_df = fetch_data(cfg, get_companies_data(table=companies_table))
    companies_df = pd.read_parquet(COMPANY_INFO_PATH, engine="pyarrow")
    companies_df_copy = companies_df.copy()
    # 매핑을 위해 clean_cust_nm 컬럼 추가
    companies_df["clean_cust_nm"] = companies_df["cust_nm"].apply(normalize_company_name)

    logger.info("벡터 DB 로드/생성 중...")
    vector_db_path = os.path.join(today_dir, "company_vector_db")
    vector_db, embedding_model = build_or_load_company_vector_db(cfg, companies_df, vector_db_path)

    start_map_idx = checkpoint["last_mapped_chunk"] + 1
    # total_chunks 계산 시, parquet_file이 여러 row_group으로 나뉘어 있지 않으면 1이 나올 수 있음.
    # 따라서 추출된 part 파일의 개수를 기반으로 total_chunks를 다시 계산하는 것이 안전함.
    extracted_files = [f for f in os.listdir(extracted_parts_dir) if f.startswith("part_") and f.endswith(".parquet")]
    total_chunks = len(extracted_files)

    if total_chunks == 0:
        logger.warning("매핑할 추출 파일이 없습니다.")

    total_mapped = 0
    total_inserted = 0
    for chunk_idx in range(start_map_idx, total_chunks):
        extracted_file = os.path.join(extracted_parts_dir, f"part_{chunk_idx:05d}.parquet")
        if not os.path.exists(extracted_file):
            continue  # 추출 결과가 없는 빈 청크

        logger.info("[매핑 & 적재] 청크 %d 시작", chunk_idx)
        extracted_df = pd.read_parquet(extracted_file)

        if extracted_df.empty:
            checkpoint["last_mapped_chunk"] = chunk_idx
            save_checkpoint(checkpoint_file, checkpoint)
            continue

        # 매핑
        mapped_df = process_one_chunk(
            news_df=extracted_df,
            companies_df=companies_df,
            companies_df_copy=companies_df_copy,
            vector_db=vector_db,
            embedding_model=embedding_model,
            similarity_threshold=SIMILARITY_THRESHOLD
        )

        # 필터링
        final_df = mapped_df[
            (mapped_df["classification"] == "essential") &
            (mapped_df["cust_no"].notna()) &
            (mapped_df["cust_no"] != "")
        ].copy()

        chunk_mapped = len(mapped_df) if mapped_df is not None and not mapped_df.empty else 0
        chunk_insert = len(final_df)
        total_mapped += chunk_mapped
        logger.info("[매핑] 청크 %d — 기업 매핑 성공: %s건", chunk_idx, f"{chunk_mapped:,}")

        # 로컬 백업 저장
        mapped_file = os.path.join(mapped_parts_dir, f"mapped_part_{chunk_idx:05d}.parquet")
        final_df.to_parquet(mapped_file, index=False)

        if DRY_RUN:
            run_dry_run(final_df, chunk_idx)

        # DB INSERT
        if not final_df.empty:
            final_df = final_df[COLUMNS].rename(columns=COLUMN_RENAME)
            final_df["seq"] = final_df["seq"].astype("Int64")

            insert_dataframe_with_id(cfg, final_df, table=target_table, batch_size=1000, verbose=False)
            total_inserted += chunk_insert
            logger.info("[적재완료] 청크 %d — DB INSERT 성공: %s건", chunk_idx, f"{chunk_insert:,}")
        else:
            logger.info("[적재] 청크 %d — 적재 대상 없음", chunk_idx)

        checkpoint["last_mapped_chunk"] = chunk_idx
        save_checkpoint(checkpoint_file, checkpoint)

    # 5. 모든 처리가 끝난 후 seq 업데이트
    save_last_processed_info(new_max_seq, new_max_part_dt)
    logger.info("last_seq 업데이트 완료: %s (파티션: %s)", f"{new_max_seq:,}", new_max_part_dt)

    elapsed = time.time() - start_time
    logger.info(
        "========== 일배치 종료 ==========\n"
        "  %-22s %s건\n"
        "  %-22s %s건\n"
        "  %-22s %s건\n"
        "  %-22s %s건\n"
        "  %-22s %.2f초",
        "덤프 데이터 크기:", f"{dump_size:,}",
        "기업명 추출 성공:", f"{total_extracted_news:,}",
        "기업 매핑 성공:", f"{total_mapped:,}",
        "DB INSERT 성공:", f"{total_inserted:,}",
        "소요시간:", elapsed,
    )

if __name__ == "__main__":
    asyncio.run(run_daily_batch())
