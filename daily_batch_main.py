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
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

from log import setup_logging, cleanup_old_output_dirs

logger = logging.getLogger("news_data")

from database.news_source import load_recent_news
from database.company_source import load_company_info
from database.parquet_store import append_to_target_parquet, resolve_target_parquet_path
from database.sqlite_store import export_parquet_to_sqlite
from database.schema import TARGET_DATA_COLUMNS
from pipelines.common.checkpoint import get_last_processed_info, save_last_processed_info, ensure_dirs, load_checkpoint, save_checkpoint
from pipelines.common.paths import path_from_cfg, project_root
from pipelines.news.extraction import process_chunk
from pipelines.company.mapping_service import build_or_load_company_vector_db, process_one_chunk, normalize_company_name
from pipelines.company.preprocess import keep_latest_per_cust_no, rename_companies_df_columns

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

    logger.warning("batch.dry_run=false 로 변경 후 재실행하세요.")
    sys.exit(0)


async def run_daily_batch():
    # CWD가 아니라 이 파일 위치(프로젝트 루트) 기준으로 conf를 찾는다.
    # 다른 폴더/워크플로우에서 import 해서 호출해도 동작하도록.
    config_dir = os.path.join(project_root(), "conf")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config")

    daily_output_dir = path_from_cfg(cfg, "daily_output_dir")
    company_info_path = path_from_cfg(cfg, "company_info_path")
    last_seq_path = path_from_cfg(cfg, "last_seq_file")
    log_dir = path_from_cfg(cfg, "log_dir")

    start_time = time.time()
    # 기본은 오늘 날짜. BATCH_RUN_DATE(YYYYMMDD) 환경변수로 덮어쓰면
    # 해당 날짜 폴더(output/daily_batch/<날짜>)를 그대로 이어서 처리한다.
    # (예: 어제 추출까지 끝났는데 매핑에서 실패한 run을 LLM 재호출 없이 재개)
    today_str = os.environ.get("BATCH_RUN_DATE") or datetime.now().strftime("%Y%m%d")
    setup_logging(log_dir=log_dir, today_str=today_str)
    logger.info("========== 일배치 시작 (%s) ==========", today_str)
    cleanup_old_output_dirs(daily_output_dir)

    news_source_path = path_from_cfg(cfg, "news_source_path")
    target_parquet_path = resolve_target_parquet_path(cfg)

    today_dir = os.path.join(daily_output_dir, today_str)
    raw_parquet_path = os.path.join(today_dir, f"raw_news_{today_str}.parquet")
    extracted_parts_dir = os.path.join(today_dir, "extracted_parts")
    mapped_parts_dir = os.path.join(today_dir, "mapped_parts")
    checkpoint_file = os.path.join(today_dir, "batch_checkpoint.json")

    ensure_dirs(today_dir, extracted_parts_dir, mapped_parts_dir)

    # 2. 신규 데이터 덤프 (원천 parquet -> 로컬 파케이, 증분 필터 적용)
    last_seq, last_part_basc_dt = get_last_processed_info(last_seq_path)
    logger.info("마지막 처리 seq: %s (파티션: %s)", f"{last_seq:,}", last_part_basc_dt)

    dump_size = 0
    if not os.path.exists(raw_parquet_path):
        logger.info("신규 뉴스 데이터 원천 파일에서 로드 중: %s", news_source_path)
        news_df = load_recent_news(news_source_path, last_seq, last_part_basc_dt)

        if news_df.empty:
            logger.info("신규 뉴스가 없습니다 (증분 필터 결과 0건). 배치를 종료합니다.")
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
    for chunk_idx, record_batch in enumerate(
        parquet_file.iter_batches(batch_size=cfg.batch.chunk_size, columns=available_cols)
    ):
        if chunk_idx < start_chunk_idx:
            continue

        logger.info("[추출] 청크 %d 시작", chunk_idx)
        chunk_df = record_batch.to_pandas()

        extracted_df = await process_chunk(
            cfg=cfg,
            chunk_df=chunk_df,
            chunk_idx=chunk_idx,
            concurrent_limit=cfg.batch.concurrent_limit,
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
    companies_df = load_company_info(company_info_path)
    companies_df = keep_latest_per_cust_no(companies_df)
    companies_df = rename_companies_df_columns(companies_df)
    companies_df_copy = companies_df.copy()
    # 매핑을 위해 clean_cust_nm 컬럼 추가
    companies_df["clean_cust_nm"] = companies_df["cust_nm"].apply(normalize_company_name)

    logger.info("벡터 DB 로드/생성 중...")
    # 날짜 무관 캐시: 회사 마스터가 동일하면 매일 재임베딩하지 않고 재사용 (CPU에서 특히 중요)
    vector_db_path = path_from_cfg(cfg, "vector_db_dir")
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
            similarity_threshold=cfg.batch.similarity_threshold,
        )

        # 필터링 (매핑 결과 없으면 컬럼이 없는 빈 DF가 반환됨)
        if mapped_df.empty:
            final_df = mapped_df.copy()
        else:
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

        if cfg.batch.dry_run:
            run_dry_run(final_df, chunk_idx)

        # parquet 원본에 append
        if not final_df.empty:
            to_append = final_df[COLUMNS].rename(columns=COLUMN_RENAME)
            to_append = to_append[TARGET_DATA_COLUMNS]
            to_append["seq"] = to_append["seq"].astype("Int64")

            appended = append_to_target_parquet(target_parquet_path, to_append)
            total_inserted += appended
            logger.info("[적재완료] 청크 %d — parquet append 성공: %s건 → %s", chunk_idx, f"{appended:,}", target_parquet_path)
        else:
            logger.info("[적재] 청크 %d — 적재 대상 없음", chunk_idx)

        checkpoint["last_mapped_chunk"] = chunk_idx
        save_checkpoint(checkpoint_file, checkpoint)

    # 5. 모든 처리가 끝난 후 seq 업데이트
    save_last_processed_info(new_max_seq, new_max_part_dt, last_seq_path)
    logger.info("last_seq 업데이트 완료: %s (파티션: %s)", f"{new_max_seq:,}", new_max_part_dt)

    # 6. 공유 공간 SQLite export (output/target parquet → .db)
    target_sqlite_path = path_from_cfg(cfg, "target_sqlite_path")
    export_parquet_to_sqlite(target_parquet_path, target_sqlite_path)

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
        "parquet append 성공:", f"{total_inserted:,}",
        "소요시간:", elapsed,
    )

def main():
    """동기 진입점. 외부(워크플로우)에서 import 해서 호출하는 용도."""
    asyncio.run(run_daily_batch())


if __name__ == "__main__":
    main()
