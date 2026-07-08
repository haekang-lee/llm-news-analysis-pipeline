# Daily news enrichment pipeline

워크플로우(WF)가 공유 경로에 내려놓은 **뉴스·기업 마스터 파일**을 읽어, LLM으로 기사별 정보를 추출하고 내부 기업 마스터와 매핑한 뒤 **누적 parquet + SQLite**로 적재하는 일 단위 배치입니다.

운영 배치(`daily_batch_main.py`)는 **Hive JDBC에 직접 연결하지 않습니다.** Hive 덤프 등 보조 작업은 `scripts/`를 사용합니다.

증분 처리(`seq`·`part_basc_dt`)와 청크 단위 중간 저장으로 장시간 배치에서도 **재시작**할 수 있도록 설계했습니다. 상세 설계는 [ARCHITECTURE.md](./ARCHITECTURE.md)를 참고하세요.

## What it does

- **증분 로드**: WF CSV/parquet에서 신규 뉴스만 읽어 `raw_news` parquet 저장
- **추출**: `asyncio` + LLM으로 기업명·분류·카테고리·요약 JSON 추출 (청크별 parquet, 파싱 실패 시 배치 계속)
- **매핑**: 기업명 정규화 → exact match → FAISS 유사 검색 → 동명이인 해소 → `(seq, cust_no)` dedup
- **적재**: `essential` + `cust_no` 있는 건만 target parquet append → SQLite export (downstream용 필터 적용)

## Flow

```
WF 입력 (recent_news, company_info)
    → 증분 raw parquet
    → LLM 추출 (extracted_parts/)
    → 기업 매핑 (mapped_parts/) + FAISS 캐시
    → output/target/NEWS_CO_MPPG.parquet (누적)
    → paths.target_sqlite_path (.db)
```

## Stack

Python · Hydra · asyncio LLM (OpenAI 호환) · GPU 임베딩 API (bge-m3) · Pandas / PyArrow · LangChain FAISS · SQLite

Hive JDBC(`jaydebeapi`)는 **`scripts/` 보조 경로**에서만 사용합니다.

## Repository layout

| 경로 | 역할 |
|------|------|
| `daily_batch_main.py` | 배치 진입점 (덤프 → 추출 → 매핑 → 적재) |
| `conf/config.yaml` | 운영 설정 (`config.example.yaml` 템플릿) |
| `pipelines/news/` | LLM 프롬프트·추출 |
| `pipelines/company/` | FAISS·매핑·동명이인 |
| `models/` | LLM·임베딩 (API / local fallback) |
| `database/` | WF CSV 리더, parquet/SQLite 적재 |
| `scripts/` | Hive 덤프 등 일회성 보조 |
| `queries/` | Hive SQL (scripts 전용) |

## Quick start

### 1. 설정

```bash
cp conf/config.example.yaml conf/config.yaml
# prompts/news_prompt.py 는 news_prompt.example.py 참고해 별도 준비
```

`conf/config.yaml`에서 최소한 아래를 채웁니다.

| 키 | 설명 |
|----|------|
| `paths.news_source_path` | WF 뉴스 파일 (CSV: `\x01` 필드 / `\|\|` 레코드, 또는 parquet) |
| `paths.company_info_path` | 기업 마스터 |
| `paths.target_sqlite_path` | downstream SQLite 경로 |
| `serve.online.*` | LLM 서빙 URL |
| `serve.embedding.*` | 임베딩 API (`dev.embedding_mode=api`) |

### 2. 실행

```bash
python daily_batch_main.py
```

Blueprint 등 외부 WF에서:

```python
from daily_batch_main import main
main()
```

### 3. 실패 run 재개

추출까지 끝난 뒤 매핑에서 실패한 경우, LLM 재호출 없이 이어가려면:

```bash
export BATCH_RUN_DATE=20260702   # 해당 run 날짜 (YYYYMMDD)
python daily_batch_main.py
```

## 임베딩 정책 (요약)

| 용도 | API | 로컬 fallback |
|------|-----|----------------|
| **① FAISS 인덱스 생성** (~8만 기업) | 필수 (실패 시 구 캐시 유지) | cold-start 재빌드 안 함 |
| **② 기업명 매핑** (쿼리·동명이인) | 우선 | API 실패 시 사용 |

자세한 분기표는 [ARCHITECTURE.md §8](./ARCHITECTURE.md#8-임베딩-아키텍처-핵심) 참고.

## Output

| 산출물 | 경로 |
|--------|------|
| 누적 결과 parquet | `output/target/NEWS_CO_MPPG.parquet` |
| downstream SQLite | `paths.target_sqlite_path` |
| FAISS 캐시 | `output/company_vector_db/` |
| 당일 중간 산출물 | `output/daily_batch/{YYYYMMDD}/` |
| 증분 커서 | `output/checkpoint/last_seq.json` |

parquet는 NOISE 포함 **전건 보존**. SQLite export 시에만 NOISE·`doc_sentiment` None 행을 제외합니다.

## Note for viewers

과제·실험을 정리한 **포트폴리오 참고용** 저장소입니다. clone 즉시 재현보다, WF 파일 입력 위에서 LLM·벡터 검색·증분 배치를 어떻게 엮었는지 보여 주는 데 초점을 두었습니다.
