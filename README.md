# Daily news enrichment pipeline

업무 관점에서 **뉴스 본문을 구조화**하는 일 단위 배치 예시입니다. Hive에 적재된 뉴스를 읽고, 대규모 언어모델(LLM)로 기업 후보·카테고리·요약 등을 뽑은 뒤, 내부 기업 마스터와 맞춰 정제된 결과를 다시 Hive에 적재하는 흐름을 보여 줍니다.

비정형 텍스트를 한 번에 쓰기보다, **증분 처리(seq·파티션 체크포인트)** 와 **청크 단위 저장**으로 장시간 배치에서도 재시작 가능한 형태를 의도했습니다.

## What it does

- **추출**: `asyncio` 기반 동시 호출로 기사별 핵심 정보를 JSON 형태로 정리하고, 파서 실패 시에도 배치가 멈추지 않도록 방어 로직을 둠
- **매핑**: 기업명 정규화 후 **정확 매칭**과 **임베딩·FAISS 유사 검색**을 함께 사용하고, 동명이인의 기업명 충돌 시 뉴스·업종 컨텍스트로 후보를 좁히는 식의 후처리 포함
- **필터·적재**: 신용 분석 관점에서 우선순위가 높은 건만 남긴 뒤 Hive 타깃 테이블에 순번을 부여해 적재

## Stack

Python · async LLM(OpenAI 호환 클라이언트) · Pandas / PyArrow · Hive(JDBC, `jaydebeapi`) · LangChain 계열 벡터 검색 · Hydra로 설정 로드

## Flow (conceptual)

```
Hive (뉴스 원천)
    → 로컬 parquet 덤프 (증분 구간)
    → LLM 배치 추출 (청크별 parquet)
    → 기업 매핑 · 우선순위 필터
    → Hive (분석용 결과 테이블)
```

## Repository layout

| 경로 | 역할 |
|------|------|
| `daily_batch_main.py` | 덤프 → 추출 → 매핑 → 적재까지 한 번에 도는 진입점 |
| `pipelines/news/` | 프롬프트 조립, LLM 호출, 응답 파싱 |
| `pipelines/company/` | 기업명 정규화, FAISS, 동명이인 처리 |
| `database/` | Hive 연결·INSERT 헬퍼 |
| `queries/` | 원천 조회 SQL, 결과 DDL 예시 등 |

## 실행에 대해

동작 재현에는 **Hive 접속 정보(JDBC URL·계정·JAR 경로), LLM 서빙 URL, 기업 마스터(parquet 등)** 가 필요합니다. 민감한 설정 파일과 운영용 프롬프트는 저장소에 넣지 않았으며, 형태만 참고할 수 있도록 `prompts/news_prompt.example.py`를 두었습니다. 실제 실행 시에는 로컬에 `conf/config.yaml`과 `prompts/news_prompt.py`를 두는 방식을 전제로 합니다.

## Note for viewers

이 저장소는 과거 과제·실험을 정리한 **포트폴리오 참고용**입니다. clone 즉시 재현 가능 여부보다, 데이터 레이크 위에서 LLM과 벡터 검색을 어떻게 엮었는지 보여 주는 데 초점을 두었습니다.
