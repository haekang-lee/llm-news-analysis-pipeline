# Daily news enrichment pipeline

금융권 업무 관점에서 **뉴스 본문을 구조화**하는 일 단위 배치 예시입니다. Hive에 적재된 뉴스를 읽고, 대규모 언어모델(LLM)로 기업 후보·카테고리·요약을 뽑은 뒤, 내부 기업 마스터와 맞춰 정제된 결과를 다시 Hive에 적재하는 흐름을 보여 줍니다.

## What it does

- **추출**: 비동기 LLM 호출로 기사별 핵심 정보를 JSON 형태로 정리
- **매핑**: 규칙 기반 정규화 매칭 + 임베딩 유사 검색(FAISS)으로 기업 코드 연결
- **적재**: 체크포인트·청크 처리로 장애 후 이어 실행 가능

## Stack

Python · async LLM(OpenAI 호환 클라이언트) · Pandas/PyArrow · Hive(JDBC) · LangChain 계열 벡터 검색 · Hydra 설정

## Flow (conceptual)

```
Hive (뉴스) → 로컬 parquet 덤프 → LLM 배치 추출 → 기업 매핑/필터 → Hive (결과)
```

세부 디렉터리는 `daily_batch_main.py`(오케스트레이션), `pipelines/news/`, `pipelines/company/`, `database/`, `queries/`에서 확인하면 됩니다.

## Note for viewers

실행에는 **Hive 접속 정보, LLM 엔드포인트, 회사 마스터 데이터 등** 환경이 필요합니다. 민감한 설정과 운영용 프롬프트는 이 저장소에 포함하지 않았고, 참고용으로 `prompts/news_prompt.example.py`만 제공합니다.

이 코드는 과거 과제·실험을 정리한 **포트폴리오 참고용**이며, 그대로 clone 해서 동작한다는 의미보다 설계 의도와 스택 구성을 보여 주는 데 초점을 두었습니다.
