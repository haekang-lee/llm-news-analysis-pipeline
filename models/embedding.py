import logging
from typing import List

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class RemoteAPIEmbeddings(Embeddings):
    """
    OpenAI 호환 임베딩 엔드포인트(GPU 서버)를 호출하는 LangChain 호환 임베딩.

    FAISS(from_documents/load_local)와 동명이인 해소 로직이 요구하는
    embed_documents / embed_query 인터페이스를 구현한다.
    langchain_core.embeddings.Embeddings 를 상속해야 FAISS가 쿼리 시
    embed_query 를 호출한다(상속 안 하면 함수로 취급해 호출 에러 발생).
    """

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", batch_size: int = 64):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.batch_size = max(1, int(batch_size))
        self.base_url = base_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [("" if t is None else str(t)) for t in texts]
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend(item.embedding for item in resp.data)
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class FallbackEmbeddings(Embeddings):
    """
    매핑용 임베딩: API 우선 호출, 실패 시 로컬 모델로 전환(세션 내 유지).
    FAISS 쿼리·동명이인 해소(process_one_chunk)에서 사용.
    """

    def __init__(self, cfg):
        self._cfg = cfg
        self._api: Embeddings | None = None
        self._local: Embeddings | None = None
        self._use_local = False
        emb_cfg = cfg.serve.embedding
        logger.info(
            "임베딩 백엔드 (매핑): API 우선 (%s, model=%s), 실패 시 local fallback",
            emb_cfg.url,
            emb_cfg.model,
        )

    def _api_backend(self) -> Embeddings:
        if self._api is None:
            self._api = _load_api_embedding(self._cfg)
        return self._api

    def _local_backend(self) -> Embeddings:
        if self._local is None:
            logger.warning("임베딩 API 사용 불가 → 로컬 모델 fallback (매핑)")
            self._local = _load_local_embedding(self._cfg)
        return self._local

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._use_local:
            return self._local_backend().embed_documents(texts)
        try:
            return self._api_backend().embed_documents(texts)
        except Exception as e:
            logger.warning("임베딩 API 실패 → 로컬 fallback (매핑): %s", e)
            self._use_local = True
            return self._local_backend().embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def _load_local_embedding(cfg):
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info(
        "임베딩 백엔드: local (%s, device=%s)",
        cfg.dev.paths.embedding_model,
        cfg.dev.system.device,
    )
    return HuggingFaceEmbeddings(
        model_name=cfg.dev.paths.embedding_model,
        model_kwargs={"device": cfg.dev.system.device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": cfg.dev.params.embed_batch_size,
        },
    )


def _load_api_embedding(cfg):
    emb_cfg = cfg.serve.embedding
    return RemoteAPIEmbeddings(
        base_url=emb_cfg.url,
        model=emb_cfg.model,
        api_key=emb_cfg.get("api_key", "EMPTY"),
        batch_size=cfg.dev.params.embed_batch_size,
    )


def load_embedding_model(cfg):
    """
    cfg.dev.embedding_mode 에 따라 임베딩 백엔드를 선택.
      - "api"   : 원격 GPU 임베딩 서버(OpenAI 호환) 호출
      - "local" : 로컬 HuggingFace 모델 직접 로드 (기본값)
    """
    mode = str(cfg.dev.get("embedding_mode", "local")).lower()

    if mode == "api":
        emb_cfg = cfg.serve.embedding
        logger.info("임베딩 백엔드: API (%s, model=%s)", emb_cfg.url, emb_cfg.model)
        return _load_api_embedding(cfg)

    return _load_local_embedding(cfg)


def load_embedding_model_for_mapping(cfg):
    """
    기업명 매핑(벡터 검색·동명이인 해소)용 임베딩.
    - embedding_mode=api: API 우선, 실패 시 로컬 fallback
    - embedding_mode=local: 로컬 모델
    """
    mode = str(cfg.dev.get("embedding_mode", "local")).lower()
    if mode != "api":
        return _load_local_embedding(cfg)
    return FallbackEmbeddings(cfg)


def load_embedding_model_for_rebuild(cfg):
    """
    vector db cold-start / 재생성 시 사용할 임베딩 백엔드.
    - embedding_mode=api: API probe 성공 시에만 반환 (실패 시 예외 — 로컬 재빌드 없음)
    - embedding_mode=local: 로컬 모델 (개발용)
    """
    mode = str(cfg.dev.get("embedding_mode", "local")).lower()
    if mode != "api":
        return _load_local_embedding(cfg)

    emb_cfg = cfg.serve.embedding
    logger.info("vector db 재생성 — 임베딩 API 시도 (%s, model=%s)", emb_cfg.url, emb_cfg.model)
    api_emb = _load_api_embedding(cfg)
    api_emb.embed_query("probe")
    logger.info("임베딩 API 사용 (vector db 재생성)")
    return api_emb
