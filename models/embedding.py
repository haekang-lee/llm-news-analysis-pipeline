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
        return RemoteAPIEmbeddings(
            base_url=emb_cfg.url,
            model=emb_cfg.model,
            api_key=emb_cfg.get("api_key", "EMPTY"),
            batch_size=cfg.dev.params.embed_batch_size,
        )

    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info("임베딩 백엔드: local (%s, device=%s)", cfg.dev.paths.embedding_model, cfg.dev.system.device)
    return HuggingFaceEmbeddings(
        model_name=cfg.dev.paths.embedding_model,
        model_kwargs={"device": cfg.dev.system.device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": cfg.dev.params.embed_batch_size,
        },
    )
