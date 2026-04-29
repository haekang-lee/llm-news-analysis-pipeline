import ast
import logging
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("torchvision").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from sklearn.metrics.pairwise import cosine_similarity


def normalize_company_name(name):
    if pd.isna(name) or not name:
        return ""

    name = str(name).strip().upper()
    name = name.replace("(주)", "").replace("주식회사", "").replace("INC.", "").replace("CO., LTD.", "")
    name = name.replace(",", "").replace(".", "").replace(" ", "")

    alpha_to_kor = {
        "A": "에이", "B": "비", "C": "씨", "D": "디", "E": "이",
        "F": "에프", "G": "지", "H": "에이치", "I": "아이", "J": "제이",
        "K": "케이", "L": "엘", "M": "엠", "N": "엔", "O": "오",
        "P": "피", "Q": "큐", "R": "알", "S": "에스", "T": "티",
        "U": "유", "V": "브이", "W": "더블유", "X": "엑스", "Y": "와이", "Z": "제트",
    }

    has_korean = any(ord("가") <= ord(c) <= ord("힣") for c in name)
    is_short_english = (not has_korean) and len(name) <= 3
    if has_korean or is_short_english:
        for char, kor in alpha_to_kor.items():
            name = name.replace(char, kor)

    return name.strip().lower()


def build_or_load_company_vector_db(cfg, companies_df, vector_db_path: str):
    # 동명이인 해소를 위해 embedding_model을 리턴해야 하므로 미리 생성
    embeddings = HuggingFaceEmbeddings(
        model_name=cfg.dev.paths.embedding_model,
        model_kwargs={"device": cfg.dev.system.device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": cfg.dev.params.embed_batch_size,
        },
    )
    
    if os.path.exists(vector_db_path):
        try:
            db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
            return db, embeddings
        except Exception as e:
            logger.warning("기존 vector db 로드 실패: %s → 재생성", e)

    docs = []
    for _, row in companies_df.iterrows():
        original_name = str(row["cust_nm"])
        clean_name = normalize_company_name(original_name)
        docs.append(
            Document(
                page_content=clean_name,
                metadata={
                    "cust_no": str(row["cust_no"]),
                    "cust_nm": original_name,
                    "clean_cust_nm": clean_name,
                },
            )
        )

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(vector_db_path)
    return db, embeddings


def resolve_homonyms(df, embedding_model):
    duplicates = df[df.duplicated(subset=["seq", "target_company"], keep=False)]
    if duplicates.empty:
        return df

    resolved_rows = []
    for (_, _), group in duplicates.groupby(["seq", "target_company"]):
        title = str(group.iloc[0].get("title", ""))
        content = str(group.iloc[0].get("content", ""))
        news_context = f"{title} {content[:500]}"

        company_contexts = []
        for _, row in group.iterrows():
            ind = str(row.get("std_inds_clas_nm", ""))
            prod = str(row.get("primy_prod_nm", ""))
            context = f"{ind} {prod}".strip()
            if not context:
                context = str(row.get("cust_nm", ""))
            company_contexts.append(context)

        news_emb = embedding_model.embed_query(news_context)
        comp_embs = embedding_model.embed_documents(company_contexts)
        scores = cosine_similarity([news_emb], comp_embs)[0]
        best_idx = int(np.argmax(scores))
        best_row = group.iloc[best_idx].copy()
        best_row["homonym_score"] = float(scores[best_idx])
        resolved_rows.append(best_row)

    non_duplicates = df.drop_duplicates(subset=["seq", "target_company"], keep=False).copy()
    non_duplicates["homonym_score"] = 0.0
    return pd.concat([non_duplicates, pd.DataFrame(resolved_rows)], ignore_index=True)


def process_one_chunk(
    news_df,
    companies_df,
    companies_df_copy,
    vector_db,
    embedding_model,
    similarity_threshold: float,
):
    def safe_literal_eval(val):
        if isinstance(val, list):
            return val
        if pd.isna(val):
            return []
        try:
            parsed = ast.literal_eval(val)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    if "company_names" not in news_df.columns:
        return pd.DataFrame()

    news_df = news_df.copy()
    news_df["company_names"] = news_df["company_names"].apply(safe_literal_eval)
    exploded_df = news_df.explode("company_names")
    exploded_df = exploded_df.rename(columns={"company_names": "target_company"})
    exploded_df = exploded_df[exploded_df["target_company"].notna() & (exploded_df["target_company"] != "")]
    if exploded_df.empty:
        return pd.DataFrame()

    exploded_df["clean_target_company"] = exploded_df["target_company"].apply(normalize_company_name)

    merged_df = pd.merge(
        exploded_df,
        companies_df,
        left_on="clean_target_company",
        right_on="clean_cust_nm",
        how="left",
    )

    mapped_mask = merged_df["cust_no"].notna()
    mapped_df = merged_df[mapped_mask].copy()
    unmapped_df = merged_df[~mapped_mask].copy()

    mapped_df["mapping_method"] = "exact_normalized"
    mapped_df["similarity_score"] = 1.0
    mapped_df["matched_clean_name"] = mapped_df["clean_cust_nm"]

    if not unmapped_df.empty:
        unique_targets = unmapped_df["target_company"].unique()
        vector_results = {}

        for target in unique_targets:
            if not target:
                continue
            clean_query = normalize_company_name(target)
            if not clean_query:
                continue
            results = vector_db.similarity_search_with_relevance_scores(clean_query, k=1)
            if results:
                doc, score = results[0]
                if score >= similarity_threshold:
                    vector_results[target] = (
                        doc.metadata["cust_nm"],
                        doc.metadata["cust_no"],
                        score,
                        doc.page_content,
                    )

        unmapped_df["mapped_cust_nm"] = unmapped_df["target_company"].map(lambda x: vector_results.get(x, (None, None, 0, None))[0])
        unmapped_df["mapped_cust_no"] = unmapped_df["target_company"].map(lambda x: vector_results.get(x, (None, None, 0, None))[1])
        unmapped_df["similarity_score"] = unmapped_df["target_company"].map(lambda x: vector_results.get(x, (None, None, 0, None))[2])
        unmapped_df["matched_clean_name"] = unmapped_df["target_company"].map(lambda x: vector_results.get(x, (None, None, 0, None))[3])
        unmapped_df["mapping_method"] = "vector_search"

        vector_success_df = unmapped_df[unmapped_df["mapped_cust_no"].notna()].copy()
        if not vector_success_df.empty:
            cols_to_drop = ["cust_no", "cust_nm", "clean_cust_nm"]
            vector_success_df.drop(columns=[c for c in cols_to_drop if c in vector_success_df.columns], inplace=True)
            vector_success_df.rename(columns={"mapped_cust_nm": "cust_nm", "mapped_cust_no": "cust_no"}, inplace=True)
            vector_success_df = pd.merge(vector_success_df, companies_df, on="cust_no", how="left", suffixes=("", "_y"))
            vector_success_df.drop(columns=["cust_nm_y", "clean_cust_nm_y"], inplace=True, errors="ignore")
            final_mapped_df = pd.concat([mapped_df, vector_success_df], ignore_index=True)
        else:
            final_mapped_df = mapped_df
    else:
        final_mapped_df = mapped_df

    if final_mapped_df.empty:
        return pd.DataFrame()

    final_resolved_df = resolve_homonyms(final_mapped_df, embedding_model)
    if final_resolved_df.empty:
        return pd.DataFrame()

    keep_cols = [
        "seq", "site_name", "basc_dt", "title", "content",
        "classification", "category", "doc_sentiment", "summary", "reference", "cust_nm", "cust_no",
    ]
    keep_cols = [c for c in keep_cols if c in final_resolved_df.columns]
    final_resolved_df = final_resolved_df[keep_cols].drop_duplicates()
    if "classification" in final_resolved_df.columns:
        final_resolved_df = final_resolved_df[final_resolved_df["classification"] == "essential"]

    if not final_resolved_df.empty:
        final_resolved_df = final_resolved_df.merge(companies_df_copy, on=["cust_no", "cust_nm"], how="left")
    return final_resolved_df

