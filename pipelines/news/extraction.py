import asyncio
import json
import logging
import os
import warnings
from typing import Dict

import pandas as pd

warnings.filterwarnings("ignore")

from models.invoke import invoke_api
from prompts.news_prompt import get_news_analysis_prompt

logger = logging.getLogger(__name__)


def build_user_prompt(row: dict) -> str:
    """
    prompts/news_prompt.py 템플릿 오류에도 안전하게 동작하도록 fallback 지원.
    """
    try:
        user_template = get_news_analysis_prompt("user_template")
        return user_template.format(
            Id=row.get("seq"),
            date=row.get("basc_dt"),
            title=row.get("title", ""),
            content=row.get("content", ""),
        )
    except Exception:
        return (
            "Analyze the following news article. Output only the result in JSON format.\n\n"
            f"ID: {row.get('seq')}\n"
            f"Date: {row.get('basc_dt')}\n"
            f"Title: {row.get('title', '')}\n"
            f"Content: {row.get('content', '')}\n"
        )


def safe_parse_response(raw_content: str) -> Dict:
    cleaned = str(raw_content).replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        open_cnt = cleaned.count("{")
        close_cnt = cleaned.count("}")
        if open_cnt > close_cnt:
            repaired = cleaned + ("}" * (open_cnt - close_cnt))
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                return {}
        return {}


async def analyze_single_news(cfg, row: dict, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        system_prompt = get_news_analysis_prompt("system")
        user_prompt = build_user_prompt(row)
        seq = row.get("seq")

        raw_content = await invoke_api(cfg, user_prompt, system_prompt)
        if not raw_content or str(raw_content).startswith("Failure:"):
            return {
                "seq": seq,
                "company_names": "[]",
                "classification": None,
                "category": None,
                "summary": None,
                "llm_error": str(raw_content),
            }

        parsed = safe_parse_response(raw_content)
        result_item = parsed.get("results", {})
        if not isinstance(result_item, dict):
            result_item = {}

        company_names = result_item.get("company_names", [])
        if isinstance(company_names, list):
            company_names = str(company_names)
        elif company_names is None:
            company_names = "[]"
        else:
            company_names = str(company_names)

        return {
            "seq": seq,
            "company_names": company_names,
            "classification": result_item.get("classification"),
            "category": result_item.get("category"),
            "summary": result_item.get("summary"),
            "llm_error": None,
        }


async def process_chunk(
    cfg, 
    chunk_df: pd.DataFrame, 
    chunk_idx: int, 
    concurrent_limit: int, 
    output_dir: str,
    save_parquet: bool = True
) -> pd.DataFrame:
    semaphore = asyncio.Semaphore(concurrent_limit)
    tasks = []
    for row in chunk_df.to_dict(orient="records"):
        tasks.append(asyncio.create_task(analyze_single_news(cfg, row, semaphore)))

    results = await asyncio.gather(*tasks)
    result_df = pd.DataFrame(results)
    final_df = chunk_df.merge(result_df, on="seq", how="left")

    # 날짜 포맷 표준화 (YYYY.MM.DD)
    date_parsed = pd.to_datetime(final_df["basc_dt"], errors="coerce", format="%Y%m%d")
    mask_na = date_parsed.isna()
    if mask_na.any():
        date_parsed_2 = pd.to_datetime(final_df.loc[mask_na, "basc_dt"], errors="coerce", format="%Y.%m.%d")
        date_parsed.loc[mask_na] = date_parsed_2
    final_df["basc_dt"] = date_parsed.dt.strftime("%Y.%m.%d").fillna(final_df["basc_dt"].astype(str))

    if "site_name" in final_df.columns:
        final_df["reference"] = "[" + final_df["site_name"].astype(str) + "]. " + final_df["title"].astype(str) + ". " + final_df["basc_dt"].astype(str)
    else:
        final_df["reference"] = final_df["title"].astype(str) + ". " + final_df["basc_dt"].astype(str)

    if save_parquet:
        out_path = os.path.join(output_dir, f"part_{chunk_idx:05d}.parquet")
        final_df.to_parquet(out_path, index=False)
        logger.info("[chunk %d] 저장 완료: %s (%d rows)", chunk_idx, out_path, len(final_df))

    return final_df

