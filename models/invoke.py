from omegaconf import DictConfig
from openai import AsyncOpenAI, base_url

async def invoke_api(cfg: DictConfig, prompt: str, system_prompt: str) -> str:
    """
    gpt-oss-120b 모델 비동기로 호출하는 함수
    """
    base_url = cfg.serve.online.url + cfg.serve.online.model

    async with AsyncOpenAI(base_url=base_url, api_key="not-needed") as client:
        try:
            response = await client.responses.create(
                model="/models",
                instructions=system_prompt,
                input=prompt,

                temperature=cfg.serve.online.temperature,
                top_p=cfg.serve.online.top_p,

                reasoning={
                    "effort": cfg.serve.online.reasoning
                },
                max_output_tokens=cfg.serve.online.max_output_tokens,
            )

            return response.output_text

        except Exception as e:
            return f"Failure: API calling: {e}"

    

