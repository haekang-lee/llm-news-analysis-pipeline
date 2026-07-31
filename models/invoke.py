from omegaconf import DictConfig
from openai import AsyncOpenAI

async def invoke_api(cfg: DictConfig, prompt: str, system_prompt: str) -> str:
    
    base_url = cfg.serve.online.base_url
    
    async with AsyncOpenAI(base_url=base_url, api_key=cfg.serve.online.api_key) as client:
        try:
            response = await client.responses.create(
                model=cfg.serve.online.model,
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
            # 에러 발생시 해당 응답 반환
            return f"Failure: API calling: {e}"