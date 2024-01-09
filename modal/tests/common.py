import random
import string

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

from shared.protocol import CompletionPayload, Params


class TestConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.dev",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_url: str
    runner_api_key: str | None = None


async def completion(
    prompt: str,
    model: str = "microsoft/phi-2",
    max_tokens: int = 1024,
    stop: list = None,
    stream: bool = False,
    api_key: str = None,
) -> str:
    test_config = TestConfig()
    api_key = api_key or test_config.runner_api_key
    stop = stop or ["</s>"]
    if not api_key:
        raise ValueError("API key is required")

    payload = CompletionPayload(
        id="".join(
            random.choices(string.ascii_lowercase + string.digits, k=10)
        ),
        prompt=prompt,
        model=model,
        stream=stream,
        params=Params(
            max_tokens=max_tokens,
            stop=stop,
        ),
    )

    async with httpx.AsyncClient() as client:
        response = await client.post(
            test_config.api_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json=payload.model_dump(),
            timeout=120,
        )

    print(response.text)
    response.raise_for_status()

    return response.text if stream else response.json()
