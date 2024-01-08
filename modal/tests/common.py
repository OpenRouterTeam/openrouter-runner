import random
import string
from importlib import resources

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

from shared.protocol import Params, Payload


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
    model: str = "Intel/neural-chat-7b-v3-1",
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

    payload = Payload(
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


def get_words_from_file(
    n: int = -1,
    filename: str = "down_and_out.txt",
    start_line: int = 0,
):
    """
    Get the first n words from the given filename (in the tests.data module),
    starting at start_line. If n == -1, everything from start_line to the end.
    """
    result = []
    with resources.open_text("tests.data", filename) as file:
        lines = file.readlines()[start_line:]
        if n <= -1:
            return "".join(lines)

        for line in lines:
            words = line.split()
            take_words = min(n - len(result), len(words))
            result.extend(words[:take_words])
            if len(result) >= n:
                break

    assert len(result) <= n
    return " ".join(result)
