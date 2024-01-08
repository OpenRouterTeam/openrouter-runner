import httpx
import pytest

from tests.common import completion, get_words_from_file


@pytest.mark.integration
@pytest.mark.asyncio
async def test_auth():
    prompt = "USER: I don't expect this to succeed."

    with pytest.raises(httpx.HTTPStatusError, match="401 Unauthorized"):
        _ = await completion(prompt, api_key="BADKEY")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_completion():
    prompt = "USER: What was Project A119 and what were its objectives?\n\n ASSISTANT:"

    output = await completion(prompt)
    assert output is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_completion_streaming():
    prompt = "USER: What was Project A119 and what were its objectives?\n\n ASSISTANT:"

    output = await completion(prompt, stream=True)
    assert output is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_large_prompt():
    """Should accept & complete a large prompt"""
    prompt = get_words_from_file(n=2500)
    output = await completion(prompt, max_tokens=5000)
    assert output is not None
