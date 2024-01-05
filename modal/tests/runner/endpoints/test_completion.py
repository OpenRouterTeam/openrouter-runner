import httpx
import pytest

from tests.common import completion


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
