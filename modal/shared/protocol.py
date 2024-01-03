from typing import List, Optional, Union

from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel


# https://github.com/vllm-project/vllm/blob/320a622ec4d098f2da5d097930f4031517e7327b/vllm/sampling_params.py#L7-L52
# Lines were sorted for consistency
class Params(BaseModel):
    # early_stopping: Union[bool, str] = False
    # length_penalty: float = 1.0
    best_of: Optional[int] = None
    frequency_penalty: float = 0.0
    ignore_eos: bool = False
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = 42
    n: int = 1
    presence_penalty: float = 0.0
    stop: Union[None, str, List[str]] = None
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    use_beam_search: bool = False
    skip_special_tokens: bool = True


class CompletionPayload(BaseModel):
    id: str
    prompt: str
    stream: bool = False
    params: Params
    model: str


class ResponseBody(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    done: bool


def create_response_text(
    text: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    done: bool = False,
) -> str:
    return ResponseBody(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        done=done,
    ).json(ensure_ascii=False)


def create_sse_data(
    text: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    done: bool = False,
) -> str:
    return f"data: {create_response_text(text, prompt_tokens, completion_tokens, done)}\n\n"


class ErrorPayload(BaseModel):
    type: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorPayload


def create_error_text(err: Exception) -> str:
    return ErrorResponse(
        error=ErrorPayload(message=f"{err}", type=f"{type(err).__name__}")
    ).json(ensure_ascii=False)


def create_error_response(status_code: int, message: str) -> JSONResponse:
    return PlainTextResponse(
        content=message,
        status_code=status_code,
    )
