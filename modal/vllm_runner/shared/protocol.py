from pydantic import BaseModel
from fastapi.responses import JSONResponse, PlainTextResponse

from typing import List, Optional, Union


# https://github.com/vllm-project/vllm/blob/320a622ec4d098f2da5d097930f4031517e7327b/vllm/sampling_params.py#L7-L52
# Lines were sorted for consistency
class Params(BaseModel):
    # early_stopping: Union[bool, str] = False
    # length_penalty: float = 1.0
    best_of: Optional[int] = None
    frequency_penalty: float = 0.0
    ignore_eos: bool = False
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    n: int = 1
    presence_penalty: float = 0.0
    stop: Union[None, str, List[str]] = None
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    use_beam_search: bool = False


class Payload(BaseModel):
    id: str
    prompt: str
    stream: bool = False
    params: Params


class CompletionResponse(BaseModel):
    text: str


class ErrorPayload(BaseModel):
    type: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorPayload


def create_error_response(status_code: int, message: str) -> JSONResponse:
    return PlainTextResponse(
        content=message,
        status_code=status_code,
    )
