from enum import Enum
from typing import Final, List, Optional, Union

from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

_COST_PER_SECOND_A100_40G: Final[float] = 0.001036
_COST_PER_SECOND_A100_80G: Final[float] = 0.001553


class GPUType(Enum):
    A100_40G = "A100_40G"
    A100_80G = "A100_80G"


class ContainerType(Enum):
    VllmContainer_3B = "VllmContainer_3B"
    VllmContainer_7B = "VllmContainer_7B"

    VllmContainerA100_40G = "VllmContainerA100_40G"

    VllmContainerA100_80G = "VllmContainerA100_80G"
    VllmContainerA100_80G_32K = "VllmContainerA100_80G_32K"

    VllmContainerA100_160G = "VllmContainerA100_160G"
    VllmContainerA100_160G_Isolated = "VllmContainerA100_160G_Isolated"

    @property
    def gpu_cost_per_second(self) -> float:
        """
        Returns:
            The quoted GPU compute cost per second for the container,
            as found on https://modal.com/pricing
        """

        # TODO: might be better to put this on the container class itself,
        #       but this is good enough(tm) for now
        match self:
            case ContainerType.VllmContainer_3B:
                return _COST_PER_SECOND_A100_40G * 1
            case ContainerType.VllmContainer_7B:
                return _COST_PER_SECOND_A100_40G * 1
            case ContainerType.VllmContainerA100_40G:
                return _COST_PER_SECOND_A100_40G * 1
            case ContainerType.VllmContainerA100_80G:
                return _COST_PER_SECOND_A100_80G * 1
            case ContainerType.VllmContainerA100_80G_32K:
                return _COST_PER_SECOND_A100_80G * 1
            case ContainerType.VllmContainerA100_160G:
                return _COST_PER_SECOND_A100_80G * 2
            case ContainerType.VllmContainerA100_160G_Isolated:
                return _COST_PER_SECOND_A100_80G * 2


# https://github.com/vllm-project/vllm/blob/320a622ec4d098f2da5d097930f4031517e7327b/vllm/sampling_params.py#L7-L52
# Lines were sorted for consistency
class Params(BaseModel):
    # early_stopping: Union[bool, str] = False
    # length_penalty: float = 1.0
    best_of: Optional[int] = None
    ignore_eos: bool = False
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = 42
    n: int = 1

    stop: Union[None, str, List[str]] = None
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0

    min_p: int = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    use_beam_search: bool = False
    skip_special_tokens: bool = True


class RunnerConfiguration(BaseModel):
    container: ContainerType


class CompletionPayload(BaseModel):
    id: str
    prompt: str
    stream: bool = False
    params: Params
    model: str
    runner: RunnerConfiguration | None = None


class Usage(BaseModel):
    """
    An OpenAI-style usage struct, containing the expected
    token counts as well as additional data about the GPU
    usage of the request.
    """

    prompt_tokens: int
    completion_tokens: int

    duration: float
    gpu_type: GPUType
    gpu_count: int


class ResponseBody(BaseModel):
    text: str
    usage: Usage
    finish_reason: str | None
    done: bool


def create_response_text(
    text: str,
    usage: Usage,
    done: bool = False,
    finish_reason: str | None = None,
) -> str:
    r = ResponseBody(
        text=text, usage=usage, done=done, finish_reason=finish_reason
    )
    return r.json(ensure_ascii=False)


def create_sse_data(response: str) -> str:
    """Wrap a given response string as an SSE message."""
    return f"data: {response}\n\n"


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
