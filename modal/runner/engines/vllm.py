import time
from typing import Optional

from modal import method
from pydantic import BaseModel

from shared.logging import get_logger, timer
from shared.protocol import (
    CompletionPayload,
    GPUType,
    ResponseBody,
    Usage,
    create_error_text,
    sse,
)

from .base import BaseEngine

logger = get_logger(__name__)


# Adapted from: https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py#L192
class VllmParams(BaseModel):
    model: str
    tokenizer: Optional[str] = None
    tokenizer_mode: str = "auto"
    trust_remote_code: bool = True
    download_dir: Optional[str] = None
    load_format: str = "auto"
    dtype: str = "auto"
    seed: int = 0
    max_model_len: Optional[int] = None
    worker_use_ray: bool = False
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    block_size: int = 16
    swap_space: int = 4  # GiB
    gpu_memory_utilization: float = 0.90
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = 256
    # max_paddings: int = 256
    disable_log_stats: bool = False
    revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    quantization: Optional[str] = None


class VllmEngine(BaseEngine):
    def __init__(
        self,
        gpu_type: GPUType,
        params: VllmParams,
    ):
        self.gpu_type = gpu_type
        self.is_first_request = True
        self.t_cold_start = time.time()

        with timer("imports", model=params.model):
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine

        self.engine_args = AsyncEngineArgs(
            **params.dict(),
            disable_log_requests=True,
        )

        with timer("engine init", model=self.engine_args.model):
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)

    @property
    def gpu_count(self) -> int:
        return self.engine_args.tensor_parallel_size

    # @method()
    # async def tokenize_prompt(self, payload: Payload) -> List[int]:
    #     return self.tokenizer(payload.prompt).input_ids

    # @method()
    # async def max_model_len(self) -> int:
    #     engine_model_config = await self.engine.get_model_config()
    #     return engine_model_config.max_model_len

    @method()
    async def generate(self, payload: CompletionPayload, params):
        assert self.engine is not None, "Engine not initialized"

        # Track usage as a running total. For the first request to the
        # container, cold-start time is included in the usage duration.
        t_start_inference = time.time()
        t_start_usage_duration = t_start_inference
        if self.is_first_request:
            self.is_first_request = False
            t_start_usage_duration = self.t_cold_start

        resp = ResponseBody(
            text="",
            usage=Usage(
                prompt_tokens=0,
                completion_tokens=0,
                duration=0.0,
                gpu_type=self.gpu_type,
                gpu_count=self.gpu_count,
            ),
        )

        try:
            results_generator = self.engine.generate(
                payload.prompt, params, payload.id
            )

            output = ""
            index = 0
            finish_reason = None
            async for current in results_generator:
                output = current.outputs[0].text
                finish_reason = current.outputs[0].finish_reason
                resp.usage.prompt_tokens = len(current.prompt_token_ids)
                resp.usage.completion_tokens = len(current.outputs[0].token_ids)
                resp.usage.duration = time.time() - t_start_usage_duration

                # Non-streaming requests continue generating w/o yielding intermediate results
                if not payload.stream:
                    yield " "  # HACK: Keep the connection alive while generating
                    continue

                # Skipping invalid UTF8 tokens:
                if output and output[-1] == "\ufffd":
                    continue

                # Streaming requests send SSE messages with each new generated part
                token = output[index:]
                index = len(output)
                resp.text = token
                resp.finish_reason = finish_reason
                yield sse(resp.json(ensure_ascii=False))

            resp.text = "" if payload.stream else output
            resp.done = True
            resp.finish_reason = finish_reason
            data = resp.json(ensure_ascii=False)
            yield sse(data) if payload.stream else data

            logger.info(
                "Completed generation",
                extra={
                    "model": self.engine_args.model,
                    "tokens": resp.usage.completion_tokens,
                    "tps": resp.usage.completion_tokens / t_start_inference,
                    "duration": resp.usage.duration,
                    "cost": resp.usage.duration * self.cost_per_second,
                },
            )
        except Exception as err:
            e = create_error_text(err)
            logger.exception(
                "Failed generation", extra={"model": self.engine_args.model}
            )
            yield sse(e) if payload.stream else e
