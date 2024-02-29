import time
from typing import Optional

from modal import method
from pydantic import BaseModel

from shared.logging import get_logger, timer
from shared.protocol import (
    CompletionPayload,
    create_error_text,
    create_response_text,
    create_sse_data,
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
    def __init__(self, params: VllmParams):
        with timer("imports", model=params.model):
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine

        self.engine_args = AsyncEngineArgs(
            **params.dict(),
            disable_log_requests=True,
        )

        with timer("engine init", model=self.engine_args.model):
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)

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

        t_start_inference = time.perf_counter()

        try:
            results_generator = self.engine.generate(
                payload.prompt, params, payload.id
            )

            final_output = None
            finish_reason = None
            if payload.stream:
                index = 0

                async for request_output in results_generator:
                    # Skipping invalid UTF8 tokens:
                    if (
                        request_output.outputs[0].text
                        and request_output.outputs[0].text[-1] == "\ufffd"
                    ):
                        continue

                    final_output = request_output
                    token = final_output.outputs[0].text[index:]
                    index = len(final_output.outputs[0].text)
                    finish_reason = final_output.outputs[0].finish_reason
                    yield create_sse_data(
                        token,
                        prompt_tokens=len(final_output.prompt_token_ids),
                        completion_tokens=len(
                            final_output.outputs[0].token_ids
                        ),
                        done=False,
                        finish_reason=finish_reason,
                    )

                output = ""
            else:
                async for request_output in results_generator:
                    final_output = request_output
                    yield " "

                output = final_output.outputs[0].text

            prompt_tokens = len(final_output.prompt_token_ids)
            completion_tokens = len(final_output.outputs[0].token_ids)
            if payload.stream:
                yield create_sse_data(
                    "",
                    prompt_tokens,
                    completion_tokens,
                    done=True,
                    finish_reason=finish_reason,
                )
            else:
                yield create_response_text(
                    output,
                    prompt_tokens,
                    completion_tokens,
                    done=True,
                    finish_reason=finish_reason,
                )

            elapsed = time.perf_counter() - t_start_inference
            logger.info(
                "Completed generation",
                extra={
                    "model": self.engine_args.model,
                    "tokens": completion_tokens,
                    "tps": completion_tokens / elapsed,
                    "duration": elapsed,
                },
            )
        except Exception as err:
            e = create_error_text(err)
            logger.exception(
                "Failed generation", extra={"model": self.engine_args.model}
            )
            if payload.stream:
                yield create_sse_data(e)
            else:
                yield e
