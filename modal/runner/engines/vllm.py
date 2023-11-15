from typing import List, Optional
from modal import method

from shared.protocol import (
    Payload,
    create_sse_data,
    create_response_text,
    create_error_text,
)
from pydantic import BaseModel

from .base import BaseEngine


# Adapted from: https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py#L192
class VllmParams(BaseModel):
    model: str
    tokenizer: Optional[str] = None
    tokenizer_mode: str = "auto"
    trust_remote_code: bool = False
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
    gpu_memory_utilization: float = 0.95
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = 256
    # max_paddings: int = 256
    disable_log_stats: bool = False
    revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    quantization: Optional[str] = None


class VllmEngine(BaseEngine):
    def __init__(self, params: VllmParams):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        engine_args = AsyncEngineArgs(
            **params.dict(),
            disable_log_requests=True,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    # @method()
    # async def tokenize_prompt(self, payload: Payload) -> List[int]:
    #     return self.tokenizer(payload.prompt).input_ids

    # @method()
    # async def max_model_len(self) -> int:
    #     engine_model_config = await self.engine.get_model_config()
    #     return engine_model_config.max_model_len

    @method()
    async def generate(self, payload: Payload, params):
        try:
            import time

            results_generator = self.engine.generate(
                payload.prompt, params, payload.id
            )

            t0 = time.time()
            index, completion_tokens = 0, 0
            output = ""
            async for request_output in results_generator:
                # Skipping invalid UTF8 tokens:
                if (
                    request_output.outputs[0].text
                    and "\ufffd" == request_output.outputs[0].text[-1]
                ):
                    continue
                token = request_output.outputs[0].text[index:]
                if payload.stream:
                    yield create_sse_data(token)
                else:
                    output += token
                index = len(request_output.outputs[0].text)
                # Token accounting

            prompt_tokens = len(request_output.prompt_token_ids)
            completion_tokens = len(request_output.outputs[0].token_ids)
            if not payload.stream:
                yield create_response_text(
                    output,
                    prompt_tokens,
                    completion_tokens,
                )
            else:
                yield create_sse_data(
                    "[DONE]",
                    prompt_tokens,
                    completion_tokens,
                )

            throughput = completion_tokens / (time.time() - t0)
            print(f"Tokens count: {completion_tokens} tokens")
            print(f"Request completed: {throughput:.4f} tokens/s")

            # yield "[DONE]"
            # print(request_output.outputs[0].text)
        except Exception as err:
            e = create_error_text(err)
            print(e)
            if payload.stream:
                yield create_sse_data(e)
            else:
                yield e
