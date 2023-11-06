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
    worker_use_ray: bool = True
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    block_size: int = 16
    swap_space: int = 4  # GiB
    gpu_memory_utilization: float = 0.95
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = 256
    disable_log_stats: bool = False
    revision: Optional[str] = None
    quantization: Optional[str] = None


class VllmEngine(BaseEngine):
    def __init__(self, params: VllmParams):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.transformers_utils.tokenizer import get_tokenizer

        import ray, torch

        # ref: https://github.com/vllm-project/vllm/issues/1116
        ray.shutdown()
        ray.init(num_gpus=torch.cuda.device_count())

        engine_args = AsyncEngineArgs(
            **params.dict(),
            disable_log_requests=True,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        # A separate tokenizer to map token IDs to strings.
        self.tokenizer = get_tokenizer(
            engine_args.tokenizer,
            tokenizer_mode=engine_args.tokenizer_mode,
            trust_remote_code=engine_args.trust_remote_code,
        )

    async def __aenter__(self):
        self.engine_model_config = await self.engine.get_model_config()
        self.max_model_len = self.engine_model_config.max_model_len

    @method()
    async def tokenize_prompt(self, payload: Payload) -> List[int]:
        return self.tokenizer(payload.prompt).input_ids

    @method()
    async def max_model_len(self) -> int:
        return self.max_model_len

    @method()
    async def generate(self, payload: Payload, params, input_ids):
        try:
            import time

            results_generator = self.engine.generate(
                payload.prompt, params, payload.id, input_ids
            )

            t0 = time.time()
            index, tokens = 0, 0
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
                tokens = len(request_output.outputs[0].token_ids)

            if not payload.stream:
                yield create_response_text(output)

            throughput = tokens / (time.time() - t0)
            print(f"Tokens count: {tokens} tokens")
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
