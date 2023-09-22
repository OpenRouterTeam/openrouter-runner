from typing import List
from modal import method

from aux.shared.protocol import (
    CompletionResponse,
    ErrorPayload,
    ErrorResponse,
    Payload,
)
from aux.engines.base import BaseEngine


# from aux.engines.vllm import VllmEngine
# NOTE: The goal is to have the code above, but due to a bug with modal resolver, we will have to place the engine here for now:
class VllmEngine(BaseEngine):
    def __init__(
        self,
        model_path: str,
        max_num_batched_tokens: int,
    ):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.transformers_utils.tokenizer import get_tokenizer

        engine_args = AsyncEngineArgs(
            model=model_path,
            tensor_parallel_size=1,
            # using 95% of GPU memory by default
            gpu_memory_utilization=0.95,
            disable_log_requests=True,
            max_num_batched_tokens=max_num_batched_tokens,
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
        self.max_model_len = self.engine_model_config.get_max_model_len()

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
                    choice = CompletionResponse(text=token).json(
                        ensure_ascii=False
                    )
                    yield f"data: {choice}\n\n"
                else:
                    output += token
                index = len(request_output.outputs[0].text)
                # Token accounting
                tokens = len(request_output.outputs[0].token_ids)

            if not payload.stream:
                yield CompletionResponse(text=output).json(ensure_ascii=False)

            throughput = tokens / (time.time() - t0)
            print(f"Tokens count: {tokens} tokens")
            print(f"Request completed: {throughput:.4f} tokens/s")

            # yield "[DONE]"
            # print(request_output.outputs[0].text)
        except Exception as err:
            error_response = ErrorResponse(
                error=ErrorPayload(
                    message=f"{err}", type=f"{type(err).__name__}"
                )
            ).json(ensure_ascii=False)

            if payload.stream:
                yield f"data: {error_response}\n\n"
            else:
                yield error_response


# Each container comprises of:
# 1. An image
# 2. A stub class wrapping an engine

from modal import gpu, Image

from aux.shared.common import stub, models_path


_vllm_image = Image.from_registry(
    "nvcr.io/nvidia/pytorch:22.12-py3"
).pip_install(
    "vllm == 0.1.7",
    # Pinned to Sep/11/2023
    # "vllm @ git+https://github.com/vllm-project/vllm.git@b9cecc26359794af863b3484a3464108b7d5ee5f",
    # Pinned to 08/15/2023
    # "vllm @ git+https://github.com/vllm-project/vllm.git@805de738f618f8b47ab0d450423d23db1e636fa2",
    "typing-extensions==4.5.0",  # >=4.6 causes typing issues
)


@stub.cls(
    volumes={str(models_path): stub.models_volume},
    image=_vllm_image,
    gpu=gpu.A100(count=1),
    allow_concurrent_inputs=16,
    container_idle_timeout=5 * 60,  # 5 minutes
)
class Vllm13BContainer(VllmEngine):
    pass
