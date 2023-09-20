from vllm_runner.shared.protocol import (
    Payload,
    CompletionResponse,
    ErrorPayload,
    ErrorResponse,
)
from vllm_runner.shared.sampling_params import SamplingParams

from typing import List
from modal import Secret, method, gpu, Image

from ..shared.common import stub, config


model_id = "Undi95/ReMM-SLERP-L2-13B"


def download_model():
    import os
    from huggingface_hub import snapshot_download
    from pathlib import Path

    # make MODEL_DIR if not existed
    Path(config.download_dir).mkdir(parents=True, exist_ok=True)

    snapshot_download(
        model_id,
        local_dir=config.download_dir,
        token=os.environ["HUGGINGFACE_TOKEN"],
    )


image = (
    Image.from_registry("nvcr.io/nvidia/pytorch:22.12-py3")
    .pip_install(
        "vllm == 0.1.7",
        # Pinned to Sep/11/2023
        # "vllm @ git+https://github.com/vllm-project/vllm.git@b9cecc26359794af863b3484a3464108b7d5ee5f",
        # Pinned to 08/15/2023
        # "vllm @ git+https://github.com/vllm-project/vllm.git@805de738f618f8b47ab0d450423d23db1e636fa2",
        "typing-extensions==4.5.0",  # >=4.6 causes typing issues
    )
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .pip_install("hf-transfer~=0.1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model,
        secret=Secret.from_name("huggingface"),
        timeout=60 * 60,
    )
)


@stub.cls(
    image=image,
    gpu=gpu.A100(count=config.num_gpu),
    secret=Secret.from_name("huggingface"),
    allow_concurrent_inputs=config.concurrency,
    container_idle_timeout=config.idle_timeout,
)
class RemmSlerp13BModel:
    async def __aenter__(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.transformers_utils.tokenizer import get_tokenizer

        engine_args = AsyncEngineArgs(
            model=config.download_dir,
            tensor_parallel_size=config.num_gpu,
            # using 95% of GPU memory by default
            gpu_memory_utilization=0.95,
            disable_log_requests=True,
            max_num_batched_tokens=config.max_batched_tokens,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        # A separate tokenizer to map token IDs to strings.
        self.tokenizer = get_tokenizer(
            engine_args.tokenizer,
            tokenizer_mode=engine_args.tokenizer_mode,
            trust_remote_code=engine_args.trust_remote_code,
        )

        self.engine_model_config = await self.engine.get_model_config()
        self.max_model_len = self.engine_model_config.get_max_model_len()

    @method()
    async def tokenize_prompt(self, payload: Payload) -> List[int]:
        return self.tokenizer(payload.prompt).input_ids

    @method()
    async def max_model_len(self) -> int:
        return self.max_model_len

    @method()
    async def generate(
        self, payload: Payload, params: SamplingParams, input_ids
    ):
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
