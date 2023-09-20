# PREREQUISITES:
# 1. Create a modal secret group
#   HUGGINGFACE_TOKEN = <your huggingface token>
#   with name "huggingface"
# 2. Create a modal secret group
#   MYTHALION_API_KEY = <generate a random key>
#   with name "ext-api-key"
# 3. modal deploy


import os

from typing import List, Union
from fastapi import Depends, HTTPException, status
from modal import Image, Secret, Stub, method, gpu, web_endpoint
from pydantic import validator, ValidationError

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse

from vllm_runner.shared.protocol import (
    create_error_response,
    Payload as BasePayload,
    CompletionResponse,
    ErrorPayload,
    ErrorResponse,
)

from vllm_runner.shared.config import Config

MODELS = [
    "PygmalionAI/mythalion-13b",
    "Gryphe/MythoMax-L2-13b",
    "Undi95/ReMM-SLERP-L2-13B",
    "meta-llama/Llama-2-13b-chat-hf",
    "NousResearch/Nous-Hermes-Llama2-13b",
]

NORMALIZED_MODELS = {model.lower() for model in MODELS}


class Payload(BasePayload):
    model: str

    # Allow insensitive casing for the HF identifier
    @validator("model", pre=True, always=True)
    def validate_model(cls, model: str) -> str:
        if model.lower() in NORMALIZED_MODELS:
            return model
        raise ValueError(f"Invalid model: {model}")


config = Config(
    name="auxilary_13b",
    api_key_id="AUXILARY_13B_API_KEY",
    model_dir="/models",
    num_gpu=1,
    max_batched_tokens=4096,
    idle_timeout=5 * 60,  # 5 minutes
    concurrency=24,
)


def download_models():
    from huggingface_hub import snapshot_download
    from pathlib import Path

    model_dir_path = Path(config.model_dir)
    # make MODEL_DIR if not existed
    model_dir_path.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        repo_id = model

        local_dir_path = model_dir_path / repo_id.lower()

        # Ensure the directory exists
        local_dir_path.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir_path),  # Convert Path object to string
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
        download_models,
        secret=Secret.from_name("huggingface"),
        timeout=60 * 20,
    )
)

stub = Stub(config.name, image=image)


@stub.cls(
    gpu=gpu.A100(count=config.num_gpu),
    secret=Secret.from_name("huggingface"),
    allow_concurrent_inputs=config.concurrency,
    container_idle_timeout=config.idle_timeout,
)
class Model:
    def __init__(self, model: str) -> None:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.transformers_utils.tokenizer import get_tokenizer
        from pathlib import Path

        model_dir_path = Path(config.model_dir) / model.lower()

        engine_args = AsyncEngineArgs(
            model=str(model_dir_path),
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


auth_scheme = HTTPBearer()


@stub.function(
    secret=Secret.from_name("ext-api-key"),
    timeout=60 * 60,
    allow_concurrent_inputs=config.concurrency,
)
@web_endpoint(method="POST")
def completion(
    payload: Payload,
    token: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
    if token.credentials != os.environ[config.api_key_id]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    from vllm.sampling_params import SamplingParams

    model = Model(payload.model)

    max_model_len = model.max_model_len.remote()
    input_ids = model.tokenize_prompt.remote(payload)
    token_num = len(input_ids)

    if payload.params.max_tokens is None:
        max_tokens = max_model_len - token_num
    else:
        max_tokens = payload.params.max_tokens

    is_too_high = (token_num + max_tokens) > max_model_len

    if is_too_high:
        return create_error_response(
            status.HTTP_400_BAD_REQUEST,
            f"This model's maximum context length is {max_model_len} tokens. "
            f"However, you requested {max_tokens + token_num} tokens "
            f"({token_num} in the messages, "
            f"{max_tokens} in the completion). "
            f"Please reduce the length of the messages or completion.",
        )

    try:
        sampling_params = SamplingParams(
            # early_stopping=payload.params.early_stopping,
            # length_penalty=payload.params.length_penalty,
            best_of=payload.params.best_of,
            frequency_penalty=payload.params.frequency_penalty,
            ignore_eos=payload.params.ignore_eos,
            logprobs=payload.params.logprobs,
            max_tokens=max_tokens,
            n=payload.params.n,
            presence_penalty=payload.params.presence_penalty,
            stop=payload.params.stop,
            temperature=payload.params.temperature,
            top_k=payload.params.top_k,
            top_p=payload.params.top_p,
            use_beam_search=payload.params.use_beam_search,
        )
        print(sampling_params)
    except ValueError as e:
        return create_error_response(status.HTTP_400_BAD_REQUEST, str(e))

    return StreamingResponse(
        model.generate.remote_gen(payload, sampling_params, input_ids),
        media_type="text/event-stream",
    )
