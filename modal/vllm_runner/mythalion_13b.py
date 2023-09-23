# PREREQUISITES:
# 1. Create a modal secret group
#   HUGGINGFACE_TOKEN = <your huggingface token>
#   with name "huggingface"
# 2. Create a modal secret group
#   MYTHALION_API_KEY = <generate a random key>
#   with name "ext-api-key"
# 3. modal deploy


import os

from typing import List, Optional, Union

from fastapi import Depends, HTTPException, status
from modal import Image, Secret, Stub, method, gpu, web_endpoint
from modal.config import Config
from pydantic import BaseModel
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse

NAME = "mythalion"
MODEL_DIR = "/model"

NUM_GPU = 2
MODEL = "PygmalionAI/mythalion-13b"

config = Config()

KEEP_WARM = None
IDLE_TIMEOUT = 60 * 5  # 5 minutes

API_KEY_ID = "MYTHALION_API_KEY"
# MODEL = "Undi95/ReMM-SLERP-L2-13B"
# MODEL = "Gryphe/MythoMax-L2-13b"


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


auth_scheme = HTTPBearer()


def download_model_to_folder():
    from huggingface_hub import snapshot_download
    from pathlib import Path

    # make MODEL_DIR if not existed
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    snapshot_download(
        MODEL,
        local_dir=MODEL_DIR,
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
        download_model_to_folder,
        secret=Secret.from_name("huggingface"),
        timeout=60 * 20,
    )
)

stub = Stub(NAME, image=image)


@stub.cls(
    gpu=gpu.L4(count=NUM_GPU),
    secret=Secret.from_name("huggingface"),
    allow_concurrent_inputs=12,
    container_idle_timeout=IDLE_TIMEOUT,
    keep_warm=KEEP_WARM,
)
class Model:
    async def __aenter__(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.transformers_utils.tokenizer import get_tokenizer

        engine_args = AsyncEngineArgs(
            model=MODEL_DIR,
            tensor_parallel_size=NUM_GPU,
            # using 95% of GPU memory by default
            gpu_memory_utilization=0.95,
            disable_log_requests=True,
            max_num_batched_tokens=4096,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        self.engine_model_config = await self.engine.get_model_config()
        self.max_model_len = self.engine_model_config.get_max_model_len()

        # A separate tokenizer to map token IDs to strings.
        self.tokenizer = get_tokenizer(
            engine_args.tokenizer,
            tokenizer_mode=engine_args.tokenizer_mode,
            trust_remote_code=engine_args.trust_remote_code,
        )

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


@stub.function(
    secret=Secret.from_name("ext-api-key"),
    timeout=60 * 60,
    allow_concurrent_inputs=12,
    keep_warm=KEEP_WARM,
)
@web_endpoint(method="POST")
def completion(
    payload: Payload,
    token: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
    if token.credentials != os.environ[API_KEY_ID]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    from vllm.sampling_params import SamplingParams

    model = Model()

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


@stub.local_entrypoint()
def main():
    model = Model()
    questions = [
        "Implement a Python function to compute the Fibonacci numbers.",
        "Write a Rust function that performs binary exponentiation.",
    ]

    for question in questions:
        tokens = []
        for token_text in model.generate.remote_gen(question):
            tokens.append(token_text)
