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
from pydantic import BaseModel
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


MODEL_DIR = "/model"

MODEL = "PygmalionAI/mythalion-13b"
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
    max_tokens: int = 16
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


auth_scheme = HTTPBearer()


def download_model_to_folder():
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL,
        local_dir=MODEL_DIR,
        token=os.environ["HUGGINGFACE_TOKEN"],
    )


image = (
    Image.from_registry("nvcr.io/nvidia/pytorch:22.12-py3")
    .pip_install(
        "torch==2.0.1", index_url="https://download.pytorch.org/whl/cu118"
    )
    # Pinned to Sep/07/2023
    .pip_install(
        "vllm @ git+https://github.com/vllm-project/vllm.git@805de738f618f8b47ab0d450423d23db1e636fa2",
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

stub = Stub("mythalion", image=image)


@stub.cls(
    gpu=gpu.A100(),
    secret=Secret.from_name("huggingface"),
    allow_concurrent_inputs=12,
    container_idle_timeout=600,
    keep_warm=1,
)
class Model:
    def __enter__(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        engine_args = AsyncEngineArgs(
            model=MODEL_DIR,
            tensor_parallel_size=1,
            # using 95% of GPU memory by default
            gpu_memory_utilization=0.95,
            disable_log_requests=False,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @method()
    async def generate(self, payload: Payload, params):
        import time

        results_generator = self.engine.generate(
            payload.prompt, params, payload.id
        )

        t0 = time.time()
        index, tokens = 0, 0
        async for request_output in results_generator:
            if (
                request_output.outputs[0].text
                and "\ufffd" == request_output.outputs[0].text[-1]
            ):
                continue
            token = request_output.outputs[0].text[index:]
            if payload.stream:
                choice = CompletionResponse(text=token).json(ensure_ascii=False)
                yield f"data: {choice}\n\n"
            else:
                yield token
            index = len(request_output.outputs[0].text)
            # Token accounting
            tokens = len(request_output.outputs[0].token_ids)

        throughput = tokens / (time.time() - t0)
        print(f"Tokens count: {tokens} tokens")
        print(f"Request completed: {throughput:.4f} tokens/s")

        # yield "[DONE]"
        # print(request_output.outputs[0].text)


@stub.function(
    secret=Secret.from_name("ext-api-key"),
    timeout=60 * 60,
    allow_concurrent_inputs=12,
    keep_warm=1,
)
@web_endpoint(method="POST")
def completion(
    payload: Payload,
    token: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
    if token.credentials != os.environ["MYTHALION_API_KEY"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    from fastapi.responses import StreamingResponse
    from vllm.sampling_params import SamplingParams

    sampling_params = SamplingParams(
        # early_stopping=payload.params.early_stopping,
        # length_penalty=payload.params.length_penalty,
        best_of=payload.params.best_of,
        frequency_penalty=payload.params.frequency_penalty,
        ignore_eos=payload.params.ignore_eos,
        logprobs=payload.params.logprobs,
        max_tokens=payload.params.max_tokens,
        n=payload.params.n,
        presence_penalty=payload.params.presence_penalty,
        stop=payload.params.stop,
        temperature=payload.params.temperature,
        top_k=payload.params.top_k,
        top_p=payload.params.top_p,
        use_beam_search=payload.params.use_beam_search,
    )

    print(sampling_params)

    return StreamingResponse(
        Model().generate.remote_gen(payload, sampling_params),
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
