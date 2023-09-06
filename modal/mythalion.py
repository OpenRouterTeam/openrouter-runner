import os

from typing import List, Optional, Union

from fastapi import Depends, HTTPException, status, Request

from modal import Image, Secret, Stub, method, gpu, web_endpoint
from pydantic import BaseModel
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


MODEL_DIR = "/model"

MODEL = "PygmalionAI/mythalion-13b"


# https://github.com/vllm-project/vllm/blob/320a622ec4d098f2da5d097930f4031517e7327b/vllm/sampling_params.py#L7-L52


class Params(BaseModel):
    n: int = (1,)
    best_of: Optional[int] = (None,)
    presence_penalty: float = (0.0,)
    frequency_penalty: float = (0.0,)
    temperature: float = (1.0,)
    top_p: float = (1.0,)
    top_k: int = (-1,)
    use_beam_search: bool = (False,)
    length_penalty: float = (1.0,)
    early_stopping: Union[bool, str] = (False,)
    stop: Union[None, str, List[str]] = (None,)
    ignore_eos: bool = (False,)
    max_tokens: int = (16,)
    logprobs: Optional[int] = (None,)


class Payload(BaseModel):
    id: str
    prompt: str
    params: Params


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
    .pip_install("torch==2.0.1", index_url="https://download.pytorch.org/whl/cu118")
    # Pinned to 08/15/2023
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
)
class Model:
    def __enter__(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        engine_args = AsyncEngineArgs(
            model=MODEL_DIR,
            tensor_parallel_size=1,
            # Only uses 90% of GPU memory by default
            gpu_memory_utilization=0.95,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.template = TEMPLATE

    @method()
    async def generate(self, payload: Payload):
        from vllm.utils import random_uuid

        import time

        results_generator = self.engine.generate(payload.id, payload.params, payload.id)
        # print(prompt)

        t0 = time.time()
        index, tokens = 0, 0
        async for request_output in results_generator:
            if "\ufffd" == request_output.outputs[0].text[-1]:
                continue
            yield request_output.outputs[0].text[index:]
            index = len(request_output.outputs[0].text)

            # Token accounting
            tokens = len(request_output.outputs[0].token_ids)

        throughput = tokens / (time.time() - t0)
        print(f"Tokens count: {tokens} tokens")
        print(f"Request completed: {throughput:.4f} tokens/s")

        yield "[DONE]"
        # print(request_output.outputs[0].text)


@stub.function(
    secret=Secret.from_name("mythalion"),
    timeout=60 * 10,
    allow_concurrent_inputs=12,
    keep_warm=1,
)
@web_endpoint(method="POST")
def completion(
    payload: Payload,
    request: Request,
    token: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
    if token.credentials != os.environ["AUTH_TOKEN"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    from itertools import chain

    from fastapi.responses import StreamingResponse

    return StreamingResponse(
        chain(
            Model().generate.remote_gen(payload),
        ),
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
