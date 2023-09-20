# PREREQUISITES:
# 1. Create a modal secret group
#   HUGGINGFACE_TOKEN = <your huggingface token>
#   with name "huggingface"
# 2. Create a modal secret group
#   MYTHALION_API_KEY = <generate a random key>
#   with name "ext-api-key"
# 3. modal deploy


import os

from fastapi import Depends, HTTPException, status
from modal import Image, Secret, Stub, method, gpu, web_endpoint
from pydantic import validator

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse

from vllm_runner.shared.protocol import (
    create_error_response,
    Payload as BasePayload,
)

from auxilary.shared.common import config, stub

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


def download_models():
    from huggingface_hub import snapshot_download
    from pathlib import Path

    model_dir_path = Path(config.download_dir)
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
        timeout=60 * 60,
    )
)


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
