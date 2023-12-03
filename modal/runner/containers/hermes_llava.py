from typing import List, Optional
from modal import method

from shared.protocol import (
    Payload,
    create_sse_data,
    create_response_text,
    create_error_text,
)
from pydantic import BaseModel

from runner.engines.vllm import VllmEngine, VllmParams

# Each container comprises of:
# 1. An image
# 2. A stub class wrapping an engine

from modal import gpu, Image

from shared.volumes import models_path
from runner.shared.common import stub

_gpu = gpu.A100(count=1)

_vllm_image = (
    Image.from_registry(
        "nvcr.io/nvidia/pytorch:23.09-py3"
        # "nvcr.io/nvidia/pytorch:22.12-py3"
    )
    # Use latest torch
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
    )
    # Grab llava from hermes-llava
    .pip_install("llava @ git+https://github.com/qnguyen3/hermes-llava@main")
)


@stub.cls(
    volumes={str(models_path): stub.models_volume},
    image=_vllm_image,
    gpu=_gpu,
    allow_concurrent_inputs=8,
    container_idle_timeout=20 * 60,
    timeout=10 * 60,
)
class HermesLlavaContainer:
    def __init__(
        self,
        model_path: str,
    ):
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
        )

    @method()
    async def generate(self, payload: Payload, params):
        pass
