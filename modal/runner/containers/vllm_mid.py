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
    .pip_install(
        "transformers @ git+https://github.com/huggingface/transformers@main"
    )
    .pip_install("vllm @ git+https://github.com/vllm-project/vllm.git@main")
)


@stub.cls(
    volumes={str(models_path): stub.models_volume},
    image=_vllm_image,
    gpu=_gpu,
    allow_concurrent_inputs=8,
    container_idle_timeout=20 * 60,
    timeout=10 * 60,
)
class VllmMidContainer(VllmEngine):
    def __init__(
        self,
        model_path: str,
    ):
        super().__init__(
            VllmParams(
                model=model_path,
                tensor_parallel_size=_gpu.count,
            )
        )
