from runner.engines.vllm import VllmEngine, VllmParams

# Each container comprises of:
# 1. An image
# 2. A stub class wrapping an engine

from modal import gpu, Image

from runner.shared.common import stub, models_path

_gpu = gpu.A100(count=1, memory=80)

_vllm_image = (
    Image.from_registry(
        # "nvcr.io/nvidia/pytorch:23.09-py3"
        "nvcr.io/nvidia/pytorch:22.12-py3"
    )
    # Use latest torch
    .pip_install(
        "torch==2.0.1+cu118", index_url="https://download.pytorch.org/whl/cu118"
    )
    # Pinned to 10/16/23
    .pip_install(
        "vllm @ git+https://github.com/vllm-project/vllm.git@651c614aa43e497a2e2aab473493ba295201ab20",
        "typing-extensions==4.5.0",  # >=4.6 causes typing issues
    )
)


@stub.cls(
    volumes={str(models_path): stub.models_volume},
    image=_vllm_image,
    gpu=_gpu,
    allow_concurrent_inputs=16,
    container_idle_timeout=10 * 60,  # 5 minutes
)
class VllmAWQ(VllmEngine):
    def __init__(
        self,
        model_path: str,
    ):
        super().__init__(
            VllmParams(
                model=model_path,
                tensor_parallel_size=_gpu.count,
                quantization="awq",
            )
        )
