from runner.engines.vllm import VllmEngine, VllmParams

# Each container comprises of:
# 1. An image
# 2. A stub class wrapping an engine

from modal import gpu, Image

from runner.shared.common import stub, models_path

GPU_COUNT = 1

_vllm_image = Image.from_registry(
    # "nvcr.io/nvidia/pytorch:23.09-py3"
    "nvcr.io/nvidia/pytorch:22.12-py3"
).pip_install(
    "vllm == 0.2.0",
    "typing-extensions==4.5.0",  # >=4.6 causes typing issues
)


@stub.cls(
    volumes={str(models_path): stub.models_volume},
    image=_vllm_image,
    gpu=gpu.A100(count=GPU_COUNT, memory=80),
    allow_concurrent_inputs=16,
    container_idle_timeout=5 * 60,  # 5 minutes
)
class VllmAWQ(VllmEngine):
    def __init__(
        self,
        model_path: str,
    ):
        super().__init__(
            VllmParams(
                model=model_path,
                tensor_parallel_size=GPU_COUNT,
                quantization="awq",
            )
        )
