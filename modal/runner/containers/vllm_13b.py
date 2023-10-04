from runner.engines.vllm import VllmEngine, VllmParams

# Each container comprises of:
# 1. An image
# 2. A stub class wrapping an engine

from modal import gpu, Image

from runner.shared.common import stub, models_path

GPU_COUNT = 2

_vllm_image = Image.from_registry(
    # "nvcr.io/nvidia/pytorch:23.09-py3"
    "nvcr.io/nvidia/pytorch:22.12-py3"
).pip_install(
    "vllm == 0.2.0",
    # "vllm @ git+https://github.com/vllm-project/vllm.git@main",
    "typing-extensions==4.5.0",  # >=4.6 causes typing issues
)


@stub.cls(
    volumes={str(models_path): stub.models_volume},
    image=_vllm_image,
    gpu=gpu.A10G(count=GPU_COUNT),
    allow_concurrent_inputs=16,
    container_idle_timeout=10 * 60,  # 5 minutes
)
class Vllm13BContainer(VllmEngine):
    def __init__(
        self,
        model_path: str,
    ):
        super().__init__(
            VllmParams(
                model=model_path,
                tensor_parallel_size=GPU_COUNT,
            )
        )
