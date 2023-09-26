from runner.engines.vllm import VllmEngine, VllmParams

# Each container comprises of:
# 1. An image
# 2. A stub class wrapping an engine

from modal import gpu, Image

from runner.shared.common import stub, models_path


_vllm_image = Image.from_registry(
    "nvcr.io/nvidia/pytorch:22.12-py3"
).pip_install(
    "vllm == 0.1.7",
    # Pinned to Sep/11/2023
    # "vllm @ git+https://github.com/vllm-project/vllm.git@b9cecc26359794af863b3484a3464108b7d5ee5f",
    # Pinned to 08/15/2023
    # "vllm @ git+https://github.com/vllm-project/vllm.git@805de738f618f8b47ab0d450423d23db1e636fa2",
    "typing-extensions==4.5.0",  # >=4.6 causes typing issues
)

GPU_COUNT = 2


@stub.cls(
    volumes={str(models_path): stub.models_volume},
    image=_vllm_image,
    gpu=gpu.L4(count=GPU_COUNT),
    allow_concurrent_inputs=16,
    container_idle_timeout=5 * 60,  # 5 minutes
)
class Vllm13BContainer(VllmEngine):
    def __init__(
        self,
        model_path: str,
        max_context_size: int,
    ):
        super().__init__(
            VllmParams(
                model=model_path,
                max_num_batched_tokens=max_context_size,
                tensor_parallel_size=GPU_COUNT,
            )
        )
