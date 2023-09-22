# Each container comprises of:
# 1. An image
# 2. A stub class wrapping an engine

from modal import gpu, Image
from aux.engines.vllm import VllmEngine
from aux.shared.common import stub
from aux.shared.volume import models_path, models_volume

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


@stub.cls(
    volumes={str(models_path): models_volume},
    image=_vllm_image,
    gpu=gpu.A100(count=1),
    allow_concurrent_inputs=16,
    container_idle_timeout=5 * 60,  # 5 minutes
)
class Vllm13BContainer(VllmEngine):
    pass
