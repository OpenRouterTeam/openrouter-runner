from runner.engines.lorean import LoreanEngine, LoreanParams

# Each container comprises of:
# 1. An image
# 2. A stub class wrapping an engine

from modal import gpu, Image
from shared.volumes import models_path, loras_path
from runner.shared.common import stub

# TODO: Swap to lower-end GPU on prod
_gpu = gpu.A100(count=1, memory=80)
_image = Image.from_registry(
    "nvcr.io/nvidia/pytorch:23.09-py3"
    # "nvcr.io/nvidia/pytorch:22.12-py3"
).pip_install(
    "bitsandbytes",
    "transformers",
    "peft",
    "accelerate",
    "datasets",
    "scipy",
    "ipywidgets",
    "wandb",
)


@stub.cls(
    volumes={
        str(loras_path): stub.loras_volume,
        str(models_path): stub.models_volume,
    },
    image=_image,
    gpu=_gpu,
    allow_concurrent_inputs=8,
    container_idle_timeout=10 * 60,  # 5 minutes
)
class Lorean7BContainer(LoreanEngine):
    def __init__(
        self,
        model_path: str,
    ):
        super().__init__(
            LoreanParams(
                model=model_path,
            )
        )
