# Each container comprises of:
# 1. An image
# 2. A stub class wrapping an engine
from modal import Image, gpu

from runner.engines.lorean import LoreanEngine, LoreanParams
from runner.shared.common import stub
from shared.volumes import loras_path, loras_volume, models_path, models_volume

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
        loras_path: loras_volume,
        models_path: models_volume,
    },
    image=_image,
    gpu=_gpu,
    allow_concurrent_inputs=8,
    container_idle_timeout=20 * 60,
    timeout=10 * 60,
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
