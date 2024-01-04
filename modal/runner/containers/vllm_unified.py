from typing import Optional

import modal.gpu
from modal import Image

from runner.engines.vllm import VllmEngine, VllmParams
from runner.shared.common import stub
from shared.volumes import models_path

_vllm_image = Image.from_registry(
    "nvidia/cuda:12.1.0-base-ubuntu22.04",
    add_python="3.10",
).pip_install(
    "vllm==0.2.6",
)


def _make_container(
    name: str, num_gpus: int = 1, memory: int = 0, concurrent_inputs: int = 8
):
    "Helper function to create a container with the given GPU configuration."

    gpu = modal.gpu.A100(count=num_gpus, memory=memory)

    class _VllmContainer(VllmEngine):
        def __init__(
            self,
            model_path: str,
            max_model_len: Optional[int] = None,
        ):
            if num_gpus > 1:
                # Patch issue from https://github.com/vllm-project/vllm/issues/1116
                import ray

                ray.shutdown()
                ray.init(num_gpus=num_gpus, ignore_reinit_error=True)

            super().__init__(
                VllmParams(
                    model=model_path,
                    tensor_parallel_size=num_gpus,
                    max_model_len=max_model_len,
                )
            )

            # Performance improvement from https://github.com/vllm-project/vllm/issues/2073#issuecomment-1853422529
            if num_gpus > 1:
                import subprocess

                RAY_CORE_PIN_OVERRIDE = "cpuid=0 ; for pid in $(ps xo '%p %c' | grep ray:: | awk '{print $1;}') ; do taskset -cp $cpuid $pid ; cpuid=$(($cpuid + 1)) ; done"
                subprocess.call(RAY_CORE_PIN_OVERRIDE, shell=True)

    _VllmContainer.__name__ = name

    wrap = stub.cls(
        volumes={str(models_path): stub.models_volume},
        image=_vllm_image,
        gpu=gpu,
        allow_concurrent_inputs=concurrent_inputs,
        container_idle_timeout=20 * 60,
        timeout=10 * 60,
    )
    return wrap(_VllmContainer)


VllmContainer_7B = _make_container(
    "VllmContainer_7B", num_gpus=1, concurrent_inputs=100
)
VllmContainerA100_40G = _make_container(
    "VllmContainerA100_40G", num_gpus=1, concurrent_inputs=32
)
VllmContainerA100_80G = _make_container(
    "VllmContainerA100_80G", num_gpus=1, memory=80
)
VllmContainerA100_160G = _make_container(
    "VllmContainerA100_160G", num_gpus=2, memory=80, concurrent_inputs=5
)
