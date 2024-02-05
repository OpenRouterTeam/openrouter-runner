import os
from pathlib import Path
from typing import Optional

import modal.gpu
import sentry_sdk
from modal import Image

from runner.engines.vllm import VllmEngine, VllmParams
from runner.shared.common import stub
from shared.logging import (
    add_observability,
    get_logger,
    get_observability_secrets,
)
from shared.volumes import does_model_exist, models_path, models_volume

_vllm_image = add_observability(
    Image.from_registry(
        "nvidia/cuda:12.1.0-base-ubuntu22.04",
        add_python="3.10",
    ).pip_install("vllm==0.2.6", "sentry-sdk==1.39.1")
)


def _make_container(
    name: str,
    num_gpus: int = 1,
    memory: int = 0,
    concurrent_inputs: int = 8,
    num_containers: Optional[int] = None,
):
    "Helper function to create a container with the given GPU configuration."

    gpu = modal.gpu.A100(count=num_gpus, memory=memory)

    class _VllmContainer(VllmEngine):
        def __init__(
            self,
            model_path: Path,
            max_model_len: Optional[int] = None,
        ):
            logger = get_logger(name)
            try:
                if not does_model_exist(model_path):
                    raise Exception("Unable to locate model {}", model_path)

                if num_gpus > 1:
                    # HACK[1-20-2024]: Yesterday, Modal started populating this env var
                    # with GPU UUIDs. This breaks some assumption in Ray, so just unset
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

                    # Patch issue from https://github.com/vllm-project/vllm/issues/1116
                    import ray

                    ray.shutdown()
                    ray.init(num_gpus=num_gpus, ignore_reinit_error=True)

                super().__init__(
                    VllmParams(
                        model=str(model_path),
                        tensor_parallel_size=num_gpus,
                        max_model_len=max_model_len,
                    )
                )

                # Performance improvement from https://github.com/vllm-project/vllm/issues/2073#issuecomment-1853422529
                if num_gpus > 1:
                    import subprocess

                    RAY_CORE_PIN_OVERRIDE = "cpuid=0 ; for pid in $(ps xo '%p %c' | grep ray:: | awk '{print $1;}') ; do taskset -cp $cpuid $pid ; cpuid=$(($cpuid + 1)) ; done"
                    subprocess.call(RAY_CORE_PIN_OVERRIDE, shell=True)
            except Exception as e:
                # We have to manually capture and re-raise because Modal catches the exception upstream
                sentry_sdk.capture_exception(e)
                logger.exception(
                    "Failed to initialize VLLM engine",
                    extra={"model": str(model_path)},
                )
                raise e

    _VllmContainer.__name__ = name

    wrap = stub.cls(
        volumes={models_path: models_volume},
        image=_vllm_image,
        # Default CPU memory is 128 on modal. Request more memory for larger
        # windows of vLLM's batch loading weights into GPU memory.
        memory=1024,
        gpu=gpu,
        retries=1,
        allow_concurrent_inputs=concurrent_inputs,
        # Timeout for idle containers waiting for inputs to shut down (10 min)
        container_idle_timeout=10 * 60,
        # maximum execution time (10 min)
        timeout=10 * 60,
        keep_warm=num_containers,
        concurrency_limit=num_containers,
        secrets=[*get_observability_secrets()],
    )
    return wrap(_VllmContainer)


VllmContainer_3B = _make_container(
    "VllmContainer_3B", num_gpus=1, concurrent_inputs=120
)

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
    "VllmContainerA100_160G",
    num_gpus=2,
    memory=80,
    concurrent_inputs=4,
    num_containers=1,
)

# Allow new models to be tested on the isolated container
VllmContainerA100_160G_Isolated = _make_container(
    "VllmContainerA100_160G_Isolated",
    num_gpus=2,
    memory=80,
    concurrent_inputs=4,
    num_containers=1,
)
