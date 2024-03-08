import os
from pathlib import Path
from typing import Optional

import modal.gpu
import sentry_sdk

from runner.engines.vllm import VllmEngine, VllmParams, vllm_image
from runner.shared.common import stub
from shared.logging import (
    get_logger,
    get_observability_secrets,
)
from shared.protocol import GPUType
from shared.volumes import does_model_exist, models_path, models_volume


def _make_container(
    name: str,
    gpu: modal.gpu = modal.gpu.A100(count=1, memory=40),
    concurrent_inputs: int = 8,
    max_containers: int = None,
):
    """Helper function to create a container with the given GPU configuration."""

    num_gpus = gpu.count
    if isinstance(gpu, modal.gpu.A100):
        gpu_type = GPUType.A100_80G if gpu.memory == 80 else GPUType.A100_40G
    elif isinstance(gpu, modal.gpu.H100):
        gpu_type = GPUType.H100_80G
    else:
        raise ValueError(f"Unknown GPU type: {gpu}")

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
                    gpu_type=gpu_type,
                    params=VllmParams(
                        model=str(model_path),
                        tensor_parallel_size=num_gpus,
                        max_model_len=max_model_len,
                    ),
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
        image=vllm_image,
        # Default CPU memory is 128 on modal. Request more memory for larger
        # windows of vLLM's batch loading weights into GPU memory.
        memory=1024,
        gpu=gpu,
        allow_concurrent_inputs=concurrent_inputs,
        container_idle_timeout=20 * 60,
        timeout=10 * 60,
        secrets=[*get_observability_secrets()],
        concurrency_limit=max_containers,
    )
    return wrap(_VllmContainer)


VllmContainer_MicrosoftPhi2 = _make_container(
    name="VllmContainer_MicrosoftPhi2",
    gpu=modal.gpu.A100(count=1, memory=40),
    concurrent_inputs=120,
)
VllmContainer_IntelNeuralChat7B = _make_container(
    name="VllmContainer_IntelNeuralChat7B",
    gpu=modal.gpu.A100(count=1, memory=40),
    concurrent_inputs=100,
)
VllmContainer_JebCarterPsyfighter13B = _make_container(
    "VllmContainer_JebCarterPsyfighter13B",
    gpu=modal.gpu.A100(count=1, memory=40),
    concurrent_inputs=32,
)
VllmContainer_KoboldAIPsyfighter2 = _make_container(
    name="VllmContainer_KoboldAIPsyfighter2",
    gpu=modal.gpu.A100(count=1, memory=40),
    concurrent_inputs=32,
)
VllmContainer_NeverSleepNoromaidMixtral8x7B = _make_container(
    name="VllmContainer_NeverSleepNoromaidMixtral8x7B",
    gpu=modal.gpu.A100(count=2, memory=80),
    concurrent_inputs=4,
    max_containers=3,
)
VllmContainer_JohnDurbinBagel34B = _make_container(
    name="VllmContainer_JohnDurbinBagel34B",
    gpu=modal.gpu.A100(count=2, memory=80),
    concurrent_inputs=4,
    max_containers=1,
)
