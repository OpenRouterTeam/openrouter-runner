import os

import modal.gpu
import sentry_sdk

from runner.engines.vllm import VllmEngine, VllmParams, vllm_image
from runner.shared.common import stub
from shared.config import is_env_dev
from shared.logging import (
    get_logger,
    get_observability_secrets,
)
from shared.protocol import GPUType
from shared.volumes import (
    does_model_exist,
    get_model_path,
    models_path,
    models_volume,
)


def _make_container(
    name: str,
    model_name: str,
    gpu: modal.gpu = modal.gpu.A100(count=1, memory=40),
    concurrent_inputs: int = 8,
    max_containers: int = None,
    keep_warm: int = None,
    **vllm_opts,
):
    """Helper function to create a container with the given GPU configuration."""

    num_gpus = gpu.count
    if isinstance(gpu, modal.gpu.A100):
        gpu_type = GPUType.A100_80G if gpu.memory == 80 else GPUType.A100_40G
    elif isinstance(gpu, modal.gpu.H100):
        gpu_type = GPUType.H100_80G
    else:
        raise ValueError(f"Unknown GPU type: {gpu}")

    # Avoid wasting resources & money in dev
    if keep_warm and is_env_dev():
        print("Dev environment detected, disabling keep_warm for", name)
        keep_warm = None

    class _VllmContainer(VllmEngine):
        def __init__(self):
            logger = get_logger(name)
            try:
                model_path = get_model_path(model_name=model_name)
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
                        **vllm_opts,
                    ),
                )

                # For any containers with keep_warm, we need to skip cold-start usage
                # billing. This is because the first request might be minutes after
                # the container is started, and we don't want to record that time as
                # usage.
                if keep_warm:
                    self.is_first_request = False

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
        keep_warm=keep_warm,
    )
    _cls = wrap(_VllmContainer)
    REGISTERED_CONTAINERS[model_name] = _cls
    return _cls


# A mapping of model names to their respective container classes.
# Automatically populated by _make_container.
REGISTERED_CONTAINERS = {}

VllmContainer_MicrosoftPhi2 = _make_container(
    name="VllmContainer_MicrosoftPhi2",
    model_name="microsoft/phi-2",
    gpu=modal.gpu.A100(count=1, memory=40),
    concurrent_inputs=120,
)
VllmContainer_IntelNeuralChat7B = _make_container(
    name="VllmContainer_IntelNeuralChat7B",
    model_name="Intel/neural-chat-7b-v3-1",
    gpu=modal.gpu.A100(count=1, memory=40),
    concurrent_inputs=100,
)
VllmContainer_JebCarterPsyfighter13B = _make_container(
    "VllmContainer_JebCarterPsyfighter13B",
    model_name="jebcarter/Psyfighter-13B",
    gpu=modal.gpu.A100(count=1, memory=40),
    concurrent_inputs=32,
)
VllmContainer_KoboldAIPsyfighter2 = _make_container(
    name="VllmContainer_KoboldAIPsyfighter2",
    model_name="KoboldAI/LLaMA2-13B-Psyfighter2",
    gpu=modal.gpu.A100(count=1, memory=40),
    concurrent_inputs=32,
)

_noromaid = "TheBloke/Noromaid-v0.1-mixtral-8x7b-Instruct-v3-GPTQ"
VllmContainer_NeverSleepNoromaidMixtral8x7B = _make_container(
    name="VllmContainer_NeverSleepNoromaidMixtral8x7B",
    model_name=_noromaid,
    gpu=modal.gpu.A100(count=1, memory=40),
    concurrent_inputs=4,
    max_containers=3,
    keep_warm=1,
    quantization="GPTQ",
    dtype="float16",  # vLLM errors when using dtype="auto" with this model
)

_bagel = "TheBloke/bagel-34b-v0.2-GPTQ"
VllmContainer_JohnDurbinBagel34B = _make_container(
    name="VllmContainer_JohnDurbinBagel34B",
    model_name=_bagel,
    gpu=modal.gpu.A100(count=1, memory=40),
    concurrent_inputs=4,
    max_containers=1,
    keep_warm=1,
    max_model_len=8_000,  # Reduced from original 200k
    quantization="GPTQ",
    dtype="float16",  # vLLM errors when using dtype="auto" with this model
)

_midnight_rose = "sambarnes/Midnight-Rose-70B-v2.0.3-GPTQ-naive"
VllmContainer_MidnightRose70B = _make_container(
    name="VllmContainer_MidnightRose70B",
    model_name=_midnight_rose,
    gpu=modal.gpu.H100(count=1),
    concurrent_inputs=4,
    max_containers=1,
    quantization="GPTQ",
    dtype="float16",  # vLLM errors when using dtype="auto" with this model
)

# A re-mapping of model names to their respective quantized models.
# From the outside, the model name is the original, but internally,
# we use the quantized model name.
QUANTIZED_MODELS = {
    "NeverSleep/Noromaid-v0.1-mixtral-8x7b-Instruct-v3": _noromaid,
    "jondurbin/bagel-34b-v0.2": _bagel,
    "sophosympatheia/Midnight-Rose-70B-v2.0.3": _midnight_rose,
}
