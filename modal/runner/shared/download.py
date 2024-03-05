from os import environ as env

import sentry_sdk
from modal import Secret

from runner.shared.common import stub
from shared.images import BASE_IMAGE
from shared.logging import get_logger, get_observability_secrets
from shared.volumes import (
    get_model_path,
    get_model_revision,
    get_repo_id,
    models_path,
    models_volume,
)

logger = get_logger(__name__)

cache_path = get_model_path("__cache__")

downloader_image = (
    BASE_IMAGE
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .pip_install("huggingface_hub==0.20.2")
    .pip_install("hf-transfer==0.1.4")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        }
    )
)


@stub.function(
    image=downloader_image,
    volumes={models_path: models_volume},
    secrets=[Secret.from_name("huggingface"), *get_observability_secrets()],
    timeout=3600,  # 1 hour per model
)
def download_model(model_name: str, force: bool = False):
    from huggingface_hub import HfApi

    hf = HfApi(token=env["HUGGINGFACE_TOKEN"])

    try:
        model_path = get_model_path(model_name)
        model_path.mkdir(parents=True, exist_ok=True)

        repo_id = get_repo_id(model_name)
        revision = get_model_revision(model_name)

        # only download safetensors if available
        has_safetensors = any(
            fn.lower().endswith(".safetensors")
            for fn in hf.list_repo_files(
                repo_id=repo_id,
                revision=revision,
                repo_type="model",
            )
        )
        ignore_patterns: list[str] = []
        if has_safetensors:
            ignore_patterns.append("*.pt")
            ignore_patterns.append("*.bin")

        # Clean doesn't remove the cache, so using `local_files_only` here returns the cache even when the local dir is empty.
        logger.info(f"Checking for {model_name}")
        hf.snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=model_path,
            cache_dir=cache_path,
            force_download=force,
            ignore_patterns=ignore_patterns,
        )
        logger.info(f"Volume now contains {model_name}")
        models_volume.commit()
    except Exception:
        logger.exception(f"Failed to download {model_name}")
        sentry_sdk.capture_exception()
        raise
