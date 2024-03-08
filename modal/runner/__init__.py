from modal import Secret, asgi_app

from runner.containers.vllm_unified import REGISTERED_CONTAINERS
from runner.shared.clean import clean_models_volume
from runner.shared.common import stub
from runner.shared.download import download_model, downloader_image
from shared.images import BASE_IMAGE
from shared.logging import get_logger, get_observability_secrets
from shared.volumes import models_path, models_volume


@stub.function(
    image=BASE_IMAGE,
    secrets=[
        Secret.from_name("ext-api-key"),
        *get_observability_secrets(),
    ],
    timeout=60 * 15,
    allow_concurrent_inputs=100,
    volumes={models_path: models_volume},
    cpu=2,
    memory=1024,
)
@asgi_app()
def completion():  # named for backwards compatibility with the Modal URL
    from .api import api_app

    return api_app


@stub.function(
    image=downloader_image,
    timeout=3600,  # 1 hour
    volumes={models_path: models_volume},
    secrets=[
        Secret.from_name("huggingface"),
        *get_observability_secrets(),
    ],
)
def download(force: bool = False):
    logger = get_logger("download")
    logger.info("Downloading all models...")
    for model in REGISTERED_CONTAINERS:
        # Can't be parallelized because of a modal volume corruption issue
        download_model.local(model, force=force)
    logger.info("ALL DONE!")


@stub.function(
    image=BASE_IMAGE,
    volumes={models_path: models_volume},
    secrets=[
        *get_observability_secrets(),
    ],
)
def clean(all: bool = False, dry: bool = False):
    logger = get_logger("clean")
    logger.info(f"Cleaning models volume. ALL: {all}. DRY: {dry}")
    remaining_models = [] if all else [m.lower() for m in REGISTERED_CONTAINERS]
    clean_models_volume(remaining_models, dry)
