from modal import Secret, asgi_app

from runner.containers import (
    DEFAULT_CONTAINER_TYPES,
)
from runner.shared.clean import clean_models_volume
from runner.shared.common import stub
from runner.shared.download import download_model
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
    image=BASE_IMAGE,
    timeout=3600,  # 1 hour
    secrets=[
        *get_observability_secrets(),
    ],
)
def download():
    logger = get_logger("download")
    logger.info("Downloading all models...")
    results = list(download_model.map(DEFAULT_CONTAINER_TYPES.keys()))
    if not results:
        raise Exception("Failed to perform remote calls")
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
    remaining_models = (
        [] if all else [m.lower() for m in DEFAULT_CONTAINER_TYPES]
    )
    clean_models_volume(remaining_models, dry)
