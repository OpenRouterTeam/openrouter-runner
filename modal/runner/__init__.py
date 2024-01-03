from fastapi import Depends, FastAPI
from fastapi.security import HTTPAuthorizationCredentials
from modal import Image, Secret, asgi_app

from runner.containers import all_models, all_models_lower
from runner.endpoints.completion import completion
from runner.endpoints.models import AddModelPayload, add_model
from runner.shared.clean import clean_models_volume
from runner.shared.common import config, stub
from runner.shared.download import download_models
from shared.protocol import CompletionPayload
from shared.volumes import models_path, models_volume

image = (
    Image.debian_slim()
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .pip_install("huggingface_hub~=0.17.1")
    .pip_install("hf-transfer~=0.1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

api_app = FastAPI()


@api_app.post("/")
async def post_completion(
    payload: CompletionPayload,
    _token: HTTPAuthorizationCredentials = Depends(config.auth),
):
    return completion(payload)


@api_app.post("/models")
async def post_model(
    payload: AddModelPayload,
    _token: HTTPAuthorizationCredentials = Depends(config.auth),
):
    return add_model(payload)


@stub.function(
    image=image,
    secrets=[Secret.from_name("ext-api-key"), Secret.from_name("huggingface")],
    timeout=60 * 15,
    allow_concurrent_inputs=100,
    volumes={models_path: models_volume},
)
@asgi_app()
def app():
    return api_app


@stub.function(
    image=image,
    volumes={models_path: models_volume},
    secret=Secret.from_name("huggingface"),
    timeout=len(all_models) * 3600,  # 1 hour per model
)
def download():
    download_models(all_models)


@stub.function(
    image=image,
    volumes={models_path: models_volume},
    secret=Secret.from_name("huggingface"),
    timeout=len(all_models) * 3600,  # 1 hours per model
)
def clean(all: bool = False, dry: bool = False):
    print(f"Cleaning models volume. ALL: {all}. DRY: {dry}")
    remaining_models = [] if all else all_models_lower
    clean_models_volume(remaining_models, dry)
