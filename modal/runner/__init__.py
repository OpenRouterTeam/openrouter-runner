from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
from modal import Secret, asgi_app
from pydantic import BaseModel

from runner.containers import (
    DEFAULT_CONTAINER_TYPES,
)
from runner.endpoints.completion import completion as completion_endpoint
from runner.shared.clean import clean_models_volume
from runner.shared.common import config, stub
from runner.shared.download import download_model
from shared.protocol import CompletionPayload
from shared.volumes import models_path, models_volume

api_app = FastAPI()


@api_app.middleware("http")
async def log_errors(request: Request, call_next):
    response = await call_next(request)
    if response.status_code >= 400:
        # Log full request URL for easier debugging
        print(
            f"Request: {request.method} {request.url}, Response: {response.status_code}"
        )
    return response


@api_app.post("/")  # for backwards compatibility with the Modal URL
@api_app.post("/completion")
async def post_completion(
    payload: CompletionPayload,
    _token: HTTPAuthorizationCredentials = Depends(config.auth),
):
    return completion_endpoint(payload)


class AddModelPayload(BaseModel):
    name: str


@api_app.post("/models")
async def post_model(
    payload: AddModelPayload,
    _token: HTTPAuthorizationCredentials = Depends(config.auth),
):
    remote_call = download_model.spawn(payload.name)
    if not remote_call:
        raise Exception("Failed to spawn remote call")

    return {"job_id": remote_call.object_id}


@api_app.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    _token: HTTPAuthorizationCredentials = Depends(config.auth),
):
    from modal.functions import FunctionCall

    function_call = FunctionCall.from_id(job_id)
    try:
        result = function_call.get(timeout=0)
    except TimeoutError:
        return JSONResponse(content="", status_code=202)

    return result


@stub.function(
    secret=Secret.from_name("ext-api-key"),
    timeout=60 * 15,
    allow_concurrent_inputs=100,
    volumes={models_path: models_volume},
    cpu=2,
    memory=1024,
)
@asgi_app()
def completion():  # named for backwards compatibility with the Modal URL
    return api_app


@stub.function(
    timeout=3600,  # 1 hour
)
def download():
    print("Downloading all models...")
    results = list(download_model.map(DEFAULT_CONTAINER_TYPES.keys()))
    if not results:
        raise Exception("Failed to perform remote calls")
    print("ALL DONE!")


@stub.function(
    volumes={models_path: models_volume},
)
def clean(all: bool = False, dry: bool = False):
    print(f"Cleaning models volume. ALL: {all}. DRY: {dry}")
    remaining_models = (
        [] if all else [m.lower() for m in DEFAULT_CONTAINER_TYPES]
    )
    clean_models_volume(remaining_models, dry)
