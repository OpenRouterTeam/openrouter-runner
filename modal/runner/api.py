import modal
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel

from runner.endpoints.completion import completion as completion_endpoint
from runner.shared.common import config
from runner.shared.download import download_model
from shared.logging import get_logger
from shared.protocol import CompletionPayload

api_app = FastAPI()
logger = get_logger(__name__)


@api_app.middleware("http")
async def log_errors(request: Request, call_next):
    response = await call_next(request)
    if response.status_code >= 400:
        # Log full request URL for easier debugging
        logger.warning(
            f"Request: {request.method} {request.url}, Response: {response.status_code}"
        )
    return response


@api_app.post("/")  # for backwards compatibility with the Modal URL
@api_app.post("/completion")
async def post_completion(
    payload: CompletionPayload,
    request: Request,
    _token: HTTPAuthorizationCredentials = Depends(config.auth),
):
    return completion_endpoint(request, payload)


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
    except modal.exception.ExecutionError as err:
        logger.warning(
            f"Error calling remote function: {err}"
        )  # skip sentry since it'll be logged by the underlying container
        return JSONResponse(
            content={"error": f"{err}"},
            status_code=500,
        )
    except Exception as err:
        logger.exception(err)
        return JSONResponse(
            content={"error": f"{err}"},
            status_code=500,
        )

    return result
