from fastapi import Depends
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials

from tuner.containers.mistral_7b_lora import Mistral7BLoraContainer
from tuner.shared.common import config


def create_lora(_token: HTTPAuthorizationCredentials = Depends(config.auth)):
    tuner = Mistral7BLoraContainer()
    return StreamingResponse(
        tuner.generate.remote_gen(),
        media_type="text/event-stream",
    )
