import os

from fastapi import Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from tuner.containers.mistral_7b_lora import Mistral7BLoraContainer
from tuner.shared.common import config

auth_scheme = HTTPBearer()


def create_lora(
    token: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
    if token.credentials != os.environ[config.api_key_id]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    tuner = Mistral7BLoraContainer()
    return StreamingResponse(
        tuner.generate.remote_gen(),
        media_type="text/event-stream",
    )
