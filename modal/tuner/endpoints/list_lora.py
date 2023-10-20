import os

from fastapi import Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from tuner.shared.common import config, loras_path

auth_scheme = HTTPBearer()


def list_lora(
    token: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
    if token.credentials != os.environ[config.api_key_id]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Get all files from the loras volume

    files = os.listdir(loras_path)

    # Return the list of files
    return JSONResponse(
        content=files,
        status_code=status.HTTP_200_OK,
    )
