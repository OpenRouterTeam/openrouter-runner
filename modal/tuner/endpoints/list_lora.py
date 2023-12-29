import os

from fastapi import Depends, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
from shared.volumes import loras_path
from tuner.shared.common import config


def list_lora(_token: HTTPAuthorizationCredentials = Depends(config.auth)):
    # Get all files from the loras volume

    files = os.listdir(loras_path)

    # Return the list of files
    return JSONResponse(
        content=files,
        status_code=status.HTTP_200_OK,
    )
