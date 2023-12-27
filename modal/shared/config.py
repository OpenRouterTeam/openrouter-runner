import os
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

_auth = HTTPBearer()


class Config(BaseModel):
    name: str
    api_key_id: str

    def auth(self, token: Annotated[HTTPAuthorizationCredentials, Depends(_auth)]) -> HTTPAuthorizationCredentials:
        """
        API Authentication dependency for protected endpoints. Checks that the request's bearer token
        matches the server's configured API key.

        Raises: HTTPException(401) if the token is invalid.
        """
        if token.credentials != os.environ[self.api_key_id]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return token
