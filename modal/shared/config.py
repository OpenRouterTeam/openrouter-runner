import os
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

_auth = HTTPBearer()


def is_env_dev() -> bool:
    """Returns whether this is running in a development environment."""
    return os.getenv("DD_ENV", "development") == "development"


class Config(BaseModel):
    name: str
    api_key_id: str

    def auth(
        self, token: HTTPAuthorizationCredentials = Depends(_auth)
    ) -> HTTPAuthorizationCredentials:
        """
        API Authentication dependency for protected endpoints. Checks that the request's bearer token
        matches the server's configured API key.

        Raises:
            * HTTPException(403) if no token is provided.
            * HTTPException(401) if the token is invalid.
        """

        # Timing attacks possible through direct comparison. Prevent it with a constant time comparison here.
        got_credential = token.credentials.encode()
        want_credential = os.environ[self.api_key_id].encode()
        if not secrets.compare_digest(got_credential, want_credential):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return token
