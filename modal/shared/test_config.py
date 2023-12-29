import os

from fastapi import Depends, FastAPI, testclient
from fastapi.security import HTTPAuthorizationCredentials

from shared.config import Config


def test_auth():
    """The API auth dependency should prevent unauthorized requests."""

    app = FastAPI()
    config = Config(name="test", api_key_id="RUNNER_API_KEY")
    os.environ["RUNNER_API_KEY"] = "abc123"

    @app.get("/test")
    def test(_token: HTTPAuthorizationCredentials = Depends(config.auth)):
        return "OK"

    with testclient.TestClient(app) as client:
        response = client.get("/test")
        assert response.status_code == 403

        response = client.get(
            "/test", headers={"Authorization": "Bearer invalid"}
        )
        assert response.status_code == 401

        response = client.get(
            "/test", headers={"Authorization": "Bearer abc123"}
        )
        assert response.status_code == 200
