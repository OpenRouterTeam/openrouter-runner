from modal import Stub, Volume
from pydantic import BaseModel
from pathlib import Path


class Config(BaseModel):
    name: str
    api_key_id: str


config = Config(
    name="runner",
    api_key_id="RUNNER_API_KEY",
)

stub = Stub(config.name)


stub.models_volume = Volume.persisted("models-volume")

models_path = Path("/models")


def get_model_path(model_name: str):
    return models_path / model_name.lower()
