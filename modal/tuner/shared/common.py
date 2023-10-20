from modal import Stub, Volume
from pydantic import BaseModel
from pathlib import Path


class Config(BaseModel):
    name: str
    api_key_id: str


config = Config(
    name="lora",
    api_key_id="RUNNER_API_KEY",
)

stub = Stub(config.name)


stub.loras_volume = Volume.persisted("loras-volume")

loras_path = Path("/loras")


def get_lora_path(user_name: str, lora_name: str):
    return loras_path / user_name.lower() / lora_name.lower()
