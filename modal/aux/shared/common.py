from modal import Stub
from pydantic import BaseModel


class Config(BaseModel):
    name: str
    api_key_id: str


config = Config(
    name="aux",
    api_key_id="AUX_API_KEY",
)

stub = Stub(config.name)
