from pydantic import BaseModel


class Config(BaseModel):
    name: str
    api_key_id: str
