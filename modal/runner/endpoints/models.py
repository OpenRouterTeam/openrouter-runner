from pydantic import BaseModel

from runner.shared.download import download_models


class AddModelPayload(BaseModel):
    name: str


def add_model(
    payload: AddModelPayload,
):
    download_models([payload.name])
