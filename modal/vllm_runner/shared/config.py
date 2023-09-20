from modal.config import Config as ModalConfig
from pydantic import BaseModel

modal_config = ModalConfig()
keep_warm = None if modal_config.get(key="environment") == "dev" else 1


class Config(BaseModel):
    name: str
    api_key_id: str

    download_dir: str

    num_gpu: int
    max_batched_tokens: int

    idle_timeout: int
    concurrency: int
