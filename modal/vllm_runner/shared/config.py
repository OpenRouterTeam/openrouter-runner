from modal import Stub, image
from modal.config import Config

modal_config = Config()

if modal_config.get(key="environment") == "dev":
    keep_warm = None
else:
    keep_warm = 1


model_path: str = "/model"
gpu_count: int = 1

api_key_id: str

model: str = ""
app_name: str

concurrent_inputs: int
