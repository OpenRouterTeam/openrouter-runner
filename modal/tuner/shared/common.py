from modal import Stub
from shared.config import Config
from shared.volumes import loras_volume, models_volume

config = Config(
    name="tuner",
    api_key_id="RUNNER_API_KEY",
)

stub = Stub(config.name)

stub.loras_volume = loras_volume
stub.models_volume = models_volume
