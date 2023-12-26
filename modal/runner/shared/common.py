from shared.config import Config
from shared.volumes import loras_volume, models_volume

from modal import Stub

config = Config(
    name="runner",
    api_key_id="RUNNER_API_KEY",
)

BACKLOG_THRESHOLD = 42

stub = Stub(config.name)

stub.models_volume = models_volume
stub.loras_volume = loras_volume
