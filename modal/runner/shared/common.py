from modal import Stub

from shared import Config
from shared.volumes import models_volume


config = Config(
    name="runner",
    api_key_id="RUNNER_API_KEY",
)

stub = Stub(config.name)


stub.models_volume = models_volume
