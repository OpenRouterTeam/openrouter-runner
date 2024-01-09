from modal import Stub

from shared.config import Config

config = Config(
    name="tuner",
    api_key_id="RUNNER_API_KEY",
)

stub = Stub(config.name)
