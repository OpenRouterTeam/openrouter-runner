from modal import Stub

from shared.config import Config

config = Config(
    name="runner",
    api_key_id="RUNNER_API_KEY",
)

BACKLOG_THRESHOLD = 30

stub = Stub(config.name)
