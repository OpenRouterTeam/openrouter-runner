from modal.config import Config
from modal import Stub
from os import environ

modal_config = Config()
keep_warm = None if modal_config.get(key="environment") == "dev" else 1

stub = Stub(name=environ["RUNNER_NAME"])
