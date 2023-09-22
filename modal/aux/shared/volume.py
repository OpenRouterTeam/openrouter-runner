from pathlib import Path
from modal import Volume

models_volume = Volume.new()

models_path = Path("/models")


def get_model_path(model_name: str):
    return models_path / model_name.lower()
