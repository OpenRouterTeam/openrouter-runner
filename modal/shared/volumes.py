from pathlib import Path

from modal import Volume

models_volume = Volume.persisted("models-volume")
models_path = Path("/models")


def get_model_path(model_name: str):
    return models_path / model_name.lower()


def does_model_exist(model_path: Path):
    if not model_path.exists():
        models_volume.reload()
    return model_path.exists()


loras_volume = Volume.persisted("loras-volume")
loras_path = Path("/loras")


def get_lora_path(user_name: str, lora_name: str):
    return loras_path / user_name.lower() / lora_name.lower()
