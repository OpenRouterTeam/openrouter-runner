from pathlib import Path

from modal import Volume

models_volume = Volume.from_name("models-volume", create_if_missing=True)
models_path = Path("/models")


def get_model_path(model_name: str):
    return models_path / model_name.lower()


# gets passed a str in some circumstances despite being typed as Path
def does_model_exist(model_path: Path | str):
    model_path = Path(model_path)
    if not model_path.exists():
        models_volume.reload()
    return model_path.exists()


# get the model revision, if one is specified by a colon
# if there is no model revision, it returns None (which will select the main revision)
# example: TheBloke/Llama-2-70B-GPTQ:gptq-3bit--1g-actorder_True
#      returns -> gptq-3bit--1g-actorder_True
def get_model_revision(model_name: str):
    parts = model_name.split(":", 1)
    return parts[1] if len(parts) > 1 else None


# get the huggingface username/model id, without the revision
def get_repo_id(model_name: str):
    return model_name.split(":", 1)[0]
