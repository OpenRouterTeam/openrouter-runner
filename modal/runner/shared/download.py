from os import environ as env

from modal import Image, Secret

from runner.shared.common import stub
from shared.volumes import get_model_path, models_path, models_volume

cache_path = get_model_path("__cache__")

downloader_image = (
    Image.debian_slim()
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .pip_install("huggingface_hub~=0.17.1")
    .pip_install("hf-transfer~=0.1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@stub.function(
    image=downloader_image,
    volumes={models_path: models_volume},
    secret=Secret.from_name("huggingface"),
    timeout=3600,  # 1 hour per model
)
def download_model(model_name: str):
    from huggingface_hub import list_repo_files, snapshot_download

    model_path = get_model_path(model_name)
    model_path.mkdir(parents=True, exist_ok=True)

    # only download safetensors if available
    has_safetensors = any(
        fn.lower().endswith(".safetensors")
        for fn in list_repo_files(
            model_name,
            repo_type="model",
            token=env["HUGGINGFACE_TOKEN"],
        )
    )
    patterns = ["tokenizer.model", "*.json"]
    if has_safetensors:
        patterns.append("*.safetensors")
    else:
        patterns.append("*.bin")
    # TODO: Use these patterns?

    # Clean doesn't remove the cache, so using `local_files_only` here returns the cache even when the local dir is empty.
    print(f"Checking for {model_name}")
    snapshot_download(
        model_name,
        local_dir=model_path,
        cache_dir=cache_path,
        local_dir_use_symlinks=False,
        token=env["HUGGINGFACE_TOKEN"],
    )
    print(f"Volume now contains {model_name}")
    models_volume.commit()
