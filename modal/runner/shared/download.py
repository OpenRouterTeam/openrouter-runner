from typing import List

from modal import Image

from shared.volumes import get_model_path, models_volume

downloader_image = (
    Image.debian_slim()
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .pip_install("huggingface_hub~=0.17.1")
    .pip_install("hf-transfer~=0.1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


def download_models(all_models: List[str]):
    from os import environ as env

    from huggingface_hub import list_repo_files, snapshot_download

    cache_path = get_model_path("__cache__")
    for model_name in all_models:
        model_path = get_model_path(model_name)
        model_path.mkdir(parents=True, exist_ok=True)

        try:
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

            snapshot_download(
                model_name,
                local_dir=model_path,
                local_files_only=True,
                cache_dir=cache_path,
                token=env["HUGGINGFACE_TOKEN"],
            )
            print(f"Volume contains {model_name}")
        except FileNotFoundError:
            print(f"Downloading {model_name} ...")
            snapshot_download(
                model_name,
                local_dir=model_path,
                cache_dir=cache_path,
                token=env["HUGGINGFACE_TOKEN"],
            )
            models_volume.commit()
    print("ALL DONE!")
