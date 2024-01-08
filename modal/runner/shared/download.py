from typing import List

from modal import Image

from shared.volumes import get_model_path

from .common import stub

downloader_image = (
    Image.debian_slim()
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .pip_install(
        "huggingface_hub==0.19.4",
    )
    .pip_install(
        "hf-transfer==0.1.4",
    )
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
            ignore_patterns = []
            if has_safetensors:
                ignore_patterns.append("*.pt")  # Using safetensors

            print(f"Checking for {model_name}")
            snapshot_download(
                model_name,
                local_dir=model_path,
                cache_dir=cache_path,
                local_files_only=True,
                local_dir_use_symlinks=False,
                ignore_patterns=ignore_patterns,
                token=env["HUGGINGFACE_TOKEN"],
            )
            print(f"Volume now contains {model_name}")
            stub.models_volume.commit()
        except FileNotFoundError:
            print(f"Downloading {model_name} ...")
            snapshot_download(
                model_name,
                local_dir=model_path,
                cache_dir=cache_path,
                ignore_patterns=ignore_patterns,
                token=env["HUGGINGFACE_TOKEN"],
            )
            stub.models_volume.commit()

    print("ALL DONE!")
