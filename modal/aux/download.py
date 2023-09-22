from modal import Secret, Image
from .shared.common import stub

from .shared.volume import get_model_path, models_path, models_volume
from .engines import vllm_13b_model_ids

_downloader_image = (
    Image.debian_slim()
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .pip_install("hf-transfer~=0.1").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

all_models = [*vllm_13b_model_ids]


# Run with:
#   modal run aux/download.py
@stub.function(
    image=_downloader_image,
    volumes={str(models_path): models_volume},
    secret=Secret.from_name("huggingface"),
)
def download():
    from huggingface_hub import snapshot_download
    from os import environ as env

    for model_name in all_models:
        model_path = get_model_path(model_name)
        model_path.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                model_name,
                local_dir=str(model_path),
                local_files_only=True,
                token=env["HUGGINGFACE_TOKEN"],
            )
            print(f"Volume contains {model_name}.")
        except FileNotFoundError:
            print(f"Downloading {model_name} ...")
            snapshot_download(
                model_name,
                local_dir=str(model_path),
                token=env["HUGGINGFACE_TOKEN"],
            )
    models_volume.commit()
