import os


def download_model_to_folder(model: str, model_dir: str):
    from huggingface_hub import snapshot_download
    from pathlib import Path

    # make MODEL_DIR if not existed
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    snapshot_download(
        model,
        local_dir=model_dir,
        token=os.environ["HUGGINGFACE_TOKEN"],
    )
