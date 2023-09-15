def download_model():
    from huggingface_hub import snapshot_download
    from pathlib import Path
    import os

    model_path = os.environ["RUNNER_MODEL_PATH"]

    # make MODEL_DIR if not existed
    Path(model_path).mkdir(parents=True, exist_ok=True)

    snapshot_download(
        os.environ["RUNNER_MODEL"],
        local_dir=model_path,
        token=os.environ["HUGGINGFACE_TOKEN"],
    )
