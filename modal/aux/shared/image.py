from modal import Image, Secret, Stub, method, gpu, web_endpoint

from .common import config


def mk_gpu_image(model: str):
    def download_model():
        import os
        from huggingface_hub import snapshot_download
        from pathlib import Path

        # make MODEL_DIR if not existed
        Path(config.download_dir).mkdir(parents=True, exist_ok=True)

        snapshot_download(
            model,
            local_dir=config.download_dir,
            token=os.environ["HUGGINGFACE_TOKEN"],
        )

    return (
        Image.from_registry("nvcr.io/nvidia/pytorch:22.12-py3")
        .pip_install(
            "vllm == 0.1.7",
            # Pinned to Sep/11/2023
            # "vllm @ git+https://github.com/vllm-project/vllm.git@b9cecc26359794af863b3484a3464108b7d5ee5f",
            # Pinned to 08/15/2023
            # "vllm @ git+https://github.com/vllm-project/vllm.git@805de738f618f8b47ab0d450423d23db1e636fa2",
            "typing-extensions==4.5.0",  # >=4.6 causes typing issues
        )
        # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
        .pip_install("hf-transfer~=0.1")
        .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
        .run_function(
            download_model,
            secret=Secret.from_name("huggingface"),
            timeout=60 * 60,
        )
    )
