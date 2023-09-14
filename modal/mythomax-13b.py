# PREREQUISITES:
# 1. Create a modal secret group
#   HUGGINGFACE_TOKEN = <your huggingface token>
#   with name "huggingface"
# 2. Create a modal secret group
#   MYTHALION_API_KEY = <generate a random key>
#   with name "ext-api-key"
# 3. modal deploy

from modal import Image, Secret, Stub
from modal.config import Config

from shared.download import download_model_to_folder
from shared.gpu_model import create_gpu_model, GpuParams
from shared.cpu_endpoint import create_cpu_completion_endpoint, CpuParams

NAME = "mythomax-13b"
MODEL_DIR = "/model"

NUM_GPU = 1
MODEL = "Gryphe/MythoMax-L2-13b"

config = Config()

if config.get(key="environment") == "dev":
    KEEP_WARM = None
else:
    KEEP_WARM = 1

API_KEY_ID = "MYTHOMAX_API_KEY"
# MODEL = "Undi95/ReMM-SLERP-L2-13B"
# MODEL = "Gryphe/MythoMax-L2-13b"


def download():
    download_model_to_folder(MODEL, MODEL_DIR)


# image = (
#     Image.from_registry("nvcr.io/nvidia/pytorch:22.12-py3")
#     .pip_install("torch==2.0.1", index_url="https://download.pytorch.org/whl/cu118")
#     # Pinned to 08/15/2023
#     .pip_install(
#         "vllm @ git+https://github.com/vllm-project/vllm.git@805de738f618f8b47ab0d450423d23db1e636fa2",
#         "typing-extensions==4.5.0",  # >=4.6 causes typing issues
#     )
#     # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
#     .pip_install("hf-transfer~=0.1")
#     .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
#     .run_function(
#         download,
#         secret=Secret.from_name("huggingface"),
#         timeout=60 * 20,
#     )
# )

image = (
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
        download,
        secret=Secret.from_name("huggingface"),
        timeout=60 * 20,
    )
)

stub = Stub(NAME, image=image)

Model = create_gpu_model(
    stub,
    GpuParams(
        model_dir=MODEL_DIR,
        gpu_count=NUM_GPU,
        gpu_memory=20,
        keep_warm=KEEP_WARM,
    ),
)

completion_endpoint = create_cpu_completion_endpoint(
    stub,
    CpuParams(
        keep_warm=KEEP_WARM,
        api_key_id=API_KEY_ID,
    ),
    Model,
)
