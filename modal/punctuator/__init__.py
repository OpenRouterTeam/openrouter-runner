import os

from fastapi import Depends
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials
from modal import Image, Secret, gpu, method, web_endpoint
from pydantic import BaseModel

from runner.shared.common import stub
from shared.config import Config
from shared.protocol import (
    create_error_text,
    create_response_text,
)


class Payload(BaseModel):
    input: str


MODEL_DIR = "/test-model"
BASE_MODEL = "oliverguhr/fullstop-punctuation-multilang-large"


def download_models():
    from huggingface_hub import snapshot_download

    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        BASE_MODEL,
        local_dir=MODEL_DIR,
        token=os.environ["HUGGINGFACE_TOKEN"],
    )


_gpu = gpu.A10G(count=1)
_image = (
    Image.from_registry("nvcr.io/nvidia/pytorch:22.12-py3")
    .pip_install(
        "torch==2.0.1+cu118", index_url="https://download.pytorch.org/whl/cu118"
    )
    .pip_install("sentencepiece")
    .pip_install("deepmultilingualpunctuation")
    .pip_install("hf-transfer~=0.1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_models, secret=Secret.from_name("huggingface"))
)


config = Config(
    name="punctuator",
    api_key_id="RUNNER_API_KEY",
)


@stub.cls(
    image=_image,
    gpu=_gpu,
    allow_concurrent_inputs=16,
    container_idle_timeout=5 * 60,  # 5 minutes
)
class Punctuator:
    def __init__(
        self,
    ):
        from deepmultilingualpunctuation import PunctuationModel

        self.model = PunctuationModel(model=MODEL_DIR)

    @method()
    def transform(self, input_str: str):
        import threading
        import time

        output = [
            None
        ]  # Use a list to hold the output to bypass Python's scoping limitations
        output_ready = threading.Event()

        def punctuate_thread():
            try:
                output[0] = create_response_text(
                    self.model.restore_punctuation(input_str)
                )
            except Exception as err:
                output[0] = create_error_text(err)
                print(output[0])
            finally:
                output_ready.set()

        threading.Thread(target=punctuate_thread).start()

        # Continuously yield an empty space until the thread is done
        while not output_ready.is_set():
            yield " "
            time.sleep(0.1)  # Adjust sleep time as needed

        # Yield the final output
        yield output[0]


@stub.function(
    secret=Secret.from_name("ext-api-key"),
    timeout=60 * 60,
    allow_concurrent_inputs=32,
)
@web_endpoint(method="POST")
def punct(
    payload: Payload,
    _token: HTTPAuthorizationCredentials = Depends(config.auth),
):
    p = Punctuator()

    return StreamingResponse(
        p.transform.remote_gen(payload.input),
        media_type="text/event-stream",
    )
