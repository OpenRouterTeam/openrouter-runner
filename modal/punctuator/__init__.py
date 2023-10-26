from modal import Secret, web_endpoint, Stub, gpu, Image, method

from runner.shared.common import stub
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from pydantic import BaseModel

from shared.config import Config
from shared.protocol import (
    create_response_text,
    create_error_text,
)


class Payload(BaseModel):
    input: str


def download_models():
    from simpletransformers.ner import NERModel

    NERModel(
        "bert",
        "felflare/bert-restore-punctuation",
        labels=[
            "OU",
            "OO",
            ".O",
            "!O",
            ",O",
            ".U",
            "!U",
            ",U",
            ":O",
            ";O",
            ":U",
            "'O",
            "-O",
            "?O",
            "?U",
        ],
        args={"silent": True, "max_seq_length": 512},
    )


_gpu = gpu.A10G(count=1)
_image = (
    Image.from_registry(
        # "nvcr.io/nvidia/pytorch:23.09-py3"
        "nvcr.io/nvidia/pytorch:22.12-py3",
    )
    .pip_install("rpunct")
    .pip_install(
        "torch==1.8.1+cu111",
        find_links="https://download.pytorch.org/whl/torch_stable.html",
    )
    .pip_install("hf-transfer~=0.1")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_models, secret=Secret.from_name("huggingface"), gpu="any"
    )
)


config = Config(
    name="punctuator",
    api_key_id="RUNNER_API_KEY",
)

stub = Stub(config.name)

auth_scheme = HTTPBearer()


@stub.cls(
    image=_image,
    gpu=_gpu,
    allow_concurrent_inputs=16,
    container_idle_timeout=5 * 60,  # 5 minutes
)
class Punctuator:
    def __enter__(
        self,
    ):
        from rpunct import RestorePuncts

        self.rpunct = RestorePuncts()
        pass

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
                    self.rpunct.punctuate(input_str)
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
    payload: Payload, token: HTTPAuthorizationCredentials = Depends(auth_scheme)
):
    import os

    if token.credentials != os.environ[config.api_key_id]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    p = Punctuator()

    return StreamingResponse(
        p.transform.remote_gen(payload.input),
        media_type="text/event-stream",
    )
