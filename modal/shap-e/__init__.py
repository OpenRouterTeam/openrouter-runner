from modal import Secret, web_endpoint, Stub, gpu, Image, method

from runner.shared.common import stub
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from pydantic import BaseModel
from typing import Optional

from shared.config import Config
from shared.protocol import (
    create_error_text,
)


class Payload(BaseModel):
    prompt: str
    num_outputs: int
    num_inference_steps: int
    extension: Optional[str] = None


MODEL_DIR = "/test-model"
BASE_MODEL = "oliverguhr/fullstop-punctuation-multilang-large"


_gpu = gpu.T4(count=1)
_image = (
    Image.from_registry("nvcr.io/nvidia/pytorch:22.12-py3")
    .pip_install(
        "torch==2.0.1+cu118", index_url="https://download.pytorch.org/whl/cu118"
    )
    .pip_install("shap-e @ git+https://github.com/openai/shap-e.git")
)


config = Config(
    name="shap-2",
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
class Model:
    def __init__(
        self,
    ):
        import torch
        from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
        from shap_e.models.download import load_model, load_config

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.xm = load_model("transmitter", device=device)
        self.model = load_model("text300M", device=device)
        self.diffusion = diffusion_from_config(load_config("diffusion"))

    @method()
    def generate(self, payload: Payload):
        import threading
        import time
        import os
        import base64
        import uuid

        output = [
            None
        ]  # Use a list to hold the output to bypass Python's scoping limitations
        output_ready = threading.Event()

        def make_object():
            from shap_e.diffusion.sample import sample_latents
            from shap_e.util.notebooks import decode_latent_mesh

            try:
                prompt = payload.prompt
                batch_size = payload.num_outputs
                inference_steps = payload.num_inference_steps
                guidance_scale = 15.0
                latents = sample_latents(
                    batch_size=batch_size,
                    model=self.model,
                    diffusion=self.diffusion,
                    guidance_scale=guidance_scale,
                    model_kwargs=dict(texts=[prompt] * batch_size),
                    progress=True,
                    clip_denoised=True,
                    use_fp16=True,
                    use_karras=True,
                    karras_steps=inference_steps,
                    sigma_min=1e-3,
                    sigma_max=160,
                    s_churn=0,
                )
                data = []
                for i, latent in enumerate(latents):
                    t = decode_latent_mesh(self.xm, latent).tri_mesh()
                    file_id = str(uuid.uuid4())
                    filename = f"{file_id}.ply"
                    with open(filename, "wb") as f:
                        t.write_ply(f)
                    with open(filename, "rb") as f:
                        base64_data = base64.b64encode(f.read()).decode("utf-8")
                        data_uri = (
                            f"data:application/x-ply;base64,{base64_data}"
                        )
                        output[0] = data_uri
                    os.remove(filename)
                return data
            except Exception as err:
                output[0] = create_error_text(err)
            finally:
                output_ready.set()

        threading.Thread(target=make_object).start()
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
def create(
    payload: Payload, token: HTTPAuthorizationCredentials = Depends(auth_scheme)
):
    import os

    if token.credentials != os.environ[config.api_key_id]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    p = Model()

    return StreamingResponse(
        p.generate.remote_gen(payload),
        media_type="text/event-stream",
    )
