from typing import List, Optional

from fastapi import Depends
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from modal import Image, Secret, Stub, gpu, method, web_endpoint
from pydantic import BaseModel

from shared.config import Config
from shared.protocol import (
    create_error_text,
)


class Payload(BaseModel):
    prompt: str
    num_outputs: int = 1
    num_inference_steps: int = 32
    extension: Optional[str] = None


class Generation(BaseModel):
    uri: Optional[str] = None
    url: Optional[str] = None


class ResponseBody(BaseModel):
    generations: List[Generation]


def download_models():
    import torch
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
    from shap_e.models.download import load_config, load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model("transmitter", device=device)
    load_model("text300M", device=device)
    diffusion_from_config(load_config("diffusion"))


_gpu = gpu.T4(count=1)
_image = (
    Image.from_registry("nvcr.io/nvidia/pytorch:23.09-py3")
    .pip_install("torch")
    .pip_install("torchvision")
    .pip_install("ipywidgets")
    # .pip_install(
    #     "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@stable",
    #     gpu=_gpu,
    # )
    .pip_install("shap-e @ git+https://github.com/openai/shap-e.git")
    .run_function(download_models, gpu=_gpu)
)


config = Config(
    name="shap-e",
    api_key_id="RUNNER_API_KEY",
)

auth_scheme = HTTPBearer()

stub = Stub(config.name)


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
        from shap_e.models.download import load_config, load_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.xm = load_model("transmitter", device=device)
        self.model = load_model("text300M", device=device)
        self.diffusion = diffusion_from_config(load_config("diffusion"))

    @method()
    def generate(self, payload: Payload):
        import threading
        import time

        output = [
            None
        ]  # Use a list to hold the output to bypass Python's scoping limitations
        output_ready = threading.Event()

        def make_object():
            import base64
            import io

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
                outputs = []

                for latent in latents:
                    t = decode_latent_mesh(self.xm, latent).tri_mesh()
                    buffer = io.BytesIO()

                    t.write_ply(buffer)

                    buffer.seek(0)

                    # Encode the buffer content to base64
                    base64_data = base64.b64encode(buffer.read()).decode(
                        "utf-8"
                    )
                    outputs.append(
                        Generation(
                            uri=f"data:application/x-ply;base64,{base64_data}"
                        )
                    )

                output[0] = ResponseBody(generations=outputs).json(
                    ensure_ascii=False
                )

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
    payload: Payload,
    _token: HTTPAuthorizationCredentials = Depends(config.auth),
):
    p = Model()

    return StreamingResponse(
        p.generate.remote_gen(payload),
        media_type="text/event-stream",
    )
