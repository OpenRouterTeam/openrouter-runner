from shared.volumes import models_path

from modal import Secret, web_endpoint
from runner.containers import all_models
from runner.endpoints.completion import completion
from runner.shared.common import stub
from runner.shared.download import download_models, downloader_image

stub.function(
    secret=Secret.from_name("ext-api-key"),
    timeout=60 * 15,
    allow_concurrent_inputs=100,
    volumes={str(models_path): stub.models_volume},
)(web_endpoint(method="POST")(completion))


@stub.function(
    image=downloader_image,
    volumes={str(models_path): stub.models_volume},
    secret=Secret.from_name("huggingface"),
    timeout=len(all_models) * 3600,  # 1 hour per model
)
def download():
    download_models(all_models)
