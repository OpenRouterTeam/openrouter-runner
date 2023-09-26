from modal import Secret, web_endpoint

from runner.shared.common import stub, models_path

from runner.endpoints.completion import completion

stub.function(
    secret=Secret.from_name("ext-api-key"),
    timeout=60 * 60,
    allow_concurrent_inputs=32,
    volumes={str(models_path): stub.models_volume},
)(web_endpoint(method="POST")(completion))


from runner.shared.download import downloader_image, download_models
from runner.containers import all_models


@stub.function(
    image=downloader_image,
    volumes={str(models_path): stub.models_volume},
    secret=Secret.from_name("huggingface"),
    timeout=len(all_models) * 3600,  # 1 hours per model
)
def download():
    download_models(all_models)
