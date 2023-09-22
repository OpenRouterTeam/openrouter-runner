from modal import Secret, web_endpoint

from aux.shared.common import stub, models_path

from aux.endpoints.completion import completion

stub.function(
    secret=Secret.from_name("ext-api-key"),
    timeout=60 * 60,
    allow_concurrent_inputs=32,
    volumes={str(models_path): stub.models_volume},
)(web_endpoint(method="POST")(completion))


from aux.shared.download import downloader_image, download_models
from aux.containers import all_models


@stub.function(
    image=downloader_image,
    volumes={str(models_path): stub.models_volume},
    secret=Secret.from_name("huggingface"),
    timeout=all_models.count() * 120,
)
def download():
    download_models(all_models)
