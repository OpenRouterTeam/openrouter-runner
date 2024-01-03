from modal import Secret, web_endpoint

from runner.containers import all_models, all_models_lower
from runner.endpoints.completion import completion
from runner.shared.clean import clean_models_volume
from runner.shared.common import stub
from runner.shared.download import download_models, downloader_image
from shared.volumes import models_path, models_volume

stub.function(
    secret=Secret.from_name("ext-api-key"),
    timeout=60 * 15,
    allow_concurrent_inputs=100,
    volumes={models_path: models_volume},
)(web_endpoint(method="POST")(completion))


@stub.function(
    image=downloader_image,
    volumes={models_path: models_volume},
    secret=Secret.from_name("huggingface"),
    timeout=len(all_models) * 3600,  # 1 hour per model
)
def download():
    download_models(all_models)


@stub.function(
    image=downloader_image,
    volumes={models_path: models_volume},
    secret=Secret.from_name("huggingface"),
    timeout=len(all_models) * 3600,  # 1 hours per model
)
def clean(all: bool = False, dry: bool = False):
    print(f"Cleaning models volume. ALL: {all}. DRY: {dry}")
    remaining_models = [] if all else all_models_lower
    clean_models_volume(remaining_models, dry)
