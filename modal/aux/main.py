from modal import Secret, web_endpoint

from aux.shared.volume import models_path, models_volume
from aux.shared.common import stub

from aux.endpoints.completion import completion

stub.function(
    secret=Secret.from_name("ext-api-key"),
    timeout=60 * 60,
    allow_concurrent_inputs=32,
    volumes={str(models_path): models_volume},
)(web_endpoint(method="POST")(completion))
