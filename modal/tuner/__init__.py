from shared.volumes import loras_path

from modal import Secret, web_endpoint
from tuner.endpoints.create_lora import create_lora
from tuner.endpoints.list_lora import list_lora
from tuner.shared.common import stub

stub.function(
    secret=Secret.from_name("ext-api-key"),
    timeout=60 * 60 * 5,  # 5 hours
    allow_concurrent_inputs=32,
    volumes={str(loras_path): stub.loras_volume},
)(web_endpoint(method="POST")(create_lora))

stub.function(
    secret=Secret.from_name("ext-api-key"),
    timeout=60 * 60,
    allow_concurrent_inputs=32,
    volumes={str(loras_path): stub.loras_volume},
)(web_endpoint(method="GET")(list_lora))
