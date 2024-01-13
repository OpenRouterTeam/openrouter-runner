from modal import Image

from shared.logging import add_observability

BASE_IMAGE = add_observability(Image.debian_slim(python_version="3.10"))
