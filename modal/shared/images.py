from modal import Image


def add_datadog(image: Image):
    return image.pip_install("datadog-api-client==2.21.0")


BASE_IMAGE = add_datadog(Image.debian_slim(python_version="3.10"))
