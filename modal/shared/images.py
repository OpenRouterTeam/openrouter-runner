from modal import Image


def add_observability(image: Image):
    return image.pip_install("datadog-api-client==2.21.0").pip_install(
        "sentry-sdk[fastapi]==1.39.1"
    )


BASE_IMAGE = add_observability(Image.debian_slim(python_version="3.10"))
