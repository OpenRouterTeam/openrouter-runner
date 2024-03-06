import json
import logging
import os
import sys
import time
from contextlib import contextmanager

import sentry_sdk
from datadog_api_client import ApiClient, Configuration
from datadog_api_client.v2.api.logs_api import LogsApi
from datadog_api_client.v2.model.content_encoding import ContentEncoding
from datadog_api_client.v2.model.http_log import HTTPLog
from datadog_api_client.v2.model.http_log_item import HTTPLogItem
from modal import Image, Secret
from sentry_sdk.scrubber import DEFAULT_DENYLIST, EventScrubber

_sentry_denylist = DEFAULT_DENYLIST + ["prompt"]
sentry_sdk.init(
    dsn=os.environ.get("SENTRY_DSN"),
    environment=os.environ.get("SENTRY_ENVIRONMENT") or "development",
    send_default_pii=False,
    event_scrubber=EventScrubber(denylist=_sentry_denylist),
)


def get_logger(name: str):
    return logging.getLogger(name)


def get_observability_secrets():
    return [Secret.from_name("sentry"), Secret.from_name("datadog")]


def add_observability(image: Image):
    return image.pip_install("datadog-api-client==2.21.0").pip_install(
        "sentry-sdk[fastapi]==1.39.1"
    )


@contextmanager
def timer(
    action: str,
    model: str = None,
    tags: dict[str, str | int] = None,
) -> None:
    """
    A simple timer context manager with structured logging for its output.

    Args:
        action: The noun being timed
        model: Optional, used as a tag
        tags: Any additional tags to include in the structured log
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        # FIXME: this block doesnt seem to execute when an async function
        # is called from within the context manager. Look into making an
        # async variant.

        elapsed = time.perf_counter() - start

        extra = (tags or {}) | {"duration": elapsed}
        if model:
            extra["model"] = model

        logging.info(f"{action} execution profiled", extra=extra)


# skip natural LogRecord attributes
# http://docs.python.org/library/logging.html#logrecord-attributes
RESERVED_ATTRS: set[str] = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


def get_record_extras(record: logging.LogRecord) -> dict:
    """
    Extracts extra attributes from LogRecord object.

    Adapted from: https://github.com/madzak/python-json-logger/blob/master/src/pythonjsonlogger/jsonlogger.py#L47
    """
    extra = {}
    for key, value in record.__dict__.items():
        # this allows to have numeric keys
        if key not in RESERVED_ATTRS and not (
            hasattr(key, "startswith") and key.startswith("_")
        ):
            extra[key] = value
    return extra


class DatadogHandler(logging.Handler):
    def __init__(self, api_client: ApiClient):
        self.api_client = api_client

        super().__init__()

    def emit(self, record: logging.LogRecord):
        # Ignore debug messages
        if record.levelno == logging.DEBUG:
            return

        environment_name = os.getenv("DD_ENV") or "development"

        log_payload = get_record_extras(record) | {
            "python-logging": {
                "py-env": environment_name,
                "py-message": record.getMessage(),
                "py-status": record.levelname.lower(),
                "py-logger": record.name,
                "py-exception": record.exc_text,
                "py-line": record.lineno,
                "py-file": record.filename,
                "py-function": record.funcName,
                "py-level": record.levelno,
                "py-path": record.pathname,
                "py-thread": record.thread,
                "py-threadName": record.threadName,
                "py-process": record.process,
                "py-processName": record.processName,
                "py-args": record.args,
                "py-msecs": record.msecs,
                "py-relativeCreated": record.relativeCreated,
                "py-created": record.created,
                "py-module": record.module,
            },
            "message": record.getMessage(),
            "environment": environment_name,
            "source": record.name,
            "modal": {
                "cloudProvider": os.getenv("MODAL_CLOUD_PROVIDER") or "",
                "environment": os.getenv("MODAL_ENVIRONMENT") or "",
                "imageId": os.getenv("MODAL_IMAGE_ID") or "",
                "region": os.getenv("MODAL_REGION") or "",
                "taskId": os.getenv("MODAL_TASK_ID") or "",
            },
        }

        # Send the log to Datadog using the Logs API
        try:
            body = HTTPLog(
                [
                    HTTPLogItem(
                        ddsource="Python",
                        ddtags=f"env:{environment_name}",
                        hostname=os.getenv("MODAL_TASK_ID") or "",
                        message=json.dumps(log_payload),
                        service="openrouter-runner",
                        status=record.levelname.lower(),
                    ),
                ]
            )

            logs = LogsApi(self.api_client)

            logs.submit_log(
                content_encoding=ContentEncoding.DEFLATE,
                body=body,  # type: ignore
            )

        except Exception as e:
            print(f"Error sending log to Datadog: {e}")
            sentry_sdk.capture_exception(e)


handlers: list[logging.Handler] = [
    logging.StreamHandler(sys.stdout),
]

if os.environ.get("DD_API_KEY") is not None:
    config = Configuration()
    config.api_key["apiKeyAuth"] = os.environ["DD_API_KEY"]
    config.server_variables["site"] = os.environ["DD_SITE"]

    api_client = ApiClient(configuration=config)
    handlers.append(DatadogHandler(api_client))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(process)d] [%(levelname)s] %(message)s",
    handlers=handlers,
)
