import json
import logging
import os
import sys

from datadog_api_client import ApiClient, Configuration
from datadog_api_client.v2.api.logs_api import LogsApi
from datadog_api_client.v2.model.content_encoding import ContentEncoding
from datadog_api_client.v2.model.http_log import HTTPLog
from datadog_api_client.v2.model.http_log_item import HTTPLogItem


def get_logger(name: str):
    return logging.getLogger(name)


# Define a custom logging handler that sends logs to Datadog
class DatadogHandler(logging.Handler):
    def __init__(self, api_client: ApiClient):
        self.api_client = api_client

        super().__init__()

    def emit(self, record):
        # Ignore debug messages
        if record.levelno == logging.DEBUG:
            return

        environment_name = os.getenv("DD_ENV") or "development"

        log_payload = {
            "python-logging": {
                "py-env": "development",
                "py-message": record.getMessage(),
                "py-status": record.levelname.lower(),
                "py-logger": record.name,
                "py-stacktrace": record.exc_info,
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
            "model": record.__dict__.get("model"),
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

            logs = LogsApi(api_client)

            logs.submit_log(
                content_encoding=ContentEncoding.DEFLATE,
                body=body,  # type: ignore
            )

        except Exception as e:
            print(f"Error sending log to Datadog: {e}")


if os.environ.get("DD_API_KEY") is not None:
    config = Configuration()
    config.api_key["apiKeyAuth"] = os.environ["DD_API_KEY"]
    config.server_variables["site"] = os.environ["DD_SITE"]

    api_client = ApiClient(configuration=config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(process)d] [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            DatadogHandler(api_client),
        ],
    )
