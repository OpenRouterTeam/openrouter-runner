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
    logger = logging.getLogger(name)
    logger.addHandler(DatadogHandler())
    return logger


# Define a custom logging handler that sends logs to Datadog
class DatadogHandler(logging.Handler):
    def emit(self, record):
        # Ignore debug messages
        if record.levelno == logging.DEBUG:
            return

        toJson = json.dumps(
            {
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
                "environment": os.getenv("DD_ENV") or "",
                "source": record.name,
                "modal": {
                    "cloudProvider": os.getenv("MODAL_CLOUD_PROVIDER") or "",
                    "environment": os.getenv("MODAL_ENVIRONMENT") or "",
                    "imageId": os.getenv("MODAL_IMAGE_ID") or "",
                    "region": os.getenv("MODAL_REGION") or "",
                    "taskId": os.getenv("MODAL_TASK_ID") or "",
                },
            }
        )

        # Send the log to Datadog using the Logs API
        try:
            body = HTTPLog(
                [
                    HTTPLogItem(
                        ddsource="Python",
                        ddtags="env:{}".format(os.getenv("DD_ENV")),
                        hostname=os.getenv("MODAL_TASK_ID") or "",
                        message=toJson,
                        service="openrouter-runner",
                        status=record.levelname.lower(),
                    ),
                ]
            )

            config = Configuration()
            config.api_key["apiKeyAuth"] = os.environ["DD_API_KEY"]
            config.server_variables["site"] = os.environ["DD_SITE"]

            with ApiClient(configuration=config) as api_client:
                logs = LogsApi(api_client)

                logs.submit_log(
                    content_encoding=ContentEncoding.DEFLATE,
                    body=body,  # type: ignore
                )
            # print(response)

        except Exception as e:
            print(f"Error sending log to Datadog: {e}")
            raise e


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(process)d] [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler(filename="./app.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
