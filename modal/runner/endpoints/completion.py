from fastapi import Request, status
from fastapi.responses import StreamingResponse

from runner.containers import DEFAULT_CONTAINER_TYPES, get_container
from runner.shared.common import BACKLOG_THRESHOLD
from runner.shared.sampling_params import SamplingParams
from shared.logging import get_logger, timer
from shared.protocol import (
    CompletionPayload,
    create_error_response,
)
from shared.volumes import does_model_exist, get_model_path

logger = get_logger(__name__)


def completion(
    request: Request,
    payload: CompletionPayload,
):
    model_path = get_model_path(payload.model)
    logger.info(
        "Received completion request",
        extra={
            "model": str(model_path),
            "user-agent": request.headers.get("user-agent"),
            "referer": request.headers.get("referer"),
            "ip": request.headers.get("x-real-ip")
            or request.headers.get("x-forwarded-for")
            or request.client.host,
        },
    )  # use path to match runner
    if not does_model_exist(model_path):
        message = f"Unable to locate model {payload.model}"
        logger.error(message)
        return create_error_response(
            status.HTTP_400_BAD_REQUEST,
            f"Unable to locate model {payload.model}",
        )

    container_type = (
        payload.runner.container
        if payload.runner
        else DEFAULT_CONTAINER_TYPES.get(payload.model)
    )

    if container_type is None:
        message = f"Unable to locate container type for model {payload.model}"
        logger.error(message)
        return create_error_response(
            status.HTTP_400_BAD_REQUEST,
            f"Unable to locate container type for model {payload.model}",
        )

    runner = get_container(model_path, container_type)

    stats = runner.generate.get_current_stats()
    logger.info(stats)
    if stats.backlog > BACKLOG_THRESHOLD:
        message = f"Backlog is too high: {stats.backlog}"
        logger.warning(message)
        return create_error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"Backlog is too high: {stats.backlog}",
        )

    # max_model_len = runner.max_model_len.remote()
    # input_ids = runner.tokenize_prompt.remote(payload)
    # token_num = len(input_ids)

    # if payload.params.max_tokens is None:
    #     max_tokens = max_model_len - token_num
    # else:
    #     max_tokens = payload.params.max_tokens

    # is_too_high = (token_num + max_tokens) > max_model_len

    # if is_too_high:
    #     return create_error_response(
    #         status.HTTP_400_BAD_REQUEST,
    #         f"This model's maximum context length is {max_model_len} tokens. "
    #         f"However, you requested {max_tokens + token_num} tokens "
    #         f"({token_num} in the messages, "
    #         f"{max_tokens} in the completion). "
    #         f"Please reduce the length of the messages or completion.",
    #     )

    try:
        sampling_params = SamplingParams(
            # early_stopping=payload.params.early_stopping,
            # length_penalty=payload.params.length_penalty,
            **payload.params.dict(),
        )
    except ValueError as e:
        logger.exception("Invalid sampling params")
        return create_error_response(status.HTTP_400_BAD_REQUEST, str(e))

    async def generate():
        with timer("runner.generate", str(model_path), container_type):
            async for text in runner.generate.remote_gen.aio(
                payload, sampling_params
            ):
                yield text

    return StreamingResponse(
        generate(),
        # runner.generate.remote_gen(payload, sampling_params),
        media_type="text/event-stream",
    )
