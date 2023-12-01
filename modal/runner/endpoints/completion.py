import os

from fastapi import Depends, HTTPException, status

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse

from runner.shared.sampling_params import SamplingParams
from shared.protocol import (
    create_error_response,
    Payload,
)

from runner.shared.common import config, BACKLOG_THRESHOLD
from runner.containers import get_container

auth_scheme = HTTPBearer()


def completion(
    payload: Payload,
    token: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
    if token.credentials != os.environ[config.api_key_id]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        runner = get_container(payload.model)
        stats = runner.generate.get_current_stats()
        print(stats)
        if stats.backlog > BACKLOG_THRESHOLD:
            return create_error_response(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                f"Backlog is too high: {stats.backlog}",
            )
    except ValueError as e:
        return create_error_response(status.HTTP_400_BAD_REQUEST, str(e))

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
        return create_error_response(status.HTTP_400_BAD_REQUEST, str(e))

    return StreamingResponse(
        runner.generate.remote_gen(payload, sampling_params),
        media_type="text/event-stream",
    )
