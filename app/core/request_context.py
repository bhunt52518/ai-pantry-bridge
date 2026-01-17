# app/core/request_context.py
from __future__ import annotations

import contextvars

request_id_ctx = contextvars.ContextVar("request_id", default=None)


def get_request_id() -> str | None:
    return request_id_ctx.get()


def set_request_id(value: str | None) -> None:
    request_id_ctx.set(value)
