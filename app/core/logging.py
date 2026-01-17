# app/core/logging.py
from __future__ import annotations

import logging
import os
import sys
from pythonjsonlogger import jsonlogger


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()

    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s %(method)s %(path)s %(status_code)s %(duration_ms)s"
    )
    handler.setFormatter(formatter)

    root.handlers = [handler]

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = [handler]
        lg.propagate = False
