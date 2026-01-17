# app/services/health.py
from __future__ import annotations

import sqlite3
import time
from typing import Any, Dict, Optional

import httpx

from app.core import config


def _ms_since(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _check_result(status: str, latency_ms: int, error: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"status": status, "latency_ms": latency_ms}
    if error:
        out["error"] = error
    return out


def check_db() -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        conn = sqlite3.connect(str(config.PANTRY_DB), timeout=2)
        try:
            conn.execute("SELECT 1;")
        finally:
            conn.close()
        return _check_result("ok", _ms_since(start))
    except Exception as e:
        return _check_result("fail", _ms_since(start), str(e))


async def check_ollama() -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        # /api/tags is a cheap health-ish endpoint for Ollama
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/tags")
            r.raise_for_status()
        return _check_result("ok", _ms_since(start))
    except Exception as e:
        # degraded: parsing won't work, but pantry logic can
        return _check_result("degraded", _ms_since(start), str(e))


def version_payload() -> Dict[str, Any]:
    return {
        "version": config.APP_VERSION,
        "git_sha": config.GIT_SHA,
        "build_date": config.BUILD_DATE,
    }
