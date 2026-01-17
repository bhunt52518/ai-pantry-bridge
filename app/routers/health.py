# app/routers/health.py
from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.services.health import check_db, check_ollama, version_payload

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    # Liveness only: if the process is serving requests, it's up
    return {"status": "ok", **version_payload()}


@router.get("/health/ready")
async def ready(response: Response):
    db = check_db()
    ollama = await check_ollama()

    overall = "ok"
    http_status = status.HTTP_200_OK

    # DB is required
    if db["status"] != "ok":
        overall = "fail"
        http_status = status.HTTP_503_SERVICE_UNAVAILABLE
    elif ollama["status"] != "ok":
        overall = "degraded"

    response.status_code = http_status
    return {
        "status": overall,
        "checks": {"db": db, "ollama": ollama},
        **version_payload(),
    }


@router.get("/version")
def version():
    return version_payload()
