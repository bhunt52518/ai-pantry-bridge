# app/routers/generate.py
from __future__ import annotations

from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel

from app.clients.ollama import ollama
from app.services.automation_generator import generate_automation

router = APIRouter(tags=["generate"])


class GenerateReq(BaseModel):
    model: str = "qwen2.5:14b"
    intent: str
    callback_url: str


@router.post("/generate")
async def generate(req: GenerateReq):
    return await generate_automation(
        intent=req.intent,
        callback_url=req.callback_url,
        ollama_client=ollama,
        model=req.model,
    )

