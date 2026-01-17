from fastapi import APIRouter, HTTPException
import json

from models_llm import SpeechFormatRequest, SpeechFormatResponse
from app.clients.ollama import ollama
from app.core.text import extract_json  # rename from _extract_json

router = APIRouter(prefix="/speech")

@router.post("/format", response_model=SpeechFormatResponse)
async def speech_format(req: SpeechFormatRequest):
    system = (
        "You format responses to be spoken by a smart home assistant.\n"
        "Return ONLY valid JSON. No markdown, no commentary.\n"
        "Schema:\n"
        '{ "speech": "string", "display": "string|null", "ssml": "string|null" }\n'
        "Rules:\n"
        "- speech must be 1–3 short sentences.\n"
        "- Avoid reading numbers or IDs unless necessary.\n"
        "- If there are options, end with a short question.\n"
    )

    user = f"Text to format:\n{req.text}"

    out = await ollama.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        timeout_s=60,
    )

    try:
        payload = json.loads(extract_json(out))
        resp = SpeechFormatResponse(**payload)
        if not resp.speech.strip():
            raise ValueError("Empty speech")
        return resp

    except Exception:
        fix = "Your previous output was invalid. Return ONLY valid JSON matching the schema exactly."
        out2 = await ollama.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "user", "content": fix},
            ],
            temperature=0.0,
            timeout_s=60,
        )
        payload = json.loads(extract_json(out2))
        return SpeechFormatResponse(**payload)
