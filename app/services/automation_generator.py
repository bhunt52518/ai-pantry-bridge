# app/services/automation_generator.py
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

from app.core.text import extract_json

SYSTEM_AUTOMATION = """You are a Home Assistant automation generator.

RULES:
- Output ONLY valid JSON.
- Do NOT include markdown, comments, or explanations.
- Do NOT include trailing commas.
- Do NOT include text outside the JSON object.
- All strings must be double-quoted.

The JSON schema MUST be:
{
  "valid": true | false,
  "summary": "short description",
  "assumptions": [],
  "helpers": [],
  "triggers": [],
  "conditions": [],
  "actions": [],
  "notes": []
}

REQUIREMENTS:
- Use Home Assistant service names exactly as documented.
- Entity IDs must be lowercase.
- If information is missing, set "valid" to false and explain in "notes".
- Prefer simplicity over cleverness.
- Never invent entities.
"""


def _ensure_schema(obj: Any) -> Dict[str, Any]:
    """
    Make sure the returned object is a dict and has all required keys.
    If not, force valid=false with notes explaining why.
    """
    if not isinstance(obj, dict):
        return {
            "valid": False,
            "summary": "Invalid generator output",
            "assumptions": [],
            "helpers": [],
            "triggers": [],
            "conditions": [],
            "actions": [],
            "notes": ["Model output was not a JSON object."],
        }

    required = ["valid", "summary", "assumptions", "helpers", "triggers", "conditions", "actions", "notes"]
    missing = [k for k in required if k not in obj]
    if missing:
        fixed = {
            "valid": False,
            "summary": obj.get("summary") if isinstance(obj.get("summary"), str) else "Invalid generator output",
            "assumptions": obj.get("assumptions") if isinstance(obj.get("assumptions"), list) else [],
            "helpers": obj.get("helpers") if isinstance(obj.get("helpers"), list) else [],
            "triggers": obj.get("triggers") if isinstance(obj.get("triggers"), list) else [],
            "conditions": obj.get("conditions") if isinstance(obj.get("conditions"), list) else [],
            "actions": obj.get("actions") if isinstance(obj.get("actions"), list) else [],
            "notes": obj.get("notes") if isinstance(obj.get("notes"), list) else [],
        }
        fixed["notes"].append(f"Missing required keys: {', '.join(missing)}")
        return fixed

    return obj


async def _post_callback(callback_url: str, payload: Dict[str, Any]) -> None:
    """
    Preserve your existing flow: send the generated JSON to Home Assistant.
    Callback failures should not break the /generate response.
    """
    if not callback_url:
        return
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            await client.post(callback_url, json=payload)
    except Exception:
        # Preserve old behavior: swallow callback errors
        return


async def generate_automation(
    *,
    intent: str,
    callback_url: str,
    ollama_client: Any,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    prompt = f"{SYSTEM_AUTOMATION}\n\nUSER INTENT:\n{intent}\n"

    # Some OllamaClient implementations don’t accept model= kwarg.
    # To preserve your request schema, we temporarily swap the client’s model attribute if present.
    orig_model = getattr(ollama_client, "model", None)
    swapped = False
    if model and orig_model is not None and model != orig_model:
        try:
            setattr(ollama_client, "model", model)
            swapped = True
        except Exception:
            swapped = False

    try:
        out = await ollama_client.chat(
            [{"role": "system", "content": SYSTEM_AUTOMATION}, {"role": "user", "content": prompt}],
            temperature=0.0,
            timeout_s=120,
        )
    finally:
        if swapped:
            try:
                setattr(ollama_client, "model", orig_model)
            except Exception:
                pass

    try:
        obj = json.loads(extract_json(out))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM generate failed: {e}")

    payload = _ensure_schema(obj)

    # Preserve notification workflow: callback receives the payload
    await _post_callback(callback_url, payload)

    # Also return it to the caller (same as monolith-style behavior)
    return payload
