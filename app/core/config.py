import os
from pathlib import Path

# Project root = ~/ai-bridge
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PANTRY_DB = DATA_DIR / "pantry.sqlite3"
PROFILES_PATH = DATA_DIR / "pantry_profiles.json"

# --- Version / build metadata (override via systemd env) ---
APP_VERSION = os.getenv("APP_VERSION", "1.0.1")
GIT_SHA = os.getenv("GIT_SHA", "unknown")
BUILD_DATE = os.getenv("BUILD_DATE", "unknown")

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen2.5:14b"

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
