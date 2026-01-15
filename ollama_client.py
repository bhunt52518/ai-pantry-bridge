import httpx
from typing import Any, Dict, List, Optional

class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str = "qwen2.5:14b"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, timeout_s: int = 180) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(f"{self.base_url}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()

        # Ollama returns: {"message": {"role": "...", "content": "..."}, ...}
        return (data.get("message") or {}).get("content") or ""
