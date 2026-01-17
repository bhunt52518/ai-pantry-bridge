import re

def extract_json(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output")
    return m.group(0)
